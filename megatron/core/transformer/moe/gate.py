"""
Top-Any Gating for Mixture-of-Experts (Megatron-Core compatible)

Adapts the Top-Any gating logic (cosine-similarity routing with learnable
per-expert thresholds and a Straight-Through Estimator) to fit the
Megatron-Core Router interface.

Contains:
- GAMoEGateSTEBackward: Straight-Through Estimator for binary gating decisions
- TopAnyRouter: Megatron-Core compatible router with Top-Any gating
- LossFreeTopAnyRouter: Loss-free variant with dynamic threshold updates
"""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from megatron.core.transformer.moe.moe_utils import (
    MoEAuxLossAutoScaler,
    ProcessGroupCollection,
    save_to_aux_losses_tracker,
)
from megatron.core.transformer.moe.router import Router
from megatron.core.transformer.transformer_config import TransformerConfig


# ─── STE for binary gating ──────────────────────────────────────────────────

class GAMoEGateSTEBackward(torch.autograd.Function):
    """Straight-Through Estimator (STE) variant:
    - forward: hard binary decision (scores > 0).float()
    - backward: multiply incoming gradient by sigmoid'(scores) (= sigma*(1-sigma))

    This produces hard forward decisions while providing a smooth (sigmoid) shaped
    gradient in the backward pass.
    """

    @staticmethod
    def forward(ctx, scores: Tensor) -> Tensor:
        hard = (scores > 0).float()
        soft = scores.sigmoid()
        ctx.save_for_backward(soft)
        return hard

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        (soft,) = ctx.saved_tensors
        grad = grad_output * (soft * (1.0 - soft))
        return grad


# ─── Megatron-Core compatible Top-Any Router ─────────────────────────────────

class TopAnyRouter(Router):
    """Top-Any Router with learnable per-expert thresholds using Cosine Similarity.

    Replaces Megatron-Core's standard TopKRouter. Instead of routing each token
    to a fixed top-k set of experts, this router uses binary decisions to send
    each token to a *variable* number of experts (0, 1, 2, ... up to E).

    For each token, the router computes a scaled cosine similarity score for
    every expert, then applies a sigmoid and subtracts a learned per-expert
    threshold to make a binary route/don't-route decision via STE.

    The forward pass produces the standard Megatron-Core (probs, routing_map)
    output format, making it a drop-in replacement for TopKRouter.

    Args:
        config (TransformerConfig): Megatron-Core transformer configuration.
            Uses: num_moe_experts, hidden_size, moe_aux_loss_coeff, moe_z_loss_coeff,
                  calculate_per_token_loss, num_layers, mtp_num_layers, mtp_use_repeated_layer.
        pg_collection (ProcessGroupCollection, optional): Process groups for MoE ops.
        is_mtp_layer (bool): Flag indicating if this router is part of an MTP layer.
        sigmoid_target (float): Target pre-sigmoid standard deviation.
            Directly dictates the scaling factor applied to cosine similarity.
    """

    def __init__(
        self,
        config: TransformerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
        is_mtp_layer: bool = False,
        sigmoid_target: float = 3.0,
    ) -> None:
        # Call Router.__init__ which sets up self.config, self.num_experts,
        # self.weight (standard gate linear), process groups, etc.
        super().__init__(config=config, pg_collection=pg_collection, is_mtp_layer=is_mtp_layer)
        del self.weight

        model_dim = config.hidden_size
        num_experts = config.num_moe_experts

        # Override the standard self.weight from Router.__init__ with our
        # cosine-similarity-based routing matrix.
        # Shape: [Model_Dim, Num_Experts] (transposed relative to Router.weight)
        self.sim_matrix = torch.nn.Parameter(
            torch.nn.init.orthogonal_(torch.empty(model_dim, num_experts, dtype=torch.float32)),
            requires_grad=True,
        )

        # Learnable per-expert thresholds (before sigmoid scaling)
        # Initialized to -1.0 so sigmoid(threshold * scale) creates a reasonable baseline
        self.gate_thresholds = torch.nn.Parameter(torch.zeros(num_experts))
        self.gate_thresholds.data.fill_(-1.0)

        # Dynamic logit scaling: stretch the ~1/sqrt(d) cosine variance to sigmoid_target
        optimal_scale = sigmoid_target * math.sqrt(model_dim)
        self.register_buffer(
            "logit_scale",
            torch.tensor(optimal_scale, dtype=torch.float32),
        )

        self.sigmoid_target = sigmoid_target

        print(
            f"[TopAnyRouter] initialized: {num_experts} experts, "
            f"hidden_size={model_dim}, scale={optimal_scale:.1f}, "
            f"sigmoid_target={sigmoid_target}"
        )

    def _get_norm_sim_matrix(self):
        """Get L2-normalized similarity matrix (cached during eval)."""
        if self.training or not hasattr(self, '_cached_norm_sim_matrix'):
            self._cached_norm_sim_matrix = F.normalize(self.sim_matrix.float(), dim=0)
        return self._cached_norm_sim_matrix

    def forward(
        self,
        input: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the Top-Any router.

        Args:
            input (torch.Tensor): Input tensor of shape [seq_length, bsz, hidden_size].
            padding_mask (torch.Tensor, optional): Boolean mask indicating padding tokens.
                Shape [seq_length, bsz]. True for padding, False for valid. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - probs: Token-expert assignment weights, shape [num_tokens, num_experts].
                         Each row sums to 1 over the selected experts (normalized by K).
                - routing_map: Boolean mask, shape [num_tokens, num_experts].
                               True where token is routed to expert.
        """
        # Flatten to [num_tokens, hidden_size]
        original_shape = input.shape
        input_2d = input.view(-1, original_shape[-1])
        num_tokens = input_2d.shape[0]

        input_fp32 = input_2d.float()

        # --- Cosine Similarity ---
        norm_input = F.normalize(input_fp32, dim=1)
        norm_sim_matrix = self._get_norm_sim_matrix()

        # pre_sigmoid: [num_tokens, num_experts]
        pre_sigmoid = torch.matmul(norm_input, norm_sim_matrix) * self.logit_scale

        # --- Probability Bounding & Thresholding ---
        raw_logits = torch.sigmoid(pre_sigmoid)
        gates_scaled = torch.sigmoid(self.gate_thresholds.float())

        # Adjusted score: probability of match minus probability threshold
        scores = raw_logits - gates_scaled.unsqueeze(0)  # [num_tokens, num_experts]

        # --- Binary decision via STE ---
        gates = GAMoEGateSTEBackward.apply(scores)  # [num_tokens, num_experts], 0.0 or 1.0

        # --- Guarantee at least one expert per token ---
        exp_counts_per_token = gates.sum(dim=1)  # [num_tokens]
        no_expert_mask = (exp_counts_per_token == 0)  # [num_tokens] bool

        top1_idx = pre_sigmoid.argmax(dim=1)  # [num_tokens]
        gates.scatter_add_(1, top1_idx.unsqueeze(1), no_expert_mask.float().unsqueeze(1))

        # Per-token expert count K (for weight normalization)
        K = exp_counts_per_token + no_expert_mask.float()  # [num_tokens]

        # --- Log K stats (experts-per-token) ---
        if self.training and torch.is_grad_enabled():
            num_layers = self.config.num_layers
            if self.config.mtp_num_layers is not None:
                num_layers += self.config.mtp_num_layers
            layer_number = self.layer_number
            if self.is_mtp_layer:
                layer_number = self.layer_number + self.config.num_layers

            save_to_aux_losses_tracker(
                "topany_k_mean", K.detach().float().mean(), layer_number, num_layers,
                reduce_op="replace",
            )
            save_to_aux_losses_tracker(
                "topany_k_min", K.detach().min().float(), layer_number, num_layers,
                reduce_op="min",
            )
            save_to_aux_losses_tracker(
                "topany_k_max", K.detach().max().float(), layer_number, num_layers,
                reduce_op="max",
            )
            save_to_aux_losses_tracker(
                "topany_k_std", K.detach().float().std(), layer_number, num_layers,
                reduce_op="replace",
            )

            # --- Log K distribution (fraction of tokens routed to exactly i experts) ---
            k_int = K.detach().long()
            counts = torch.bincount(k_int, minlength=self.num_experts + 1)
            for i in range(1, self.num_experts + 1):
                save_to_aux_losses_tracker(
                    f"topany_k_dist_{i}", counts[i].float() / num_tokens,
                    layer_number, num_layers, reduce_op="replace",
                )

        # --- Build Megatron-Core compatible outputs ---
        # routing_map: boolean mask [num_tokens, num_experts]
        routing_map = gates.bool()

        # probs: normalized weights [num_tokens, num_experts]
        # Each token's weights are normalized by K (number of active experts)
        probs = (gates / torch.clamp(K, 1).unsqueeze(1)).to(input.dtype)

        # --- Auxiliary load-balancing loss ---
        if self.training and torch.is_grad_enabled():
            aux_loss_coeff = self.config.moe_aux_loss_coeff
            if aux_loss_coeff is not None and aux_loss_coeff > 0:
                # Quadratic load-balance loss: l_aux ∝ k²
                exp_counts = gates.sum(dim=0)  # [num_experts]
                non_zero_count = (K > 0).sum().clamp(min=1).float()
                me = exp_counts / non_zero_count
                l_aux = (
                    aux_loss_coeff
                    * torch.mean(me * me)
                    * self.num_experts
                    * self.num_experts
                )

                probs = MoEAuxLossAutoScaler.apply(probs, l_aux)

                # Log the aux loss
                num_layers = self.config.num_layers
                if self.config.mtp_num_layers is not None:
                    num_layers += self.config.mtp_num_layers
                layer_number = self.layer_number
                if self.is_mtp_layer:
                    layer_number = self.layer_number + self.config.num_layers

                save_to_aux_losses_tracker(
                    "load_balancing_loss",
                    l_aux / aux_loss_coeff,
                    layer_number,
                    num_layers,
                    reduce_group=self.tp_cp_group,
                )

        return probs, routing_map

    def routing(self, logits: torch.Tensor):
        """Not used — Top-Any routing is handled entirely in forward()."""
        raise NotImplementedError("TopAnyRouter uses forward() directly, not routing().")


class LossFreeTopAnyRouter(Router):
    """Loss-Free Top-Any Router.

    Instead of using an auxiliary load-balancing loss, this router dynamically
    updates its per-expert thresholds to maintain a target average expert
    activation (target_K). The thresholds are tracked as buffers, not parameters,
    and updated outside the autograd graph based on the previous batch's load.

    Args:
        config (TransformerConfig): Megatron-Core transformer configuration.
        pg_collection (ProcessGroupCollection, optional): Process groups for MoE ops.
        is_mtp_layer (bool): Flag indicating if this router is part of an MTP layer.
        sigmoid_target (float): Target pre-sigmoid standard deviation.
        target_K (float): Desired average number of experts per token (e.g., 2.0).
        update_rate (float): Step size for per-expert threshold updates.
    """

    def __init__(
        self,
        config: TransformerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
        is_mtp_layer: bool = False,
        sigmoid_target: float = 3.0,
        target_K: Optional[float] = None,
        update_rate: Optional[float] = None,
    ) -> None:
        super().__init__(config=config, pg_collection=pg_collection, is_mtp_layer=is_mtp_layer)
        del self.weight

        model_dim = config.hidden_size
        num_experts = config.num_moe_experts

        # Read from config, with constructor args as overrides for backward compat
        self.target_K = target_K if target_K is not None else getattr(
            config, 'moe_topany_target_k', 2.0
        )
        self.update_rate = update_rate if update_rate is not None else getattr(
            config, 'moe_topany_update_rate', 0.01
        )

        self.sim_matrix = torch.nn.Parameter(
            torch.nn.init.orthogonal_(torch.empty(model_dim, num_experts, dtype=torch.float32)),
            requires_grad=True,
        )

        # Buffer for per-expert thresholds (not optimized by gradient descent)
        self.register_buffer("gate_thresholds", torch.full((num_experts,), 0.0))

        optimal_scale = sigmoid_target * math.sqrt(model_dim)
        self.register_buffer(
            "logit_scale",
            torch.tensor(optimal_scale, dtype=torch.float32),
        )

        self.sigmoid_target = sigmoid_target

        print(
            f"[LossFreeTopAnyRouter] initialized: {num_experts} experts, "
            f"hidden_size={model_dim}, scale={optimal_scale:.1f}, "
            f"sigmoid_target={sigmoid_target}, target_K={self.target_K}, "
            f"update_rate={self.update_rate}"
        )

        # High-precision shadow tensor for threshold updates
        # (buffers can get cast to bf16/fp16 by mixed precision wrappers)
        self._fp32_thresholds = None

    def _get_norm_sim_matrix(self):
        """Get L2-normalized similarity matrix (cached during eval)."""
        if self.training or not hasattr(self, '_cached_norm_sim_matrix'):
            self._cached_norm_sim_matrix = F.normalize(self.sim_matrix.float(), dim=0)
        return self._cached_norm_sim_matrix

    def forward(
        self,
        input: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the Loss-Free Top-Any router.

        Args:
            input (torch.Tensor): Input tensor of shape [seq_length, bsz, hidden_size].
            padding_mask (torch.Tensor, optional): Padding mask. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - probs: Token-expert assignment weights [num_tokens, num_experts].
                - routing_map: Boolean mask [num_tokens, num_experts].
        """
        original_shape = input.shape
        input_2d = input.view(-1, original_shape[-1])
        num_tokens = input_2d.shape[0]

        input_fp32 = input_2d.float()

        # --- Cosine Similarity ---
        norm_input = F.normalize(input_fp32, dim=1)
        norm_sim_matrix = self._get_norm_sim_matrix()
        pre_sigmoid = torch.matmul(norm_input, norm_sim_matrix) * self.logit_scale

        # --- Sync fp32 shadow thresholds ---
        if self._fp32_thresholds is None or self._fp32_thresholds.device != input.device:
            self._fp32_thresholds = self.gate_thresholds.clone().float()

        # --- Probability Bounding & Thresholding ---
        raw_logits = torch.sigmoid(pre_sigmoid)
        # .detach() is critical: _fp32_thresholds is updated in-place each step,
        # without detach the autograd graph chains through every past update
        gates_scaled = torch.sigmoid(self._fp32_thresholds.detach())
        scores = raw_logits - gates_scaled.unsqueeze(0)

        # --- Binary decision via STE ---
        gates = GAMoEGateSTEBackward.apply(scores)

        # --- Guarantee at least one expert per token ---
        exp_counts_per_token = gates.sum(dim=1)
        no_expert_mask = (exp_counts_per_token == 0)
        top1_idx = pre_sigmoid.argmax(dim=1)
        gates.scatter_add_(1, top1_idx.unsqueeze(1), no_expert_mask.float().unsqueeze(1))

        K = exp_counts_per_token + no_expert_mask.float()

        # --- Log K stats (experts-per-token) ---
        if self.training and torch.is_grad_enabled():
            num_layers = self.config.num_layers
            if self.config.mtp_num_layers is not None:
                num_layers += self.config.mtp_num_layers
            layer_number = self.layer_number
            if self.is_mtp_layer:
                layer_number = self.layer_number + self.config.num_layers

            save_to_aux_losses_tracker(
                "topany_k_mean", K.detach().float().mean(), layer_number, num_layers,
                reduce_op="replace",
            )
            save_to_aux_losses_tracker(
                "topany_k_min", K.detach().min().float(), layer_number, num_layers,
                reduce_op="min",
            )
            save_to_aux_losses_tracker(
                "topany_k_max", K.detach().max().float(), layer_number, num_layers,
                reduce_op="max",
            )
            save_to_aux_losses_tracker(
                "topany_k_std", K.detach().float().std(), layer_number, num_layers,
                reduce_op="replace",
            )

            # --- Log K distribution (fraction of tokens routed to exactly i experts) ---
            k_int = K.detach().long()
            counts = torch.bincount(k_int, minlength=self.num_experts + 1)
            for i in range(1, self.num_experts + 1):
                save_to_aux_losses_tracker(
                    f"topany_k_dist_{i}", counts[i].float() / num_tokens,
                    layer_number, num_layers, reduce_op="replace",
                )

        # --- Threshold Update Logic (Loss-Free Balancing) ---
        if self.training:
            with torch.no_grad():
                target_c = (num_tokens * self.target_K) / self.num_experts
                actual_c = gates.sum(dim=0)
                e_i = actual_c - target_c

                self._fp32_thresholds += self.update_rate * torch.sign(e_i)
                self.gate_thresholds.copy_(self._fp32_thresholds)

        # --- Build Megatron-Core compatible outputs ---
        routing_map = gates.bool()
        probs = (gates / torch.clamp(K, 1).unsqueeze(1)).to(input.dtype)

        return probs, routing_map

    def routing(self, logits: torch.Tensor):
        """Not used — Top-Any routing is handled entirely in forward()."""
        raise NotImplementedError("LossFreeTopAnyRouter uses forward() directly, not routing().")