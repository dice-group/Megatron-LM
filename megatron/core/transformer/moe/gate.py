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
import os
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


# Guards against moe_utils treating exact 0.0 as "layer not written":
# when every MoE layer reports 0.0 for a replace-reduced metric, the tracker
# falls back to a CPU NaN tensor that mismatches the cuda accumulator → crash.
_EPS = 1e-30

# ─── STE for binary gating ──────────────────────────────────────────────────

class GAMoEGateSTEBackward(torch.autograd.Function):
    """Straight-Through Estimator (STE):
    - forward: hard binary decision (scores > 0).float()
    - backward: identity — pass gradients through unchanged.
    """

    @staticmethod
    def forward(ctx, scores: Tensor) -> Tensor:
        return (scores > 0).float()

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:
        return grad_output


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
        if hasattr(self, 'bias') and self.bias is not None:
            del self.bias
            self.bias = None

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
                "topany_k_std", K.detach().float().std() + _EPS, layer_number, num_layers,
                reduce_op="replace",
            )

            # --- Log K distribution (fraction of tokens routed to exactly i experts) ---
            k_int = K.detach().long()
            counts = torch.bincount(k_int, minlength=self.num_experts + 1)
            for i in range(1, self.num_experts + 1):
                save_to_aux_losses_tracker(
                    f"topany_k_dist_{i}", counts[i].float() / num_tokens + _EPS,
                    layer_number, num_layers, reduce_op="replace",
                )

        # --- Build Megatron-Core compatible outputs ---
        # routing_map: boolean mask [num_tokens, num_experts]
        routing_map = gates.bool()

        # probs: soft weights masked by the binary gates, renormalized to sum to 1.
        # Uses raw_logits (sigmoid of cosine similarity) as per-expert confidence so a
        # high-match expert dominates instead of being diluted to 1/K. Gradient flows
        # through both raw_logits (smooth) and gates (STE).
        soft_weights = raw_logits * gates
        probs = (
            soft_weights / soft_weights.sum(dim=1, keepdim=True).clamp_min(1e-9)
        ).to(input.dtype)

        # --- Auxiliary losses (load balance + optional K-target) ---
        if self.training and torch.is_grad_enabled():
            aux_loss_coeff = self.config.moe_aux_loss_coeff or 0.0
            # K-target loss: coeff · (K̄ − target_K)². Gradient flows through
            # STE → scores → gate_thresholds, directly anchoring K.mean() to
            # the target. Env-controlled to keep the routing config simple.
            k_tgt_coeff = float(os.environ.get("TOPANY_K_TARGET_COEFF", "0"))
            k_tgt_value = float(os.environ.get("TOPANY_K_TARGET", "2.0"))

            # --- Unified sweep CSV ---
            with torch.no_grad():
                ec_diag = gates.sum(dim=0).float()
                ec_mean_diag = ec_diag.mean().clamp_min(1e-6)
                non_zero_count_diag = (K > 0).sum().clamp(min=1).float()
                me_diag = ec_diag / non_zero_count_diag
                l_lb_raw = (me_diag.mul(me_diag).mean() * self.num_experts * self.num_experts).item()
                l_kt_raw = ((K.float().mean() - k_tgt_value) ** 2).item()
                t_diag = self.gate_thresholds.detach().float()
                _sweep_diag_log(
                    self.layer_number,
                    routing_type="topany",
                    k_mean=K.float().mean().item(),
                    k_std=K.float().std().item(),
                    k_max=K.float().max().item(),
                    no_expert_frac=no_expert_mask.float().mean().item(),
                    load_max_over_mean=(ec_diag.max() / ec_mean_diag).item(),
                    load_min_over_mean=(ec_diag.min() / ec_mean_diag).item(),
                    dead_count=(ec_diag < 0.1 * ec_mean_diag).sum().item(),
                    threshold_mean=t_diag.mean().item(),
                    threshold_abs_max=t_diag.abs().max().item(),
                    aux_loss_lb=l_lb_raw,
                    aux_loss_kt=l_kt_raw,
                    target_K=k_tgt_value,
                    num_experts=self.num_experts,
                )

            if aux_loss_coeff > 0 or k_tgt_coeff > 0:
                num_layers = self.config.num_layers
                if self.config.mtp_num_layers is not None:
                    num_layers += self.config.mtp_num_layers
                layer_number = self.layer_number
                if self.is_mtp_layer:
                    layer_number = self.layer_number + self.config.num_layers

                total_aux = torch.zeros((), device=probs.device, dtype=probs.dtype)

                if aux_loss_coeff > 0:
                    exp_counts = gates.sum(dim=0)
                    non_zero_count = (K > 0).sum().clamp(min=1).float()
                    me = exp_counts / non_zero_count
                    l_aux = (
                        aux_loss_coeff
                        * torch.mean(me * me)
                        * self.num_experts
                        * self.num_experts
                    )
                    total_aux = total_aux + l_aux.to(probs.dtype)
                    save_to_aux_losses_tracker(
                        "load_balancing_loss",
                        l_aux.detach() / aux_loss_coeff,
                        layer_number,
                        num_layers,
                        reduce_group=self.tp_cp_group,
                    )

                if k_tgt_coeff > 0:
                    l_k = k_tgt_coeff * ((K.float().mean() - k_tgt_value) ** 2)
                    total_aux = total_aux + l_k.to(probs.dtype)
                    save_to_aux_losses_tracker(
                        "k_target_loss",
                        l_k.detach() / k_tgt_coeff + _EPS,
                        layer_number,
                        num_layers,
                        reduce_op="replace",
                    )

                probs = MoEAuxLossAutoScaler.apply(probs, total_aux)

        return probs, routing_map

    def routing(self, logits: torch.Tensor):
        """Not used — Top-Any routing is handled entirely in forward()."""
        raise NotImplementedError("TopAnyRouter uses forward() directly, not routing().")


_SWEEP_DIAG_COLUMNS = (
    "run_name", "routing_type", "call",
    "k_mean", "k_std", "k_max", "no_expert_frac",
    "load_max_over_mean", "load_min_over_mean", "dead_count",
    "threshold_mean", "threshold_abs_max", "threshold_delta_abs_max",
    "aux_loss_lb", "aux_loss_kt",
    "target_K", "update_rate", "update_mode", "num_experts",
)
# Unified diagnostic-CSV state: shared across all router types in this process.
# First (rank-0, training) router instance to call _sweep_diag_log claims it;
# everyone else returns immediately. Subsamples to keep total rows bounded.
_SWEEP_DIAG = {
    "claimed_layer": None,
    "path": None,
    "call_count": 0,
    "dense_phase": int(os.environ.get("SWEEP_DIAG_DENSE", "50")),
    "stride": int(os.environ.get("SWEEP_DIAG_STRIDE", "200")),
    "initialized": False,
}


def _sweep_diag_log(layer_number: int, **fields) -> None:
    """Append one row to the unified sweep CSV.

    Schema: _SWEEP_DIAG_COLUMNS. Missing fields render as empty strings
    (e.g. lossfree leaves aux_loss_*; topany leaves threshold_delta_*).
    Only writes from rank 0 + first-instance-to-claim-this-layer.
    """
    state = _SWEEP_DIAG

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() != 0:
            return

    if state["claimed_layer"] is None:
        state["claimed_layer"] = layer_number
    if layer_number != state["claimed_layer"]:
        return

    if not state["initialized"]:
        state["initialized"] = True
        run_name = os.environ.get("RUN_NAME", "unknown")
        default = os.path.join("logs", f"sweep_diag_{run_name}.csv")
        state["path"] = os.environ.get("SWEEP_DIAG_FILE", default)
        try:
            os.makedirs(os.path.dirname(state["path"]) or ".", exist_ok=True)
            with open(state["path"], "w") as f:
                f.write(",".join(_SWEEP_DIAG_COLUMNS) + "\n")
            print(
                f"[SweepDiag] CSV: {os.path.abspath(state['path'])} "
                f"(layer={layer_number}, dense={state['dense_phase']}, stride={state['stride']})",
                flush=True,
            )
        except Exception as e:
            print(f"[SweepDiag] init failed: {e}")
            state["path"] = None

    if state["path"] is None:
        return

    state["call_count"] += 1
    c = state["call_count"]
    if c > state["dense_phase"] and (c - state["dense_phase"]) % state["stride"] != 0:
        return

    fields["call"] = c
    fields.setdefault("run_name", os.environ.get("RUN_NAME", "unknown"))

    try:
        with open(state["path"], "a") as f:
            f.write(",".join(str(fields.get(col, "")) for col in _SWEEP_DIAG_COLUMNS) + "\n")
    except Exception as e:
        if c <= 3:
            print(f"[SweepDiag] write failed: {e}")


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
        threshold_update_mode (str): "sign" for fixed-magnitude steps, "magnitude" for
            error-proportional steps.
    """

    def __init__(
        self,
        config: TransformerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
        is_mtp_layer: bool = False,
        sigmoid_target: float = 3.0,
        target_K: Optional[float] = None,
        update_rate: Optional[float] = None,
        threshold_update_mode: Optional[str] = None,
    ) -> None:
        super().__init__(config=config, pg_collection=pg_collection, is_mtp_layer=is_mtp_layer)
        del self.weight
        if hasattr(self, 'bias') and self.bias is not None:
            del self.bias
            self.bias = None

        model_dim = config.hidden_size
        num_experts = config.num_moe_experts

        # Read from config, with constructor args as overrides for backward compat
        self.target_K = target_K if target_K is not None else getattr(
            config, 'moe_topany_target_k', 2.0
        )
        self.update_rate = update_rate if update_rate is not None else getattr(
            config, 'moe_topany_update_rate', 0.01
        )
        self.threshold_update_mode = threshold_update_mode if threshold_update_mode is not None else getattr(
            config, 'moe_topany_threshold_update_mode', 'sign'
        )
        assert self.threshold_update_mode in ("sign", "magnitude"), (
            f"Unknown threshold_update_mode '{self.threshold_update_mode}'. Expected 'sign' or 'magnitude'."
        )

        self.sim_matrix = torch.nn.Parameter(
            torch.nn.init.orthogonal_(torch.empty(model_dim, num_experts, dtype=torch.float32)),
            requires_grad=True,
        )

        # Calculate statistically optimal initial threshold
        # We model the pre_sigmoid scores as roughly N(0, sigmoid_target^2)
        # We want the probability p = target_K / num_experts of exceeding the threshold T:
        # P(Z > T) = p => T = sigmoid_target * inverse_cdf_normal(1 - p)
        # icdf(x) = sqrt(2) * erfinv(2x - 1)
        p = self.target_K / num_experts
        if 0 < p < 1:
            icdf_val = math.sqrt(2.0) * torch.erfinv(torch.tensor(2.0 * (1.0 - p) - 1.0)).item()
            init_t = sigmoid_target * icdf_val
        else:
            init_t = 0.0

        # Buffer for per-expert thresholds (not optimized by gradient descent)
        self.register_buffer("gate_thresholds", torch.full((num_experts,), init_t))

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
            f"update_rate={self.update_rate}, "
            f"threshold_update_mode={self.threshold_update_mode}"
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

        # --- Log K stats (experts-per-token) + threshold/fallback diagnostics ---
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
            # _EPS guards against the moe_utils tracker treating exact 0.0 as
            # "layer not written" (its filter is `loss_list != 0.0`). When ALL
            # MoE layers report 0.0 the tracker falls back to a CPU NaN tensor,
            # which then mismatches device with the cuda accumulator → crash.
            _EPS = 1e-30
            k_int = K.detach().long()
            counts = torch.bincount(k_int, minlength=self.num_experts + 1)
            for i in range(1, self.num_experts + 1):
                save_to_aux_losses_tracker(
                    f"topany_k_dist_{i}", counts[i].float() / num_tokens + _EPS,
                    layer_number, num_layers, reduce_op="replace",
                )

            # --- Threshold + fallback diagnostics (load-bearing for tuning update_rate) ---
            t = self._fp32_thresholds.detach() if self._fp32_thresholds is not None else self.gate_thresholds.detach().float()
            save_to_aux_losses_tracker(
                "threshold_mean", t.mean() + _EPS, layer_number, num_layers, reduce_op="replace",
            )
            save_to_aux_losses_tracker(
                "threshold_std", t.std() + _EPS, layer_number, num_layers, reduce_op="replace",
            )
            save_to_aux_losses_tracker(
                "threshold_abs_max", t.abs().max() + _EPS, layer_number, num_layers, reduce_op="replace",
            )
            save_to_aux_losses_tracker(
                "no_expert_fallback_frac",
                no_expert_mask.float().mean() + _EPS, layer_number, num_layers, reduce_op="replace",
            )
            # Per-expert load imbalance (max/min ratio of normalized expert counts)
            ec = gates.sum(dim=0).detach().float()
            ec_mean = ec.mean().clamp_min(1e-6)
            save_to_aux_losses_tracker(
                "expert_load_max_over_mean", ec.max() / ec_mean + _EPS,
                layer_number, num_layers, reduce_op="replace",
            )
            save_to_aux_losses_tracker(
                "expert_load_min_over_mean", ec.min() / ec_mean + _EPS,
                layer_number, num_layers, reduce_op="replace",
            )

        # --- Threshold Update Logic (Loss-Free Balancing) ---
        if self.training:
            # Capture grad-enabled state BEFORE entering no_grad (otherwise
            # torch.is_grad_enabled() inside the block always returns False).
            log_metrics = torch.is_grad_enabled()
            with torch.no_grad():
                # Sum expert counts across all ranks that see different tokens
                # (TP x DP x CP). Without this, per-rank thresholds drift and the
                # routing decisions diverge between replicas.
                actual_c = gates.sum(dim=0).float()
                group = self.tp_dp_cp_group
                world_size = (
                    torch.distributed.get_world_size(group)
                    if (
                        torch.distributed.is_available()
                        and torch.distributed.is_initialized()
                        and group is not None
                    )
                    else 1
                )
                if world_size > 1:
                    torch.distributed.all_reduce(actual_c, group=group)

                global_num_tokens = num_tokens * world_size
                target_c = (global_num_tokens * self.target_K) / self.num_experts
                e_i = actual_c - target_c

                if self.threshold_update_mode == "sign":
                    delta = self.update_rate * torch.sign(e_i)
                else:  # "magnitude"
                    # Normalize by target_c so the update rate is independent of
                    # batch size, world size, and number of experts.
                    # Clamp to [-1, 1]: e_i/target_c is bounded below at -1
                    # (zero tokens) but unbounded above (one expert can take
                    # ~num_experts × target_c tokens), which biases the
                    # controller upward and lets a single hot batch step the
                    # threshold by rate × (E-1). With the clamp, |delta| ≤ rate.
                    delta = self.update_rate * (e_i / target_c).clamp_(-1.0, 1.0)
                self._fp32_thresholds += delta
                # Anti-windup: clamp to the sigmoid-non-saturated range.
                # Beyond ±3·sigmoid_target, sigmoid(threshold) is within ~0.001
                # of {0,1} — the expert is functionally all-pass or dead, and
                # further drift just slows recovery without changing routing.
                _t_clip = 3.0 * self.sigmoid_target
                self._fp32_thresholds.clamp_(-_t_clip, _t_clip)
                self.gate_thresholds.copy_(self._fp32_thresholds)

                if log_metrics:
                    _EPS = 1e-30
                    num_layers_d = self.config.num_layers
                    if self.config.mtp_num_layers is not None:
                        num_layers_d += self.config.mtp_num_layers
                    layer_number_d = self.layer_number
                    if self.is_mtp_layer:
                        layer_number_d = self.layer_number + self.config.num_layers
                    save_to_aux_losses_tracker(
                        "threshold_delta_abs_mean", delta.abs().mean() + _EPS,
                        layer_number_d, num_layers_d, reduce_op="replace",
                    )
                    save_to_aux_losses_tracker(
                        "threshold_delta_abs_max", delta.abs().max() + _EPS,
                        layer_number_d, num_layers_d, reduce_op="replace",
                    )

                # --- Unified sweep CSV (rank-0 + first-claimed-layer only) ---
                ec = gates.sum(dim=0).float()
                ec_mean = ec.mean().clamp_min(1e-6)
                _sweep_diag_log(
                    self.layer_number,
                    routing_type="lossfree",
                    k_mean=K.float().mean().item(),
                    k_std=K.float().std().item(),
                    k_max=K.float().max().item(),
                    no_expert_frac=no_expert_mask.float().mean().item(),
                    load_max_over_mean=(ec.max() / ec_mean).item(),
                    load_min_over_mean=(ec.min() / ec_mean).item(),
                    dead_count=(ec < 0.1 * ec_mean).sum().item(),
                    threshold_mean=self._fp32_thresholds.mean().item(),
                    threshold_abs_max=self._fp32_thresholds.abs().max().item(),
                    threshold_delta_abs_max=delta.abs().max().item(),
                    target_K=self.target_K,
                    update_rate=self.update_rate,
                    update_mode=self.threshold_update_mode,
                    num_experts=self.num_experts,
                )

        # --- Build Megatron-Core compatible outputs ---
        routing_map = gates.bool()

        # Soft weights: raw_logits masked by binary gates, renormalized per token.
        # See TopAnyRouter.forward for rationale.
        soft_weights = raw_logits * gates
        probs = (
            soft_weights / soft_weights.sum(dim=1, keepdim=True).clamp_min(1e-9)
        ).to(input.dtype)

        return probs, routing_map

    def routing(self, logits: torch.Tensor):
        """Not used — Top-Any routing is handled entirely in forward()."""
        raise NotImplementedError("LossFreeTopAnyRouter uses forward() directly, not routing().")

