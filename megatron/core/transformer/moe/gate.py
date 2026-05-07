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

        # --- Optionally guarantee at least one expert per token ---
        # TOPANY_FORCE_TOP1=1 (default): tokens with K=0 fall back to top-1.
        # TOPANY_FORCE_TOP1=0: tokens with K=0 skip MoE entirely (residual passthrough).
        exp_counts_per_token = gates.sum(dim=1)  # [num_tokens]
        no_expert_mask = (exp_counts_per_token == 0)  # [num_tokens] bool

        if int(os.environ.get("TOPANY_FORCE_TOP1", "1")):
            top1_idx = pre_sigmoid.argmax(dim=1)
            gates.scatter_add_(1, top1_idx.unsqueeze(1), no_expert_mask.float().unsqueeze(1))
            K = exp_counts_per_token + no_expert_mask.float()
        else:
            K = exp_counts_per_token

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
            # Gate before the .item() calls — they each force a host sync.
            if _sweep_diag_should_log(self.layer_number):
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


def _sweep_diag_should_log(layer_number: int) -> bool:
    """Cheap gating check — call BEFORE evaluating the host-syncing .item()
    arguments to _sweep_diag_log. Returns True iff this call should both
    compute the diagnostic tensors and append a CSV row.

    Why: callsites pass ~10 .item() values into _sweep_diag_log. Python
    evaluates kwargs eagerly, so each .item() forces a CUDA→CPU sync —
    even on calls that the stride filter would discard. Gating up-front
    avoids those syncs entirely.

    Maintains call_count + stride state, so write cadence is unchanged
    versus the pre-split implementation.
    """
    state = _SWEEP_DIAG

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() != 0:
            return False

    if state["claimed_layer"] is None:
        state["claimed_layer"] = layer_number
    if layer_number != state["claimed_layer"]:
        return False

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
        return False

    state["call_count"] += 1
    c = state["call_count"]
    if c > state["dense_phase"] and (c - state["dense_phase"]) % state["stride"] != 0:
        return False
    return True


def _sweep_diag_log(layer_number: int, **fields) -> None:
    """Append one row to the unified sweep CSV. Caller must have already
    verified _sweep_diag_should_log(layer_number) — this function does no
    rank/layer/stride filtering itself.

    Schema: _SWEEP_DIAG_COLUMNS. Missing fields render as empty strings
    (e.g. lossfree leaves aux_loss_*; topany leaves threshold_delta_*).
    """
    state = _SWEEP_DIAG
    if state["path"] is None:
        return

    fields["call"] = state["call_count"]
    fields.setdefault("run_name", os.environ.get("RUN_NAME", "unknown"))

    try:
        with open(state["path"], "a") as f:
            f.write(",".join(str(fields.get(col, "")) for col in _SWEEP_DIAG_COLUMNS) + "\n")
    except Exception as e:
        if state["call_count"] <= 3:
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

        # --- Optionally guarantee at least one expert per token ---
        exp_counts_per_token = gates.sum(dim=1)
        no_expert_mask = (exp_counts_per_token == 0)
        if int(os.environ.get("TOPANY_FORCE_TOP1", "1")):
            top1_idx = pre_sigmoid.argmax(dim=1)
            gates.scatter_add_(1, top1_idx.unsqueeze(1), no_expert_mask.float().unsqueeze(1))
            K = exp_counts_per_token + no_expert_mask.float()
        else:
            K = exp_counts_per_token

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
                # Gate before the .item() calls — they each force a host sync.
                if _sweep_diag_should_log(self.layer_number):
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


# ─── Sigmoid-Linear Top-Any Router ───────────────────────────────────────────
#
# Different parameterization of variable-K routing: instead of cosine similarity
# with learnable per-expert thresholds, use a plain Linear(d, E) → sigmoid with
# a fixed 0.5 cutoff. The K is implicit (count of experts above 0.5). Optional
# K-target loss + standard load-balance aux loss are inherited from TopAnyRouter.
#
# Decision: σ(W^T x) > 0.5  ⇔  W^T x > 0
#
# This drops three quirks of the cosine variant at once:
#   1. No learnable threshold — cutoff fixed at 0.5, removes a degree of freedom
#      the LM gradient was abusing (pushing thresholds down to dilute probs).
#   2. No cosine normalization — linear logits can absorb both routing direction
#      and magnitude (see LossFreeTopAnyRouter post-mortem: cosine pre-sigmoid
#      drifted upward as sim_matrix learned, pinning thresholds at clamp).
#   3. No scaling factor — sigmoid_target/logit_scale becomes a property of W's
#      init scale (handled by config.init_method), not a separate hyperparam.

class SigmoidGateRouter(Router):
    """Sigmoid-Linear router with binary route/no-route decisions.

    Linear(d, E) → sigmoid → STE(>0.5). Variable K per token. Standard MoE load
    balance aux loss + optional env-controlled K-target loss.

    Args:
        config (TransformerConfig): Megatron-Core transformer configuration.
        pg_collection (ProcessGroupCollection, optional): Process groups for MoE ops.
        is_mtp_layer (bool): Flag indicating if this router is part of an MTP layer.
    """

    def __init__(
        self,
        config: TransformerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
        is_mtp_layer: bool = False,
    ) -> None:
        super().__init__(config=config, pg_collection=pg_collection, is_mtp_layer=is_mtp_layer)
        # Reuse self.weight from Router.__init__ — it's the [E, d] linear we want.
        # Drop the trainable bias; cutoff is fixed at logit=0 (σ=0.5).
        if hasattr(self, 'bias') and self.bias is not None:
            del self.bias
            self.bias = None

        print(
            f"[SigmoidGateRouter] initialized: {self.num_experts} experts, "
            f"hidden_size={config.hidden_size}"
        )

    def forward(
        self,
        input: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        original_shape = input.shape
        input_2d = input.view(-1, original_shape[-1])
        num_tokens = input_2d.shape[0]

        # Linear logits via base-class gating (handles dtype, device).
        logits = self.gating(input_2d).float()  # [num_tokens, num_experts]

        raw_logits = torch.sigmoid(logits)
        scores = raw_logits - 0.5  # cutoff at 0.5 ⇔ logit > 0
        gates = GAMoEGateSTEBackward.apply(scores)  # [num_tokens, num_experts]

        # --- Optionally guarantee at least one expert per token ---
        exp_counts_per_token = gates.sum(dim=1)
        no_expert_mask = (exp_counts_per_token == 0)
        if int(os.environ.get("TOPANY_FORCE_TOP1", "1")):
            top1_idx = logits.argmax(dim=1)
            gates.scatter_add_(1, top1_idx.unsqueeze(1), no_expert_mask.float().unsqueeze(1))
            K = exp_counts_per_token + no_expert_mask.float()
        else:
            K = exp_counts_per_token

        # --- Log K stats ---
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

            k_int = K.detach().long()
            counts = torch.bincount(k_int, minlength=self.num_experts + 1)
            for i in range(1, self.num_experts + 1):
                save_to_aux_losses_tracker(
                    f"topany_k_dist_{i}", counts[i].float() / num_tokens + _EPS,
                    layer_number, num_layers, reduce_op="replace",
                )

        routing_map = gates.bool()
        soft_weights = raw_logits * gates
        probs = (
            soft_weights / soft_weights.sum(dim=1, keepdim=True).clamp_min(1e-9)
        ).to(input.dtype)

        # --- Aux losses + sweep CSV ---
        if self.training and torch.is_grad_enabled():
            aux_loss_coeff = self.config.moe_aux_loss_coeff or 0.0
            k_tgt_coeff = float(os.environ.get("TOPANY_K_TARGET_COEFF", "0"))
            k_tgt_value = float(os.environ.get("TOPANY_K_TARGET", "2.0"))

            # Gate before the .item() calls — they each force a host sync.
            if _sweep_diag_should_log(self.layer_number):
                with torch.no_grad():
                    ec_diag = gates.sum(dim=0).float()
                    ec_mean_diag = ec_diag.mean().clamp_min(1e-6)
                    non_zero_count_diag = (K > 0).sum().clamp(min=1).float()
                    me_diag = ec_diag / non_zero_count_diag
                    l_lb_raw = (me_diag.mul(me_diag).mean() * self.num_experts * self.num_experts).item()
                    l_kt_raw = ((K.float().mean() - k_tgt_value) ** 2).item()
                    # threshold_* fields don't apply (fixed 0.5 cutoff). Logit stats
                    # go in their place so the sweep CSV stays usefully populated.
                    logit_diag = logits.detach().float()
                    _sweep_diag_log(
                        self.layer_number,
                        routing_type="sigmoid",
                        k_mean=K.float().mean().item(),
                        k_std=K.float().std().item(),
                        k_max=K.float().max().item(),
                        no_expert_frac=no_expert_mask.float().mean().item(),
                        load_max_over_mean=(ec_diag.max() / ec_mean_diag).item(),
                        load_min_over_mean=(ec_diag.min() / ec_mean_diag).item(),
                        dead_count=(ec_diag < 0.1 * ec_mean_diag).sum().item(),
                        threshold_mean=logit_diag.mean().item(),
                        threshold_abs_max=logit_diag.abs().max().item(),
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
        """Not used — handled entirely in forward()."""
        raise NotImplementedError("SigmoidGateRouter uses forward() directly, not routing().")


class LossFreeSigmoidRouter(Router):
    """Loss-Free Sigmoid-Linear router with per-expert bias balancing.

    Linear(d, E) → sigmoid(logit + b_e) → STE(>0.5), where b_e is a per-expert
    bias buffer (NOT a parameter) updated outside autograd to balance load:
    overloaded experts get b_e ↓, underloaded get b_e ↑.

    Equivalent to LossFreeTopAnyRouter's threshold mechanism but on a linear
    pre-activation. Should avoid the cosine-LF failure mode where pre_sigmoid
    drifted upward as sim_matrix learned (linear weights absorb that drift).

    Args:
        config (TransformerConfig): Megatron-Core transformer configuration.
        pg_collection (ProcessGroupCollection, optional): Process groups for MoE ops.
        is_mtp_layer (bool): Flag indicating if this router is part of an MTP layer.
        target_K (float): Desired average number of experts per token (e.g., 2.0).
        update_rate (float): Step size for per-expert bias updates.
        threshold_update_mode (str): "sign" or "magnitude" (kept named for env-var
            compatibility with the cosine LF router; controls bias updates here).
    """

    def __init__(
        self,
        config: TransformerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
        is_mtp_layer: bool = False,
        target_K: Optional[float] = None,
        update_rate: Optional[float] = None,
        threshold_update_mode: Optional[str] = None,
    ) -> None:
        super().__init__(config=config, pg_collection=pg_collection, is_mtp_layer=is_mtp_layer)
        if hasattr(self, 'bias') and self.bias is not None:
            del self.bias
            self.bias = None

        num_experts = config.num_moe_experts

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

        # Per-expert bias on the pre-sigmoid logit. Buffer (not a parameter)
        # so it doesn't get gradient updates. Init at 0 (decision boundary at
        # logit=0 ⇔ σ=0.5 ⇔ no bias).
        self.register_buffer("lf_bias", torch.zeros(num_experts))
        # High-precision shadow tensor (mixed-precision can cast the buffer).
        self._fp32_lf_bias = None

        print(
            f"[LossFreeSigmoidRouter] initialized: {num_experts} experts, "
            f"hidden_size={config.hidden_size}, target_K={self.target_K}, "
            f"update_rate={self.update_rate}, mode={self.threshold_update_mode}"
        )

    def forward(
        self,
        input: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        original_shape = input.shape
        input_2d = input.view(-1, original_shape[-1])
        num_tokens = input_2d.shape[0]

        logits = self.gating(input_2d).float()  # [num_tokens, num_experts]

        if self._fp32_lf_bias is None or self._fp32_lf_bias.device != input.device:
            self._fp32_lf_bias = self.lf_bias.clone().float().to(input.device)

        biased_logits = logits + self._fp32_lf_bias.detach().unsqueeze(0)
        raw_logits = torch.sigmoid(biased_logits)
        scores = raw_logits - 0.5
        gates = GAMoEGateSTEBackward.apply(scores)

        # --- Optionally guarantee at least one expert per token ---
        exp_counts_per_token = gates.sum(dim=1)
        no_expert_mask = (exp_counts_per_token == 0)
        if int(os.environ.get("TOPANY_FORCE_TOP1", "1")):
            top1_idx = biased_logits.argmax(dim=1)
            gates.scatter_add_(1, top1_idx.unsqueeze(1), no_expert_mask.float().unsqueeze(1))
            K = exp_counts_per_token + no_expert_mask.float()
        else:
            K = exp_counts_per_token

        # --- Log K stats + bias diagnostics ---
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

            k_int = K.detach().long()
            counts = torch.bincount(k_int, minlength=self.num_experts + 1)
            for i in range(1, self.num_experts + 1):
                save_to_aux_losses_tracker(
                    f"topany_k_dist_{i}", counts[i].float() / num_tokens + _EPS,
                    layer_number, num_layers, reduce_op="replace",
                )

            b = self._fp32_lf_bias.detach()
            save_to_aux_losses_tracker(
                "threshold_mean", b.mean() + _EPS, layer_number, num_layers, reduce_op="replace",
            )
            save_to_aux_losses_tracker(
                "threshold_abs_max", b.abs().max() + _EPS, layer_number, num_layers, reduce_op="replace",
            )
            save_to_aux_losses_tracker(
                "no_expert_fallback_frac",
                no_expert_mask.float().mean() + _EPS, layer_number, num_layers, reduce_op="replace",
            )
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

        # --- Bias update (loss-free balancing, outside autograd) ---
        if self.training:
            log_metrics = torch.is_grad_enabled()
            with torch.no_grad():
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

                # Sign-flipped vs LossFreeTopAnyRouter: there, threshold↑ ⇒ less
                # selected. Here, bias↑ ⇒ MORE selected (logit + bias goes up).
                # So overloaded (e_i > 0) needs bias DOWN.
                if self.threshold_update_mode == "sign":
                    delta = -self.update_rate * torch.sign(e_i)
                else:  # "magnitude"
                    delta = -self.update_rate * (e_i / target_c).clamp_(-1.0, 1.0)

                self._fp32_lf_bias += delta
                # Anti-windup: σ(±9) saturates within ~0.0001 of {0,1}; further
                # drift just slows recovery without changing routing.
                self._fp32_lf_bias.clamp_(-9.0, 9.0)
                self.lf_bias.copy_(self._fp32_lf_bias)

                if log_metrics:
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

                # --- Sweep CSV ---
                # Gate before the .item() calls — they each force a host sync.
                if _sweep_diag_should_log(self.layer_number):
                    ec = gates.sum(dim=0).float()
                    ec_mean = ec.mean().clamp_min(1e-6)
                    _sweep_diag_log(
                        self.layer_number,
                        routing_type="sigmoid_lossfree",
                        k_mean=K.float().mean().item(),
                        k_std=K.float().std().item(),
                        k_max=K.float().max().item(),
                        no_expert_frac=no_expert_mask.float().mean().item(),
                        load_max_over_mean=(ec.max() / ec_mean).item(),
                        load_min_over_mean=(ec.min() / ec_mean).item(),
                        dead_count=(ec < 0.1 * ec_mean).sum().item(),
                        threshold_mean=self._fp32_lf_bias.mean().item(),
                        threshold_abs_max=self._fp32_lf_bias.abs().max().item(),
                        threshold_delta_abs_max=delta.abs().max().item(),
                        target_K=self.target_K,
                        update_rate=self.update_rate,
                        update_mode=self.threshold_update_mode,
                        num_experts=self.num_experts,
                    )

        routing_map = gates.bool()
        soft_weights = raw_logits * gates
        probs = (
            soft_weights / soft_weights.sum(dim=1, keepdim=True).clamp_min(1e-9)
        ).to(input.dtype)

        # Optional load-balance aux loss. Off by default (loss-free balancing
        # via the bias controller is the design). Enabling it gives the LM
        # gradient something to push back against if the linear logits drift
        # upward to compensate for an over-saturated negative bias.
        if self.training and torch.is_grad_enabled():
            aux_loss_coeff = self.config.moe_aux_loss_coeff or 0.0
            if aux_loss_coeff > 0:
                num_layers = self.config.num_layers
                if self.config.mtp_num_layers is not None:
                    num_layers += self.config.mtp_num_layers
                layer_number = self.layer_number
                if self.is_mtp_layer:
                    layer_number = self.layer_number + self.config.num_layers

                exp_counts = gates.sum(dim=0)
                non_zero_count = (K > 0).sum().clamp(min=1).float()
                me = exp_counts / non_zero_count
                l_aux = (
                    aux_loss_coeff
                    * torch.mean(me * me)
                    * self.num_experts
                    * self.num_experts
                )
                save_to_aux_losses_tracker(
                    "load_balancing_loss",
                    l_aux.detach() / aux_loss_coeff,
                    layer_number,
                    num_layers,
                    reduce_group=self.tp_cp_group,
                )
                probs = MoEAuxLossAutoScaler.apply(probs, l_aux.to(probs.dtype))

        return probs, routing_map

    def routing(self, logits: torch.Tensor):
        """Not used — handled entirely in forward()."""
        raise NotImplementedError("LossFreeSigmoidRouter uses forward() directly, not routing().")


# ─── Expert Threshold Router (kth-largest EMA) ───────────────────────────────
#
# Different load-balancing controller than LossFree*: instead of a count-error
# feedback loop ("expert took too many tokens → push bias down"), maintain a
# per-expert threshold via EMA of the kth-largest score that arrives at each
# expert. With k = target_K · num_tokens / num_experts, the threshold converges
# to the order-statistic that yields exactly target_K activations per token in
# expectation. Tracking the score distribution directly avoids the overshoot
# and oscillation seen when a count-based controller chases a moving target
# while the underlying gating weights are still drifting.
#
# Equivalent to Expert-Choice routing over an infinitely large batch, which
# restores causality (no batch-wise ranking required at any step).

class ETRouter(Router):
    """Expert Threshold router with kth-largest EMA balancing.

    Linear(d, E) → sigmoid → STE(score > c_e), where c_e is a per-expert
    threshold buffer updated outside autograd as
    ``c_e ← β·c_e + (1-β)·kth-largest(score_e)``,
    with k chosen so that uniform routing yields exactly ``target_K``
    activations per token.

    Args:
        config (TransformerConfig): Megatron-Core transformer configuration.
        pg_collection (ProcessGroupCollection, optional): Process groups for MoE ops.
        is_mtp_layer (bool): Whether this router belongs to an MTP layer.
        target_K (float, optional): Desired average activations per token.
        ema_beta (float): EMA decay for threshold tracking. Env override:
            ``ET_EMA_BETA`` (default 0.99).
    """

    def __init__(
        self,
        config: TransformerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
        is_mtp_layer: bool = False,
        target_K: Optional[float] = None,
        ema_beta: float = 0.99,
    ) -> None:
        super().__init__(config=config, pg_collection=pg_collection, is_mtp_layer=is_mtp_layer)
        if hasattr(self, 'bias') and self.bias is not None:
            del self.bias
            self.bias = None

        num_experts = config.num_moe_experts

        self.target_K = target_K if target_K is not None else getattr(
            config, 'moe_topany_target_k', 2.0
        )
        self.ema_beta = float(os.environ.get("ET_EMA_BETA", str(ema_beta)))

        # Init at the score quantile that yields target_K acceptance under a
        # uniform sigmoid distribution: with E experts and target_K kept,
        # threshold = 1 - target_K/E in [0,1].
        init_t = max(0.0, min(1.0, 1.0 - self.target_K / num_experts))
        self.register_buffer("et_threshold", torch.full((num_experts,), init_t))
        self._fp32_threshold = None

        print(
            f"[ETRouter] initialized: {num_experts} experts, "
            f"hidden_size={config.hidden_size}, target_K={self.target_K}, "
            f"ema_beta={self.ema_beta}, init_threshold={init_t:.4f}"
        )

    def forward(
        self,
        input: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        original_shape = input.shape
        input_2d = input.view(-1, original_shape[-1])
        num_tokens = input_2d.shape[0]

        logits = self.gating(input_2d).float()
        raw_logits = torch.sigmoid(logits)

        if self._fp32_threshold is None or self._fp32_threshold.device != input.device:
            self._fp32_threshold = self.et_threshold.clone().float().to(input.device)

        scores = raw_logits - self._fp32_threshold.detach().unsqueeze(0)
        gates = GAMoEGateSTEBackward.apply(scores)

        # FORCE_TOP1 fallback (consistent with other Top-Any variants).
        exp_counts_per_token = gates.sum(dim=1)
        no_expert_mask = (exp_counts_per_token == 0)
        if int(os.environ.get("TOPANY_FORCE_TOP1", "1")):
            top1_idx = logits.argmax(dim=1)
            gates.scatter_add_(1, top1_idx.unsqueeze(1), no_expert_mask.float().unsqueeze(1))
            K = exp_counts_per_token + no_expert_mask.float()
        else:
            K = exp_counts_per_token

        # K + threshold + load diagnostics
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
            k_int = K.detach().long()
            counts = torch.bincount(k_int, minlength=self.num_experts + 1)
            for i in range(1, self.num_experts + 1):
                save_to_aux_losses_tracker(
                    f"topany_k_dist_{i}", counts[i].float() / num_tokens + _EPS,
                    layer_number, num_layers, reduce_op="replace",
                )

            t = self._fp32_threshold.detach()
            save_to_aux_losses_tracker(
                "threshold_mean", t.mean() + _EPS, layer_number, num_layers, reduce_op="replace",
            )
            save_to_aux_losses_tracker(
                "threshold_abs_max", t.abs().max() + _EPS, layer_number, num_layers, reduce_op="replace",
            )
            save_to_aux_losses_tracker(
                "no_expert_fallback_frac",
                no_expert_mask.float().mean() + _EPS, layer_number, num_layers, reduce_op="replace",
            )
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

        # EMA threshold update (outside autograd)
        if self.training:
            log_metrics = torch.is_grad_enabled()
            with torch.no_grad():
                # k = expected tokens per expert under target_K balanced routing.
                k = max(1, min(num_tokens,
                               int(round(num_tokens * self.target_K / self.num_experts))))
                # kth-largest sigmoid score per expert = smallest of the top-k.
                topk_vals, _ = raw_logits.detach().topk(k, dim=0)
                kth_largest = topk_vals[-1]  # [num_experts]

                # Average across TP/DP/CP for cross-rank threshold consistency.
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
                    torch.distributed.all_reduce(kth_largest, group=group)
                    kth_largest /= world_size

                self._fp32_threshold.mul_(self.ema_beta).add_(
                    kth_largest, alpha=1.0 - self.ema_beta
                )
                self._fp32_threshold.clamp_(0.0, 1.0)
                self.et_threshold.copy_(self._fp32_threshold)

                if log_metrics and _sweep_diag_should_log(self.layer_number):
                    ec = gates.sum(dim=0).float()
                    ec_mean = ec.mean().clamp_min(1e-6)
                    _sweep_diag_log(
                        self.layer_number,
                        routing_type="et",
                        k_mean=K.float().mean().item(),
                        k_std=K.float().std().item(),
                        k_max=K.float().max().item(),
                        no_expert_frac=no_expert_mask.float().mean().item(),
                        load_max_over_mean=(ec.max() / ec_mean).item(),
                        load_min_over_mean=(ec.min() / ec_mean).item(),
                        dead_count=(ec < 0.1 * ec_mean).sum().item(),
                        threshold_mean=self._fp32_threshold.mean().item(),
                        threshold_abs_max=self._fp32_threshold.abs().max().item(),
                        target_K=self.target_K,
                        update_rate=self.ema_beta,
                        update_mode="ema_kth",
                        num_experts=self.num_experts,
                    )

        routing_map = gates.bool()
        soft_weights = raw_logits * gates
        probs = (
            soft_weights / soft_weights.sum(dim=1, keepdim=True).clamp_min(1e-9)
        ).to(input.dtype)

        return probs, routing_map

    def routing(self, logits: torch.Tensor):
        """Not used — handled entirely in forward()."""
        raise NotImplementedError("ETRouter uses forward() directly, not routing().")


# ─── Top-P (Confidence) Router ───────────────────────────────────────────────
#
# Variable K via cumulative-confidence thresholding. Per token, sort sigmoid
# scores descending and activate experts until their cumulative sum first
# reaches ``top_p``. K is implicit: confident tokens get K=1, ambiguous tokens
# pull in more experts. No threshold buffer to update; balance is encouraged
# via the standard load-balance aux loss + optional entropy-min regularizer.

class TopPRouter(Router):
    """Top-P confidence-based variable-K router.

    Linear(d, E) → sigmoid → cumulative-sum → STE-thresholded selection.
    Number of activated experts depends on token-level routing confidence.

    Args:
        config (TransformerConfig): Megatron-Core transformer configuration.
        pg_collection (ProcessGroupCollection, optional): Process groups for MoE ops.
        is_mtp_layer (bool): Whether this router belongs to an MTP layer.
        top_p (float, optional): Cumulative confidence threshold. Env override:
            ``TOPP_THRESHOLD`` (default 0.5).
    """

    def __init__(
        self,
        config: TransformerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
        is_mtp_layer: bool = False,
        top_p: Optional[float] = None,
    ) -> None:
        super().__init__(config=config, pg_collection=pg_collection, is_mtp_layer=is_mtp_layer)
        if hasattr(self, 'bias') and self.bias is not None:
            del self.bias
            self.bias = None

        default_p = top_p if top_p is not None else 0.5
        self.top_p = float(os.environ.get("TOPP_THRESHOLD", str(default_p)))

        print(
            f"[TopPRouter] initialized: {self.num_experts} experts, "
            f"hidden_size={config.hidden_size}, top_p={self.top_p}"
        )

    def forward(
        self,
        input: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        original_shape = input.shape
        input_2d = input.view(-1, original_shape[-1])
        num_tokens = input_2d.shape[0]

        logits = self.gating(input_2d).float()
        raw_logits = torch.sigmoid(logits)  # [N, E]

        # Sort descending, keep experts whose cumulative-confidence pre-sum
        # is still below top_p (i.e., include the one that crosses p).
        sorted_logits, sorted_idx = raw_logits.detach().sort(dim=1, descending=True)
        cumsum = sorted_logits.cumsum(dim=1)
        prev_cum = torch.cat(
            [torch.zeros_like(cumsum[:, :1]), cumsum[:, :-1]], dim=1
        )
        keep_sorted = (prev_cum < self.top_p)  # [N, E] bool

        keep = torch.zeros_like(keep_sorted)
        keep.scatter_(1, sorted_idx, keep_sorted)

        # STE: per-token cutoff = smallest kept sigmoid score; gradient flows
        # through `raw_logits - cutoff` so the router can learn to push scores
        # above/below the dynamic per-token boundary.
        num_kept = keep_sorted.float().sum(dim=1).long().clamp(min=1)  # [N]
        last_kept_pos = (num_kept - 1).clamp(max=self.num_experts - 1)
        cutoff_score = sorted_logits.gather(1, last_kept_pos.unsqueeze(1))  # [N,1]
        scores = raw_logits - cutoff_score
        # Forward gates match the sorted-cumsum decision exactly (continuous
        # sigmoid scores → ties are negligible). Backward = identity through scores.
        gates = GAMoEGateSTEBackward.apply(scores)

        # FORCE_TOP1 fallback (rare here since the first-ranked expert is
        # always kept by construction, but kept for parity with siblings).
        exp_counts_per_token = gates.sum(dim=1)
        no_expert_mask = (exp_counts_per_token == 0)
        if int(os.environ.get("TOPANY_FORCE_TOP1", "1")):
            top1_idx = logits.argmax(dim=1)
            gates.scatter_add_(1, top1_idx.unsqueeze(1), no_expert_mask.float().unsqueeze(1))
            K = exp_counts_per_token + no_expert_mask.float()
        else:
            K = exp_counts_per_token

        # K stats
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
            k_int = K.detach().long()
            counts = torch.bincount(k_int, minlength=self.num_experts + 1)
            for i in range(1, self.num_experts + 1):
                save_to_aux_losses_tracker(
                    f"topany_k_dist_{i}", counts[i].float() / num_tokens + _EPS,
                    layer_number, num_layers, reduce_op="replace",
                )

        routing_map = gates.bool()
        soft_weights = raw_logits * gates
        probs = (
            soft_weights / soft_weights.sum(dim=1, keepdim=True).clamp_min(1e-9)
        ).to(input.dtype)

        # Aux losses: load balance (always useful here since there's no
        # per-expert threshold) + optional entropy-min to encourage decisive
        # confidence distributions over time.
        if self.training and torch.is_grad_enabled():
            aux_loss_coeff = self.config.moe_aux_loss_coeff or 0.0
            entropy_coeff = float(os.environ.get("TOPP_ENTROPY_COEFF", "0"))

            if _sweep_diag_should_log(self.layer_number):
                with torch.no_grad():
                    ec_diag = gates.sum(dim=0).float()
                    ec_mean_diag = ec_diag.mean().clamp_min(1e-6)
                    non_zero_count_diag = (K > 0).sum().clamp(min=1).float()
                    me_diag = ec_diag / non_zero_count_diag
                    l_lb_raw = (me_diag.mul(me_diag).mean() *
                                self.num_experts * self.num_experts).item()
                    logit_diag = logits.detach().float()
                    _sweep_diag_log(
                        self.layer_number,
                        routing_type="topp",
                        k_mean=K.float().mean().item(),
                        k_std=K.float().std().item(),
                        k_max=K.float().max().item(),
                        no_expert_frac=no_expert_mask.float().mean().item(),
                        load_max_over_mean=(ec_diag.max() / ec_mean_diag).item(),
                        load_min_over_mean=(ec_diag.min() / ec_mean_diag).item(),
                        dead_count=(ec_diag < 0.1 * ec_mean_diag).sum().item(),
                        threshold_mean=logit_diag.mean().item(),
                        threshold_abs_max=logit_diag.abs().max().item(),
                        aux_loss_lb=l_lb_raw,
                        target_K=self.top_p,
                        num_experts=self.num_experts,
                    )

            if aux_loss_coeff > 0 or entropy_coeff > 0:
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

                if entropy_coeff > 0:
                    # H(p) per token over the sigmoid-normalized distribution;
                    # minimizing H pushes the router toward decisive choices.
                    p_norm = raw_logits / raw_logits.sum(dim=1, keepdim=True).clamp_min(1e-9)
                    ent = -(p_norm * (p_norm.clamp_min(1e-9)).log()).sum(dim=1).mean()
                    l_ent = entropy_coeff * ent
                    total_aux = total_aux + l_ent.to(probs.dtype)
                    save_to_aux_losses_tracker(
                        "router_entropy",
                        ent.detach() + _EPS,
                        layer_number,
                        num_layers,
                        reduce_op="replace",
                    )

                probs = MoEAuxLossAutoScaler.apply(probs, total_aux)

        return probs, routing_map

    def routing(self, logits: torch.Tensor):
        """Not used — handled entirely in forward()."""
        raise NotImplementedError("TopPRouter uses forward() directly, not routing().")


# ─── K-target annealing wrapper around LossFreeSigmoidRouter ─────────────────
#
# Same machinery as LossFreeSigmoidRouter, but ``target_K`` is cosine-annealed
# from ``TOPANY_K_ANNEAL_START`` → ``TOPANY_K_ANNEAL_END`` between
# ``TOPANY_K_ANNEAL_START_STEP`` and ``TOPANY_K_ANNEAL_END_STEP``. Step counter
# increments per training-forward call, so the schedule is in micro-batches —
# multiply by GRAD_ACCUM_STEPS if you want to think in optimizer steps.
#
# Motivation (Sigma-MoE-Tiny progressive sparsification): early layers can't
# differentiate representations well enough to support sparse routing, so
# starting at higher K and annealing down stabilizes loss and lets specialized
# experts emerge organically before being squeezed.

class LossFreeSigmoidAnnealRouter(LossFreeSigmoidRouter):
    """LossFreeSigmoidRouter with cosine K-target annealing."""

    def __init__(
        self,
        config: TransformerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
        is_mtp_layer: bool = False,
        target_K: Optional[float] = None,
        update_rate: Optional[float] = None,
        threshold_update_mode: Optional[str] = None,
    ) -> None:
        super().__init__(
            config=config,
            pg_collection=pg_collection,
            is_mtp_layer=is_mtp_layer,
            target_K=target_K,
            update_rate=update_rate,
            threshold_update_mode=threshold_update_mode,
        )
        self._anneal_start_K = float(
            os.environ.get("TOPANY_K_ANNEAL_START", str(self.target_K))
        )
        self._anneal_end_K = float(
            os.environ.get("TOPANY_K_ANNEAL_END", str(self.target_K))
        )
        self._anneal_start_step = int(os.environ.get("TOPANY_K_ANNEAL_START_STEP", "0"))
        self._anneal_end_step = int(os.environ.get("TOPANY_K_ANNEAL_END_STEP", "0"))
        self._fwd_step = 0

        print(
            f"[LossFreeSigmoidAnnealRouter] anneal K: "
            f"{self._anneal_start_K} → {self._anneal_end_K} "
            f"over steps [{self._anneal_start_step}, {self._anneal_end_step}]"
        )

    def _current_target_K(self) -> float:
        if self._anneal_end_step <= self._anneal_start_step:
            return self._anneal_start_K
        if self._fwd_step <= self._anneal_start_step:
            return self._anneal_start_K
        if self._fwd_step >= self._anneal_end_step:
            return self._anneal_end_K
        progress = (self._fwd_step - self._anneal_start_step) / (
            self._anneal_end_step - self._anneal_start_step
        )
        return self._anneal_end_K + 0.5 * (self._anneal_start_K - self._anneal_end_K) * (
            1.0 + math.cos(math.pi * progress)
        )

    def forward(
        self,
        input: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.training:
            self._fwd_step += 1
            self.target_K = self._current_target_K()
        return super().forward(input, padding_mask)


# ─── ReMoE: ReLU Routing (fully differentiable) ──────────────────────────────
#
# Wang/Chen/Zhu 2024, arXiv:2412.14711. Unlike every other variable-K router in
# this file, ReMoE has NO STE: gate = ReLU(W·x), so the gradient at the
# selection boundary is well-defined. probs ARE the gate values (un-normalized,
# per the paper) — gradient pressure for sparsity comes from a load-weighted L1
# penalty whose coefficient λ is multiplicatively adapted to track a target
# average-K. Paper claims consistent wins over TopK at multiple scales — the
# only mechanism we've added that the literature reports actually beating
# TopK on equal compute.

class ReMoERouter(Router):
    """ReLU-gated MoE router with adaptive L1 sparsity controller.

    Forward: gate = ReLU(linear(x)); routing_map = (gate > 0); probs = gate
    (NOT normalized — un-normalized gate values are what creates the gradient
    pressure for sparsity once L1 is applied).

    Sparsity control: λ_{t+1} = λ_t · (1 ± α) based on whether current average
    K exceeds or falls below ``target_K``. Load balancing folded into the L1:
    per-expert L1 weight = batch frequency f_e (over-used experts get pushed
    down harder).

    Args:
        config (TransformerConfig): Megatron-Core transformer configuration.
        pg_collection (ProcessGroupCollection, optional): Process groups.
        is_mtp_layer (bool): MTP-layer flag.
        target_K (float, optional): Desired average K. Env: ``REMOE_TARGET_K``.
    """

    def __init__(
        self,
        config: TransformerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
        is_mtp_layer: bool = False,
        target_K: Optional[float] = None,
    ) -> None:
        super().__init__(config=config, pg_collection=pg_collection, is_mtp_layer=is_mtp_layer)
        # Drop bias — we want pure linear logits feeding the ReLU.
        if hasattr(self, 'bias') and self.bias is not None:
            del self.bias
            self.bias = None

        self.target_K = float(os.environ.get(
            "REMOE_TARGET_K",
            str(target_K if target_K is not None else
                getattr(config, 'moe_topany_target_k', 2.0)),
        ))
        # Multiplicative step size for λ updates per training step.
        self.lam_alpha = float(os.environ.get("REMOE_LAMBDA_ALPHA", "0.01"))
        self.lam_init = float(os.environ.get("REMOE_LAMBDA_INIT", "1e-4"))
        # Bounds keep λ in a sane range (avoid {0, ∞}).
        self.lam_min = float(os.environ.get("REMOE_LAMBDA_MIN", "1e-8"))
        self.lam_max = float(os.environ.get("REMOE_LAMBDA_MAX", "1.0"))

        self.register_buffer("lam", torch.tensor(self.lam_init, dtype=torch.float32))
        self._fp32_lam = None

        print(
            f"[ReMoERouter] initialized: {self.num_experts} experts, "
            f"hidden_size={config.hidden_size}, target_K={self.target_K}, "
            f"lam_init={self.lam_init}, lam_alpha={self.lam_alpha}"
        )

    def forward(
        self,
        input: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        original_shape = input.shape
        input_2d = input.view(-1, original_shape[-1])
        num_tokens = input_2d.shape[0]

        logits = self.gating(input_2d).float()
        gate = torch.relu(logits)  # [num_tokens, num_experts]
        routing_map = (gate > 0)
        K = routing_map.float().sum(dim=1)  # [num_tokens]

        # FORCE_TOP1 fallback for tokens with all-negative logits.
        no_expert_mask = (K == 0)
        if int(os.environ.get("TOPANY_FORCE_TOP1", "1")):
            top1_idx = logits.argmax(dim=1)
            # Insert a small positive gate value (raw logit shifted by ε) at top-1
            # for tokens that would otherwise drop. ε ensures gradient flows.
            fallback_val = (logits.max(dim=1).values - logits.max(dim=1).values.detach() + 1e-3)
            mask_idx = no_expert_mask.unsqueeze(1)  # [num_tokens, 1]
            # Build the fallback contribution as a tensor of the same shape
            scatter_vals = torch.zeros_like(gate)
            scatter_vals.scatter_(1, top1_idx.unsqueeze(1), fallback_val.unsqueeze(1))
            gate = gate + scatter_vals * mask_idx.float()
            routing_map = routing_map | (no_expert_mask.unsqueeze(1) & (
                torch.arange(self.num_experts, device=gate.device).unsqueeze(0)
                == top1_idx.unsqueeze(1)
            ))
            K = routing_map.float().sum(dim=1)

        # Initialize fp32 shadow of λ on first forward.
        if self._fp32_lam is None or self._fp32_lam.device != input.device:
            self._fp32_lam = self.lam.clone().float().to(input.device)

        # K + load diagnostics.
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

            k_int = K.detach().long()
            counts = torch.bincount(k_int, minlength=self.num_experts + 1)
            for i in range(1, self.num_experts + 1):
                save_to_aux_losses_tracker(
                    f"topany_k_dist_{i}", counts[i].float() / num_tokens + _EPS,
                    layer_number, num_layers, reduce_op="replace",
                )

            save_to_aux_losses_tracker(
                "remoe_lambda", self._fp32_lam.detach() + _EPS,
                layer_number, num_layers, reduce_op="replace",
            )

        # Adaptive λ update + L1 sparsity penalty.
        if self.training and torch.is_grad_enabled():
            with torch.no_grad():
                current_K = K.float().mean()
                # Multiplicative update: λ↑ when too active, λ↓ when too sparse.
                if current_K > self.target_K:
                    self._fp32_lam *= (1.0 + self.lam_alpha)
                else:
                    self._fp32_lam *= (1.0 - self.lam_alpha)
                self._fp32_lam.clamp_(self.lam_min, self.lam_max)
                self.lam.copy_(self._fp32_lam)

                # Per-expert load (frequency of activation), detached so it
                # acts as a fixed weighting in the L1, not a target.
                f = routing_map.float().mean(dim=0)  # [num_experts]

            # Load-weighted L1: λ · (1/T) · Σ_t Σ_e f_e · gate_{t,e}
            # f detached; gate is the live tensor that carries gradient.
            l1 = self._fp32_lam * (f.unsqueeze(0) * gate).sum() / max(num_tokens, 1)

            if _sweep_diag_should_log(self.layer_number):
                with torch.no_grad():
                    ec = routing_map.float().sum(dim=0)
                    ec_mean = ec.mean().clamp_min(1e-6)
                    _sweep_diag_log(
                        self.layer_number,
                        routing_type="remoe",
                        k_mean=K.float().mean().item(),
                        k_std=K.float().std().item(),
                        k_max=K.float().max().item(),
                        no_expert_frac=no_expert_mask.float().mean().item(),
                        load_max_over_mean=(ec.max() / ec_mean).item(),
                        load_min_over_mean=(ec.min() / ec_mean).item(),
                        dead_count=(ec < 0.1 * ec_mean).sum().item(),
                        threshold_mean=self._fp32_lam.item(),
                        threshold_abs_max=l1.item(),
                        target_K=self.target_K,
                        update_rate=self.lam_alpha,
                        update_mode="remoe_l1",
                        num_experts=self.num_experts,
                    )

            # Apply via MoEAuxLossAutoScaler so the L1 gradient flows alongside
            # the LM gradient through the probs path.
            probs = gate.to(input.dtype)
            probs = MoEAuxLossAutoScaler.apply(probs, l1.to(probs.dtype))
        else:
            probs = gate.to(input.dtype)

        return probs, routing_map

    def routing(self, logits: torch.Tensor):
        """Not used — handled entirely in forward()."""
        raise NotImplementedError("ReMoERouter uses forward() directly, not routing().")


# ─── AdaMoE: Top-K over (real + null) experts ────────────────────────────────
#
# Zeng et al. EMNLP-Findings 2024, arXiv:2406.13233. Keeps the standard topk +
# softmax + renormalize machinery (the part that beat us in the prior sweep)
# and just enlarges the routing space with `m` null experts that always output
# 0. Token's effective real-K = number of real-expert slots in its top-(k+m').
#
# This is the diagnostic experiment: if AdaMoE matches topk (or beats it),
# the value of variable-K is real but our previous mechanisms were the wrong
# parameterization. If AdaMoE also loses by ~0.029 nats, variable-K is not the
# lever at this scale.

class AdaMoERouter(Router):
    """Top-(k+m') routing over (N real + m null) experts.

    Augments the standard router weight to shape [N+m, d]; selects top-(k+m')
    by softmax probability; null indices contribute 0; remaining real picks
    are renormalized to sum to 1. Tokens whose entire top-(k+m') was nulls
    fall back to top-1 of the real experts (FORCE_TOP1).

    Args:
        config (TransformerConfig): Megatron-Core transformer configuration.
        pg_collection (ProcessGroupCollection, optional): Process groups.
        is_mtp_layer (bool): MTP-layer flag.
        num_null (int): Number of null expert slots `m`.
            Env: ``ADAMOE_NUM_NULL`` (default 16).
        topk_aug (int): Augmented top-k (k+m'). Env: ``ADAMOE_TOPK``
            (default 3, matched with num_null=16 to give expected real-K=2).
    """

    def __init__(
        self,
        config: TransformerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
        is_mtp_layer: bool = False,
        num_null: int = 16,
        topk_aug: int = 3,
    ) -> None:
        super().__init__(config=config, pg_collection=pg_collection, is_mtp_layer=is_mtp_layer)
        if hasattr(self, 'bias') and self.bias is not None:
            del self.bias
            self.bias = None

        self.num_real = config.num_moe_experts
        self.num_null = int(os.environ.get("ADAMOE_NUM_NULL", str(num_null)))
        self.topk_aug = int(os.environ.get("ADAMOE_TOPK", str(topk_aug)))

        # Replace the parent's [N, d] weight with augmented [N+m, d].
        del self.weight
        self.weight = torch.nn.Parameter(
            torch.empty(
                (self.num_real + self.num_null, config.hidden_size),
                dtype=torch.float32,
            )
        )
        if config.perform_initialization:
            config.init_method(self.weight)
        self.weight.data = self.weight.data.to(dtype=config.params_dtype)
        setattr(self.weight, 'sequence_parallel', config.sequence_parallel)

        print(
            f"[AdaMoERouter] initialized: {self.num_real} real + {self.num_null} null "
            f"experts, top-{self.topk_aug}, expected real-K = "
            f"{self.topk_aug * self.num_real / (self.num_real + self.num_null):.2f}"
        )

    def forward(
        self,
        input: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        original_shape = input.shape
        input_2d = input.view(-1, original_shape[-1])
        num_tokens = input_2d.shape[0]

        # Augmented logits over N+m experts.
        logits = self.gating(input_2d).float()  # [num_tokens, N+m]
        full_probs = F.softmax(logits, dim=-1)

        # Top-(k+m'): selects k_aug indices per token. Probs from softmax;
        # indices may include null slots.
        topk_vals, topk_idx = full_probs.topk(self.topk_aug, dim=-1)  # [num_tokens, k_aug]

        # Mask null picks (indices ≥ num_real).
        is_real = (topk_idx < self.num_real)  # [num_tokens, k_aug]
        # Number of REAL experts this token actually routed to.
        K_real = is_real.float().sum(dim=1)  # [num_tokens]

        # Build full routing_map / weights tensor over [N+m], mask nulls, then
        # take real columns.
        full_routing = torch.zeros_like(full_probs)
        # Scatter real probs into their positions; null positions stay 0.
        full_routing.scatter_(1, topk_idx, topk_vals * is_real.float())
        weights_real = full_routing[:, : self.num_real]  # [num_tokens, N]
        routing_map = (weights_real > 0)

        # FORCE_TOP1 fallback: tokens whose entire top-(k+m') was nulls.
        no_expert_mask = (K_real == 0)
        if int(os.environ.get("TOPANY_FORCE_TOP1", "1")) and no_expert_mask.any():
            real_logits = logits[:, : self.num_real]
            top1_real_idx = real_logits.argmax(dim=1)  # [num_tokens]
            top1_real_prob = F.softmax(real_logits, dim=-1).gather(
                1, top1_real_idx.unsqueeze(1)
            ).squeeze(1)  # [num_tokens]
            # Set the top-1 real expert as the only routed pick for these tokens.
            mask = no_expert_mask.unsqueeze(1)  # [num_tokens, 1]
            scatter_vals = torch.zeros_like(weights_real)
            scatter_vals.scatter_(1, top1_real_idx.unsqueeze(1), top1_real_prob.unsqueeze(1))
            weights_real = torch.where(mask, scatter_vals, weights_real)
            routing_map = (weights_real > 0)
            K_real = routing_map.float().sum(dim=1)

        # Renormalize over real picks so each token's probs sum to 1.
        probs = (
            weights_real / weights_real.sum(dim=1, keepdim=True).clamp_min(1e-9)
        ).to(input.dtype)

        # K diagnostics.
        if self.training and torch.is_grad_enabled():
            num_layers = self.config.num_layers
            if self.config.mtp_num_layers is not None:
                num_layers += self.config.mtp_num_layers
            layer_number = self.layer_number
            if self.is_mtp_layer:
                layer_number = self.layer_number + self.config.num_layers

            save_to_aux_losses_tracker(
                "topany_k_mean", K_real.detach().float().mean(), layer_number, num_layers,
                reduce_op="replace",
            )
            save_to_aux_losses_tracker(
                "topany_k_min", K_real.detach().min().float(), layer_number, num_layers,
                reduce_op="min",
            )
            save_to_aux_losses_tracker(
                "topany_k_max", K_real.detach().max().float(), layer_number, num_layers,
                reduce_op="max",
            )
            save_to_aux_losses_tracker(
                "topany_k_std", K_real.detach().float().std() + _EPS, layer_number, num_layers,
                reduce_op="replace",
            )
            k_int = K_real.detach().long()
            counts = torch.bincount(k_int, minlength=self.num_real + 1)
            for i in range(1, self.num_real + 1):
                save_to_aux_losses_tracker(
                    f"topany_k_dist_{i}", counts[i].float() / num_tokens + _EPS,
                    layer_number, num_layers, reduce_op="replace",
                )

            null_pick_frac = (~is_real).float().mean()
            save_to_aux_losses_tracker(
                "adamoe_null_pick_frac", null_pick_frac + _EPS,
                layer_number, num_layers, reduce_op="replace",
            )

            if _sweep_diag_should_log(self.layer_number):
                with torch.no_grad():
                    ec = routing_map.float().sum(dim=0)
                    ec_mean = ec.mean().clamp_min(1e-6)
                    _sweep_diag_log(
                        self.layer_number,
                        routing_type="adamoe",
                        k_mean=K_real.float().mean().item(),
                        k_std=K_real.float().std().item(),
                        k_max=K_real.float().max().item(),
                        no_expert_frac=no_expert_mask.float().mean().item(),
                        load_max_over_mean=(ec.max() / ec_mean).item(),
                        load_min_over_mean=(ec.min() / ec_mean).item(),
                        dead_count=(ec < 0.1 * ec_mean).sum().item(),
                        threshold_mean=null_pick_frac.item(),
                        threshold_abs_max=0.0,
                        target_K=self.topk_aug * self.num_real / (self.num_real + self.num_null),
                        num_experts=self.num_real,
                        update_mode=f"m{self.num_null}_k{self.topk_aug}",
                    )

        # Optional load-balance aux loss (standard form, over the N+m space).
        if self.training and torch.is_grad_enabled():
            aux_loss_coeff = self.config.moe_aux_loss_coeff or 0.0
            if aux_loss_coeff > 0:
                num_layers = self.config.num_layers
                if self.config.mtp_num_layers is not None:
                    num_layers += self.config.mtp_num_layers
                layer_number = self.layer_number
                if self.is_mtp_layer:
                    layer_number = self.layer_number + self.config.num_layers
                # Standard switch-style: f_i * P_i, summed over experts.
                # Augmented over N+m so null experts get a balance signal too.
                pick_mask = torch.zeros_like(full_probs)
                pick_mask.scatter_(1, topk_idx, 1.0)
                f_aug = pick_mask.float().mean(dim=0)  # [N+m]
                P_aug = full_probs.mean(dim=0)         # [N+m]
                l_aux = (
                    aux_loss_coeff
                    * (f_aug * P_aug).sum()
                    * (self.num_real + self.num_null)
                )
                save_to_aux_losses_tracker(
                    "load_balancing_loss",
                    l_aux.detach() / aux_loss_coeff,
                    layer_number,
                    num_layers,
                    reduce_group=self.tp_cp_group,
                )
                probs = MoEAuxLossAutoScaler.apply(probs, l_aux.to(probs.dtype))

        return probs, routing_map

    def routing(self, logits: torch.Tensor):
        """Not used — handled entirely in forward()."""
        raise NotImplementedError("AdaMoERouter uses forward() directly, not routing().")


# ─── Dynamic Top-P (PI-controlled cumulative threshold) ──────────────────────
#
# Subclass of TopPRouter: the static ``top_p`` becomes a state variable
# updated by a Proportional-Integral controller targeting a desired
# average-K (instead of a fixed cumulative-confidence threshold).
#
# Fixes the failure modes we saw with static p=0.5:
#   - Initial collapse to K=1 (highest score alone covers p=0.5).
#   - Eventual runaway to K=9 (scores collapsed under aux-loss pressure).
#
# Reference: Sparsity-Controllable Dynamic Top-p MoE, arXiv:2512.13996 (2025).

class DynamicTopPRouter(TopPRouter):
    """Top-P router with PI-controlled cumulative threshold.

    Maintains ``self.top_p`` as a buffer; at the end of each training forward
    measures actual average-K and applies a PI update toward the target.

    Args:
        config: Megatron-Core transformer configuration.
        pg_collection (optional): Process groups.
        is_mtp_layer (bool): MTP-layer flag.
        target_K (float, optional): Desired average K. Env ``DTOPP_TARGET_K``.
        kp (float): Proportional gain. Env ``DTOPP_KP``.
        ki (float): Integral gain.    Env ``DTOPP_KI``.
        p_init (float): Initial p value. Env ``DTOPP_P_INIT``.
    """

    def __init__(
        self,
        config: TransformerConfig,
        pg_collection: Optional[ProcessGroupCollection] = None,
        is_mtp_layer: bool = False,
        target_K: Optional[float] = None,
        kp: float = 0.05,
        ki: float = 0.005,
        p_init: float = 1.0,
    ) -> None:
        # Initialize parent with the initial p value.
        os.environ.setdefault("TOPP_THRESHOLD", str(os.environ.get("DTOPP_P_INIT", str(p_init))))
        super().__init__(config=config, pg_collection=pg_collection, is_mtp_layer=is_mtp_layer)

        self.target_K = float(os.environ.get(
            "DTOPP_TARGET_K",
            str(target_K if target_K is not None else
                getattr(config, 'moe_topany_target_k', 2.0)),
        ))
        self.kp = float(os.environ.get("DTOPP_KP", str(kp)))
        self.ki = float(os.environ.get("DTOPP_KI", str(ki)))
        self.p_min = float(os.environ.get("DTOPP_P_MIN", "0.05"))
        self.p_max = float(os.environ.get("DTOPP_P_MAX", "8.0"))

        # State for PI controller. Stored as Python floats — small scalars,
        # no gradient, no need for buffers.
        self._error_integral = 0.0

        print(
            f"[DynamicTopPRouter] init p={self.top_p}, target_K={self.target_K}, "
            f"kp={self.kp}, ki={self.ki}"
        )

    def forward(
        self,
        input: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Run the parent forward at the current (frozen) p.
        probs, routing_map = super().forward(input, padding_mask)

        # PI update for next step. p increases when K too low, decreases when
        # K too high. Sign convention: error = target_K - measured_K.
        if self.training:
            with torch.no_grad():
                current_K = routing_map.float().sum(dim=1).mean().item()
                error = self.target_K - current_K
                self._error_integral += error
                new_p = self.top_p + self.kp * error + self.ki * self._error_integral
                new_p = max(self.p_min, min(self.p_max, new_p))
                # Anti-windup on integral when clamped.
                if new_p == self.p_min or new_p == self.p_max:
                    self._error_integral -= error
                self.top_p = new_p

        return probs, routing_map

    def routing(self, logits: torch.Tensor):
        """Not used — handled entirely in forward()."""
        raise NotImplementedError("DynamicTopPRouter uses forward() directly, not routing().")

