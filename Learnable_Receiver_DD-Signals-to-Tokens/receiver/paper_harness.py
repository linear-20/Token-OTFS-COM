"""Paper-finalization harness (Step 17).

Reports, metrics, ablations, and complexity estimates for the fixed
sparse OTFS receiver chain.  No algorithm changes -- only diagnostics,
counting, and config management.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import torch

from .config import ReceiverConfig

if TYPE_CHECKING:
    from .model import LearnableOTFSReceiver, ReceiverOutput


# --- dataclasses -------------------------------------------------------

@dataclass
class ReceiverComplexityReport:
    """Parameter counts and estimated operations.

    Attributes:
        total_parameters: Total parameters (trainable + frozen).
        trainable_parameters: Parameters with requires_grad=True.
        classifier_parameters: Parameters in token classifier.
        denoiser_parameters: Parameters in residual DD denoiser.
        equalizer_parameters: Parameters in unfolded equalizer.
        offgrid_parameters: Parameters in off-grid refinement net.
        fusion_parameters: Parameters in logit fusion module.
        other_parameters: Remaining (CE, DD operator, etc).
        estimated_sparse_operator_ops: Per-sample analytic op count.
        estimated_detector_ops: Per-sample analytic op count.
        estimated_classifier_ops: Per-sample analytic op count.
        estimated_token_prox_ops: Per-sample analytic op count.
        notes: Arbitrary supplementary key-value pairs.
    """

    total_parameters: int = 0
    trainable_parameters: int = 0
    classifier_parameters: int = 0
    denoiser_parameters: int = 0
    equalizer_parameters: int = 0
    offgrid_parameters: int = 0
    fusion_parameters: int = 0
    other_parameters: int = 0
    estimated_sparse_operator_ops: int = 0
    estimated_detector_ops: int = 0
    estimated_classifier_ops: int = 0
    estimated_token_prox_ops: int = 0
    notes: dict[str, object] = field(default_factory=dict)


@dataclass
class ReceiverPaperTrace:
    """Traceable paper-level metadata from a single forward pass.

    Attributes:
        config_snapshot: Subset of ReceiverConfig values.
        enabled_modules: Which stages were active.
        tensor_shapes: Key output shapes.
        scalar_metrics: Summary scalars from the forward pass.
        algorithm_flags: Derived algorithm settings.
        ablation_name: Optional ablation label.
    """

    config_snapshot: dict[str, object] = field(default_factory=dict)
    enabled_modules: dict[str, bool] = field(default_factory=dict)
    tensor_shapes: dict[str, tuple] = field(default_factory=dict)
    scalar_metrics: dict[str, float] = field(default_factory=dict)
    algorithm_flags: dict[str, object] = field(default_factory=dict)
    ablation_name: str | None = None


# --- config summary ----------------------------------------------------

def summarize_receiver_config(config: ReceiverConfig) -> dict[str, object]:
    """Return a dictionary of paper-relevant config keys."""
    return {
        "M": config.M,
        "N": config.N,
        "vocab_size": config.vocab_size,
        "token_embedding_dim": config.token_embedding_dim,
        "topk_paths": config.topk_paths,
        "hidden_channels": config.hidden_channels,
        "num_unfolded_layers": config.num_unfolded_layers,
        "channel_estimator_mode": config.channel_estimator_mode,
        "pilot_delay": config.pilot_delay,
        "pilot_doppler": config.pilot_doppler,
        "pilot_guard_delay": config.pilot_guard_delay,
        "pilot_guard_doppler": config.pilot_guard_doppler,
        "pilot_obs_delay_radius": config.pilot_obs_delay_radius,
        "pilot_obs_doppler_radius": config.pilot_obs_doppler_radius,
        "use_offgrid_refinement": config.use_offgrid_refinement,
        "offgrid_kernel_type": config.offgrid_kernel_type,
        "dd_operator_mode": config.dd_operator_mode,
        "detector_prox_mode": config.detector_prox_mode,
        "detector_num_layers": config.num_unfolded_layers,
        "token_posterior_topk": config.token_posterior_topk,
        "token_candidate_pruning_mode": config.token_candidate_pruning_mode,
        "token_candidate_sketch_mode": config.token_candidate_sketch_mode,
        "token_candidate_count": config.token_candidate_count,
        "token_candidate_adaptive": config.token_candidate_adaptive,
        "token_candidate_min_count": config.token_candidate_min_count,
        "token_candidate_max_count": config.token_candidate_max_count,
        "token_candidate_entropy_threshold": config.token_candidate_entropy_threshold,
        "token_candidate_margin_threshold": config.token_candidate_margin_threshold,
        "token_candidate_expand_factor": config.token_candidate_expand_factor,
        "token_candidate_allow_full_fallback": config.token_candidate_allow_full_fallback,
        "token_candidate_reliability_floor": config.token_candidate_reliability_floor,
        "token_candidate_score_mode": config.token_candidate_score_mode,
        "noise_var": config.noise_var,
        "use_denoiser": (
            config.use_denoiser if config.use_denoiser is not None
            else config.use_refinement_net
        ),
        "use_data_consistency_correction": config.use_data_consistency_correction,
        "classifier_num_evidence_heads": config.classifier_num_evidence_heads,
        "classifier_head_fusion": config.classifier_head_fusion,
        "use_token_logit_fusion": config.use_token_logit_fusion,
        "token_logit_fusion_mode": config.token_logit_fusion_mode,
        "use_physics_guided_gain_refinement": config.use_physics_guided_gain_refinement,
        "use_variance_tracking": config.use_variance_tracking,
    }


# --- parameter counting ------------------------------------------------

def count_receiver_parameters(model: "LearnableOTFSReceiver") -> ReceiverComplexityReport:
    """Count parameters per functional block."""
    report = ReceiverComplexityReport()

    def _count(module) -> int:
        return int(sum(p.numel() for p in module.parameters()))

    def _trainable(module) -> int:
        return int(sum(p.numel() for p in module.parameters() if p.requires_grad))

    total = _count(model)
    trainable = _trainable(model)

    # classifier
    if hasattr(model, "token_classifier"):
        report.classifier_parameters = _count(model.token_classifier)

    # denoiser
    if hasattr(model, "denoiser") and model.denoiser is not None:
        report.denoiser_parameters = _count(model.denoiser)

    # equalizer
    if hasattr(model, "equalizer"):
        report.equalizer_parameters = _count(model.equalizer)

    # offgrid / refinement
    if hasattr(model, "sparse_ce") and hasattr(model.sparse_ce, "offgrid_refinement_net"):
        if model.sparse_ce.offgrid_refinement_net is not None:
            report.offgrid_parameters = _count(model.sparse_ce.offgrid_refinement_net)

    # fusion
    if hasattr(model, "logit_fusion") and model.logit_fusion is not None:
        report.fusion_parameters = _count(model.logit_fusion)

    known = (report.classifier_parameters + report.denoiser_parameters
             + report.equalizer_parameters + report.offgrid_parameters
             + report.fusion_parameters)
    report.other_parameters = max(0, total - known)
    report.total_parameters = total
    report.trainable_parameters = trainable
    return report


# --- rough op estimation -----------------------------------------------

def estimate_receiver_ops(config: ReceiverConfig) -> dict[str, int]:
    """Analytic rough operation counts (per-sample, multiply-add equivalents).

    Returns a dict of string keys to integer estimates.  ``B`` = 1 is
    assumed; per-batch costs scale linearly.
    """
    M = config.M
    N = config.N
    K = config.topk_paths
    L = config.num_unfolded_layers
    V = config.vocab_size
    D = config.token_embedding_dim
    H = config.classifier_num_evidence_heads
    hidden = config.hidden_channels

    ops: dict[str, int] = {}

    # sparse DD operator: one apply or matched_filter = O(K * M * N)
    ops["sparse_operator_per_pass"] = K * M * N

    # detector: each layer has H.apply + H^H.apply + L1/prox + variance
    detector_ops = L * (2 * K * M * N + M * N)  # apply + matched_filter + soft-threshold
    ops["detector"] = detector_ops

    # token posterior prox
    pruning_mode = config.token_candidate_pruning_mode
    candidate_count = config.token_candidate_count or config.token_posterior_topk
    sketch_size = config.token_candidate_sketch_size
    if sketch_size is None:
        sketch_size = min(M * N, 64)
    if config.detector_prox_mode == "l1":
        ops["token_prox"] = 0
    elif pruning_mode in ("sketch", "explicit") and candidate_count is not None:
        Kc = candidate_count
        if config.token_candidate_adaptive and config.token_candidate_max_count is not None:
            Kc = max(Kc, config.token_candidate_max_count)
        ops["token_prox_sketch"] = V * sketch_size + Kc * M * N
        ops["token_prox"] = ops["token_prox_sketch"]
    else:
        ops["token_prox_full_vocab"] = V * M * N
        ops["token_prox"] = ops["token_prox_full_vocab"]

    # classifier: CNN + H*D per position + H*V*D for codebook similarity
    cnn_ops = hidden * M * N * 9  # rough 3x3 conv
    embed_ops = H * D * M * N
    sim_ops = H * V * D
    ops["classifier"] = cnn_ops + embed_ops + sim_ops

    # data consistency: one H.apply + H^H.apply
    ops["data_consistency"] = 2 * K * M * N

    return ops


# --- paper trace -------------------------------------------------------

def build_receiver_paper_trace(
    output: "ReceiverOutput",
    config: ReceiverConfig,
    ablation_name: str | None = None,
) -> ReceiverPaperTrace:
    """Build a ReceiverPaperTrace from a forward pass."""
    trace = ReceiverPaperTrace(ablation_name=ablation_name)

    # config snapshot
    trace.config_snapshot = summarize_receiver_config(config)

    # enabled modules
    trace.enabled_modules = {
        "offgrid_refinement": config.use_offgrid_refinement,
        "physics_gain_refinement": config.use_physics_guided_gain_refinement,
        "dd_operator_offgrid": (
            config.dd_operator_mode == "offgrid"
            or (config.dd_operator_mode == "auto" and config.use_offgrid_refinement)
        ),
        "token_posterior_prox": config.detector_prox_mode in ("token_posterior", "hybrid"),
        "candidate_pruning": config.token_candidate_pruning_mode != "none",
        "variance_tracking": config.use_variance_tracking,
        "denoiser": (
            config.use_denoiser if config.use_denoiser is not None
            else config.use_refinement_net
        ),
        "data_consistency_correction": config.use_data_consistency_correction,
        "multihead_classifier": config.classifier_num_evidence_heads > 1,
        "logit_fusion": config.use_token_logit_fusion,
    }

    # tensor shapes
    shapes: dict[str, tuple] = {}
    if output.x_equalized is not None:
        shapes["x_equalized"] = tuple(output.x_equalized.shape)
    if output.x_refined is not None:
        shapes["x_refined"] = tuple(output.x_refined.shape)
    if output.x_consistent is not None:
        shapes["x_consistent"] = tuple(output.x_consistent.shape)
    if output.token_logits is not None:
        shapes["token_logits"] = tuple(output.token_logits.shape)
    if output.rx_embedding is not None:
        shapes["rx_embedding"] = tuple(output.rx_embedding.shape)
    if output.sparse_estimate is not None:
        shapes["h_dd"] = tuple(output.sparse_estimate.h_dd.shape)
        shapes["path_indices"] = tuple(output.sparse_estimate.path_indices.shape)
    trace.tensor_shapes = shapes

    # scalar metrics from receiver_trace
    rt = output.receiver_trace or {}
    metrics: dict[str, float] = {}
    for key in ("classifier_confidence_mean", "classifier_entropy_mean",
                "reliability_scalar_mean", "expected_error_proxy_mean",
                "token_logit_fusion_gate_mean", "dc_residual_energy_mean",
                "physics_residual_power_mean",
                "classifier_head_entropy_mean", "classifier_head_fusion_entropy_mean",
                "token_candidate_entropy_last_mean", "token_candidate_margin_last_mean",
                "token_candidate_fallback_rate_last",
                "physics_fit_residual_power_mean", "physics_full_residual_power_mean"):
        val = rt.get(key)
        if val is not None and isinstance(val, (int, float)):
            metrics[key] = float(val)
    trace.scalar_metrics = metrics

    # algorithm flags
    trace.algorithm_flags = {
        "operator_mode": config.dd_operator_mode,
        "prox_mode": config.detector_prox_mode,
        "candidate_mode": config.token_candidate_pruning_mode,
        "classifier_heads": config.classifier_num_evidence_heads,
        "head_fusion": config.classifier_head_fusion,
        "logit_fusion_mode": config.token_logit_fusion_mode,
        "channel_estimator": config.channel_estimator_mode,
    }

    return trace


# --- paper ablations ---------------------------------------------------

def named_receiver_paper_ablation(name: str, config: ReceiverConfig) -> ReceiverConfig:
    """Return a paper-level ablation configuration.

    Supported names: "full", "no_offgrid", "no_physics_gain_refinement",
    "ongrid_operator", "no_token_posterior_prox", "no_candidate_pruning",
    "no_denoiser", "no_data_consistency", "single_head_classifier",
    "no_logit_fusion", "classifier_only", "no_reliability_calibration".

    Args:
        name: Ablation name.
        config: Baseline ReceiverConfig.

    Returns:
        ReceiverConfig with ablation flags changed.
    """
    if name == "full":
        return config
    if name == "no_offgrid":
        return replace(config, use_offgrid_refinement=False, dd_operator_mode="ongrid")
    if name == "no_physics_gain_refinement":
        return replace(config, use_physics_guided_gain_refinement=False)
    if name == "ongrid_operator":
        return replace(config, dd_operator_mode="ongrid")
    if name == "no_token_posterior_prox":
        return replace(config, detector_prox_mode="l1")
    if name == "no_candidate_pruning":
        return replace(config, token_candidate_pruning_mode="none")
    if name == "no_denoiser":
        return replace(config, use_denoiser=False)
    if name == "no_data_consistency":
        return replace(config, use_data_consistency_correction=False)
    if name == "single_head_classifier":
        return replace(config, classifier_num_evidence_heads=1, classifier_head_fusion="mean")
    if name == "no_logit_fusion":
        return replace(config, use_token_logit_fusion=False)
    if name == "classifier_only":
        return replace(
            config,
            detector_prox_mode="l1",
            use_denoiser=False,
            use_data_consistency_correction=False,
            use_token_logit_fusion=False,
            token_candidate_pruning_mode="none",
        )
    if name == "no_adaptive_candidate_pruning":
        return replace(config, token_candidate_adaptive=False)
    if name == "no_energy_weighted_candidate_score":
        return replace(config, token_candidate_score_mode="distance")
    if name == "no_reliability_calibration":
        # Calibration is a loss/evaluation setting rather than an inference
        # architecture switch.  Keep the receiver config unchanged; disable
        # calibration through ReceiverLossWeights in the experiment harness.
        return config
    raise ValueError(f"Unknown paper ablation: {name}")


# --- full summary ------------------------------------------------------

def receiver_paper_summary(
    model: "LearnableOTFSReceiver",
    config: ReceiverConfig,
    output: "ReceiverOutput | None" = None,
    ablation_name: str | None = None,
) -> dict[str, object]:
    """Return a combined paper summary dict.

    Keys: "config", "complexity", and optionally "trace".
    """
    summary: dict[str, object] = {
        "config": summarize_receiver_config(config),
        "complexity": count_receiver_parameters(model),
        "ops": estimate_receiver_ops(config),
    }
    if output is not None:
        summary["trace"] = build_receiver_paper_trace(
            output, config, ablation_name=ablation_name,
        )
    return summary
