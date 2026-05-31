"""Named paper ablation configurations for the sparse OTFS receiver."""

from __future__ import annotations

from dataclasses import replace

from .config import ReceiverConfig


def named_receiver_ablation(name: str, config: ReceiverConfig) -> ReceiverConfig:
    """Return a named receiver ablation config for DD tensors [B, M, N].

    Args:
        name: Ablation name. Supported names are "full", "no_offgrid",
            "ongrid_operator", "no_denoiser",
            "no_equalizer_residual_uncertainty", "no_classifier_uncertainty",
            and "fallback_ce_baseline".
        config: ReceiverConfig describing receiver tensor shapes [B, M, N]
            and token logits [B, vocab_size].

    Returns:
        ReceiverConfig with only the named ablation switches changed.
    """

    if name == "full":
        return config
    if name == "no_offgrid":
        return replace(config, use_offgrid_refinement=False, dd_operator_mode="ongrid")
    if name == "ongrid_operator":
        return replace(config, dd_operator_mode="ongrid")
    if name == "no_denoiser":
        return replace(config, use_denoiser=False)
    if name == "no_equalizer_residual_uncertainty":
        return replace(config, use_equalizer_residual_for_denoiser=False)
    if name == "no_classifier_uncertainty":
        return replace(config, classifier_use_uncertainty=False)
    if name == "fallback_ce_baseline":
        return replace(
            config,
            channel_estimator_mode="fallback_topk",
            use_offgrid_refinement=False,
            dd_operator_mode="ongrid",
        )
    if name == "no_data_consistency_correction":
        return replace(config, use_data_consistency_correction=False)
    raise ValueError(f"Unknown receiver ablation: {name}")
