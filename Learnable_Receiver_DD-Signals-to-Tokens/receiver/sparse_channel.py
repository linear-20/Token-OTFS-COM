"""Sparse DD channel support estimation for the fixed OTFS receiver.

All public tensors use DD shape [B, M, N]. The MVP fallback uses y_dd as a
coarse channel proxy when h_dd is not provided; this is not a full pilot-aided
channel estimator.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.nn as nn
import torch.nn.functional as F

from .complex_utils import (
    mask_to_real,
    validate_complex_dd,
    validate_optional_complex_dd,
    validate_support_mask,
)
from .config import ReceiverConfig
from .offgrid_refinement import OffGridRefinementNet, apply_offgrid_refinement
from .pilot_ce import EmbeddedPilotConfig, PilotSparseChannelEstimator


@dataclass
class SparseChannelEstimate:
    """Sparse channel estimate for downstream sparse DD operators.

    Attributes:
        h_dd: Complex sparse channel estimate with shape [B, M, N].
        support_mask: Float support mask with shape [B, M, N].
        path_indices: Long path indices with shape [B, K, 2], ordered as
            [delay_idx, doppler_idx].
        path_gains: Complex path gains with shape [B, K].
        confidence: Optional float confidence with shape [B, K].
        support_logits: Optional float logits with shape [B, M, N].
        path_confidence_map: Optional float confidence map with shape [B, M, N].
        fractional_offsets: Optional float off-grid offsets with shape [B, K, 2].
            These are estimated after on-grid CE when off-grid refinement is
            enabled. They prepare the later parametric H_theta operator.
        pilot_residual: Optional complex pilot-window residual with shape
            [B, P, Q].
        pilot_residual_power: Optional float pilot residual power with shape
            [B].
        noise_var: Optional scalar noise variance used by pilot CE.
        physics_residual: Optional complex physics fitting residual with shape
            [B, M, N].
        physics_residual_power: Optional float residual power with shape [B].
        gain_covariance_diag: Optional float gain covariance diagonal with
            shape [B, K].
        physics_normal_diag: Optional float normal matrix diagonal with shape
            [B, K].
        physics_refined: Whether physics-guided gain refinement was applied.
        estimator_mode: Estimator mode string: "oracle", "pilot", or
            "fallback_topk".
        ce_fit_mask_source: How the CE fitting mask was obtained:
            "pilot_observation", "explicit_ce_data_mask", "none", or
            "full_grid_debug".
        physics_probe_position: Optional (delay, doppler) tuple of the probe
            impulse used for physics dictionary construction.
    """

    h_dd: torch.Tensor
    support_mask: torch.Tensor
    path_indices: torch.Tensor
    path_gains: torch.Tensor
    confidence: torch.Tensor | None = None
    support_logits: torch.Tensor | None = None
    path_confidence_map: torch.Tensor | None = None
    fractional_offsets: torch.Tensor | None = None
    pilot_residual: torch.Tensor | None = None
    pilot_residual_power: torch.Tensor | None = None
    noise_var: float | None = None
    physics_residual: torch.Tensor | None = None
    physics_residual_power: torch.Tensor | None = None
    physics_fit_residual_power: torch.Tensor | None = None
    physics_full_residual_power: torch.Tensor | None = None
    gain_covariance_diag: torch.Tensor | None = None
    physics_normal_diag: torch.Tensor | None = None
    physics_refined: bool = False
    estimator_mode: str = "fallback_topk"
    ce_fit_mask_source: str = "none"
    physics_probe_position: tuple[int, int] | None = None


def _validate_k(k: int, total_bins: int) -> int:
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer.")
    return min(k, total_bins)


def _flat_to_delay_doppler(flat_indices: torch.Tensor, n_doppler: int) -> torch.Tensor:
    """Convert flat indices [B, K] to [delay_idx, doppler_idx] [B, K, 2]."""

    delay_idx = torch.div(flat_indices, n_doppler, rounding_mode="floor")
    doppler_idx = flat_indices.remainder(n_doppler)
    return torch.stack((delay_idx, doppler_idx), dim=-1).long()


def _gather_path_gains(h_dd: torch.Tensor, path_indices: torch.Tensor) -> torch.Tensor:
    """Gather complex path gains from h_dd [B, M, N] at path_indices [B, K, 2]."""

    batch = torch.arange(h_dd.shape[0], device=h_dd.device).unsqueeze(-1)
    return h_dd[batch, path_indices[..., 0], path_indices[..., 1]]


def _scatter_path_gains_to_h_dd(
    path_gains: torch.Tensor,
    path_indices: torch.Tensor,
    support_mask: torch.Tensor,
    confidence: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scatter path_gains [B, K] to diagnostic h_dd projection [B, M, N]."""

    batch_size, delay_bins, doppler_bins = support_mask.shape
    flat_indices = path_indices[..., 0] * doppler_bins + path_indices[..., 1]
    valid = torch.ones_like(path_gains.real, dtype=torch.bool)
    if confidence is not None:
        valid = confidence > 0
    values = torch.where(valid, path_gains, torch.zeros_like(path_gains))
    flat = torch.zeros(batch_size, delay_bins * doppler_bins, device=path_gains.device, dtype=path_gains.dtype)
    flat.scatter_add_(dim=1, index=flat_indices.to(device=path_gains.device), src=values)
    return flat.reshape(batch_size, delay_bins, doppler_bins) * support_mask.to(
        device=path_gains.device,
        dtype=path_gains.real.dtype,
    )


def _scatter_path_confidence_to_map(
    confidence: torch.Tensor,
    path_indices: torch.Tensor,
    support_mask: torch.Tensor,
) -> torch.Tensor:
    """Scatter confidence [B, K] to path confidence map [B, M, N]."""

    batch_size, delay_bins, doppler_bins = support_mask.shape
    flat_indices = path_indices[..., 0] * doppler_bins + path_indices[..., 1]
    flat = torch.zeros(batch_size, delay_bins * doppler_bins, device=confidence.device, dtype=confidence.dtype)
    flat.scatter_add_(dim=1, index=flat_indices.to(device=confidence.device), src=confidence.clamp(0.0, 1.0))
    return flat.reshape(batch_size, delay_bins, doppler_bins).clamp(0.0, 1.0)


def _estimate_from_flat_indices(
    coarse_h_dd: torch.Tensor,
    flat_indices: torch.Tensor,
    confidence: torch.Tensor,
    support_mask: torch.Tensor,
    support_logits: torch.Tensor | None = None,
    estimator_mode: str = "fallback_topk",
) -> SparseChannelEstimate:
    """Build SparseChannelEstimate from flat indices [B, K] and mask [B, M, N]."""

    path_indices = _flat_to_delay_doppler(flat_indices, coarse_h_dd.shape[-1])
    h_sparse = coarse_h_dd * support_mask.to(dtype=coarse_h_dd.real.dtype)
    path_gains = _gather_path_gains(h_sparse, path_indices)
    valid_gain = confidence > 0
    path_gains = torch.where(valid_gain, path_gains, torch.zeros_like(path_gains))
    return SparseChannelEstimate(
        h_dd=h_sparse,
        support_mask=support_mask,
        path_indices=path_indices,
        path_gains=path_gains,
        confidence=confidence,
        support_logits=support_logits,
        estimator_mode=estimator_mode,
    )


class TopKSparseEstimator(nn.Module):
    """Select top-K DD taps by abs(coarse_h_dd) for debug/smoke tests.

    Input:
        coarse_h_dd: Complex tensor with shape [B, M, N].
        k: Optional number of taps K. Defaults to config.topk_paths.

    Output:
        SparseChannelEstimate with h_dd [B, M, N], support_mask [B, M, N],
        path_indices [B, K, 2], path_gains [B, K], confidence [B, K].

    Note:
        This is not the paper receiver's channel estimator. It is retained as
        fallback_topk for legacy tests, shape smoke tests, and debugging.
    """

    def __init__(self, config: ReceiverConfig):
        super().__init__()
        self.config = config

    def forward(self, coarse_h_dd: torch.Tensor, k: int | None = None) -> SparseChannelEstimate:
        """Return top-K sparse estimate from coarse_h_dd [B, M, N]."""

        validate_complex_dd(coarse_h_dd, self.config, "coarse_h_dd")
        requested_k = self.config.topk_paths if k is None else k
        total_bins = self.config.M * self.config.N
        actual_k = _validate_k(requested_k, total_bins)

        flat_abs = coarse_h_dd.abs().reshape(coarse_h_dd.shape[0], -1)
        values, flat_indices = torch.topk(flat_abs, k=actual_k, dim=-1)

        if actual_k < requested_k:
            pad_count = requested_k - actual_k
            flat_indices = F.pad(flat_indices, (0, pad_count))
            values = F.pad(values, (0, pad_count))

        flat_mask = torch.zeros_like(flat_abs, dtype=coarse_h_dd.real.dtype)
        flat_mask.scatter_(dim=-1, index=flat_indices[:, :actual_k], value=1.0)
        support_mask = flat_mask.reshape(coarse_h_dd.shape[0], self.config.M, self.config.N)
        return _estimate_from_flat_indices(coarse_h_dd, flat_indices, values, support_mask)


class ThresholdSparseEstimator(nn.Module):
    """Select DD taps whose abs(coarse_h_dd) exceeds a threshold.

    Input:
        coarse_h_dd: Complex tensor with shape [B, M, N].
        threshold: Scalar threshold applied to abs(coarse_h_dd).

    Output:
        SparseChannelEstimate with h_dd [B, M, N], support_mask [B, M, N],
        path_indices [B, K, 2], path_gains [B, K], confidence [B, K], where
        K is config.topk_paths and padded entries have zero confidence.

    Note:
        This is a legacy on-grid utility, not the embedded-pilot paper CE path.
    """

    def __init__(self, config: ReceiverConfig):
        super().__init__()
        self.config = config

    def forward(self, coarse_h_dd: torch.Tensor, threshold: float) -> SparseChannelEstimate:
        """Return thresholded sparse estimate from coarse_h_dd [B, M, N]."""

        validate_complex_dd(coarse_h_dd, self.config, "coarse_h_dd")
        if not isinstance(threshold, (int, float)):
            raise TypeError("threshold must be a scalar float.")

        batch_size = coarse_h_dd.shape[0]
        requested_k = self.config.topk_paths
        actual_k = min(requested_k, self.config.M * self.config.N)
        flat_abs = coarse_h_dd.abs().reshape(batch_size, -1)
        candidate_abs = torch.where(
            flat_abs >= float(threshold),
            flat_abs,
            torch.full_like(flat_abs, -1.0),
        )
        values, flat_indices = torch.topk(candidate_abs, k=actual_k, dim=-1)
        valid = values >= float(threshold)

        if actual_k < requested_k:
            pad_count = requested_k - actual_k
            values = F.pad(values, (0, pad_count), value=-1.0)
            flat_indices = F.pad(flat_indices, (0, pad_count))
            valid = F.pad(valid, (0, pad_count), value=False)

        confidence = torch.where(valid, values.clamp_min(0.0), torch.zeros_like(values))
        flat_mask = torch.zeros_like(flat_abs, dtype=coarse_h_dd.real.dtype)
        flat_mask.scatter_add_(dim=-1, index=flat_indices, src=valid.to(dtype=coarse_h_dd.real.dtype))
        flat_mask = flat_mask.clamp_max(1.0)
        support_mask = flat_mask.reshape(batch_size, self.config.M, self.config.N)
        flat_indices = torch.where(valid, flat_indices, torch.zeros_like(flat_indices))
        return _estimate_from_flat_indices(coarse_h_dd, flat_indices, confidence, support_mask)


class SparseRefinementNet(nn.Module):
    """Light refinement constrained by sparse support/gating.

    Inputs:
        coarse_h_dd: Complex coarse channel with shape [B, M, N].
        support_mask: Float support mask with shape [B, M, N].
        snr_db: Optional scalar or [B] tensor.

    Outputs:
        refined_h_dd: Complex refined sparse channel with shape [B, M, N].
        support_logits: Float support logits with shape [B, M, N].
    """

    def __init__(self, config: ReceiverConfig):
        super().__init__()
        self.config = config
        in_channels = 5  # real, imag, abs, support_mask, snr map
        hidden = config.hidden_channels
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.delta_head = nn.Conv2d(hidden, 2, kernel_size=1)
        self.support_head = nn.Conv2d(hidden, 1, kernel_size=1)

    def forward(
        self,
        coarse_h_dd: torch.Tensor,
        support_mask: torch.Tensor,
        snr_db: torch.Tensor | float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Refine coarse_h_dd [B, M, N] under support_mask [B, M, N]."""

        validate_complex_dd(coarse_h_dd, self.config, "coarse_h_dd")
        validate_support_mask(support_mask, self.config, "support_mask")
        if support_mask.shape[0] != coarse_h_dd.shape[0]:
            raise ValueError("support_mask batch size must match coarse_h_dd.")

        support = support_mask.to(device=coarse_h_dd.device, dtype=coarse_h_dd.real.dtype)
        snr_map = self._snr_map(snr_db, coarse_h_dd)
        inputs = torch.stack(
            (
                coarse_h_dd.real,
                coarse_h_dd.imag,
                coarse_h_dd.abs(),
                support,
                snr_map,
            ),
            dim=1,
        )
        hidden = self.features(inputs)
        delta_channels = self.delta_head(hidden)
        delta = torch.complex(delta_channels[:, 0], delta_channels[:, 1])
        support_logits = self.support_head(hidden).squeeze(1)

        learned_gate = torch.sigmoid(support_logits)
        sparse_gate = support * learned_gate
        refined_h_dd = (coarse_h_dd + delta) * sparse_gate
        return refined_h_dd, support_logits

    def _snr_map(self, snr_db: torch.Tensor | float | None, coarse_h_dd: torch.Tensor) -> torch.Tensor:
        """Return normalized SNR map with shape [B, M, N]."""

        if snr_db is None:
            snr = torch.zeros(coarse_h_dd.shape[0], device=coarse_h_dd.device, dtype=coarse_h_dd.real.dtype)
        else:
            snr = torch.as_tensor(snr_db, device=coarse_h_dd.device, dtype=coarse_h_dd.real.dtype)
            if snr.ndim == 0:
                snr = snr.expand(coarse_h_dd.shape[0])
            elif snr.shape != (coarse_h_dd.shape[0],):
                raise ValueError("snr_db must be scalar or have shape [B].")
        snr = snr / 30.0
        return snr.reshape(-1, 1, 1).expand(-1, self.config.M, self.config.N)


class SparseChannelEstimator(nn.Module):
    """Build sparse DD support/channel estimate from y_dd [B, M, N].

    This module preserves the fixed receiver stage Y_DD -> sparse CE/support.
    The paper path is mode="pilot", which estimates sparse on-grid channel
    taps from an embedded pilot observation window. mode="oracle" uses a
    supplied h_dd [B, M, N] for controlled evaluation. mode="fallback_topk"
    retains the original y_dd/coarse_h_dd top-k behavior only for debugging
    and smoke tests; it is not intended for paper experiments.
    """

    def __init__(self, config: ReceiverConfig):
        super().__init__()
        self.config = config
        self.topk_estimator = TopKSparseEstimator(config)
        use_channel_refinement = bool(getattr(config, "use_channel_refinement", False))
        self.refinement_net = SparseRefinementNet(config) if use_channel_refinement else None
        use_offgrid_refinement = bool(getattr(config, "use_offgrid_refinement", False))
        self.offgrid_refinement_net = (
            OffGridRefinementNet(
                radius_delay=config.offgrid_patch_delay_radius,
                radius_doppler=config.offgrid_patch_doppler_radius,
                max_offset=config.offgrid_max_offset,
                confidence_floor=config.offgrid_confidence_floor,
                gain_correction_scale=config.offgrid_gain_correction_scale,
                hidden_channels=config.hidden_channels,
            )
            if use_offgrid_refinement
            else None
        )
        if bool(getattr(config, "use_physics_guided_gain_refinement", False)):
            from .physics_refinement import PhysicsGuidedPathGainRefiner

            self.physics_refiner = PhysicsGuidedPathGainRefiner(config)
        else:
            self.physics_refiner = None

    def forward(
        self,
        y_dd: torch.Tensor,
        h_dd: torch.Tensor | None = None,
        support_mask: torch.Tensor | None = None,
        snr_db: torch.Tensor | float | None = None,
        mode: str | None = None,
        pilot_config: EmbeddedPilotConfig | None = None,
        pilot_layout: EmbeddedPilotConfig | None = None,
        pilot_threshold: float | None = None,
        noise_var: float | None = None,
        data_mask: torch.Tensor | None = None,
        ce_fit_mask: torch.Tensor | None = None,
    ) -> SparseChannelEstimate:
        """Estimate sparse DD support/channel from y_dd [B, M, N].

        Args:
            y_dd: Complex received DD grid with shape [B, M, N].
            h_dd: Optional complex coarse/known channel with shape [B, M, N].
            support_mask: Optional bool/float support mask with shape [B, M, N].
            snr_db: Optional scalar or [B] tensor for refinement features.
            mode: Optional CE mode: "oracle", "pilot", or "fallback_topk".
            pilot_config: Optional EmbeddedPilotConfig for pilot CE.
            pilot_layout: Optional alias for pilot_config.
            pilot_threshold: Optional pilot CE magnitude threshold.
            noise_var: Optional scalar noise variance metadata for pilot CE.
            data_mask: Deprecated alias for ``ce_fit_mask``.  Prefer
                ``ce_fit_mask`` for physics-guided gain refinement.
            ce_fit_mask: Optional real/bool CE *observation* fitting mask with
                shape [B, M, N] or [1, M, N].  Only positions inside the
                known CE observation window (pilot region) should be active.

        Returns:
            SparseChannelEstimate.
        """

        validate_complex_dd(y_dd, self.config, "y_dd")
        validate_optional_complex_dd(h_dd, self.config, "h_dd")
        if h_dd is not None and h_dd.shape[0] != y_dd.shape[0]:
            raise ValueError("h_dd batch size must match y_dd.")
        if support_mask is not None:
            validate_support_mask(support_mask, self.config, "support_mask")
            if support_mask.shape[0] != y_dd.shape[0]:
                raise ValueError("support_mask batch size must match y_dd.")

        selected_mode = self.config.channel_estimator_mode if mode is None else mode
        if selected_mode not in {"oracle", "pilot", "fallback_topk"}:
            raise ValueError('mode must be one of "oracle", "pilot", or "fallback_topk".')

        if selected_mode == "oracle":
            estimate = self._estimate_oracle(h_dd, support_mask)
        elif selected_mode == "pilot":
            layout = pilot_config if pilot_config is not None else pilot_layout
            estimate = self._estimate_from_pilot(
                y_dd,
                layout,
                pilot_threshold=pilot_threshold,
                noise_var=noise_var,
            )
        else:
            estimate = self._estimate_fallback_topk(y_dd, h_dd, support_mask)

        if self.refinement_net is not None:
            estimate = self._apply_channel_refinement(estimate, snr_db)
        if self.offgrid_refinement_net is not None:
            estimate = self._apply_offgrid_refinement(estimate)
        if self.physics_refiner is not None and selected_mode != "oracle":
            layout = pilot_config if pilot_config is not None else pilot_layout
            ce_mask, mask_source, probe_pos, probe_val = self._resolve_ce_fitting_context(
                y_dd, ce_fit_mask, data_mask, layout, selected_mode
            )
            if ce_mask is not None:
                estimate = self._apply_physics_refinement(
                    estimate, y_dd, ce_mask,
                    probe_position=probe_pos,
                    probe_value=probe_val,
                )
                estimate = replace(estimate, ce_fit_mask_source=mask_source, physics_probe_position=probe_pos)
            else:
                estimate = replace(estimate, ce_fit_mask_source=mask_source, physics_probe_position=probe_pos)
        return estimate

    def _apply_channel_refinement(
        self,
        estimate: SparseChannelEstimate,
        snr_db: torch.Tensor | float | None,
    ) -> SparseChannelEstimate:
        """Apply channel refinement to estimate h_dd [B, M, N]."""

        refined_h_dd, support_logits = self.refinement_net(
            estimate.h_dd,
            estimate.support_mask,
            snr_db=snr_db,
        )
        path_gains = _gather_path_gains(refined_h_dd, estimate.path_indices)
        if estimate.confidence is not None:
            path_gains = torch.where(estimate.confidence > 0, path_gains, torch.zeros_like(path_gains))
        return replace(
            estimate,
            h_dd=refined_h_dd,
            path_gains=path_gains,
            support_logits=support_logits,
        )

    def _apply_offgrid_refinement(self, estimate: SparseChannelEstimate) -> SparseChannelEstimate:
        """Apply off-grid refinement to sparse paths [B, K, 2]."""

        refinement = self.offgrid_refinement_net(
            estimate.h_dd,
            estimate.path_indices,
            estimate.path_gains,
            confidence=estimate.confidence,
        )
        return apply_offgrid_refinement(estimate, refinement)

    def _apply_physics_refinement(
        self,
        estimate: SparseChannelEstimate,
        y_dd: torch.Tensor,
        fit_mask: torch.Tensor | None,
        probe_position: tuple[int, int] | None = None,
        probe_value: complex = 1.0 + 0.0j,
    ) -> SparseChannelEstimate:
        """Apply physics-guided gain refinement to sparse paths [B, K, 2]."""

        output = self.physics_refiner(
            y_dd,
            estimate.path_indices,
            estimate.path_gains,
            confidence=estimate.confidence,
            fractional_offsets=estimate.fractional_offsets,
            fit_mask=fit_mask,
            probe_position=probe_position,
            probe_value=probe_value,
        )
        h_dd = _scatter_path_gains_to_h_dd(
            output.path_gains,
            estimate.path_indices,
            estimate.support_mask,
            confidence=output.confidence,
        )
        path_confidence_map = _scatter_path_confidence_to_map(
            output.confidence,
            estimate.path_indices,
            estimate.support_mask,
        )
        return replace(
            estimate,
            h_dd=h_dd,
            path_gains=output.path_gains,
            confidence=output.confidence,
            path_confidence_map=path_confidence_map,
            physics_residual=output.residual,
            physics_residual_power=output.fit_residual_power,
            physics_fit_residual_power=output.fit_residual_power,
            physics_full_residual_power=output.full_residual_power,
            gain_covariance_diag=output.gain_covariance_diag,
            physics_normal_diag=output.normal_matrix_diag,
            physics_refined=True,
        )

    def _estimate_oracle(
        self,
        h_dd: torch.Tensor | None,
        support_mask: torch.Tensor | None,
    ) -> SparseChannelEstimate:
        """Estimate from oracle h_dd [B, M, N] and optional support [B, M, N]."""

        if h_dd is None:
            raise ValueError('h_dd is required when mode="oracle".')
        if support_mask is None:
            return replace(self.topk_estimator(h_dd), estimator_mode="oracle")
        support = mask_to_real(support_mask, h_dd.real.dtype, h_dd.device)
        return self._estimate_from_given_support(h_dd, support, estimator_mode="oracle")

    def _estimate_from_pilot(
        self,
        y_dd: torch.Tensor,
        pilot_config: EmbeddedPilotConfig | None,
        pilot_threshold: float | None,
        noise_var: float | None,
    ) -> SparseChannelEstimate:
        """Estimate from embedded pilot in y_dd [B, M, N]."""

        layout = pilot_config if pilot_config is not None else self._pilot_config_from_receiver_config()
        topk_paths = self.config.pilot_topk_paths if self.config.pilot_topk_paths is not None else self.config.topk_paths
        threshold = pilot_threshold if pilot_threshold is not None else self.config.pilot_threshold
        estimator = PilotSparseChannelEstimator(
            layout,
            topk_paths=topk_paths,
            threshold=threshold,
            noise_var=self.config.noise_var if noise_var is None else noise_var,
            confidence_temperature=self.config.confidence_temperature,
        )
        return estimator(y_dd)

    def _estimate_fallback_topk(
        self,
        y_dd: torch.Tensor,
        h_dd: torch.Tensor | None,
        support_mask: torch.Tensor | None,
    ) -> SparseChannelEstimate:
        """Legacy fallback_topk estimate for debug/smoke tests on [B, M, N]."""

        coarse_h_dd = h_dd if h_dd is not None else y_dd
        if support_mask is None:
            return replace(self.topk_estimator(coarse_h_dd), estimator_mode="fallback_topk")
        support = mask_to_real(support_mask, coarse_h_dd.real.dtype, coarse_h_dd.device)
        return self._estimate_from_given_support(coarse_h_dd, support, estimator_mode="fallback_topk")

    def _resolve_ce_fitting_context(
        self,
        y_dd: torch.Tensor,
        ce_fit_mask: torch.Tensor | None,
        data_mask: torch.Tensor | None,
        pilot_config: EmbeddedPilotConfig | None,
        estimator_mode: str,
    ) -> tuple[torch.Tensor | None, str, tuple[int, int] | None, complex]:
        """Resolve CE fitting mask, source label, probe position, and probe value.

        Priority:
        1. Explicit ``ce_fit_mask`` (or deprecated ``data_mask``).
        2. Pilot-mode: auto-constructed from pilot observation window.
        3. Fallback: None unless full-grid debug is explicitly enabled.

        Returns:
            (fit_mask, source_label, probe_position, probe_value).
            fit_mask is None when physics refinement must be skipped.
        """
        explicit_mask = ce_fit_mask if ce_fit_mask is not None else data_mask
        if explicit_mask is not None:
            probe_pos, probe_val = self._probe_from_layout(pilot_config)
            return (
                _resolve_spatial_mask_to_batch(explicit_mask, y_dd),
                "explicit_ce_data_mask",
                probe_pos,
                probe_val,
            )

        if estimator_mode == "pilot" and pilot_config is not None:
            probe_pos = (pilot_config.pilot_delay, pilot_config.pilot_doppler)
            probe_val = pilot_config.pilot_value
            obs_mask = _build_pilot_observation_mask(y_dd.shape, pilot_config, device=y_dd.device)
            return obs_mask, "pilot_observation", probe_pos, probe_val

        if self.config.physics_refinement_allow_full_grid_debug:
            probe_pos, probe_val = self._probe_from_layout(pilot_config)
            return (
                torch.ones_like(y_dd.real),
                "full_grid_debug",
                probe_pos,
                probe_val,
            )

        probe_pos, probe_val = self._probe_from_layout(pilot_config)
        return None, "none", probe_pos, probe_val

    def _probe_from_layout(
        self, pilot_config: EmbeddedPilotConfig | None
    ) -> tuple[tuple[int, int] | None, complex]:
        """Extract probe position and value from pilot config, with defaults."""
        if pilot_config is not None:
            return (
                (pilot_config.pilot_delay, pilot_config.pilot_doppler),
                pilot_config.pilot_value,
            )
        if self.config.pilot_delay is not None and self.config.pilot_doppler is not None:
            return (
                (self.config.pilot_delay, self.config.pilot_doppler),
                complex(self.config.pilot_value_real, self.config.pilot_value_imag),
            )
        return None, 1.0 + 0.0j

    def _pilot_config_from_receiver_config(self) -> EmbeddedPilotConfig:
        """Build EmbeddedPilotConfig for DD tensors [B, M, N] from ReceiverConfig."""

        if self.config.pilot_delay is None or self.config.pilot_doppler is None:
            raise ValueError(
                'pilot_delay and pilot_doppler must be set in ReceiverConfig or pilot_config when mode="pilot".'
            )
        return EmbeddedPilotConfig(
            pilot_delay=self.config.pilot_delay,
            pilot_doppler=self.config.pilot_doppler,
            guard_delay=self.config.pilot_guard_delay,
            guard_doppler=self.config.pilot_guard_doppler,
            obs_delay_radius=self.config.pilot_obs_delay_radius,
            obs_doppler_radius=self.config.pilot_obs_doppler_radius,
            pilot_value=complex(self.config.pilot_value_real, self.config.pilot_value_imag),
            wrap_around=False,
            require_obs_within_guard=self.config.require_obs_within_guard,
            cfar_scale=self.config.pilot_cfar_scale,
            min_confidence=self.config.pilot_min_confidence,
        )

    def _estimate_from_given_support(
        self,
        coarse_h_dd: torch.Tensor,
        support_mask: torch.Tensor,
        estimator_mode: str = "fallback_topk",
    ) -> SparseChannelEstimate:
        """Build estimate from coarse_h_dd [B, M, N] and support_mask [B, M, N]."""

        batch_size = coarse_h_dd.shape[0]
        requested_k = self.config.topk_paths
        actual_k = min(requested_k, self.config.M * self.config.N)
        flat_abs = (coarse_h_dd.abs() * support_mask).reshape(batch_size, -1)
        flat_support = support_mask.reshape(batch_size, -1) > 0
        candidate_abs = torch.where(flat_support, flat_abs, torch.full_like(flat_abs, -1.0))
        values, flat_indices = torch.topk(candidate_abs, k=actual_k, dim=-1)
        valid = values >= 0.0
        if actual_k < requested_k:
            pad_count = requested_k - actual_k
            values = F.pad(values, (0, pad_count), value=-1.0)
            flat_indices = F.pad(flat_indices, (0, pad_count))
            valid = F.pad(valid, (0, pad_count), value=False)
        confidence = torch.where(valid, values, torch.zeros_like(values))
        flat_indices = torch.where(valid, flat_indices, torch.zeros_like(flat_indices))
        h_sparse = coarse_h_dd * support_mask.to(dtype=coarse_h_dd.real.dtype)
        path_indices = _flat_to_delay_doppler(flat_indices, self.config.N)
        path_gains = _gather_path_gains(h_sparse, path_indices)
        path_gains = torch.where(confidence > 0, path_gains, torch.zeros_like(path_gains))
        return SparseChannelEstimate(
            h_dd=h_sparse,
            support_mask=support_mask,
            path_indices=path_indices,
            path_gains=path_gains,
            confidence=confidence,
            estimator_mode=estimator_mode,
        )


def _resolve_spatial_mask_to_batch(mask: torch.Tensor, y_dd: torch.Tensor) -> torch.Tensor:
    """Resolve a [B, M, N] or [1, M, N] mask to [B, M, N] for the given y_dd."""
    if not torch.is_tensor(mask) or mask.ndim != 3 or torch.is_complex(mask):
        raise ValueError("mask must have real shape [B, M, N] or [1, M, N].")
    if mask.shape[-2:] != y_dd.shape[-2:]:
        raise ValueError("mask spatial dims must match y_dd.")
    if mask.shape[0] not in {1, y_dd.shape[0]}:
        raise ValueError("mask batch dim must be 1 or match y_dd.")
    resolved = mask.to(device=y_dd.device, dtype=y_dd.real.dtype).clamp(0.0, 1.0)
    if resolved.shape[0] == 1 and y_dd.shape[0] != 1:
        resolved = resolved.expand(y_dd.shape[0], -1, -1)
    return resolved


def _build_pilot_observation_mask(
    y_shape: tuple[int, int, int],
    pilot_config: "EmbeddedPilotConfig",
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build CE fitting mask [1, M, N] covering the pilot observation window."""
    batch_size, M, N = y_shape
    mask = torch.zeros(1, M, N, device=device)
    p_delay = pilot_config.pilot_delay
    p_doppler = pilot_config.pilot_doppler
    r_delay = pilot_config.obs_delay_radius
    r_doppler = pilot_config.obs_doppler_radius

    for d in range(p_delay - r_delay, p_delay + r_delay + 1):
        if 0 <= d < M:
            for dopp in range(p_doppler - r_doppler, p_doppler + r_doppler + 1):
                if 0 <= dopp < N:
                    mask[0, d, dopp] = 1.0
    return mask


def channel_nmse(h_hat: torch.Tensor, h_true: torch.Tensor) -> torch.Tensor:
    """Return NMSE for h_hat [B, M, N] against h_true [B, M, N]."""

    if h_hat.shape != h_true.shape:
        raise ValueError("h_hat and h_true must have the same shape [B, M, N].")
    error = (h_hat - h_true).abs().pow(2).sum()
    denom = h_true.abs().pow(2).sum().clamp_min(1e-12)
    return error / denom


def support_bce_loss(support_logits: torch.Tensor, support_true: torch.Tensor) -> torch.Tensor:
    """Return BCE loss for support_logits [B, M, N] and support_true [B, M, N]."""

    if support_logits.shape != support_true.shape:
        raise ValueError("support_logits and support_true must share shape [B, M, N].")
    return F.binary_cross_entropy_with_logits(
        support_logits,
        support_true.to(device=support_logits.device, dtype=support_logits.dtype),
    )


def sparsity_l1_loss(h_dd: torch.Tensor) -> torch.Tensor:
    """Return L1 sparsity loss for complex h_dd [B, M, N]."""

    if not torch.is_tensor(h_dd) or h_dd.ndim != 3:
        raise ValueError("h_dd must have shape [B, M, N].")
    return h_dd.abs().mean()
