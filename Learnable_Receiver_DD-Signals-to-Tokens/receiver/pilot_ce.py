"""Embedded-pilot on-grid sparse channel estimation for DD tensors.

This module implements pilot/guard layout utilities and a model-driven
on-grid sparse CE stage. It estimates sparse DD taps only from the embedded
pilot observation window; it does not use the whole y_dd grid as a channel
proxy and does not perform off-grid fractional refinement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from .sparse_channel import SparseChannelEstimate


@dataclass(frozen=True)
class EmbeddedPilotConfig:
    """Embedded pilot layout for DD grids with shape [B, M, N].

    Attributes:
        pilot_delay: Delay-bin index of the pilot in a DD grid [M, N].
        pilot_doppler: Doppler-bin index of the pilot in a DD grid [M, N].
        guard_delay: Non-negative rectangular guard radius along delay.
        guard_doppler: Non-negative rectangular guard radius along Doppler.
        obs_delay_radius: Non-negative observation radius along delay.
        obs_doppler_radius: Non-negative observation radius along Doppler.
        pilot_value: Complex scalar written at the pilot bin.
        wrap_around: If true, pilot/guard/observation windows wrap on [M, N].
            The default false clips windows at DD-grid boundaries.
        require_obs_within_guard: If true, observation radii must fit inside
            the guard radii so pilot CE uses only pilot/guard bins.
        cfar_scale: Positive scale for sqrt(noise_var) CFAR-style thresholding.
        min_confidence: Minimum normalized path confidence in [0, 1].
    """

    pilot_delay: int
    pilot_doppler: int
    guard_delay: int
    guard_doppler: int
    obs_delay_radius: int
    obs_doppler_radius: int
    pilot_value: complex = 1.0 + 0.0j
    wrap_around: bool = False
    require_obs_within_guard: bool = True
    cfar_scale: float = 3.0
    min_confidence: float = 0.0


@dataclass(frozen=True)
class PilotMasks:
    """Embedded pilot masks for one DD grid.

    Attributes:
        pilot_mask: Boolean mask with shape [M, N] and exactly one true entry.
        guard_mask: Boolean guard mask with shape [M, N], excluding the pilot.
        data_mask: Boolean data mask with shape [M, N], excluding pilot/guard.
    """

    pilot_mask: torch.Tensor
    guard_mask: torch.Tensor
    data_mask: torch.Tensor


@dataclass(frozen=True)
class PilotObservation:
    """Pilot observation window extracted from a received DD grid.

    Attributes:
        observation: Complex pilot observation tensor with shape [B, P, Q].
        delay_indices: Long global delay indices with shape [P].
        doppler_indices: Long global Doppler indices with shape [Q].
        center_delay: Integer pilot delay index in the source DD grid.
        center_doppler: Integer pilot Doppler index in the source DD grid.
    """

    observation: torch.Tensor
    delay_indices: torch.Tensor
    doppler_indices: torch.Tensor
    center_delay: int
    center_doppler: int


def build_embedded_pilot_masks(M: int, N: int, config: EmbeddedPilotConfig) -> PilotMasks:
    """Build pilot, guard, and data masks for one DD grid.

    Args:
        M: Number of delay bins in DD tensors with shape [B, M, N].
        N: Number of Doppler bins in DD tensors with shape [B, M, N].
        config: EmbeddedPilotConfig describing pilot and guard layout.

    Returns:
        PilotMasks containing boolean tensors pilot_mask [M, N], guard_mask
        [M, N], and data_mask [M, N]. The masks are mutually disjoint.
    """

    validate_pilot_ce_layout(M, N, config)
    pilot_mask = torch.zeros(M, N, dtype=torch.bool)
    guard_mask = torch.zeros(M, N, dtype=torch.bool)

    pilot_mask[config.pilot_delay, config.pilot_doppler] = True
    delay_indices = _window_indices(config.pilot_delay, config.guard_delay, M, config.wrap_around)
    doppler_indices = _window_indices(config.pilot_doppler, config.guard_doppler, N, config.wrap_around)
    for delay in delay_indices.tolist():
        for doppler in doppler_indices.tolist():
            guard_mask[int(delay), int(doppler)] = True
    guard_mask = guard_mask & ~pilot_mask
    data_mask = ~(pilot_mask | guard_mask)
    return PilotMasks(pilot_mask=pilot_mask, guard_mask=guard_mask, data_mask=data_mask)


def insert_embedded_pilot(
    x_dd: torch.Tensor,
    config: EmbeddedPilotConfig,
    masks: PilotMasks | None = None,
) -> torch.Tensor:
    """Insert an embedded pilot into a DD data grid.

    Args:
        x_dd: Complex DD data tensor with shape [B, M, N].
        config: EmbeddedPilotConfig containing the pilot scalar and layout.
        masks: Optional PilotMasks with pilot_mask [M, N], guard_mask [M, N],
            and data_mask [M, N]. If omitted, masks are built from x_dd shape.

    Returns:
        Complex DD tensor x_pilot with shape [B, M, N]. The pilot bin is set
        to config.pilot_value, guard bins are zero, and data bins preserve x_dd.
    """

    _validate_complex_dd_unconfigured(x_dd, "x_dd")
    M, N = x_dd.shape[-2:]
    validate_pilot_ce_layout(M, N, config)
    masks = build_embedded_pilot_masks(M, N, config) if masks is None else masks
    _validate_masks(masks, M, N)

    pilot_mask = masks.pilot_mask.to(device=x_dd.device)
    guard_mask = masks.guard_mask.to(device=x_dd.device)
    x_pilot = x_dd.clone()
    pilot_value = torch.as_tensor(config.pilot_value, device=x_dd.device, dtype=x_dd.dtype)
    x_pilot[:, pilot_mask] = pilot_value
    x_pilot[:, guard_mask] = torch.zeros((), device=x_dd.device, dtype=x_dd.dtype)
    return x_pilot


def extract_pilot_observation(y_dd: torch.Tensor, config: EmbeddedPilotConfig) -> PilotObservation:
    """Extract the embedded-pilot observation window from y_dd.

    Args:
        y_dd: Complex received DD tensor with shape [B, M, N].
        config: EmbeddedPilotConfig defining the pilot center and observation
            radii.

    Returns:
        PilotObservation with observation [B, P, Q], delay_indices [P], and
        doppler_indices [Q]. With wrap_around=False, P and Q are clipped at
        the DD-grid boundaries.
    """

    _validate_complex_dd_unconfigured(y_dd, "y_dd")
    M, N = y_dd.shape[-2:]
    validate_pilot_ce_layout(M, N, config)
    delay_indices = _window_indices(
        config.pilot_delay,
        config.obs_delay_radius,
        M,
        config.wrap_around,
        device=y_dd.device,
    )
    doppler_indices = _window_indices(
        config.pilot_doppler,
        config.obs_doppler_radius,
        N,
        config.wrap_around,
        device=y_dd.device,
    )
    observation = y_dd.index_select(-2, delay_indices).index_select(-1, doppler_indices)
    return PilotObservation(
        observation=observation,
        delay_indices=delay_indices,
        doppler_indices=doppler_indices,
        center_delay=config.pilot_delay,
        center_doppler=config.pilot_doppler,
    )


class PilotSparseChannelEstimator(nn.Module):
    """Embedded-pilot on-grid sparse CE for received DD tensors [B, M, N].

    The estimator extracts a local pilot observation [B, P, Q], divides by the
    known pilot scalar, detects sparse on-grid taps, and returns a
    SparseChannelEstimate with h_dd [B, M, N], support_mask [B, M, N],
    path_indices [B, K, 2], path_gains [B, K], confidence [B, K], and
    path_confidence_map [B, M, N]. If threshold is supplied, it is used
    directly. Else if noise_var is supplied, the effective threshold is
    cfar_scale * sqrt(noise_var). Else non-CFAR top-K is used, which is only
    suitable for controlled/noiseless settings. It does not perform off-grid
    refinement.
    """

    def __init__(
        self,
        config: EmbeddedPilotConfig,
        topk_paths: int,
        threshold: float | None = None,
        noise_var: float | None = None,
        confidence_temperature: float = 1.0,
    ):
        super().__init__()
        if not isinstance(topk_paths, int) or topk_paths <= 0:
            raise ValueError("topk_paths must be a positive integer.")
        if threshold is not None and (not isinstance(threshold, (int, float)) or threshold < 0):
            raise ValueError("threshold must be None or a non-negative scalar.")
        if noise_var is not None and (not isinstance(noise_var, (int, float)) or noise_var < 0):
            raise ValueError("noise_var must be None or a non-negative scalar.")
        if not isinstance(confidence_temperature, (int, float)) or confidence_temperature <= 0:
            raise ValueError("confidence_temperature must be positive.")
        if abs(config.pilot_value) <= 0:
            raise ValueError("config.pilot_value must be nonzero.")
        self.config = config
        self.topk_paths = topk_paths
        self.threshold = None if threshold is None else float(threshold)
        self.noise_var = None if noise_var is None else float(noise_var)
        self.confidence_temperature = float(confidence_temperature)
        self.cfar_scale = float(config.cfar_scale)
        self.min_confidence = float(config.min_confidence)

    def forward(self, y_dd: torch.Tensor) -> "SparseChannelEstimate":
        """Estimate sparse on-grid channel taps from y_dd [B, M, N].

        Args:
            y_dd: Complex received DD tensor with shape [B, M, N].

        Returns:
            SparseChannelEstimate with h_dd [B, M, N], support_mask [B, M, N],
            path_indices [B, K, 2], path_gains [B, K], confidence [B, K],
            path_confidence_map [B, M, N], support_logits [B, M, N],
            pilot_residual [B, P, Q], and pilot_residual_power [B].
        """

        from .sparse_channel import SparseChannelEstimate

        _validate_complex_dd_unconfigured(y_dd, "y_dd")
        batch_size, M, N = y_dd.shape
        validate_pilot_ce_layout(M, N, self.config)
        obs = extract_pilot_observation(y_dd, self.config)
        pilot_value = torch.as_tensor(self.config.pilot_value, device=y_dd.device, dtype=y_dd.dtype)
        h_obs = obs.observation / pilot_value
        flat_h = h_obs.reshape(batch_size, -1)
        flat_abs = flat_h.abs()

        values, flat_indices, candidate_valid = self._select_support(flat_abs)
        confidence = self._confidence(values, candidate_valid)
        valid = candidate_valid & (confidence >= self.min_confidence)
        confidence = torch.where(valid, confidence, torch.zeros_like(confidence))
        path_indices = self._flat_obs_to_path_indices(flat_indices, valid, obs, M, N)
        path_gains = _gather_flat_values(flat_h, flat_indices, valid)

        h_dd = torch.zeros(batch_size, M, N, device=y_dd.device, dtype=y_dd.dtype)
        support_mask = torch.zeros(batch_size, M, N, device=y_dd.device, dtype=y_dd.real.dtype)
        confidence_map = torch.zeros_like(support_mask)
        self._scatter_paths(h_dd, support_mask, confidence_map, path_indices, path_gains, confidence, valid)

        pilot_residual = self._pilot_residual(h_obs, flat_indices, path_gains, valid)
        pilot_residual_power = pilot_residual.abs().pow(2).mean(dim=(-2, -1))
        support_logits = _confidence_to_logits(confidence_map)
        return SparseChannelEstimate(
            h_dd=h_dd,
            support_mask=support_mask,
            path_indices=path_indices,
            path_gains=path_gains,
            confidence=confidence,
            support_logits=support_logits,
            path_confidence_map=confidence_map,
            pilot_residual=pilot_residual,
            pilot_residual_power=pilot_residual_power,
            noise_var=self.noise_var,
            estimator_mode="pilot",
        )

    def _select_support(self, flat_abs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select support from pilot magnitudes flat_abs [B, P*Q]."""

        actual_k = min(self.topk_paths, flat_abs.shape[-1])
        effective_threshold = self._effective_threshold()
        if effective_threshold is None:
            values, flat_indices = torch.topk(flat_abs, k=actual_k, dim=-1)
            valid = values > 0
        else:
            rejected = torch.full_like(flat_abs, -1.0)
            candidates = torch.where(flat_abs >= effective_threshold, flat_abs, rejected)
            values, flat_indices = torch.topk(candidates, k=actual_k, dim=-1)
            valid = (values >= effective_threshold) & (values > 0)

        if actual_k < self.topk_paths:
            pad_count = self.topk_paths - actual_k
            values = torch.nn.functional.pad(values, (0, pad_count))
            flat_indices = torch.nn.functional.pad(flat_indices, (0, pad_count))
            valid = torch.nn.functional.pad(valid, (0, pad_count), value=False)
        return values, flat_indices, valid

    def _effective_threshold(self) -> float | None:
        """Return scalar threshold for pilot magnitudes flat_abs [B, P*Q]."""

        if self.threshold is not None:
            return self.threshold
        if self.noise_var is not None:
            return self.cfar_scale * (self.noise_var ** 0.5)
        return None

    def _flat_obs_to_path_indices(
        self,
        flat_indices: torch.Tensor,
        valid: torch.Tensor,
        obs: PilotObservation,
        M: int,
        N: int,
    ) -> torch.Tensor:
        """Map flat observation indices [B, K] to DD path indices [B, K, 2]."""

        q_size = obs.doppler_indices.numel()
        obs_delay_pos = torch.div(flat_indices, q_size, rounding_mode="floor")
        obs_doppler_pos = flat_indices.remainder(q_size)
        delay_global = obs.delay_indices[obs_delay_pos]
        doppler_global = obs.doppler_indices[obs_doppler_pos]
        delay_tap = (delay_global - int(obs.center_delay)).remainder(M)
        doppler_tap = (doppler_global - int(obs.center_doppler)).remainder(N)
        path_indices = torch.stack((delay_tap, doppler_tap), dim=-1).long()
        return torch.where(valid.unsqueeze(-1), path_indices, torch.zeros_like(path_indices))

    def _confidence(self, values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Normalize selected magnitudes values [B, K] to confidence [B, K]."""

        magnitude = values.clamp_min(0.0)
        denom = magnitude.max(dim=-1, keepdim=True).values.clamp_min(1e-12)
        confidence = (magnitude / denom).clamp(0.0, 1.0)
        confidence = confidence.pow(1.0 / self.confidence_temperature)
        return torch.where(valid, confidence, torch.zeros_like(confidence))

    def _scatter_paths(
        self,
        h_dd: torch.Tensor,
        support_mask: torch.Tensor,
        confidence_map: torch.Tensor,
        path_indices: torch.Tensor,
        path_gains: torch.Tensor,
        confidence: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        """Scatter selected paths [B, K] into DD tensors [B, M, N]."""

        batch_indices = torch.arange(h_dd.shape[0], device=h_dd.device)
        for path_pos in range(path_indices.shape[1]):
            active = valid[:, path_pos]
            if not bool(active.any()):
                continue
            batch = batch_indices[active]
            delay = path_indices[active, path_pos, 0]
            doppler = path_indices[active, path_pos, 1]
            h_dd[batch, delay, doppler] = path_gains[active, path_pos]
            support_mask[batch, delay, doppler] = 1.0
            confidence_map[batch, delay, doppler] = confidence[active, path_pos]

    def _pilot_residual(
        self,
        h_obs: torch.Tensor,
        flat_indices: torch.Tensor,
        path_gains: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Return pilot residual h_obs - sparse(h_obs) with shape [B, P, Q]."""

        h_obs_hat_flat = torch.zeros_like(h_obs.reshape(h_obs.shape[0], -1))
        for path_pos in range(flat_indices.shape[1]):
            active = valid[:, path_pos]
            if not bool(active.any()):
                continue
            h_obs_hat_flat[active, flat_indices[active, path_pos]] = path_gains[active, path_pos]
        return h_obs - h_obs_hat_flat.reshape_as(h_obs)


def validate_pilot_ce_layout(M: int, N: int, config: EmbeddedPilotConfig) -> None:
    """Validate embedded-pilot CE layout for DD tensors with shape [B, M, N].

    Args:
        M: Number of delay bins in DD tensors with shape [B, M, N].
        N: Number of Doppler bins in DD tensors with shape [B, M, N].
        config: EmbeddedPilotConfig carrying pilot, guard, and observation
            radii. If config.require_obs_within_guard is true, the observation
            window [P, Q] must lie inside the pilot+guard rectangle.

    Raises:
        ValueError: If pilot indices, guard/radius values, CFAR parameters, or
            observation-within-guard constraints are invalid.
    """

    if not isinstance(M, int) or M <= 0:
        raise ValueError("M must be a positive integer.")
    if not isinstance(N, int) or N <= 0:
        raise ValueError("N must be a positive integer.")
    if not isinstance(config.pilot_delay, int) or config.pilot_delay < 0 or config.pilot_delay >= M:
        raise ValueError("pilot_delay must be an integer in [0, M).")
    if not isinstance(config.pilot_doppler, int) or config.pilot_doppler < 0 or config.pilot_doppler >= N:
        raise ValueError("pilot_doppler must be an integer in [0, N).")
    non_negative_fields = {
        "guard_delay": config.guard_delay,
        "guard_doppler": config.guard_doppler,
        "obs_delay_radius": config.obs_delay_radius,
        "obs_doppler_radius": config.obs_doppler_radius,
    }
    for name, value in non_negative_fields.items():
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer.")
    if not isinstance(config.wrap_around, bool):
        raise ValueError("wrap_around must be a bool.")
    if not isinstance(config.require_obs_within_guard, bool):
        raise ValueError("require_obs_within_guard must be a bool.")
    if not isinstance(config.cfar_scale, (int, float)) or config.cfar_scale <= 0:
        raise ValueError("cfar_scale must be positive.")
    if not isinstance(config.min_confidence, (int, float)) or not 0.0 <= config.min_confidence <= 1.0:
        raise ValueError("min_confidence must be in [0, 1].")
    if config.require_obs_within_guard:
        if config.obs_delay_radius > config.guard_delay:
            raise ValueError("obs_delay_radius must be <= guard_delay when require_obs_within_guard=True.")
        if config.obs_doppler_radius > config.guard_doppler:
            raise ValueError("obs_doppler_radius must be <= guard_doppler when require_obs_within_guard=True.")


def _validate_complex_dd_unconfigured(value: torch.Tensor, name: str) -> None:
    if not torch.is_tensor(value) or value.ndim != 3 or not torch.is_complex(value):
        raise TypeError(f"{name} must be a complex tensor with shape [B, M, N].")


def _validate_masks(masks: PilotMasks, M: int, N: int) -> None:
    for name, mask in (
        ("pilot_mask", masks.pilot_mask),
        ("guard_mask", masks.guard_mask),
        ("data_mask", masks.data_mask),
    ):
        if not torch.is_tensor(mask) or mask.shape != (M, N) or mask.dtype != torch.bool:
            raise ValueError(f"{name} must be a bool tensor with shape [M, N].")
    if bool((masks.pilot_mask & masks.guard_mask).any()):
        raise ValueError("pilot_mask and guard_mask must be disjoint.")
    if bool((masks.pilot_mask & masks.data_mask).any()):
        raise ValueError("pilot_mask and data_mask must be disjoint.")
    if bool((masks.guard_mask & masks.data_mask).any()):
        raise ValueError("guard_mask and data_mask must be disjoint.")


def _window_indices(
    center: int,
    radius: int,
    limit: int,
    wrap_around: bool,
    device: torch.device | None = None,
) -> torch.Tensor:
    raw = torch.arange(center - radius, center + radius + 1, dtype=torch.long, device=device)
    if wrap_around:
        return raw.remainder(limit)
    return raw[(raw >= 0) & (raw < limit)]


def _gather_flat_values(flat_h: torch.Tensor, flat_indices: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    gathered = torch.gather(flat_h, dim=-1, index=flat_indices)
    return torch.where(valid, gathered, torch.zeros_like(gathered))


def _confidence_to_logits(confidence_map: torch.Tensor) -> torch.Tensor:
    eps = 1e-6
    return torch.logit(confidence_map.clamp(eps, 1.0 - eps))
