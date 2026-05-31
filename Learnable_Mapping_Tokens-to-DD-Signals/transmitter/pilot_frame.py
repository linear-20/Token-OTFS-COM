"""Pilot, guard, and data mask construction for DD grids.

All masks use shape [1, M, N] with boolean dtype. The guard region is
rectangular with boundary clipping (no wrap-around).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import TransmitterConfig


@dataclass(frozen=True)
class TransmitterPilotMasks:
    """Pilot, guard, and data masks for one DD grid.

    Attributes:
        pilot_mask: Boolean tensor [1, M, N] with exactly one True at the pilot.
        guard_mask: Boolean tensor [1, M, N] for the rectangular guard region,
            excluding the pilot bin.
        data_mask: Boolean tensor [1, M, N] for data-carrying bins.
    """

    pilot_mask: torch.Tensor
    guard_mask: torch.Tensor
    data_mask: torch.Tensor


def build_transmitter_pilot_masks(config: TransmitterConfig) -> TransmitterPilotMasks:
    """Build pilot, guard, and data masks for a DD grid.

    The pilot mask has a single True at (pilot_delay, pilot_doppler). The guard
    mask covers a rectangular region around the pilot with the configured guard
    radii, clipped at DD-grid boundaries (no wrap-around), with the pilot bin
    excluded. The data mask is the complement of pilot + guard.

    Args:
        config: TransmitterConfig with validated pilot/guard layout.

    Returns:
        TransmitterPilotMasks with pilot_mask [1, M, N], guard_mask [1, M, N],
        and data_mask [1, M, N]. All masks are boolean and mutually exclusive.

    Raises:
        ValueError: If data_mask contains no True entries.
    """
    M, N = config.M, config.N

    pilot_mask = torch.zeros(1, M, N, dtype=torch.bool)
    pilot_mask[0, config.pilot_delay, config.pilot_doppler] = True

    guard_mask = torch.zeros(1, M, N, dtype=torch.bool)
    delay_start = max(0, config.pilot_delay - config.pilot_guard_delay)
    delay_end = min(M, config.pilot_delay + config.pilot_guard_delay + 1)
    doppler_start = max(0, config.pilot_doppler - config.pilot_guard_doppler)
    doppler_end = min(N, config.pilot_doppler + config.pilot_guard_doppler + 1)
    guard_mask[0, delay_start:delay_end, doppler_start:doppler_end] = True
    guard_mask = guard_mask & ~pilot_mask

    data_mask = ~(pilot_mask | guard_mask)

    if not data_mask.any():
        raise ValueError(
            "data_mask must contain at least one True entry. "
            f"The pilot+guard layout covers the entire DD grid [M={M}, N={N}]."
        )

    return TransmitterPilotMasks(
        pilot_mask=pilot_mask,
        guard_mask=guard_mask,
        data_mask=data_mask,
    )


def insert_transmitter_pilot(
    codewords: torch.Tensor,
    config: TransmitterConfig,
    masks: TransmitterPilotMasks | None = None,
) -> torch.Tensor:
    """Apply hard pilot/guard insertion to selected DD codewords.

    Converts batch selected DD codewords [B, M, N] into a pilot-safe DD frame
    x_dd [B, M, N] by: (1) setting the pilot bin to config.pilot_value,
    (2) zeroing the guard region, and (3) preserving codeword values in the
    data region.

    This function performs hard pilot/guard insertion only. It does NOT perform
    codebook generation, OTFS modulation, or bit/QAM mapping.

    Args:
        codewords: Complex DD codewords with shape [B, M, N].
        config: TransmitterConfig with validated pilot/guard layout.
        masks: Optional TransmitterPilotMasks with pilot_mask [1, M, N],
            guard_mask [1, M, N], and data_mask [1, M, N]. If None, masks are
            built from config via build_transmitter_pilot_masks.

    Returns:
        Complex DD frame x_dd with shape [B, M, N], same device and dtype as
        codewords. The input codewords tensor is not modified in-place.

    Raises:
        TypeError: If codewords is not a torch.Tensor or is not complex.
        ValueError: If codewords shape, config, or masks are invalid.
    """
    _validate_codewords(codewords, config)

    if masks is None:
        masks = build_transmitter_pilot_masks(config)
    else:
        _validate_insertion_masks(masks, config)

    M, N = config.M, config.N
    pilot_mask_2d = masks.pilot_mask[0].to(device=codewords.device)
    guard_mask_2d = masks.guard_mask[0].to(device=codewords.device)

    x_dd = codewords.clone()
    x_dd[:, guard_mask_2d] = 0.0
    pilot_value = torch.as_tensor(config.pilot_value, device=codewords.device, dtype=codewords.dtype)
    x_dd[:, pilot_mask_2d] = pilot_value

    return x_dd


def _validate_codewords(codewords: torch.Tensor, config: TransmitterConfig) -> None:
    """Validate codewords tensor for insert_transmitter_pilot.

    Args:
        codewords: Complex DD codewords with shape [B, M, N].
        config: TransmitterConfig for spatial shape and dtype reference.

    Raises:
        TypeError: If codewords is not a torch.Tensor or is not complex.
        ValueError: If shape or spatial dimensions are invalid.
    """
    if not torch.is_tensor(codewords):
        raise TypeError(f"codewords must be a torch.Tensor, got {type(codewords).__name__}.")
    if not torch.is_complex(codewords):
        raise TypeError(f"codewords must be a complex tensor, got dtype {codewords.dtype}.")
    if codewords.ndim != 3:
        raise ValueError(
            f"codewords must have 3 dimensions [B, M, N], got ndim={codewords.ndim}."
        )
    B, M_cw, N_cw = codewords.shape
    if B <= 0:
        raise ValueError(f"codewords batch size B must be positive, got B={B}.")
    if M_cw != config.M or N_cw != config.N:
        raise ValueError(
            f"codewords spatial shape [{M_cw}, {N_cw}] does not match "
            f"config.M={config.M}, config.N={config.N}."
        )


def _validate_insertion_masks(masks: TransmitterPilotMasks, config: TransmitterConfig) -> None:
    """Validate externally supplied masks for insert_transmitter_pilot.

    Validates that masks are a TransmitterPilotMasks instance, have correct
    shape/dtype, are mutually exclusive, and strictly equal the no-wrap
    rectangular layout defined by config via build_transmitter_pilot_masks.

    Args:
        masks: TransmitterPilotMasks to validate.
        config: TransmitterConfig for expected spatial shape and layout.

    Raises:
        TypeError: If masks is not a TransmitterPilotMasks, or if any mask
            tensor is not bool dtype.
        ValueError: If shapes, mutual exclusivity, data-mask identity,
            data-mask emptiness, pilot position, or layout equality against
            build_transmitter_pilot_masks(config) are violated.
    """
    if not isinstance(masks, TransmitterPilotMasks):
        raise TypeError(
            f"masks must be a TransmitterPilotMasks instance, "
            f"got {type(masks).__name__}."
        )

    M, N = config.M, config.N

    for name in ("pilot_mask", "guard_mask", "data_mask"):
        mask = getattr(masks, name)
        if not torch.is_tensor(mask):
            raise TypeError(f"masks.{name} must be a torch.Tensor, got {type(mask).__name__}.")
        if mask.dtype != torch.bool:
            raise TypeError(
                f"masks.{name} must be a bool tensor, got dtype {mask.dtype}."
            )
        if mask.shape != (1, M, N):
            raise ValueError(
                f"masks.{name} must have shape [1, {M}, {N}], got {list(mask.shape)}."
            )

    pilot = masks.pilot_mask
    guard = masks.guard_mask
    data = masks.data_mask

    if bool((pilot & guard).any()):
        raise ValueError("masks.pilot_mask and masks.guard_mask must be disjoint.")
    if bool((pilot & data).any()):
        raise ValueError("masks.pilot_mask and masks.data_mask must be disjoint.")
    if bool((guard & data).any()):
        raise ValueError("masks.guard_mask and masks.data_mask must be disjoint.")

    expected_data = ~(pilot | guard)
    if not torch.equal(data, expected_data):
        raise ValueError("masks.data_mask must equal ~(pilot_mask | guard_mask).")

    if not data.any():
        raise ValueError("masks.data_mask must contain at least one True entry.")

    if int(pilot.sum().item()) != 1:
        raise ValueError(
            f"masks.pilot_mask must have exactly one True entry, "
            f"got {int(pilot.sum().item())}."
        )
    if not pilot[0, config.pilot_delay, config.pilot_doppler].item():
        raise ValueError(
            f"masks.pilot_mask True must be at "
            f"[0, pilot_delay={config.pilot_delay}, pilot_doppler={config.pilot_doppler}], "
            f"but that position is False."
        )

    expected = build_transmitter_pilot_masks(config)
    exp_pilot = expected.pilot_mask.to(device=pilot.device)
    exp_guard = expected.guard_mask.to(device=guard.device)
    exp_data = expected.data_mask.to(device=data.device)

    if not torch.equal(pilot, exp_pilot):
        raise ValueError(
            "masks.pilot_mask does not match the config-defined "
            "no-wrap rectangular pilot layout."
        )
    if not torch.equal(guard, exp_guard):
        raise ValueError(
            "masks.guard_mask does not match the config-defined "
            "no-wrap rectangular guard layout."
        )
    if not torch.equal(data, exp_data):
        raise ValueError(
            "masks.data_mask does not match the config-defined "
            "no-wrap rectangular data layout."
        )
