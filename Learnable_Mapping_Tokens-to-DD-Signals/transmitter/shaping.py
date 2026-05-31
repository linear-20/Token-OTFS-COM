"""Sparse integer DD shift metadata for scenario construction."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SparseShiftSet:
    """Set of sparse integer DD shifts with optional weights.

    Each row in shifts [S, 2] is a (delay_shift, doppler_shift) pair
    representing a sparse DD displacement.

    Attributes:
        shifts: Long tensor [S, 2] of integer delay/Doppler shift pairs.
        weights: Optional real floating tensor [S] of non-negative weights.
            If None, uniform weights are used in normalized_weights().
        name: Human-readable identifier for this shift set.
    """

    shifts: torch.Tensor
    weights: torch.Tensor | None = None
    name: str = "integer_shift_set"

    def __post_init__(self) -> None:
        _validate_sparse_shift_set(self)

    def normalized_weights(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Return normalized shift weights [S] summing to 1.

        If self.weights is None, uniform weights are returned. Otherwise
        self.weights is normalized to sum to 1. The input weights tensor
        is never modified in-place.

        Args:
            device: Optional target device. Defaults to the source tensor's
                device.
            dtype: Optional target real dtype. Defaults to the source tensor's
                dtype (or torch.float32 for uniform weights).

        Returns:
            Real tensor [S] of normalized weights summing to 1.
        """
        if self.weights is None:
            resolved_device = device if device is not None else self.shifts.device
            resolved_dtype = dtype if dtype is not None else torch.float32
            S = self.shifts.shape[0]
            return torch.full(
                (S,), 1.0 / S, device=resolved_device, dtype=resolved_dtype
            )

        resolved_device = device if device is not None else self.weights.device
        resolved_dtype = dtype if dtype is not None else self.weights.dtype
        w = self.weights.to(device=resolved_device, dtype=resolved_dtype)
        return w / w.sum()


def _validate_sparse_shift_set(ss: SparseShiftSet) -> None:
    """Validate SparseShiftSet fields.

    Raises:
        TypeError: If shifts is not a tensor, weights is not a tensor, or
            dtypes are invalid.
        ValueError: If shapes, finiteness, non-negativity, or name constraints
            are violated.
    """
    if not torch.is_tensor(ss.shifts):
        raise TypeError(f"shifts must be a torch.Tensor, got {type(ss.shifts).__name__}.")
    if ss.shifts.ndim != 2:
        raise ValueError(f"shifts must have ndim==2, got ndim={ss.shifts.ndim}.")
    if ss.shifts.shape[-1] != 2:
        raise ValueError(f"shifts last dim must be 2, got {ss.shifts.shape[-1]}.")
    if ss.shifts.dtype not in (torch.long, torch.int64):
        raise TypeError(f"shifts dtype must be torch.long or torch.int64, got {ss.shifts.dtype}.")
    if ss.shifts.shape[0] <= 0:
        raise ValueError(f"shifts must have at least one row, got shape {list(ss.shifts.shape)}.")

    if not isinstance(ss.name, str) or len(ss.name) == 0:
        raise ValueError(f"name must be a non-empty str, got {ss.name!r}.")

    if ss.weights is not None:
        if not torch.is_tensor(ss.weights):
            raise TypeError(f"weights must be a torch.Tensor or None, got {type(ss.weights).__name__}.")
        if not torch.is_floating_point(ss.weights):
            raise TypeError(f"weights must be a real floating tensor, got dtype {ss.weights.dtype}.")
        S = ss.shifts.shape[0]
        if ss.weights.shape != (S,):
            raise ValueError(f"weights must have shape [{S}], got {list(ss.weights.shape)}.")
        if not torch.isfinite(ss.weights).all():
            raise ValueError("weights must contain only finite values.")
        if (ss.weights < 0).any():
            raise ValueError("weights must be non-negative.")
        if ss.weights.sum() <= 0:
            raise ValueError("weights sum must be positive.")


def default_integer_shift_set(
    max_delay: int,
    max_doppler: int,
    include_zero: bool = False,
    name: str = "default_integer_shift_set",
) -> SparseShiftSet:
    """Build a dense integer shift grid within [-max, +max] in each dimension.

    Constructs all (delay_shift, doppler_shift) pairs with delay_shift in
    [-max_delay, max_delay] and doppler_shift in [-max_doppler, max_doppler].
    If include_zero is False, the zero shift [0, 0] is excluded.

    Args:
        max_delay: Non-negative maximum absolute delay shift.
        max_doppler: Non-negative maximum absolute Doppler shift.
        include_zero: If True, keep the [0, 0] shift entry.
        name: Identifier for the constructed SparseShiftSet.

    Returns:
        SparseShiftSet with shifts [S, 2] of dtype torch.long.

    Raises:
        ValueError: If max_delay or max_doppler are negative, include_zero
            is not bool, name is empty, or the resulting shift set is empty.
    """
    if not isinstance(max_delay, int) or max_delay < 0:
        raise ValueError(f"max_delay must be a non-negative integer, got {max_delay}.")
    if not isinstance(max_doppler, int) or max_doppler < 0:
        raise ValueError(f"max_doppler must be a non-negative integer, got {max_doppler}.")
    if not isinstance(include_zero, bool):
        raise ValueError(f"include_zero must be a bool, got {type(include_zero).__name__}.")
    if not isinstance(name, str) or len(name) == 0:
        raise ValueError(f"name must be a non-empty str, got {name!r}.")

    delay_range = torch.arange(-max_delay, max_delay + 1, dtype=torch.long)
    doppler_range = torch.arange(-max_doppler, max_doppler + 1, dtype=torch.long)
    delay_grid, doppler_grid = torch.meshgrid(delay_range, doppler_range, indexing="ij")
    shifts = torch.stack([delay_grid.reshape(-1), doppler_grid.reshape(-1)], dim=-1)

    if not include_zero:
        keep = (shifts[:, 0] != 0) | (shifts[:, 1] != 0)
        shifts = shifts[keep]

    if shifts.shape[0] == 0:
        raise ValueError(
            "Resulting shift set is empty. Set include_zero=True or increase "
            "max_delay/max_doppler."
        )

    return SparseShiftSet(shifts=shifts, weights=None, name=name)
