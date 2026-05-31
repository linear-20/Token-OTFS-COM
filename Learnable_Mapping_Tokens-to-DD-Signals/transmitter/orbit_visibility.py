"""Sparse DD shift-orbit visibility kernel for optional ablations.

This pre-17C2 diagnostic is not part of the default transmitter objective.

Public API:
    masked_shift_orbit_visibility(codeword_book, shifts, evidence_mask)
        -> real [V, S]

Computes the fraction of each codeword's total energy that remains
inside the evidence region after a circular DD shift:

    eta[v, s] = ||M_e * Pi_s(C_v)||_F^2 / ||C_v||_F^2

where Pi_s is the receiver-aligned integer DD circular shift and
M_e is a hard binary evidence mask applied AFTER shifting.
"""

from __future__ import annotations

import torch

from .dd_shifts import dd_circular_shift_bank
from .orbit_correlation import _prepare_evidence_mask, _real_dtype_for


def masked_shift_orbit_visibility(
    codeword_book: torch.Tensor,
    shifts: torch.Tensor,
    evidence_mask: torch.Tensor,
) -> torch.Tensor:
    """Masked shift-orbit energy visibility.

    For each codeword v and shift s, computes:
        eta[v, s] =
            ||M_e * Pi_s(C_v)||_F^2 / ||C_v||_F^2

    The denominator is the full physical codeword energy (not masked).
    Evidence mask M_e is applied AFTER shifting.  eta == 0 is valid
    and means the shifted codeword has no energy in the evidence region.
    Zero shift [0, 0] is legal for visibility.

    Args:
        codeword_book: Complex tensor [V, M, N] of physical normalized
            codewords. All dims > 0, all values finite.
        shifts: Long tensor [S, 2]. Each row is [delay_shift, doppler_shift].
        evidence_mask: hard binary support mask [1, M, N] or [V, M, N],
            bool or real 0/1. Applied after shifting.

    Returns:
        Real tensor [V, S] with dtype float32 (complex64 input) or
        float64 (complex128 input), on the same device as codeword_book.

    Raises:
        TypeError: If codeword_book, shifts, or mask types are invalid.
        ValueError: If shapes, finiteness, or zero full codeword energy.
    """
    _validate_visibility_inputs(codeword_book, shifts)

    # Prepare evidence mask (reuses Step 13 mask contract).
    mask = _prepare_evidence_mask(evidence_mask, codeword_book)  # [1,M,N] or [V,M,N]
    # Broadcast to [V, 1, M, N] for shift bank.
    mask_bc = mask.reshape(mask.shape[0], 1, mask.shape[1], mask.shape[2])

    # Full codeword energy (denominator).
    full_energy = codeword_book.abs().pow(2).sum(dim=(-2, -1))  # [V]
    if bool((full_energy <= 0).any().item()):
        raise ValueError(
            "masked_shift_orbit_visibility: full codeword energy "
            "<= 0 for at least one token."
        )

    # Shift bank [V, S, M, N]; mask applied AFTER shift.
    shifted_bank = dd_circular_shift_bank(codeword_book, shifts)  # [V, S, M, N]
    masked = shifted_bank * mask_bc  # [V, S, M, N]

    shifted_energy = masked.abs().pow(2).sum(dim=(-2, -1))  # [V, S]

    real_dtype = _real_dtype_for(codeword_book)
    return (shifted_energy / full_energy.unsqueeze(1)).to(dtype=real_dtype)


def _validate_visibility_inputs(
    codeword_book: torch.Tensor,
    shifts: torch.Tensor,
) -> None:
    """Validate codeword_book [V,M,N] complex and shifts long [S,2]."""
    if not torch.is_tensor(codeword_book):
        raise TypeError(
            f"codeword_book must be a torch.Tensor, "
            f"got {type(codeword_book).__name__}."
        )
    if not torch.is_complex(codeword_book):
        raise TypeError(
            f"codeword_book must be a complex tensor, "
            f"got dtype {codeword_book.dtype}."
        )
    if codeword_book.ndim != 3:
        raise ValueError(
            f"codeword_book must have 3 dimensions [V, M, N], "
            f"got ndim={codeword_book.ndim}."
        )
    V, M, N = codeword_book.shape
    if V <= 0:
        raise ValueError(f"codeword_book V must be > 0, got V={V}.")
    if M <= 0:
        raise ValueError(f"codeword_book M must be > 0, got M={M}.")
    if N <= 0:
        raise ValueError(f"codeword_book N must be > 0, got N={N}.")
    if not torch.isfinite(codeword_book).all():
        raise ValueError("codeword_book must contain only finite values.")

    if not torch.is_tensor(shifts):
        raise TypeError(
            f"shifts must be a torch.Tensor, got {type(shifts).__name__}."
        )
    if shifts.dtype != torch.long:
        raise TypeError(
            f"shifts dtype must be torch.long, got {shifts.dtype}."
        )
    if shifts.ndim != 2:
        raise ValueError(
            f"shifts must have 2 dimensions [S, 2], "
            f"got ndim={shifts.ndim}."
        )
    if shifts.shape[1] != 2:
        raise ValueError(
            f"shifts last dim must be 2, got {shifts.shape[1]}."
        )
    if shifts.shape[0] <= 0:
        raise ValueError(
            f"shifts S must be > 0, got S={shifts.shape[0]}."
        )
