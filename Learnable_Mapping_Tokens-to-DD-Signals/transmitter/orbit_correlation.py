"""Masked normalized DD correlation kernels for orbit ablations.

These pre-17C2 diagnostics are not part of the default transmitter objective.

Public API:
    masked_normalized_dd_correlation(reference_dd, candidate_dd, evidence_mask)
        -> real [B]
    masked_shift_orbit_correlation(reference_dd, candidate_dd, shifts, evidence_mask)
        -> real [B, S]

These compute the core Sparse-Aware Token Orbit Shaping metric:
    rho(u, v, delta) =
        |<M_e * C_u, M_e * Pi_delta(C_v)>|
        / (||M_e * C_u||_2 * ||M_e * Pi_delta(C_v)||_2)

where Pi_delta is the receiver-aligned integer DD circular shift and
M_e is a hard binary evidence mask (applied at correlation time, not
at shift time). No epsilon, no clamping, no soft mask projection.
"""

from __future__ import annotations

import torch

from .dd_shifts import dd_circular_shift_bank


def masked_normalized_dd_correlation(
    reference_dd: torch.Tensor,
    candidate_dd: torch.Tensor,
    evidence_mask: torch.Tensor,
) -> torch.Tensor:
    """Batch-wise masked normalized DD correlation.

    Computes for each batch element:
        score[b] = |<M_e * ref[b], M_e * cand[b]>|
                   / (||M_e * ref[b]||_2 * ||M_e * cand[b]||_2)

    The complex inner product uses conj(reference) * candidate.

    Args:
        reference_dd: Complex tensor [B, M, N].
        candidate_dd: Complex tensor [B, M, N] with same shape, device,
            and dtype as reference_dd.
        evidence_mask: hard binary support mask [1, M, N] or [B, M, N],
            bool or real 0/1. Fractional values are rejected.

    Returns:
        Real tensor [B] on the same device as the DD tensors, with
        dtype float32 (complex64 input) or float64 (complex128 input).

    Raises:
        TypeError: If inputs are not tensors, not complex, or mask is
            complex.
        ValueError: If shapes, finiteness, zero masked energy, or
            fractional evidence_mask are violated.
    """
    # Validation (does not modify inputs).
    _validate_dd_pair(reference_dd, candidate_dd)
    mask = _prepare_evidence_mask(evidence_mask, reference_dd)

    return _masked_normalized_correlation_core(
        reference_dd, candidate_dd, mask,
    )


def masked_shift_orbit_correlation(
    reference_dd: torch.Tensor,
    candidate_dd: torch.Tensor,
    shifts: torch.Tensor,
    evidence_mask: torch.Tensor,
) -> torch.Tensor:
    """Masked normalized correlation over an integer DD shift orbit.

    For each shift s in shifts [S, 2], computes:
        rho[b, s] =
            |<M_e * ref[b], M_e * Pi_{s}(cand[b])>|
            / (||M_e * ref[b]||_2 * ||M_e * Pi_{s}(cand[b])||_2)

    Evidence mask M_e is applied AFTER shifting, i.e. the candidate is
    first circularly shifted, then both reference and shifted candidate
    are multiplied by the evidence mask before the inner product.

    This is the core metric for Sparse-Aware Token Orbit Shaping and
    does NOT construct dense matrices or compute full V^2 pair matrices.

    Args:
        reference_dd: Complex tensor [B, M, N].
        candidate_dd: Complex tensor [B, M, N] with same shape, device,
            and dtype as reference_dd.
        shifts: Long tensor [S, 2]. Each row is [delay_shift, doppler_shift].
            Passed to dd_circular_shift_bank.
        evidence_mask: hard binary support mask [1, M, N] or [B, M, N],
            bool or real 0/1. Applied after shifting (mask-after-shift).

    Returns:
        Real tensor [B, S] on the same device as the DD tensors, with
        dtype float32 (complex64 input) or float64 (complex128 input).

    Raises:
        TypeError: If DD inputs, mask, or shifts are invalid.
        ValueError: If shapes, finiteness, or zero masked energy are
            violated (including after any shift).
    """
    _validate_dd_pair(reference_dd, candidate_dd)
    mask = _prepare_evidence_mask(evidence_mask, reference_dd)

    # Shift candidate in batch; no mask projection here.
    shifted_bank = dd_circular_shift_bank(candidate_dd, shifts)  # [B, S, M, N]

    # Vectorized correlation over the S dimension.
    # reference_dd: [B, M, N] -> unsqueeze to [B, 1, M, N]
    ref = reference_dd.unsqueeze(1)  # [B, 1, M, N]
    cand = shifted_bank             # [B, S, M, N]

    # Broadcast mask to [1, 1, M, N] so it broadcasts over B and S.
    mask_bc = mask.reshape(mask.shape[0], 1, mask.shape[1], mask.shape[2])

    ref_m = ref * mask_bc   # [B, 1, M, N]  or  [1, 1, M, N] if mask is [1,M,N]
    cand_m = cand * mask_bc  # [B, S, M, N]

    # Complex inner products.
    num = torch.abs(
        (torch.conj(ref_m) * cand_m).sum(dim=(-2, -1))
    )  # [B, S]  or  [B, 1] broadcast -> [B, S]

    ref_energy = ref_m.abs().pow(2).sum(dim=(-2, -1))  # [B, 1]
    cand_energy = cand_m.abs().pow(2).sum(dim=(-2, -1))  # [B, S]

    if bool((ref_energy <= 0).any().item()):
        raise ValueError(
            "masked_shift_orbit_correlation: masked reference energy "
            "<= 0 for at least one batch element."
        )
    if bool((cand_energy <= 0).any().item()):
        raise ValueError(
            "masked_shift_orbit_correlation: masked candidate energy "
            "<= 0 for at least one (batch, shift) element."
        )

    denom = torch.sqrt(ref_energy * cand_energy)  # [B, S]
    real_dtype = _real_dtype_for(reference_dd)
    return (num / denom).to(dtype=real_dtype)


# -- private helpers -----------------------------------------------------------


def _validate_dd_pair(
    reference_dd: torch.Tensor,
    candidate_dd: torch.Tensor,
) -> None:
    """Validate reference_dd and candidate_dd are matching complex [B,M,N]."""
    _validate_one_dd(reference_dd, "reference_dd")
    _validate_one_dd(candidate_dd, "candidate_dd")
    if reference_dd.shape != candidate_dd.shape:
        raise ValueError(
            "reference_dd and candidate_dd must have the same shape; "
            f"got {list(reference_dd.shape)} and {list(candidate_dd.shape)}."
        )
    if reference_dd.device != candidate_dd.device:
        raise ValueError(
            "reference_dd and candidate_dd must be on the same device; "
            f"got {reference_dd.device} and {candidate_dd.device}."
        )
    if reference_dd.dtype != candidate_dd.dtype:
        raise TypeError(
            "reference_dd and candidate_dd must have the same dtype; "
            f"got {reference_dd.dtype} and {candidate_dd.dtype}."
        )


def _validate_one_dd(t: torch.Tensor, name: str) -> None:
    """Validate a single DD tensor is complex [B,M,N] with all dims > 0."""
    if not torch.is_tensor(t):
        raise TypeError(
            f"{name} must be a torch.Tensor, got {type(t).__name__}."
        )
    if not torch.is_complex(t):
        raise TypeError(
            f"{name} must be a complex tensor, got dtype {t.dtype}."
        )
    if t.ndim != 3:
        raise ValueError(
            f"{name} must have 3 dimensions [B, M, N], got ndim={t.ndim}."
        )
    B, M, N = t.shape
    if B <= 0:
        raise ValueError(f"{name} batch size B must be > 0, got B={B}.")
    if M <= 0:
        raise ValueError(f"{name} M must be > 0, got M={M}.")
    if N <= 0:
        raise ValueError(f"{name} N must be > 0, got N={N}.")
    if not torch.isfinite(t).all():
        raise ValueError(
            f"{name} must contain only finite values "
            "(real and imag parts must both be finite)."
        )


def _prepare_evidence_mask(
    evidence_mask: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Validate and prepare evidence_mask for correlation.

    Returns a real 0/1 mask broadcast-compatible with reference_dd,
    on the reference device and real dtype.
    """
    if not torch.is_tensor(evidence_mask):
        raise TypeError(
            f"evidence_mask must be a torch.Tensor, "
            f"got {type(evidence_mask).__name__}."
        )
    if torch.is_complex(evidence_mask):
        raise TypeError("evidence_mask must not be a complex tensor.")

    if evidence_mask.ndim != 3:
        raise ValueError(
            f"evidence_mask must have 3 dimensions, "
            f"got ndim={evidence_mask.ndim}."
        )
    B_ref, M_ref, N_ref = reference.shape
    if evidence_mask.shape not in ((1, M_ref, N_ref), (B_ref, M_ref, N_ref)):
        raise ValueError(
            f"evidence_mask must have shape [1, {M_ref}, {N_ref}] or "
            f"[{B_ref}, {M_ref}, {N_ref}], got {list(evidence_mask.shape)}."
        )

    # Preserve source dtype so conversion cannot round fractional values to 0/1.
    mask_cpu = evidence_mask.detach().to(
        device=torch.device("cpu"), copy=True,
    )
    if not torch.isfinite(mask_cpu).all():
        raise ValueError("evidence_mask must contain only finite values.")
    invalid = (mask_cpu != 0) & (mask_cpu != 1)
    if invalid.any():
        raise ValueError(
            "evidence_mask must be a hard binary support mask "
            "(only 0 or 1 values). Fractional values are not allowed."
        )

    # Move to reference device/real dtype.
    real_dtype = _real_dtype_for(reference)
    mask = evidence_mask.to(device=reference.device, dtype=real_dtype)

    # Check at least one active bin per batch element (after broadcast).
    active = mask > 0
    if not active.any():
        raise ValueError(
            "evidence_mask has no active bins. "
            "At least one position must be True/1."
        )
    if mask.shape[0] == B_ref:
        if not active.any(dim=(-2, -1)).all():
            raise ValueError(
                "evidence_mask has batch-wise elements with no active bins."
            )

    return mask


def _masked_normalized_correlation_core(
    reference_dd: torch.Tensor,
    candidate_dd: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Core batched masked normalized correlation (no validation)."""
    ref = reference_dd * mask
    cand = candidate_dd * mask

    num = torch.abs(
        (torch.conj(ref) * cand).sum(dim=(-2, -1))
    )  # [B]

    ref_energy = ref.abs().pow(2).sum(dim=(-2, -1))  # [B]
    cand_energy = cand.abs().pow(2).sum(dim=(-2, -1))  # [B]

    if bool((ref_energy <= 0).any().item()):
        raise ValueError(
            "masked_normalized_dd_correlation: masked reference energy "
            "<= 0 for at least one batch element."
        )
    if bool((cand_energy <= 0).any().item()):
        raise ValueError(
            "masked_normalized_dd_correlation: masked candidate energy "
            "<= 0 for at least one batch element."
        )

    denom = torch.sqrt(ref_energy * cand_energy)
    real_dtype = _real_dtype_for(reference_dd)
    return (num / denom).to(dtype=real_dtype)


def _real_dtype_for(t: torch.Tensor) -> torch.dtype:
    """Map complex tensor dtype to corresponding real dtype."""
    return {torch.complex64: torch.float32, torch.complex128: torch.float64}[t.dtype]
