"""Optional pre-17C2 orbit regularizers for controlled ablations.

These objectives are not part of the default transmitter algorithm. The main
training objective is sparse_multipath_operator_margin_loss.

Public API:
    cross_token_orbit_confusion_scores(codeword_book, token_pairs, shift_set,
        evidence_mask) -> real [P, S]
    cross_token_orbit_confusion_loss(codeword_book, token_pairs, shift_set,
        evidence_mask, *, mode, risk_temperature) -> scalar []
    self_shift_orbit_sidelobe_scores(codeword_book, shift_set,
        evidence_mask) -> real [V, S]
    self_shift_orbit_sidelobe_loss(codeword_book, shift_set, evidence_mask,
        *, mode, risk_temperature) -> scalar []
    shift_orbit_visibility_scores(codeword_book, shift_set, evidence_mask)
        -> real [V, S]
    shift_orbit_visibility_loss(codeword_book, shift_set, evidence_mask,
        *, minimum_visibility) -> scalar []

This is the first deterministic mathematical kernel for SATO-Shaping
(Sparse-Aware Token Orbit). It computes cross-token confusion over
explicitly enumerated pairs only -- never a full V^2 pair matrix.
"""

from __future__ import annotations

import math

import torch

from .orbit_correlation import masked_shift_orbit_correlation
from .orbit_visibility import masked_shift_orbit_visibility
from .shaping import SparseShiftSet


def cross_token_orbit_confusion_scores(
    codeword_book: torch.Tensor,
    token_pairs: torch.Tensor,
    shift_set: SparseShiftSet,
    evidence_mask: torch.Tensor,
) -> torch.Tensor:
    """Cross-token sparse DD shift-orbit coherence scores.

    For each pair p = (u, v) and shift s, computes:
        rho[p, s] =
            |<M_e * C_u, M_e * Pi_s(C_v)>|
            / (||M_e * C_u||_F * ||M_e * Pi_s(C_v)||_F)

    where Pi_s is the receiver-aligned integer DD circular shift and
    M_e is a hard binary evidence mask applied after shifting.

    Complexity: O(P * S * M * N).  No V^2 matrix is constructed.

    Lifecycle: codeword_book should be the physical normalized book
    produced by TokenDDCodebook.forward(data_mask), i.e. hard data-mask
    projected and equal-power normalized. This function only validates
    tensor contract; it cannot detect whether the caller passed raw
    learnable parameters (raw_real/raw_imag) by mistake.

    Args:
        codeword_book: Complex tensor [V, M, N] of physical normalized
            codewords. All dims > 0, all values finite.
        token_pairs: Long tensor [P, 2] of (u, v) index pairs with
            P > 0, indices in [0, V), and u != v.
        shift_set: SparseShiftSet whose .shifts are long [S, 2].
        evidence_mask: hard binary support mask [1, M, N] or [P, M, N],
            bool or real 0/1. Applied after shifting.

    Returns:
        Real tensor [P, S] with dtype float32 (complex64 input) or
        float64 (complex128 input), on the same device as codeword_book.

    Raises:
        TypeError: If codeword_book, token_pairs, shift_set, or mask
            types are invalid.
        ValueError: If shapes, ranges, or u == v are violated.
    """
    _validate_physical_codeword_book(codeword_book)
    V = codeword_book.shape[0]
    _validate_cross_token_pairs(token_pairs, V)

    if not isinstance(shift_set, SparseShiftSet):
        raise TypeError(
            f"shift_set must be a SparseShiftSet, "
            f"got {type(shift_set).__name__}."
        )

    shifts = shift_set.shifts  # long [S, 2]

    # Index into codeword_book by token ids.
    # token_pairs may be on CPU while codeword_book is on GPU.
    u_idx = token_pairs[:, 0].to(device=codeword_book.device)
    v_idx = token_pairs[:, 1].to(device=codeword_book.device)

    reference_dd = codeword_book[u_idx]   # [P, M, N]
    candidate_dd = codeword_book[v_idx]   # [P, M, N]

    return masked_shift_orbit_correlation(
        reference_dd,
        candidate_dd,
        shifts,
        evidence_mask,
    )


def cross_token_orbit_confusion_loss(
    codeword_book: torch.Tensor,
    token_pairs: torch.Tensor,
    shift_set: SparseShiftSet,
    evidence_mask: torch.Tensor,
    *,
    mode: str = "smooth_peak",
    risk_temperature: float = 0.1,
) -> torch.Tensor:
    """Aggregate cross-token orbit confusion scores into a scalar loss.

    Computes the confusion scores via cross_token_orbit_confusion_scores,
    then aggregates them according to the chosen mode.

    Modes:
        "weighted_isl"  -- mean_p sum_s q[s] * rho[p,s]^2
        "peak"          -- max over all configured shifts, ignoring q
        "smooth_peak"   -- tau * log(mean_p sum_s q[s] * exp(rho / tau))

    As tau -> 0, smooth_peak approaches max over (p, s) with q[s] > 0.
    It only approaches the unconditional global peak when all q[s] > 0.
    q[s] == 0 is legal and excludes that shift from the weighted tail-risk.
    No epsilon is added; the formula is used as stated.

    Shift weights q are obtained from shift_set.normalized_weights().

    Args:
        codeword_book: Complex tensor [V, M, N].
        token_pairs: Long tensor [P, 2].
        shift_set: SparseShiftSet.
        evidence_mask: hard binary support mask [1, M, N] or [P, M, N].
        mode: Aggregation mode: "weighted_isl", "peak", or "smooth_peak".
        risk_temperature: Positive finite float (not bool) for smooth_peak
            mode. Validated for all modes.

    Returns:
        Scalar real tensor [] on the same device as codeword_book.

    Raises:
        TypeError/ValueError: From score computation, mode, temperature,
            or shift weights validation.
    """
    # Fail-fast: validate mode, temperature, and raw shift weights
    # BEFORE the expensive O(P*S*M*N) score computation.
    _validate_mode(mode)
    _validate_risk_temperature(risk_temperature)
    _validate_current_shift_weights(shift_set)

    scores = cross_token_orbit_confusion_scores(
        codeword_book, token_pairs, shift_set, evidence_mask,
    )  # [P, S]

    q = _normalized_shift_weights(shift_set, scores)

    if mode == "weighted_isl":
        loss = (scores.pow(2) * q.view(1, -1)).sum(dim=1).mean()
    elif mode == "peak":
        # q is validated but deliberately not used.
        loss = scores.max()
    elif mode == "smooth_peak":
        loss = _smooth_peak_loss(scores, q, risk_temperature)
    else:
        raise ValueError(f"Unknown mode: '{mode}'.")  # pragma: no cover

    return loss


# -- private validators --------------------------------------------------------


def _validate_physical_codeword_book(t: torch.Tensor) -> None:
    """Validate codeword_book is a complex finite [V, M, N] tensor."""
    if not torch.is_tensor(t):
        raise TypeError(
            f"codeword_book must be a torch.Tensor, got {type(t).__name__}."
        )
    if not torch.is_complex(t):
        raise TypeError(
            f"codeword_book must be a complex tensor, got dtype {t.dtype}."
        )
    if t.ndim != 3:
        raise ValueError(
            f"codeword_book must have 3 dimensions [V, M, N], "
            f"got ndim={t.ndim}."
        )
    V, M, N = t.shape
    if V <= 0:
        raise ValueError(f"codeword_book V must be > 0, got V={V}.")
    if M <= 0:
        raise ValueError(f"codeword_book M must be > 0, got M={M}.")
    if N <= 0:
        raise ValueError(f"codeword_book N must be > 0, got N={N}.")
    if not torch.isfinite(t).all():
        raise ValueError(
            "codeword_book must contain only finite values."
        )


def _validate_cross_token_pairs(
    token_pairs: torch.Tensor,
    V: int,
) -> None:
    """Validate token_pairs is a long [P, 2] tensor with valid indices."""
    if not torch.is_tensor(token_pairs):
        raise TypeError(
            f"token_pairs must be a torch.Tensor, "
            f"got {type(token_pairs).__name__}."
        )
    if token_pairs.dtype != torch.long:
        raise TypeError(
            f"token_pairs dtype must be torch.long, got {token_pairs.dtype}."
        )
    if token_pairs.ndim != 2:
        raise ValueError(
            f"token_pairs must have 2 dimensions [P, 2], "
            f"got ndim={token_pairs.ndim}."
        )
    if token_pairs.shape[1] != 2:
        raise ValueError(
            f"token_pairs last dim must be 2, got {token_pairs.shape[1]}."
        )
    P = token_pairs.shape[0]
    if P <= 0:
        raise ValueError(f"token_pairs P must be > 0, got P={P}.")

    if (token_pairs < 0).any() or (token_pairs >= V).any():
        raise ValueError(
            f"token_pairs indices must be in [0, V={V})."
        )
    if (token_pairs[:, 0] == token_pairs[:, 1]).any():
        raise ValueError(
            "token_pairs must not contain self-pairs (u == v). "
            "This function computes cross-token orbit confusion only."
        )


def _validate_mode(mode: str) -> None:
    """Validate mode is one of the supported aggregation modes."""
    if not isinstance(mode, str):
        raise TypeError(
            f"mode must be str, got {type(mode).__name__}."
        )
    if mode not in {"weighted_isl", "peak", "smooth_peak"}:
        raise ValueError(
            f"mode must be 'weighted_isl', 'peak', or 'smooth_peak', "
            f"got '{mode}'."
        )


def _validate_risk_temperature(value: float) -> None:
    """Validate risk_temperature is a finite positive number, not bool."""
    if isinstance(value, bool):
        raise TypeError(
            f"risk_temperature must be a number, not bool."
        )
    if not isinstance(value, (int, float)):
        raise TypeError(
            f"risk_temperature must be a number, "
            f"got {type(value).__name__}."
        )
    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            f"risk_temperature must be finite positive, got {value}."
        )


def _validate_current_shift_weights(shift_set: SparseShiftSet) -> None:
    """Re-validate raw shift weights to catch post-construction corruption.

    If shift_set.weights is None (uniform), accept immediately.
    Otherwise re-check: tensor, real floating, shape [S], finite,
    non-negative, sum > 0.  Negative raw weights must be rejected
    before normalization can hide them.
    """
    if not isinstance(shift_set, SparseShiftSet):
        raise TypeError(
            f"shift_set must be a SparseShiftSet, "
            f"got {type(shift_set).__name__}."
        )
    w = shift_set.weights
    if w is None:
        return
    if not torch.is_tensor(w):
        raise TypeError(
            f"shift_set.weights must be a torch.Tensor or None, "
            f"got {type(w).__name__}."
        )
    if not torch.is_floating_point(w):
        raise TypeError(
            f"shift_set.weights must be a real floating tensor, "
            f"got dtype {w.dtype}."
        )
    S = shift_set.shifts.shape[0]
    if w.shape != (S,):
        raise ValueError(
            f"shift_set.weights must have shape [{S}], "
            f"got {list(w.shape)}."
        )
    if not torch.isfinite(w).all():
        raise ValueError("shift_set.weights must contain only finite values.")
    if (w < 0).any():
        raise ValueError(
            "shift_set.weights must be non-negative. "
            "Negative raw weights cannot be fixed by normalization."
        )
    if w.sum().item() <= 0:
        raise ValueError("shift_set.weights sum must be > 0.")


def _normalized_shift_weights(
    shift_set: SparseShiftSet,
    scores: torch.Tensor,
) -> torch.Tensor:
    """Obtain and validate normalized shift weights [S]."""
    q = shift_set.normalized_weights(
        device=scores.device, dtype=scores.dtype,
    )
    if not torch.is_tensor(q):
        raise TypeError(
            "shift_set.normalized_weights() must return a torch.Tensor."
        )
    if q.ndim != 1:
        raise ValueError(
            f"normalized_weights must return 1-D tensor [S], "
            f"got ndim={q.ndim}."
        )
    S_scores = scores.shape[1]
    if q.shape[0] != S_scores:
        raise ValueError(
            f"normalized_weights shape [{q.shape[0]}] must match "
            f"shifts S={S_scores}."
        )
    if torch.is_complex(q):
        raise TypeError("normalized_weights must return a real tensor.")
    if not torch.isfinite(q).all():
        raise ValueError("normalized_weights must be finite.")
    if (q < 0).any():
        raise ValueError("normalized_weights must be non-negative.")
    if q.sum().item() <= 0:
        raise ValueError("normalized_weights sum must be > 0.")
    return q


def _smooth_peak_loss(
    scores: torch.Tensor,
    q: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """Compute smooth_peak loss via numerically stable logsumexp.

    loss = tau * log( mean_p sum_s q[s] * exp(rho[p,s] / tau) )

    q zeros are handled naturally: log(0) = -inf excludes them from
    the logsumexp. No epsilon is added.
    """
    P = scores.shape[0]
    log_q = torch.log(q)  # [S]; log(0) = -inf is acceptable

    # scores / tau: [P, S]; log_q: [S] -> broadcast to [P, S]
    log_terms = scores / tau + log_q.view(1, -1)  # [P, S]
    # Flatten to [P*S] for single logsumexp over all pairs and shifts.
    lse = torch.logsumexp(log_terms.reshape(-1), dim=0)  # scalar
    return tau * (lse - math.log(P))


# -- self-shift orbit sidelobe ------------------------------------------------


def self_shift_orbit_sidelobe_scores(
    codeword_book: torch.Tensor,
    shift_set: SparseShiftSet,
    evidence_mask: torch.Tensor,
) -> torch.Tensor:
    """Self-shift masked DD orbit sidelobe scores.

    For each token v and nonzero shift s, computes:
        rho_self[v, s] =
            |<M_e * C_v, M_e * Pi_s(C_v)>|
            / (||M_e * C_v||_F * ||M_e * Pi_s(C_v)||_F)

    Zero shift [0, 0] is the main lobe (score always 1) and is
    explicitly rejected. This function only measures nonzero-shift
    sidelobe coherence of each codeword with its own shifted copies.

    Lifecycle: codeword_book should be the physical normalized book
    produced by TokenDDCodebook.forward(data_mask).

    Complexity: O(V * S * M * N).  No V^2 matrix is constructed.

    Args:
        codeword_book: Complex tensor [V, M, N] of physical normalized
            codewords. All dims > 0, all values finite.
        shift_set: SparseShiftSet whose .shifts are long [S, 2] and
            contain no zero shift [0, 0].
        evidence_mask: hard binary support mask [1, M, N] or [V, M, N],
            bool or real 0/1. Applied after shifting.

    Returns:
        Real tensor [V, S] with dtype float32 (complex64 input) or
        float64 (complex128 input), on the same device as codeword_book.

    Raises:
        TypeError: If codeword_book, shift_set, or mask types are invalid.
        ValueError: If shapes, finiteness, or zero shift is present.
    """
    _validate_physical_codeword_book(codeword_book)
    shifts = _validate_nonzero_shift_set(shift_set)

    # Self-shift: reference and candidate are both the full codeword_book.
    return masked_shift_orbit_correlation(
        codeword_book,
        codeword_book,
        shifts,
        evidence_mask,
    )


def self_shift_orbit_sidelobe_loss(
    codeword_book: torch.Tensor,
    shift_set: SparseShiftSet,
    evidence_mask: torch.Tensor,
    *,
    mode: str = "smooth_peak",
    risk_temperature: float = 0.1,
) -> torch.Tensor:
    """Aggregate self-shift sidelobe scores into a scalar loss.

    Aggregation modes are the same as cross_token_orbit_confusion_loss:
        "weighted_isl"  -- mean_v sum_s q[s] * rho_self[v,s]^2
        "peak"          -- max over all configured nonzero shifts
        "smooth_peak"   -- tau * log(mean_v sum_s q[s] * exp(rho / tau))

    Zero shift [0, 0] is rejected. Shift weights q come from
    shift_set.normalized_weights().  q[s] == 0 is legal and excludes
    that shift from weighted aggregation.

    Args:
        codeword_book: Complex tensor [V, M, N].
        shift_set: SparseShiftSet with nonzero shifts only.
        evidence_mask: hard binary support mask [1, M, N] or [V, M, N].
        mode: "weighted_isl", "peak", or "smooth_peak".
        risk_temperature: Positive finite float (not bool).

    Returns:
        Scalar real tensor [] on the same device as codeword_book.
    """
    # Fail-fast checks before expensive O(V*S*M*N) computation.
    _validate_mode(mode)
    _validate_risk_temperature(risk_temperature)
    _validate_nonzero_shift_set(shift_set)
    _validate_current_shift_weights(shift_set)

    scores = self_shift_orbit_sidelobe_scores(
        codeword_book, shift_set, evidence_mask,
    )  # [V, S]

    q = _normalized_shift_weights(shift_set, scores)

    if mode == "weighted_isl":
        loss = (scores.pow(2) * q.view(1, -1)).sum(dim=1).mean()
    elif mode == "peak":
        loss = scores.max()
    elif mode == "smooth_peak":
        loss = _smooth_peak_loss(scores, q, risk_temperature)
    else:
        raise ValueError(f"Unknown mode: '{mode}'.")  # pragma: no cover

    return loss


def _validate_nonzero_shift_set(
    shift_set: SparseShiftSet,
) -> torch.Tensor:
    """Validate shift_set is a SparseShiftSet with nonzero shifts [S, 2].

    Returns the validated shifts tensor.
    """
    if not isinstance(shift_set, SparseShiftSet):
        raise TypeError(
            f"shift_set must be a SparseShiftSet, "
            f"got {type(shift_set).__name__}."
        )
    shifts = shift_set.shifts
    if not torch.is_tensor(shifts):
        raise TypeError(
            "shift_set.shifts must be a torch.Tensor, "
            f"got {type(shifts).__name__}."
        )
    if shifts.dtype != torch.long:
        raise TypeError(
            f"shift_set.shifts dtype must be torch.long, "
            f"got {shifts.dtype}."
        )
    if shifts.ndim != 2:
        raise ValueError(
            f"shift_set.shifts must have 2 dimensions [S, 2], "
            f"got ndim={shifts.ndim}."
        )
    if shifts.shape[1] != 2:
        raise ValueError(
            f"shift_set.shifts last dim must be 2, "
            f"got {shifts.shape[1]}."
        )
    S = shifts.shape[0]
    if S <= 0:
        raise ValueError(
            f"shift_set.shifts S must be > 0, got S={S}."
        )
    # Reject zero shift [0, 0].
    if ((shifts[:, 0] == 0) & (shifts[:, 1] == 0)).any():
        raise ValueError(
            "shift_set.shifts must not contain the zero shift [0, 0]. "
            "Zero shift is the main lobe (score always 1) and must not "
            "be included in sidelobe loss."
        )
    return shifts


# -- shift-orbit visibility ---------------------------------------------------


def shift_orbit_visibility_scores(
    codeword_book: torch.Tensor,
    shift_set: SparseShiftSet,
    evidence_mask: torch.Tensor,
) -> torch.Tensor:
    """Shift-orbit energy visibility scores for each codeword and shift.

    Computes eta[v, s] = ||M_e * Pi_s(C_v)||_F^2 / ||C_v||_F^2.

    Zero shift [0, 0] is legal.  eta == 0 is valid.

    Lifecycle: codeword_book should be the physical normalized book
    produced by TokenDDCodebook.forward(data_mask).

    Args:
        codeword_book: Complex tensor [V, M, N] of physical normalized
            codewords.
        shift_set: SparseShiftSet whose .shifts are long [S, 2].
        evidence_mask: hard binary support mask [1, M, N] or [V, M, N].

    Returns:
        Real tensor [V, S] on the same device as codeword_book.
    """
    _validate_physical_codeword_book(codeword_book)
    if not isinstance(shift_set, SparseShiftSet):
        raise TypeError(
            f"shift_set must be a SparseShiftSet, "
            f"got {type(shift_set).__name__}."
        )
    return masked_shift_orbit_visibility(
        codeword_book, shift_set.shifts, evidence_mask,
    )


def shift_orbit_visibility_loss(
    codeword_book: torch.Tensor,
    shift_set: SparseShiftSet,
    evidence_mask: torch.Tensor,
    *,
    minimum_visibility: float = 0.5,
) -> torch.Tensor:
    """Weighted hinge loss penalizing visibility below a floor.

        L = mean_v sum_s q[s] * relu(minimum_visibility - eta[v,s])^2

    Shift weights q come from shift_set.normalized_weights().
    minimum_visibility == 0 yields loss == 0 identically.

    Args:
        codeword_book: Complex tensor [V, M, N].
        shift_set: SparseShiftSet.
        evidence_mask: hard binary support mask [1, M, N] or [V, M, N].
        minimum_visibility: Float in [0, 1] (not bool).

    Returns:
        Scalar real tensor [] on the same device as codeword_book.
    """
    # Fail-fast: validate parameters before expensive computation.
    _validate_minimum_visibility(minimum_visibility)
    _validate_shift_set_for_regularizer(shift_set)
    _validate_current_shift_weights(shift_set)

    eta = shift_orbit_visibility_scores(
        codeword_book, shift_set, evidence_mask,
    )  # [V, S]

    q = _normalized_shift_weights(shift_set, eta)
    # Hinge: relu(floor - eta)^2, weighted by q.
    penalty = torch.relu(minimum_visibility - eta).pow(2)  # [V, S]
    return (penalty * q.view(1, -1)).sum(dim=1).mean()  # scalar []


# -- additional private validators --------------------------------------------


def _validate_minimum_visibility(value: float) -> None:
    """Validate minimum_visibility is a finite number in [0, 1], not bool."""
    if isinstance(value, bool):
        raise TypeError(
            "minimum_visibility must be a number, not bool."
        )
    if not isinstance(value, (int, float)):
        raise TypeError(
            f"minimum_visibility must be a number, "
            f"got {type(value).__name__}."
        )
    if not math.isfinite(value):
        raise ValueError(
            f"minimum_visibility must be finite, got {value}."
        )
    if value < 0.0 or value > 1.0:
        raise ValueError(
            f"minimum_visibility must be in [0, 1], got {value}."
        )


def _validate_shift_set_for_regularizer(
    shift_set: SparseShiftSet,
) -> None:
    """Validate shift_set is a SparseShiftSet with valid shifts [S, 2].

    Unlike _validate_nonzero_shift_set, zero shift [0, 0] is allowed.
    """
    if not isinstance(shift_set, SparseShiftSet):
        raise TypeError(
            f"shift_set must be a SparseShiftSet, "
            f"got {type(shift_set).__name__}."
        )
    shifts = shift_set.shifts
    if not torch.is_tensor(shifts):
        raise TypeError(
            "shift_set.shifts must be a torch.Tensor, "
            f"got {type(shifts).__name__}."
        )
    if shifts.dtype != torch.long:
        raise TypeError(
            f"shift_set.shifts dtype must be torch.long, "
            f"got {shifts.dtype}."
        )
    if shifts.ndim != 2:
        raise ValueError(
            f"shift_set.shifts must have 2 dimensions [S, 2], "
            f"got ndim={shifts.ndim}."
        )
    if shifts.shape[1] != 2:
        raise ValueError(
            f"shift_set.shifts last dim must be 2, "
            f"got {shifts.shape[1]}."
        )
    if shifts.shape[0] <= 0:
        raise ValueError(
            f"shift_set.shifts S must be > 0, got S={shifts.shape[0]}."
        )
