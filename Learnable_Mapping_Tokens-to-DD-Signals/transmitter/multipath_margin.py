"""Receiver-aligned sparse multipath operator separation scores.

Public API:
    sparse_multipath_operator_separation_scores(codeword_book, token_pairs,
        scenario_bank, evidence_mask) -> real [P, R]

This is the separation-energy kernel for the multipath margin loss.
It is NOT a margin loss itself and NOT a posterior probability.
"""

from __future__ import annotations

import math

import torch

from .multipath_scenarios import SparseMultipathScenarioBank
from .sparse_multipath import (
    SparseMultipathDDChannel,
    apply_sparse_multipath_dd_operator,
)


def sparse_multipath_operator_separation_scores(
    codeword_book: torch.Tensor,
    token_pairs: torch.Tensor,
    scenario_bank: SparseMultipathScenarioBank,
    evidence_mask: torch.Tensor,
) -> torch.Tensor:
    """Receiver-aligned sparse multipath operator separation scores.

    For each token pair p = (u, v) and fixed multipath scenario r:
        d[p, r] = ||M_e * H_r(C_u - C_v)||_F^2 / |Omega_e|

    where H_r applies the sparse multipath operator and M_e is a
    shared hard binary evidence mask applied AFTER the operator.

    Destructive cancellation d == 0 is valid and does not raise.

    Complexity: O(P * R * K * M * N). No dense MN x MN matrix is built.
    Scores are a received DD separation-energy proxy, NOT a posterior
    probability, NOT a TER/BER estimate, and NOT a full OTFS channel.

    Args:
        codeword_book: Complex tensor [V, M, N] of physical normalized
            codewords (from TokenDDCodebook.forward(data_mask)).
        token_pairs: Long tensor [P, 2] of ordered non-self pairs with
            indices in [0, V). Duplicates are preserved.
        scenario_bank: SparseMultipathScenarioBank with R scenarios
            each having K path slots.
        evidence_mask: Shared hard binary support mask [1, M, N],
            bool or real 0/1. Applied after H_r.

    Returns:
        Real tensor [P, R] with dtype float32 (complex64 input) or
        float64 (complex128 input), on codeword_book.device.
    """
    _validate_separation_inputs(
        codeword_book, token_pairs, scenario_bank, evidence_mask,
    )

    V, M, N = codeword_book.shape
    P = token_pairs.shape[0]
    real_dtype = {torch.complex64: torch.float32,
                  torch.complex128: torch.float64}[codeword_book.dtype]

    # -- prepare mask on target device ----------------------------------------
    mask = _prepare_shared_evidence_mask(evidence_mask, M, N,
                                         codeword_book.device, real_dtype)
    active_count = mask.sum()  # scalar, |Omega_e|

    # -- delta = C_u - C_v ----------------------------------------------------
    u_idx = token_pairs[:, 0].to(device=codeword_book.device)
    v_idx = token_pairs[:, 1].to(device=codeword_book.device)
    delta = codeword_book[u_idx] - codeword_book[v_idx]  # [P, M, N]

    # -- materialize and expand scenario channel ------------------------------
    ch_R = scenario_bank.materialize_channel(
        device=codeword_book.device, dtype=codeword_book.dtype,
    )
    R = ch_R.path_shifts.shape[0]
    K = ch_R.path_shifts.shape[1]

    ch_flat = _expand_scenario_channel_for_pairs(ch_R, P, R, K)
    delta_flat = delta.unsqueeze(1).expand(P, R, M, N).reshape(P * R, M, N)

    # -- apply operator -------------------------------------------------------
    y_flat = apply_sparse_multipath_dd_operator(delta_flat, ch_flat)
    y = y_flat.reshape(P, R, M, N)  # [P, R, M, N]

    # -- apply evidence mask AFTER operator -----------------------------------
    masked_y = y * mask.reshape(1, 1, M, N)
    scores = masked_y.abs().pow(2).sum(dim=(-2, -1)) / active_count  # [P, R]

    return scores.to(dtype=real_dtype)


# -- private helpers -----------------------------------------------------------


def _validate_separation_inputs(
    codeword_book: torch.Tensor,
    token_pairs: torch.Tensor,
    scenario_bank: SparseMultipathScenarioBank,
    evidence_mask: torch.Tensor,
) -> None:
    """Validate all inputs for separation score computation."""
    # codeword_book
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

    # token_pairs
    if not torch.is_tensor(token_pairs):
        raise TypeError(
            f"token_pairs must be a torch.Tensor, "
            f"got {type(token_pairs).__name__}."
        )
    if token_pairs.dtype != torch.long:
        raise TypeError(
            f"token_pairs dtype must be torch.long, "
            f"got {token_pairs.dtype}."
        )
    if token_pairs.ndim != 2 or token_pairs.shape[1] != 2:
        raise ValueError(
            f"token_pairs must have shape [P, 2], "
            f"got {list(token_pairs.shape)}."
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
            "token_pairs must not contain self-pairs (u == v)."
        )

    # scenario_bank
    if not isinstance(scenario_bank, SparseMultipathScenarioBank):
        raise TypeError(
            f"scenario_bank must be a SparseMultipathScenarioBank, "
            f"got {type(scenario_bank).__name__}."
        )

    # evidence_mask: shared [1, M, N] only
    if not torch.is_tensor(evidence_mask):
        raise TypeError(
            f"evidence_mask must be a torch.Tensor, "
            f"got {type(evidence_mask).__name__}."
        )
    if torch.is_complex(evidence_mask):
        raise TypeError("evidence_mask must not be a complex tensor.")
    if evidence_mask.ndim != 3:
        raise ValueError(
            f"evidence_mask must have 3 dimensions [1, M, N], "
            f"got ndim={evidence_mask.ndim}."
        )
    if evidence_mask.shape != (1, M, N):
        raise ValueError(
            f"evidence_mask must have shape [1, {M}, {N}], "
            f"got {list(evidence_mask.shape)}."
        )


def _prepare_shared_evidence_mask(
    evidence_mask: torch.Tensor,
    M: int,
    N: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Validate shared [1, M, N] hard binary mask and move to device."""
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
    mask = evidence_mask.to(device=device, dtype=dtype)
    if mask.sum().item() <= 0:
        raise ValueError(
            "evidence_mask active bin count |Omega_e| must be > 0."
        )
    return mask


def _expand_scenario_channel_for_pairs(
    ch_R: SparseMultipathDDChannel,
    P: int,
    R: int,
    K: int,
) -> SparseMultipathDDChannel:
    """Expand a [R, K, *] channel to [P*R, K, *] for pair-scenario batch."""
    shifts = ch_R.path_shifts.unsqueeze(0).expand(P, R, K, 2).reshape(
        P * R, K, 2,
    )
    gains = ch_R.path_gains.unsqueeze(0).expand(P, R, K).reshape(
        P * R, K,
    )
    mask = None
    if ch_R.path_active_mask is not None:
        mask = (ch_R.path_active_mask.unsqueeze(0)
                .expand(P, R, K).reshape(P * R, K))
    return SparseMultipathDDChannel(
        path_shifts=shifts,
        path_gains=gains,
        path_active_mask=mask,
    )


# -- margin loss --------------------------------------------------------------


def sparse_multipath_operator_margin_loss(
    codeword_book: torch.Tensor,
    token_pairs: torch.Tensor,
    scenario_bank: SparseMultipathScenarioBank,
    evidence_mask: torch.Tensor,
    *,
    target_margin: float = 1.0,
) -> torch.Tensor:
    """Scenario-weighted sparse multipath operator hinge margin loss.

        L = mean_p sum_r w[r] * relu(target_margin - d[p,r])^2

    where d[p,r] (real [P, R]) comes from sparse_multipath_operator_separation_scores
    and w[r] = scenario_bank.normalized_weights(...).

    target_margin is one global physical hyperparameter shared across
    all token pairs and scenarios. No per-token margin, no semantic
    weighting, no unequal error protection, no resource allocation.

    This is a receiver-aligned sparse multipath separation-energy
    hinge margin proxy. It is NOT a posterior probability, NOT a
    TER/BER, NOT a complete OTFS channel model.

    target_margin == 0 is legal and keeps the autograd graph intact.

    Args:
        codeword_book: Complex tensor [V, M, N] of physical normalized
            codewords.
        token_pairs: Long tensor [P, 2] of ordered non-self pairs.
        scenario_bank: SparseMultipathScenarioBank with R scenarios.
        evidence_mask: Shared hard binary support mask [1, M, N].
        target_margin: Global physical margin >= 0 (float, not bool).

    Returns:
        Scalar real tensor [] on codeword_book.device.
    """
    # Fail-fast: validate target_margin and scenario weights before
    # expensive O(P*R*K*M*N) score computation.
    _validate_target_margin(target_margin, codeword_book)
    _validate_scenario_bank_weights_current(scenario_bank)

    scores = sparse_multipath_operator_separation_scores(
        codeword_book, token_pairs, scenario_bank, evidence_mask,
    )  # [P, R]

    w = scenario_bank.normalized_weights(
        device=scores.device, dtype=scores.dtype,
    )
    _validate_normalized_scenario_weights(w, scores)

    margin = torch.as_tensor(target_margin, device=scores.device,
                             dtype=scores.dtype)
    penalties = torch.relu(margin - scores).pow(2)  # [P, R]
    loss = (penalties * w.view(1, -1)).sum(dim=1).mean()  # scalar []

    return loss


# -- additional private validators --------------------------------------------


def _validate_target_margin(
    value: float,
    codeword_book: torch.Tensor,
) -> None:
    """Validate target_margin is finite, non-negative, and representable."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"target_margin must be a number, "
            f"got {type(value).__name__}."
        )
    try:
        is_finite = math.isfinite(value)
    except OverflowError:
        is_finite = False
    if not is_finite:
        raise ValueError(
            f"target_margin must be finite, got {value}."
        )
    if value < 0.0:
        raise ValueError(
            f"target_margin must be >= 0, got {value}."
        )
    if (torch.is_tensor(codeword_book)
            and codeword_book.dtype in (torch.complex64, torch.complex128)):
        real_dtype = {
            torch.complex64: torch.float32,
            torch.complex128: torch.float64,
        }[codeword_book.dtype]
        try:
            converted = torch.as_tensor(value, dtype=real_dtype)
        except (RuntimeError, OverflowError, TypeError) as exc:
            raise ValueError(
                f"target_margin must be representable as {real_dtype}, "
                f"got {value}."
            ) from exc
        if not torch.isfinite(converted).item():
            raise ValueError(
                f"target_margin must be representable as {real_dtype}, "
                f"got {value}."
            )


def _validate_scenario_bank_weights_current(
    scenario_bank: SparseMultipathScenarioBank,
) -> None:
    """Re-validate raw scenario_weights to catch post-construction corruption."""
    if not isinstance(scenario_bank, SparseMultipathScenarioBank):
        raise TypeError(
            f"scenario_bank must be a SparseMultipathScenarioBank, "
            f"got {type(scenario_bank).__name__}."
        )
    sw = scenario_bank.scenario_weights
    if sw is None:
        return
    R = scenario_bank.channel.path_gains.shape[0]
    if not torch.is_tensor(sw):
        raise TypeError(
            f"scenario_bank.scenario_weights must be a torch.Tensor, "
            f"got {type(sw).__name__}."
        )
    if not sw.dtype.is_floating_point:
        raise TypeError(
            f"scenario_bank.scenario_weights must be real floating, "
            f"got dtype {sw.dtype}."
        )
    if sw.shape != (R,):
        raise ValueError(
            f"scenario_bank.scenario_weights must have shape [{R}], "
            f"got {list(sw.shape)}."
        )
    if not torch.isfinite(sw).all():
        raise ValueError(
            "scenario_bank.scenario_weights must be finite."
        )
    if (sw < 0).any():
        raise ValueError(
            "scenario_bank.scenario_weights must be non-negative."
        )
    if sw.sum().item() <= 0:
        raise ValueError(
            "scenario_bank.scenario_weights sum must be > 0."
        )
    if sw.requires_grad:
        raise ValueError(
            "scenario_bank.scenario_weights.requires_grad must be False."
        )


def _validate_normalized_scenario_weights(
    w: torch.Tensor,
    scores: torch.Tensor,
) -> None:
    """Validate normalized scenario weights match scores contract."""
    if not torch.is_tensor(w):
        raise TypeError(
            f"normalized_weights must return a torch.Tensor, "
            f"got {type(w).__name__}."
        )
    if torch.is_complex(w):
        raise TypeError("normalized_weights must not be complex.")
    if w.ndim != 1:
        raise ValueError(
            f"normalized_weights must be 1-D, got ndim={w.ndim}."
        )
    R = scores.shape[1]
    if w.shape != (R,):
        raise ValueError(
            f"normalized_weights must have shape [{R}], "
            f"got {list(w.shape)}."
        )
    if w.device != scores.device:
        raise ValueError(
            f"normalized_weights device ({w.device}) must match "
            f"scores device ({scores.device})."
        )
    if w.dtype != scores.dtype:
        raise TypeError(
            f"normalized_weights dtype ({w.dtype}) must match "
            f"scores dtype ({scores.dtype})."
        )
    if not torch.isfinite(w).all():
        raise ValueError("normalized_weights must be finite.")
    if (w < 0).any():
        raise ValueError("normalized_weights must be non-negative.")
    if w.requires_grad:
        raise ValueError(
            "normalized_weights.requires_grad must be False."
        )
    if not torch.allclose(w.sum(), torch.tensor(1.0, device=w.device,
                                                dtype=w.dtype)):
        raise ValueError(
            "normalized_weights must sum to 1."
        )
