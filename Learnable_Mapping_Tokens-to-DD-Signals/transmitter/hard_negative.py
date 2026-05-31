"""Candidate-pool cross-token hard-negative mining for orbit ablations.

This pre-17C2 utility is not part of the default transmitter algorithm.

Public API:
    HardNegativeTokenPairs  -- frozen dataclass for selected hard negatives
    mine_cross_token_hard_negatives(codeword_book, candidate_pairs, shift_set,
        evidence_mask, num_hard_pairs) -> HardNegativeTokenPairs

This is a discrete selection-only utility. It is NOT a differentiable
loss. Training must recompute cross_token_orbit_confusion_loss with the
returned token_pairs to obtain gradients.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .regularizers import cross_token_orbit_confusion_scores
from .shaping import SparseShiftSet


@dataclass(frozen=True)
class HardNegativeTokenPairs:
    """Selection metadata for top-K hard-negative cross-token pairs.

    All tensors are detached and non-differentiable. Duplicates are
    legal and preserved. This is selection-only metadata; the training
    loop should recompute cross_token_orbit_confusion_loss with the
    returned token_pairs to obtain an autograd graph.

    Attributes:
        token_pairs: Long tensor [K, 2] of selected (u, v) pairs.
        source_indices: Long tensor [K] recording the row index in the
            candidate pool each selected pair came from.
        peak_scores: Real floating tensor [K] = max over shifts of
            the orbit correlation for each selected pair.
        orbit_scores: Real floating tensor [K, S] of full orbit
            correlation scores for the selected pairs.
    """

    token_pairs: torch.Tensor
    source_indices: torch.Tensor
    peak_scores: torch.Tensor
    orbit_scores: torch.Tensor

    def __post_init__(self) -> None:
        # -- type checks -------------------------------------------------------
        for name in ("token_pairs", "source_indices", "peak_scores",
                     "orbit_scores"):
            t = getattr(self, name)
            if not torch.is_tensor(t):
                raise TypeError(
                    f"{name} must be a torch.Tensor, "
                    f"got {type(t).__name__}."
                )

        # dtypes
        if self.token_pairs.dtype != torch.long:
            raise TypeError(
                f"token_pairs dtype must be torch.long, "
                f"got {self.token_pairs.dtype}."
            )
        if self.source_indices.dtype != torch.long:
            raise TypeError(
                f"source_indices dtype must be torch.long, "
                f"got {self.source_indices.dtype}."
            )
        if not self.peak_scores.dtype.is_floating_point:
            raise TypeError(
                f"peak_scores must be real floating, "
                f"got {self.peak_scores.dtype}."
            )
        if not self.orbit_scores.dtype.is_floating_point:
            raise TypeError(
                f"orbit_scores must be real floating, "
                f"got {self.orbit_scores.dtype}."
            )
        if self.peak_scores.dtype != self.orbit_scores.dtype:
            raise TypeError(
                "peak_scores and orbit_scores must have the same dtype, "
                f"got {self.peak_scores.dtype} and {self.orbit_scores.dtype}."
            )

        # -- shape checks ------------------------------------------------------
        if self.token_pairs.ndim != 2 or self.token_pairs.shape[1] != 2:
            raise ValueError(
                f"token_pairs must have shape [K, 2], "
                f"got {list(self.token_pairs.shape)}."
            )
        K = self.token_pairs.shape[0]
        if K <= 0:
            raise ValueError(f"K must be > 0, got K={K}.")

        if self.source_indices.shape != (K,):
            raise ValueError(
                f"source_indices must have shape [K={K}], "
                f"got {list(self.source_indices.shape)}."
            )
        if self.peak_scores.shape != (K,):
            raise ValueError(
                f"peak_scores must have shape [K={K}], "
                f"got {list(self.peak_scores.shape)}."
            )

        if self.orbit_scores.ndim != 2:
            raise ValueError(
                f"orbit_scores must have 2 dims [K, S], "
                f"got ndim={self.orbit_scores.ndim}."
            )
        if self.orbit_scores.shape[0] != K:
            raise ValueError(
                f"orbit_scores dim 0 must be K={K}, "
                f"got {self.orbit_scores.shape[0]}."
            )
        S = self.orbit_scores.shape[1]
        if S <= 0:
            raise ValueError(f"orbit_scores S must be > 0, got S={S}.")

        # -- device consistency -------------------------------------------------
        dev = self.token_pairs.device
        for name in ("source_indices", "peak_scores", "orbit_scores"):
            t = getattr(self, name)
            if t.device != dev:
                raise ValueError(
                    f"All tensors must be on the same device. "
                    f"token_pairs: {dev}, {name}: {t.device}."
                )

        # -- value checks -------------------------------------------------------
        if (self.token_pairs < 0).any():
            raise ValueError("token_pairs indices must be >= 0.")
        if (self.token_pairs[:, 0] == self.token_pairs[:, 1]).any():
            raise ValueError(
                "token_pairs must not contain self-pairs (u == v)."
            )
        if (self.source_indices < 0).any():
            raise ValueError("source_indices must be >= 0.")
        if not torch.isfinite(self.peak_scores).all():
            raise ValueError("peak_scores must be finite.")
        if not torch.isfinite(self.orbit_scores).all():
            raise ValueError("orbit_scores must be finite.")

        # peak_scores must match orbit_scores.max(dim=1).values
        computed_peak = self.orbit_scores.max(dim=1).values
        if not torch.allclose(self.peak_scores, computed_peak):
            raise ValueError(
                "peak_scores must equal orbit_scores.max(dim=1).values."
            )

        # -- requires_grad must be False ----------------------------------------
        for name in ("token_pairs", "source_indices", "peak_scores",
                     "orbit_scores"):
            t = getattr(self, name)
            if t.requires_grad:
                raise ValueError(
                    f"{name}.requires_grad must be False. "
                    "HardNegativeTokenPairs is selection metadata, "
                    "not a differentiable loss."
                )


def mine_cross_token_hard_negatives(
    codeword_book: torch.Tensor,
    candidate_pairs: torch.Tensor,
    shift_set: SparseShiftSet,
    evidence_mask: torch.Tensor,
    num_hard_pairs: int,
) -> HardNegativeTokenPairs:
    """Select top-K cross-token pairs by peak orbit confusion.

    Scores all candidate pairs via cross_token_orbit_confusion_scores
    under torch.no_grad(), then selects the K pairs with the highest
    peak correlation over shifts. Output is detached selection metadata.

    Hard mining is intentionally biased toward the high-confusion tail
    and is NOT an unbiased Monte Carlo estimator.

    The returned selection is detached and requires no gradient.
    Training must recompute cross_token_orbit_confusion_loss with the
    selected token_pairs to obtain a differentiable loss. If evidence_mask
    is pair-specific [P_pool, M, N], select its source_indices rows before
    recomputing the loss.

    Complexity: O(P_pool * S * M * N). No V^2 matrix is constructed.

    Args:
        codeword_book: Complex tensor [V, M, N] of physical normalized
            codewords.
        candidate_pairs: Long tensor [P_pool, 2] of ordered non-self
            candidate token pairs.
        shift_set: SparseShiftSet.
        evidence_mask: hard binary support mask [1, M, N] or [P_pool, M, N].
        num_hard_pairs: K > 0, K <= P_pool (int, not bool).

    Returns:
        HardNegativeTokenPairs with:
            token_pairs    long [K, 2]
            source_indices long [K]
            peak_scores    real [K]
            orbit_scores   real [K, S]
        All detached and non-differentiable.

    Raises:
        TypeError/ValueError: From validation, scoring, or topk.
    """
    # Fail-fast validation before expensive scoring.
    _validate_hard_mining_args(candidate_pairs, shift_set, num_hard_pairs)

    with torch.no_grad():
        orbit_scores_full = cross_token_orbit_confusion_scores(
            codeword_book,
            candidate_pairs,
            shift_set,
            evidence_mask,
        )  # [P_pool, S]

        pair_peak = orbit_scores_full.max(dim=1).values  # [P_pool]

        peak_scores, source_indices = torch.topk(
            pair_peak,
            k=num_hard_pairs,
            largest=True,
            sorted=True,
        )

    # Select the winning rows and move to codeword_book device.
    out_device = codeword_book.device
    source_on_dev = source_indices.to(device=out_device)
    pairs_on_dev = candidate_pairs.to(device=out_device)

    selected_pairs = pairs_on_dev[source_on_dev]            # [K, 2]
    selected_orbit = orbit_scores_full[source_indices]      # [K, S]
    selected_peak = peak_scores.to(device=out_device)       # [K]

    return HardNegativeTokenPairs(
        token_pairs=selected_pairs.detach().to(dtype=torch.long),
        source_indices=source_on_dev.detach().to(dtype=torch.long),
        peak_scores=selected_peak.detach(),
        orbit_scores=selected_orbit.detach(),
    )


# -- private validators --------------------------------------------------------


def _validate_hard_mining_args(
    candidate_pairs: torch.Tensor,
    shift_set: SparseShiftSet,
    num_hard_pairs: int,
) -> None:
    """Fail-fast validation of hard-mining arguments."""
    if not torch.is_tensor(candidate_pairs):
        raise TypeError(
            f"candidate_pairs must be a torch.Tensor, "
            f"got {type(candidate_pairs).__name__}."
        )
    if candidate_pairs.dtype != torch.long:
        raise TypeError(
            f"candidate_pairs dtype must be torch.long, "
            f"got {candidate_pairs.dtype}."
        )
    if candidate_pairs.ndim != 2 or candidate_pairs.shape[1] != 2:
        raise ValueError(
            f"candidate_pairs must have shape [P_pool, 2], "
            f"got {list(candidate_pairs.shape)}."
        )
    P_pool = candidate_pairs.shape[0]
    if P_pool <= 0:
        raise ValueError(
            f"candidate_pairs P_pool must be > 0, got P_pool={P_pool}."
        )

    if isinstance(num_hard_pairs, bool) or not isinstance(num_hard_pairs, int):
        raise TypeError(
            f"num_hard_pairs must be int, "
            f"got {type(num_hard_pairs).__name__}."
        )
    if num_hard_pairs <= 0:
        raise ValueError(
            f"num_hard_pairs must be > 0, got {num_hard_pairs}."
        )
    if num_hard_pairs > P_pool:
        raise ValueError(
            f"num_hard_pairs ({num_hard_pairs}) must be <= "
            f"P_pool ({P_pool})."
        )

    if not isinstance(shift_set, SparseShiftSet):
        raise TypeError(
            f"shift_set must be a SparseShiftSet, "
            f"got {type(shift_set).__name__}."
        )
