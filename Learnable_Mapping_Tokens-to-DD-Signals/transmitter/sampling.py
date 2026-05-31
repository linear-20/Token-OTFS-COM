"""Scalable Monte Carlo pair sampling and optional orbit-shift sampling.

Public API:
    SampledSparseShiftSet  -- frozen dataclass for sampled shift metadata
    sample_uniform_cross_token_pairs(vocab_size, num_pairs, ...) -> long [P, 2]
    sample_weighted_sparse_shift_set(source_shift_set, num_shift_samples, ...)
        -> SampledSparseShiftSet

Uniform token-pair sampling is used by the core Step 17C2 objective. Weighted
shift-set sampling is retained only for pre-17C2 orbit ablations.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .shaping import SparseShiftSet


@dataclass(frozen=True)
class SampledSparseShiftSet:
    """Result of weighted shift set sampling.

    Attributes:
        shift_set: SparseShiftSet with .shifts long [S_sample, 2] and
            .weights strictly None (uniform). This prevents downstream
            regularizers from multiplying q a second time.
        source_indices: Long tensor [S_sample] recording which row of
            the source shift set each sampled shift came from.
            Duplicates are preserved.
    """

    shift_set: SparseShiftSet
    source_indices: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.shift_set, SparseShiftSet):
            raise TypeError(
                f"shift_set must be a SparseShiftSet, "
                f"got {type(self.shift_set).__name__}."
            )
        ss = self.shift_set
        if not torch.is_tensor(ss.shifts):
            raise TypeError(
                "shift_set.shifts must be a torch.Tensor, "
                f"got {type(ss.shifts).__name__}."
            )
        if ss.shifts.dtype != torch.long:
            raise TypeError(
                f"shift_set.shifts dtype must be torch.long, "
                f"got {ss.shifts.dtype}."
            )
        if ss.shifts.ndim != 2 or ss.shifts.shape[1] != 2:
            raise ValueError(
                f"shift_set.shifts must have shape [S_sample, 2], "
                f"got {list(ss.shifts.shape)}."
            )
        if ss.shifts.shape[0] <= 0:
            raise ValueError(
                f"shift_set.shifts S_sample must be > 0, "
                f"got {ss.shifts.shape[0]}."
            )
        if ss.weights is not None:
            raise ValueError(
                "shift_set.weights must be None for sampled shifts. "
                "Downstream regularizers use uniform weights to avoid "
                "double-counting the sampling distribution q."
            )

        si = self.source_indices
        if not torch.is_tensor(si):
            raise TypeError(
                f"source_indices must be a torch.Tensor, "
                f"got {type(si).__name__}."
            )
        if si.dtype != torch.long:
            raise TypeError(
                f"source_indices dtype must be torch.long, "
                f"got {si.dtype}."
            )
        if si.shape != (ss.shifts.shape[0],):
            raise ValueError(
                f"source_indices must have shape [S_sample={ss.shifts.shape[0]}], "
                f"got {list(si.shape)}."
            )
        if si.device != ss.shifts.device:
            raise ValueError(
                f"source_indices device ({si.device}) must match "
                f"shift_set.shifts device ({ss.shifts.device})."
            )
        if (si < 0).any():
            raise ValueError("source_indices must be >= 0.")


def sample_uniform_cross_token_pairs(
    vocab_size: int,
    num_pairs: int,
    *,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Uniformly sample ordered non-self token pairs with replacement.

    Draws from the V*(V-1) possible ordered non-self pairs without
    constructing the full pair table.  Uses an O(P) rejection-free
    mapping from flat index to (u, v).

    Args:
        vocab_size: V >= 2 (int, not bool).
        num_pairs: P > 0 (int, not bool).
        generator: Optional CPU torch.Generator for reproducibility.
            GPU generators are rejected.
        device: Output device (default CPU).

    Returns:
        Long tensor [P, 2] of (u, v) pairs with u != v, indices in
        [0, V). Duplicates are legal and preserved.
    """
    _validate_pair_sampling_args(vocab_size, num_pairs)
    if generator is not None:
        _validate_cpu_generator(generator)

    V = vocab_size
    P = num_pairs
    total = V * (V - 1)

    # Sample flat indices uniformly.
    flat = torch.randint(
        0, total, (P,), generator=generator, dtype=torch.long,
    )

    # O(P) rejection-free mapping: flat -> (u, v).
    u = flat // (V - 1)
    offset = flat % (V - 1)
    v = offset + (offset >= u).long()

    pairs = torch.stack([u, v], dim=-1)  # [P, 2]

    if device is not None:
        pairs = pairs.to(device=torch.device(device))

    return pairs


def sample_weighted_sparse_shift_set(
    source_shift_set: SparseShiftSet,
    num_shift_samples: int,
    *,
    generator: torch.Generator | None = None,
) -> SampledSparseShiftSet:
    """Sample shifts with replacement according to source weights q[s].

    Returns a SampledSparseShiftSet whose .shift_set.weights is None
    (uniform).  Downstream regularizers use uniform averaging, which
    combined with q-weighted sampling yields a Monte Carlo estimator
    of the q-weighted objective.  This avoids multiplying q twice.

    Args:
        source_shift_set: SparseShiftSet with .shifts long [S, 2].
            If .weights is None, uniform sampling is used.
        num_shift_samples: S_sample > 0 (int, not bool).
        generator: Optional CPU torch.Generator.

    Returns:
        SampledSparseShiftSet with:
            .shift_set.shifts   long [S_sample, 2]
            .shift_set.weights  None
            .source_indices     long [S_sample]
        Duplicates and zero shifts are preserved.
    """
    _validate_weighted_shift_sampling_args(
        source_shift_set, num_shift_samples,
    )
    if generator is not None:
        _validate_cpu_generator(generator)

    S_src = source_shift_set.shifts.shape[0]

    # Obtain normalized weights on CPU float64 for multinomial.
    q_cpu = source_shift_set.normalized_weights(
        device=torch.device("cpu"), dtype=torch.float64,
    )
    if not torch.isfinite(q_cpu).all():
        raise ValueError("normalized source shift weights must be finite.")
    if (q_cpu < 0).any():
        raise ValueError("normalized source shift weights must be non-negative.")
    if q_cpu.sum().item() <= 0:
        raise ValueError("normalized source shift weights sum must be > 0.")

    sampled_indices_cpu = torch.multinomial(
        q_cpu,
        num_shift_samples,
        replacement=True,
        generator=generator,
    )  # long [S_sample] on CPU

    # Move indices to source shifts device for indexing.
    src_device = source_shift_set.shifts.device
    sampled_indices = sampled_indices_cpu.to(device=src_device)

    sampled_shifts = source_shift_set.shifts[sampled_indices]  # [S_sample, 2]

    return SampledSparseShiftSet(
        shift_set=SparseShiftSet(
            shifts=sampled_shifts,
            weights=None,
            name=f"sampled_{source_shift_set.name}",
        ),
        source_indices=sampled_indices,
    )


# -- private validators --------------------------------------------------------


def _validate_pair_sampling_args(vocab_size: int, num_pairs: int) -> None:
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int):
        raise TypeError(
            f"vocab_size must be int, got {type(vocab_size).__name__}."
        )
    if vocab_size < 2:
        raise ValueError(
            f"vocab_size must be >= 2 for non-self pairs, got {vocab_size}."
        )
    if isinstance(num_pairs, bool) or not isinstance(num_pairs, int):
        raise TypeError(
            f"num_pairs must be int, got {type(num_pairs).__name__}."
        )
    if num_pairs <= 0:
        raise ValueError(
            f"num_pairs must be > 0, got {num_pairs}."
        )


def _validate_cpu_generator(generator: torch.Generator) -> None:
    if not isinstance(generator, torch.Generator):
        raise TypeError(
            f"generator must be a torch.Generator, "
            f"got {type(generator).__name__}."
        )
    if generator.device.type != "cpu":
        raise ValueError(
            f"generator must be a CPU torch.Generator, "
            f"got device {generator.device}."
        )


def _validate_weighted_shift_sampling_args(
    source_shift_set: SparseShiftSet,
    num_shift_samples: int,
) -> None:
    if not isinstance(source_shift_set, SparseShiftSet):
        raise TypeError(
            f"source_shift_set must be a SparseShiftSet, "
            f"got {type(source_shift_set).__name__}."
        )
    ss = source_shift_set
    if not torch.is_tensor(ss.shifts):
        raise TypeError(
            "source_shift_set.shifts must be a torch.Tensor, "
            f"got {type(ss.shifts).__name__}."
        )
    if ss.shifts.dtype != torch.long:
        raise TypeError(
            f"source_shift_set.shifts dtype must be torch.long, "
            f"got {ss.shifts.dtype}."
        )
    if ss.shifts.ndim != 2 or ss.shifts.shape[1] != 2:
        raise ValueError(
            f"source_shift_set.shifts must have shape [S, 2], "
            f"got {list(ss.shifts.shape)}."
        )
    S = ss.shifts.shape[0]
    if S <= 0:
        raise ValueError(
            f"source_shift_set.shifts S must be > 0, got S={S}."
        )

    if ss.weights is not None:
        w = ss.weights
        if not torch.is_tensor(w):
            raise TypeError(
                f"source_shift_set.weights must be a torch.Tensor, "
                f"got {type(w).__name__}."
            )
        if not torch.is_floating_point(w):
            raise TypeError(
                f"source_shift_set.weights must be real floating, "
                f"got dtype {w.dtype}."
            )
        if w.shape != (S,):
            raise ValueError(
                f"source_shift_set.weights must have shape [{S}], "
                f"got {list(w.shape)}."
            )
        if not torch.isfinite(w).all():
            raise ValueError(
                "source_shift_set.weights must be finite."
            )
        if (w < 0).any():
            raise ValueError(
                "source_shift_set.weights must be non-negative."
            )
        if w.sum().item() <= 0:
            raise ValueError(
                "source_shift_set.weights sum must be > 0."
            )
        if w.requires_grad:
            raise ValueError(
                "source_shift_set.weights.requires_grad must be False. "
                "Shift weights are fixed physical statistics; they must "
                "not be learnable parameters."
            )

    if isinstance(num_shift_samples, bool) or not isinstance(
        num_shift_samples, int,
    ):
        raise TypeError(
            f"num_shift_samples must be int, "
            f"got {type(num_shift_samples).__name__}."
        )
    if num_shift_samples <= 0:
        raise ValueError(
            f"num_shift_samples must be > 0, got {num_shift_samples}."
        )
