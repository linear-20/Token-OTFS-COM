"""Tail-risk helpers for TX-only sparse-DD surrogate experiments.

These helpers change risk aggregation, not transmitter architecture.
The physical penalty remains:

    penalty[p, r] = relu(target_margin - d[p, r]) ** 2

For a uniform sampled pair-scenario bank, empirical_tail_cvar_margin_loss
returns the mean of the largest ceil(tail_fraction * P * R) penalties.
This is the empirical CVaR of the penalty at confidence 1 - tail_fraction.
"""

from __future__ import annotations

import math

import torch


def empirical_tail_cvar_margin_loss(
    scores: torch.Tensor,
    *,
    target_margin: float,
    tail_fraction: float,
) -> torch.Tensor:
    """Return empirical CVaR of squared margin deficits.

    Args:
        scores: Real finite separation scores [P, R].
        target_margin: Shared global physical margin gamma >= 0.
        tail_fraction: Fraction in (0, 1] retained from the worst penalties.

    Returns:
        Scalar tensor preserving the autograd graph.
    """
    _validate_scores(scores)
    _validate_target_margin(target_margin)
    _validate_tail_fraction(tail_fraction)
    margin = torch.as_tensor(
        target_margin, device=scores.device, dtype=scores.dtype,
    )
    penalties = torch.relu(margin - scores).pow(2).reshape(-1)
    tail_count = max(1, math.ceil(tail_fraction * penalties.numel()))
    return torch.topk(
        penalties, k=tail_count, largest=True, sorted=False,
    ).values.mean()


def summarize_tail_separation_scores(
    scores: torch.Tensor,
    *,
    target_margin: float,
    tail_fraction: float,
) -> dict:
    """Return auditable tail diagnostics for separation scores [P, R]."""
    _validate_scores(scores)
    _validate_target_margin(target_margin)
    _validate_tail_fraction(tail_fraction)
    flat = scores.reshape(-1)
    return {
        "separation_p001": torch.quantile(
            flat, 0.001, interpolation="linear",
        ).item(),
        "separation_p005": torch.quantile(
            flat, 0.005, interpolation="linear",
        ).item(),
        "separation_p05": torch.quantile(
            flat, 0.05, interpolation="linear",
        ).item(),
        "tail_cvar_fraction": tail_fraction,
        "eval_tail_cvar_l_core": empirical_tail_cvar_margin_loss(
            scores,
            target_margin=target_margin,
            tail_fraction=tail_fraction,
        ).item(),
    }


def _validate_scores(scores: torch.Tensor) -> None:
    if not torch.is_tensor(scores):
        raise TypeError(
            f"scores must be a torch.Tensor, got {type(scores).__name__}."
        )
    if torch.is_complex(scores) or not torch.is_floating_point(scores):
        raise TypeError(
            f"scores must be a real floating tensor, got dtype {scores.dtype}."
        )
    if scores.ndim != 2:
        raise ValueError(
            f"scores must have shape [P, R], got ndim={scores.ndim}."
        )
    if scores.shape[0] <= 0 or scores.shape[1] <= 0:
        raise ValueError(
            f"scores dimensions must be > 0, got {list(scores.shape)}."
        )
    if not torch.isfinite(scores).all():
        raise ValueError("scores must contain only finite values.")


def _validate_target_margin(value: float) -> None:
    if (isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0):
        raise ValueError(
            f"target_margin must be finite and >= 0, got {value}."
        )


def _validate_tail_fraction(value: float) -> None:
    if (isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
            or value > 1):
        raise ValueError(
            f"tail_fraction must be finite and in (0, 1], got {value}."
        )
