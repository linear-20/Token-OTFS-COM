"""Receiver-aligned integer DD circular shift primitives.

Public API:
    dd_circular_shift(x_dd, delay_shift, doppler_shift) -> [B, M, N]
    dd_circular_shift_bank(x_dd, shifts [S, 2]) -> [B, S, M, N]

These are integer on-grid circular DD shift operators. They do NOT
construct dense matrices, apply data-mask projection, fractional
leakage, or sparse multipath composition. Shift metadata are discrete
integers and are not part of the autograd graph.
"""

from __future__ import annotations

import torch


def dd_circular_shift(
    x_dd: torch.Tensor,
    delay_shift: int,
    doppler_shift: int,
) -> torch.Tensor:
    """Apply a receiver-aligned integer circular DD shift.

    Uses torch.roll along dims (-2, -1), matching the receiver's
    on-grid sparse path primitive:

        torch.roll(x_dd[batch_idx], shifts=(delay, doppler), dims=(0, 1))

    No data-mask projection is applied after shifting. Energy may
    move into the pilot/guard region.

    Args:
        x_dd: Complex tensor [B, M, N] (B can also represent a token
            dimension V, e.g. codeword_book [V, M, N]).
        delay_shift: Integer shift along the delay axis (dim -2,
            size M). Negative and wrap-around shifts are supported
            via torch.roll circular semantics.
        doppler_shift: Integer shift along the Doppler axis (dim -1,
            size N). Negative and wrap-around shifts are supported.

    Returns:
        Complex tensor [B, M, N] with the same device and dtype as x_dd.
        The autograd graph of x_dd is preserved.

    Raises:
        TypeError: If x_dd is not a complex tensor, or delay_shift/
            doppler_shift are not ints (including bool rejection).
        ValueError: If x_dd shape is not [B, M, N] with all dims > 0.
    """
    _validate_complex_dd_collection(x_dd)
    _validate_integer_shift(delay_shift, "delay_shift")
    _validate_integer_shift(doppler_shift, "doppler_shift")

    return torch.roll(
        x_dd,
        shifts=(delay_shift, doppler_shift),
        dims=(-2, -1),
    )


def dd_circular_shift_bank(
    x_dd: torch.Tensor,
    shifts: torch.Tensor,
) -> torch.Tensor:
    """Apply a bank of integer DD circular shifts.

    Each row of shifts specifies one (delay_shift, doppler_shift) pair.
    The result stacks the shifted copies along a new dimension S.

    Args:
        x_dd: Complex tensor [B, M, N].
        shifts: Long tensor [S, 2]. Each row is [delay_shift, doppler_shift]
            in that order. Shifts are discrete metadata only and do not
            need to be on the same device as x_dd.

    Returns:
        Complex tensor [B, S, M, N] where output[:, s] corresponds to
        shifts[s]. Same device and dtype as x_dd.

    Raises:
        TypeError: If x_dd is not a complex tensor, or shifts is not a
            torch.long tensor.
        ValueError: If shapes or dimensions are invalid.
    """
    _validate_complex_dd_collection(x_dd)
    _validate_shift_bank(shifts)

    S = shifts.shape[0]
    shifted_list = [
        dd_circular_shift(x_dd, int(shifts[s, 0].item()), int(shifts[s, 1].item()))
        for s in range(S)
    ]
    return torch.stack(shifted_list, dim=1)


# -- private validators --------------------------------------------------------


def _validate_complex_dd_collection(x_dd: torch.Tensor) -> None:
    """Validate x_dd is a complex [B, M, N] tensor with all dims > 0."""
    if not torch.is_tensor(x_dd):
        raise TypeError(
            f"x_dd must be a torch.Tensor, got {type(x_dd).__name__}."
        )
    if not torch.is_complex(x_dd):
        raise TypeError(
            f"x_dd must be a complex tensor, got dtype {x_dd.dtype}."
        )
    if x_dd.ndim != 3:
        raise ValueError(
            f"x_dd must have 3 dimensions [B, M, N], got ndim={x_dd.ndim}."
        )
    B, M, N = x_dd.shape
    if B <= 0:
        raise ValueError(f"x_dd batch size B must be > 0, got B={B}.")
    if M <= 0:
        raise ValueError(f"x_dd M (delay bins) must be > 0, got M={M}.")
    if N <= 0:
        raise ValueError(f"x_dd N (Doppler bins) must be > 0, got N={N}.")


def _validate_integer_shift(value: int, name: str) -> None:
    """Validate value is a Python int (reject bool)."""
    if isinstance(value, bool):
        raise TypeError(
            f"{name} must be int, not bool. Got {name}={value}."
        )
    if not isinstance(value, int):
        raise TypeError(
            f"{name} must be int, got {type(value).__name__}."
        )


def _validate_shift_bank(shifts: torch.Tensor) -> None:
    """Validate shifts is a long tensor [S, 2] with S > 0."""
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
            f"shifts must have 2 dimensions [S, 2], got ndim={shifts.ndim}."
        )
    if shifts.shape[1] != 2:
        raise ValueError(
            f"shifts last dim must be 2, got {shifts.shape[1]}."
        )
    if shifts.shape[0] <= 0:
        raise ValueError(
            f"shifts S must be > 0, got S={shifts.shape[0]}."
        )
