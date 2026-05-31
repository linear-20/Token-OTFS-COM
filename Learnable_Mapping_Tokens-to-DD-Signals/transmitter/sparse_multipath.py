"""Receiver-aligned on-grid sparse multipath DD operator.

Public API:
    SparseMultipathDDChannel  -- frozen dataclass for sparse path metadata
    apply_sparse_multipath_dd_operator(x_dd, channel) -> complex [B, M, N]

This is a receiver-aligned on-grid sparse DD circular-shift superposition
operator. It is NOT full OTFS twisted convolution and does NOT model
fractional Doppler or off-grid leakage.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .dd_shifts import dd_circular_shift


@dataclass(frozen=True)
class SparseMultipathDDChannel:
    """Sparse on-grid multipath DD channel metadata and gains.

    Models:
        H_theta x = sum_k active_k * h_k * Pi_{delta_k}(x)

    where Pi is the receiver-aligned integer DD circular shift.
    Duplicate shifts use coherent complex addition of their gains.
    The active mask is a strict bool gate only -- not a soft confidence,
    not a learnable reliability, not a path gain.

    Attributes:
        path_shifts: Long tensor [B, K, 2] of (delay_shift, doppler_shift)
            pairs. Signed and wrap-around shifts are legal.
        path_gains: Complex tensor [B, K] of path gains. May require grad.
        path_active_mask: Optional bool tensor [B, K]; True = active,
            False = padded/inactive. If None, all paths are active.
    """

    path_shifts: torch.Tensor
    path_gains: torch.Tensor
    path_active_mask: torch.Tensor | None = None

    def __post_init__(self) -> None:
        # -- path_shifts validation -------------------------------------------
        if not torch.is_tensor(self.path_shifts):
            raise TypeError(
                f"path_shifts must be a torch.Tensor, "
                f"got {type(self.path_shifts).__name__}."
            )
        if self.path_shifts.dtype != torch.long:
            raise TypeError(
                f"path_shifts dtype must be torch.long, "
                f"got {self.path_shifts.dtype}."
            )
        if self.path_shifts.ndim != 3 or self.path_shifts.shape[2] != 2:
            raise ValueError(
                f"path_shifts must have shape [B, K, 2], "
                f"got {list(self.path_shifts.shape)}."
            )
        B_sh, K_sh, _ = self.path_shifts.shape
        if B_sh <= 0:
            raise ValueError(f"path_shifts B must be > 0, got B={B_sh}.")
        if K_sh <= 0:
            raise ValueError(f"path_shifts K must be > 0, got K={K_sh}.")

        # -- path_gains validation --------------------------------------------
        if not torch.is_tensor(self.path_gains):
            raise TypeError(
                f"path_gains must be a torch.Tensor, "
                f"got {type(self.path_gains).__name__}."
            )
        if not torch.is_complex(self.path_gains):
            raise TypeError(
                f"path_gains must be a complex tensor, "
                f"got dtype {self.path_gains.dtype}."
            )
        if self.path_gains.shape != (B_sh, K_sh):
            raise ValueError(
                f"path_gains must have shape [{B_sh}, {K_sh}], "
                f"got {list(self.path_gains.shape)}."
            )
        if not torch.isfinite(self.path_gains).all():
            raise ValueError("path_gains must contain only finite values.")

        # -- path_active_mask validation ---------------------------------------
        if self.path_active_mask is not None:
            mask = self.path_active_mask
            if not torch.is_tensor(mask):
                raise TypeError(
                    f"path_active_mask must be a torch.Tensor, "
                    f"got {type(mask).__name__}."
                )
            if mask.dtype != torch.bool:
                raise TypeError(
                    f"path_active_mask dtype must be torch.bool, "
                    f"got {mask.dtype}."
                )
            if mask.shape != (B_sh, K_sh):
                raise ValueError(
                    f"path_active_mask must have shape [{B_sh}, {K_sh}], "
                    f"got {list(mask.shape)}."
                )


def apply_sparse_multipath_dd_operator(
    x_dd: torch.Tensor,
    channel: SparseMultipathDDChannel,
) -> torch.Tensor:
    """Apply sparse on-grid multipath DD circular-shift superposition.

    For each batch element b:
        y_dd[b] = sum_{k} active[b,k] * gain[b,k] * Pi_{shift[b,k]}(x_dd[b])

    where Pi is the receiver-aligned integer DD circular shift and
    active[b,k] is 1 if path_active_mask is None or True, else 0.

    This is a receiver-aligned on-grid sparse DD shift superposition
    approximation. It is NOT full OTFS twisted convolution and does
    NOT model fractional Doppler or off-grid leakage.

    Complexity: O(B * K * M * N).  No dense MN x MN matrix.

    Args:
        x_dd: Complex tensor [B, M, N]. All dims > 0, finite.
        channel: SparseMultipathDDChannel with path_shifts [B, K, 2],
            path_gains complex [B, K] (same device/dtype as x_dd).

    Returns:
        Complex tensor [B, M, N], same device and dtype as x_dd.

    Raises:
        TypeError/ValueError: If inputs fail validation.
    """
    _validate_operator_inputs(x_dd, channel)

    B, M, N = x_dd.shape
    K = channel.path_shifts.shape[1]
    shifts = channel.path_shifts  # [B, K, 2], may be CPU or CUDA
    gains = channel.path_gains    # [B, K], must match x_dd device/dtype
    mask = channel.path_active_mask  # [B, K] bool or None

    y_list = []
    for b in range(B):
        # Keep a zero-gradient graph anchor even when every padded path is
        # inactive. This preserves backward() semantics without changing y.
        y_b = x_dd[b] * 0.0 + gains[b].sum() * 0.0
        for k in range(K):
            # Active gate.
            active = True if mask is None else bool(mask[b, k].item())
            if not active:
                continue

            g = gains[b, k]  # complex scalar with autograd
            d = int(shifts[b, k, 0].item())
            v = int(shifts[b, k, 1].item())

            # Shift x_dd[b] (preserves autograd on x_dd).
            shifted = dd_circular_shift(
                x_dd[b:b + 1], d, v,
            )[0]  # [M, N]

            y_b = y_b + g * shifted

        y_list.append(y_b)

    return torch.stack(y_list, dim=0)  # [B, M, N]


def _validate_operator_inputs(
    x_dd: torch.Tensor,
    channel: SparseMultipathDDChannel,
) -> None:
    """Validate x_dd and channel for the multipath operator."""
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
            f"x_dd must have 3 dimensions [B, M, N], "
            f"got ndim={x_dd.ndim}."
        )
    B_x, M_x, N_x = x_dd.shape
    if B_x <= 0:
        raise ValueError(f"x_dd B must be > 0, got B={B_x}.")
    if M_x <= 0:
        raise ValueError(f"x_dd M must be > 0, got M={M_x}.")
    if N_x <= 0:
        raise ValueError(f"x_dd N must be > 0, got N={N_x}.")
    if not torch.isfinite(x_dd).all():
        raise ValueError("x_dd must contain only finite values.")

    if not isinstance(channel, SparseMultipathDDChannel):
        raise TypeError(
            f"channel must be a SparseMultipathDDChannel, "
            f"got {type(channel).__name__}."
        )

    # Batch size match.
    ch_B = channel.path_shifts.shape[0]
    if ch_B != B_x:
        raise ValueError(
            f"channel batch size ({ch_B}) must match "
            f"x_dd batch size ({B_x})."
        )

    # path_gains must be on same device and dtype as x_dd.
    if channel.path_gains.device != x_dd.device:
        raise ValueError(
            f"channel.path_gains device ({channel.path_gains.device}) "
            f"must match x_dd device ({x_dd.device})."
        )
    if channel.path_gains.dtype != x_dd.dtype:
        raise TypeError(
            f"channel.path_gains dtype ({channel.path_gains.dtype}) "
            f"must match x_dd dtype ({x_dd.dtype})."
        )
