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
    device = x_dd.device
    shifts = channel.path_shifts.to(device=device)  # [B, K, 2]
    gains = channel.path_gains  # [B, K], same device/dtype as x_dd

    # torch.roll(x, shifts=(d, v)) maps output[m, n] to
    # input[(m - d) % M, (n - v) % N]. Build all source indices at once
    # so the hot path contains no Python loops and no per-path scalar
    # extraction synchronizations on CUDA.
    delay_out = torch.arange(M, device=device).reshape(1, 1, M, 1)
    doppler_out = torch.arange(N, device=device).reshape(1, 1, 1, N)
    delay_src = (
        delay_out - shifts[:, :, 0].reshape(B, K, 1, 1)
    ).remainder(M)
    doppler_src = (
        doppler_out - shifts[:, :, 1].reshape(B, K, 1, 1)
    ).remainder(N)
    flat_src = (delay_src * N + doppler_src).expand(B, K, M, N)

    x_flat = x_dd.reshape(B, 1, M * N).expand(B, K, M * N)
    shifted = torch.gather(
        x_flat, dim=2, index=flat_src.reshape(B, K, M * N),
    ).reshape(B, K, M, N)

    effective_gains = gains
    if channel.path_active_mask is not None:
        effective_gains = effective_gains * channel.path_active_mask.to(
            device=device, dtype=gains.real.dtype,
        )

    # Multiplication by a zero bool gate preserves a zero-gradient graph
    # for all-inactive scenarios without a special-case branch.
    return (
        shifted * effective_gains.reshape(B, K, 1, 1)
    ).sum(dim=1)


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
