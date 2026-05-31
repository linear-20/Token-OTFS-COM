"""Delay-Doppler position encodings for token classification."""

from __future__ import annotations

import math

import torch


def build_dd_position_encoding(
    M: int,
    N: int,
    num_channels: int,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Build DD position encoding with shape [1, C, M, N].

    Args:
        M: Number of delay bins in DD tensors with shape [B, M, N].
        N: Number of Doppler bins in DD tensors with shape [B, M, N].
        num_channels: Positive output channel count C.
        device: Optional torch device for the returned tensor.
        dtype: Optional floating dtype for the returned tensor.

    Returns:
        Float position encoding with shape [1, num_channels, M, N]. The base
        channels contain normalized delay/Doppler coordinates and sinusoidal
        components. If num_channels exceeds the base size, the base channels
        are repeated; if smaller, they are truncated.
    """

    if not isinstance(M, int) or M <= 0:
        raise ValueError("M must be a positive integer.")
    if not isinstance(N, int) or N <= 0:
        raise ValueError("N must be a positive integer.")
    if not isinstance(num_channels, int) or num_channels <= 0:
        raise ValueError("num_channels must be a positive integer.")
    dtype = torch.float32 if dtype is None else dtype
    if not dtype.is_floating_point:
        raise TypeError("dtype must be floating point.")

    delay = _normalized_axis(M, device=device, dtype=dtype).reshape(M, 1).expand(M, N)
    doppler = _normalized_axis(N, device=device, dtype=dtype).reshape(1, N).expand(M, N)
    base = torch.stack(
        (
            delay,
            doppler,
            torch.sin(math.pi * delay),
            torch.cos(math.pi * delay),
            torch.sin(math.pi * doppler),
            torch.cos(math.pi * doppler),
        ),
        dim=0,
    )
    repeats = math.ceil(num_channels / base.shape[0])
    channels = base.repeat(repeats, 1, 1)[:num_channels]
    return channels.unsqueeze(0)


def _normalized_axis(length: int, device: torch.device | None, dtype: torch.dtype) -> torch.Tensor:
    if length == 1:
        return torch.zeros(length, device=device, dtype=dtype)
    return torch.linspace(-1.0, 1.0, steps=length, device=device, dtype=dtype)
