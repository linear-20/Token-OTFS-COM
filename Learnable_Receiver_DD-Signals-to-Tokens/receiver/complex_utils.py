"""Complex tensor helpers for DD-domain receiver modules.

The receiver keeps complex tensors as torch complex with shape [B, M, N].
Conversion to real/imag channels is only used before CNN layers.
"""

from __future__ import annotations

import torch

from .config import ReceiverConfig


def validate_complex_dd(x: torch.Tensor, config: ReceiverConfig, name: str) -> None:
    """Validate a complex DD tensor with shape [B, M, N]."""

    if not torch.is_tensor(x):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if x.ndim != 3:
        raise ValueError(f"{name} must have shape [B, M, N]; got {list(x.shape)}.")
    if tuple(x.shape[1:]) != (config.M, config.N):
        raise ValueError(
            f"{name} must have spatial shape [{config.M}, {config.N}]; "
            f"got {list(x.shape[1:])}."
        )
    if not torch.is_complex(x):
        raise TypeError(f"{name} must be a complex tensor; got {x.dtype}.")


def validate_optional_complex_dd(
    x: torch.Tensor | None,
    config: ReceiverConfig,
    name: str,
) -> None:
    """Validate an optional complex DD tensor with shape [B, M, N]."""

    if x is not None:
        validate_complex_dd(x, config, name)


def validate_support_mask(mask: torch.Tensor, config: ReceiverConfig, name: str) -> None:
    """Validate a support mask with shape [B, M, N]."""

    if not torch.is_tensor(mask):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if mask.ndim != 3:
        raise ValueError(f"{name} must have shape [B, M, N]; got {list(mask.shape)}.")
    if tuple(mask.shape[1:]) != (config.M, config.N):
        raise ValueError(
            f"{name} must have spatial shape [{config.M}, {config.N}]; "
            f"got {list(mask.shape[1:])}."
        )


def complex_to_channels(x: torch.Tensor) -> torch.Tensor:
    """Convert complex [B, M, N] to real channels [B, 2, M, N]."""

    if not torch.is_complex(x):
        raise TypeError(f"x must be complex; got {x.dtype}.")
    return torch.stack((x.real, x.imag), dim=1)


def channels_to_complex(x: torch.Tensor) -> torch.Tensor:
    """Convert real channels [B, 2, M, N] to complex [B, M, N]."""

    if x.ndim != 4 or x.shape[1] != 2:
        raise ValueError(f"x must have shape [B, 2, M, N]; got {list(x.shape)}.")
    return torch.complex(x[:, 0], x[:, 1])


def mask_to_real(mask: torch.Tensor, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Convert support_mask [B, M, N] to real-valued [B, M, N]."""

    return mask.to(device=device, dtype=dtype)

