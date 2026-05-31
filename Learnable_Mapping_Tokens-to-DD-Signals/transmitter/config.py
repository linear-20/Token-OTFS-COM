"""Transmitter configuration for OTFS DD signal generation.

All DD tensors use shape [B, M, N]. Masks use shape [1, M, N].
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class TransmitterConfig:
    """Transmitter hyperparameters for DD signal generation.

    Attributes:
        M: Number of delay bins in DD tensors [B, M, N].
        N: Number of Doppler bins in DD tensors [B, M, N].
        vocab_size: Number of tokens in the learned codebook vocabulary.
        pilot_delay: Delay-bin index of the embedded pilot in [0, M).
        pilot_doppler: Doppler-bin index of the embedded pilot in [0, N).
        pilot_guard_delay: Non-negative rectangular guard radius along delay.
        pilot_guard_doppler: Non-negative rectangular guard radius along Doppler.
        pilot_obs_delay_radius: Non-negative pilot observation radius along delay.
        pilot_obs_doppler_radius: Non-negative pilot observation radius along
            Doppler.
        data_power: Positive average power scaling for data DD bins.
        pilot_value_real: Real part of the embedded pilot scalar.
        pilot_value_imag: Imaginary part of the embedded pilot scalar.
        max_channel_delay: Maximum expected channel delay spread (non-negative).
        max_channel_doppler: Maximum expected channel Doppler spread (non-negative).
        complex_dtype: Torch complex dtype: "complex64" or "complex128".
    """

    M: int
    N: int
    vocab_size: int
    pilot_delay: int
    pilot_doppler: int
    pilot_guard_delay: int
    pilot_guard_doppler: int
    pilot_obs_delay_radius: int
    pilot_obs_doppler_radius: int
    data_power: float = 1.0
    pilot_value_real: float = 1.0
    pilot_value_imag: float = 0.0
    max_channel_delay: int = 0
    max_channel_doppler: int = 0
    complex_dtype: str = "complex64"

    def __post_init__(self) -> None:
        _validate_transmitter_config(self)

    @property
    def pilot_value(self) -> complex:
        """Complex pilot scalar."""
        return complex(self.pilot_value_real, self.pilot_value_imag)

    @property
    def torch_complex_dtype(self) -> torch.dtype:
        """Torch complex dtype corresponding to complex_dtype."""
        return {"complex64": torch.complex64, "complex128": torch.complex128}[self.complex_dtype]

    @property
    def torch_real_dtype(self) -> torch.dtype:
        """Torch real dtype corresponding to complex_dtype."""
        return {"complex64": torch.float32, "complex128": torch.float64}[self.complex_dtype]


def _validate_transmitter_config(cfg: TransmitterConfig) -> None:
    """Validate all fields and constraints of TransmitterConfig.

    Args:
        cfg: TransmitterConfig instance to validate.

    Raises:
        ValueError: If any field or constraint is violated, with a clear message
            naming the failing field or condition.
    """
    if isinstance(cfg.M, bool) or not isinstance(cfg.M, int) or cfg.M <= 0:
        raise ValueError(f"M must be a positive integer, got {cfg.M}.")
    if isinstance(cfg.N, bool) or not isinstance(cfg.N, int) or cfg.N <= 0:
        raise ValueError(f"N must be a positive integer, got {cfg.N}.")
    if (isinstance(cfg.vocab_size, bool)
            or not isinstance(cfg.vocab_size, int)
            or cfg.vocab_size <= 0):
        raise ValueError(f"vocab_size must be a positive integer, got {cfg.vocab_size}.")

    if (isinstance(cfg.pilot_delay, bool)
            or not isinstance(cfg.pilot_delay, int)
            or cfg.pilot_delay < 0
            or cfg.pilot_delay >= cfg.M):
        raise ValueError(
            f"pilot_delay must be an integer in [0, M={cfg.M}), got {cfg.pilot_delay}."
        )
    if (isinstance(cfg.pilot_doppler, bool)
            or not isinstance(cfg.pilot_doppler, int)
            or cfg.pilot_doppler < 0
            or cfg.pilot_doppler >= cfg.N):
        raise ValueError(
            f"pilot_doppler must be an integer in [0, N={cfg.N}), got {cfg.pilot_doppler}."
        )

    if (isinstance(cfg.data_power, bool)
            or not isinstance(cfg.data_power, (int, float))
            or not math.isfinite(cfg.data_power)
            or cfg.data_power <= 0):
        raise ValueError(
            f"data_power must be finite and positive, got {cfg.data_power}."
        )

    for name, value in {
        "pilot_value_real": cfg.pilot_value_real,
        "pilot_value_imag": cfg.pilot_value_imag,
    }.items():
        if (isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)):
            raise ValueError(
                f"{name} must be a finite real number, got {value}."
            )

    non_negative_ints = {
        "pilot_guard_delay": cfg.pilot_guard_delay,
        "pilot_guard_doppler": cfg.pilot_guard_doppler,
        "pilot_obs_delay_radius": cfg.pilot_obs_delay_radius,
        "pilot_obs_doppler_radius": cfg.pilot_obs_doppler_radius,
        "max_channel_delay": cfg.max_channel_delay,
        "max_channel_doppler": cfg.max_channel_doppler,
    }
    for name, value in non_negative_ints.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer, got {value}.")

    if (not isinstance(cfg.complex_dtype, str)
            or cfg.complex_dtype not in {"complex64", "complex128"}):
        raise ValueError(
            f"complex_dtype must be 'complex64' or 'complex128', got '{cfg.complex_dtype}'."
        )

    if cfg.pilot_guard_delay < cfg.max_channel_delay + cfg.pilot_obs_delay_radius:
        raise ValueError(
            f"pilot_guard_delay ({cfg.pilot_guard_delay}) must be >= "
            f"max_channel_delay ({cfg.max_channel_delay}) + "
            f"pilot_obs_delay_radius ({cfg.pilot_obs_delay_radius}) = "
            f"{cfg.max_channel_delay + cfg.pilot_obs_delay_radius}."
        )
    if cfg.pilot_guard_doppler < cfg.max_channel_doppler + cfg.pilot_obs_doppler_radius:
        raise ValueError(
            f"pilot_guard_doppler ({cfg.pilot_guard_doppler}) must be >= "
            f"max_channel_doppler ({cfg.max_channel_doppler}) + "
            f"pilot_obs_doppler_radius ({cfg.pilot_obs_doppler_radius}) = "
            f"{cfg.max_channel_doppler + cfg.pilot_obs_doppler_radius}."
        )

    max_extent_delay = cfg.max_channel_delay + cfg.pilot_obs_delay_radius
    max_extent_doppler = cfg.max_channel_doppler + cfg.pilot_obs_doppler_radius

    if cfg.pilot_delay - max_extent_delay < 0:
        raise ValueError(
            f"No-wrap boundary safety violated: pilot_delay ({cfg.pilot_delay}) - "
            f"(max_channel_delay ({cfg.max_channel_delay}) + "
            f"pilot_obs_delay_radius ({cfg.pilot_obs_delay_radius})) = "
            f"{cfg.pilot_delay - max_extent_delay} < 0."
        )
    if cfg.pilot_delay + max_extent_delay >= cfg.M:
        raise ValueError(
            f"No-wrap boundary safety violated: pilot_delay ({cfg.pilot_delay}) + "
            f"(max_channel_delay ({cfg.max_channel_delay}) + "
            f"pilot_obs_delay_radius ({cfg.pilot_obs_delay_radius})) = "
            f"{cfg.pilot_delay + max_extent_delay} >= M ({cfg.M})."
        )
    if cfg.pilot_doppler - max_extent_doppler < 0:
        raise ValueError(
            f"No-wrap boundary safety violated: pilot_doppler ({cfg.pilot_doppler}) - "
            f"(max_channel_doppler ({cfg.max_channel_doppler}) + "
            f"pilot_obs_doppler_radius ({cfg.pilot_obs_doppler_radius})) = "
            f"{cfg.pilot_doppler - max_extent_doppler} < 0."
        )
    if cfg.pilot_doppler + max_extent_doppler >= cfg.N:
        raise ValueError(
            f"No-wrap boundary safety violated: pilot_doppler ({cfg.pilot_doppler}) + "
            f"(max_channel_doppler ({cfg.max_channel_doppler}) + "
            f"pilot_obs_doppler_radius ({cfg.pilot_obs_doppler_radius})) = "
            f"{cfg.pilot_doppler + max_extent_doppler} >= N ({cfg.N})."
        )
