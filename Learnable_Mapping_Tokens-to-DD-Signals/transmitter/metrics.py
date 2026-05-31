"""DD-domain / DD-frame power metrics and time-domain waveform metrics.

Public API:
    data_codeword_power            - per-token mean |codeword|^2 over data support
    pilot_power                    - scalar |pilot_value|^2
    dd_frame_power                 - batch-wise mean |x_dd|^2
    summarize_dd_transmitter_metrics - unified DD metrics dict for model
    time_signal_power              - batch-wise mean |s|^2 over time dim
    peak_to_average_power_ratio    - batch-wise PAPR after OTFS modulation
    summarize_time_transmitter_metrics - unified time metrics dict
"""

from __future__ import annotations

import torch


def data_codeword_power(
    codeword_book: torch.Tensor,
    data_mask: torch.Tensor,
) -> torch.Tensor:
    """Per-token mean |codeword|^2 over the active data support region.

    Args:
        codeword_book: Complex tensor [V, M, N].
        data_mask: Boolean or real binary (0/1) tensor [1, M, N] or [V, M, N].

    Returns:
        Real tensor [V] of per-token average active-region power.

    Raises:
        TypeError: If codeword_book is not a complex tensor, or data_mask is
            complex.
        ValueError: If shapes are invalid, data_mask is fractional or
            non-finite, or any token has an empty active region.
    """
    _validate_complex_tensor_3d(codeword_book, "codeword_book")
    target_device = codeword_book.device
    target_dtype = _complex_to_real_dtype(codeword_book)
    mask = _validate_binary_support_mask(
        data_mask, codeword_book.shape, "data_mask",
        device=target_device, dtype=target_dtype,
    )

    active = mask > 0
    active_count = active.sum(dim=(-2, -1))
    if (active_count <= 0).any():
        raise ValueError(
            "data_mask has no active bins for at least one token. "
            "Each token must have at least one active data-region bin."
        )

    projected = codeword_book * mask
    power = projected.abs().pow(2).sum(dim=(-2, -1)) / active_count.clamp_min(1)
    return power.to(dtype=target_dtype)


def pilot_power(
    pilot_value: complex | torch.Tensor,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Scalar pilot power |pilot_value|^2, shape [].

    Args:
        pilot_value: Python complex/real or scalar torch.Tensor (ndim==0).
        device: Target device. Defaults to pilot_value.device for tensors,
            or CPU for Python scalars.
        dtype: Target floating dtype. Must be a real floating type.

    Returns:
        Scalar tensor [] with value |pilot_value|^2.

    Raises:
        TypeError: If pilot_value is a non-scalar tensor or dtype is not
            floating.
    """
    if not dtype.is_floating_point:
        raise TypeError(f"dtype must be a floating dtype, got {dtype}.")

    if torch.is_tensor(pilot_value):
        if pilot_value.ndim != 0:
            raise ValueError(
                f"pilot_value tensor must be scalar (ndim==0), "
                f"got ndim={pilot_value.ndim}."
            )
        abs_sq = pilot_value.abs() ** 2
        dev = device if device is not None else pilot_value.device
        return abs_sq.to(device=dev, dtype=dtype)

    if not isinstance(pilot_value, (int, float, complex)):
        raise TypeError(
            f"pilot_value must be a number or scalar tensor, "
            f"got {type(pilot_value).__name__}."
        )
    dev = device if device is not None else torch.device("cpu")
    abs_sq = abs(pilot_value) ** 2
    return torch.as_tensor(abs_sq, device=dev, dtype=dtype)


def dd_frame_power(x_dd: torch.Tensor) -> torch.Tensor:
    """Batch-wise mean DD frame power, shape [B].

    Args:
        x_dd: Complex tensor [B, M, N].

    Returns:
        Real tensor [B] with mean(|x_dd|^2) over spatial dimensions.

    Raises:
        TypeError: If x_dd is not a complex tensor.
        ValueError: If shape is invalid or batch size is zero.
    """
    if not torch.is_tensor(x_dd):
        raise TypeError(f"x_dd must be a torch.Tensor, got {type(x_dd).__name__}.")
    if not torch.is_complex(x_dd):
        raise TypeError(f"x_dd must be a complex tensor, got dtype {x_dd.dtype}.")
    if x_dd.ndim != 3:
        raise ValueError(f"x_dd must have 3 dimensions [B, M, N], got ndim={x_dd.ndim}.")
    B, M, N_sp = x_dd.shape
    if B <= 0:
        raise ValueError(f"x_dd batch size must be positive, got B={B}.")
    if M <= 0 or N_sp <= 0:
        raise ValueError(f"x_dd spatial dimensions [M, N] must be > 0, got [{M}, {N_sp}].")

    return x_dd.abs().pow(2).mean(dim=(-2, -1)).to(dtype=_complex_to_real_dtype(x_dd))


def summarize_dd_transmitter_metrics(
    codeword_book: torch.Tensor,
    x_dd: torch.Tensor,
    data_mask: torch.Tensor,
    pilot_value: complex | torch.Tensor,
) -> dict:
    """Unified metrics dict for LearnableTokenDDTransmitter.forward().

    Args:
        codeword_book: Complex tensor [V, M, N].
        x_dd: Complex tensor [B, M, N].
        data_mask: Boolean or real binary tensor [1, M, N] or [V, M, N].
        pilot_value: Python complex/real or scalar tensor.

    Returns:
        Dict with keys:
            "data_codeword_power"  - real [V]
            "pilot_power"          - scalar []
            "total_dd_frame_power" - real [B]
    """
    real_dtype = _complex_to_real_dtype(x_dd)
    return {
        "data_codeword_power": data_codeword_power(codeword_book, data_mask),
        "pilot_power": pilot_power(
            pilot_value, device=x_dd.device, dtype=real_dtype,
        ),
        "total_dd_frame_power": dd_frame_power(x_dd),
    }


def _complex_to_real_dtype(t: torch.Tensor) -> torch.dtype:
    """Map a complex tensor's dtype to its real component dtype."""
    return {torch.complex64: torch.float32, torch.complex128: torch.float64}[t.dtype]


def _validate_complex_tensor_3d(
    t: torch.Tensor,
    name: str,
) -> None:
    """Raise if t is not a complex 3-D tensor with all dimensions > 0."""
    if not torch.is_tensor(t):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(t).__name__}.")
    if not torch.is_complex(t):
        raise TypeError(f"{name} must be a complex tensor, got dtype {t.dtype}.")
    if t.ndim != 3:
        raise ValueError(f"{name} must have 3 dimensions, got ndim={t.ndim}.")
    if t.shape[0] <= 0 or t.shape[1] <= 0 or t.shape[2] <= 0:
        raise ValueError(
            f"{name} must have all dimensions > 0, got {list(t.shape)}."
        )


def _validate_binary_support_mask(
    mask: torch.Tensor,
    ref_shape: torch.Size,
    name: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Validate and convert a binary support mask for DD operations.

    Accepts bool or real-valued 0/1 masks. Returns a real mask on the
    specified device and dtype, with 0/1 entries only.

    Args:
        mask: Input mask tensor (bool or real-valued binary).
        ref_shape: Expected [V, M, N] shape of the codeword book. The mask
            must have shape [1, M, N] or [V, M, N].
        name: Name for error messages.
        device: Target device for the returned mask.
        dtype: Target real dtype for the returned mask.

    Returns:
        Real mask tensor with values 0.0 or 1.0 on the requested device/dtype.

    Raises:
        TypeError: If mask is complex or not a tensor.
        ValueError: If shape, finiteness, or non-binary values are detected.
    """
    if not torch.is_tensor(mask):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(mask).__name__}.")
    if torch.is_complex(mask):
        raise TypeError(f"{name} must not be a complex tensor.")

    if mask.ndim != 3:
        raise ValueError(f"{name} must have 3 dimensions, got ndim={mask.ndim}.")
    V, M, N = ref_shape
    if mask.shape not in ((1, M, N), (V, M, N)):
        raise ValueError(
            f"{name} must have shape [1, {M}, {N}] or [{V}, {M}, {N}], "
            f"got {list(mask.shape)}."
        )

    # Preserve source dtype so conversion cannot round fractional values to 0/1.
    mask_check = mask.detach().to(device=torch.device("cpu"), copy=True)
    if not torch.isfinite(mask_check).all():
        raise ValueError(f"{name} must contain only finite values.")

    invalid = (mask_check != 0) & (mask_check != 1)
    if invalid.any():
        raise ValueError(
            f"{name} must be a hard binary support mask (only 0 or 1 values). "
            "Fractional values are not allowed."
        )
    return mask.to(device=device, dtype=dtype)


# ---- time-domain waveform metrics (computed after OTFS modulation) -----------


def _validate_complex_time_signal(time_signal: torch.Tensor) -> None:
    """Validate time_signal is a finite complex tensor [B, T] with B > 0, T > 0.

    Args:
        time_signal: Complex tensor to validate.

    Raises:
        TypeError: If time_signal is not a torch.Tensor or is not complex.
        ValueError: If ndim != 2, B == 0, T == 0, or non-finite values found.
    """
    if not torch.is_tensor(time_signal):
        raise TypeError(
            f"time_signal must be a torch.Tensor, got {type(time_signal).__name__}."
        )
    if not torch.is_complex(time_signal):
        raise TypeError(
            f"time_signal must be a complex tensor, got dtype {time_signal.dtype}."
        )
    if time_signal.ndim != 2:
        raise ValueError(
            f"time_signal must have 2 dimensions [B, T], "
            f"got ndim={time_signal.ndim}."
        )
    B, T = time_signal.shape
    if B <= 0:
        raise ValueError(f"time_signal batch size B must be > 0, got B={B}.")
    if T <= 0:
        raise ValueError(f"time_signal time length T must be > 0, got T={T}.")
    if not torch.isfinite(time_signal).all():
        raise ValueError(
            "time_signal must contain only finite values "
            "(real and imag parts must both be finite)."
        )


def time_signal_power(time_signal: torch.Tensor) -> torch.Tensor:
    """Batch-wise mean power of a complex time-domain waveform.

    Computes P[b] = (1 / T) * sum_t |s[b,t]|^2 over the time dimension.
    The metric is computed on the actual time-domain waveform, which may
    include cyclic prefix samples.

    Args:
        time_signal: Complex tensor [B, T] with B > 0, T > 0.

    Returns:
        Real tensor [B] with dtype float32 (complex64 input) or float64
        (complex128 input), on the same device as time_signal.

    Raises:
        TypeError: If time_signal is not a complex torch.Tensor.
        ValueError: If shape or finiteness is invalid.
    """
    _validate_complex_time_signal(time_signal)
    real_dtype = _complex_to_real_dtype(time_signal)
    power = time_signal.abs().pow(2).mean(dim=1)
    return power.to(dtype=real_dtype)


def peak_to_average_power_ratio(time_signal: torch.Tensor) -> torch.Tensor:
    """Batch-wise Peak-to-Average Power Ratio (PAPR).

    Computes PAPR[b] = max_t |s[b,t]|^2 / mean_t |s[b,t]|^2.
    PAPR is computed on the actual time-domain waveform after OTFS
    modulation and includes cyclic prefix samples. It is NOT computed
    on DD-domain codewords or x_dd.

    A zero-power waveform has an undefined PAPR and raises ValueError.
    No epsilon floor is applied.

    Args:
        time_signal: Complex tensor [B, T] with B > 0, T > 0.

    Returns:
        Real tensor [B] with dtype float32 (complex64 input) or float64
        (complex128 input), on the same device as time_signal.

    Raises:
        TypeError: If time_signal is not a complex torch.Tensor.
        ValueError: If shape or finiteness is invalid, or any batch
            waveform has zero average power.
    """
    power = time_signal_power(time_signal)
    if bool((power <= 0).any().item()):
        raise ValueError(
            "peak_to_average_power_ratio is undefined for zero-power "
            "waveforms. At least one batch waveform has average power <= 0."
        )
    peak = time_signal.abs().pow(2).max(dim=1).values
    return peak / power.to(dtype=peak.dtype)


def summarize_time_transmitter_metrics(
    time_signal: torch.Tensor,
) -> dict:
    """Unified time-domain metrics dict for transmitter diagnostics.

    Args:
        time_signal: Complex tensor [B, T] with B > 0, T > 0.

    Returns:
        Dict with keys:
            "time_signal_power"  - real [B]
            "papr"               - real [B]
    """
    return {
        "time_signal_power": time_signal_power(time_signal),
        "papr": peak_to_average_power_ratio(time_signal),
    }
