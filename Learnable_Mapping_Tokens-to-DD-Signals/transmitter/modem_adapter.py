"""OTFS modem adapter: bridge from DD frame to time-domain signal.

Public API:
    modulate_otfs_dd_frame(x_dd, modem) -> time_signal

The adapter validates the modem contract and x_dd, calls
modem.modulate(x_dd) without any clone/detach/device-conversion,
and then validates the modem output shape/device/dtype.
"""

from __future__ import annotations

import torch


def modulate_otfs_dd_frame(
    x_dd: torch.Tensor,
    modem: object,
) -> torch.Tensor:
    """Modulate a DD frame to a time-domain signal via an OTFS modem.

    Calls modem.modulate(x_dd) directly. The adapter does NOT clone,
    detach, cast, or move x_dd; the autograd graph is preserved through
    the modulation call so that gradients can flow back through x_dd.

    Args:
        x_dd: Complex tensor [B, M, N] with B > 0, M > 0, N > 0.
        modem: An object satisfying the OTFS modem protocol:
            - modem.modulate(x_dd) must be callable and return a complex
              tensor [B, N * (M + cp_len)].
            - modem.dd_shape must be a tuple[int, int] of (M, N) with
              positive ints (not bool).
            - modem.cp_len must be a non-negative int (not bool) and
              satisfy cp_len <= M.

    Returns:
        Complex time_signal [B, N * (M + cp_len)] with the same device
        and dtype as x_dd.

    Raises:
        TypeError: If x_dd is not a complex tensor, or modem contract
            types are violated.
        ValueError: If x_dd shape is invalid, modem contract values are
            invalid, or modem output fails validation.
    """
    _validate_x_dd(x_dd)
    _validate_modem_contract(modem, x_dd)

    # Call modem.modulate directly; do NOT clone/detach/to/cast x_dd.
    time_signal = modem.modulate(x_dd)

    _validate_modem_output(time_signal, x_dd, modem)
    return time_signal


# -- private validators --------------------------------------------------------


def _validate_x_dd(x_dd: torch.Tensor) -> None:
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


def _validate_modem_contract(
    modem: object,
    x_dd: torch.Tensor,
) -> None:
    """Validate the modem satisfies the OTFS modem protocol.

    Checks modulate, dd_shape, and cp_len.  Raises TypeError for type
    violations and ValueError for value violations.
    """
    if modem is None:
        raise TypeError("modem must not be None.")

    # -- modulate ---------------------------------------------------------------
    if not hasattr(modem, "modulate"):
        raise TypeError("modem must have a 'modulate' attribute.")
    if not callable(modem.modulate):
        raise TypeError("modem.modulate must be callable.")

    # -- dd_shape ---------------------------------------------------------------
    if not hasattr(modem, "dd_shape"):
        raise TypeError("modem must have a 'dd_shape' attribute.")
    dd_shape = modem.dd_shape

    if not isinstance(dd_shape, tuple):
        raise TypeError(
            f"modem.dd_shape must be a tuple, got {type(dd_shape).__name__}."
        )
    if len(dd_shape) != 2:
        raise ValueError(
            f"modem.dd_shape must have length 2, got len={len(dd_shape)}."
        )
    if any(isinstance(x, bool) for x in dd_shape):
        raise TypeError(
            f"modem.dd_shape elements must be int, not bool. "
            f"Got dd_shape={dd_shape}."
        )
    if not all(isinstance(x, int) and x > 0 for x in dd_shape):
        raise ValueError(
            f"modem.dd_shape must be a tuple of two positive ints, "
            f"got {dd_shape}."
        )

    M_modem, N_modem = dd_shape
    B, M_xdd, N_xdd = x_dd.shape
    if M_modem != M_xdd or N_modem != N_xdd:
        raise ValueError(
            f"modem.dd_shape ({M_modem}, {N_modem}) must match "
            f"x_dd spatial shape ({M_xdd}, {N_xdd})."
        )

    # -- cp_len -----------------------------------------------------------------
    if not hasattr(modem, "cp_len"):
        raise TypeError("modem must have a 'cp_len' attribute.")
    cp_len = modem.cp_len

    if isinstance(cp_len, bool):
        raise TypeError(
            f"modem.cp_len must be int, not bool. Got cp_len={cp_len}."
        )
    if not isinstance(cp_len, int):
        raise TypeError(
            f"modem.cp_len must be int, got {type(cp_len).__name__}."
        )
    if cp_len < 0:
        raise ValueError(
            f"modem.cp_len must be >= 0, got cp_len={cp_len}."
        )
    if cp_len > M_modem:
        raise ValueError(
            f"modem.cp_len ({cp_len}) must be <= M ({M_modem})."
        )


def _validate_modem_output(
    time_signal: torch.Tensor,
    x_dd: torch.Tensor,
    modem: object,
) -> None:
    """Validate modem output matches expected shape, device, and dtype."""
    if not torch.is_tensor(time_signal):
        raise TypeError(
            "modem.modulate must return a torch.Tensor, "
            f"got {type(time_signal).__name__}."
        )
    if not torch.is_complex(time_signal):
        raise TypeError(
            "modem.modulate must return a complex tensor, "
            f"got dtype {time_signal.dtype}."
        )

    B_x, M_x, N_x = x_dd.shape
    cp = modem.cp_len
    expected_len = N_x * (M_x + cp)

    if time_signal.ndim != 2:
        raise ValueError(
            f"modem.modulate output must have 2 dimensions [B, time], "
            f"got ndim={time_signal.ndim}."
        )
    B_ts, T_ts = time_signal.shape
    if B_ts != B_x:
        raise ValueError(
            f"modem.modulate output batch size ({B_ts}) must match "
            f"x_dd batch size ({B_x})."
        )
    if T_ts != expected_len:
        raise ValueError(
            f"modem.modulate output time length ({T_ts}) must equal "
            f"N * (M + cp_len) = {N_x} * ({M_x} + {cp}) = {expected_len}."
        )

    if time_signal.device != x_dd.device:
        raise ValueError(
            f"modem.modulate output device ({time_signal.device}) must "
            f"match x_dd device ({x_dd.device})."
        )
    if time_signal.dtype != x_dd.dtype:
        raise TypeError(
            f"modem.modulate output dtype ({time_signal.dtype}) must "
            f"match x_dd dtype ({x_dd.dtype})."
        )
