"""Deterministic OTFS modulation/demodulation transform chain.

Public API:
    normalized_mse(reference, estimate) -> float
    max_abs_error(reference, estimate) -> float
    class OTFSModem
        OTFSModem.isfft(x_dd: [B, M, N]) -> x_tf: [B, M, N]
        OTFSModem.sfft(x_tf: [B, M, N]) -> x_dd: [B, M, N]
        OTFSModem.modulate(x_dd: [B, M, N]) -> time_signal: [B, N*(M+cp_len)]
        OTFSModem.demodulate(time_signal: [B, N*(M+cp_len)] | [N*(M+cp_len)]) -> y_dd: [B, M, N]
        OTFSModem.roundtrip(x_dd: [B, M, N]) -> (time_signal, recovered_dd)

Transform chain:
    X_DD -> ISFFT -> X_TF -> OFDM modulation -> time waveform
         -> OFDM demodulation -> Y_TF -> SFFT -> Y_DD

There is no wireless channel, AWGN, token decoding, bit conversion, QAM, or
training. The transform definitions are chosen to be self-consistent inverses.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch


def _torch_load(path: Path) -> Any:
    """Load a PyTorch object on CPU, compatible with old and new torch versions."""

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


def _shape_list(tensor: torch.Tensor) -> list[int]:
    return list(tensor.shape)


def normalized_mse(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    """Normalized mean squared error between reference and estimate.

    Computes ||estimate - reference||^2 / ||reference||^2.

    Args:
        reference: Complex tensor of any shape.
        estimate: Complex tensor with the same shape as reference.

    Returns:
        Non-negative float scalar.

    Raises:
        TypeError: If either argument is not a torch.Tensor.
        ValueError: If shapes do not match, or reference has zero total power.
    """
    if not torch.is_tensor(reference) or not torch.is_tensor(estimate):
        raise TypeError("reference and estimate must be torch.Tensor objects.")
    if reference.shape != estimate.shape:
        raise ValueError(
            "reference and estimate must have the same shape; "
            f"got {_shape_list(reference)} and {_shape_list(estimate)}."
        )
    denom = reference.abs().pow(2).sum()
    if denom.item() == 0:
        raise ValueError("normalized_mse is undefined for a zero-power reference.")
    error = (estimate - reference).abs().pow(2).sum()
    return (error / denom).item()


def max_abs_error(reference: torch.Tensor, estimate: torch.Tensor) -> float:
    """Maximum absolute element-wise error between reference and estimate.

    Computes max(|estimate - reference|).

    Args:
        reference: Tensor of any shape. Must be non-empty.
        estimate: Tensor with the same shape as reference.

    Returns:
        Non-negative float scalar.

    Raises:
        TypeError: If either argument is not a torch.Tensor.
        ValueError: If shapes do not match, or either tensor is empty.
    """
    if not torch.is_tensor(reference) or not torch.is_tensor(estimate):
        raise TypeError("reference and estimate must be torch.Tensor objects.")
    if reference.numel() == 0:
        raise ValueError("max_abs_error is undefined for empty tensors.")
    if reference.shape != estimate.shape:
        raise ValueError(
            "reference and estimate must have the same shape; "
            f"got {_shape_list(reference)} and {_shape_list(estimate)}."
        )
    return (estimate - reference).abs().max().item()


class OTFSModem:
    """Deterministic OTFS modulator/demodulator.

    Implements the transform chain:
        X_DD [B, M, N] -> ISFFT -> X_TF [B, M, N]
            -> OFDM modulation (IFFT over subcarriers, transpose, optional CP)
            -> time_signal [B, N*(M+cp_len)]

    and its strict inverse:
        time_signal -> OFDM demodulation (remove CP, transpose, FFT)
            -> Y_TF [B, M, N] -> SFFT -> Y_DD [B, M, N]

    ISFFT: ifft over Doppler (dim=2), then fft over delay (dim=1), norm="ortho".
    SFFT:  inverse of ISFFT (ifft over dim=1, fft over dim=2).

    Args:
        dd_shape: (M, N) tuple of positive integers. M is delay bins
            (subcarriers), N is Doppler bins (time slots).
        cp_len: Non-negative cyclic prefix length in samples. Must satisfy
            0 <= cp_len <= M.
        device: Torch device string or None for CPU.
        dtype: Complex torch dtype; must be torch.complex64 or torch.complex128.

    Raises:
        TypeError: If dd_shape elements or cp_len are bool, or dtype is not
            a complex dtype.
        ValueError: If dd_shape elements are not positive ints, cp_len is
            negative or exceeds M, or dd_shape is not length-2.
    """

    def __init__(
        self,
        dd_shape: tuple[int, int] = (32, 32),
        cp_len: int = 0,
        device: str | None = None,
        dtype: torch.dtype = torch.complex64,
    ):
        if isinstance(dd_shape, tuple) and any(
            isinstance(x, bool) for x in dd_shape
        ):
            raise TypeError(
                "dd_shape elements must be int, not bool. "
                f"Got dd_shape={dd_shape}."
            )
        if (
            not isinstance(dd_shape, tuple)
            or len(dd_shape) != 2
            or not all(isinstance(x, int) and x > 0 for x in dd_shape)
        ):
            raise ValueError("dd_shape must be a tuple of two positive integers.")
        if isinstance(cp_len, bool):
            raise TypeError(
                f"cp_len must be int, not bool. Got cp_len={cp_len}."
            )
        if not isinstance(cp_len, int) or cp_len < 0:
            raise ValueError("cp_len must be a non-negative integer.")
        if cp_len > dd_shape[0]:
            raise ValueError(
                f"cp_len must be <= M ({dd_shape[0]}) so a full cyclic prefix exists; "
                f"got cp_len={cp_len}."
            )
        if dtype not in (torch.complex64, torch.complex128):
            raise TypeError(
                f"dtype must be torch.complex64 or torch.complex128, got {dtype}."
            )

        self.dd_shape = dd_shape
        self.M = dd_shape[0]
        self.N = dd_shape[1]
        self.cp_len = cp_len
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype

    def _validate_dd_tensor(self, x_dd: torch.Tensor, name: str = "x_dd") -> None:
        """Validate that x_dd is a complex [B, M, N] tensor with B > 0.

        Args:
            x_dd: Complex tensor to validate.
            name: Name used in error messages.

        Raises:
            TypeError: If x_dd is not a torch.Tensor or is not complex.
            ValueError: If ndim != 3, spatial shape does not match dd_shape,
                or batch size B == 0.
        """
        if not torch.is_tensor(x_dd):
            raise TypeError(f"{name} must be a torch.Tensor.")
        if x_dd.ndim != 3:
            raise ValueError(
                f"{name} must have shape [B, M, N]; got shape {_shape_list(x_dd)}."
            )
        if x_dd.shape[0] == 0:
            raise ValueError(f"{name} batch size must be > 0.")
        if tuple(x_dd.shape[1:]) != self.dd_shape:
            raise ValueError(
                f"{name} spatial shape must match dd_shape {self.dd_shape}; "
                f"got {tuple(x_dd.shape[1:])}."
            )
        if not torch.is_complex(x_dd):
            raise TypeError(f"{name} must be a complex tensor; got {x_dd.dtype}.")

    def isfft(self, x_dd: torch.Tensor) -> torch.Tensor:
        """Inverse Symplectic FFT: X_DD [B, M, N] -> X_TF [B, M, N].

        Applies ifft over Doppler (dim=2), then fft over delay (dim=1),
        both with norm="ortho". The input is cast to self.device/self.dtype.

        Args:
            x_dd: Complex tensor [B, M, N] with B > 0.

        Returns:
            Complex tensor [B, M, N] on self.device with self.dtype.
        """
        self._validate_dd_tensor(x_dd, "x_dd")
        x_dd = x_dd.to(device=self.device, dtype=self.dtype)
        return torch.fft.fft(
            torch.fft.ifft(x_dd, dim=2, norm="ortho"),
            dim=1,
            norm="ortho",
        )

    def sfft(self, x_tf: torch.Tensor) -> torch.Tensor:
        """Symplectic FFT: X_TF [B, M, N] -> X_DD [B, M, N].

        Applies ifft over delay (dim=1), then fft over Doppler (dim=2),
        both with norm="ortho". This is the inverse of ISFFT.

        Args:
            x_tf: Complex tensor [B, M, N] with B > 0.

        Returns:
            Complex tensor [B, M, N] on self.device with self.dtype.
        """
        self._validate_dd_tensor(x_tf, "x_tf")
        x_tf = x_tf.to(device=self.device, dtype=self.dtype)
        return torch.fft.fft(
            torch.fft.ifft(x_tf, dim=1, norm="ortho"),
            dim=2,
            norm="ortho",
        )

    def modulate(self, x_dd: torch.Tensor) -> torch.Tensor:
        """OTFS modulation: X_DD [B, M, N] -> time_signal [B, N*(M+cp_len)].

        Steps:
            1. ISFFT:        X_DD [B, M, N] -> X_TF [B, M, N]
            2. OFDM IFFT:    ifft over subcarriers (dim=1), norm="ortho"
            3. Transpose:    [B, M, N] -> [B, N, M]
            4. CP (optional): prepend last cp_len samples along dim=-1
            5. Flatten:      [B, N, M+cp_len] -> [B, N*(M+cp_len)]

        Args:
            x_dd: Complex tensor [B, M, N] with B > 0.

        Returns:
            Complex time_signal [B, N*(M+cp_len)] on self.device with
            self.dtype.
        """
        self._validate_dd_tensor(x_dd, "x_dd")
        x_tf = self.isfft(x_dd)

        # OFDM/Heisenberg step: IFFT over subcarriers for each time slot.
        x_symbols = torch.fft.ifft(x_tf, dim=1, norm="ortho")
        x_symbols = x_symbols.transpose(1, 2)  # [B, N, M]

        if self.cp_len > 0:
            cp = x_symbols[:, :, -self.cp_len :]
            x_symbols = torch.cat([cp, x_symbols], dim=-1)

        return x_symbols.reshape(x_symbols.shape[0], -1)

    def demodulate(self, time_signal: torch.Tensor) -> torch.Tensor:
        """OTFS demodulation: time_signal -> Y_DD [B, M, N].

        Strict inverse of modulate():
            1. Unflatten:    [B, N*(M+cp_len)] -> [B, N, M+cp_len]
            2. Remove CP:    strip first cp_len samples along dim=-1
            3. Transpose:    [B, N, M] -> [B, M, N]
            4. OFDM FFT:     fft over subcarriers (dim=1), norm="ortho"
            5. SFFT:         Y_TF [B, M, N] -> Y_DD [B, M, N]

        Accepts 1-D input [time] and treats it as batch size 1.

        Args:
            time_signal: Complex tensor [B, N*(M+cp_len)] with B > 0,
                or 1-D tensor [N*(M+cp_len)].

        Returns:
            Complex Y_DD [B, M, N] on self.device with self.dtype.

        Raises:
            TypeError: If time_signal is not a tensor or is not complex.
            ValueError: If ndim is not 1 or 2, flattened length does not
                match N*(M+cp_len), or batch size B == 0.
        """
        if not torch.is_tensor(time_signal):
            raise TypeError("time_signal must be a torch.Tensor.")
        if time_signal.ndim == 1:
            time_signal = time_signal.unsqueeze(0)
        elif time_signal.ndim != 2:
            raise ValueError(
                "time_signal must have shape [B, time] or [time]; "
                f"got shape {_shape_list(time_signal)}."
            )
        if time_signal.shape[0] == 0:
            raise ValueError("time_signal batch size must be > 0.")
        if not torch.is_complex(time_signal):
            raise TypeError(f"time_signal must be a complex tensor; got {time_signal.dtype}.")

        expected_len = self.N * (self.M + self.cp_len)
        if time_signal.shape[1] != expected_len:
            raise ValueError(
                f"time_signal length must be N * (M + cp_len) = {expected_len}; "
                f"got {time_signal.shape[1]}."
            )

        time_signal = time_signal.to(device=self.device, dtype=self.dtype)
        y_symbols = time_signal.reshape(time_signal.shape[0], self.N, self.M + self.cp_len)
        if self.cp_len > 0:
            y_symbols = y_symbols[:, :, self.cp_len :]

        y_symbols = y_symbols.transpose(1, 2)  # [B, M, N]
        y_tf = torch.fft.fft(y_symbols, dim=1, norm="ortho")
        return self.sfft(y_tf)

    def roundtrip(self, x_dd: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Full modulation-demodulation roundtrip.

        modulate(x_dd) -> time_signal, then demodulate(time_signal) -> recovered_dd.
        The recovered Y_DD should match X_DD up to numerical tolerance.

        Args:
            x_dd: Complex tensor [B, M, N] with B > 0.

        Returns:
            Tuple of (time_signal [B, N*(M+cp_len)], recovered_dd [B, M, N]).
        """
        time_signal = self.modulate(x_dd)
        recovered_dd = self.demodulate(time_signal)
        return time_signal, recovered_dd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a minimal OTFS modulation/demodulation numerical loop.",
    )
    parser.add_argument("--input-dd", type=Path, required=True)
    parser.add_argument("--output-time", type=Path, required=True)
    parser.add_argument("--output-dd", type=Path, required=True)
    parser.add_argument("--cp-len", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = _torch_load(args.input_dd)
    if not isinstance(payload, dict):
        raise ValueError(f"{args.input_dd} must contain a dict.")
    if "dd_grid" not in payload:
        raise KeyError(f"{args.input_dd} is missing required key: 'dd_grid'.")

    dd_grid = payload["dd_grid"]
    if not torch.is_tensor(dd_grid):
        raise TypeError("payload['dd_grid'] must be a torch.Tensor.")
    if dd_grid.ndim != 3:
        raise ValueError(
            "payload['dd_grid'] must have shape [B, M, N]; "
            f"got {_shape_list(dd_grid)}."
        )
    if not torch.is_complex(dd_grid):
        raise TypeError(f"payload['dd_grid'] must be complex; got {dd_grid.dtype}.")

    if "dd_shape" in payload:
        dd_shape = tuple(payload["dd_shape"])
    else:
        dd_shape = tuple(dd_grid.shape[-2:])

    modem = OTFSModem(
        dd_shape=dd_shape,
        cp_len=args.cp_len,
        device=args.device,
        dtype=dd_grid.dtype,
    )
    time_signal, recovered_dd = modem.roundtrip(dd_grid)
    nmse = normalized_mse(dd_grid.to(recovered_dd.device), recovered_dd)
    max_err = max_abs_error(dd_grid.to(recovered_dd.device), recovered_dd)

    input_power = dd_grid.abs().pow(2).mean().item()
    time_power = time_signal.abs().pow(2).mean().item()
    recovered_power = recovered_dd.abs().pow(2).mean().item()

    args.output_time.parent.mkdir(parents=True, exist_ok=True)
    args.output_dd.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "time_signal": time_signal.cpu(),
            "dd_shape": dd_shape,
            "cp_len": args.cp_len,
            "input_dd_shape": tuple(dd_grid.shape),
        },
        args.output_time,
    )
    torch.save(
        {
            "dd_grid": recovered_dd.cpu(),
            "source_dd_grid": dd_grid.cpu(),
            "dd_shape": dd_shape,
            "cp_len": args.cp_len,
            "normalized_mse": nmse,
            "max_abs_error": max_err,
        },
        args.output_dd,
    )

    print(f"input DD shape: {_shape_list(dd_grid)}")
    print(f"DD dtype: {dd_grid.dtype}")
    print(f"DD shape M,N: {modem.M},{modem.N}")
    print(f"cp_len: {modem.cp_len}")
    print(f"time signal shape: {_shape_list(time_signal)}")
    print(f"recovered DD shape: {_shape_list(recovered_dd)}")
    print(f"input average power: {input_power}")
    print(f"time average power: {time_power}")
    print(f"recovered average power: {recovered_power}")
    print(f"normalized DD MSE: {nmse}")
    print(f"max abs error: {max_err}")
    print(f"output time path: {args.output_time}")
    print(f"output dd path: {args.output_dd}")


if __name__ == "__main__":
    main()
