"""
Time-varying multipath channel with complex AWGN.

The channel is independent of OTFS. Later code can pass any transmitted
complex-baseband waveform into this PyTorch layer:

    y[n] = sum_p h_p exp(j 2*pi*f_p*n/fs) x[n - tau_p] + w[n]

Only PyTorch is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover
    raise ImportError("channel_model.py requires PyTorch.") from exc


Number = Union[int, float]


@dataclass
class ChannelConfig:
    """Configurable channel parameters."""

    num_paths: int = 6
    sample_rate: float = 15.36e6
    snr_db: Optional[float] = 20.0

    max_delay_samples: float = 12.0
    max_doppler_hz: float = 1500.0
    path_delays: Optional[Sequence[float]] = None
    path_dopplers_hz: Optional[Sequence[float]] = None
    path_powers_db: Optional[Sequence[float]] = None

    fading: str = "rayleigh"  # "rayleigh", "rician", or "fixed"
    rician_k_db: float = 8.0
    doppler_distribution: str = "jakes"  # "jakes" or "uniform"
    normalize_path_powers: bool = True

    add_awgn: bool = True
    randomize_each_forward: bool = False
    fractional_delays: bool = True
    complex_dtype: str = "complex64"  # "complex64" or "complex128"
    seed: Optional[int] = 7


@dataclass
class ChannelOutput:
    """Optional detailed output for debugging or neural-receiver conditioning."""

    y: torch.Tensor
    clean: torch.Tensor
    noise: torch.Tensor
    path_gains: torch.Tensor
    delays: torch.Tensor
    dopplers_hz: torch.Tensor
    time_varying_gains: torch.Tensor
    conditioning: torch.Tensor
    snr_db: Optional[torch.Tensor]


def max_doppler_from_velocity(speed_kmh: Number, carrier_hz: Number) -> float:
    """Return maximum Doppler shift in Hz."""

    speed_mps = float(speed_kmh) / 3.6
    return speed_mps * float(carrier_hz) / 299_792_458.0


class TimeVaryingMultipathChannel(nn.Module):
    """Complex baseband channel layer.

    Input shape:
        [time] or [batch, time], real or complex.

    Default output:
        y only, so the module behaves like a normal PyTorch layer.

    Detailed output:
        call channel(x, return_info=True).

    The channel has no nn.Parameter objects. Delays, Dopplers, path powers, and
    fixed path gains are stored as buffers, so gradients flow through the input
    waveform but optimizers will not update the channel itself.
    """

    def __init__(self, config: Optional[ChannelConfig] = None):
        super().__init__()
        self.config = config or ChannelConfig()
        self._validate_config()

        self.complex_dtype = self._complex_dtype(self.config.complex_dtype)
        self.real_dtype = torch.float64 if self.complex_dtype == torch.complex128 else torch.float32

        generator = None
        if self.config.seed is not None:
            generator = torch.Generator()
            generator.manual_seed(int(self.config.seed))

        delays = self._initial_delays(generator)
        dopplers = self._initial_dopplers(generator)
        powers = self._path_powers()
        gains = self._draw_gains(batch_size=1, powers=powers, device=torch.device("cpu"), generator=generator)[0]

        self.register_buffer("delay_samples", delays)
        self.register_buffer("dopplers_hz", dopplers)
        self.register_buffer("path_powers", powers)
        self.register_buffer("path_gains", gains)

    def forward(
        self,
        x: torch.Tensor,
        snr_db: Optional[Union[Number, torch.Tensor]] = None,
        path_gains: Optional[torch.Tensor] = None,
        path_delays: Optional[torch.Tensor] = None,
        path_dopplers_hz: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        return_info: bool = False,
    ) -> Union[torch.Tensor, ChannelOutput]:
        squeeze_batch = x.ndim == 1
        if squeeze_batch:
            x = x.unsqueeze(0)
        if x.ndim != 2:
            raise ValueError("x must have shape [time] or [batch, time].")

        if not torch.is_complex(x):
            x = torch.complex(x, torch.zeros_like(x))
        x = x.to(dtype=self.complex_dtype)

        batch_size, num_samples = x.shape
        delays, dopplers, gains = self._resolve_paths(
            batch_size=batch_size,
            device=x.device,
            path_delays=path_delays,
            path_dopplers_hz=path_dopplers_hz,
            path_gains=path_gains,
        )

        time = torch.arange(num_samples, device=x.device, dtype=self.real_dtype) / self.config.sample_rate
        phase = torch.exp(1j * 2.0 * torch.pi * dopplers.unsqueeze(-1) * time).to(self.complex_dtype)
        time_varying_gains = gains.unsqueeze(-1) * phase

        clean = torch.zeros_like(x)
        for path_idx in range(self.config.num_paths):
            clean = clean + time_varying_gains[:, path_idx, :] * self._delay(x, delays[:, path_idx])

        used_snr = self.config.snr_db if snr_db is None else snr_db
        y, used_noise, snr_tensor = self._add_noise(clean, used_snr, noise)

        if squeeze_batch:
            y = y.squeeze(0)
        if not return_info:
            return y

        conditioning = self.make_conditioning(time_varying_gains, delays, dopplers)
        if squeeze_batch:
            clean = clean.squeeze(0)
            used_noise = used_noise.squeeze(0)
            gains = gains.squeeze(0)
            delays = delays.squeeze(0)
            dopplers = dopplers.squeeze(0)
            time_varying_gains = time_varying_gains.squeeze(0)
            conditioning = conditioning.squeeze(0)

        return ChannelOutput(
            y=y,
            clean=clean,
            noise=used_noise,
            path_gains=gains,
            delays=delays,
            dopplers_hz=dopplers,
            time_varying_gains=time_varying_gains,
            conditioning=conditioning,
            snr_db=snr_tensor,
        )

    def set_parameters(
        self,
        delays: Optional[Sequence[float]] = None,
        dopplers_hz: Optional[Sequence[float]] = None,
        path_powers_db: Optional[Sequence[float]] = None,
        snr_db: Optional[float] = None,
    ) -> None:
        """Update fixed channel parameters after initialization."""

        with torch.no_grad():
            if delays is not None:
                self.delay_samples.copy_(self._as_delay_vector(delays, "delays").to(self.delay_samples))
            if dopplers_hz is not None:
                self.dopplers_hz.copy_(self._as_vector(dopplers_hz, "dopplers_hz").to(self.dopplers_hz))
            if path_powers_db is not None:
                self.path_powers.copy_(self._powers_from_db(path_powers_db).to(self.path_powers))
                gains = self._draw_gains(1, self.path_powers, self.path_powers.device, generator=None)[0]
                self.path_gains.copy_(gains.to(self.path_gains))
            if snr_db is not None:
                self.config.snr_db = float(snr_db)

    def make_conditioning(
        self,
        time_varying_gains: torch.Tensor,
        delays: torch.Tensor,
        dopplers_hz: torch.Tensor,
    ) -> torch.Tensor:
        """Return real features [batch, time, 4*num_paths].

        Feature order: real gain, imaginary gain, normalized delay,
        normalized Doppler.
        """

        if time_varying_gains.ndim != 3:
            raise ValueError("time_varying_gains must have shape [batch, paths, time].")

        batch_size, num_paths, num_samples = time_varying_gains.shape
        delay_scale = max(float(self.config.max_delay_samples), 1.0)
        doppler_scale = max(float(self.config.max_doppler_hz), 1.0)

        gain_real = time_varying_gains.real.transpose(1, 2)
        gain_imag = time_varying_gains.imag.transpose(1, 2)
        delay_feat = (delays / delay_scale).unsqueeze(1).expand(batch_size, num_samples, num_paths)
        doppler_feat = (dopplers_hz / doppler_scale).unsqueeze(1).expand(batch_size, num_samples, num_paths)
        return torch.cat([gain_real, gain_imag, delay_feat, doppler_feat], dim=-1)

    def impulse_response(self, num_samples: int, batch_size: int = 1, device: Optional[torch.device] = None) -> torch.Tensor:
        """Approximate dense impulse response [batch, time, delay_bin]."""

        device = device or self.path_powers.device
        delays, dopplers, gains = self._resolve_paths(batch_size, device, None, None, None)
        time = torch.arange(num_samples, device=device, dtype=self.real_dtype) / self.config.sample_rate
        phase = torch.exp(1j * 2.0 * torch.pi * dopplers.unsqueeze(-1) * time).to(self.complex_dtype)
        time_gains = gains.unsqueeze(-1) * phase

        max_delay = int(torch.ceil(delays.max()).item()) + 1
        h = torch.zeros(batch_size, num_samples, max_delay, device=device, dtype=self.complex_dtype)
        bins = torch.round(delays).long().clamp(0, max_delay - 1)
        for batch_idx in range(batch_size):
            for path_idx in range(self.config.num_paths):
                h[batch_idx, :, bins[batch_idx, path_idx]] += time_gains[batch_idx, path_idx, :]
        return h

    def _resolve_paths(
        self,
        batch_size: int,
        device: torch.device,
        path_delays: Optional[torch.Tensor],
        path_dopplers_hz: Optional[torch.Tensor],
        path_gains: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.config.randomize_each_forward:
            delays = self._sample_delays(batch_size, device)
            dopplers = self._sample_dopplers(batch_size, device)
            gains = self._draw_gains(batch_size, self.path_powers.to(device), device, generator=None)
        else:
            delays = self._expand(self.delay_samples, batch_size, device, "path_delays")
            dopplers = self._expand(self.dopplers_hz, batch_size, device, "path_dopplers_hz")
            gains = self._expand(self.path_gains, batch_size, device, "path_gains", complex_value=True)

        if path_delays is not None:
            delays = self._expand(path_delays, batch_size, device, "path_delays")
        if path_dopplers_hz is not None:
            dopplers = self._expand(path_dopplers_hz, batch_size, device, "path_dopplers_hz")
        if path_gains is not None:
            gains = self._expand(path_gains, batch_size, device, "path_gains", complex_value=True)
        return delays, dopplers, gains

    def _delay(self, x: torch.Tensor, delay: torch.Tensor) -> torch.Tensor:
        if self.config.fractional_delays:
            return self._fractional_delay(x, delay)
        return self._integer_delay(x, delay)

    def _fractional_delay(self, x: torch.Tensor, delay: torch.Tensor) -> torch.Tensor:
        batch_size, num_samples = x.shape
        sample_index = torch.arange(num_samples, device=x.device, dtype=self.real_dtype)
        source_index = sample_index.unsqueeze(0) - delay.reshape(batch_size, 1)
        x_grid = torch.zeros_like(source_index) if num_samples == 1 else 2.0 * source_index / (num_samples - 1) - 1.0
        y_grid = torch.zeros_like(x_grid)
        grid = torch.stack([x_grid, y_grid], dim=-1).unsqueeze(1)
        real_imag = torch.stack([x.real, x.imag], dim=1).unsqueeze(2)
        sampled = F.grid_sample(real_imag, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        return torch.complex(sampled[:, 0, 0, :], sampled[:, 1, 0, :]).to(self.complex_dtype)

    @staticmethod
    def _integer_delay(x: torch.Tensor, delay: torch.Tensor) -> torch.Tensor:
        batch_size, num_samples = x.shape
        out = torch.zeros_like(x)
        delay_int = torch.round(delay).long().clamp(min=0)
        for batch_idx in range(batch_size):
            d = int(delay_int[batch_idx].item())
            if d == 0:
                out[batch_idx] = x[batch_idx]
            elif d < num_samples:
                out[batch_idx, d:] = x[batch_idx, : num_samples - d]
        return out

    def _add_noise(
        self,
        clean: torch.Tensor,
        snr_db: Optional[Union[Number, torch.Tensor]],
        noise: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if noise is not None:
            noise = noise.to(device=clean.device, dtype=self.complex_dtype)
            return clean + noise, noise, None
        if not self.config.add_awgn or snr_db is None:
            return clean, torch.zeros_like(clean), None

        snr = torch.as_tensor(snr_db, device=clean.device, dtype=self.real_dtype)
        if snr.ndim == 1:
            snr = snr.reshape(-1, 1)
        snr_linear = 10.0 ** (snr / 10.0)
        signal_power = clean.abs().pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-12)
        noise_power = signal_power / snr_linear
        scale = torch.sqrt(noise_power / 2.0)
        noise = torch.complex(torch.randn_like(clean.real) * scale, torch.randn_like(clean.real) * scale)
        noise = noise.to(self.complex_dtype)
        return clean + noise, noise, torch.as_tensor(snr_db, device=clean.device, dtype=self.real_dtype)

    def _sample_delays(self, batch_size: int, device: torch.device) -> torch.Tensor:
        delays = torch.rand(batch_size, self.config.num_paths, device=device, dtype=self.real_dtype)
        delays = torch.sort(delays * float(self.config.max_delay_samples), dim=-1).values
        delays[:, 0] = 0.0
        return delays

    def _sample_dopplers(self, batch_size: int, device: torch.device) -> torch.Tensor:
        shape = (batch_size, self.config.num_paths)
        if self.config.doppler_distribution == "jakes":
            angles = 2.0 * torch.pi * torch.rand(shape, device=device, dtype=self.real_dtype)
            return float(self.config.max_doppler_hz) * torch.cos(angles)
        values = torch.rand(shape, device=device, dtype=self.real_dtype)
        return (2.0 * values - 1.0) * float(self.config.max_doppler_hz)

    def _initial_delays(self, generator: Optional[torch.Generator]) -> torch.Tensor:
        if self.config.path_delays is not None:
            return self._as_delay_vector(self.config.path_delays, "path_delays")
        delays = torch.rand(self.config.num_paths, dtype=self.real_dtype, generator=generator)
        delays = torch.sort(delays * float(self.config.max_delay_samples)).values
        delays[0] = 0.0
        return delays

    def _initial_dopplers(self, generator: Optional[torch.Generator]) -> torch.Tensor:
        if self.config.path_dopplers_hz is not None:
            return self._as_vector(self.config.path_dopplers_hz, "path_dopplers_hz")
        if self.config.doppler_distribution == "jakes":
            angles = 2.0 * torch.pi * torch.rand(self.config.num_paths, dtype=self.real_dtype, generator=generator)
            return float(self.config.max_doppler_hz) * torch.cos(angles)
        values = torch.rand(self.config.num_paths, dtype=self.real_dtype, generator=generator)
        return (2.0 * values - 1.0) * float(self.config.max_doppler_hz)

    def _path_powers(self) -> torch.Tensor:
        if self.config.path_powers_db is not None:
            return self._powers_from_db(self.config.path_powers_db)
        idx = torch.arange(self.config.num_paths, dtype=self.real_dtype)
        powers = torch.exp(-idx / max(float(self.config.num_paths) / 2.0, 1.0))
        return self._normalize(powers)

    def _powers_from_db(self, values_db: Sequence[float]) -> torch.Tensor:
        powers = 10.0 ** (self._as_vector(values_db, "path_powers_db") / 10.0)
        return self._normalize(powers)

    def _draw_gains(
        self,
        batch_size: int,
        powers: torch.Tensor,
        device: torch.device,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        powers = powers.to(device=device, dtype=self.real_dtype)
        shape = (batch_size, self.config.num_paths)

        if self.config.fading == "fixed":
            phase = 2.0 * torch.pi * torch.rand(self.config.num_paths, device=device, dtype=self.real_dtype, generator=generator)
            gains = torch.sqrt(powers) * torch.exp(1j * phase)
            return gains.to(self.complex_dtype).unsqueeze(0).expand(batch_size, -1).clone()

        real = torch.randn(shape, device=device, dtype=self.real_dtype, generator=generator)
        imag = torch.randn(shape, device=device, dtype=self.real_dtype, generator=generator)
        if self.config.fading == "rayleigh":
            scale = torch.sqrt(powers / 2.0).unsqueeze(0)
            return torch.complex(real * scale, imag * scale).to(self.complex_dtype)

        k = 10.0 ** (float(self.config.rician_k_db) / 10.0)
        los = torch.sqrt(powers * k / (k + 1.0)).unsqueeze(0)
        scatter = torch.sqrt(powers / (2.0 * (k + 1.0))).unsqueeze(0)
        return torch.complex(los + real * scatter, imag * scatter).to(self.complex_dtype)

    def _expand(
        self,
        value: torch.Tensor,
        batch_size: int,
        device: torch.device,
        name: str,
        complex_value: bool = False,
    ) -> torch.Tensor:
        dtype = self.complex_dtype if complex_value else self.real_dtype
        value = value.to(device=device, dtype=dtype)
        if value.ndim == 1:
            if value.numel() != self.config.num_paths:
                raise ValueError(f"{name} must contain {self.config.num_paths} values.")
            return value.unsqueeze(0).expand(batch_size, -1)
        if value.shape != (batch_size, self.config.num_paths):
            raise ValueError(f"{name} must have shape [{batch_size}, {self.config.num_paths}].")
        return value

    def _as_vector(self, values: Sequence[float], name: str) -> torch.Tensor:
        if len(values) != self.config.num_paths:
            raise ValueError(f"{name} must contain {self.config.num_paths} values.")
        return torch.tensor(values, dtype=self.real_dtype)

    def _as_delay_vector(self, values: Sequence[float], name: str) -> torch.Tensor:
        delays = self._as_vector(values, name)
        if torch.any(delays < 0):
            raise ValueError(f"{name} must be non-negative.")
        return delays

    def _normalize(self, powers: torch.Tensor) -> torch.Tensor:
        if self.config.normalize_path_powers:
            return powers / powers.sum().clamp_min(1e-12)
        return powers

    def _validate_config(self) -> None:
        if self.config.num_paths <= 0:
            raise ValueError("num_paths must be positive.")
        if self.config.sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")
        if self.config.max_delay_samples < 0:
            raise ValueError("max_delay_samples must be non-negative.")
        if self.config.max_doppler_hz < 0:
            raise ValueError("max_doppler_hz must be non-negative.")
        if self.config.fading not in {"rayleigh", "rician", "fixed"}:
            raise ValueError("fading must be 'rayleigh', 'rician', or 'fixed'.")
        if self.config.doppler_distribution not in {"jakes", "uniform"}:
            raise ValueError("doppler_distribution must be 'jakes' or 'uniform'.")
        self._complex_dtype(self.config.complex_dtype)

    @staticmethod
    def _complex_dtype(name: str) -> torch.dtype:
        if name == "complex64":
            return torch.complex64
        if name == "complex128":
            return torch.complex128
        raise ValueError("complex_dtype must be 'complex64' or 'complex128'.")


TimeVaryingMultipathConfig = ChannelConfig


__all__ = [
    "ChannelConfig",
    "ChannelOutput",
    "TimeVaryingMultipathChannel",
    "TimeVaryingMultipathConfig",
    "max_doppler_from_velocity",
]
