"""Support-gated residual DD denoiser stage.

The denoiser refines the unfolded equalizer output with a residual correction:
x_refined = x_eq + delta. It does not replace the equalizer or produce a
standalone receiver estimate. All complex DD tensors use shape [B, M, N].
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .complex_utils import (
    validate_complex_dd,
    validate_optional_complex_dd,
    validate_support_mask,
)
from .config import ReceiverConfig


@dataclass
class DenoiserOutput:
    """Detailed denoiser output.

    Attributes:
        x_refined: Complex refined DD tensor with shape [B, M, N].
        delta: Complex residual correction with shape [B, M, N].
        features: Optional real feature tensor with shape [B, C, M, N].
        correction_gate: Optional float correction gate with shape [B, 1, M, N].
        confidence_map: Optional float confidence map with shape [B, M, N].
        uncertainty_map: Optional float uncertainty map with shape [B, M, N].
    """

    x_refined: torch.Tensor
    delta: torch.Tensor
    features: torch.Tensor | None = None
    correction_gate: torch.Tensor | None = None
    confidence_map: torch.Tensor | None = None
    uncertainty_map: torch.Tensor | None = None


def build_confidence_map(
    support_mask: torch.Tensor,
    path_confidence_map: torch.Tensor | None = None,
    confidence: torch.Tensor | None = None,
    path_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build a DD confidence map with shape [B, M, N].

    Args:
        support_mask: Float/bool sparse support mask with shape [B, M, N].
        path_confidence_map: Optional precomputed confidence map with shape
            [B, M, N]. If supplied, this is used first.
        confidence: Optional sparse path confidence with shape [B, K].
        path_indices: Optional long path indices with shape [B, K, 2].

    Returns:
        Float confidence map with shape [B, M, N], clamped to [0, 1].
    """

    _validate_real_dd_like(support_mask, "support_mask")
    device = support_mask.device
    dtype = torch.float32 if support_mask.dtype == torch.bool else support_mask.dtype
    support = support_mask.to(device=device, dtype=dtype)
    if path_confidence_map is not None:
        _validate_real_dd_like(path_confidence_map, "path_confidence_map")
        if path_confidence_map.shape != support_mask.shape:
            raise ValueError("path_confidence_map must have shape [B, M, N] matching support_mask.")
        return path_confidence_map.to(device=device, dtype=dtype).clamp(0.0, 1.0)

    if confidence is not None or path_indices is not None:
        if confidence is None or path_indices is None:
            raise ValueError("confidence and path_indices must be provided together.")
        if not torch.is_tensor(path_indices) or path_indices.ndim != 3 or path_indices.shape[-1] != 2:
            raise ValueError("path_indices must have shape [B, K, 2].")
        if not torch.is_tensor(confidence) or confidence.shape != path_indices.shape[:2]:
            raise ValueError("confidence must have shape [B, K].")
        if path_indices.shape[0] != support_mask.shape[0]:
            raise ValueError("path_indices batch size must match support_mask.")
        confidence_map = torch.zeros_like(support)
        batch_indices = torch.arange(support.shape[0], device=device)
        path_indices = path_indices.to(device=device, dtype=torch.long)
        confidence = confidence.to(device=device, dtype=dtype).clamp(0.0, 1.0)
        for path_pos in range(path_indices.shape[1]):
            delay = path_indices[:, path_pos, 0].remainder(support.shape[-2])
            doppler = path_indices[:, path_pos, 1].remainder(support.shape[-1])
            confidence_map[batch_indices, delay, doppler] = confidence[:, path_pos]
        return confidence_map.clamp(0.0, 1.0)

    return support.clamp(0.0, 1.0)


def build_uncertainty_map(
    confidence_map: torch.Tensor,
    residual_energy: torch.Tensor | None = None,
    pilot_residual_power: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build a non-negative uncertainty map with shape [B, M, N].

    Args:
        confidence_map: Float confidence map with shape [B, M, N].
        residual_energy: Optional equalizer residual energy with shape
            [B, M, N].
        pilot_residual_power: Optional pilot residual power with shape [B].

    Returns:
        Float uncertainty map with shape [B, M, N], finite and non-negative.
    """

    _validate_real_dd_like(confidence_map, "confidence_map")
    dtype = confidence_map.dtype
    device = confidence_map.device
    uncertainty = (1.0 - confidence_map.to(device=device, dtype=dtype).clamp(0.0, 1.0)).clamp_min(0.0)

    if residual_energy is not None:
        _validate_real_dd_like(residual_energy, "residual_energy")
        if residual_energy.shape != confidence_map.shape:
            raise ValueError("residual_energy must have shape [B, M, N] matching confidence_map.")
        residual = residual_energy.to(device=device, dtype=dtype).clamp_min(0.0)
        residual_scale = residual.flatten(1).amax(dim=-1).clamp_min(1e-12).reshape(-1, 1, 1)
        uncertainty = uncertainty + residual / residual_scale

    if pilot_residual_power is not None:
        if not torch.is_tensor(pilot_residual_power) or pilot_residual_power.shape != (confidence_map.shape[0],):
            raise ValueError("pilot_residual_power must have shape [B].")
        pilot = pilot_residual_power.to(device=device, dtype=dtype).clamp_min(0.0)
        pilot = pilot / (pilot + 1.0)
        uncertainty = uncertainty + pilot.reshape(-1, 1, 1)

    return torch.nan_to_num(uncertainty.clamp_min(0.0), nan=0.0, posinf=3.0, neginf=0.0).clamp_max(3.0)


class ResidualDDDenoiser(nn.Module):
    """Support-gated residual DD denoiser for x_eq [B, M, N].

    The CNN predicts only delta [B, M, N], and the returned estimate is always
    x_eq + delta. If support_mask is omitted, it uses an all-ones mask [B, M, N]
    to indicate that no sparse support prior was supplied.
    """

    def __init__(self, config: ReceiverConfig):
        super().__init__()
        self.config = config
        self.input_channels = 12
        hidden = config.hidden_channels

        self.input_proj = nn.Conv2d(self.input_channels, hidden, kernel_size=3, padding=1)
        self.support_gate = nn.Conv2d(1, hidden, kernel_size=1)
        self.confidence_gate = nn.Conv2d(1, hidden, kernel_size=1)
        self.uncertainty_gate = nn.Conv2d(1, hidden, kernel_size=1)
        self.hidden = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.gate_head = nn.Conv2d(hidden, 1, kernel_size=1)
        self.delta_head = nn.Conv2d(hidden, 2, kernel_size=3, padding=1)
        self._init_identity()

    def forward(
        self,
        x_eq: torch.Tensor,
        y_dd: torch.Tensor | None = None,
        h_dd: torch.Tensor | None = None,
        support_mask: torch.Tensor | None = None,
        snr_db: torch.Tensor | float | None = None,
        return_delta: bool = False,
        confidence_map: torch.Tensor | None = None,
        path_confidence: torch.Tensor | None = None,
        path_indices: torch.Tensor | None = None,
        uncertainty_map: torch.Tensor | None = None,
        residual_energy: torch.Tensor | None = None,
        pilot_residual_power: torch.Tensor | None = None,
    ) -> torch.Tensor | DenoiserOutput:
        """Refine equalized DD tensor x_eq [B, M, N].

        Args:
            x_eq: Complex unfolded equalizer output with shape [B, M, N].
            y_dd: Optional complex received DD grid with shape [B, M, N]. If
                omitted, its real/imag channels are zeros [B, M, N].
            h_dd: Optional complex sparse DD channel with shape [B, M, N]. If
                omitted, its real/imag/abs channels are zeros [B, M, N].
            support_mask: Optional bool/float mask with shape [B, M, N]. If
                omitted, an all-ones mask [B, M, N] is used for support gating.
            snr_db: Optional scalar or [B] tensor. If omitted, snr map is zeros
                [B, M, N].
            return_delta: If false, return x_refined [B, M, N]. If true,
                return DenoiserOutput with x_refined [B, M, N] and delta
                [B, M, N].
            confidence_map: Optional CE confidence map with shape [B, M, N].
            path_confidence: Optional sparse path confidence with shape [B, K].
            path_indices: Optional sparse path indices with shape [B, K, 2].
            uncertainty_map: Optional precomputed uncertainty map with shape
                [B, M, N].
            residual_energy: Optional equalizer residual energy with shape
                [B, M, N].
            pilot_residual_power: Optional pilot CE residual power with shape
                [B].

        Returns:
            Complex x_refined [B, M, N] or DenoiserOutput.
        """

        validate_complex_dd(x_eq, self.config, "x_eq")
        validate_optional_complex_dd(y_dd, self.config, "y_dd")
        validate_optional_complex_dd(h_dd, self.config, "h_dd")
        self._validate_optional_batch(y_dd, x_eq, "y_dd")
        self._validate_optional_batch(h_dd, x_eq, "h_dd")
        if support_mask is not None:
            validate_support_mask(support_mask, self.config, "support_mask")
            if support_mask.shape[0] != x_eq.shape[0]:
                raise ValueError("support_mask batch size must match x_eq.")
        self._validate_optional_real_dd(confidence_map, x_eq, "confidence_map")
        self._validate_optional_real_dd(uncertainty_map, x_eq, "uncertainty_map")
        self._validate_optional_real_dd(residual_energy, x_eq, "residual_energy")
        if pilot_residual_power is not None:
            if not torch.is_tensor(pilot_residual_power) or pilot_residual_power.shape != (x_eq.shape[0],):
                raise ValueError("pilot_residual_power must have shape [B].")

        support = self._support(support_mask, x_eq)
        resolved_confidence_map = build_confidence_map(
            support,
            path_confidence_map=confidence_map,
            confidence=path_confidence,
            path_indices=path_indices,
        ).to(device=x_eq.device, dtype=x_eq.real.dtype)
        resolved_uncertainty_map = (
            uncertainty_map.to(device=x_eq.device, dtype=x_eq.real.dtype)
            if uncertainty_map is not None
            else build_uncertainty_map(
                resolved_confidence_map,
                residual_energy=residual_energy,
                pilot_residual_power=pilot_residual_power,
            )
        ).to(device=x_eq.device, dtype=x_eq.real.dtype)

        if not self.config.use_refinement_net:
            delta = torch.zeros_like(x_eq)
            x_refined = x_eq + delta
            correction_gate = torch.zeros(x_eq.shape[0], 1, self.config.M, self.config.N, device=x_eq.device, dtype=x_eq.real.dtype)
            output = DenoiserOutput(
                x_refined=x_refined,
                delta=delta,
                features=None,
                correction_gate=correction_gate,
                confidence_map=resolved_confidence_map,
                uncertainty_map=resolved_uncertainty_map,
            )
            return output if return_delta else x_refined

        channels, support, confidence, uncertainty = self._build_inputs(
            x_eq,
            y_dd,
            h_dd,
            support,
            snr_db,
            resolved_confidence_map,
            resolved_uncertainty_map,
        )
        features = self.input_proj(channels)
        support_gate = 0.5 + torch.sigmoid(self.support_gate(support.unsqueeze(1)))
        confidence_gate = (0.5 + torch.sigmoid(self.confidence_gate(confidence.unsqueeze(1)))) * (
            1.25 - 0.75 * confidence.unsqueeze(1)
        )
        uncertainty_gate = (0.5 + torch.sigmoid(self.uncertainty_gate(uncertainty.unsqueeze(1)))) * (
            0.75 + uncertainty.clamp(0.0, 1.0).unsqueeze(1)
        )
        features = self.hidden(features * support_gate * confidence_gate * uncertainty_gate)
        delta_channels = self.delta_head(features)
        raw_delta = torch.complex(delta_channels[:, 0], delta_channels[:, 1]).to(dtype=x_eq.dtype)
        correction_gate = torch.sigmoid(self.gate_head(features)) * (0.25 + uncertainty.clamp(0.0, 1.0).unsqueeze(1))
        correction_gate = correction_gate.to(dtype=x_eq.real.dtype)
        delta = raw_delta * correction_gate.squeeze(1).to(dtype=x_eq.dtype)
        x_refined = x_eq + delta

        output = DenoiserOutput(
            x_refined=x_refined,
            delta=delta,
            features=features,
            correction_gate=correction_gate,
            confidence_map=confidence,
            uncertainty_map=uncertainty,
        )
        return output if return_delta else x_refined

    def _build_inputs(
        self,
        x_eq: torch.Tensor,
        y_dd: torch.Tensor | None,
        h_dd: torch.Tensor | None,
        support_mask: torch.Tensor,
        snr_db: torch.Tensor | float | None,
        confidence_map: torch.Tensor,
        uncertainty_map: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build real CNN inputs [B, C, M, N] and maps [B, M, N]."""

        real_dtype = self.input_proj.weight.dtype
        device = x_eq.device
        zeros = torch.zeros_like(x_eq.real, dtype=real_dtype, device=device)

        y_real = zeros if y_dd is None else y_dd.real.to(dtype=real_dtype)
        y_imag = zeros if y_dd is None else y_dd.imag.to(dtype=real_dtype)
        h_real = zeros if h_dd is None else h_dd.real.to(dtype=real_dtype)
        h_imag = zeros if h_dd is None else h_dd.imag.to(dtype=real_dtype)
        h_abs = zeros if h_dd is None else h_dd.abs().to(dtype=real_dtype)
        support = support_mask.to(device=device, dtype=real_dtype)
        snr = self._snr_map(snr_db, x_eq).to(dtype=real_dtype)
        confidence = confidence_map.to(device=device, dtype=real_dtype).clamp(0.0, 1.0)
        uncertainty = uncertainty_map.to(device=device, dtype=real_dtype).clamp_min(0.0)

        channels = torch.stack(
            (
                x_eq.real.to(dtype=real_dtype),
                x_eq.imag.to(dtype=real_dtype),
                y_real,
                y_imag,
                x_eq.abs().to(dtype=real_dtype),
                support,
                h_real,
                h_imag,
                h_abs,
                snr,
                confidence,
                uncertainty,
            ),
            dim=1,
        )
        return channels, support, confidence, uncertainty

    def _support(self, support_mask: torch.Tensor | None, x_eq: torch.Tensor) -> torch.Tensor:
        """Return support mask [B, M, N], defaulting to all ones."""

        if support_mask is None:
            return torch.ones(x_eq.shape, device=x_eq.device, dtype=x_eq.real.dtype)
        return support_mask.to(device=x_eq.device, dtype=x_eq.real.dtype)

    def _snr_map(self, snr_db: torch.Tensor | float | None, x_eq: torch.Tensor) -> torch.Tensor:
        """Return normalized SNR map with shape [B, M, N]."""

        if snr_db is None:
            snr = torch.zeros(x_eq.shape[0], device=x_eq.device, dtype=x_eq.real.dtype)
        else:
            snr = torch.as_tensor(snr_db, device=x_eq.device, dtype=x_eq.real.dtype)
            if snr.ndim == 0:
                snr = snr.expand(x_eq.shape[0])
            elif snr.shape != (x_eq.shape[0],):
                raise ValueError("snr_db must be scalar or have shape [B].")
        return (snr / 30.0).reshape(-1, 1, 1).expand(-1, self.config.M, self.config.N)

    def _validate_optional_batch(
        self,
        value: torch.Tensor | None,
        reference: torch.Tensor,
        name: str,
    ) -> None:
        """Validate optional value batch for tensors [B, M, N]."""

        if value is not None and value.shape[0] != reference.shape[0]:
            raise ValueError(f"{name} batch size must match x_eq.")

    def _validate_optional_real_dd(
        self,
        value: torch.Tensor | None,
        reference: torch.Tensor,
        name: str,
    ) -> None:
        """Validate optional real tensor value with shape [B, M, N]."""

        if value is None:
            return
        if not torch.is_tensor(value) or value.shape != reference.shape or torch.is_complex(value):
            raise ValueError(f"{name} must have real shape [B, M, N].")

    def _init_identity(self) -> None:
        """Initialize delta head so initial delta [B, M, N] is exactly zero."""

        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.zeros_(self.gate_head.bias)


def _validate_real_dd_like(value: torch.Tensor, name: str) -> None:
    if not torch.is_tensor(value) or value.ndim != 3 or torch.is_complex(value):
        raise ValueError(f"{name} must have real shape [B, M, N].")
