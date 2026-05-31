"""Local off-grid sparse path refinement for on-grid DD channel estimates.

This module refines sparse path metadata by predicting fractional delay and
Doppler offsets around on-grid support. It prepares the receiver for a later
parametric H_theta operator; it does not upgrade the current DD operator and
does not replace the sparse model-driven receiver with a dense neural receiver.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from .sparse_channel import SparseChannelEstimate


@dataclass
class OffGridPathRefinement:
    """Off-grid refinement metadata for sparse DD paths.

    Attributes:
        fractional_offsets: Float offsets with shape [B, K, 2], ordered as
            [delay_offset, doppler_offset] and bounded by max_offset.
        gain_correction: Complex gain correction with shape [B, K].
        confidence_update: Float confidence update with shape [B, K] in [0, 1].
        patch_features: Optional local patch features with shape
            [B, K, C, P, Q].
    """

    fractional_offsets: torch.Tensor
    gain_correction: torch.Tensor
    confidence_update: torch.Tensor
    patch_features: torch.Tensor | None = None


class LocalPatchExtractor(nn.Module):
    """Extract circular local DD patches around sparse paths.

    The extractor consumes h_coarse [B, M, N] and path_indices [B, K, 2], then
    returns real patch features [B, K, C, P, Q], where P = 2*radius_delay + 1,
    Q = 2*radius_doppler + 1, and C = 6 for real, imag, abs, normalized local
    delay coordinate, normalized local Doppler coordinate, and confidence.
    """

    def __init__(self, radius_delay: int, radius_doppler: int):
        super().__init__()
        if not isinstance(radius_delay, int) or radius_delay < 0:
            raise ValueError("radius_delay must be a non-negative integer.")
        if not isinstance(radius_doppler, int) or radius_doppler < 0:
            raise ValueError("radius_doppler must be a non-negative integer.")
        self.radius_delay = radius_delay
        self.radius_doppler = radius_doppler
        self.num_channels = 6

    def forward(
        self,
        h_coarse: torch.Tensor,
        path_indices: torch.Tensor,
        confidence: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return local patch features [B, K, C, P, Q].

        Args:
            h_coarse: Complex coarse channel projection with shape [B, M, N].
            path_indices: Long on-grid path indices with shape [B, K, 2].
            confidence: Optional float confidence with shape [B, K].

        Returns:
            Float patch_features with shape [B, K, 6, P, Q], using circular
            wrap-around on the DD grid.
        """

        _validate_complex_dd_unconfigured(h_coarse, "h_coarse")
        _validate_path_indices(h_coarse, path_indices)
        if confidence is not None:
            _validate_confidence(path_indices, confidence)

        batch_size, delay_bins, doppler_bins = h_coarse.shape
        num_paths = path_indices.shape[1]
        p_size = 2 * self.radius_delay + 1
        q_size = 2 * self.radius_doppler + 1
        device = h_coarse.device
        real_dtype = h_coarse.real.dtype
        path_indices = path_indices.to(device=device, dtype=torch.long)

        complex_patches = torch.zeros(
            batch_size,
            num_paths,
            p_size,
            q_size,
            device=device,
            dtype=h_coarse.dtype,
        )
        batch_index = torch.arange(batch_size, device=device).reshape(batch_size, 1).expand(-1, num_paths)
        delay_offsets = torch.arange(-self.radius_delay, self.radius_delay + 1, device=device)
        doppler_offsets = torch.arange(-self.radius_doppler, self.radius_doppler + 1, device=device)
        for delay_pos, delay_offset in enumerate(delay_offsets.tolist()):
            delay_index = (path_indices[..., 0] + int(delay_offset)).remainder(delay_bins)
            for doppler_pos, doppler_offset in enumerate(doppler_offsets.tolist()):
                doppler_index = (path_indices[..., 1] + int(doppler_offset)).remainder(doppler_bins)
                complex_patches[:, :, delay_pos, doppler_pos] = h_coarse[batch_index, delay_index, doppler_index]

        delay_coord = _normalized_coordinate(
            delay_offsets,
            self.radius_delay,
            batch_size,
            num_paths,
            p_size,
            q_size,
            real_dtype,
        )
        doppler_coord = _normalized_coordinate(
            doppler_offsets,
            self.radius_doppler,
            batch_size,
            num_paths,
            q_size,
            p_size,
            real_dtype,
        ).transpose(-2, -1)
        if confidence is None:
            confidence_channel = torch.ones(
                batch_size,
                num_paths,
                p_size,
                q_size,
                device=device,
                dtype=real_dtype,
            )
        else:
            confidence_channel = confidence.to(device=device, dtype=real_dtype).reshape(batch_size, num_paths, 1, 1)
            confidence_channel = confidence_channel.expand(-1, -1, p_size, q_size)

        return torch.stack(
            (
                complex_patches.real,
                complex_patches.imag,
                complex_patches.abs(),
                delay_coord.to(device=device),
                doppler_coord.to(device=device),
                confidence_channel,
            ),
            dim=2,
        )


class OffGridRefinementNet(nn.Module):
    """Predict local fractional offsets for sparse DD paths.

    Inputs to forward are h_coarse [B, M, N], path_indices [B, K, 2],
    path_gains [B, K], and optional confidence [B, K]. Outputs are
    fractional_offsets [B, K, 2], gain_correction [B, K], confidence_update
    [B, K], and patch_features [B, K, C, P, Q]. This is a small local
    patch network and does not replace the model-driven sparse receiver.
    """

    def __init__(
        self,
        radius_delay: int,
        radius_doppler: int,
        max_offset: float,
        confidence_floor: float,
        gain_correction_scale: float,
        hidden_channels: int = 16,
    ):
        super().__init__()
        if not isinstance(hidden_channels, int) or hidden_channels <= 0:
            raise ValueError("hidden_channels must be a positive integer.")
        if not isinstance(max_offset, (int, float)) or not 0.0 < max_offset <= 0.5:
            raise ValueError("max_offset must be in (0, 0.5].")
        if not isinstance(confidence_floor, (int, float)) or not 0.0 <= confidence_floor <= 1.0:
            raise ValueError("confidence_floor must be in [0, 1].")
        if not isinstance(gain_correction_scale, (int, float)) or gain_correction_scale < 0:
            raise ValueError("gain_correction_scale must be non-negative.")

        self.patch_extractor = LocalPatchExtractor(radius_delay, radius_doppler)
        self.max_offset = float(max_offset)
        self.confidence_floor = float(confidence_floor)
        self.gain_correction_scale = float(gain_correction_scale)
        self.patch_net = nn.Sequential(
            nn.Conv2d(self.patch_extractor.num_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.offset_head = nn.Linear(hidden_channels, 2)
        self.gain_head = nn.Linear(hidden_channels, 2)
        self.confidence_head = nn.Linear(hidden_channels, 1)
        self._init_small_updates()

    def forward(
        self,
        h_coarse: torch.Tensor,
        path_indices: torch.Tensor,
        path_gains: torch.Tensor,
        confidence: torch.Tensor | None = None,
    ) -> OffGridPathRefinement:
        """Return off-grid refinement for sparse paths.

        Args:
            h_coarse: Complex on-grid channel projection with shape [B, M, N].
            path_indices: Long on-grid path indices with shape [B, K, 2].
            path_gains: Complex path gains with shape [B, K].
            confidence: Optional float path confidence with shape [B, K].

        Returns:
            OffGridPathRefinement containing fractional_offsets [B, K, 2],
            gain_correction [B, K], confidence_update [B, K], and
            patch_features [B, K, C, P, Q].
        """

        _validate_complex_dd_unconfigured(h_coarse, "h_coarse")
        _validate_path_indices(h_coarse, path_indices)
        _validate_path_gains(path_indices, path_gains)
        if confidence is not None:
            _validate_confidence(path_indices, confidence)
        path_gains = path_gains.to(device=h_coarse.device)
        if confidence is not None:
            confidence = confidence.to(device=h_coarse.device)

        patch_features = self.patch_extractor(h_coarse, path_indices, confidence=confidence)
        batch_size, num_paths, channels, p_size, q_size = patch_features.shape
        features = patch_features.reshape(batch_size * num_paths, channels, p_size, q_size)
        features = features.to(dtype=self.patch_net[0].weight.dtype)
        hidden = self.patch_net(features).reshape(batch_size, num_paths, -1)

        offsets_raw = self.offset_head(hidden)
        fractional_offsets = self.max_offset * torch.tanh(offsets_raw)
        gain_raw = self.gain_head(hidden)
        gain_correction = self.gain_correction_scale * torch.complex(gain_raw[..., 0], gain_raw[..., 1])
        gain_correction = gain_correction.to(device=path_gains.device, dtype=path_gains.dtype)
        confidence_update = torch.sigmoid(self.confidence_head(hidden).squeeze(-1))

        active = self._active_mask(path_indices, confidence, hidden)
        fractional_offsets = fractional_offsets.to(device=h_coarse.device, dtype=h_coarse.real.dtype) * active.unsqueeze(-1)
        gain_correction = gain_correction * active.to(device=gain_correction.device, dtype=path_gains.real.dtype)
        confidence_update = confidence_update.to(device=h_coarse.device, dtype=h_coarse.real.dtype) * active
        return OffGridPathRefinement(
            fractional_offsets=fractional_offsets,
            gain_correction=gain_correction,
            confidence_update=confidence_update,
            patch_features=patch_features,
        )

    def _active_mask(
        self,
        path_indices: torch.Tensor,
        confidence: torch.Tensor | None,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        """Return active path mask [B, K] from confidence [B, K]."""

        if confidence is None:
            return torch.ones(path_indices.shape[:2], device=hidden.device, dtype=hidden.dtype)
        return (confidence.to(device=hidden.device, dtype=hidden.dtype) > self.confidence_floor).to(dtype=hidden.dtype)

    def _init_small_updates(self) -> None:
        """Initialize offset/gain heads for small updates [B, K, 2] and [B, K]."""

        nn.init.normal_(self.offset_head.weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.offset_head.bias)
        nn.init.normal_(self.gain_head.weight, mean=0.0, std=1e-4)
        nn.init.zeros_(self.gain_head.bias)
        nn.init.zeros_(self.confidence_head.weight)
        nn.init.constant_(self.confidence_head.bias, math.log(0.99 / 0.01))


def apply_offgrid_refinement(
    estimate: "SparseChannelEstimate",
    refinement: OffGridPathRefinement,
) -> "SparseChannelEstimate":
    """Apply off-grid metadata to a sparse channel estimate.

    Args:
        estimate: SparseChannelEstimate with h_dd [B, M, N], path_indices
            [B, K, 2], path_gains [B, K], and optional confidence [B, K].
        refinement: OffGridPathRefinement with fractional_offsets [B, K, 2],
            gain_correction [B, K], and confidence_update [B, K].

    Returns:
        SparseChannelEstimate with fractional_offsets [B, K, 2], corrected
        path_gains [B, K], and confidence [B, K]. support_mask [B, M, N] and
        path_indices [B, K, 2] are preserved. h_dd remains an on-grid
        diagnostic projection; fractional_offsets are the representation used
        by the later parametric H_theta operator.
    """

    _validate_refinement_shapes(estimate, refinement)
    corrected_gains = estimate.path_gains + refinement.gain_correction.to(
        device=estimate.path_gains.device,
        dtype=estimate.path_gains.dtype,
    )
    if estimate.confidence is None:
        confidence = refinement.confidence_update.to(device=corrected_gains.device, dtype=corrected_gains.real.dtype)
    else:
        confidence = estimate.confidence.to(device=corrected_gains.device, dtype=corrected_gains.real.dtype)
        confidence = confidence * refinement.confidence_update.to(device=corrected_gains.device, dtype=confidence.dtype)

    h_projection = estimate.h_dd.clone()
    active = confidence > 0
    _scatter_corrected_gains(h_projection, estimate.path_indices, corrected_gains, active)
    return replace(
        estimate,
        h_dd=h_projection,
        path_gains=corrected_gains,
        confidence=confidence,
        fractional_offsets=refinement.fractional_offsets.to(
            device=estimate.path_indices.device,
            dtype=estimate.h_dd.real.dtype,
        ),
    )


def fractional_offset_l2_loss(
    pred_offsets: torch.Tensor,
    true_offsets: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return L2 loss for fractional offsets pred/true [B, K, 2].

    Args:
        pred_offsets: Float predicted offsets with shape [B, K, 2].
        true_offsets: Float target offsets with shape [B, K, 2].
        mask: Optional float/bool mask with shape [B, K].

    Returns:
        Scalar tensor containing masked mean-squared offset error.
    """

    if pred_offsets.shape != true_offsets.shape or pred_offsets.ndim != 3 or pred_offsets.shape[-1] != 2:
        raise ValueError("pred_offsets and true_offsets must have shape [B, K, 2].")
    error = (pred_offsets - true_offsets).pow(2).sum(dim=-1)
    return _masked_mean(error, mask)


def gain_correction_l2_loss(
    pred_gains: torch.Tensor,
    true_gains: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return L2 loss for complex gain corrections pred/true [B, K].

    Args:
        pred_gains: Complex predicted gains with shape [B, K].
        true_gains: Complex target gains with shape [B, K].
        mask: Optional float/bool mask with shape [B, K].

    Returns:
        Scalar tensor containing masked mean-squared complex gain error.
    """

    if pred_gains.shape != true_gains.shape or pred_gains.ndim != 2:
        raise ValueError("pred_gains and true_gains must have shape [B, K].")
    if not torch.is_complex(pred_gains) or not torch.is_complex(true_gains):
        raise TypeError("pred_gains and true_gains must be complex tensors with shape [B, K].")
    error = (pred_gains - true_gains).abs().pow(2)
    return _masked_mean(error, mask)


def _validate_complex_dd_unconfigured(value: torch.Tensor, name: str) -> None:
    if not torch.is_tensor(value) or value.ndim != 3 or not torch.is_complex(value):
        raise TypeError(f"{name} must be a complex tensor with shape [B, M, N].")


def _validate_path_indices(reference_dd: torch.Tensor, path_indices: torch.Tensor) -> None:
    if not torch.is_tensor(path_indices) or path_indices.ndim != 3 or path_indices.shape[-1] != 2:
        raise ValueError("path_indices must have shape [B, K, 2].")
    if path_indices.shape[0] != reference_dd.shape[0]:
        raise ValueError("path_indices batch size must match h_coarse.")


def _validate_path_gains(path_indices: torch.Tensor, path_gains: torch.Tensor) -> None:
    if not torch.is_tensor(path_gains) or path_gains.ndim != 2 or path_gains.shape != path_indices.shape[:2]:
        raise ValueError("path_gains must have shape [B, K].")
    if not torch.is_complex(path_gains):
        raise TypeError("path_gains must be complex with shape [B, K].")


def _validate_confidence(path_indices: torch.Tensor, confidence: torch.Tensor) -> None:
    if not torch.is_tensor(confidence) or confidence.shape != path_indices.shape[:2]:
        raise ValueError("confidence must have shape [B, K].")


def _normalized_coordinate(
    offsets: torch.Tensor,
    radius: int,
    batch_size: int,
    num_paths: int,
    primary_size: int,
    secondary_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    denom = float(max(radius, 1))
    coord = offsets.to(dtype=dtype) / denom
    return coord.reshape(1, 1, primary_size, 1).expand(batch_size, num_paths, primary_size, secondary_size)


def _validate_refinement_shapes(estimate: "SparseChannelEstimate", refinement: OffGridPathRefinement) -> None:
    if refinement.fractional_offsets.shape != estimate.path_indices.shape:
        raise ValueError("fractional_offsets must have shape [B, K, 2] matching path_indices.")
    if refinement.gain_correction.shape != estimate.path_gains.shape:
        raise ValueError("gain_correction must have shape [B, K] matching path_gains.")
    if refinement.confidence_update.shape != estimate.path_gains.shape:
        raise ValueError("confidence_update must have shape [B, K] matching path_gains.")
    if not torch.is_complex(refinement.gain_correction):
        raise TypeError("gain_correction must be complex with shape [B, K].")


def _scatter_corrected_gains(
    h_projection: torch.Tensor,
    path_indices: torch.Tensor,
    corrected_gains: torch.Tensor,
    active: torch.Tensor,
) -> None:
    batch_indices = torch.arange(h_projection.shape[0], device=h_projection.device)
    for path_pos in range(path_indices.shape[1]):
        active_path = active[:, path_pos].to(device=h_projection.device)
        if not bool(active_path.any()):
            continue
        batch = batch_indices[active_path]
        delay = path_indices[active_path, path_pos, 0].to(device=h_projection.device, dtype=torch.long)
        doppler = path_indices[active_path, path_pos, 1].to(device=h_projection.device, dtype=torch.long)
        h_projection[batch, delay, doppler] = corrected_gains[active_path, path_pos].to(
            device=h_projection.device,
            dtype=h_projection.dtype,
        )


def _masked_mean(error: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return error.mean()
    if mask.shape != error.shape:
        raise ValueError("mask must have shape [B, K].")
    mask_real = mask.to(device=error.device, dtype=error.dtype)
    return (error * mask_real).sum() / mask_real.sum().clamp_min(1e-12)
