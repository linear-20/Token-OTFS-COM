"""Sparse DD circular convolution operator for the fixed OTFS receiver.

The main operator uses explicit sparse DD paths and never constructs a dense
MN x MN channel matrix. All DD grids use complex shape [B, M, N].
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .complex_utils import validate_complex_dd, validate_support_mask
from .config import ReceiverConfig
from .sparse_channel import SparseChannelEstimate


@dataclass
class SparseDDOperatorState:
    """Sparse DD circular convolution state.

    Attributes:
        h_dd: Complex sparse DD channel with shape [B, M, N].
        support_mask: Float support mask with shape [B, M, N].
        path_indices: Long path indices with shape [B, K, 2], ordered as
            [delay_idx, doppler_idx].
        path_gains: Complex path gains with shape [B, K].
        confidence: Optional float confidence with shape [B, K].
        fractional_offsets: Optional float offsets with shape [B, K, 2],
            ordered as [delay_offset, doppler_offset].
        operator_mode: Active operator mode: "ongrid" or "offgrid".
        kernel_radius: Off-grid leakage kernel radius R, giving P=Q=2R+1.
        kernel_type: Off-grid leakage kernel type: "linear" or "sinc".
        normalize_kernel: Whether off-grid kernel weights are normalized.
    """

    h_dd: torch.Tensor
    support_mask: torch.Tensor
    path_indices: torch.Tensor
    path_gains: torch.Tensor
    confidence: torch.Tensor | None = None
    fractional_offsets: torch.Tensor | None = None
    operator_mode: str = "ongrid"
    kernel_radius: int = 0
    kernel_type: str = "linear"
    normalize_kernel: bool = True


def dd_circular_convolve_sparse(
    x_dd: torch.Tensor,
    path_indices: torch.Tensor,
    path_gains: torch.Tensor,
    confidence: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply sparse DD circular convolution.

    Args:
        x_dd: Complex DD input with shape [B, M, N].
        path_indices: Long path indices with shape [B, K, 2], ordered as
            [delay_idx, doppler_idx].
        path_gains: Complex path gains with shape [B, K].
        confidence: Optional float confidence with shape [B, K]. Paths with
            confidence <= 0 do not participate.

    Returns:
        Complex DD output with shape [B, M, N].
    """

    _validate_sparse_path_inputs(x_dd, path_indices, path_gains, confidence)
    y_dd = torch.zeros_like(x_dd)
    batch_size, _, _ = x_dd.shape
    num_paths = path_indices.shape[1]

    for batch_idx in range(batch_size):
        for path_idx in range(num_paths):
            if confidence is not None and confidence[batch_idx, path_idx].item() <= 0:
                continue
            delay = int(path_indices[batch_idx, path_idx, 0].item())
            doppler = int(path_indices[batch_idx, path_idx, 1].item())
            gain = path_gains[batch_idx, path_idx]
            shifted = torch.roll(x_dd[batch_idx], shifts=(delay, doppler), dims=(0, 1))
            y_dd[batch_idx] = y_dd[batch_idx] + gain * shifted
    return y_dd


def dd_circular_correlation_sparse(
    y_dd: torch.Tensor,
    path_indices: torch.Tensor,
    path_gains: torch.Tensor,
    confidence: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply sparse DD circular correlation H^H y.

    Args:
        y_dd: Complex DD input with shape [B, M, N].
        path_indices: Long path indices with shape [B, K, 2], ordered as
            [delay_idx, doppler_idx].
        path_gains: Complex path gains with shape [B, K].
        confidence: Optional float confidence with shape [B, K]. Paths with
            confidence <= 0 do not participate.

    Returns:
        Complex DD output with shape [B, M, N].
    """

    _validate_sparse_path_inputs(y_dd, path_indices, path_gains, confidence)
    x_dd = torch.zeros_like(y_dd)
    batch_size, _, _ = y_dd.shape
    num_paths = path_indices.shape[1]

    for batch_idx in range(batch_size):
        for path_idx in range(num_paths):
            if confidence is not None and confidence[batch_idx, path_idx].item() <= 0:
                continue
            delay = int(path_indices[batch_idx, path_idx, 0].item())
            doppler = int(path_indices[batch_idx, path_idx, 1].item())
            gain = torch.conj(path_gains[batch_idx, path_idx])
            shifted = torch.roll(y_dd[batch_idx], shifts=(-delay, -doppler), dims=(0, 1))
            x_dd[batch_idx] = x_dd[batch_idx] + gain * shifted
    return x_dd


def offgrid_leakage_kernel(
    offset_delay: torch.Tensor,
    offset_doppler: torch.Tensor,
    kernel_radius: int,
    kernel_type: str = "linear",
    normalize: bool = True,
) -> torch.Tensor:
    """Build differentiable approximate off-grid leakage kernel [P, Q].

    Args:
        offset_delay: Scalar tensor for fractional delay offset.
        offset_doppler: Scalar tensor for fractional Doppler offset.
        kernel_radius: Non-negative radius R, producing P=Q=2R+1.
        kernel_type: "linear" for triangular interpolation or "sinc" for a
            truncated separable sinc approximation.
        normalize: If true, normalize by the absolute kernel sum.

    Returns:
        Float leakage kernel with shape [P, Q]. This is a differentiable
        local leakage approximation for off-grid sparse DD paths, not a full
        OTFS pulse-shaping model.
    """

    if not isinstance(kernel_radius, int) or kernel_radius < 0:
        raise ValueError("kernel_radius must be a non-negative integer.")
    if kernel_type not in {"linear", "sinc"}:
        raise ValueError('kernel_type must be "linear" or "sinc".')
    device = offset_delay.device
    dtype = offset_delay.dtype if offset_delay.dtype.is_floating_point else torch.float32
    offsets = torch.arange(-kernel_radius, kernel_radius + 1, device=device, dtype=dtype)
    delay_offset = offset_delay.to(device=device, dtype=dtype)
    doppler_offset = offset_doppler.to(device=device, dtype=dtype)

    if kernel_type == "linear":
        delay_weights = torch.relu(1.0 - (offsets - delay_offset).abs())
        doppler_weights = torch.relu(1.0 - (offsets - doppler_offset).abs())
    else:
        delay_weights = torch.sinc(offsets - delay_offset)
        doppler_weights = torch.sinc(offsets - doppler_offset)

    kernel = delay_weights[:, None] * doppler_weights[None, :]
    if normalize:
        kernel = kernel / kernel.abs().sum().clamp_min(1e-12)
    return kernel


def dd_circular_convolve_offgrid_sparse(
    x_dd: torch.Tensor,
    path_indices: torch.Tensor,
    path_gains: torch.Tensor,
    fractional_offsets: torch.Tensor,
    confidence: torch.Tensor | None = None,
    kernel_radius: int = 1,
    kernel_type: str = "linear",
    normalize_kernel: bool = True,
) -> torch.Tensor:
    """Apply approximate off-grid sparse DD circular convolution.

    Args:
        x_dd: Complex DD input with shape [B, M, N].
        path_indices: Long integer path indices with shape [B, K, 2].
        path_gains: Complex path gains with shape [B, K].
        fractional_offsets: Float offsets with shape [B, K, 2], ordered as
            [delay_offset, doppler_offset].
        confidence: Optional float confidence with shape [B, K]. Paths with
            confidence <= 0 do not participate.
        kernel_radius: Non-negative leakage radius R, producing P=Q=2R+1.
        kernel_type: "linear" or "sinc" leakage approximation.
        normalize_kernel: If true, normalize local kernel weights.

    Returns:
        Complex DD output with shape [B, M, N]. No dense MN x MN matrix is
        constructed.
    """

    _validate_offgrid_path_inputs(x_dd, path_indices, path_gains, fractional_offsets, confidence)
    y_dd = torch.zeros_like(x_dd)
    batch_size = x_dd.shape[0]
    num_paths = path_indices.shape[1]
    fractional_offsets = fractional_offsets.to(device=x_dd.device, dtype=x_dd.real.dtype)

    for batch_idx in range(batch_size):
        for path_idx in range(num_paths):
            if confidence is not None and confidence[batch_idx, path_idx].item() <= 0:
                continue
            delay = int(path_indices[batch_idx, path_idx, 0].item())
            doppler = int(path_indices[batch_idx, path_idx, 1].item())
            gain = path_gains[batch_idx, path_idx]
            kernel = offgrid_leakage_kernel(
                fractional_offsets[batch_idx, path_idx, 0],
                fractional_offsets[batch_idx, path_idx, 1],
                kernel_radius=kernel_radius,
                kernel_type=kernel_type,
                normalize=normalize_kernel,
            )
            y_dd[batch_idx] = y_dd[batch_idx] + _apply_kernel_convolution_to_one(
                x_dd[batch_idx],
                delay,
                doppler,
                gain,
                kernel,
                kernel_radius,
            )
    return y_dd


def dd_circular_correlation_offgrid_sparse(
    y_dd: torch.Tensor,
    path_indices: torch.Tensor,
    path_gains: torch.Tensor,
    fractional_offsets: torch.Tensor,
    confidence: torch.Tensor | None = None,
    kernel_radius: int = 1,
    kernel_type: str = "linear",
    normalize_kernel: bool = True,
) -> torch.Tensor:
    """Apply approximate off-grid sparse DD matched filter H_theta^H y.

    Args:
        y_dd: Complex DD input with shape [B, M, N].
        path_indices: Long integer path indices with shape [B, K, 2].
        path_gains: Complex path gains with shape [B, K].
        fractional_offsets: Float offsets with shape [B, K, 2], ordered as
            [delay_offset, doppler_offset].
        confidence: Optional float confidence with shape [B, K]. Paths with
            confidence <= 0 do not participate.
        kernel_radius: Non-negative leakage radius R, producing P=Q=2R+1.
        kernel_type: "linear" or "sinc" leakage approximation.
        normalize_kernel: If true, normalize local kernel weights.

    Returns:
        Complex DD output with shape [B, M, N], implemented as the approximate
        adjoint of dd_circular_convolve_offgrid_sparse.
    """

    _validate_offgrid_path_inputs(y_dd, path_indices, path_gains, fractional_offsets, confidence)
    x_dd = torch.zeros_like(y_dd)
    batch_size = y_dd.shape[0]
    num_paths = path_indices.shape[1]
    fractional_offsets = fractional_offsets.to(device=y_dd.device, dtype=y_dd.real.dtype)

    for batch_idx in range(batch_size):
        for path_idx in range(num_paths):
            if confidence is not None and confidence[batch_idx, path_idx].item() <= 0:
                continue
            delay = int(path_indices[batch_idx, path_idx, 0].item())
            doppler = int(path_indices[batch_idx, path_idx, 1].item())
            gain = path_gains[batch_idx, path_idx]
            kernel = offgrid_leakage_kernel(
                fractional_offsets[batch_idx, path_idx, 0],
                fractional_offsets[batch_idx, path_idx, 1],
                kernel_radius=kernel_radius,
                kernel_type=kernel_type,
                normalize=normalize_kernel,
            )
            x_dd[batch_idx] = x_dd[batch_idx] + _apply_kernel_correlation_to_one(
                y_dd[batch_idx],
                delay,
                doppler,
                gain,
                kernel,
                kernel_radius,
            )
    return x_dd


def dd_circular_convolve_dense_fft(x_dd: torch.Tensor, h_dd: torch.Tensor) -> torch.Tensor:
    """Apply dense FFT baseline circular convolution.

    Args:
        x_dd: Complex DD input with shape [B, M, N].
        h_dd: Complex DD channel with shape [B, M, N].

    Returns:
        Complex DD output with shape [B, M, N].
    """

    if not torch.is_tensor(x_dd) or x_dd.ndim != 3 or not torch.is_complex(x_dd):
        raise TypeError("x_dd must be a complex tensor with shape [B, M, N].")
    if not torch.is_tensor(h_dd) or h_dd.ndim != 3 or not torch.is_complex(h_dd):
        raise TypeError("h_dd must be a complex tensor with shape [B, M, N].")
    if x_dd.shape != h_dd.shape:
        raise ValueError("x_dd and h_dd must have matching shape [B, M, N].")
    return torch.fft.ifft2(torch.fft.fft2(x_dd, dim=(-2, -1)) * torch.fft.fft2(h_dd, dim=(-2, -1)), dim=(-2, -1))


class SparseDDOperator(nn.Module):
    """Build/apply sparse DD circular convolution for tensors [B, M, N]."""

    def __init__(self, config: ReceiverConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        sparse_estimate_or_h_dd: SparseChannelEstimate | torch.Tensor,
        support_mask: torch.Tensor | None = None,
        path_indices: torch.Tensor | None = None,
        path_gains: torch.Tensor | None = None,
        confidence: torch.Tensor | None = None,
        fractional_offsets: torch.Tensor | None = None,
    ) -> SparseDDOperatorState:
        """Build operator state.

        Supported calls:
            forward(sparse_estimate), where sparse_estimate carries h_dd
                [B, M, N], support_mask [B, M, N], path_indices [B, K, 2],
                path_gains [B, K], optional confidence [B, K], and optional
                fractional_offsets [B, K, 2].
            forward(h_dd, support_mask, path_indices=None, path_gains=None,
                confidence=None, fractional_offsets=None), where h_dd is
                complex [B, M, N].

        Returns:
            SparseDDOperatorState with h_dd [B, M, N], support_mask [B, M, N],
            path_indices [B, K, 2], path_gains [B, K], and optional confidence
            [B, K]. If off-grid mode is active, the state also carries
            fractional_offsets [B, K, 2].
        """

        if isinstance(sparse_estimate_or_h_dd, SparseChannelEstimate):
            estimate = sparse_estimate_or_h_dd
            return self._build_state(
                estimate.h_dd,
                estimate.support_mask,
                estimate.path_indices,
                estimate.path_gains,
                estimate.confidence,
                estimate.fractional_offsets,
            )

        h_dd = sparse_estimate_or_h_dd
        if support_mask is None:
            raise ValueError("support_mask is required when calling SparseDDOperator with h_dd.")
        return self._build_state(h_dd, support_mask, path_indices, path_gains, confidence, fractional_offsets)

    def apply(self, x_dd: torch.Tensor, state: SparseDDOperatorState) -> torch.Tensor:
        """Apply sparse DD convolution to x_dd [B, M, N], returning [B, M, N]."""

        validate_complex_dd(x_dd, self.config, "x_dd")
        if state.operator_mode == "offgrid":
            return dd_circular_convolve_offgrid_sparse(
                x_dd,
                state.path_indices,
                state.path_gains,
                state.fractional_offsets,
                confidence=state.confidence,
                kernel_radius=state.kernel_radius,
                kernel_type=state.kernel_type,
                normalize_kernel=state.normalize_kernel,
            )
        return dd_circular_convolve_sparse(
            x_dd,
            state.path_indices,
            state.path_gains,
            confidence=state.confidence,
        )

    def matched_filter(self, y_dd: torch.Tensor, state: SparseDDOperatorState) -> torch.Tensor:
        """Apply sparse DD matched filter to y_dd [B, M, N], returning [B, M, N]."""

        validate_complex_dd(y_dd, self.config, "y_dd")
        if state.operator_mode == "offgrid":
            return dd_circular_correlation_offgrid_sparse(
                y_dd,
                state.path_indices,
                state.path_gains,
                state.fractional_offsets,
                confidence=state.confidence,
                kernel_radius=state.kernel_radius,
                kernel_type=state.kernel_type,
                normalize_kernel=state.normalize_kernel,
            )
        return dd_circular_correlation_sparse(
            y_dd,
            state.path_indices,
            state.path_gains,
            confidence=state.confidence,
        )

    def _build_state(
        self,
        h_dd: torch.Tensor,
        support_mask: torch.Tensor,
        path_indices: torch.Tensor | None,
        path_gains: torch.Tensor | None,
        confidence: torch.Tensor | None,
        fractional_offsets: torch.Tensor | None,
    ) -> SparseDDOperatorState:
        """Build state from h_dd [B, M, N] and optional sparse paths."""

        validate_complex_dd(h_dd, self.config, "h_dd")
        validate_support_mask(support_mask, self.config, "support_mask")
        if support_mask.shape[0] != h_dd.shape[0]:
            raise ValueError("support_mask batch size must match h_dd.")
        support = support_mask.to(device=h_dd.device, dtype=h_dd.real.dtype)
        h_sparse = h_dd * support

        if path_indices is None or path_gains is None:
            path_indices, path_gains, confidence = self._extract_paths_from_mask(h_sparse, support)
        else:
            path_indices = path_indices.to(device=h_dd.device, dtype=torch.long)
            path_gains = path_gains.to(device=h_dd.device, dtype=h_dd.dtype)
            _validate_path_metadata(h_dd, path_indices, path_gains, confidence)
            if confidence is not None:
                confidence = confidence.to(device=h_dd.device, dtype=h_dd.real.dtype)
        fractional_offsets = self._prepare_fractional_offsets(h_dd, path_indices, fractional_offsets)
        operator_mode = self._resolve_operator_mode(fractional_offsets)

        return SparseDDOperatorState(
            h_dd=h_sparse,
            support_mask=support,
            path_indices=path_indices,
            path_gains=path_gains,
            confidence=confidence,
            fractional_offsets=fractional_offsets,
            operator_mode=operator_mode,
            kernel_radius=self.config.offgrid_kernel_radius if operator_mode == "offgrid" else 0,
            kernel_type=self.config.offgrid_kernel_type,
            normalize_kernel=self.config.offgrid_normalize_kernel,
        )

    def _resolve_operator_mode(self, fractional_offsets: torch.Tensor | None) -> str:
        """Resolve DD operator mode for fractional_offsets [B, K, 2]."""

        mode = self.config.dd_operator_mode
        if mode == "ongrid":
            return "ongrid"
        if mode == "offgrid":
            if fractional_offsets is None:
                raise ValueError('fractional_offsets are required when dd_operator_mode="offgrid".')
            return "offgrid"
        if fractional_offsets is not None:
            return "offgrid"
        return "ongrid"

    def _prepare_fractional_offsets(
        self,
        h_dd: torch.Tensor,
        path_indices: torch.Tensor,
        fractional_offsets: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Validate optional fractional_offsets [B, K, 2]."""

        if fractional_offsets is None:
            return None
        _validate_fractional_offsets(path_indices, fractional_offsets)
        return fractional_offsets.to(device=h_dd.device, dtype=h_dd.real.dtype)

    def _extract_paths_from_mask(
        self,
        h_dd: torch.Tensor,
        support_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract up to topk_paths nonzero paths from h_dd [B, M, N]."""

        batch_size = h_dd.shape[0]
        requested_k = self.config.topk_paths
        actual_k = min(requested_k, self.config.M * self.config.N)
        flat_abs = (h_dd.abs() * support_mask).reshape(batch_size, -1)
        values, flat_indices = torch.topk(flat_abs, k=actual_k, dim=-1)
        valid = values > 0
        if actual_k < requested_k:
            pad_count = requested_k - actual_k
            values = torch.nn.functional.pad(values, (0, pad_count))
            flat_indices = torch.nn.functional.pad(flat_indices, (0, pad_count))
            valid = torch.nn.functional.pad(valid, (0, pad_count), value=False)

        flat_indices = torch.where(valid, flat_indices, torch.zeros_like(flat_indices))
        delay_idx = torch.div(flat_indices, self.config.N, rounding_mode="floor")
        doppler_idx = flat_indices.remainder(self.config.N)
        path_indices = torch.stack((delay_idx, doppler_idx), dim=-1).long()
        batch = torch.arange(batch_size, device=h_dd.device).unsqueeze(-1)
        path_gains = h_dd[batch, path_indices[..., 0], path_indices[..., 1]]
        path_gains = torch.where(valid, path_gains, torch.zeros_like(path_gains))
        confidence = valid.to(dtype=h_dd.real.dtype)
        return path_indices, path_gains, confidence


def _validate_sparse_path_inputs(
    x_dd: torch.Tensor,
    path_indices: torch.Tensor,
    path_gains: torch.Tensor,
    confidence: torch.Tensor | None,
) -> None:
    """Validate sparse operator inputs x_dd [B, M, N], paths [B, K, 2]."""

    if not torch.is_tensor(x_dd) or x_dd.ndim != 3 or not torch.is_complex(x_dd):
        raise TypeError("x_dd must be a complex tensor with shape [B, M, N].")
    _validate_path_metadata(x_dd, path_indices, path_gains, confidence)


def _validate_offgrid_path_inputs(
    x_dd: torch.Tensor,
    path_indices: torch.Tensor,
    path_gains: torch.Tensor,
    fractional_offsets: torch.Tensor,
    confidence: torch.Tensor | None,
) -> None:
    """Validate off-grid sparse inputs x_dd [B, M, N], offsets [B, K, 2]."""

    _validate_sparse_path_inputs(x_dd, path_indices, path_gains, confidence)
    _validate_fractional_offsets(path_indices, fractional_offsets)


def _validate_fractional_offsets(path_indices: torch.Tensor, fractional_offsets: torch.Tensor) -> None:
    """Validate fractional_offsets [B, K, 2] matching path_indices [B, K, 2]."""

    if not torch.is_tensor(fractional_offsets):
        raise ValueError("fractional_offsets must be a tensor with shape [B, K, 2].")
    if fractional_offsets.shape != path_indices.shape:
        raise ValueError("fractional_offsets must have shape [B, K, 2] matching path_indices.")
    if not fractional_offsets.dtype.is_floating_point:
        raise TypeError("fractional_offsets must be floating point with shape [B, K, 2].")


def _validate_path_metadata(
    reference_dd: torch.Tensor,
    path_indices: torch.Tensor,
    path_gains: torch.Tensor,
    confidence: torch.Tensor | None,
) -> None:
    """Validate path_indices [B, K, 2], path_gains [B, K], confidence [B, K]."""

    if not torch.is_tensor(path_indices) or path_indices.ndim != 3 or path_indices.shape[-1] != 2:
        raise ValueError("path_indices must have shape [B, K, 2].")
    if path_indices.shape[0] != reference_dd.shape[0]:
        raise ValueError("path_indices batch size must match DD tensor.")
    if not torch.is_tensor(path_gains) or path_gains.ndim != 2:
        raise ValueError("path_gains must have shape [B, K].")
    if path_gains.shape != path_indices.shape[:2]:
        raise ValueError("path_gains must have shape [B, K] matching path_indices.")
    if not torch.is_complex(path_gains):
        raise TypeError("path_gains must be complex with shape [B, K].")
    if confidence is not None:
        if not torch.is_tensor(confidence) or confidence.shape != path_gains.shape:
            raise ValueError("confidence must have shape [B, K].")


def _apply_kernel_convolution_to_one(
    x_dd: torch.Tensor,
    delay: int,
    doppler: int,
    gain: torch.Tensor,
    kernel: torch.Tensor,
    kernel_radius: int,
) -> torch.Tensor:
    """Apply one off-grid path kernel to x_dd [M, N], returning [M, N]."""

    y_dd = torch.zeros_like(x_dd)
    for delay_pos in range(kernel.shape[0]):
        delay_shift = delay + delay_pos - kernel_radius
        for doppler_pos in range(kernel.shape[1]):
            doppler_shift = doppler + doppler_pos - kernel_radius
            weight = kernel[delay_pos, doppler_pos].to(device=x_dd.device, dtype=x_dd.real.dtype)
            shifted = torch.roll(x_dd, shifts=(delay_shift, doppler_shift), dims=(0, 1))
            y_dd = y_dd + gain * weight.to(dtype=x_dd.dtype) * shifted
    return y_dd


def _apply_kernel_correlation_to_one(
    y_dd: torch.Tensor,
    delay: int,
    doppler: int,
    gain: torch.Tensor,
    kernel: torch.Tensor,
    kernel_radius: int,
) -> torch.Tensor:
    """Apply one off-grid adjoint kernel to y_dd [M, N], returning [M, N]."""

    x_dd = torch.zeros_like(y_dd)
    for delay_pos in range(kernel.shape[0]):
        delay_shift = delay + delay_pos - kernel_radius
        for doppler_pos in range(kernel.shape[1]):
            doppler_shift = doppler + doppler_pos - kernel_radius
            weight = kernel[delay_pos, doppler_pos].to(device=y_dd.device, dtype=y_dd.real.dtype)
            shifted = torch.roll(y_dd, shifts=(-delay_shift, -doppler_shift), dims=(0, 1))
            x_dd = x_dd + torch.conj(gain * weight.to(dtype=y_dd.dtype)) * shifted
    return x_dd
