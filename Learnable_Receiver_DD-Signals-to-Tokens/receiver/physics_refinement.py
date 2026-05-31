"""Physics-guided sparse path gain re-estimation for DD channel estimates."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .config import ReceiverConfig
from .dd_ops import (
    dd_circular_convolve_offgrid_sparse,
    dd_circular_convolve_sparse,
)


@dataclass
class PhysicsRefinementOutput:
    """Output of physics-guided sparse path gain refinement.

    Attributes:
        path_gains: Complex refined path gains with shape [B, K].
        residual: Complex full-grid fitting residual with shape [B, M, N].
        residual_power: Float fit-masked residual power with shape [B].
            This is computed only over the CE fitting mask region and is
            the value used for confidence updates.  Kept for backward
            compatibility; prefer ``fit_residual_power`` for clarity.
        fit_residual_power: Float fit-masked residual power with shape [B].
            This is the per-sample mean of |residual|^2 weighted by the
            CE fitting mask, i.e. sum(fit_mask * |residual|^2) / sum(fit_mask).
            Used for path confidence updates.
        full_residual_power: Float full-grid residual power with shape [B].
            Diagnostic only — mean of |residual|^2 over the entire DD grid.
            Never used for confidence computation.
        gain_covariance_diag: Float approximate covariance diagonal with
            shape [B, K].
        confidence: Float updated path confidence with shape [B, K].
        normal_matrix_diag: Float normal-matrix diagonal with shape [B, K].
    """

    path_gains: torch.Tensor
    residual: torch.Tensor
    residual_power: torch.Tensor
    fit_residual_power: torch.Tensor | None = None
    full_residual_power: torch.Tensor | None = None
    gain_covariance_diag: torch.Tensor | None = None
    confidence: torch.Tensor | None = None
    normal_matrix_diag: torch.Tensor | None = None


def build_path_dictionary(
    y_shape: tuple[int, int, int],
    path_indices: torch.Tensor,
    fractional_offsets: torch.Tensor | None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.complex64,
    kernel_radius: int = 1,
    kernel_type: str = "linear",
    normalize_kernel: bool = True,
    probe_position: tuple[int, int] | None = None,
    probe_value: complex = 1.0 + 0.0j,
) -> torch.Tensor:
    """Build approximate path dictionary [B, M*N, K].

    Each column is the sparse DD operator response when a single path acts on
    a unit impulse placed at ``probe_position`` (default (0,0) for backward
    compatibility).  For embedded-pilot CE the probe position must match the
    known pilot location so the dictionary columns correctly model how each
    sparse path transforms the pilot impulse.

    Args:
        y_shape: Tuple (B, M, N) for the DD observation grid.
        path_indices: Long path indices with shape [B, K, 2].
        fractional_offsets: Optional float offsets with shape [B, K, 2].
        device: Optional torch device for the dictionary.
        dtype: Complex dtype for dictionary entries.
        kernel_radius: Off-grid leakage kernel radius.
        kernel_type: Off-grid leakage kernel type, "linear" or "sinc".
        normalize_kernel: If true, normalize off-grid leakage kernels.
        probe_position: Optional (delay, doppler) index of the probe impulse.
            If None, defaults to (0, 0).
        probe_value: Complex scalar written at the probe position.

    Returns:
        Complex dictionary tensor with shape [B, M*N, K].
    """

    if len(y_shape) != 3:
        raise ValueError("y_shape must be a tuple (B, M, N).")
    batch_size, delay_bins, doppler_bins = y_shape
    if not torch.is_tensor(path_indices) or path_indices.shape[:1] != (batch_size,) or path_indices.ndim != 3:
        raise ValueError("path_indices must have shape [B, K, 2].")
    if path_indices.shape[-1] != 2:
        raise ValueError("path_indices must have shape [B, K, 2].")
    if not dtype.is_complex:
        raise TypeError("dtype must be a complex dtype.")
    resolved_device = path_indices.device if device is None else device
    path_indices = path_indices.to(device=resolved_device, dtype=torch.long)
    if fractional_offsets is not None:
        if fractional_offsets.shape != path_indices.shape or not fractional_offsets.dtype.is_floating_point:
            raise ValueError("fractional_offsets must have float shape [B, K, 2].")
        fractional_offsets = fractional_offsets.to(device=resolved_device)

    if probe_position is None:
        probe_delay, probe_doppler = 0, 0
    else:
        probe_delay, probe_doppler = probe_position
    if not (0 <= probe_delay < delay_bins and 0 <= probe_doppler < doppler_bins):
        raise ValueError(
            f"probe_position ({probe_delay}, {probe_doppler}) is outside DD grid [0..{delay_bins-1}, 0..{doppler_bins-1}]."
        )

    probe_value_tensor = torch.as_tensor(probe_value, device=resolved_device, dtype=dtype)
    probe = torch.zeros(batch_size, delay_bins, doppler_bins, device=resolved_device, dtype=dtype)
    probe[:, probe_delay, probe_doppler] = probe_value_tensor
    num_paths = path_indices.shape[1]
    columns: list[torch.Tensor] = []
    unit_gain = torch.ones(batch_size, 1, device=resolved_device, dtype=dtype)
    unit_confidence = torch.ones(batch_size, 1, device=resolved_device, dtype=probe.real.dtype)
    for path_idx in range(num_paths):
        one_path = path_indices[:, path_idx : path_idx + 1]
        if fractional_offsets is None:
            response = dd_circular_convolve_sparse(
                probe,
                one_path,
                unit_gain,
                confidence=unit_confidence,
            )
        else:
            response = dd_circular_convolve_offgrid_sparse(
                probe,
                one_path,
                unit_gain,
                fractional_offsets[:, path_idx : path_idx + 1],
                confidence=unit_confidence,
                kernel_radius=kernel_radius,
                kernel_type=kernel_type,
                normalize_kernel=normalize_kernel,
            )
        columns.append(response.reshape(batch_size, -1))
    return torch.stack(columns, dim=-1)


class PhysicsGuidedPathGainRefiner(nn.Module):
    """Approximate physics-guided sparse path gain LS/MMSE refiner.

    Inputs:
        y_dd: Complex observation with shape [B, M, N].
        path_indices: Long path indices with shape [B, K, 2].
        path_gains: Complex initial path gains with shape [B, K].
        confidence: Optional float confidence with shape [B, K].
        fractional_offsets: Optional float offsets with shape [B, K, 2].
        fit_mask: Optional real/bool CE fitting mask with shape [B, M, N] or
            [1, M, N].  Only positions where fit_mask > 0 participate in the
            LS normal equations.  This must be the CE *observation* region
            (pilot window), not the token data region.
        data_mask: Deprecated alias for ``fit_mask``.  Prefer ``fit_mask``.
        probe_position: Optional (delay, doppler) tuple for the known pilot
            impulse position used by ``build_path_dictionary``.  If None,
            defaults to (0, 0) for backward compatibility.
        probe_value: Complex pilot scalar at the probe position.

    Output:
        PhysicsRefinementOutput with refined path_gains [B, K], residual
        [B, M, N], residual_power [B], covariance/normal diagonals [B, K],
        and confidence [B, K].

    Note:
        This module performs small-K ridge LS/MMSE gain re-estimation with an
        approximate leakage-kernel sparse DD dictionary.  It does not use
        ground-truth channel labels, does not construct a dense MN×MN matrix,
        and must only fit against known CE observation regions.
    """

    def __init__(self, config: ReceiverConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        y_dd: torch.Tensor,
        path_indices: torch.Tensor,
        path_gains: torch.Tensor,
        confidence: torch.Tensor | None = None,
        fractional_offsets: torch.Tensor | None = None,
        fit_mask: torch.Tensor | None = None,
        data_mask: torch.Tensor | None = None,
        probe_position: tuple[int, int] | None = None,
        probe_value: complex = 1.0 + 0.0j,
    ) -> PhysicsRefinementOutput:
        """Refine path_gains [B, K] from y_dd [B, M, N].

        Args:
            y_dd: Complex observation with shape [B, M, N].
            path_indices: Long path indices with shape [B, K, 2].
            path_gains: Complex initial path gains with shape [B, K].
            confidence: Optional float confidence with shape [B, K].
            fractional_offsets: Optional float offsets with shape [B, K, 2].
            fit_mask: Optional real CE fitting mask with shape [B, M, N] or
                [1, M, N].  Restricts LS normal equations to known observation
                positions (e.g. pilot window).  Takes priority over data_mask.
            data_mask: Deprecated alias for ``fit_mask``.
            probe_position: Optional (delay, doppler) of the pilot impulse.
            probe_value: Complex pilot scalar.

        Returns:
            PhysicsRefinementOutput.
        """

        _validate_inputs(y_dd, path_indices, path_gains, confidence, fractional_offsets)
        batch_size, delay_bins, doppler_bins = y_dd.shape

        resolved_fit_mask = _resolve_fit_mask(fit_mask, data_mask, y_dd)

        dictionary = build_path_dictionary(
            y_dd.shape,
            path_indices,
            fractional_offsets,
            device=y_dd.device,
            dtype=y_dd.dtype,
            kernel_radius=self.config.offgrid_kernel_radius,
            kernel_type=self.config.offgrid_kernel_type,
            normalize_kernel=self.config.offgrid_normalize_kernel,
            probe_position=probe_position,
            probe_value=probe_value,
        )
        fit_mask_flat = resolved_fit_mask.reshape(batch_size, -1)
        weighted_a = dictionary * fit_mask_flat.unsqueeze(-1).to(dtype=y_dd.dtype)
        weighted_y = y_dd.reshape(batch_size, -1) * fit_mask_flat.to(dtype=y_dd.dtype)

        refined_gains = []
        residuals = []
        fit_residual_powers = []
        full_residual_powers = []
        normal_diags = []
        covariance_diags = []
        confidence_updates = []
        ridge = float(self.config.physics_refinement_ridge)
        min_confidence = float(self.config.physics_refinement_min_confidence)
        eye = torch.eye(path_indices.shape[1], device=y_dd.device, dtype=y_dd.dtype)
        for batch_idx in range(batch_size):
            a_b = weighted_a[batch_idx]
            y_b = weighted_y[batch_idx]
            initial_gain = path_gains[batch_idx]
            active = _active_paths(confidence, batch_idx, initial_gain, min_confidence)
            column_scale = active.to(device=y_dd.device, dtype=y_dd.real.dtype)
            a_scaled = a_b * column_scale.to(dtype=y_dd.dtype).unsqueeze(0)
            normal = a_scaled.conj().transpose(0, 1) @ a_scaled + ridge * eye
            rhs = a_scaled.conj().transpose(0, 1) @ y_b + ridge * initial_gain
            solved = torch.linalg.solve(normal, rhs.unsqueeze(-1)).squeeze(-1)
            refined = torch.where(active, solved, initial_gain)
            fitted = dictionary[batch_idx] @ refined
            residual = y_dd[batch_idx].reshape(-1) - fitted

            # Fit-masked residual power: only over CE observation region
            mask_b = resolved_fit_mask[batch_idx].reshape(-1)
            mask_sum = mask_b.sum().clamp_min(1e-8)
            fit_res_pow = (mask_b * residual.abs().pow(2)).sum() / mask_sum

            # Full-grid residual power: diagnostic only, not used for confidence
            full_res_pow = residual.abs().pow(2).mean()

            normal_diag = normal.diagonal().real.clamp_min(ridge)
            covariance_diag = normal_diag.reciprocal()
            gain_power = refined.abs().pow(2)
            conf = gain_power / (gain_power + fit_res_pow + covariance_diag + 1e-8)
            if confidence is not None:
                base_conf = confidence[batch_idx].to(device=y_dd.device, dtype=y_dd.real.dtype).clamp(0.0, 1.0)
                conf = torch.where(active, conf, 0.5 * base_conf)
            else:
                conf = torch.where(active, conf, torch.zeros_like(conf))
            refined_gains.append(refined)
            residuals.append(residual.reshape(delay_bins, doppler_bins))
            fit_residual_powers.append(fit_res_pow)
            full_residual_powers.append(full_res_pow)
            normal_diags.append(normal_diag)
            covariance_diags.append(covariance_diag)
            confidence_updates.append(conf.clamp(0.0, 1.0))

        residual_map = torch.stack(residuals, dim=0)
        fit_pow = torch.stack(fit_residual_powers, dim=0)
        full_pow = torch.stack(full_residual_powers, dim=0)
        return PhysicsRefinementOutput(
            path_gains=torch.stack(refined_gains, dim=0),
            residual=residual_map,
            residual_power=fit_pow,
            fit_residual_power=fit_pow,
            full_residual_power=full_pow,
            gain_covariance_diag=torch.stack(covariance_diags, dim=0),
            confidence=torch.stack(confidence_updates, dim=0),
            normal_matrix_diag=torch.stack(normal_diags, dim=0),
        )


def _validate_inputs(
    y_dd: torch.Tensor,
    path_indices: torch.Tensor,
    path_gains: torch.Tensor,
    confidence: torch.Tensor | None,
    fractional_offsets: torch.Tensor | None,
) -> None:
    if not torch.is_tensor(y_dd) or y_dd.ndim != 3 or not torch.is_complex(y_dd):
        raise TypeError("y_dd must be complex with shape [B, M, N].")
    if not torch.is_tensor(path_indices) or path_indices.ndim != 3 or path_indices.shape[-1] != 2:
        raise ValueError("path_indices must have shape [B, K, 2].")
    if path_indices.shape[0] != y_dd.shape[0]:
        raise ValueError("path_indices batch size must match y_dd.")
    if not torch.is_tensor(path_gains) or path_gains.shape != path_indices.shape[:2] or not torch.is_complex(path_gains):
        raise ValueError("path_gains must be complex with shape [B, K].")
    if confidence is not None and (not torch.is_tensor(confidence) or confidence.shape != path_gains.shape):
        raise ValueError("confidence must have shape [B, K].")
    if fractional_offsets is not None:
        if fractional_offsets.shape != path_indices.shape or not fractional_offsets.dtype.is_floating_point:
            raise ValueError("fractional_offsets must have float shape [B, K, 2].")


def _resolve_fit_mask(
    fit_mask: torch.Tensor | None,
    data_mask: torch.Tensor | None,
    y_dd: torch.Tensor,
) -> torch.Tensor:
    """Resolve CE fitting mask from fit_mask (preferred) or data_mask (legacy).

    Returns a real tensor with shape [B, M, N] clamped to [0, 1].  If neither
    is supplied, returns an all-ones mask (full-grid fallback — caller must
    decide whether this is safe).
    """
    mask = fit_mask if fit_mask is not None else data_mask
    if mask is None:
        return torch.ones_like(y_dd.real)
    if not torch.is_tensor(mask) or mask.ndim != 3 or torch.is_complex(mask):
        raise ValueError("fit_mask/data_mask must have real shape [B, M, N] or [1, M, N].")
    if mask.shape[-2:] != y_dd.shape[-2:] or mask.shape[0] not in {1, y_dd.shape[0]}:
        raise ValueError("fit_mask/data_mask must have real shape [B, M, N] or [1, M, N].")
    resolved = mask.to(device=y_dd.device, dtype=y_dd.real.dtype).clamp(0.0, 1.0)
    if resolved.shape[0] == 1 and y_dd.shape[0] != 1:
        resolved = resolved.expand(y_dd.shape[0], -1, -1)
    return resolved


def _active_paths(
    confidence: torch.Tensor | None,
    batch_idx: int,
    path_gains: torch.Tensor,
    min_confidence: float,
) -> torch.Tensor:
    if confidence is None:
        return path_gains.abs() > 0
    return confidence[batch_idx].to(device=path_gains.device, dtype=path_gains.real.dtype) > min_confidence
