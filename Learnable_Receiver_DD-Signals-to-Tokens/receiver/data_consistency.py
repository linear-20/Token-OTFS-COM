"""Data-consistency correction block (Step 12).

Placed between the Residual DD Denoiser and the Token Embedding Classifier,
this module takes one model-driven gradient step that reduces the residual
Y - H(x) using only the sparse DD operator.  It is identity-initialised so
the default forward pass is unchanged.

This is a weighted data-consistency gradient step, NOT a range-space
projection.  The observation-domain weighting (for the LS gradient) and the
x-domain update gate (for the correction magnitude) are cleanly separated.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .complex_utils import validate_complex_dd
from .config import ReceiverConfig
from .dd_ops import SparseDDOperator, SparseDDOperatorState


@dataclass
class DataConsistencyOutput:
    """Step 12 correction output.

    Attributes:
        x_corrected: Complex DD tensor after correction [B, M, N].
        correction: Complex correction term applied [B, M, N].
        residual: Complex data residual y_hat - y_dd [B, M, N].
        gradient: Complex matched-filtered weighted residual [B, M, N].
        step_size: Scalar learnable step size (float tensor).
        observation_weight: Real per-position weight used to form the
            weighted LS residual before matched_filter [B, M, N].
            Formula: data_mask * reliab / (1 + uncert).
        update_gate: Real per-position gate applied to the gradient step
            [B, M, N].  Scalar learnable gate broadcast spatially,
            optionally multiplied by reliability_map.
        correction_gate: Backward-compatible alias for ``update_gate``.
        residual_energy: Real per-position residual energy [B, M, N].
    """

    x_corrected: torch.Tensor
    correction: torch.Tensor
    residual: torch.Tensor
    gradient: torch.Tensor
    step_size: torch.Tensor
    observation_weight: torch.Tensor | None = None
    update_gate: torch.Tensor | None = None
    correction_gate: torch.Tensor | None = None
    residual_energy: torch.Tensor | None = None


class DataConsistencyCorrection(nn.Module):
    """Weighted data-consistency gradient step for x_refined [B, M, N].

    Produces one model-driven update with separated obs-weight and update-gate:

        y_hat     = H(x)
        r         = y_hat - y
        obs_w     = data_mask * reliab / (1 + uncert)
        r_w       = obs_w * r
        g         = H^H(r_w)              -- weighted LS gradient
        gate      = scalar_gate            -- learnable x-domain gate
        correction = step * gate * g
        x_out     = x - correction

    where H is the sparse DD operator ``apply`` / ``matched_filter``.
    The step size is learnable and initialised near zero so the block
    defaults to identity.  The observation weight ``obs_w`` is only used
    inside H^H(w*r); it does NOT appear in the x-domain update gate.

    If ``config.dc_gate_uses_reliability`` is True, ``gate`` is multiplied
    by ``reliability_map`` (a heuristic to softly disable the correction
    in low-confidence regions).  This is a training-time knob separate
    from the LS gradient weighting.

    Learnable parameters:
        raw_step (scalar) -- after softplus * dc_step_scale = step_size
        raw_gate (scalar) -- after sigmoid = learnable per-position gate
    """

    def __init__(self, config: ReceiverConfig, dd_operator: SparseDDOperator):
        super().__init__()
        self.config = config
        self.dd_operator = dd_operator

        init_step = float(config.dc_step_init)
        if init_step <= 0:
            raw_init = -10.0
        else:
            import math
            raw_init = math.log(math.exp(init_step / max(config.dc_step_scale, 1e-8)) - 1.0)
        self.raw_step = nn.Parameter(torch.tensor(raw_init))

        gate_init = float(config.dc_gate_floor)
        if gate_init <= 0.0:
            raw_gate_init = -5.0
        else:
            import math
            raw_gate_init = math.log(gate_init / (1.0 - gate_init))
        self.raw_gate = nn.Parameter(torch.tensor(raw_gate_init))

    @property
    def step_size(self) -> torch.Tensor:
        scale = float(self.config.dc_step_scale)
        return torch.nn.functional.softplus(self.raw_step) * scale

    @property
    def gate_value(self) -> torch.Tensor:
        """Scalar learnable update gate."""
        floor = float(self.config.dc_gate_floor)
        return torch.sigmoid(self.raw_gate) * (1.0 - floor) + floor

    def forward(
        self,
        x_dd: torch.Tensor,
        y_dd: torch.Tensor,
        operator_state: SparseDDOperatorState,
        data_mask: torch.Tensor | None = None,
        reliability_map: torch.Tensor | None = None,
        uncertainty_map: torch.Tensor | None = None,
        return_details: bool = False,
    ) -> torch.Tensor | DataConsistencyOutput:
        """Apply one weighted data-consistency gradient correction.

        Args:
            x_dd: Complex input DD tensor [B, M, N] (from denoiser).
            y_dd: Complex received DD grid [B, M, N].
            operator_state: SparseDDOperatorState with paths and gains.
            data_mask: Optional real [B, M, N] or [1, M, N].
            reliability_map: Optional real [B, M, N] or [1, M, N].
            uncertainty_map: Optional real [B, M, N] or [1, M, N].
            return_details: If true, return DataConsistencyOutput.

        Returns:
            Complex x_corrected [B, M, N] or DataConsistencyOutput.
        """
        validate_complex_dd(x_dd, self.config, "x_dd")
        validate_complex_dd(y_dd, self.config, "y_dd")

        device = x_dd.device
        real_dtype = x_dd.real.dtype

        # ---- residual: y_hat = H(x), r = y_hat - y ----
        y_hat = self.dd_operator.apply(x_dd, operator_state)
        residual = y_hat - y_dd

        # ---- observation-domain weight (for LS gradient only) ----
        dm = _resolve_map(data_mask, x_dd, default=1.0)
        rel = _resolve_map(reliability_map, x_dd, default=1.0)
        unc = _resolve_map(uncertainty_map, x_dd, default=0.0).clamp_min(0.0)

        if not self.config.dc_use_reliability:
            rel = torch.ones_like(rel)
        if not self.config.dc_use_uncertainty:
            unc = torch.zeros_like(unc)

        obs_weight = dm * rel / (1.0 + unc)

        # ---- weighted LS gradient via matched filter ----
        weighted_residual = obs_weight * residual
        gradient = self.dd_operator.matched_filter(weighted_residual, operator_state)

        # ---- x-domain update gate (scalar) ----
        scalar_gate = self.gate_value.to(device)
        gate_map = scalar_gate.expand_as(x_dd.real)

        if getattr(self.config, "dc_gate_uses_reliability", False):
            gate_map = gate_map * rel

        # ---- correction (no obs_weight here) ----
        step = self.step_size.to(device)
        correction = step * gate_map * gradient
        x_corrected = x_dd - correction

        residual_energy = residual.abs().pow(2)

        if not return_details:
            return x_corrected

        return DataConsistencyOutput(
            x_corrected=x_corrected,
            correction=correction,
            residual=residual,
            gradient=gradient,
            step_size=step.detach(),
            observation_weight=obs_weight,
            update_gate=gate_map,
            correction_gate=gate_map,
            residual_energy=residual_energy,
        )


def _resolve_map(
    value: torch.Tensor | None,
    reference: torch.Tensor,
    default: float,
) -> torch.Tensor:
    if value is None:
        return torch.full_like(reference.real, float(default))
    if not torch.is_tensor(value) or value.ndim != 3 or torch.is_complex(value):
        raise ValueError("map must have real shape [B, M, N] or [1, M, N].")
    if value.shape[-2:] != reference.shape[-2:] or value.shape[0] not in {1, reference.shape[0]}:
        raise ValueError("map spatial dims must match reference [B, M, N].")
    resolved = value.to(device=reference.device, dtype=reference.real.dtype)
    if resolved.shape[0] == 1 and reference.shape[0] != 1:
        resolved = resolved.expand(reference.shape[0], -1, -1)
    return resolved
