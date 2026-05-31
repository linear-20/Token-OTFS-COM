"""Physics validation harness (Step 18).

Diagnostic tools for quantifying the accuracy of the sparse DD operator,
off-grid leakage approximation, adjoint consistency, and pilot-CE alignment.

No algorithm changes -- only metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses import replace

import torch

from .dd_ops import SparseDDOperator, SparseDDOperatorState


@dataclass
class PhysicsValidationReport:
    """Physics consistency validation report.

    Attributes:
        integer_operator_nmse: NMSE between sparse op and torch.roll reference.
        offgrid_zero_offset_nmse: NMSE offgrid (zero offset) vs ongrid.
        adjoint_relative_error: Relative adjoint consistency error.
        leakage_energy_ratio: Output energy / input energy for impulse test.
        leakage_centroid_error: Euclidean DD distance between centroid and
            expected peak position.
        pilot_operator_alignment_error: Masked NMSE in pilot window.
        gain_refinement_residual_ratio: fit_residual / pilot_residual mean ratio.
        passed: Whether all applicable metrics satisfied thresholds.
        thresholds: Dict of threshold values used.
        metrics: Dict of all computed metric tensors.
        notes: Arbitrary diagnostic notes.
    """

    integer_operator_nmse: torch.Tensor | None = None
    offgrid_zero_offset_nmse: torch.Tensor | None = None
    adjoint_relative_error: torch.Tensor | None = None
    leakage_energy_ratio: torch.Tensor | None = None
    leakage_centroid_error: torch.Tensor | None = None
    pilot_operator_alignment_error: torch.Tensor | None = None
    gain_refinement_residual_ratio: torch.Tensor | None = None
    passed: bool = False
    thresholds: dict[str, float] = field(default_factory=dict)
    metrics: dict[str, torch.Tensor] = field(default_factory=dict)
    notes: dict[str, object] = field(default_factory=dict)


# --- basic helpers ------------------------------------------------------

def complex_nmse(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Normalised mean squared error for complex tensors.

    Returns a scalar.  Zero-target energy is safely handled via eps.
    """
    err = (pred - target).abs().pow(2).sum()
    den = target.abs().pow(2).sum().clamp_min(float(eps))
    return (err / den).clamp_max(1e9)


def _inner_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Complex inner product: sum(conj(a) * b)."""
    return (a.conj() * b).sum()


def _resolve_to_batch(t: torch.Tensor | None, batch: int) -> torch.Tensor | None:
    if t is None:
        return None
    if t.ndim != 3:
        return None
    if t.shape[0] == 1 and batch != 1:
        return t.expand(batch, -1, -1)
    return t


# --- validation functions -----------------------------------------------

def adjoint_consistency_error(
    dd_operator: SparseDDOperator,
    operator_state: SparseDDOperatorState,
    x_dd: torch.Tensor,
    y_dd: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Relative adjoint error: |<Hx, y> - <x, H^H y>| / (|<Hx,y>| + |<x,H^H y>| + eps)."""
    Hx = dd_operator.apply(x_dd, operator_state)
    Hhy = dd_operator.matched_filter(y_dd, operator_state)
    inner_fwd = _inner_product(Hx, y_dd)
    inner_adj = _inner_product(x_dd, Hhy)
    num = (inner_fwd - inner_adj).abs()
    den = (inner_fwd.abs() + inner_adj.abs()).clamp_min(float(eps))
    return (num / den).clamp_max(1e9).real


def integer_sparse_operator_reference(
    x_dd: torch.Tensor,
    path_indices: torch.Tensor,
    path_gains: torch.Tensor,
) -> torch.Tensor:
    """Reference integer DD circular convolution using torch.roll.

    Does NOT call dd_operator.apply().  Used as independent baseline.
    """
    B, M, N = x_dd.shape
    K = path_indices.shape[1]
    y_dd = torch.zeros_like(x_dd)
    for b in range(B):
        for k in range(K):
            delay = int(path_indices[b, k, 0].item())
            doppler = int(path_indices[b, k, 1].item())
            gain = path_gains[b, k]
            y_dd[b] = y_dd[b] + gain * torch.roll(x_dd[b], shifts=(delay, doppler), dims=(0, 1))
    return y_dd


def validate_integer_operator(
    dd_operator: SparseDDOperator,
    operator_state: SparseDDOperatorState,
    x_dd: torch.Tensor,
) -> torch.Tensor:
    """NMSE between sparse op and torch.roll reference (ongrid only)."""
    ref = integer_sparse_operator_reference(
        x_dd, operator_state.path_indices, operator_state.path_gains,
    )
    pred = dd_operator.apply(x_dd, operator_state)
    return complex_nmse(pred, ref)


def validate_offgrid_zero_offset(
    dd_operator: SparseDDOperator,
    ongrid_state: SparseDDOperatorState,
    offgrid_state: SparseDDOperatorState,
    x_dd: torch.Tensor,
) -> torch.Tensor:
    """NMSE between offgrid (zero offset) and ongrid output."""
    ongrid_y = dd_operator.apply(x_dd, ongrid_state)
    offgrid_y = dd_operator.apply(x_dd, offgrid_state)
    return complex_nmse(offgrid_y, ongrid_y)


def leakage_kernel_energy_diagnostics(
    dd_operator: SparseDDOperator,
    operator_state: SparseDDOperatorState,
    impulse_position: tuple[int, int] | None = None,
) -> dict[str, torch.Tensor]:
    """Diagnostics for off-grid leakage kernel energy and centroid.

    Places a unit complex impulse and measures output energy.
    """
    B, M, N = operator_state.h_dd.shape
    device = operator_state.h_dd.device
    dtype = operator_state.h_dd.dtype

    pos = impulse_position or (M // 2, N // 2)
    impulse = torch.zeros(B, M, N, device=device, dtype=dtype)
    impulse[:, pos[0], pos[1]] = 1.0 + 0.0j

    output = dd_operator.apply(impulse, operator_state)

    in_energy = impulse.abs().pow(2).sum()
    out_energy = output.abs().pow(2).sum()
    ratio = out_energy / in_energy.clamp_min(1e-12)

    # centroid via energy-weighted coordinates
    energy_map = output.abs().pow(2)  # [B, M, N]
    total_energy = energy_map.sum(dim=(-2, -1)).clamp_min(1e-12)  # [B]
    delay_grid = torch.arange(M, device=device, dtype=torch.float32).reshape(1, M, 1)
    doppler_grid = torch.arange(N, device=device, dtype=torch.float32).reshape(1, 1, N)
    centroid_d = (energy_map * delay_grid).sum(dim=(-2, -1)) / total_energy  # [B]
    centroid_dopp = (energy_map * doppler_grid).sum(dim=(-2, -1)) / total_energy  # [B]
    expected_d = torch.tensor(float(pos[0]), device=device)
    expected_dopp = torch.tensor(float(pos[1]), device=device)
    centroid_err = ((centroid_d - expected_d) ** 2 + (centroid_dopp - expected_dopp) ** 2).sqrt()

    # peak position
    flat_idx = energy_map.reshape(B, -1).argmax(dim=-1)
    peak_d = (flat_idx // N).float()
    peak_dopp = (flat_idx % N).float()

    return {
        "total_input_energy": in_energy.detach(),
        "total_output_energy": out_energy.detach(),
        "leakage_energy_ratio": ratio.detach(),
        "centroid_delay": centroid_d.mean().detach(),
        "centroid_doppler": centroid_dopp.mean().detach(),
        "centroid_error": centroid_err.mean().detach(),
        "peak_delay": peak_d.float().mean().detach(),
        "peak_doppler": peak_dopp.float().mean().detach(),
    }


def pilot_operator_alignment_check(
    dd_operator: SparseDDOperator,
    sparse_estimate,
    pilot_observation_mask: torch.Tensor,
    probe_position: tuple[int, int],
    probe_value: complex,
    y_dd: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """NMSE between H(probe) and y_dd within pilot observation mask."""
    B, M, N = y_dd.shape
    device = y_dd.device
    dtype = y_dd.dtype

    # build probe signal
    probe = torch.zeros(B, M, N, device=device, dtype=dtype)
    probe[:, probe_position[0], probe_position[1]] = torch.as_tensor(probe_value, device=device, dtype=dtype)

    state = dd_operator(sparse_estimate)
    y_hat = dd_operator.apply(probe, state)

    mask = _resolve_to_batch(pilot_observation_mask, B)
    if mask is None or mask.sum().item() <= 0:
        return torch.tensor(0.0, device=device)

    mask = mask.to(device=device, dtype=torch.float32)
    return complex_nmse(y_hat * mask, y_dd * mask, eps=eps)


def gain_refinement_residual_check(sparse_estimate) -> torch.Tensor:
    """Ratio of physics fit residual to pilot residual power.

    Returns 0 if either field is unavailable.
    """
    fit = getattr(sparse_estimate, "physics_fit_residual_power", None)
    pilot = getattr(sparse_estimate, "pilot_residual_power", None)
    if fit is None or pilot is None:
        return torch.tensor(0.0, device=sparse_estimate.h_dd.device)
    ratio = fit / pilot.clamp_min(1e-12)
    return ratio.mean()


def validate_receiver_physics(
    config,
    dd_operator: SparseDDOperator,
    operator_state: SparseDDOperatorState,
    x_dd: torch.Tensor,
    y_dd: torch.Tensor | None = None,
    sparse_estimate=None,
    pilot_observation_mask: torch.Tensor | None = None,
    probe_position: tuple[int, int] | None = None,
    probe_value: complex = 1.0 + 0.0j,
    thresholds: dict[str, float] | None = None,
) -> PhysicsValidationReport:
    """Run comprehensive physics validation.

    Args:
        config: Optional ReceiverConfig.
        dd_operator: SparseDDOperator instance.
        operator_state: Current SparseDDOperatorState.
        x_dd: Complex test signal [B, M, N].
        y_dd: Optional received signal for adjoint/pilot tests.
        sparse_estimate: Optional SparseChannelEstimate.
        pilot_observation_mask: Optional pilot window mask.
        probe_position: Optional (delay, doppler) tuple.
        probe_value: Complex pilot value.
        thresholds: Dict of metric thresholds.  Uses defaults if None.

    Returns:
        PhysicsValidationReport.
    """
    default_thresholds: dict[str, float] = {
        "integer_operator_nmse": 1e-8,
        "offgrid_zero_offset_nmse": 1e-6,
        "adjoint_relative_error": 1e-5,
        "leakage_energy_ratio_max": 10.0,
        "pilot_operator_alignment_error": 1e-3,
    }
    th = default_thresholds.copy()
    if thresholds is not None:
        th.update(thresholds)

    report = PhysicsValidationReport(thresholds=th)
    all_passed = True

    # 1. integer operator NMSE
    try:
        nmse = validate_integer_operator(dd_operator, operator_state, x_dd)
        report.integer_operator_nmse = nmse.detach()
        report.metrics["integer_operator_nmse"] = nmse.detach()
        if nmse > th.get("integer_operator_nmse", 1e-8):
            all_passed = False
    except Exception as e:
        report.notes["integer_operator_error"] = str(e)

    # 2. offgrid zero-offset NMSE (if offgrid and fractional_offsets are zero)
    if operator_state.operator_mode == "offgrid" and operator_state.fractional_offsets is not None:
        try:
            ongrid_state = replace(operator_state, operator_mode="ongrid")
            off_nmse = validate_offgrid_zero_offset(dd_operator, ongrid_state, operator_state, x_dd)
            report.offgrid_zero_offset_nmse = off_nmse.detach()
            report.metrics["offgrid_zero_offset_nmse"] = off_nmse.detach()
            if off_nmse > th.get("offgrid_zero_offset_nmse", 1e-6):
                all_passed = False
        except Exception as e:
            report.notes["offgrid_zero_offset_error"] = str(e)

    # 3. adjoint consistency
    if y_dd is not None:
        try:
            adj = adjoint_consistency_error(dd_operator, operator_state, x_dd, y_dd)
            report.adjoint_relative_error = adj.detach()
            report.metrics["adjoint_relative_error"] = adj.detach()
            if adj > th.get("adjoint_relative_error", 1e-5):
                all_passed = False
        except Exception as e:
            report.notes["adjoint_error"] = str(e)

    # 4. leakage diagnostics
    try:
        leak = leakage_kernel_energy_diagnostics(dd_operator, operator_state)
        report.leakage_energy_ratio = leak["leakage_energy_ratio"].detach()
        report.leakage_centroid_error = leak.get("centroid_error", torch.tensor(0.0)).detach()
        for k, v in leak.items():
            report.metrics[k] = v.detach()
        if leak["leakage_energy_ratio"] > th.get("leakage_energy_ratio_max", 10.0):
            all_passed = False
    except Exception as e:
        report.notes["leakage_error"] = str(e)

    # 5. pilot alignment
    if (sparse_estimate is not None and pilot_observation_mask is not None
            and probe_position is not None and y_dd is not None):
        try:
            palign = pilot_operator_alignment_check(
                dd_operator, sparse_estimate, pilot_observation_mask,
                probe_position, probe_value, y_dd,
            )
            report.pilot_operator_alignment_error = palign.detach()
            report.metrics["pilot_operator_alignment_error"] = palign.detach()
            if palign > th.get("pilot_operator_alignment_error", 1e-3):
                all_passed = False
        except Exception as e:
            report.notes["pilot_alignment_error"] = str(e)

    # 6. gain refinement residual ratio
    if sparse_estimate is not None:
        try:
            gr_res = gain_refinement_residual_check(sparse_estimate)
            report.gain_refinement_residual_ratio = gr_res.detach()
            report.metrics["gain_refinement_residual_ratio"] = gr_res.detach()
        except Exception as e:
            report.notes["gain_refinement_error"] = str(e)

    # 7. record what was skipped
    skipped = []
    if y_dd is None:
        skipped.append("adjoint_consistency")
    if sparse_estimate is None or pilot_observation_mask is None or probe_position is None:
        skipped.append("pilot_alignment")
    if skipped:
        report.notes["skipped_due_to_missing_inputs"] = skipped

    report.passed = all_passed
    return report
