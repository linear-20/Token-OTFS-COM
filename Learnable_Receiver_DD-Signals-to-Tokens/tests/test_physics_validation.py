from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from receiver import (
    LearnableOTFSReceiver,
    PhysicsValidationReport,
    ReceiverConfig,
    SparseDDOperator,
    SparseDDOperatorState,
    adjoint_consistency_error,
    complex_nmse,
    gain_refinement_residual_check,
    integer_sparse_operator_reference,
    leakage_kernel_energy_diagnostics,
    pilot_operator_alignment_check,
    validate_integer_operator,
    validate_offgrid_zero_offset,
    validate_receiver_physics,
)
from receiver.sparse_channel import SparseChannelEstimate


def _config(**kwargs) -> ReceiverConfig:
    values = dict(
        M=6, N=5, vocab_size=11, token_embedding_dim=7,
        num_unfolded_layers=2, topk_paths=3, hidden_channels=6,
        noise_var=0.1, use_refinement_net=True, use_offgrid_refinement=True,
    )
    values.update(kwargs)
    return ReceiverConfig(**values)


def _complex_dd(batch, config):
    real = torch.randn(batch, config.M, config.N)
    imag = torch.randn(batch, config.M, config.N)
    return torch.complex(real, imag)


def _ongrid_state(config, batch=1):
    x_dd = _complex_dd(batch, config)
    path_indices = torch.zeros(batch, 2, 2, dtype=torch.long)
    path_indices[0, 0] = torch.tensor([1, 2])
    path_indices[0, 1] = torch.tensor([0, 0])
    if batch > 1:
        path_indices[1, 0] = torch.tensor([0, 1])
        path_indices[1, 1] = torch.tensor([2, 2])
    path_gains = torch.ones(batch, 2, dtype=torch.complex64)
    return x_dd, SparseDDOperatorState(
        h_dd=x_dd, support_mask=torch.ones_like(x_dd.real),
        path_indices=path_indices, path_gains=path_gains,
        operator_mode="ongrid", kernel_radius=0,
        kernel_type="linear", normalize_kernel=True,
    )


class ComplexNMSETests(unittest.TestCase):
    def test_identical_returns_near_zero(self):
        config = _config()
        x = _complex_dd(2, config)
        self.assertLess(float(complex_nmse(x, x.clone())), 1e-8)

    def test_zero_target_does_not_nan(self):
        config = _config()
        x = _complex_dd(1, config)
        target = torch.zeros_like(x)
        loss = complex_nmse(x, target)
        self.assertTrue(torch.isfinite(loss))


class IntegerReferenceTests(unittest.TestCase):
    def test_single_zero_path_equals_input(self):
        config = _config()
        x = _complex_dd(1, config)
        idx = torch.tensor([[[0, 0]]])
        gain = torch.ones(1, 1, dtype=torch.complex64)
        ref = integer_sparse_operator_reference(x, idx, gain)
        self.assertTrue(torch.allclose(ref, x, atol=1e-6))

    def test_single_shifted_path(self):
        config = _config()
        x = _complex_dd(1, config)
        idx = torch.tensor([[[2, 3]]])
        gain = torch.ones(1, 1, dtype=torch.complex64)
        ref = integer_sparse_operator_reference(x, idx, gain)
        expected = torch.roll(x, shifts=(2, 3), dims=(1, 2))
        self.assertTrue(torch.allclose(ref, expected, atol=1e-6))

    def test_multipath_finite(self):
        config = _config()
        x = _complex_dd(1, config)
        idx = torch.tensor([[[0, 1], [2, 0]]])
        gain = torch.tensor([[0.5 + 0.0j, 0.3 + 0.0j]], dtype=torch.complex64)
        ref = integer_sparse_operator_reference(x, idx, gain)
        self.assertTrue(torch.isfinite(ref).all())


class ValidateIntegerOperatorTests(unittest.TestCase):
    def test_ongrid_nmse_small(self):
        config = _config()
        dd_op = SparseDDOperator(config)
        x_dd, state = _ongrid_state(config)
        nmse = validate_integer_operator(dd_op, state, x_dd)
        self.assertLess(float(nmse.item()), 1e-8)


class AdjointConsistencyTests(unittest.TestCase):
    def test_ongrid_adjoint_small(self):
        config = _config()
        dd_op = SparseDDOperator(config)
        x_dd, state = _ongrid_state(config)
        y_dd = _complex_dd(1, config)
        err = adjoint_consistency_error(dd_op, state, x_dd, y_dd)
        self.assertLess(float(err.item()), 1e-6)

    def test_offgrid_adjoint_finite(self):
        config = _config()
        dd_op = SparseDDOperator(config)
        x_dd = _complex_dd(1, config)
        y_dd = _complex_dd(1, config)
        path_indices = torch.tensor([[[1, 2]]])
        path_gains = torch.ones(1, 1, dtype=torch.complex64)
        frac = torch.tensor([[[0.25, -0.1]]])
        state = SparseDDOperatorState(
            h_dd=x_dd, support_mask=torch.ones_like(x_dd.real),
            path_indices=path_indices, path_gains=path_gains,
            fractional_offsets=frac, operator_mode="offgrid",
            kernel_radius=1, kernel_type="linear", normalize_kernel=True,
        )
        err = adjoint_consistency_error(dd_op, state, x_dd, y_dd)
        self.assertTrue(torch.isfinite(err))


class OffgridZeroOffsetTests(unittest.TestCase):
    def test_offgrid_zero_matches_ongrid(self):
        config = _config()
        dd_op = SparseDDOperator(config)
        x_dd = _complex_dd(1, config)
        path_indices = torch.tensor([[[1, 0]]])
        path_gains = torch.ones(1, 1, dtype=torch.complex64)
        frac_zero = torch.zeros(1, 1, 2)
        ongrid = SparseDDOperatorState(
            h_dd=x_dd, support_mask=torch.ones_like(x_dd.real),
            path_indices=path_indices, path_gains=path_gains,
            operator_mode="ongrid",
        )
        offgrid = SparseDDOperatorState(
            h_dd=x_dd, support_mask=torch.ones_like(x_dd.real),
            path_indices=path_indices, path_gains=path_gains,
            fractional_offsets=frac_zero, operator_mode="offgrid",
            kernel_radius=1, kernel_type="linear", normalize_kernel=True,
        )
        nmse = validate_offgrid_zero_offset(dd_op, ongrid, offgrid, x_dd)
        self.assertLess(float(nmse.item()), 1e-6)


class LeakageDiagnosticsTests(unittest.TestCase):
    def test_returns_required_keys(self):
        config = _config()
        dd_op = SparseDDOperator(config)
        x_dd, state = _ongrid_state(config)
        leak = leakage_kernel_energy_diagnostics(dd_op, state)
        for key in ("total_input_energy", "total_output_energy",
                     "leakage_energy_ratio", "centroid_error"):
            self.assertIn(key, leak)

    def test_energy_ratio_finite_positive(self):
        config = _config()
        dd_op = SparseDDOperator(config)
        x_dd, state = _ongrid_state(config)
        leak = leakage_kernel_energy_diagnostics(dd_op, state)
        self.assertTrue(torch.isfinite(leak["leakage_energy_ratio"]))
        self.assertGreater(float(leak["leakage_energy_ratio"].item()), 0.0)

    def test_peak_valid(self):
        config = _config()
        dd_op = SparseDDOperator(config)
        x_dd, state = _ongrid_state(config)
        leak = leakage_kernel_energy_diagnostics(dd_op, state,
                                                  impulse_position=(2, 2))
        peak_d = int(leak["peak_delay"].item())
        peak_dopp = int(leak["peak_doppler"].item())
        self.assertGreaterEqual(peak_d, 0)
        self.assertLess(peak_d, config.M)


class PilotAlignmentTests(unittest.TestCase):
    def test_one_path_case_small_error(self):
        config = _config()
        dd_op = SparseDDOperator(config)
        # one-path pilot: path at (1,0), probe at (2,1), gain=1
        est = SparseChannelEstimate(
            h_dd=torch.zeros(1, config.M, config.N, dtype=torch.complex64),
            support_mask=torch.zeros(1, config.M, config.N),
            path_indices=torch.tensor([[[1, 0]]]),
            path_gains=torch.ones(1, 1, dtype=torch.complex64),
            estimator_mode="pilot",
        )
        est.h_dd[0, 1, 0] = 1.0 + 0.0j
        est.support_mask[0, 1, 0] = 1.0
        probe_pos = (2, 1)
        probe_val = 1.0 + 0.0j
        mask = torch.zeros(1, config.M, config.N)
        mask[:, 1:4, 0:3] = 1.0

        # y_dd = H(probe) = circular shift of probe by (1,0)
        probe = torch.zeros(1, config.M, config.N, dtype=torch.complex64)
        probe[:, 2, 1] = 1.0 + 0.0j
        y_dd = integer_sparse_operator_reference(
            probe, est.path_indices, est.path_gains,
        )
        err = pilot_operator_alignment_check(
            dd_op, est, mask, probe_pos, probe_val, y_dd,
        )
        self.assertTrue(torch.isfinite(err))
        self.assertLess(float(err.item()), 1e-8)

    def test_alignment_uses_passed_operator_configuration(self):
        config = _config(dd_operator_mode="offgrid", offgrid_kernel_radius=2)
        dd_op = SparseDDOperator(config)
        est = SparseChannelEstimate(
            h_dd=torch.zeros(1, config.M, config.N, dtype=torch.complex64),
            support_mask=torch.zeros(1, config.M, config.N),
            path_indices=torch.tensor([[[1, 0]]]),
            path_gains=torch.ones(1, 1, dtype=torch.complex64),
            fractional_offsets=torch.tensor([[[0.25, 0.0]]]),
            estimator_mode="pilot",
        )
        est.h_dd[0, 1, 0] = 1.0 + 0.0j
        est.support_mask[0, 1, 0] = 1.0
        probe_pos = (2, 1)
        probe_val = 1.0 + 0.0j
        probe = torch.zeros(1, config.M, config.N, dtype=torch.complex64)
        probe[:, probe_pos[0], probe_pos[1]] = probe_val
        state = dd_op(est)
        y_dd = dd_op.apply(probe, state)
        mask = torch.ones(1, config.M, config.N)

        err = pilot_operator_alignment_check(
            dd_op, est, mask, probe_pos, probe_val, y_dd,
        )

        self.assertLess(float(err.item()), 1e-8)

    def test_all_zero_mask_returns_finite_zero(self):
        config = _config()
        dd_op = SparseDDOperator(config)
        est = SparseChannelEstimate(
            h_dd=torch.zeros(1, config.M, config.N, dtype=torch.complex64),
            support_mask=torch.zeros(1, config.M, config.N),
            path_indices=torch.zeros(1, 1, 2, dtype=torch.long),
            path_gains=torch.ones(1, 1, dtype=torch.complex64),
            estimator_mode="pilot",
        )
        mask = torch.zeros(1, config.M, config.N)
        y_dd = _complex_dd(1, config)
        err = pilot_operator_alignment_check(
            dd_op, est, mask, (0, 0), 1.0 + 0.0j, y_dd,
        )
        self.assertLessEqual(float(err.item()), 1e-6)


class GainRefinementTests(unittest.TestCase):
    def test_with_fields_returns_finite(self):
        config = _config()
        est = SparseChannelEstimate(
            h_dd=_complex_dd(1, config),
            support_mask=torch.ones(1, config.M, config.N),
            path_indices=torch.zeros(1, 2, 2, dtype=torch.long),
            path_gains=torch.ones(1, 2, dtype=torch.complex64),
            pilot_residual_power=torch.tensor([0.01]),
            physics_fit_residual_power=torch.tensor([0.005]),
            physics_refined=True,
        )
        ratio = gain_refinement_residual_check(est)
        self.assertTrue(torch.isfinite(ratio))

    def test_missing_fields_returns_zero(self):
        config = _config()
        est = SparseChannelEstimate(
            h_dd=_complex_dd(1, config),
            support_mask=torch.ones(1, config.M, config.N),
            path_indices=torch.zeros(1, 2, 2, dtype=torch.long),
            path_gains=torch.ones(1, 2, dtype=torch.complex64),
        )
        ratio = gain_refinement_residual_check(est)
        self.assertLessEqual(float(ratio.item()), 1e-6)


class ValidatePhysicsTests(unittest.TestCase):
    def test_returns_report_with_passed_bool(self):
        config = _config()
        dd_op = SparseDDOperator(config)
        x_dd, state = _ongrid_state(config)
        y_dd = _complex_dd(1, config)

        report = validate_receiver_physics(
            config, dd_op, state, x_dd, y_dd=y_dd,
        )
        self.assertIsInstance(report, PhysicsValidationReport)
        self.assertIsInstance(report.passed, bool)
        self.assertIsNotNone(report.integer_operator_nmse)

    def test_missing_optional_inputs_notes(self):
        config = _config()
        dd_op = SparseDDOperator(config)
        x_dd, state = _ongrid_state(config)

        report = validate_receiver_physics(config, dd_op, state, x_dd)
        self.assertIn("skipped_due_to_missing_inputs", report.notes)

    def test_metrics_contains_keys(self):
        config = _config()
        dd_op = SparseDDOperator(config)
        x_dd, state = _ongrid_state(config)
        y_dd = _complex_dd(1, config)

        report = validate_receiver_physics(
            config, dd_op, state, x_dd, y_dd=y_dd,
        )
        self.assertIn("integer_operator_nmse", report.metrics)


class ConstraintsTests(unittest.TestCase):
    def test_no_main_algorithm_change(self):
        config = _config()
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)
        logits = model(y_dd)
        self.assertEqual(logits.shape, (2, config.vocab_size))


if __name__ == "__main__":
    unittest.main()
