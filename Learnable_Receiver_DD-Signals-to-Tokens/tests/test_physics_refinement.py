from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from receiver import ReceiverConfig
from receiver.physics_refinement import (
    PhysicsGuidedPathGainRefiner,
    PhysicsRefinementOutput,
    build_path_dictionary,
)


def _config(**kwargs) -> ReceiverConfig:
    values = dict(
        M=5,
        N=4,
        vocab_size=11,
        token_embedding_dim=8,
        num_unfolded_layers=2,
        topk_paths=3,
        hidden_channels=6,
        noise_var=0.1,
        use_refinement_net=False,
        physics_refinement_ridge=1e-4,
    )
    values.update(kwargs)
    return ReceiverConfig(**values)


class PhysicsRefinementTests(unittest.TestCase):
    def test_output_shapes(self):
        config = _config()
        refiner = PhysicsGuidedPathGainRefiner(config)
        y_dd = torch.randn(2, config.M, config.N, dtype=torch.complex64)
        path_indices = torch.tensor([[[0, 0], [1, 2], [3, 1]], [[0, 1], [2, 2], [4, 3]]])
        path_gains = torch.ones(2, 3, dtype=torch.complex64)

        output = refiner(y_dd, path_indices, path_gains)

        self.assertIsInstance(output, PhysicsRefinementOutput)
        self.assertEqual(output.path_gains.shape, (2, 3))
        self.assertEqual(output.residual.shape, (2, config.M, config.N))
        self.assertEqual(output.residual_power.shape, (2,))
        self.assertEqual(output.gain_covariance_diag.shape, (2, 3))
        self.assertEqual(output.confidence.shape, (2, 3))
        self.assertEqual(output.normal_matrix_diag.shape, (2, 3))

    def test_on_grid_single_path_gain_recovery(self):
        config = _config(physics_refinement_ridge=1e-6)
        refiner = PhysicsGuidedPathGainRefiner(config)
        path_indices = torch.tensor([[[1, 2]]])
        true_gain = torch.tensor([[2.0 - 0.5j]], dtype=torch.complex64)
        dictionary = build_path_dictionary((1, config.M, config.N), path_indices, None, dtype=torch.complex64)
        y_dd = (dictionary @ true_gain.unsqueeze(-1)).squeeze(-1).reshape(1, config.M, config.N)
        initial_gain = torch.tensor([[0.1 + 0.1j]], dtype=torch.complex64)

        output = refiner(y_dd, path_indices, initial_gain, confidence=torch.ones(1, 1))

        self.assertTrue(torch.allclose(output.path_gains, true_gain, atol=2e-3, rtol=2e-3))

    def test_ridge_solve_finite_with_duplicate_paths(self):
        config = _config(physics_refinement_ridge=1e-2)
        refiner = PhysicsGuidedPathGainRefiner(config)
        y_dd = torch.randn(1, config.M, config.N, dtype=torch.complex64)
        path_indices = torch.tensor([[[1, 1], [1, 1], [1, 1]]])
        path_gains = torch.randn(1, 3, dtype=torch.complex64)

        output = refiner(y_dd, path_indices, path_gains, confidence=torch.ones(1, 3))

        self.assertFalse(torch.isnan(output.path_gains.real).any())
        self.assertFalse(torch.isnan(output.gain_covariance_diag).any())
        self.assertFalse(torch.isinf(output.gain_covariance_diag).any())

    def test_fractional_offsets_optional_path_runs(self):
        config = _config(use_offgrid_refinement=True)
        refiner = PhysicsGuidedPathGainRefiner(config)
        y_dd = torch.randn(1, config.M, config.N, dtype=torch.complex64)
        path_indices = torch.tensor([[[1, 2], [3, 1]]])
        path_gains = torch.ones(1, 2, dtype=torch.complex64)
        fractional_offsets = torch.tensor([[[0.25, -0.2], [0.0, 0.1]]])

        output = refiner(
            y_dd,
            path_indices,
            path_gains,
            confidence=torch.ones(1, 2),
            fractional_offsets=fractional_offsets,
        )

        self.assertEqual(output.path_gains.shape, (1, 2))
        self.assertEqual(output.residual.shape, (1, config.M, config.N))

    def test_confidence_shape_and_range(self):
        config = _config()
        refiner = PhysicsGuidedPathGainRefiner(config)
        y_dd = torch.randn(2, config.M, config.N, dtype=torch.complex64)
        path_indices = torch.tensor([[[0, 0], [1, 1]], [[2, 2], [3, 3]]])
        path_gains = torch.ones(2, 2, dtype=torch.complex64)

        output = refiner(y_dd, path_indices, path_gains)

        self.assertEqual(output.confidence.shape, (2, 2))
        self.assertTrue(torch.all(output.confidence >= 0.0))
        self.assertTrue(torch.all(output.confidence <= 1.0))

    def test_backward_flows_to_y_dd_and_initial_path_gains(self):
        config = _config()
        refiner = PhysicsGuidedPathGainRefiner(config)
        y_dd = torch.randn(1, config.M, config.N, dtype=torch.complex64, requires_grad=True)
        path_indices = torch.tensor([[[0, 0], [1, 2]]])
        path_gains = torch.randn(1, 2, dtype=torch.complex64, requires_grad=True)

        output = refiner(y_dd, path_indices, path_gains, confidence=torch.ones(1, 2))
        loss = output.path_gains.abs().mean() + output.residual.abs().mean()
        loss.backward()

        self.assertIsNotNone(y_dd.grad)
        self.assertIsNotNone(path_gains.grad)
        self.assertGreater(float(y_dd.grad.abs().sum().item()), 0.0)
        self.assertGreater(float(path_gains.grad.abs().sum().item()), 0.0)

    def test_data_mask_limits_fitting_region(self):
        config = _config(physics_refinement_ridge=1e-6)
        refiner = PhysicsGuidedPathGainRefiner(config)
        path_indices = torch.tensor([[[0, 0]]])
        initial_gain = torch.zeros(1, 1, dtype=torch.complex64)
        y_dd = torch.zeros(1, config.M, config.N, dtype=torch.complex64)
        y_dd[:, 0, 0] = 2.0 + 0.0j
        y_dd[:, 4, 3] = 100.0 + 0.0j
        data_mask = torch.zeros(1, config.M, config.N)
        data_mask[:, 0, 0] = 1.0

        output = refiner(y_dd, path_indices, initial_gain, confidence=torch.ones(1, 1), data_mask=data_mask)

        self.assertTrue(torch.allclose(output.path_gains, torch.tensor([[2.0 + 0.0j]]), atol=2e-3, rtol=2e-3))

    def test_fit_mask_takes_priority_over_data_mask_alias(self):
        config = _config(physics_refinement_ridge=1e-6)
        refiner = PhysicsGuidedPathGainRefiner(config)
        path_indices = torch.tensor([[[0, 0]]])
        initial_gain = torch.zeros(1, 1, dtype=torch.complex64)
        y_dd = torch.zeros(1, config.M, config.N, dtype=torch.complex64)
        y_dd[:, 0, 0] = 3.0 + 0.0j
        y_dd[:, 4, 3] = 100.0 + 0.0j
        fit_mask = torch.zeros(1, config.M, config.N)
        fit_mask[:, 0, 0] = 1.0
        data_mask = torch.ones(1, config.M, config.N)

        output = refiner(y_dd, path_indices, initial_gain, confidence=torch.ones(1, 1),
                         fit_mask=fit_mask, data_mask=data_mask)

        self.assertTrue(torch.allclose(output.path_gains, torch.tensor([[3.0 + 0.0j]]), atol=2e-3, rtol=2e-3))

    def test_nonzero_probe_position_dictionary_response(self):
        config = _config()
        path_indices = torch.tensor([[[1, 0]]])
        dict_default = build_path_dictionary(
            (1, config.M, config.N), path_indices, None, dtype=torch.complex64,
        )
        dict_offset = build_path_dictionary(
            (1, config.M, config.N), path_indices, None, dtype=torch.complex64,
            probe_position=(2, 1),
        )
        peak_default = dict_default.abs().argmax().item()
        peak_offset = dict_offset.abs().argmax().item()
        self.assertNotEqual(peak_default, peak_offset)

    def test_nonzero_probe_position_gain_recovery(self):
        config = _config(physics_refinement_ridge=1e-6)
        refiner = PhysicsGuidedPathGainRefiner(config)
        pilot_pos = (2, 1)
        pilot_val = 0.5 + 0.5j
        path_indices = torch.tensor([[[1, 0]]])
        true_gain = torch.tensor([[2.0 - 0.5j]], dtype=torch.complex64)
        dictionary = build_path_dictionary(
            (1, config.M, config.N), path_indices, None, dtype=torch.complex64,
            probe_position=pilot_pos, probe_value=pilot_val,
        )
        y_dd = (dictionary @ true_gain.unsqueeze(-1)).squeeze(-1).reshape(1, config.M, config.N)
        initial_gain = torch.zeros(1, 1, dtype=torch.complex64)

        output = refiner(
            y_dd, path_indices, initial_gain, confidence=torch.ones(1, 1),
            probe_position=pilot_pos, probe_value=pilot_val,
        )

        self.assertTrue(torch.allclose(output.path_gains, true_gain, atol=2e-3, rtol=2e-3))

    def test_fit_mask_limits_ls_region_only(self):
        config = _config(physics_refinement_ridge=1e-6)
        refiner = PhysicsGuidedPathGainRefiner(config)
        path_indices = torch.tensor([[[0, 0]]])
        true_gain = torch.tensor([[2.0 + 0.0j]], dtype=torch.complex64)
        dictionary = build_path_dictionary(
            (1, config.M, config.N), path_indices, None, dtype=torch.complex64,
        )
        y_dd = (dictionary @ true_gain.unsqueeze(-1)).squeeze(-1).reshape(1, config.M, config.N)
        y_dd[:, 4, 3] = 999.0 + 0.0j
        initial_gain = torch.zeros(1, 1, dtype=torch.complex64)
        fit_mask = torch.ones(1, config.M, config.N)
        fit_mask[:, 4, 3] = 0.0

        output = refiner(y_dd, path_indices, initial_gain, confidence=torch.ones(1, 1), fit_mask=fit_mask)

        self.assertTrue(torch.allclose(output.path_gains, true_gain, atol=2e-3, rtol=2e-3))

    def test_data_mask_alias_still_compatible(self):
        config = _config(physics_refinement_ridge=1e-6)
        refiner = PhysicsGuidedPathGainRefiner(config)
        path_indices = torch.tensor([[[0, 0]]])
        initial_gain = torch.zeros(1, 1, dtype=torch.complex64)
        y_dd = torch.zeros(1, config.M, config.N, dtype=torch.complex64)
        y_dd[:, 0, 0] = 1.5 + 0.0j
        data_mask = torch.zeros(1, config.M, config.N)
        data_mask[:, 0, 0] = 1.0

        output = refiner(y_dd, path_indices, initial_gain, confidence=torch.ones(1, 1), data_mask=data_mask)

        self.assertTrue(torch.allclose(output.path_gains, torch.tensor([[1.5 + 0.0j]]), atol=2e-3, rtol=2e-3))

    def test_confidence_uses_fit_residual_not_full_residual(self):
        """Confidence must not be suppressed by large data-region residuals."""
        config = _config(physics_refinement_ridge=1e-6)
        refiner = PhysicsGuidedPathGainRefiner(config)
        path_indices = torch.tensor([[[0, 0]]])
        true_gain = torch.tensor([[2.0 + 0.0j]], dtype=torch.complex64)
        dictionary = build_path_dictionary(
            (1, config.M, config.N), path_indices, None, dtype=torch.complex64,
        )
        y_dd = (dictionary @ true_gain.unsqueeze(-1)).squeeze(-1).reshape(1, config.M, config.N)
        # Inject massive interference outside the CE fitting region
        y_dd[:, 4, 3] = 500.0 + 0.0j
        fit_mask = torch.ones(1, config.M, config.N)
        fit_mask[:, 4, 3] = 0.0  # exclude the interference position
        initial_gain = torch.zeros(1, 1, dtype=torch.complex64)

        output = refiner(y_dd, path_indices, initial_gain, confidence=torch.ones(1, 1), fit_mask=fit_mask)

        # Gain should still be recovered correctly
        self.assertTrue(torch.allclose(output.path_gains, true_gain, atol=2e-3, rtol=2e-3))
        # Confidence uses fit residual (~0.8 due to covariance), not full residual (~0.0003)
        self.assertGreater(float(output.confidence[0, 0].item()), 0.5)
        # fit_residual_power should be small
        self.assertLess(float(output.fit_residual_power[0].item()), 1e-3)
        # full_residual_power should be large (includes the 500 at position 4,3)
        self.assertGreater(float(output.full_residual_power[0].item()), 10.0)
        # backward-compat residual_power should equal fit_residual_power
        self.assertTrue(torch.allclose(output.residual_power, output.fit_residual_power))

    def test_fit_and_full_residual_powers_are_exposed(self):
        config = _config(physics_refinement_ridge=1e-4)
        refiner = PhysicsGuidedPathGainRefiner(config)
        y_dd = torch.randn(2, config.M, config.N, dtype=torch.complex64)
        path_indices = torch.tensor([[[0, 0], [1, 2]], [[0, 1], [2, 2]]])
        path_gains = torch.ones(2, 2, dtype=torch.complex64)

        output = refiner(y_dd, path_indices, path_gains)

        self.assertIsNotNone(output.fit_residual_power)
        self.assertIsNotNone(output.full_residual_power)
        self.assertEqual(output.fit_residual_power.shape, (2,))
        self.assertEqual(output.full_residual_power.shape, (2,))


if __name__ == "__main__":
    unittest.main()
