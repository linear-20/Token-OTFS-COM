from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from receiver import (
    DataConsistencyCorrection,
    DataConsistencyOutput,
    ReceiverConfig,
)
from receiver.dd_ops import SparseDDOperator, SparseDDOperatorState


def _config(**kwargs) -> ReceiverConfig:
    values = dict(
        M=5,
        N=4,
        vocab_size=11,
        token_embedding_dim=7,
        num_unfolded_layers=2,
        topk_paths=3,
        hidden_channels=6,
        noise_var=0.1,
        use_refinement_net=True,
        use_offgrid_refinement=True,
        use_data_consistency_correction=True,
        dc_step_init=0.0,
        dc_step_scale=1.0,
    )
    values.update(kwargs)
    return ReceiverConfig(**values)


def _complex_dd(batch, config, **kwargs):
    real = torch.randn(batch, config.M, config.N, **kwargs)
    imag = torch.randn(batch, config.M, config.N, **kwargs)
    return torch.complex(real, imag)


def _make_state(config, batch, device=None):
    y_dd = _complex_dd(batch, config)
    if device is not None:
        y_dd = y_dd.to(device=device)
    K = config.topk_paths
    if batch == 1:
        path_indices = torch.zeros(1, K, 2, dtype=torch.long)
        path_indices[0, 0] = torch.tensor([0, 0])
        path_indices[0, 1] = torch.tensor([1, 2])
        path_indices[0, 2] = torch.tensor([3, 1])
    else:
        path_indices = torch.zeros(batch, K, 2, dtype=torch.long)
        path_indices[0, 0] = torch.tensor([0, 0])
        path_indices[0, 1] = torch.tensor([1, 2])
        path_indices[0, 2] = torch.tensor([3, 1])
        path_indices[1, 0] = torch.tensor([0, 1])
        path_indices[1, 1] = torch.tensor([2, 2])
        path_indices[1, 2] = torch.tensor([4, 3])
    path_gains = torch.ones(batch, K, dtype=torch.complex64)
    if device is not None:
        path_indices = path_indices.to(device=device)
        path_gains = path_gains.to(device=device)
    return SparseDDOperatorState(
        h_dd=y_dd,
        support_mask=torch.ones_like(y_dd.real),
        path_indices=path_indices,
        path_gains=path_gains,
        operator_mode="ongrid",
        kernel_radius=0,
        kernel_type="linear",
        normalize_kernel=True,
    )


class DataConsistencyCorrectionTests(unittest.TestCase):
    def test_output_shape_and_dtype(self):
        config = _config()
        dd_op = SparseDDOperator(config)
        module = DataConsistencyCorrection(config, dd_op)
        x_dd = _complex_dd(2, config)
        y_dd = _complex_dd(2, config)
        state = _make_state(config, 2)

        x_out = module(x_dd, y_dd, state)

        self.assertEqual(x_out.shape, (2, config.M, config.N))
        self.assertTrue(torch.is_complex(x_out))

    def test_default_zero_step_identity(self):
        config = _config(dc_step_init=0.0, dc_step_scale=0.1)
        dd_op = SparseDDOperator(config)
        module = DataConsistencyCorrection(config, dd_op)
        x_dd = _complex_dd(2, config)
        y_dd = _complex_dd(2, config)
        state = _make_state(config, 2)

        x_out = module(x_dd, y_dd, state)

        # at init, step ~ 0 so x_out ~ x_dd
        diff = (x_out - x_dd).abs().mean()
        self.assertLess(float(diff.item()), 1e-2)

    def test_return_details_gives_all_fields(self):
        config = _config()
        dd_op = SparseDDOperator(config)
        module = DataConsistencyCorrection(config, dd_op)
        x_dd = _complex_dd(1, config)
        y_dd = _complex_dd(1, config)
        state = _make_state(config, 1)

        output = module(x_dd, y_dd, state, return_details=True)

        self.assertIsInstance(output, DataConsistencyOutput)
        self.assertEqual(output.x_corrected.shape, (1, config.M, config.N))
        self.assertEqual(output.correction.shape, (1, config.M, config.N))
        self.assertEqual(output.residual.shape, (1, config.M, config.N))
        self.assertEqual(output.gradient.shape, (1, config.M, config.N))
        self.assertIsNotNone(output.observation_weight)
        self.assertIsNotNone(output.update_gate)
        self.assertEqual(output.observation_weight.shape, (1, config.M, config.N))
        self.assertEqual(output.update_gate.shape, (1, config.M, config.N))
        # backward-compat alias
        self.assertIsNotNone(output.correction_gate)
        self.assertEqual(output.correction_gate.shape, (1, config.M, config.N))
        self.assertEqual(output.residual_energy.shape, (1, config.M, config.N))

    def test_residual_is_hx_minus_y(self):
        config = _config()
        dd_op = SparseDDOperator(config)
        module = DataConsistencyCorrection(config, dd_op)
        x_dd = _complex_dd(1, config)
        y_dd = _complex_dd(1, config)
        state = _make_state(config, 1)

        output = module(x_dd, y_dd, state, return_details=True)
        y_hat = dd_op.apply(x_dd, state)

        self.assertTrue(torch.allclose(output.residual, y_hat - y_dd, atol=1e-6))

    def test_gradient_is_hh_of_weighted_residual(self):
        config = _config(dc_step_init=0.5, dc_step_scale=1.0)
        dd_op = SparseDDOperator(config)
        module = DataConsistencyCorrection(config, dd_op)
        x_dd = _complex_dd(1, config)
        y_dd = _complex_dd(1, config)
        state = _make_state(config, 1)
        mask = torch.ones(1, config.M, config.N)
        mask[:, 0, 0] = 0.0

        output = module(x_dd, y_dd, state, return_details=True, data_mask=mask)
        y_hat = dd_op.apply(x_dd, state)
        residual = y_hat - y_dd
        obs_w = output.observation_weight
        expected_grad = dd_op.matched_filter(obs_w * residual, state)

        self.assertTrue(torch.allclose(output.gradient, expected_grad, atol=1e-5))

    def test_correction_does_not_duplicate_obs_weight(self):
        config = _config(dc_step_init=1.0, dc_step_scale=1.0)
        dd_op = SparseDDOperator(config)
        module = DataConsistencyCorrection(config, dd_op)
        x_dd = _complex_dd(1, config)
        y_dd = _complex_dd(1, config)
        state = _make_state(config, 1)
        mask = torch.ones(1, config.M, config.N)
        mask[:, 0, 0] = 0.0

        output = module(x_dd, y_dd, state, return_details=True, data_mask=mask)
        step = output.step_size
        gate = output.update_gate
        grad = output.gradient
        expected_correction = step * gate * grad

        self.assertTrue(torch.allclose(output.correction, expected_correction, atol=1e-5))

    def test_data_mask_changes_correction(self):
        config = _config(dc_step_init=0.5, dc_step_scale=1.0)
        dd_op = SparseDDOperator(config)
        module = DataConsistencyCorrection(config, dd_op)
        x_dd = _complex_dd(1, config)
        y_dd = _complex_dd(1, config)
        state = _make_state(config, 1)
        mask = torch.ones(1, config.M, config.N)
        mask[:, 0, 0] = 0.0  # exclude one position

        out_full = module(x_dd, y_dd, state)
        out_masked = module(x_dd, y_dd, state, data_mask=mask)

        self.assertFalse(torch.allclose(out_full, out_masked))

    def test_reliability_changes_correction(self):
        config = _config(dc_step_init=0.5, dc_step_scale=1.0)
        dd_op = SparseDDOperator(config)
        module = DataConsistencyCorrection(config, dd_op)
        x_dd = _complex_dd(1, config)
        y_dd = _complex_dd(1, config)
        state = _make_state(config, 1)
        rel_high = torch.ones(1, config.M, config.N)
        rel_low = 0.1 * torch.ones(1, config.M, config.N)

        out_high = module(x_dd, y_dd, state, reliability_map=rel_high)
        out_low = module(x_dd, y_dd, state, reliability_map=rel_low)

        self.assertFalse(torch.allclose(out_high, out_low))

    def test_uncertainty_suppresses_correction(self):
        config = _config(dc_step_init=0.5, dc_step_scale=1.0)
        dd_op = SparseDDOperator(config)
        module = DataConsistencyCorrection(config, dd_op)
        x_dd = _complex_dd(1, config)
        y_dd = _complex_dd(1, config)
        state = _make_state(config, 1)
        unc_high = 10.0 * torch.ones(1, config.M, config.N)

        out_no_unc = module(x_dd, y_dd, state)
        out_high_unc = module(x_dd, y_dd, state, uncertainty_map=unc_high)

        diff_no_unc = (out_no_unc - x_dd).abs().mean()
        diff_unc = (out_high_unc - x_dd).abs().mean()
        # high uncertainty should keep correction closer to identity
        self.assertLess(float(diff_unc.item()), float(diff_no_unc.item()) + 1e-3)

    def test_backward_flows_to_x_dd_and_params(self):
        config = _config(dc_step_init=0.5, dc_step_scale=1.0)
        dd_op = SparseDDOperator(config)
        module = DataConsistencyCorrection(config, dd_op)
        x_dd = _complex_dd(1, config)
        x_dd.requires_grad_(True)
        y_dd = _complex_dd(1, config)
        state = _make_state(config, 1)

        out = module(x_dd, y_dd, state)
        loss = out.abs().mean()
        loss.backward()

        self.assertIsNotNone(x_dd.grad)
        self.assertGreater(float(x_dd.grad.abs().sum().item()), 0.0)
        self.assertIsNotNone(module.raw_step.grad)
        self.assertIsNotNone(module.raw_gate.grad)

    def test_dc_gate_uses_reliability_flag_multiplies_gate(self):
        config = _config(dc_step_init=0.5, dc_step_scale=1.0,
                         dc_gate_uses_reliability=True)
        dd_op = SparseDDOperator(config)
        module = DataConsistencyCorrection(config, dd_op)
        x_dd = _complex_dd(1, config)
        y_dd = _complex_dd(1, config)
        state = _make_state(config, 1)
        rel_map = 0.5 * torch.ones(1, config.M, config.N)

        output = module(x_dd, y_dd, state, return_details=True,
                        reliability_map=rel_map)

        # update_gate should carry the reliability modulation
        scalar_gate = module.gate_value
        expected_gate = scalar_gate * rel_map.to(scalar_gate.device)
        self.assertTrue(torch.allclose(output.update_gate, expected_gate, atol=1e-4))

    def test_no_dense_matrix_constructed(self):
        config = _config()
        dd_op = SparseDDOperator(config)
        module = DataConsistencyCorrection(config, dd_op)
        x_dd = _complex_dd(1, config)
        y_dd = _complex_dd(1, config)
        state = _make_state(config, 1)

        # forward should not build or reference state.h_eff
        out = module(x_dd, y_dd, state, return_details=True)

        self.assertFalse(hasattr(state, "h_eff"))


if __name__ == "__main__":
    unittest.main()
