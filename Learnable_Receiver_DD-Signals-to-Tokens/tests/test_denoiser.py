from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from receiver import DenoiserOutput, LearnableOTFSReceiver, ReceiverConfig
from receiver.denoiser import ResidualDDDenoiser, build_confidence_map, build_uncertainty_map


def _config(use_refinement_net: bool = True) -> ReceiverConfig:
    return ReceiverConfig(
        M=6,
        N=5,
        vocab_size=13,
        token_embedding_dim=8,
        num_unfolded_layers=2,
        topk_paths=3,
        hidden_channels=7,
        noise_var=0.1,
        use_refinement_net=use_refinement_net,
    )


def _complex_dd(batch: int, config: ReceiverConfig) -> torch.Tensor:
    real = torch.randn(batch, config.M, config.N)
    imag = torch.randn(batch, config.M, config.N)
    return torch.complex(real, imag)


class DenoiserTests(unittest.TestCase):
    def test_confidence_map_helper_from_path_confidence(self):
        support_mask = torch.zeros(2, 4, 5)
        path_indices = torch.tensor(
            [
                [[1, 2], [3, 4]],
                [[0, 1], [2, 3]],
            ],
            dtype=torch.long,
        )
        confidence = torch.tensor([[0.8, 0.3], [1.0, 0.2]])

        confidence_map = build_confidence_map(
            support_mask,
            confidence=confidence,
            path_indices=path_indices,
        )

        self.assertEqual(confidence_map.shape, (2, 4, 5))
        self.assertAlmostEqual(float(confidence_map[0, 1, 2].item()), 0.8, places=6)
        self.assertAlmostEqual(float(confidence_map[0, 3, 4].item()), 0.3, places=6)
        self.assertAlmostEqual(float(confidence_map[1, 0, 1].item()), 1.0, places=6)
        self.assertAlmostEqual(float(confidence_map[1, 2, 3].item()), 0.2, places=6)

    def test_uncertainty_map_shape_and_range(self):
        confidence_map = torch.rand(2, 4, 5)
        residual_energy = torch.rand(2, 4, 5)
        pilot_residual_power = torch.tensor([0.2, 1.0])

        uncertainty_map = build_uncertainty_map(
            confidence_map,
            residual_energy=residual_energy,
            pilot_residual_power=pilot_residual_power,
        )

        self.assertEqual(uncertainty_map.shape, confidence_map.shape)
        self.assertFalse(torch.isnan(uncertainty_map).any())
        self.assertFalse(torch.isinf(uncertainty_map).any())
        self.assertTrue(torch.all(uncertainty_map >= 0))

    def test_output_shape_and_dtype_match_x_eq(self):
        config = _config()
        denoiser = ResidualDDDenoiser(config)
        x_eq = _complex_dd(2, config)
        y_dd = _complex_dd(2, config)
        h_dd = _complex_dd(2, config)
        support_mask = torch.ones(2, config.M, config.N)

        x_refined = denoiser(x_eq, y_dd=y_dd, h_dd=h_dd, support_mask=support_mask, snr_db=torch.tensor([10.0, 12.0]))

        self.assertEqual(x_refined.shape, x_eq.shape)
        self.assertEqual(x_refined.dtype, x_eq.dtype)
        self.assertTrue(torch.is_complex(x_refined))

    def test_forward_accepts_confidence_and_uncertainty_inputs(self):
        config = _config()
        denoiser = ResidualDDDenoiser(config)
        x_eq = _complex_dd(2, config)
        confidence_map = torch.rand(2, config.M, config.N)
        uncertainty_map = torch.rand(2, config.M, config.N)

        x_refined = denoiser(
            x_eq,
            confidence_map=confidence_map,
            uncertainty_map=uncertainty_map,
        )

        self.assertEqual(x_refined.shape, x_eq.shape)
        self.assertEqual(x_refined.dtype, x_eq.dtype)

    def test_return_delta_output_shapes(self):
        config = _config()
        denoiser = ResidualDDDenoiser(config)
        x_eq = _complex_dd(3, config)

        output = denoiser(x_eq, return_delta=True)

        self.assertIsInstance(output, DenoiserOutput)
        self.assertEqual(output.x_refined.shape, x_eq.shape)
        self.assertEqual(output.delta.shape, x_eq.shape)
        self.assertIsNotNone(output.features)
        self.assertEqual(output.features.shape[0], x_eq.shape[0])
        self.assertEqual(output.features.shape[-2:], (config.M, config.N))

    def test_return_delta_includes_gate_and_maps(self):
        config = _config()
        denoiser = ResidualDDDenoiser(config)
        x_eq = _complex_dd(2, config)
        confidence_map = torch.rand(2, config.M, config.N)
        uncertainty_map = torch.rand(2, config.M, config.N)

        output = denoiser(
            x_eq,
            confidence_map=confidence_map,
            uncertainty_map=uncertainty_map,
            return_delta=True,
        )

        self.assertEqual(output.correction_gate.shape, (2, 1, config.M, config.N))
        self.assertEqual(output.confidence_map.shape, (2, config.M, config.N))
        self.assertEqual(output.uncertainty_map.shape, (2, config.M, config.N))

    def test_initial_output_is_identity_due_to_zero_delta_head(self):
        config = _config()
        denoiser = ResidualDDDenoiser(config)
        x_eq = _complex_dd(2, config)
        y_dd = _complex_dd(2, config)
        h_dd = _complex_dd(2, config)
        support_mask = torch.randint(0, 2, (2, config.M, config.N)).float()
        confidence_map = torch.rand(2, config.M, config.N)
        uncertainty_map = torch.rand(2, config.M, config.N)

        x_refined = denoiser(
            x_eq,
            y_dd=y_dd,
            h_dd=h_dd,
            support_mask=support_mask,
            snr_db=15.0,
            confidence_map=confidence_map,
            uncertainty_map=uncertainty_map,
        )

        self.assertTrue(torch.allclose(x_refined, x_eq, atol=1e-7, rtol=1e-7))

    def test_support_mask_changes_gating_path_without_error(self):
        config = _config()
        denoiser = ResidualDDDenoiser(config)
        x_eq = _complex_dd(1, config)
        support_zeros = torch.zeros(1, config.M, config.N)
        support_ones = torch.ones(1, config.M, config.N)

        output_zero = denoiser(x_eq, support_mask=support_zeros, return_delta=True)
        output_one = denoiser(x_eq, support_mask=support_ones, return_delta=True)

        self.assertEqual(output_zero.x_refined.shape, x_eq.shape)
        self.assertEqual(output_one.x_refined.shape, x_eq.shape)
        self.assertIsNotNone(output_zero.features)
        self.assertIsNotNone(output_one.features)

    def test_missing_optional_inputs_still_runs(self):
        config = _config()
        denoiser = ResidualDDDenoiser(config)
        x_eq = _complex_dd(2, config)

        x_refined = denoiser(x_eq, y_dd=None, h_dd=None, support_mask=None, snr_db=None)

        self.assertEqual(x_refined.shape, x_eq.shape)
        self.assertEqual(x_refined.dtype, x_eq.dtype)

    def test_disabled_denoiser_return_delta_fields_have_shapes(self):
        config = _config(use_refinement_net=False)
        denoiser = ResidualDDDenoiser(config)
        x_eq = _complex_dd(2, config)

        output = denoiser(x_eq, return_delta=True)

        self.assertTrue(torch.all(output.delta == 0))
        self.assertEqual(output.correction_gate.shape, (2, 1, config.M, config.N))
        self.assertEqual(output.confidence_map.shape, (2, config.M, config.N))
        self.assertEqual(output.uncertainty_map.shape, (2, config.M, config.N))

    def test_backward_flows_to_x_eq_and_network_parameters(self):
        config = _config()
        denoiser = ResidualDDDenoiser(config)
        x_eq = _complex_dd(2, config).requires_grad_()
        y_dd = _complex_dd(2, config)
        support_mask = torch.ones(2, config.M, config.N)
        confidence_map = torch.rand(2, config.M, config.N)
        uncertainty_map = torch.rand(2, config.M, config.N)

        x_refined = denoiser(
            x_eq,
            y_dd=y_dd,
            support_mask=support_mask,
            confidence_map=confidence_map,
            uncertainty_map=uncertainty_map,
        )
        loss = x_refined.abs().pow(2).mean()
        loss.backward()

        self.assertIsNotNone(x_eq.grad)
        self.assertGreater(float(x_eq.grad.abs().sum().item()), 0.0)
        parameter_grad = sum(
            float(param.grad.abs().sum().item())
            for param in denoiser.parameters()
            if param.grad is not None
        )
        self.assertGreater(parameter_grad, 0.0)

    def test_full_receiver_forward_still_runs(self):
        config = _config()
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)

        token_logits = model(y_dd)

        self.assertEqual(token_logits.shape, (2, config.vocab_size))
        self.assertTrue(token_logits.dtype.is_floating_point)


if __name__ == "__main__":
    unittest.main()
