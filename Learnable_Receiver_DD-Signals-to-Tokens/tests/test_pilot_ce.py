from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from receiver import LearnableOTFSReceiver, ReceiverConfig
from receiver.dd_ops import dd_circular_convolve_sparse
from receiver.pilot_ce import (
    EmbeddedPilotConfig,
    PilotSparseChannelEstimator,
    build_embedded_pilot_masks,
    extract_pilot_observation,
    insert_embedded_pilot,
    validate_pilot_ce_layout,
)
from receiver.sparse_channel import SparseChannelEstimator


def _config(**kwargs) -> ReceiverConfig:
    values = dict(
        M=6,
        N=7,
        vocab_size=13,
        token_embedding_dim=8,
        num_unfolded_layers=2,
        topk_paths=3,
        hidden_channels=6,
        noise_var=0.1,
        use_refinement_net=False,
    )
    values.update(kwargs)
    return ReceiverConfig(**values)


def _pilot_config() -> EmbeddedPilotConfig:
    return EmbeddedPilotConfig(
        pilot_delay=3,
        pilot_doppler=3,
        guard_delay=1,
        guard_doppler=2,
        obs_delay_radius=1,
        obs_doppler_radius=1,
        pilot_value=1.0 + 0.0j,
        wrap_around=False,
    )


def _complex_dd(batch: int, config: ReceiverConfig) -> torch.Tensor:
    real = torch.randn(batch, config.M, config.N)
    imag = torch.randn(batch, config.M, config.N)
    return torch.complex(real, imag)


class PilotCETests(unittest.TestCase):
    def test_build_embedded_pilot_masks_shape_and_disjoint(self):
        config = _pilot_config()

        masks = build_embedded_pilot_masks(6, 7, config)

        self.assertEqual(masks.pilot_mask.shape, (6, 7))
        self.assertEqual(masks.guard_mask.shape, (6, 7))
        self.assertEqual(masks.data_mask.shape, (6, 7))
        self.assertEqual(int(masks.pilot_mask.sum().item()), 1)
        self.assertFalse(bool((masks.pilot_mask & masks.guard_mask).any()))
        self.assertFalse(bool((masks.pilot_mask & masks.data_mask).any()))
        self.assertFalse(bool((masks.guard_mask & masks.data_mask).any()))

    def test_pilot_layout_requires_obs_inside_guard(self):
        config = EmbeddedPilotConfig(
            pilot_delay=2,
            pilot_doppler=2,
            guard_delay=0,
            guard_doppler=1,
            obs_delay_radius=1,
            obs_doppler_radius=1,
            require_obs_within_guard=True,
        )
        y_dd = torch.zeros(1, 5, 5, dtype=torch.complex64)

        with self.assertRaises(ValueError):
            validate_pilot_ce_layout(5, 5, config)
        with self.assertRaises(ValueError):
            build_embedded_pilot_masks(5, 5, config)
        with self.assertRaises(ValueError):
            extract_pilot_observation(y_dd, config)
        with self.assertRaises(ValueError):
            PilotSparseChannelEstimator(config, topk_paths=2)(y_dd)

    def test_insert_embedded_pilot_writes_pilot_zeros_guard_and_preserves_data(self):
        config = _pilot_config()
        masks = build_embedded_pilot_masks(6, 7, config)
        x_dd = torch.randn(2, 6, 7, dtype=torch.complex64)

        x_pilot = insert_embedded_pilot(x_dd, config, masks=masks)

        self.assertEqual(x_pilot.shape, x_dd.shape)
        self.assertEqual(x_pilot.dtype, x_dd.dtype)
        self.assertTrue(torch.all(x_pilot[:, config.pilot_delay, config.pilot_doppler] == 1.0 + 0.0j))
        self.assertTrue(torch.allclose(x_pilot[:, masks.guard_mask], torch.zeros_like(x_pilot[:, masks.guard_mask])))
        self.assertTrue(torch.allclose(x_pilot[:, masks.data_mask], x_dd[:, masks.data_mask]))

    def test_extract_pilot_observation_shape_and_indices(self):
        config = _pilot_config()
        y_dd = torch.randn(2, 6, 7, dtype=torch.complex64)

        observation = extract_pilot_observation(y_dd, config)

        self.assertEqual(observation.observation.shape, (2, 3, 3))
        self.assertTrue(torch.equal(observation.delay_indices.cpu(), torch.tensor([2, 3, 4])))
        self.assertTrue(torch.equal(observation.doppler_indices.cpu(), torch.tensor([2, 3, 4])))

    def test_pilot_sparse_channel_estimator_detects_largest_on_grid_tap(self):
        config = _pilot_config()
        y_dd = torch.zeros(1, 6, 7, dtype=torch.complex64)
        y_dd[0, 3, 3] = 0.5 + 0.0j
        y_dd[0, 2, 3] = 1.0 - 0.25j
        y_dd[0, 4, 4] = 2.0 + 1.0j
        estimator = PilotSparseChannelEstimator(config, topk_paths=3)

        estimate = estimator(y_dd)

        self.assertEqual(estimate.h_dd.shape, (1, 6, 7))
        self.assertEqual(estimate.support_mask.shape, (1, 6, 7))
        self.assertEqual(estimate.path_indices.shape, (1, 3, 2))
        self.assertEqual(estimate.path_gains.shape, (1, 3))
        self.assertEqual(estimate.confidence.shape, (1, 3))
        self.assertEqual(estimate.path_confidence_map.shape, (1, 6, 7))
        self.assertEqual(estimate.estimator_mode, "pilot")
        self.assertTrue(torch.equal(estimate.path_indices[0, 0].cpu(), torch.tensor([1, 1])))
        self.assertTrue(torch.allclose(estimate.path_gains[0, 0], torch.tensor(2.0 + 1.0j)))
        self.assertGreater(float(estimate.confidence[0, 0].item()), 0.99)

    def test_noise_var_creates_cfar_threshold(self):
        config = _pilot_config()
        y_dd = torch.zeros(1, 6, 7, dtype=torch.complex64)
        y_dd[0, 3, 3] = 0.1 + 0.0j
        estimator = PilotSparseChannelEstimator(config, topk_paths=3, noise_var=1.0)

        estimate = estimator(y_dd)

        self.assertEqual(float(estimate.support_mask.sum().item()), 0.0)
        self.assertTrue(torch.all(estimate.confidence == 0))
        self.assertTrue(torch.all(estimate.path_gains == 0))

    def test_all_zero_observation_has_no_valid_support(self):
        config = _pilot_config()
        y_dd = torch.zeros(2, 6, 7, dtype=torch.complex64)
        estimator = PilotSparseChannelEstimator(config, topk_paths=3)

        estimate = estimator(y_dd)

        self.assertEqual(float(estimate.support_mask.sum().item()), 0.0)
        self.assertTrue(torch.all(estimate.confidence == 0))
        self.assertTrue(torch.all(estimate.path_gains == 0))
        self.assertEqual(estimate.pilot_residual.shape, (2, 3, 3))
        self.assertEqual(estimate.pilot_residual_power.shape, (2,))

    def test_threshold_keeps_only_valid_paths(self):
        config = _pilot_config()
        y_dd = torch.zeros(1, 6, 7, dtype=torch.complex64)
        y_dd[0, 3, 3] = 0.2 + 0.0j
        y_dd[0, 2, 3] = 0.9 + 0.0j
        y_dd[0, 4, 4] = 1.3 + 0.0j
        estimator = PilotSparseChannelEstimator(config, topk_paths=3, threshold=1.0)

        estimate = estimator(y_dd)

        self.assertEqual(float(estimate.support_mask.sum().item()), 1.0)
        self.assertTrue(torch.equal(estimate.path_indices[0, 0].cpu(), torch.tensor([1, 1])))
        self.assertTrue(torch.allclose(estimate.path_gains[0, 0], torch.tensor(1.3 + 0.0j)))
        self.assertTrue(torch.all(estimate.confidence[0, 1:] == 0))

    def test_end_to_end_pilot_sparse_channel_recovery_with_dd_operator(self):
        config = _pilot_config()
        x_dd = torch.zeros(1, 6, 7, dtype=torch.complex64)
        x_pilot = insert_embedded_pilot(x_dd, config)
        path_indices = torch.tensor([[[1, 1], [0, 0]]], dtype=torch.long)
        path_gains = torch.tensor([[1.5 - 0.25j, 0.1 + 0.0j]], dtype=torch.complex64)
        y_dd = dd_circular_convolve_sparse(x_pilot, path_indices, path_gains)
        estimator = PilotSparseChannelEstimator(config, topk_paths=2, threshold=0.5)

        estimate = estimator(y_dd)

        self.assertTrue(torch.equal(estimate.path_indices[0, 0].cpu(), torch.tensor([1, 1])))
        self.assertTrue(torch.allclose(estimate.path_gains[0, 0], path_gains[0, 0], atol=1e-6, rtol=1e-6))

    def test_sparse_channel_estimator_modes(self):
        config = _config()
        estimator = SparseChannelEstimator(config)
        y_dd = torch.zeros(1, config.M, config.N, dtype=torch.complex64)
        h_dd = torch.zeros_like(y_dd)
        h_dd[0, 1, 2] = 3.0 + 0.5j

        oracle_estimate = estimator(y_dd, h_dd=h_dd, mode="oracle")
        fallback_estimate = estimator(y_dd, h_dd=h_dd, mode="fallback_topk")
        pilot_estimate = estimator(y_dd, mode="pilot", pilot_config=_pilot_config())

        self.assertEqual(oracle_estimate.estimator_mode, "oracle")
        self.assertTrue(torch.allclose(oracle_estimate.h_dd[0, 1, 2], h_dd[0, 1, 2]))
        self.assertEqual(fallback_estimate.estimator_mode, "fallback_topk")
        self.assertEqual(pilot_estimate.estimator_mode, "pilot")
        with self.assertRaises(ValueError):
            estimator(y_dd, mode="bad_mode")

    def test_learnable_receiver_default_and_pilot_modes_run(self):
        config = _config(use_refinement_net=True)
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)

        logits = model(y_dd)
        pilot_output = model(
            y_dd,
            channel_estimator_mode="pilot",
            pilot_config=_pilot_config(),
            return_details=True,
        )

        self.assertEqual(logits.shape, (2, config.vocab_size))
        self.assertEqual(pilot_output.token_logits.shape, (2, config.vocab_size))
        self.assertEqual(pilot_output.sparse_estimate.estimator_mode, "pilot")


if __name__ == "__main__":
    unittest.main()
