from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from receiver import ReceiverConfig
from receiver.sparse_channel import (
    SparseChannelEstimator,
    ThresholdSparseEstimator,
    TopKSparseEstimator,
    _build_pilot_observation_mask,
    channel_nmse,
    sparsity_l1_loss,
    support_bce_loss,
)
from receiver.pilot_ce import EmbeddedPilotConfig


def _config(use_refinement_net: bool = False, use_channel_refinement: bool = False, **kwargs) -> ReceiverConfig:
    values = dict(
        M=5,
        N=4,
        vocab_size=11,
        token_embedding_dim=8,
        num_unfolded_layers=2,
        topk_paths=3,
        hidden_channels=6,
        noise_var=0.1,
        use_refinement_net=use_refinement_net,
        use_channel_refinement=use_channel_refinement,
    )
    values.update(kwargs)
    return ReceiverConfig(**values)


def _complex_dd(batch: int, config: ReceiverConfig) -> torch.Tensor:
    real = torch.randn(batch, config.M, config.N)
    imag = torch.randn(batch, config.M, config.N)
    return torch.complex(real, imag)


class SparseChannelTests(unittest.TestCase):
    def test_topk_sparse_estimator_shapes(self):
        config = _config()
        estimator = TopKSparseEstimator(config)
        coarse_h_dd = _complex_dd(2, config)

        estimate = estimator(coarse_h_dd)

        self.assertEqual(estimate.h_dd.shape, (2, config.M, config.N))
        self.assertEqual(estimate.support_mask.shape, (2, config.M, config.N))
        self.assertEqual(estimate.path_indices.shape, (2, config.topk_paths, 2))
        self.assertEqual(estimate.path_gains.shape, (2, config.topk_paths))
        self.assertTrue(torch.is_complex(estimate.path_gains))

    def test_topk_support_mask_has_exactly_k_entries_per_batch(self):
        config = _config()
        estimator = TopKSparseEstimator(config)
        coarse_h_dd = _complex_dd(4, config)

        estimate = estimator(coarse_h_dd)

        support_counts = estimate.support_mask.sum(dim=(1, 2))
        self.assertTrue(torch.all(support_counts == config.topk_paths))

    def test_threshold_sparse_estimator_pads_when_fewer_than_k(self):
        config = _config()
        estimator = ThresholdSparseEstimator(config)
        coarse_h_dd = torch.zeros(1, config.M, config.N, dtype=torch.complex64)
        coarse_h_dd[0, 1, 2] = torch.tensor(3.0 + 1.0j)

        estimate = estimator(coarse_h_dd, threshold=1.0)

        self.assertEqual(estimate.path_indices.shape, (1, config.topk_paths, 2))
        self.assertEqual(estimate.path_gains.shape, (1, config.topk_paths))
        self.assertEqual(int(estimate.support_mask.sum().item()), 1)
        self.assertEqual(int((estimate.confidence > 0).sum().item()), 1)
        self.assertTrue(torch.all(estimate.confidence[0, 1:] == 0))
        self.assertTrue(torch.all(estimate.path_gains[0, 1:] == 0))

    def test_sparse_channel_estimator_accepts_h_dd_and_support_mask(self):
        config = _config(use_refinement_net=True, use_channel_refinement=True)
        estimator = SparseChannelEstimator(config)
        y_dd = _complex_dd(2, config)
        h_dd = _complex_dd(2, config)
        support_mask = torch.zeros(2, config.M, config.N)
        support_mask[:, 0, 0] = 1.0
        support_mask[:, 2, 3] = 1.0

        estimate = estimator(y_dd, h_dd=h_dd, support_mask=support_mask, snr_db=torch.tensor([10.0, 12.0]))

        self.assertEqual(estimate.h_dd.shape, (2, config.M, config.N))
        self.assertEqual(estimate.support_mask.shape, (2, config.M, config.N))
        self.assertEqual(estimate.path_indices.shape, (2, config.topk_paths, 2))
        self.assertEqual(estimate.path_gains.shape, (2, config.topk_paths))
        self.assertIsNotNone(estimate.support_logits)
        self.assertEqual(estimate.support_logits.shape, (2, config.M, config.N))

    def test_channel_refinement_flag_is_separate(self):
        denoiser_only_config = _config(use_refinement_net=True, use_channel_refinement=False)
        channel_refine_config = _config(use_refinement_net=False, use_channel_refinement=True)

        denoiser_only_estimator = SparseChannelEstimator(denoiser_only_config)
        channel_refine_estimator = SparseChannelEstimator(channel_refine_config)

        self.assertIsNone(denoiser_only_estimator.refinement_net)
        self.assertIsNotNone(channel_refine_estimator.refinement_net)

    def test_channel_nmse_identical_tensor_is_zero(self):
        config = _config()
        h_dd = _complex_dd(2, config)

        nmse = channel_nmse(h_dd, h_dd.clone())

        self.assertLess(float(nmse.item()), 1e-8)

    def test_sparse_losses_return_scalars(self):
        config = _config()
        h_dd = _complex_dd(2, config)
        logits = torch.randn(2, config.M, config.N)
        support = torch.randint(0, 2, (2, config.M, config.N)).float()

        self.assertEqual(sparsity_l1_loss(h_dd).ndim, 0)
        self.assertEqual(support_bce_loss(logits, support).ndim, 0)

    def test_physics_guided_gain_refinement_updates_estimate_fields(self):
        config = _config(use_physics_guided_gain_refinement=True, physics_refinement_ridge=1e-4,
                         physics_refinement_allow_full_grid_debug=True)
        estimator = SparseChannelEstimator(config)
        y_dd = torch.zeros(2, config.M, config.N, dtype=torch.complex64)
        y_dd[:, 1, 2] = 2.0 - 0.5j
        y_dd[:, 3, 1] = 0.5 + 0.25j

        estimate = estimator(y_dd)

        self.assertTrue(estimate.physics_refined)
        self.assertIsNotNone(estimate.physics_residual)
        self.assertIsNotNone(estimate.physics_residual_power)
        self.assertIsNotNone(estimate.physics_fit_residual_power)
        self.assertIsNotNone(estimate.physics_full_residual_power)
        self.assertIsNotNone(estimate.gain_covariance_diag)
        self.assertIsNotNone(estimate.physics_normal_diag)
        self.assertEqual(estimate.physics_residual.shape, (2, config.M, config.N))
        self.assertEqual(estimate.physics_residual_power.shape, (2,))
        self.assertEqual(estimate.physics_fit_residual_power.shape, (2,))
        self.assertEqual(estimate.physics_full_residual_power.shape, (2,))
        self.assertEqual(estimate.gain_covariance_diag.shape, (2, config.topk_paths))
        self.assertEqual(estimate.physics_normal_diag.shape, (2, config.topk_paths))
        self.assertEqual(estimate.confidence.shape, (2, config.topk_paths))

    def test_default_physics_refinement_is_disabled(self):
        config = _config()
        estimator = SparseChannelEstimator(config)
        y_dd = _complex_dd(1, config)

        estimate = estimator(y_dd)

        self.assertFalse(estimate.physics_refined)
        self.assertIsNone(estimate.physics_residual)
        self.assertIsNone(estimate.physics_residual_power)

    def test_pilot_mode_physics_refinement_uses_observation_mask(self):
        config = _config(use_physics_guided_gain_refinement=True, physics_refinement_ridge=1e-4)
        pilot_cfg = EmbeddedPilotConfig(
            pilot_delay=2, pilot_doppler=1,
            guard_delay=2, guard_doppler=2,
            obs_delay_radius=1, obs_doppler_radius=1,
        )
        estimator = SparseChannelEstimator(config)
        y_dd = torch.zeros(2, config.M, config.N, dtype=torch.complex64)
        y_dd[:, 2, 1] = 1.0 + 0.0j
        y_dd[:, 1, 1] = 0.5 + 0.0j

        estimate = estimator(y_dd, mode="pilot", pilot_config=pilot_cfg)

        self.assertTrue(estimate.physics_refined)
        self.assertEqual(estimate.ce_fit_mask_source, "pilot_observation")

    def test_no_pilot_no_ce_mask_skips_physics_refinement_by_default(self):
        config = _config(use_physics_guided_gain_refinement=True,
                         physics_refinement_require_fit_mask=True,
                         physics_refinement_allow_full_grid_debug=False)
        estimator = SparseChannelEstimator(config)
        y_dd = _complex_dd(1, config)

        estimate = estimator(y_dd, mode="fallback_topk")

        self.assertFalse(estimate.physics_refined)
        self.assertEqual(estimate.ce_fit_mask_source, "none")

    def test_full_grid_debug_flag_allows_fallback(self):
        config = _config(use_physics_guided_gain_refinement=True,
                         physics_refinement_require_fit_mask=True,
                         physics_refinement_allow_full_grid_debug=True)
        estimator = SparseChannelEstimator(config)
        y_dd = _complex_dd(1, config)

        estimate = estimator(y_dd, mode="fallback_topk")

        self.assertTrue(estimate.physics_refined)
        self.assertEqual(estimate.ce_fit_mask_source, "full_grid_debug")

    def test_explicit_ce_fit_mask_takes_priority(self):
        config = _config(use_physics_guided_gain_refinement=True)
        pilot_cfg = EmbeddedPilotConfig(
            pilot_delay=2, pilot_doppler=1,
            guard_delay=2, guard_doppler=2,
            obs_delay_radius=1, obs_doppler_radius=1,
        )
        estimator = SparseChannelEstimator(config)
        y_dd = _complex_dd(2, config)
        explicit_mask = torch.zeros(1, config.M, config.N)
        explicit_mask[:, 1, 1] = 1.0

        estimate = estimator(y_dd, mode="pilot", pilot_config=pilot_cfg, ce_fit_mask=explicit_mask)

        self.assertTrue(estimate.physics_refined)
        self.assertEqual(estimate.ce_fit_mask_source, "explicit_ce_data_mask")

    def test_build_pilot_observation_mask_covers_obs_window(self):
        config = EmbeddedPilotConfig(
            pilot_delay=2, pilot_doppler=1,
            guard_delay=2, guard_doppler=2,
            obs_delay_radius=1, obs_doppler_radius=1,
        )
        mask = _build_pilot_observation_mask((1, 5, 4), config)
        self.assertEqual(mask.shape, (1, 5, 4))
        self.assertGreater(float(mask.sum().item()), 0.0)
        self.assertEqual(float(mask[0, 2, 1].item()), 1.0)


if __name__ == "__main__":
    unittest.main()
