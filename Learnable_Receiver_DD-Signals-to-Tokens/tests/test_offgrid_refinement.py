from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from receiver import ReceiverConfig
from receiver.offgrid_refinement import (
    LocalPatchExtractor,
    OffGridPathRefinement,
    OffGridRefinementNet,
    apply_offgrid_refinement,
)
from receiver.sparse_channel import SparseChannelEstimate, SparseChannelEstimator


def _config(use_offgrid_refinement: bool = False) -> ReceiverConfig:
    return ReceiverConfig(
        M=6,
        N=7,
        vocab_size=13,
        token_embedding_dim=8,
        num_unfolded_layers=2,
        topk_paths=3,
        hidden_channels=6,
        noise_var=0.1,
        use_refinement_net=False,
        use_offgrid_refinement=use_offgrid_refinement,
        offgrid_patch_delay_radius=1,
        offgrid_patch_doppler_radius=2,
        offgrid_max_offset=0.4,
        offgrid_confidence_floor=0.0,
        offgrid_gain_correction_scale=0.1,
    )


def _complex_dd(batch: int, config: ReceiverConfig) -> torch.Tensor:
    real = torch.randn(batch, config.M, config.N)
    imag = torch.randn(batch, config.M, config.N)
    return torch.complex(real, imag)


def _path_inputs(batch: int = 2, paths: int = 3) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    path_indices = torch.tensor(
        [
            [[0, 0], [1, 2], [5, 6]],
            [[2, 3], [4, 1], [0, 6]],
        ],
        dtype=torch.long,
    )[:batch, :paths]
    path_gains = torch.randn(batch, paths, dtype=torch.complex64)
    confidence = torch.ones(batch, paths)
    return path_indices, path_gains, confidence


def _estimate(config: ReceiverConfig) -> SparseChannelEstimate:
    h_dd = torch.zeros(1, config.M, config.N, dtype=torch.complex64)
    path_indices = torch.tensor([[[1, 2], [3, 4], [0, 0]]], dtype=torch.long)
    path_gains = torch.tensor([[1.0 + 0.5j, -0.2 + 0.7j, 0.0 + 0.0j]], dtype=torch.complex64)
    confidence = torch.tensor([[1.0, 0.5, 0.0]])
    support_mask = torch.zeros(1, config.M, config.N)
    for path_pos in range(path_indices.shape[1]):
        delay, doppler = path_indices[0, path_pos]
        h_dd[0, delay, doppler] = path_gains[0, path_pos]
        if confidence[0, path_pos] > 0:
            support_mask[0, delay, doppler] = 1.0
    return SparseChannelEstimate(
        h_dd=h_dd,
        support_mask=support_mask,
        path_indices=path_indices,
        path_gains=path_gains,
        confidence=confidence,
        estimator_mode="oracle",
    )


class OffGridRefinementTests(unittest.TestCase):
    def test_local_patch_extractor_shape(self):
        config = _config()
        h_coarse = _complex_dd(2, config)
        path_indices, _, confidence = _path_inputs(batch=2, paths=3)
        extractor = LocalPatchExtractor(radius_delay=1, radius_doppler=2)

        patch_features = extractor(h_coarse, path_indices, confidence=confidence)

        self.assertEqual(patch_features.shape, (2, 3, 6, 3, 5))
        self.assertTrue(patch_features.dtype.is_floating_point)

    def test_offsets_are_within_range(self):
        config = _config()
        h_coarse = _complex_dd(2, config)
        path_indices, path_gains, confidence = _path_inputs(batch=2, paths=3)
        net = OffGridRefinementNet(
            radius_delay=1,
            radius_doppler=1,
            max_offset=0.25,
            confidence_floor=0.0,
            gain_correction_scale=0.1,
            hidden_channels=6,
        )

        refinement = net(h_coarse, path_indices, path_gains, confidence=confidence)

        self.assertEqual(refinement.fractional_offsets.shape, (2, 3, 2))
        self.assertTrue(torch.all(refinement.fractional_offsets <= 0.25))
        self.assertTrue(torch.all(refinement.fractional_offsets >= -0.25))

    def test_gain_correction_is_complex_with_path_shape(self):
        config = _config()
        h_coarse = _complex_dd(2, config)
        path_indices, path_gains, confidence = _path_inputs(batch=2, paths=3)
        net = OffGridRefinementNet(1, 1, 0.5, 0.0, 0.1, hidden_channels=6)

        refinement = net(h_coarse, path_indices, path_gains, confidence=confidence)

        self.assertEqual(refinement.gain_correction.shape, (2, 3))
        self.assertTrue(torch.is_complex(refinement.gain_correction))

    def test_low_confidence_masks_updates(self):
        config = _config()
        h_coarse = _complex_dd(2, config)
        path_indices, path_gains, _ = _path_inputs(batch=2, paths=3)
        confidence = torch.zeros(2, 3)
        net = OffGridRefinementNet(1, 1, 0.5, 0.0, 0.1, hidden_channels=6)

        refinement = net(h_coarse, path_indices, path_gains, confidence=confidence)

        self.assertTrue(torch.all(refinement.fractional_offsets == 0))
        self.assertTrue(torch.all(refinement.gain_correction == 0))
        self.assertTrue(torch.all(refinement.confidence_update == 0))

    def test_apply_offgrid_refinement_preserves_sparse_shapes(self):
        config = _config()
        estimate = _estimate(config)
        fractional_offsets = torch.full((1, config.topk_paths, 2), 0.1)
        gain_correction = torch.full((1, config.topk_paths), 0.01 + 0.02j, dtype=torch.complex64)
        confidence_update = torch.tensor([[1.0, 0.5, 0.0]])
        refinement = OffGridPathRefinement(
            fractional_offsets=fractional_offsets,
            gain_correction=gain_correction,
            confidence_update=confidence_update,
            patch_features=None,
        )

        refined = apply_offgrid_refinement(estimate, refinement)

        self.assertEqual(refined.fractional_offsets.shape, (1, config.topk_paths, 2))
        self.assertEqual(refined.path_indices.shape, estimate.path_indices.shape)
        self.assertEqual(refined.support_mask.shape, estimate.support_mask.shape)
        self.assertEqual(refined.path_gains.shape, estimate.path_gains.shape)

    def test_sparse_channel_estimator_offgrid_integration(self):
        offgrid_config = _config(use_offgrid_refinement=True)
        plain_config = _config(use_offgrid_refinement=False)
        h_dd = torch.zeros(1, offgrid_config.M, offgrid_config.N, dtype=torch.complex64)
        h_dd[0, 1, 2] = 1.0 + 0.5j
        h_dd[0, 2, 3] = 0.8 - 0.1j
        y_dd = torch.zeros_like(h_dd)

        offgrid_estimate = SparseChannelEstimator(offgrid_config)(y_dd, h_dd=h_dd, mode="oracle")
        plain_estimate = SparseChannelEstimator(plain_config)(y_dd, h_dd=h_dd, mode="oracle")

        self.assertIsNotNone(offgrid_estimate.fractional_offsets)
        self.assertEqual(offgrid_estimate.fractional_offsets.shape, (1, offgrid_config.topk_paths, 2))
        self.assertIsNone(plain_estimate.fractional_offsets)
        self.assertEqual(offgrid_estimate.estimator_mode, "oracle")

    def test_backward_flows_to_refinement_net_parameters(self):
        config = _config()
        h_coarse = _complex_dd(2, config)
        path_indices, path_gains, confidence = _path_inputs(batch=2, paths=3)
        net = OffGridRefinementNet(1, 1, 0.5, 0.0, 0.1, hidden_channels=6)

        refinement = net(h_coarse, path_indices, path_gains, confidence=confidence)
        loss = refinement.fractional_offsets.pow(2).mean() + refinement.gain_correction.abs().mean()
        loss.backward()

        grad_sum = sum(
            float(param.grad.abs().sum().item())
            for param in net.parameters()
            if param.grad is not None
        )
        self.assertGreater(grad_sum, 0.0)


if __name__ == "__main__":
    unittest.main()
