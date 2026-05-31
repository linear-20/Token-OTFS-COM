from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from receiver import LearnableOTFSReceiver, ReceiverConfig, token_cross_entropy
from receiver.complex_utils import channels_to_complex, complex_to_channels
from receiver.sparse_channel import SparseChannelEstimator


def _config() -> ReceiverConfig:
    return ReceiverConfig(
        M=8,
        N=6,
        vocab_size=17,
        token_embedding_dim=12,
        num_unfolded_layers=2,
        topk_paths=5,
        hidden_channels=8,
        noise_var=0.05,
        use_refinement_net=True,
    )


def _complex_dd(batch: int, config: ReceiverConfig) -> torch.Tensor:
    real = torch.randn(batch, config.M, config.N)
    imag = torch.randn(batch, config.M, config.N)
    return torch.complex(real, imag)


class ReceiverShapeTests(unittest.TestCase):
    def test_package_import_and_forward_shape_without_optional_inputs(self):
        config = _config()
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(3, config)

        token_logits = model(y_dd)

        self.assertEqual(token_logits.shape, (3, config.vocab_size))
        self.assertTrue(token_logits.dtype.is_floating_point)

    def test_forward_shape_with_h_dd_and_support_mask(self):
        config = _config()
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)
        h_dd = _complex_dd(2, config)
        support_mask = torch.zeros(2, config.M, config.N, dtype=torch.bool)
        support_mask[:, :2, :3] = True

        token_logits = model(y_dd, h_dd=h_dd, support_mask=support_mask)

        self.assertEqual(token_logits.shape, (2, config.vocab_size))
        self.assertTrue(token_logits.dtype.is_floating_point)

    def test_sparse_channel_estimator_topk_support_shape_and_count(self):
        config = _config()
        estimator = SparseChannelEstimator(config)
        y_dd = _complex_dd(4, config)

        estimate = estimator(y_dd)

        self.assertEqual(estimate.h_dd.shape, (4, config.M, config.N))
        self.assertEqual(estimate.support_mask.shape, (4, config.M, config.N))
        self.assertTrue(torch.all(estimate.support_mask.sum(dim=(1, 2)) == config.topk_paths))

    def test_complex_channel_conversion_roundtrip_shape(self):
        config = _config()
        x_dd = _complex_dd(2, config)

        channels = complex_to_channels(x_dd)
        recovered = channels_to_complex(channels)

        self.assertEqual(channels.shape, (2, 2, config.M, config.N))
        self.assertEqual(recovered.shape, x_dd.shape)
        self.assertTrue(torch.allclose(recovered, x_dd))

    def test_token_cross_entropy_shape(self):
        config = _config()
        logits = torch.randn(5, config.vocab_size)
        targets = torch.randint(0, config.vocab_size, (5,))

        loss = token_cross_entropy(logits, targets)

        self.assertEqual(loss.ndim, 0)


if __name__ == "__main__":
    unittest.main()
