from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from receiver import ReceiverConfig
from receiver.dd_ops import (
    SparseDDOperator,
    dd_circular_convolve_dense_fft,
    dd_circular_convolve_sparse,
    dd_circular_correlation_sparse,
)
from receiver.sparse_channel import SparseChannelEstimate


def _config() -> ReceiverConfig:
    return ReceiverConfig(
        M=5,
        N=6,
        vocab_size=13,
        token_embedding_dim=8,
        num_unfolded_layers=2,
        topk_paths=4,
        hidden_channels=6,
        noise_var=0.1,
        use_refinement_net=False,
    )


def _complex_dd(batch: int, config: ReceiverConfig) -> torch.Tensor:
    real = torch.randn(batch, config.M, config.N)
    imag = torch.randn(batch, config.M, config.N)
    return torch.complex(real, imag)


def _inner_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.vdot(a.reshape(-1), b.reshape(-1))


class SparseDDOperatorTests(unittest.TestCase):
    def test_single_zero_path_is_identity(self):
        config = _config()
        x_dd = _complex_dd(2, config)
        path_indices = torch.zeros(2, 1, 2, dtype=torch.long)
        path_gains = torch.ones(2, 1, dtype=x_dd.dtype)

        y_dd = dd_circular_convolve_sparse(x_dd, path_indices, path_gains)

        self.assertTrue(torch.allclose(y_dd, x_dd))

    def test_single_shifted_path_matches_roll(self):
        config = _config()
        x_dd = _complex_dd(1, config)
        path_indices = torch.tensor([[[1, 2]]], dtype=torch.long)
        path_gains = torch.ones(1, 1, dtype=x_dd.dtype)

        y_dd = dd_circular_convolve_sparse(x_dd, path_indices, path_gains)
        expected = torch.roll(x_dd, shifts=(1, 2), dims=(1, 2))

        self.assertTrue(torch.allclose(y_dd, expected))

    def test_multipath_sparse_convolution_matches_manual_roll_sum(self):
        config = _config()
        x_dd = _complex_dd(1, config)
        path_indices = torch.tensor([[[0, 0], [1, 2], [4, 5]]], dtype=torch.long)
        path_gains = torch.tensor([[1.0 + 0.0j, 0.5 - 0.25j, -0.2 + 0.7j]], dtype=x_dd.dtype)

        y_dd = dd_circular_convolve_sparse(x_dd, path_indices, path_gains)
        expected = (
            path_gains[0, 0] * x_dd
            + path_gains[0, 1] * torch.roll(x_dd, shifts=(1, 2), dims=(1, 2))
            + path_gains[0, 2] * torch.roll(x_dd, shifts=(4, 5), dims=(1, 2))
        )

        self.assertTrue(torch.allclose(y_dd, expected))

    def test_dense_fft_convolution_matches_sparse_paths(self):
        config = _config()
        x_dd = _complex_dd(2, config)
        path_indices = torch.tensor(
            [
                [[0, 0], [1, 2], [3, 4]],
                [[0, 1], [2, 3], [4, 5]],
            ],
            dtype=torch.long,
        )
        path_gains = torch.tensor(
            [
                [1.0 + 0.0j, 0.4 + 0.2j, -0.1 + 0.3j],
                [0.7 - 0.1j, -0.5 + 0.6j, 0.2 + 0.2j],
            ],
            dtype=x_dd.dtype,
        )
        h_dd = torch.zeros_like(x_dd)
        for batch_idx in range(x_dd.shape[0]):
            for path_idx in range(path_indices.shape[1]):
                delay, doppler = path_indices[batch_idx, path_idx]
                h_dd[batch_idx, delay, doppler] = path_gains[batch_idx, path_idx]

        sparse_y = dd_circular_convolve_sparse(x_dd, path_indices, path_gains)
        dense_y = dd_circular_convolve_dense_fft(x_dd, h_dd)

        self.assertTrue(torch.allclose(sparse_y, dense_y, atol=1e-5, rtol=1e-5))

    def test_sparse_correlation_is_adjoint(self):
        config = _config()
        x_dd = _complex_dd(2, config)
        y_dd = _complex_dd(2, config)
        path_indices = torch.tensor(
            [
                [[0, 0], [1, 2], [3, 1]],
                [[2, 4], [1, 1], [0, 5]],
            ],
            dtype=torch.long,
        )
        path_gains = torch.tensor(
            [
                [1.0 + 0.5j, -0.3 + 0.2j, 0.1 - 0.4j],
                [0.6 - 0.1j, 0.2 + 0.8j, -0.7 + 0.3j],
            ],
            dtype=x_dd.dtype,
        )

        hx = dd_circular_convolve_sparse(x_dd, path_indices, path_gains)
        hhy = dd_circular_correlation_sparse(y_dd, path_indices, path_gains)

        self.assertTrue(torch.allclose(_inner_product(hx, y_dd), _inner_product(x_dd, hhy), atol=1e-5, rtol=1e-5))

    def test_sparse_operator_forward_estimate_and_apply(self):
        config = _config()
        operator = SparseDDOperator(config)
        x_dd = _complex_dd(1, config)
        h_dd = torch.zeros(1, config.M, config.N, dtype=x_dd.dtype)
        h_dd[0, 1, 2] = 1.0 + 0.0j
        support_mask = torch.zeros(1, config.M, config.N)
        support_mask[0, 1, 2] = 1.0
        path_indices = torch.tensor([[[1, 2]]], dtype=torch.long)
        path_gains = torch.tensor([[1.0 + 0.0j]], dtype=x_dd.dtype)
        confidence = torch.ones(1, 1)
        estimate = SparseChannelEstimate(
            h_dd=h_dd,
            support_mask=support_mask,
            path_indices=path_indices,
            path_gains=path_gains,
            confidence=confidence,
        )

        state = operator(estimate)
        y_dd = operator.apply(x_dd, state)

        self.assertEqual(state.path_indices.shape, (1, 1, 2))
        self.assertEqual(state.path_gains.shape, (1, 1))
        self.assertTrue(torch.allclose(y_dd, torch.roll(x_dd, shifts=(1, 2), dims=(1, 2))))


if __name__ == "__main__":
    unittest.main()
