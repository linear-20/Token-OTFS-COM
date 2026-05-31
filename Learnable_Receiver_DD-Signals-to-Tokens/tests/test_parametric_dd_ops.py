from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from receiver import ReceiverConfig
from receiver.dd_ops import (
    SparseDDOperator,
    dd_circular_convolve_offgrid_sparse,
    dd_circular_convolve_sparse,
    dd_circular_correlation_offgrid_sparse,
)
from receiver.sparse_channel import SparseChannelEstimate


def _config(**kwargs) -> ReceiverConfig:
    values = dict(
        M=5,
        N=6,
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


def _complex_dd(batch: int, config: ReceiverConfig) -> torch.Tensor:
    real = torch.randn(batch, config.M, config.N)
    imag = torch.randn(batch, config.M, config.N)
    return torch.complex(real, imag)


def _inner_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.vdot(a.reshape(-1), b.reshape(-1))


def _estimate(config: ReceiverConfig, fractional_offsets: torch.Tensor | None = None) -> SparseChannelEstimate:
    h_dd = torch.zeros(1, config.M, config.N, dtype=torch.complex64)
    support_mask = torch.zeros(1, config.M, config.N)
    path_indices = torch.tensor([[[1, 2], [0, 0]]], dtype=torch.long)
    path_gains = torch.tensor([[1.0 + 0.25j, 0.5 - 0.1j]], dtype=torch.complex64)
    confidence = torch.ones(1, 2)
    for path_pos in range(path_indices.shape[1]):
        delay, doppler = path_indices[0, path_pos]
        h_dd[0, delay, doppler] = path_gains[0, path_pos]
        support_mask[0, delay, doppler] = 1.0
    return SparseChannelEstimate(
        h_dd=h_dd,
        support_mask=support_mask,
        path_indices=path_indices,
        path_gains=path_gains,
        confidence=confidence,
        fractional_offsets=fractional_offsets,
    )


class ParametricDDOperatorTests(unittest.TestCase):
    def test_offgrid_zero_offsets_matches_ongrid(self):
        config = _config()
        x_dd = _complex_dd(2, config)
        path_indices = torch.tensor(
            [
                [[0, 0], [1, 2]],
                [[2, 3], [4, 5]],
            ],
            dtype=torch.long,
        )
        path_gains = torch.tensor(
            [
                [1.0 + 0.1j, -0.2 + 0.5j],
                [0.3 - 0.4j, 0.8 + 0.2j],
            ],
            dtype=x_dd.dtype,
        )
        fractional_offsets = torch.zeros(2, 2, 2)

        ongrid = dd_circular_convolve_sparse(x_dd, path_indices, path_gains)
        offgrid = dd_circular_convolve_offgrid_sparse(
            x_dd,
            path_indices,
            path_gains,
            fractional_offsets,
            kernel_radius=1,
            kernel_type="linear",
        )

        self.assertTrue(torch.allclose(offgrid, ongrid, atol=1e-6, rtol=1e-6))

    def test_offgrid_small_offset_spreads_energy(self):
        config = _config()
        x_dd = _complex_dd(1, config)
        path_indices = torch.tensor([[[1, 2]]], dtype=torch.long)
        path_gains = torch.tensor([[1.0 + 0.0j]], dtype=x_dd.dtype)
        fractional_offsets = torch.tensor([[[0.25, 0.25]]], dtype=torch.float32)

        ongrid = dd_circular_convolve_sparse(x_dd, path_indices, path_gains)
        offgrid = dd_circular_convolve_offgrid_sparse(
            x_dd,
            path_indices,
            path_gains,
            fractional_offsets,
            kernel_radius=1,
            kernel_type="linear",
        )

        self.assertFalse(torch.allclose(offgrid, ongrid, atol=1e-5, rtol=1e-5))
        self.assertFalse(torch.isnan(offgrid).any())
        self.assertFalse(torch.isinf(offgrid).any())
        self.assertGreater(float(offgrid.abs().pow(2).sum().item()), 0.0)

    def test_offgrid_correlation_is_adjoint(self):
        config = _config()
        x_dd = _complex_dd(2, config)
        y_dd = _complex_dd(2, config)
        path_indices = torch.tensor(
            [
                [[0, 0], [1, 2], [4, 5]],
                [[2, 3], [3, 1], [0, 5]],
            ],
            dtype=torch.long,
        )
        path_gains = torch.tensor(
            [
                [1.0 + 0.2j, -0.3 + 0.4j, 0.2 - 0.1j],
                [0.5 - 0.2j, 0.7 + 0.1j, -0.4 + 0.6j],
            ],
            dtype=x_dd.dtype,
        )
        fractional_offsets = torch.tensor(
            [
                [[0.2, -0.1], [0.25, 0.3], [-0.2, 0.1]],
                [[-0.1, 0.2], [0.3, -0.25], [0.1, 0.1]],
            ],
            dtype=torch.float32,
        )

        hx = dd_circular_convolve_offgrid_sparse(
            x_dd,
            path_indices,
            path_gains,
            fractional_offsets,
            kernel_radius=1,
            kernel_type="linear",
        )
        hhy = dd_circular_correlation_offgrid_sparse(
            y_dd,
            path_indices,
            path_gains,
            fractional_offsets,
            kernel_radius=1,
            kernel_type="linear",
        )

        self.assertTrue(torch.allclose(_inner_product(hx, y_dd), _inner_product(x_dd, hhy), atol=1e-4, rtol=1e-4))

    def test_sparse_operator_auto_uses_offgrid_when_offsets_present(self):
        config = _config(dd_operator_mode="auto", offgrid_kernel_radius=1)
        operator = SparseDDOperator(config)
        x_dd = _complex_dd(1, config)
        fractional_offsets = torch.tensor([[[0.1, 0.2], [0.0, 0.0]]], dtype=torch.float32)
        estimate = _estimate(config, fractional_offsets=fractional_offsets)

        state = operator(estimate)
        y_dd = operator.apply(x_dd, state)

        self.assertEqual(state.operator_mode, "offgrid")
        self.assertEqual(y_dd.shape, x_dd.shape)

    def test_sparse_operator_forced_offgrid_requires_offsets(self):
        config = _config(dd_operator_mode="offgrid")
        operator = SparseDDOperator(config)
        estimate = _estimate(config, fractional_offsets=None)

        with self.assertRaises(ValueError):
            operator(estimate)

    def test_forced_ongrid_ignores_fractional_offsets(self):
        config = _config(dd_operator_mode="ongrid")
        operator = SparseDDOperator(config)
        x_dd = _complex_dd(1, config)
        fractional_offsets = torch.tensor([[[0.25, 0.25], [0.2, -0.2]]], dtype=torch.float32)
        estimate = _estimate(config, fractional_offsets=fractional_offsets)

        state = operator(estimate)
        y_dd = operator.apply(x_dd, state)
        expected = dd_circular_convolve_sparse(
            x_dd,
            estimate.path_indices,
            estimate.path_gains,
            confidence=estimate.confidence,
        )

        self.assertEqual(state.operator_mode, "ongrid")
        self.assertTrue(torch.allclose(y_dd, expected, atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
