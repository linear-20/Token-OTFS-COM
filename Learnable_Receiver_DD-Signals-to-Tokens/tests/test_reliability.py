from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from receiver.reliability import build_symbol_reliability_map


class SymbolReliabilityTests(unittest.TestCase):
    def test_output_shape(self):
        data_mask = torch.ones(2, 4, 5)
        path_confidence = torch.tensor([[1.0, 0.5, 0.0], [0.25, 0.0, 0.0]])

        reliability = build_symbol_reliability_map(data_mask, path_confidence=path_confidence)

        self.assertEqual(reliability.shape, (2, 4, 5))

    def test_all_ones_data_mask_is_finite_and_in_unit_interval(self):
        data_mask = torch.ones(2, 4, 5)
        path_confidence = torch.rand(2, 3)

        reliability = build_symbol_reliability_map(data_mask, path_confidence=path_confidence)

        self.assertFalse(torch.isnan(reliability).any())
        self.assertFalse(torch.isinf(reliability).any())
        self.assertGreaterEqual(float(reliability.min().item()), 0.0)
        self.assertLessEqual(float(reliability.max().item()), 1.0)

    def test_sparse_path_confidence_map_is_reduced_to_global_reliability(self):
        data_mask = torch.ones(1, 4, 5)
        path_confidence_map = torch.zeros(1, 4, 5)
        path_confidence_map[:, 0, 0] = 1.0
        path_confidence_map[:, 2, 3] = 0.5

        reliability = build_symbol_reliability_map(
            data_mask,
            path_confidence_map=path_confidence_map,
        )

        self.assertTrue(torch.all(reliability > 0.0))
        self.assertTrue(torch.allclose(reliability, torch.full_like(reliability, 0.75)))

    def test_higher_residual_energy_lowers_reliability(self):
        data_mask = torch.ones(1, 3, 4)
        low = build_symbol_reliability_map(data_mask, residual_energy=torch.full((1, 3, 4), 0.1))
        high = build_symbol_reliability_map(data_mask, residual_energy=torch.full((1, 3, 4), 10.0))

        self.assertTrue(torch.all(high < low))

    def test_higher_uncertainty_lowers_reliability(self):
        data_mask = torch.ones(1, 3, 4)
        low = build_symbol_reliability_map(data_mask, uncertainty_map=torch.full((1, 3, 4), 0.1))
        high = build_symbol_reliability_map(data_mask, uncertainty_map=torch.full((1, 3, 4), 5.0))

        self.assertTrue(torch.all(high < low))

    def test_higher_posterior_variance_lowers_reliability(self):
        data_mask = torch.ones(1, 3, 4)
        low = build_symbol_reliability_map(data_mask, posterior_variance=torch.full((1, 3, 4), 0.1))
        high = build_symbol_reliability_map(data_mask, posterior_variance=torch.full((1, 3, 4), 5.0))

        self.assertTrue(torch.all(high < low))

    def test_zero_data_mask_positions_have_zero_reliability(self):
        data_mask = torch.ones(1, 3, 4)
        data_mask[:, 1, 2] = 0.0

        reliability = build_symbol_reliability_map(data_mask, path_confidence=torch.ones(1, 2))

        self.assertEqual(float(reliability[:, 1, 2].item()), 0.0)
        self.assertGreater(float(reliability[:, 0, 0].item()), 0.0)


if __name__ == "__main__":
    unittest.main()
