"""Tests for candidate-pool hard-negative mining (Step 16B)."""

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
TRANSMITTER_ROOT = ROOT / "Learnable_Mapping_Tokens-to-DD-Signals"
sys.path.insert(0, str(TRANSMITTER_ROOT))

from transmitter import (
    SparseShiftSet,
    sample_uniform_cross_token_pairs,
)
from transmitter.ablations import (
    HardNegativeTokenPairs,
    cross_token_orbit_confusion_loss,
    cross_token_orbit_confusion_scores,
    mine_cross_token_hard_negatives,
    sample_weighted_sparse_shift_set,
)


def _complex_randn(*shape, dtype=torch.complex64):
    rd = {torch.complex64: torch.float32, torch.complex128: torch.float64}[dtype]
    return torch.randn(*shape, dtype=rd) + 1j * torch.randn(*shape, dtype=rd)


def _candidate_pool(V=6, P=12, seed=42):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return sample_uniform_cross_token_pairs(V, P, generator=gen)


# ============================================================================
# A. Dataclass tests
# ============================================================================

class DataclassTests(unittest.TestCase):

    def _valid_hntp(self, K=3, S=2):
        return HardNegativeTokenPairs(
            token_pairs=torch.tensor([[0, 1], [2, 3], [4, 5]],
                                     dtype=torch.long),
            source_indices=torch.tensor([0, 2, 4], dtype=torch.long),
            peak_scores=torch.tensor([0.9, 0.8, 0.7]),
            orbit_scores=torch.tensor(
                [[0.9, 0.3], [0.8, 0.2], [0.7, 0.1]],
            ),
        )

    def test_valid_construction(self):
        obj = self._valid_hntp()
        self.assertIsInstance(obj, HardNegativeTokenPairs)

    def test_token_pairs_shape_K2(self):
        obj = self._valid_hntp()
        self.assertEqual(obj.token_pairs.shape, (3, 2))

    def test_source_indices_shape_K(self):
        obj = self._valid_hntp()
        self.assertEqual(obj.source_indices.shape, (3,))

    def test_peak_scores_shape_K(self):
        obj = self._valid_hntp()
        self.assertEqual(obj.peak_scores.shape, (3,))

    def test_orbit_scores_shape_KS(self):
        obj = self._valid_hntp()
        self.assertEqual(obj.orbit_scores.shape, (3, 2))

    def test_dtype_correct(self):
        obj = self._valid_hntp()
        self.assertEqual(obj.token_pairs.dtype, torch.long)
        self.assertEqual(obj.source_indices.dtype, torch.long)
        self.assertTrue(obj.peak_scores.dtype.is_floating_point)
        self.assertTrue(obj.orbit_scores.dtype.is_floating_point)

    def test_device_consistent(self):
        obj = self._valid_hntp()
        dev = obj.token_pairs.device
        self.assertEqual(obj.source_indices.device, dev)
        self.assertEqual(obj.peak_scores.device, dev)
        self.assertEqual(obj.orbit_scores.device, dev)

    # -- reject: non-tensor ----------------------------------------------------

    def test_rejects_token_pairs_non_tensor(self):
        with self.assertRaises(TypeError):
            HardNegativeTokenPairs(
                token_pairs=[[0, 1], [2, 3]],
                source_indices=torch.tensor([0, 1], dtype=torch.long),
                peak_scores=torch.tensor([0.9, 0.8]),
                orbit_scores=torch.tensor([[0.9, 0.3], [0.8, 0.2]]),
            )

    def test_rejects_source_indices_non_tensor(self):
        with self.assertRaises(TypeError):
            HardNegativeTokenPairs(
                token_pairs=torch.tensor([[0, 1], [2, 3]], dtype=torch.long),
                source_indices=[0, 1],
                peak_scores=torch.tensor([0.9, 0.8]),
                orbit_scores=torch.tensor([[0.9, 0.3], [0.8, 0.2]]),
            )

    # -- reject: wrong dtype ---------------------------------------------------

    def test_rejects_token_pairs_wrong_dtype(self):
        with self.assertRaises(TypeError):
            HardNegativeTokenPairs(
                token_pairs=torch.tensor([[0, 1]], dtype=torch.int32),
                source_indices=torch.tensor([0], dtype=torch.long),
                peak_scores=torch.tensor([0.9]),
                orbit_scores=torch.tensor([[0.9, 0.3]]),
            )

    def test_rejects_source_indices_wrong_dtype(self):
        with self.assertRaises(TypeError):
            HardNegativeTokenPairs(
                token_pairs=torch.tensor([[0, 1]], dtype=torch.long),
                source_indices=torch.tensor([0], dtype=torch.int32),
                peak_scores=torch.tensor([0.9]),
                orbit_scores=torch.tensor([[0.9, 0.3]]),
            )

    def test_rejects_score_dtype_mismatch(self):
        with self.assertRaises(TypeError):
            HardNegativeTokenPairs(
                token_pairs=torch.tensor([[0, 1]], dtype=torch.long),
                source_indices=torch.tensor([0], dtype=torch.long),
                peak_scores=torch.tensor([0.9], dtype=torch.float32),
                orbit_scores=torch.tensor([[0.9, 0.3]], dtype=torch.float64),
            )

    # -- reject: wrong shape ---------------------------------------------------

    def test_rejects_wrong_shape_K_zero(self):
        with self.assertRaises(ValueError):
            HardNegativeTokenPairs(
                token_pairs=torch.empty(0, 2, dtype=torch.long),
                source_indices=torch.empty(0, dtype=torch.long),
                peak_scores=torch.empty(0),
                orbit_scores=torch.empty(0, 2),
            )

    def test_rejects_wrong_shape_S_zero(self):
        with self.assertRaises(ValueError):
            HardNegativeTokenPairs(
                token_pairs=torch.tensor([[0, 1]], dtype=torch.long),
                source_indices=torch.tensor([0], dtype=torch.long),
                peak_scores=torch.tensor([0.9]),
                orbit_scores=torch.empty(1, 0),
            )

    def test_rejects_peak_scores_wrong_shape(self):
        with self.assertRaises(ValueError):
            HardNegativeTokenPairs(
                token_pairs=torch.tensor([[0, 1]], dtype=torch.long),
                source_indices=torch.tensor([0], dtype=torch.long),
                peak_scores=torch.tensor([0.9, 0.8]),
                orbit_scores=torch.tensor([[0.9, 0.3]]),
            )

    # -- reject: device mismatch -----------------------------------------------

    def test_rejects_device_mismatch(self):
        try:
            peak_cuda = torch.tensor([0.9]).to("cuda")
        except (AssertionError, RuntimeError):
            self.skipTest("CUDA not available")
        with self.assertRaises(ValueError):
            HardNegativeTokenPairs(
                token_pairs=torch.tensor([[0, 1]], dtype=torch.long),
                source_indices=torch.tensor([0], dtype=torch.long),
                peak_scores=peak_cuda,
                orbit_scores=torch.tensor([[0.9, 0.3]]),
            )

    # -- reject: value violations ----------------------------------------------

    def test_rejects_negative_token_index(self):
        with self.assertRaises(ValueError):
            HardNegativeTokenPairs(
                token_pairs=torch.tensor([[-1, 0]], dtype=torch.long),
                source_indices=torch.tensor([0], dtype=torch.long),
                peak_scores=torch.tensor([0.9]),
                orbit_scores=torch.tensor([[0.9, 0.3]]),
            )

    def test_rejects_self_pair(self):
        with self.assertRaises(ValueError):
            HardNegativeTokenPairs(
                token_pairs=torch.tensor([[1, 1]], dtype=torch.long),
                source_indices=torch.tensor([0], dtype=torch.long),
                peak_scores=torch.tensor([0.9]),
                orbit_scores=torch.tensor([[0.9, 0.3]]),
            )

    def test_rejects_negative_source_index(self):
        with self.assertRaises(ValueError):
            HardNegativeTokenPairs(
                token_pairs=torch.tensor([[0, 1]], dtype=torch.long),
                source_indices=torch.tensor([-1], dtype=torch.long),
                peak_scores=torch.tensor([0.9]),
                orbit_scores=torch.tensor([[0.9, 0.3]]),
            )

    def test_rejects_nonfinite_peak_scores(self):
        with self.assertRaises(ValueError):
            HardNegativeTokenPairs(
                token_pairs=torch.tensor([[0, 1]], dtype=torch.long),
                source_indices=torch.tensor([0], dtype=torch.long),
                peak_scores=torch.tensor([float("nan")]),
                orbit_scores=torch.tensor([[0.9, 0.3]]),
            )

    def test_rejects_nonfinite_orbit_scores(self):
        with self.assertRaises(ValueError):
            HardNegativeTokenPairs(
                token_pairs=torch.tensor([[0, 1]], dtype=torch.long),
                source_indices=torch.tensor([0], dtype=torch.long),
                peak_scores=torch.tensor([0.9]),
                orbit_scores=torch.tensor([[float("inf"), 0.3]]),
            )

    def test_rejects_peak_mismatch_orbit_max(self):
        with self.assertRaises(ValueError):
            HardNegativeTokenPairs(
                token_pairs=torch.tensor([[0, 1]], dtype=torch.long),
                source_indices=torch.tensor([0], dtype=torch.long),
                peak_scores=torch.tensor([0.5]),
                orbit_scores=torch.tensor([[0.9, 0.3]]),
            )

    def test_rejects_requires_grad_peak_scores(self):
        with self.assertRaises(ValueError):
            HardNegativeTokenPairs(
                token_pairs=torch.tensor([[0, 1]], dtype=torch.long),
                source_indices=torch.tensor([0], dtype=torch.long),
                peak_scores=torch.tensor([0.9], requires_grad=True),
                orbit_scores=torch.tensor([[0.9, 0.3]]),
            )

    def test_rejects_requires_grad_orbit_scores(self):
        with self.assertRaises(ValueError):
            HardNegativeTokenPairs(
                token_pairs=torch.tensor([[0, 1]], dtype=torch.long),
                source_indices=torch.tensor([0], dtype=torch.long),
                peak_scores=torch.tensor([0.9]),
                orbit_scores=torch.tensor([[0.9, 0.3]],
                                          requires_grad=True),
            )

    def test_duplicates_legal(self):
        obj = HardNegativeTokenPairs(
            token_pairs=torch.tensor([[0, 1], [0, 1]], dtype=torch.long),
            source_indices=torch.tensor([0, 0], dtype=torch.long),
            peak_scores=torch.tensor([0.9, 0.9]),
            orbit_scores=torch.tensor([[0.9, 0.3], [0.9, 0.3]]),
        )
        self.assertEqual(obj.token_pairs.shape, (2, 2))

    def test_docstring_mentions_shapes(self):
        doc = HardNegativeTokenPairs.__doc__
        self.assertIn("[K, 2]", doc)
        self.assertIn("[K, S]", doc)
        self.assertIn("[K]", doc)


# ============================================================================
# B. Miner tests
# ============================================================================

class MinerTests(unittest.TestCase):

    def setUp(self):
        self.cw = _complex_randn(6, 8, 10)
        self.pairs = _candidate_pool(V=6, P=12)
        self.ss = SparseShiftSet(
            shifts=torch.tensor([[0, 0], [1, 0]], dtype=torch.long),
        )
        self.mask = torch.ones(1, 8, 10)

    def test_returns_hard_negative_token_pairs(self):
        result = mine_cross_token_hard_negatives(
            self.cw, self.pairs, self.ss, self.mask, num_hard_pairs=3,
        )
        self.assertIsInstance(result, HardNegativeTokenPairs)

    def test_output_token_pairs_shape_K2(self):
        result = mine_cross_token_hard_negatives(
            self.cw, self.pairs, self.ss, self.mask, num_hard_pairs=4,
        )
        self.assertEqual(result.token_pairs.shape, (4, 2))

    def test_peak_scores_descending_sorted(self):
        result = mine_cross_token_hard_negatives(
            self.cw, self.pairs, self.ss, self.mask, num_hard_pairs=5,
        )
        self.assertTrue(
            (result.peak_scores[:-1] >= result.peak_scores[1:]).all(),
        )

    def test_selected_pairs_match_candidate_pool(self):
        result = mine_cross_token_hard_negatives(
            self.cw, self.pairs, self.ss, self.mask, num_hard_pairs=4,
        )
        expected_pairs = self.pairs[result.source_indices]
        self.assertTrue(torch.equal(result.token_pairs, expected_pairs))

    def test_selected_orbit_matches_full_scores(self):
        result = mine_cross_token_hard_negatives(
            self.cw, self.pairs, self.ss, self.mask, num_hard_pairs=4,
        )
        full = cross_token_orbit_confusion_scores(
            self.cw, self.pairs, self.ss, self.mask,
        )
        expected = full[result.source_indices.cpu()]
        self.assertTrue(torch.allclose(
            result.orbit_scores.cpu(), expected, atol=1e-5,
        ))

    def test_peak_matches_orbit_max(self):
        result = mine_cross_token_hard_negatives(
            self.cw, self.pairs, self.ss, self.mask, num_hard_pairs=3,
        )
        computed = result.orbit_scores.max(dim=1).values
        self.assertTrue(torch.allclose(result.peak_scores, computed))

    def test_all_output_requires_grad_false(self):
        result = mine_cross_token_hard_negatives(
            self.cw, self.pairs, self.ss, self.mask, num_hard_pairs=3,
        )
        self.assertFalse(result.token_pairs.requires_grad)
        self.assertFalse(result.source_indices.requires_grad)
        self.assertFalse(result.peak_scores.requires_grad)
        self.assertFalse(result.orbit_scores.requires_grad)

    def test_output_on_codeword_book_device(self):
        result = mine_cross_token_hard_negatives(
            self.cw, self.pairs, self.ss, self.mask, num_hard_pairs=3,
        )
        self.assertEqual(result.token_pairs.device, self.cw.device)
        self.assertEqual(result.orbit_scores.device, self.cw.device)

    def test_does_not_modify_codeword_book(self):
        cw_copy = self.cw.clone()
        mine_cross_token_hard_negatives(
            self.cw, self.pairs, self.ss, self.mask, num_hard_pairs=3,
        )
        self.assertTrue(torch.equal(self.cw, cw_copy))

    def test_does_not_modify_candidate_pairs(self):
        pairs_copy = self.pairs.clone()
        mine_cross_token_hard_negatives(
            self.cw, self.pairs, self.ss, self.mask, num_hard_pairs=3,
        )
        self.assertTrue(torch.equal(self.pairs, pairs_copy))

    def test_duplicates_preserved(self):
        cw = torch.zeros(3, 1, 2, dtype=torch.complex64)
        cw[0, 0, 0] = 1.0 + 0j
        cw[1, 0, 0] = 1.0 + 0j
        cw[2, 0, 1] = 1.0 + 0j
        pairs = torch.tensor([[0, 1], [0, 1], [0, 2]], dtype=torch.long)
        ss = SparseShiftSet(
            shifts=torch.tensor([[0, 0]], dtype=torch.long),
        )
        result = mine_cross_token_hard_negatives(
            cw, pairs, ss, torch.ones(1, 1, 2), num_hard_pairs=2,
        )
        self.assertTrue(torch.equal(
            result.token_pairs,
            torch.tensor([[0, 1], [0, 1]], dtype=torch.long),
        ))
        self.assertTrue(torch.equal(
            result.source_indices.sort().values,
            torch.tensor([0, 1], dtype=torch.long),
        ))

    def test_deterministic_highest_confusion_pair_selected(self):
        cw = torch.zeros(3, 1, 2, dtype=torch.complex64)
        cw[0, 0, 0] = 1.0 + 0j
        cw[1, 0, 0] = 1.0 + 0j
        cw[2, 0, 1] = 1.0 + 0j
        pairs = torch.tensor([[0, 2], [0, 1], [1, 2]], dtype=torch.long)
        ss = SparseShiftSet(
            shifts=torch.tensor([[0, 0]], dtype=torch.long),
        )
        result = mine_cross_token_hard_negatives(
            cw, pairs, ss, torch.ones(1, 1, 2), num_hard_pairs=1,
        )
        self.assertTrue(torch.equal(
            result.token_pairs,
            torch.tensor([[0, 1]], dtype=torch.long),
        ))
        self.assertTrue(torch.equal(
            result.source_indices,
            torch.tensor([1], dtype=torch.long),
        ))
        self.assertAlmostEqual(result.peak_scores.item(), 1.0, places=6)

    def test_zero_shift_legal(self):
        ss = SparseShiftSet(
            shifts=torch.tensor([[0, 0], [1, 0]], dtype=torch.long),
        )
        result = mine_cross_token_hard_negatives(
            self.cw, self.pairs, ss, self.mask, num_hard_pairs=3,
        )
        self.assertIsInstance(result, HardNegativeTokenPairs)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cpu_pairs_cuda_cw(self):
        cw = _complex_randn(6, 8, 10).to("cuda")
        result = mine_cross_token_hard_negatives(
            cw, self.pairs, self.ss, self.mask, num_hard_pairs=3,
        )
        self.assertTrue(result.token_pairs.is_cuda)

    # -- reject paths ----------------------------------------------------------

    def test_rejects_num_hard_pairs_bool(self):
        with self.assertRaises(TypeError):
            mine_cross_token_hard_negatives(
                self.cw, self.pairs, self.ss, self.mask,
                num_hard_pairs=True,
            )

    def test_rejects_num_hard_pairs_non_int(self):
        with self.assertRaises(TypeError):
            mine_cross_token_hard_negatives(
                self.cw, self.pairs, self.ss, self.mask,
                num_hard_pairs=3.0,
            )

    def test_rejects_num_hard_pairs_zero(self):
        with self.assertRaises(ValueError):
            mine_cross_token_hard_negatives(
                self.cw, self.pairs, self.ss, self.mask,
                num_hard_pairs=0,
            )

    def test_rejects_num_hard_pairs_exceeds_P_pool(self):
        with self.assertRaises(ValueError):
            mine_cross_token_hard_negatives(
                self.cw, self.pairs, self.ss, self.mask,
                num_hard_pairs=20,
            )

    def test_rejects_candidate_pairs_non_tensor(self):
        with self.assertRaises(TypeError):
            mine_cross_token_hard_negatives(
                self.cw, [[0, 1]], self.ss, self.mask, num_hard_pairs=1,
            )

    def test_rejects_candidate_pairs_non_long(self):
        with self.assertRaises(TypeError):
            mine_cross_token_hard_negatives(
                self.cw,
                torch.tensor([[0, 1]], dtype=torch.int32),
                self.ss, self.mask, num_hard_pairs=1,
            )

    def test_rejects_candidate_pairs_wrong_shape(self):
        with self.assertRaises(ValueError):
            mine_cross_token_hard_negatives(
                self.cw,
                torch.tensor([0, 1], dtype=torch.long),
                self.ss, self.mask, num_hard_pairs=1,
            )

    def test_rejects_candidate_pairs_empty(self):
        with self.assertRaises(ValueError):
            mine_cross_token_hard_negatives(
                self.cw,
                torch.empty(0, 2, dtype=torch.long),
                self.ss, self.mask, num_hard_pairs=1,
            )

    def test_rejects_non_sparseshiftset(self):
        with self.assertRaises(TypeError):
            mine_cross_token_hard_negatives(
                self.cw, self.pairs, "bad", self.mask, num_hard_pairs=2,
            )

    # -- fail-fast -------------------------------------------------------------

    def test_fail_fast_invalid_K_before_scores(self):
        from unittest.mock import patch
        target = ("transmitter.hard_negative."
                  "cross_token_orbit_confusion_scores")
        with patch(target) as mock_scores:
            with self.assertRaises(ValueError):
                mine_cross_token_hard_negatives(
                    self.cw, self.pairs, self.ss, self.mask,
                    num_hard_pairs=0,
                )
            mock_scores.assert_not_called()

    def test_fail_fast_invalid_pairs_before_scores(self):
        from unittest.mock import patch
        target = ("transmitter.hard_negative."
                  "cross_token_orbit_confusion_scores")
        with patch(target) as mock_scores:
            with self.assertRaises(TypeError):
                mine_cross_token_hard_negatives(
                    self.cw, [[0, 1]], self.ss, self.mask,
                    num_hard_pairs=2,
                )
            mock_scores.assert_not_called()

    def test_topk_largest_true_sorted_true(self):
        from unittest.mock import patch
        with patch("torch.topk", wraps=torch.topk) as mock_topk:
            mine_cross_token_hard_negatives(
                self.cw, self.pairs, self.ss, self.mask,
                num_hard_pairs=3,
            )
            self.assertTrue(mock_topk.call_args[1]["largest"])
            self.assertTrue(mock_topk.call_args[1]["sorted"])

    def test_scores_run_under_no_grad(self):
        from unittest.mock import patch
        observed_grad_modes = []

        def fake_scores(codeword_book, token_pairs, shift_set, evidence_mask):
            observed_grad_modes.append(torch.is_grad_enabled())
            return torch.ones(
                token_pairs.shape[0],
                shift_set.shifts.shape[0],
                device=codeword_book.device,
            )

        target = ("transmitter.hard_negative."
                  "cross_token_orbit_confusion_scores")
        with patch(target, side_effect=fake_scores):
            mine_cross_token_hard_negatives(
                self.cw, self.pairs, self.ss, self.mask,
                num_hard_pairs=3,
            )
        self.assertEqual(observed_grad_modes, [False])

    def test_zero_masked_energy_propagates_value_error(self):
        cw = torch.zeros(2, 2, 2, dtype=torch.complex64)
        cw[:, 0, 0] = 1.0 + 0j
        pairs = torch.tensor([[0, 1]], dtype=torch.long)
        ss = SparseShiftSet(
            shifts=torch.tensor([[0, 1]], dtype=torch.long),
        )
        mask = torch.zeros(1, 2, 2)
        mask[0, 0, 0] = 1.0
        with self.assertRaises(ValueError):
            mine_cross_token_hard_negatives(
                cw, pairs, ss, mask, num_hard_pairs=1,
            )


# ============================================================================
# C. Autograd contract tests
# ============================================================================

class AutogradContractTests(unittest.TestCase):

    def setUp(self):
        self.cw = _complex_randn(6, 8, 10)
        self.pairs = _candidate_pool(V=6, P=10)
        self.ss = SparseShiftSet(
            shifts=torch.tensor([[1, 0], [0, 1]], dtype=torch.long),
        )
        self.mask = torch.ones(1, 8, 10)

    def test_miner_output_all_detached_no_grad(self):
        result = mine_cross_token_hard_negatives(
            self.cw, self.pairs, self.ss, self.mask, num_hard_pairs=3,
        )
        self.assertFalse(result.token_pairs.requires_grad)
        self.assertFalse(result.orbit_scores.requires_grad)
        self.assertFalse(result.peak_scores.requires_grad)

    def test_recompute_loss_with_selected_pairs_produces_grad(self):
        cw = self.cw.clone().requires_grad_(True)
        miner = mine_cross_token_hard_negatives(
            cw, self.pairs, self.ss, self.mask, num_hard_pairs=3,
        )
        # Recompute loss with selected pairs to establish autograd.
        loss = cross_token_orbit_confusion_loss(
            cw, miner.token_pairs, self.ss, self.mask, mode="weighted_isl",
        )
        loss.backward()
        self.assertIsNotNone(cw.grad)
        self.assertTrue(torch.isfinite(cw.grad).all())

    def test_miner_peak_scores_not_trainable(self):
        """Miner peak_scores must not be used as training loss."""
        miner = mine_cross_token_hard_negatives(
            self.cw, self.pairs, self.ss, self.mask, num_hard_pairs=3,
        )
        self.assertFalse(miner.peak_scores.requires_grad)


# ============================================================================
# D. Step 16A composition tests
# ============================================================================

class CompositionTests(unittest.TestCase):

    def setUp(self):
        self.cw = _complex_randn(6, 8, 10)
        self.mask = torch.ones(1, 8, 10)

    def test_sample_uniform_pairs_direct_to_miner(self):
        pairs = sample_uniform_cross_token_pairs(6, 15)
        ss = SparseShiftSet(
            shifts=torch.tensor([[0, 0], [1, 0]], dtype=torch.long),
        )
        result = mine_cross_token_hard_negatives(
            self.cw, pairs, ss, self.mask, num_hard_pairs=5,
        )
        self.assertEqual(result.token_pairs.shape, (5, 2))

    def test_sample_weighted_shifts_direct_to_miner(self):
        src_ss = SparseShiftSet(
            shifts=torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 2]],
                                dtype=torch.long),
            weights=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        )
        sampled = sample_weighted_sparse_shift_set(src_ss, 4)
        self.assertIsNone(sampled.shift_set.weights)
        pairs = sample_uniform_cross_token_pairs(6, 10)
        result = mine_cross_token_hard_negatives(
            self.cw, pairs, sampled.shift_set, self.mask, num_hard_pairs=5,
        )
        self.assertIsInstance(result, HardNegativeTokenPairs)

    def test_duplicates_preserved_through_chain(self):
        pairs = sample_uniform_cross_token_pairs(3, 30)
        ss = SparseShiftSet(
            shifts=torch.tensor([[0, 0]], dtype=torch.long),
        )
        result = mine_cross_token_hard_negatives(
            self.cw, pairs, ss, self.mask, num_hard_pairs=10,
        )
        self.assertEqual(result.token_pairs.shape, (10, 2))


# ============================================================================
# E. Quality tests
# ============================================================================

class QualityTests(unittest.TestCase):

    def test_hard_negative_py_ascii_only(self):
        path = TRANSMITTER_ROOT / "transmitter" / "hard_negative.py"
        content = path.read_text(encoding="utf-8")
        for i, ch in enumerate(content):
            self.assertTrue(ord(ch) < 128,
                            f"Non-ASCII U+{ord(ch):04X} at offset {i}")

    def test_dataclass_docstring_exists(self):
        self.assertIsNotNone(HardNegativeTokenPairs.__doc__)
        self.assertTrue(len(HardNegativeTokenPairs.__doc__.strip()) > 0)

    def test_miner_docstring_exists(self):
        self.assertIsNotNone(mine_cross_token_hard_negatives.__doc__)
        self.assertTrue(
            len(mine_cross_token_hard_negatives.__doc__.strip()) > 0,
        )

    def test_dataclass_docstring_mentions_shapes(self):
        doc = HardNegativeTokenPairs.__doc__
        self.assertIn("[K, 2]", doc)
        self.assertIn("[K, S]", doc)

    def test_miner_docstring_mentions_shapes(self):
        doc = mine_cross_token_hard_negatives.__doc__
        self.assertIn("[P_pool, 2]", doc)
        self.assertIn("[K, 2]", doc)

    def test_miner_docstring_no_grad_selection(self):
        doc = mine_cross_token_hard_negatives.__doc__
        self.assertTrue("no_grad" in doc or "detached" in doc.lower())

    def test_miner_docstring_recompute_for_autograd(self):
        doc = mine_cross_token_hard_negatives.__doc__
        self.assertTrue(
            "recompute" in doc.lower()
            or "cross_token_orbit_confusion_loss" in doc,
        )

    def test_exports_importable(self):
        from transmitter.ablations import (
            HardNegativeTokenPairs as _HNTP,
            mine_cross_token_hard_negatives as _m,
        )
        self.assertTrue(callable(_m))
        self.assertIsInstance(_HNTP, type)


# ============================================================================
# F. CUDA tests
# ============================================================================

@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class CUDATests(unittest.TestCase):

    def test_cuda_mining(self):
        cw = _complex_randn(6, 8, 10).to("cuda")
        pairs = _candidate_pool(V=6, P=10)
        ss = SparseShiftSet(
            shifts=torch.tensor([[0, 0], [1, 0]], dtype=torch.long),
        )
        mask = torch.ones(1, 8, 10)
        result = mine_cross_token_hard_negatives(
            cw, pairs, ss, mask, num_hard_pairs=3,
        )
        self.assertTrue(result.token_pairs.is_cuda)
        self.assertTrue(result.source_indices.is_cuda)
        self.assertTrue(result.peak_scores.is_cuda)
        self.assertTrue(result.orbit_scores.is_cuda)
        self.assertFalse(result.token_pairs.requires_grad)
        self.assertFalse(result.orbit_scores.requires_grad)


if __name__ == "__main__":
    unittest.main()
