"""Tests for Monte Carlo pair/shift sampling primitives (Step 16A)."""

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
    SampledSparseShiftSet,
    sample_weighted_sparse_shift_set,
)


# ============================================================================
# A. SampledSparseShiftSet tests
# ============================================================================

class SampledSparseShiftSetTests(unittest.TestCase):

    def test_valid_construction(self):
        ss = SparseShiftSet(
            shifts=torch.tensor([[1, 0], [0, 1]], dtype=torch.long),
            weights=None,
        )
        si = torch.tensor([0, 1], dtype=torch.long)
        obj = SampledSparseShiftSet(shift_set=ss, source_indices=si)
        self.assertIsInstance(obj, SampledSparseShiftSet)

    def test_shifts_shape(self):
        ss = SparseShiftSet(
            shifts=torch.tensor([[1, 0], [0, 1], [2, 3]], dtype=torch.long),
            weights=None,
        )
        si = torch.tensor([0, 1, 0], dtype=torch.long)
        obj = SampledSparseShiftSet(shift_set=ss, source_indices=si)
        self.assertEqual(obj.shift_set.shifts.shape, (3, 2))

    def test_source_indices_shape(self):
        ss = SparseShiftSet(
            shifts=torch.tensor([[1, 0], [0, 1]], dtype=torch.long),
            weights=None,
        )
        si = torch.tensor([0, 1], dtype=torch.long)
        obj = SampledSparseShiftSet(shift_set=ss, source_indices=si)
        self.assertEqual(obj.source_indices.shape, (2,))

    def test_dtype_device_correct(self):
        ss = SparseShiftSet(
            shifts=torch.tensor([[1, 0]], dtype=torch.long),
            weights=None,
        )
        si = torch.tensor([0], dtype=torch.long)
        obj = SampledSparseShiftSet(shift_set=ss, source_indices=si)
        self.assertEqual(obj.source_indices.dtype, torch.long)
        self.assertEqual(obj.source_indices.device, ss.shifts.device)

    def test_duplicate_shifts_legal(self):
        ss = SparseShiftSet(
            shifts=torch.tensor([[1, 0], [1, 0]], dtype=torch.long),
            weights=None,
        )
        si = torch.tensor([0, 0], dtype=torch.long)
        obj = SampledSparseShiftSet(shift_set=ss, source_indices=si)
        self.assertEqual(obj.shift_set.shifts.shape, (2, 2))

    def test_duplicate_source_indices_legal(self):
        ss = SparseShiftSet(
            shifts=torch.tensor([[1, 0], [0, 1]], dtype=torch.long),
            weights=None,
        )
        si = torch.tensor([0, 0], dtype=torch.long)
        obj = SampledSparseShiftSet(shift_set=ss, source_indices=si)
        self.assertEqual(obj.source_indices.tolist(), [0, 0])

    def test_source_indices_nonnegative(self):
        ss = SparseShiftSet(
            shifts=torch.tensor([[1, 0]], dtype=torch.long),
            weights=None,
        )
        si = torch.tensor([0], dtype=torch.long)
        obj = SampledSparseShiftSet(shift_set=ss, source_indices=si)
        self.assertTrue((obj.source_indices >= 0).all())

    # -- reject paths ----------------------------------------------------------

    def test_rejects_non_sparseshiftset(self):
        with self.assertRaises(TypeError):
            SampledSparseShiftSet(
                shift_set="bad",
                source_indices=torch.tensor([0], dtype=torch.long),
            )

    def test_rejects_non_none_weights(self):
        ss = SparseShiftSet(
            shifts=torch.tensor([[1, 0]], dtype=torch.long),
            weights=torch.tensor([1.0]),
        )
        with self.assertRaises(ValueError):
            SampledSparseShiftSet(
                shift_set=ss,
                source_indices=torch.tensor([0], dtype=torch.long),
            )

    def test_rejects_source_indices_non_tensor(self):
        with self.assertRaises(TypeError):
            SampledSparseShiftSet(
                shift_set=SparseShiftSet(
                    shifts=torch.tensor([[1, 0]], dtype=torch.long),
                    weights=None,
                ),
                source_indices=[0],
            )

    def test_rejects_source_indices_non_long(self):
        with self.assertRaises(TypeError):
            SampledSparseShiftSet(
                shift_set=SparseShiftSet(
                    shifts=torch.tensor([[1, 0]], dtype=torch.long),
                    weights=None,
                ),
                source_indices=torch.tensor([0], dtype=torch.int32),
            )

    def test_rejects_source_indices_wrong_ndim(self):
        with self.assertRaises(ValueError):
            SampledSparseShiftSet(
                shift_set=SparseShiftSet(
                    shifts=torch.tensor([[1, 0]], dtype=torch.long),
                    weights=None,
                ),
                source_indices=torch.tensor([[0]], dtype=torch.long),
            )

    def test_rejects_source_indices_wrong_length(self):
        with self.assertRaises(ValueError):
            SampledSparseShiftSet(
                shift_set=SparseShiftSet(
                    shifts=torch.tensor([[1, 0]], dtype=torch.long),
                    weights=None,
                ),
                source_indices=torch.tensor([0, 1], dtype=torch.long),
            )

    def test_rejects_source_indices_device_mismatch(self):
        ss = SparseShiftSet(
            shifts=torch.tensor([[1, 0]], dtype=torch.long),
            weights=None,
        )
        try:
            si_cuda = torch.tensor([0], dtype=torch.long).to("cuda")
        except (AssertionError, RuntimeError):
            self.skipTest("CUDA not available for device mismatch test")
        with self.assertRaises(ValueError):
            SampledSparseShiftSet(shift_set=ss, source_indices=si_cuda)

    def test_rejects_negative_source_indices(self):
        with self.assertRaises(ValueError):
            SampledSparseShiftSet(
                shift_set=SparseShiftSet(
                    shifts=torch.tensor([[1, 0]], dtype=torch.long),
                    weights=None,
                ),
                source_indices=torch.tensor([-1], dtype=torch.long),
            )

    def test_docstring_mentions_shapes(self):
        doc = SampledSparseShiftSet.__doc__
        self.assertIsNotNone(doc)
        self.assertIn("[S_sample, 2]", doc)
        self.assertIn("[S_sample]", doc)


# ============================================================================
# B. Pair sampling tests
# ============================================================================

class PairSamplingTests(unittest.TestCase):

    def test_output_long_P2(self):
        pairs = sample_uniform_cross_token_pairs(6, 5)
        self.assertEqual(pairs.dtype, torch.long)
        self.assertEqual(pairs.shape, (5, 2))

    def test_all_indices_in_range(self):
        V = 10
        pairs = sample_uniform_cross_token_pairs(V, 50)
        self.assertTrue((pairs >= 0).all())
        self.assertTrue((pairs < V).all())

    def test_all_u_not_equal_v(self):
        pairs = sample_uniform_cross_token_pairs(6, 100)
        self.assertFalse((pairs[:, 0] == pairs[:, 1]).any())

    def test_ordered_pairs_semantics(self):
        """Ordered: (u,v) and (v,u) are different samples."""
        from unittest.mock import patch
        # For V=4, flat 0 -> (0,1) and flat 3 -> (1,0).
        flat = torch.tensor([0, 3], dtype=torch.long)
        with patch("torch.randint", return_value=flat):
            pairs = sample_uniform_cross_token_pairs(4, 2)
        self.assertTrue(torch.equal(
            pairs,
            torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        ))

    def test_deterministic_with_same_seed(self):
        gen1 = torch.Generator(device="cpu").manual_seed(42)
        gen2 = torch.Generator(device="cpu").manual_seed(42)
        p1 = sample_uniform_cross_token_pairs(10, 7, generator=gen1)
        p2 = sample_uniform_cross_token_pairs(10, 7, generator=gen2)
        self.assertTrue(torch.equal(p1, p2))

    def test_duplicates_legal(self):
        from unittest.mock import patch
        flat = torch.tensor([0, 0], dtype=torch.long)
        with patch("torch.randint", return_value=flat):
            pairs = sample_uniform_cross_token_pairs(3, 2)
        self.assertTrue(torch.equal(pairs[0], pairs[1]))

    def test_mapping_correctness(self):
        """Manually verify flat -> (u,v) mapping for a small V."""
        V = 3
        P = 50
        gen = torch.Generator(device="cpu").manual_seed(123)
        pairs = sample_uniform_cross_token_pairs(V, P, generator=gen)
        gen2 = torch.Generator(device="cpu").manual_seed(123)
        total = V * (V - 1)
        flat = torch.randint(0, total, (P,), generator=gen2, dtype=torch.long)
        u_exp = flat // (V - 1)
        offset = flat % (V - 1)
        v_exp = offset + (offset >= u_exp).long()
        expected = torch.stack([u_exp, v_exp], dim=-1)
        self.assertTrue(torch.equal(pairs, expected))

    def test_large_vocab_size_small_P_no_O_V2(self):
        """V=1_000_000 with P=5 should run instantly, proving no V^2."""
        pairs = sample_uniform_cross_token_pairs(1_000_000, 5)
        self.assertEqual(pairs.shape, (5, 2))

    def test_no_randperm_called(self):
        from unittest.mock import patch
        with patch("torch.randperm") as mock_rp:
            sample_uniform_cross_token_pairs(100, 10)
            mock_rp.assert_not_called()

    # -- reject paths ----------------------------------------------------------

    def test_rejects_vocab_size_bool(self):
        with self.assertRaises(TypeError):
            sample_uniform_cross_token_pairs(True, 5)

    def test_rejects_vocab_size_lt_2(self):
        with self.assertRaises(ValueError):
            sample_uniform_cross_token_pairs(1, 5)

    def test_rejects_vocab_size_non_int(self):
        with self.assertRaises(TypeError):
            sample_uniform_cross_token_pairs(6.0, 5)

    def test_rejects_num_pairs_bool(self):
        with self.assertRaises(TypeError):
            sample_uniform_cross_token_pairs(6, True)

    def test_rejects_num_pairs_zero(self):
        with self.assertRaises(ValueError):
            sample_uniform_cross_token_pairs(6, 0)

    def test_rejects_num_pairs_non_int(self):
        with self.assertRaises(TypeError):
            sample_uniform_cross_token_pairs(6, 5.0)

    def test_rejects_invalid_generator_type(self):
        with self.assertRaises(TypeError):
            sample_uniform_cross_token_pairs(6, 5, generator="bad")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_rejects_gpu_generator(self):
        gen = torch.Generator(device="cuda")
        with self.assertRaises(ValueError):
            sample_uniform_cross_token_pairs(6, 5, generator=gen)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_output(self):
        pairs = sample_uniform_cross_token_pairs(6, 5, device="cuda")
        self.assertTrue(pairs.is_cuda)


# ============================================================================
# C. Weighted shift sampling tests
# ============================================================================

class WeightedShiftSamplingTests(unittest.TestCase):

    def setUp(self):
        self.src_ss = SparseShiftSet(
            shifts=torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 2]],
                                dtype=torch.long),
            weights=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        )

    def test_output_is_sampled_sparse_shift_set(self):
        result = sample_weighted_sparse_shift_set(self.src_ss, 5)
        self.assertIsInstance(result, SampledSparseShiftSet)

    def test_output_shifts_shape(self):
        result = sample_weighted_sparse_shift_set(self.src_ss, 7)
        self.assertEqual(result.shift_set.shifts.shape, (7, 2))

    def test_output_source_indices_shape(self):
        result = sample_weighted_sparse_shift_set(self.src_ss, 7)
        self.assertEqual(result.source_indices.shape, (7,))

    def test_output_weights_is_none(self):
        result = sample_weighted_sparse_shift_set(self.src_ss, 5)
        self.assertIsNone(result.shift_set.weights)

    def test_deterministic_with_same_seed(self):
        gen1 = torch.Generator(device="cpu").manual_seed(42)
        gen2 = torch.Generator(device="cpu").manual_seed(42)
        r1 = sample_weighted_sparse_shift_set(self.src_ss, 8, generator=gen1)
        r2 = sample_weighted_sparse_shift_set(self.src_ss, 8, generator=gen2)
        self.assertTrue(torch.equal(r1.shift_set.shifts, r2.shift_set.shifts))
        self.assertTrue(torch.equal(r1.source_indices, r2.source_indices))

    def test_one_hot_source_weights_picks_only_that_shift(self):
        ss = SparseShiftSet(
            shifts=torch.tensor([[1, 0], [2, 0], [3, 0]], dtype=torch.long),
            weights=torch.tensor([0.0, 100.0, 0.0]),
        )
        result = sample_weighted_sparse_shift_set(ss, 10)
        # All sampled shifts should be [2, 0].
        self.assertTrue(
            (result.shift_set.shifts == torch.tensor([2, 0])).all()
        )

    def test_zero_weight_shift_never_sampled(self):
        ss = SparseShiftSet(
            shifts=torch.tensor([[1, 0], [2, 0]], dtype=torch.long),
            weights=torch.tensor([0.0, 1.0]),
        )
        result = sample_weighted_sparse_shift_set(ss, 20)
        self.assertFalse(
            (result.shift_set.shifts == torch.tensor([1, 0])).all(1).any()
        )

    def test_duplicates_preserved(self):
        result = sample_weighted_sparse_shift_set(self.src_ss, 20)
        si = result.source_indices.tolist()
        self.assertGreater(len(si), len(set(si)),
                           "20 samples from 4 shifts must have duplicates")

    def test_zero_shift_preserved_if_sampled(self):
        ss = SparseShiftSet(
            shifts=torch.tensor([[0, 0], [1, 0]], dtype=torch.long),
            weights=torch.tensor([100.0, 0.0]),
        )
        result = sample_weighted_sparse_shift_set(ss, 5)
        self.assertTrue(
            (result.shift_set.shifts == torch.tensor([0, 0])).all()
        )

    def test_output_device_preserved(self):
        result = sample_weighted_sparse_shift_set(self.src_ss, 3)
        self.assertEqual(result.shift_set.shifts.device,
                         self.src_ss.shifts.device)
        self.assertEqual(result.source_indices.device,
                         self.src_ss.shifts.device)

    def test_source_input_not_modified(self):
        orig_shifts = self.src_ss.shifts.clone()
        orig_weights = self.src_ss.weights.clone()
        sample_weighted_sparse_shift_set(self.src_ss, 5)
        self.assertTrue(torch.equal(self.src_ss.shifts, orig_shifts))
        self.assertTrue(torch.equal(self.src_ss.weights, orig_weights))

    # -- reject: invalid source type -------------------------------------------

    def test_rejects_non_sparseshiftset(self):
        with self.assertRaises(TypeError):
            sample_weighted_sparse_shift_set("bad", 5)

    # -- reject: corrupted shifts ----------------------------------------------

    def test_rejects_shifts_non_tensor(self):
        ss = SparseShiftSet(
            shifts=torch.tensor([[1, 0]], dtype=torch.long),
            weights=None,
        )
        object.__setattr__(ss, "shifts", [[1, 0]])
        with self.assertRaises(TypeError):
            sample_weighted_sparse_shift_set(ss, 3)

    def test_rejects_shifts_wrong_dtype(self):
        ss = SparseShiftSet(
            shifts=torch.tensor([[1, 0]], dtype=torch.long),
            weights=None,
        )
        object.__setattr__(ss, "shifts",
                           torch.tensor([[1, 0]], dtype=torch.int32))
        with self.assertRaises(TypeError):
            sample_weighted_sparse_shift_set(ss, 3)

    def test_rejects_shifts_wrong_ndim(self):
        ss = SparseShiftSet(
            shifts=torch.tensor([[1, 0]], dtype=torch.long),
            weights=None,
        )
        object.__setattr__(ss, "shifts",
                           torch.tensor([1, 0], dtype=torch.long))
        with self.assertRaises(ValueError):
            sample_weighted_sparse_shift_set(ss, 3)

    def test_rejects_shifts_last_dim_not_2(self):
        ss = SparseShiftSet(
            shifts=torch.tensor([[1, 0]], dtype=torch.long),
            weights=None,
        )
        object.__setattr__(ss, "shifts",
                           torch.tensor([[1, 0, 2]], dtype=torch.long))
        with self.assertRaises(ValueError):
            sample_weighted_sparse_shift_set(ss, 3)

    def test_rejects_shifts_S_zero(self):
        ss = SparseShiftSet(
            shifts=torch.tensor([[1, 0]], dtype=torch.long),
            weights=None,
        )
        object.__setattr__(ss, "shifts",
                           torch.empty(0, 2, dtype=torch.long))
        with self.assertRaises(ValueError):
            sample_weighted_sparse_shift_set(ss, 3)

    # -- reject: corrupted weights ---------------------------------------------

    def test_rejects_weights_non_tensor(self):
        ss = SparseShiftSet(
            shifts=torch.tensor([[1, 0]], dtype=torch.long),
            weights=torch.tensor([1.0]),
        )
        object.__setattr__(ss, "weights", [1.0])
        with self.assertRaises(TypeError):
            sample_weighted_sparse_shift_set(ss, 3)

    def test_rejects_weights_non_floating(self):
        ss = SparseShiftSet(
            shifts=torch.tensor([[1, 0]], dtype=torch.long),
            weights=torch.tensor([1.0]),
        )
        object.__setattr__(ss, "weights", torch.tensor([1], dtype=torch.long))
        with self.assertRaises(TypeError):
            sample_weighted_sparse_shift_set(ss, 3)

    def test_rejects_weights_wrong_shape(self):
        ss = SparseShiftSet(
            shifts=torch.tensor([[1, 0]], dtype=torch.long),
            weights=torch.tensor([1.0]),
        )
        object.__setattr__(ss, "weights", torch.tensor([1.0, 2.0]))
        with self.assertRaises(ValueError):
            sample_weighted_sparse_shift_set(ss, 3)

    def test_rejects_weights_nan(self):
        ss = SparseShiftSet(
            shifts=torch.tensor([[1, 0]], dtype=torch.long),
            weights=torch.tensor([1.0]),
        )
        object.__setattr__(ss, "weights",
                           torch.tensor([float("nan")]))
        with self.assertRaises(ValueError):
            sample_weighted_sparse_shift_set(ss, 3)

    def test_rejects_weights_negative(self):
        ss = SparseShiftSet(
            shifts=torch.tensor([[1, 0]], dtype=torch.long),
            weights=torch.tensor([1.0]),
        )
        object.__setattr__(ss, "weights", torch.tensor([-1.0]))
        with self.assertRaises(ValueError):
            sample_weighted_sparse_shift_set(ss, 3)

    def test_rejects_weights_zero_sum(self):
        ss = SparseShiftSet(
            shifts=torch.tensor([[1, 0]], dtype=torch.long),
            weights=torch.tensor([1.0]),
        )
        object.__setattr__(ss, "weights", torch.tensor([0.0]))
        with self.assertRaises(ValueError):
            sample_weighted_sparse_shift_set(ss, 3)

    def test_rejects_weights_requires_grad_true(self):
        w = torch.tensor([1.0, 1.0], requires_grad=True)
        ss = SparseShiftSet(
            shifts=torch.tensor([[1, 0], [0, 1]], dtype=torch.long),
            weights=w,
        )
        with self.assertRaises(ValueError):
            sample_weighted_sparse_shift_set(ss, 3)

    # -- reject: num_shift_samples ---------------------------------------------

    def test_rejects_num_shift_samples_bool(self):
        with self.assertRaises(TypeError):
            sample_weighted_sparse_shift_set(self.src_ss, True)

    def test_rejects_num_shift_samples_zero(self):
        with self.assertRaises(ValueError):
            sample_weighted_sparse_shift_set(self.src_ss, 0)

    def test_rejects_num_shift_samples_non_int(self):
        with self.assertRaises(TypeError):
            sample_weighted_sparse_shift_set(self.src_ss, 5.0)

    # -- reject: invalid generator ---------------------------------------------

    def test_rejects_invalid_generator_type(self):
        with self.assertRaises(TypeError):
            sample_weighted_sparse_shift_set(
                self.src_ss, 3, generator="bad",
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_rejects_gpu_generator(self):
        gen = torch.Generator(device="cuda")
        with self.assertRaises(ValueError):
            sample_weighted_sparse_shift_set(
                self.src_ss, 3, generator=gen,
            )

    # -- CUDA ----------------------------------------------------------------

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_source_shifts(self):
        src = SparseShiftSet(
            shifts=torch.tensor([[0, 0], [1, 0]], dtype=torch.long,
                                device="cuda"),
            weights=torch.tensor([1.0, 1.0]),
        )
        result = sample_weighted_sparse_shift_set(src, 3)
        self.assertTrue(result.shift_set.shifts.is_cuda)
        self.assertTrue(result.source_indices.is_cuda)
        self.assertIsNone(result.shift_set.weights)


# ============================================================================
# D. Monte Carlo weighting semantics
# ============================================================================

class MCWeightingSemanticsTests(unittest.TestCase):

    def test_multinomial_receives_normalized_q_once(self):
        from unittest.mock import patch
        ss = SparseShiftSet(
            shifts=torch.tensor([[0, 0], [1, 0]], dtype=torch.long),
            weights=torch.tensor([1.0, 3.0]),
        )
        with patch(
            "torch.multinomial",
            return_value=torch.tensor([1, 0], dtype=torch.long),
        ) as mock_multinomial:
            result = sample_weighted_sparse_shift_set(ss, 2)
        q_cpu = mock_multinomial.call_args.args[0]
        self.assertTrue(torch.equal(
            q_cpu,
            torch.tensor([0.25, 0.75], dtype=torch.float64),
        ))
        self.assertIsNone(result.shift_set.weights)

    def test_sampled_weights_is_none_after_weighted_sampling(self):
        ss = SparseShiftSet(
            shifts=torch.tensor([[0, 0], [1, 0]], dtype=torch.long),
            weights=torch.tensor([1.0, 2.0]),
        )
        result = sample_weighted_sparse_shift_set(ss, 5)
        self.assertIsNone(result.shift_set.weights)

    def test_one_hot_all_sampled_equal_source_k(self):
        ss = SparseShiftSet(
            shifts=torch.tensor([[1, 0], [2, 0], [3, 0]], dtype=torch.long),
            weights=torch.tensor([0.0, 100.0, 0.0]),
        )
        result = sample_weighted_sparse_shift_set(ss, 5)
        self.assertTrue(
            (result.shift_set.shifts == torch.tensor([2, 0])).all()
        )
        self.assertTrue(
            (result.source_indices == 1).all()
        )
        self.assertIsNone(result.shift_set.weights)


# ============================================================================
# E. Quality tests
# ============================================================================

class QualityTests(unittest.TestCase):

    def test_sampling_py_ascii_only(self):
        path = TRANSMITTER_ROOT / "transmitter" / "sampling.py"
        content = path.read_text(encoding="utf-8")
        for i, ch in enumerate(content):
            self.assertTrue(ord(ch) < 128,
                            f"Non-ASCII U+{ord(ch):04X} at offset {i}")

    def test_sampled_ss_docstring_exists(self):
        self.assertIsNotNone(SampledSparseShiftSet.__doc__)
        self.assertTrue(len(SampledSparseShiftSet.__doc__.strip()) > 0)

    def test_pair_sampling_docstring_exists(self):
        self.assertIsNotNone(sample_uniform_cross_token_pairs.__doc__)
        self.assertTrue(
            len(sample_uniform_cross_token_pairs.__doc__.strip()) > 0,
        )

    def test_shift_sampling_docstring_exists(self):
        self.assertIsNotNone(sample_weighted_sparse_shift_set.__doc__)
        self.assertTrue(
            len(sample_weighted_sparse_shift_set.__doc__.strip()) > 0,
        )

    def test_pair_docstring_mentions_P2(self):
        doc = sample_uniform_cross_token_pairs.__doc__
        self.assertIn("[P, 2]", doc)

    def test_shift_docstring_mentions_S_sample_2(self):
        doc = sample_weighted_sparse_shift_set.__doc__
        self.assertIn("[S_sample, 2]", doc)

    def test_shift_docstring_weights_none_double_weighting(self):
        doc = sample_weighted_sparse_shift_set.__doc__
        self.assertTrue(
            "weights is None" in doc
            or "weights=None" in doc
            or "weights is strictly None" in doc
            or "double" in doc.lower(),
        )

    def test_exports_importable(self):
        from transmitter import sample_uniform_cross_token_pairs as _sup
        from transmitter.ablations import (
            SampledSparseShiftSet as _SSS,
            sample_weighted_sparse_shift_set as _sws,
        )
        self.assertTrue(callable(_sup))
        self.assertTrue(callable(_sws))
        self.assertIsInstance(_SSS, type)


if __name__ == "__main__":
    unittest.main()
