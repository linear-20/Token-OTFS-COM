"""Tests for auditable sparse multipath scenario bank (Step 17B)."""

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
TRANSMITTER_ROOT = ROOT / "Learnable_Mapping_Tokens-to-DD-Signals"
sys.path.insert(0, str(TRANSMITTER_ROOT))

from transmitter import (
    SparseMultipathDDChannel,
    SparseMultipathScenarioBank,
    SparseShiftSet,
    apply_sparse_multipath_dd_operator,
    sample_normalized_sparse_multipath_scenario_bank,
)


def _valid_bank(R=3, K=2):
    shifts = torch.zeros(R, K, 2, dtype=torch.long)
    gains = torch.ones(R, K, dtype=torch.complex64) / (K ** 0.5)
    return SparseMultipathScenarioBank(
        channel=SparseMultipathDDChannel(
            path_shifts=shifts, path_gains=gains,
            path_active_mask=torch.ones(R, K, dtype=torch.bool),
        ),
    )


def _src_shift_set():
    return SparseShiftSet(
        shifts=torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 2]],
                            dtype=torch.long),
        weights=torch.tensor([1.0, 2.0, 3.0, 4.0]),
    )


# ============================================================================
# A. Dataclass tests
# ============================================================================

class DataclassTests(unittest.TestCase):

    def test_valid_bank(self):
        bank = _valid_bank()
        self.assertIsInstance(bank, SparseMultipathScenarioBank)

    def test_shifts_shape_RK2(self):
        bank = _valid_bank(R=5, K=3)
        self.assertEqual(bank.channel.path_shifts.shape, (5, 3, 2))

    def test_gains_shape_RK(self):
        bank = _valid_bank(R=5, K=3)
        self.assertEqual(bank.channel.path_gains.shape, (5, 3))

    def test_active_mask_shape_RK(self):
        bank = _valid_bank()
        self.assertEqual(bank.channel.path_active_mask.shape, (3, 2))

    def test_scenario_weights_none_legal(self):
        bank = _valid_bank()
        self.assertIsNone(bank.scenario_weights)

    def test_explicit_scenario_weights_legal(self):
        shifts = torch.zeros(3, 2, 2, dtype=torch.long)
        gains = torch.ones(3, 2, dtype=torch.complex64) / (2 ** 0.5)
        bank = SparseMultipathScenarioBank(
            channel=SparseMultipathDDChannel(
                path_shifts=shifts, path_gains=gains,
                path_active_mask=torch.ones(3, 2, dtype=torch.bool),
            ),
            scenario_weights=torch.tensor([1.0, 2.0, 3.0]),
        )
        self.assertIsNotNone(bank.scenario_weights)

    def test_active_path_energy_is_one(self):
        shifts = torch.zeros(2, 3, 2, dtype=torch.long)
        g = torch.tensor([[1.0 + 0j, 2.0 + 0j, 0.0 + 0j]],
                         dtype=torch.complex64).expand(2, -1)
        g = g / g.abs().pow(2).sum(dim=1, keepdim=True).sqrt()
        mask = torch.tensor([[True, True, False]]).expand(2, -1)
        bank = SparseMultipathScenarioBank(
            channel=SparseMultipathDDChannel(
                path_shifts=shifts, path_gains=g, path_active_mask=mask,
            ),
        )
        e = (g.abs().pow(2) * mask.to(dtype=torch.float32)).sum(dim=1)
        self.assertTrue(torch.allclose(e, torch.ones(2), atol=1e-5))

    def test_active_mask_only_active_counted(self):
        shifts = torch.zeros(1, 2, 2, dtype=torch.long)
        g = torch.tensor([[1.0 + 0j, 10.0 + 0j]], dtype=torch.complex64)
        # Active path k=0 has gain 1.0, get normalized to 1.0.
        mask = torch.tensor([[True, False]])
        # Normalize only active paths: |1|^2 = 1 -> unit already.
        # But the bank _validate_per_scenario_energy checks active-only sum.
        # We need active_energy == 1. So with mask [True,False],
        # gains = [1, 10] -> active_energy = 1. That's valid.
        bank = SparseMultipathScenarioBank(
            channel=SparseMultipathDDChannel(
                path_shifts=shifts, path_gains=g, path_active_mask=mask,
            ),
        )
        e_active = (g.abs().pow(2) * mask.to(dtype=torch.float32)).sum(dim=1)
        self.assertTrue(torch.allclose(e_active, torch.ones(1), atol=1e-5))

    # -- reject ----------------------------------------------------------------

    def test_rejects_non_sparse_multipath_channel(self):
        with self.assertRaises(TypeError):
            SparseMultipathScenarioBank(channel="bad")

    def test_rejects_gains_requires_grad(self):
        shifts = torch.zeros(2, 2, 2, dtype=torch.long)
        gains = torch.ones(2, 2, dtype=torch.complex64) / (2 ** 0.5)
        gains.requires_grad_(True)
        with self.assertRaises(ValueError):
            SparseMultipathScenarioBank(
                channel=SparseMultipathDDChannel(
                    path_shifts=shifts, path_gains=gains,
                    path_active_mask=torch.ones(2, 2, dtype=torch.bool),
                ),
            )

    def test_rejects_active_energy_not_one(self):
        shifts = torch.zeros(1, 2, 2, dtype=torch.long)
        gains = torch.tensor([[1.0 + 0j, 2.0 + 0j]], dtype=torch.complex64)
        # Not normalized.
        with self.assertRaises(ValueError):
            SparseMultipathScenarioBank(
                channel=SparseMultipathDDChannel(
                    path_shifts=shifts, path_gains=gains,
                    path_active_mask=torch.ones(1, 2, dtype=torch.bool),
                ),
            )

    def test_rejects_all_inactive_scenario(self):
        shifts = torch.zeros(1, 2, 2, dtype=torch.long)
        gains = torch.ones(1, 2, dtype=torch.complex64) / (2 ** 0.5)
        with self.assertRaises(ValueError):
            SparseMultipathScenarioBank(
                channel=SparseMultipathDDChannel(
                    path_shifts=shifts, path_gains=gains,
                    path_active_mask=torch.zeros(1, 2, dtype=torch.bool),
                ),
            )

    def test_rejects_weights_non_tensor(self):
        with self.assertRaises(TypeError):
            SparseMultipathScenarioBank(
                channel=_valid_bank().channel, scenario_weights=[1.0],
            )

    def test_rejects_weights_non_floating(self):
        with self.assertRaises(TypeError):
            SparseMultipathScenarioBank(
                channel=_valid_bank().channel,
                scenario_weights=torch.tensor([1, 2, 3]),
            )

    def test_rejects_weights_wrong_shape(self):
        with self.assertRaises(ValueError):
            SparseMultipathScenarioBank(
                channel=_valid_bank().channel,
                scenario_weights=torch.tensor([1.0, 2.0]),
            )

    def test_rejects_weights_nan(self):
        with self.assertRaises(ValueError):
            SparseMultipathScenarioBank(
                channel=_valid_bank().channel,
                scenario_weights=torch.tensor([float("nan"), 1.0, 1.0]),
            )

    def test_rejects_weights_negative(self):
        with self.assertRaises(ValueError):
            SparseMultipathScenarioBank(
                channel=_valid_bank().channel,
                scenario_weights=torch.tensor([-1.0, 1.0, 1.0]),
            )

    def test_rejects_weights_zero_sum(self):
        with self.assertRaises(ValueError):
            SparseMultipathScenarioBank(
                channel=_valid_bank().channel,
                scenario_weights=torch.zeros(3),
            )

    def test_rejects_weights_requires_grad(self):
        with self.assertRaises(ValueError):
            SparseMultipathScenarioBank(
                channel=_valid_bank().channel,
                scenario_weights=torch.tensor([1.0, 1.0, 1.0],
                                              requires_grad=True),
            )

    def test_rejects_source_indices_non_tensor(self):
        with self.assertRaises(TypeError):
            SparseMultipathScenarioBank(
                channel=_valid_bank().channel,
                source_shift_indices=[[0, 1]],
            )

    def test_rejects_source_indices_non_long(self):
        with self.assertRaises(TypeError):
            SparseMultipathScenarioBank(
                channel=_valid_bank(R=1, K=2).channel,
                source_shift_indices=torch.tensor([[0, 1]],
                                                  dtype=torch.int32),
            )

    def test_rejects_source_indices_wrong_shape(self):
        with self.assertRaises(ValueError):
            SparseMultipathScenarioBank(
                channel=_valid_bank(R=2, K=2).channel,
                source_shift_indices=torch.tensor([[0, 1, 2]],
                                                  dtype=torch.long),
            )

    def test_rejects_source_indices_negative(self):
        with self.assertRaises(ValueError):
            SparseMultipathScenarioBank(
                channel=_valid_bank(R=1, K=2).channel,
                source_shift_indices=torch.tensor([[-1, 0]],
                                                  dtype=torch.long),
            )

    def test_rejects_name_non_str(self):
        with self.assertRaises(TypeError):
            SparseMultipathScenarioBank(channel=_valid_bank().channel, name=1)

    def test_rejects_name_empty(self):
        with self.assertRaises(ValueError):
            SparseMultipathScenarioBank(channel=_valid_bank().channel,
                                        name="   ")

    def test_docstring_contains_shapes(self):
        doc = SparseMultipathScenarioBank.__doc__
        self.assertIn("[R, K, 2]", doc)
        self.assertIn("[R, K]", doc)

    def test_docstring_fixed_scenario_semantics(self):
        doc = SparseMultipathScenarioBank.__doc__
        self.assertTrue("fixed" in doc.lower()
                        or "non-learnable" in doc.lower())


# ============================================================================
# B. normalized_weights tests
# ============================================================================

class NormalizedWeightsTests(unittest.TestCase):

    def test_default_uniform(self):
        bank = _valid_bank(R=4)
        q = bank.normalized_weights()
        self.assertTrue(torch.allclose(q, torch.full((4,), 0.25)))

    def test_explicit_weights_match_manual(self):
        bank = _valid_bank(R=3)
        object.__setattr__(bank, "scenario_weights",
                           torch.tensor([1.0, 2.0, 3.0]))
        q = bank.normalized_weights()
        expected = torch.tensor([1.0, 2.0, 3.0]) / 6.0
        self.assertTrue(torch.allclose(q, expected))

    def test_output_device_argument(self):
        bank = _valid_bank(R=3)
        q = bank.normalized_weights(device="cpu")
        self.assertEqual(q.device, torch.device("cpu"))

    def test_output_dtype_argument(self):
        bank = _valid_bank(R=3)
        q = bank.normalized_weights(dtype=torch.float64)
        self.assertEqual(q.dtype, torch.float64)

    def test_rejects_non_floating_dtype(self):
        bank = _valid_bank()
        with self.assertRaises(TypeError):
            bank.normalized_weights(dtype=torch.long)

    def test_rejects_non_dtype_object(self):
        bank = _valid_bank()
        with self.assertRaises(TypeError):
            bank.normalized_weights(dtype="float32")

    def test_does_not_modify_source(self):
        bank = _valid_bank(R=3)
        w_orig = bank.scenario_weights
        bank.normalized_weights()
        self.assertIs(bank.scenario_weights, w_orig)


# ============================================================================
# C. materialize_channel tests
# ============================================================================

class MaterializeTests(unittest.TestCase):

    def setUp(self):
        self.bank = _valid_bank(R=3, K=2)

    def test_materialize_complex64(self):
        ch = self.bank.materialize_channel(
            device="cpu", dtype=torch.complex64,
        )
        self.assertIsInstance(ch, SparseMultipathDDChannel)
        self.assertEqual(ch.path_gains.dtype, torch.complex64)

    def test_materialize_complex128(self):
        ch = self.bank.materialize_channel(
            device="cpu", dtype=torch.complex128,
        )
        self.assertEqual(ch.path_gains.dtype, torch.complex128)

    def test_materialized_shape_kept(self):
        ch = self.bank.materialize_channel(
            device="cpu", dtype=torch.complex64,
        )
        self.assertEqual(ch.path_shifts.shape, (3, 2, 2))
        self.assertEqual(ch.path_gains.shape, (3, 2))

    def test_materialized_gains_requires_grad_false(self):
        ch = self.bank.materialize_channel(
            device="cpu", dtype=torch.complex64,
        )
        self.assertFalse(ch.path_gains.requires_grad)

    def test_source_bank_not_modified(self):
        orig_gains = self.bank.channel.path_gains.clone()
        self.bank.materialize_channel(device="cpu", dtype=torch.complex128)
        self.assertTrue(torch.equal(self.bank.channel.path_gains, orig_gains))

    def test_materialized_tensors_independent(self):
        ch = self.bank.materialize_channel(
            device="cpu", dtype=torch.complex64,
        )
        ch.path_gains[0, 0] = 999.0
        self.assertNotEqual(
            self.bank.channel.path_gains[0, 0].item(),
            ch.path_gains[0, 0].item(),
        )

    def test_rejects_invalid_dtype(self):
        with self.assertRaises(TypeError):
            self.bank.materialize_channel(device="cpu", dtype=torch.float32)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_materialize_cuda_connect_operator(self):
        ch = self.bank.materialize_channel(
            device="cuda", dtype=torch.complex64,
        )
        x = torch.randn(3, 4, 6, dtype=torch.complex64).to("cuda")
        x.requires_grad_(True)
        y = apply_sparse_multipath_dd_operator(x, ch)
        self.assertEqual(y.shape, (3, 4, 6))
        loss = y.abs().pow(2).sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertFalse(ch.path_gains.requires_grad)


# ============================================================================
# D. Sampling tests
# ============================================================================

class SamplingTests(unittest.TestCase):

    def test_output_is_scenario_bank(self):
        bank = sample_normalized_sparse_multipath_scenario_bank(
            _src_shift_set(), 3, 2,
        )
        self.assertIsInstance(bank, SparseMultipathScenarioBank)

    def test_output_canonical_cpu(self):
        bank = sample_normalized_sparse_multipath_scenario_bank(
            _src_shift_set(), 3, 2,
        )
        self.assertEqual(bank.channel.path_shifts.device.type, "cpu")
        self.assertEqual(bank.channel.path_gains.device.type, "cpu")

    def test_shapes_correct(self):
        bank = sample_normalized_sparse_multipath_scenario_bank(
            _src_shift_set(), 5, 3,
        )
        self.assertEqual(bank.channel.path_shifts.shape, (5, 3, 2))
        self.assertEqual(bank.channel.path_gains.shape, (5, 3))
        self.assertEqual(bank.channel.path_active_mask.shape, (5, 3))
        self.assertEqual(bank.source_shift_indices.shape, (5, 3))

    def test_active_mask_all_true(self):
        bank = sample_normalized_sparse_multipath_scenario_bank(
            _src_shift_set(), 3, 2,
        )
        self.assertTrue(bank.channel.path_active_mask.all())

    def test_scenario_weights_is_none(self):
        bank = sample_normalized_sparse_multipath_scenario_bank(
            _src_shift_set(), 3, 2,
        )
        self.assertIsNone(bank.scenario_weights)

    def test_source_indices_map_correctly(self):
        bank = sample_normalized_sparse_multipath_scenario_bank(
            _src_shift_set(), 3, 2,
        )
        src = _src_shift_set()
        expected_shifts = src.shifts[bank.source_shift_indices]
        self.assertTrue(torch.equal(bank.channel.path_shifts, expected_shifts))

    def test_per_scenario_unit_energy(self):
        bank = sample_normalized_sparse_multipath_scenario_bank(
            _src_shift_set(), 5, 4,
        )
        e = bank.channel.path_gains.abs().pow(2).sum(dim=1)
        self.assertTrue(torch.allclose(e, torch.ones(5), atol=1e-4))

    def test_deterministic_same_seed(self):
        gen1 = torch.Generator(device="cpu").manual_seed(42)
        gen2 = torch.Generator(device="cpu").manual_seed(42)
        b1 = sample_normalized_sparse_multipath_scenario_bank(
            _src_shift_set(), 3, 2, generator=gen1,
        )
        b2 = sample_normalized_sparse_multipath_scenario_bank(
            _src_shift_set(), 3, 2, generator=gen2,
        )
        self.assertTrue(torch.equal(b1.channel.path_shifts,
                                    b2.channel.path_shifts))
        self.assertTrue(torch.equal(b1.channel.path_gains,
                                    b2.channel.path_gains))

    def test_one_hot_q_only_sampled_shift(self):
        ss = SparseShiftSet(
            shifts=torch.tensor([[7, 3], [1, 0], [0, 1]], dtype=torch.long),
            weights=torch.tensor([100.0, 0.0, 0.0]),
        )
        bank = sample_normalized_sparse_multipath_scenario_bank(ss, 3, 2)
        self.assertTrue(
            (bank.channel.path_shifts == torch.tensor([7, 3])).all()
        )

    def test_duplicates_retained(self):
        ss = SparseShiftSet(
            shifts=torch.tensor([[7, 3], [1, 0]], dtype=torch.long),
            weights=torch.tensor([1.0, 0.0]),
        )
        bank = sample_normalized_sparse_multipath_scenario_bank(ss, 2, 3)
        self.assertTrue(torch.equal(
            bank.source_shift_indices,
            torch.zeros(2, 3, dtype=torch.long),
        ))
        self.assertTrue(
            (bank.channel.path_shifts == torch.tensor([7, 3])).all()
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_source_shifts_still_return_canonical_cpu(self):
        ss = SparseShiftSet(
            shifts=torch.tensor(
                [[0, 0], [1, 0]], dtype=torch.long, device="cuda",
            ),
            weights=torch.tensor([1.0, 1.0]),
        )
        bank = sample_normalized_sparse_multipath_scenario_bank(ss, 2, 2)
        self.assertEqual(bank.channel.path_shifts.device.type, "cpu")
        self.assertEqual(bank.channel.path_gains.device.type, "cpu")
        self.assertEqual(bank.source_shift_indices.device.type, "cpu")

    def test_source_not_modified(self):
        ss = _src_shift_set()
        orig_shifts = ss.shifts.clone()
        orig_weights = ss.weights.clone() if ss.weights is not None else None
        sample_normalized_sparse_multipath_scenario_bank(ss, 3, 2)
        self.assertTrue(torch.equal(ss.shifts, orig_shifts))
        if orig_weights is not None:
            self.assertTrue(torch.equal(ss.weights, orig_weights))

    def test_complex128_output(self):
        bank = sample_normalized_sparse_multipath_scenario_bank(
            _src_shift_set(), 2, 2, complex_dtype=torch.complex128,
        )
        self.assertEqual(bank.channel.path_gains.dtype, torch.complex128)

    # -- reject ----------------------------------------------------------------

    def test_rejects_non_sparseshiftset(self):
        with self.assertRaises(TypeError):
            sample_normalized_sparse_multipath_scenario_bank("bad", 3, 2)

    def test_rejects_shifts_non_tensor(self):
        ss = _src_shift_set()
        object.__setattr__(ss, "shifts", [[0, 0]])
        with self.assertRaises(TypeError):
            sample_normalized_sparse_multipath_scenario_bank(ss, 3, 2)

    def test_rejects_shifts_wrong_dtype(self):
        ss = _src_shift_set()
        object.__setattr__(ss, "shifts",
                           torch.tensor([[0, 0]], dtype=torch.int32))
        with self.assertRaises(TypeError):
            sample_normalized_sparse_multipath_scenario_bank(ss, 3, 2)

    def test_rejects_shifts_wrong_shape(self):
        ss = _src_shift_set()
        object.__setattr__(ss, "shifts",
                           torch.tensor([0, 0], dtype=torch.long))
        with self.assertRaises(ValueError):
            sample_normalized_sparse_multipath_scenario_bank(ss, 3, 2)

    def test_rejects_weights_nan(self):
        ss = _src_shift_set()
        object.__setattr__(ss, "weights", torch.tensor([float("nan"), 1.0, 1.0, 1.0]))
        with self.assertRaises(ValueError):
            sample_normalized_sparse_multipath_scenario_bank(ss, 3, 2)

    def test_rejects_weights_negative(self):
        ss = _src_shift_set()
        object.__setattr__(ss, "weights", torch.tensor([-1.0, 1.0, 1.0, 1.0]))
        with self.assertRaises(ValueError):
            sample_normalized_sparse_multipath_scenario_bank(ss, 3, 2)

    def test_rejects_weights_requires_grad(self):
        w = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        ss = SparseShiftSet(
            shifts=torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 2]],
                                dtype=torch.long),
            weights=w,
        )
        with self.assertRaises(ValueError):
            sample_normalized_sparse_multipath_scenario_bank(ss, 3, 2)

    def test_rejects_num_scenarios_bool(self):
        with self.assertRaises(TypeError):
            sample_normalized_sparse_multipath_scenario_bank(
                _src_shift_set(), True, 2,
            )

    def test_rejects_num_scenarios_zero(self):
        with self.assertRaises(ValueError):
            sample_normalized_sparse_multipath_scenario_bank(
                _src_shift_set(), 0, 2,
            )

    def test_rejects_num_paths_bool(self):
        with self.assertRaises(TypeError):
            sample_normalized_sparse_multipath_scenario_bank(
                _src_shift_set(), 3, True,
            )

    def test_rejects_num_paths_zero(self):
        with self.assertRaises(ValueError):
            sample_normalized_sparse_multipath_scenario_bank(
                _src_shift_set(), 3, 0,
            )

    def test_rejects_invalid_complex_dtype(self):
        with self.assertRaises(TypeError):
            sample_normalized_sparse_multipath_scenario_bank(
                _src_shift_set(), 3, 2, complex_dtype=torch.float32,
            )

    def test_rejects_invalid_generator(self):
        with self.assertRaises(TypeError):
            sample_normalized_sparse_multipath_scenario_bank(
                _src_shift_set(), 3, 2, generator="bad",
            )

    def test_multinomial_called_once(self):
        from unittest.mock import patch
        with patch("torch.multinomial", wraps=torch.multinomial) as mock_mn:
            sample_normalized_sparse_multipath_scenario_bank(
                _src_shift_set(), 3, 2,
            )
            self.assertEqual(mock_mn.call_count, 1)
            self.assertTrue(torch.equal(
                mock_mn.call_args.args[0],
                torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64),
            ))

    def test_zero_raw_gain_norm_raises_value_error(self):
        from unittest.mock import patch
        with patch("torch.randn", return_value=torch.zeros(2, 2)):
            with self.assertRaises(ValueError):
                sample_normalized_sparse_multipath_scenario_bank(
                    _src_shift_set(), 2, 2,
                )

    def test_no_randperm_called(self):
        from unittest.mock import patch
        with patch("torch.randperm") as mock_rp:
            sample_normalized_sparse_multipath_scenario_bank(
                _src_shift_set(), 3, 2,
            )
            mock_rp.assert_not_called()


# ============================================================================
# E. Composition tests
# ============================================================================

class CompositionTests(unittest.TestCase):

    def test_sampled_bank_materialize_and_apply_operator(self):
        bank = sample_normalized_sparse_multipath_scenario_bank(
            _src_shift_set(), 4, 3,
        )
        ch = bank.materialize_channel(device="cpu", dtype=torch.complex64)
        x = torch.randn(4, 4, 6, dtype=torch.complex64)
        x.requires_grad_(True)
        y = apply_sparse_multipath_dd_operator(x, ch)
        self.assertEqual(y.shape, (4, 4, 6))
        loss = y.abs().pow(2).sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        # Scenario gains must be fixed, no grad.
        self.assertFalse(ch.path_gains.requires_grad)


# ============================================================================
# F. Double-weighting tests
# ============================================================================

class DoubleWeightingTests(unittest.TestCase):

    def test_sampled_bank_scenario_weights_is_none(self):
        bank = sample_normalized_sparse_multipath_scenario_bank(
            _src_shift_set(), 4, 3,
        )
        self.assertIsNone(bank.scenario_weights)

    def test_sampled_bank_normalized_weights_is_uniform(self):
        bank = sample_normalized_sparse_multipath_scenario_bank(
            _src_shift_set(), 4, 3,
        )
        q = bank.normalized_weights()
        self.assertTrue(torch.allclose(q, torch.full((4,), 0.25)))

    def test_q_used_only_in_sampling_not_as_scenario_weights(self):
        ss = SparseShiftSet(
            shifts=torch.tensor([[7, 3], [1, 0], [0, 1]], dtype=torch.long),
            weights=torch.tensor([100.0, 0.0, 0.0]),
        )
        bank = sample_normalized_sparse_multipath_scenario_bank(ss, 5, 2)
        # q=[1,0,0] biased sampling toward shift [7,3], but scenario_weights=None.
        self.assertIsNone(bank.scenario_weights)
        # Uniform normalized_weights, NOT [1,0,0].
        q = bank.normalized_weights()
        self.assertTrue(torch.allclose(q, torch.full((5,), 0.2)))


# ============================================================================
# G. Quality tests
# ============================================================================

class QualityTests(unittest.TestCase):

    def test_multipath_scenarios_py_ascii_only(self):
        path = TRANSMITTER_ROOT / "transmitter" / "multipath_scenarios.py"
        content = path.read_text(encoding="utf-8")
        for i, ch in enumerate(content):
            self.assertTrue(ord(ch) < 128,
                            f"Non-ASCII U+{ord(ch):04X} at offset {i}")

    def test_bank_docstring_exists(self):
        self.assertIsNotNone(SparseMultipathScenarioBank.__doc__)
        self.assertTrue(len(SparseMultipathScenarioBank.__doc__.strip()) > 0)

    def test_sampling_docstring_exists(self):
        self.assertIsNotNone(
            sample_normalized_sparse_multipath_scenario_bank.__doc__,
        )
        self.assertTrue(len(
            sample_normalized_sparse_multipath_scenario_bank.__doc__.strip()
        ) > 0)

    def test_norm_weights_docstring_exists(self):
        self.assertIsNotNone(
            SparseMultipathScenarioBank.normalized_weights.__doc__,
        )

    def test_materialize_docstring_exists(self):
        self.assertIsNotNone(
            SparseMultipathScenarioBank.materialize_channel.__doc__,
        )

    def test_docstrings_contain_shapes(self):
        doc = SparseMultipathScenarioBank.__doc__
        self.assertIn("[R, K, 2]", doc)
        doc_s = sample_normalized_sparse_multipath_scenario_bank.__doc__
        self.assertIn("[R, K, 2]", doc_s)
        self.assertIn("[R, K]", doc_s)
        self.assertIn("[R]", doc_s)

    def test_docstrings_unit_energy(self):
        doc = SparseMultipathScenarioBank.__doc__
        self.assertTrue("unit" in doc.lower() or "normaliz" in doc.lower())

    def test_docstrings_fixed_non_learnable(self):
        doc = SparseMultipathScenarioBank.__doc__
        self.assertTrue("fixed" in doc.lower()
                        or "non-learnable" in doc.lower())

    def test_exports_importable(self):
        from transmitter import (
            SparseMultipathScenarioBank as _SB,
            sample_normalized_sparse_multipath_scenario_bank as _sf,
        )
        self.assertIsInstance(_SB, type)
        self.assertTrue(callable(_sf))


if __name__ == "__main__":
    unittest.main()
