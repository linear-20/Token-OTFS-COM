"""Tests for multipath operator separation scores (Step 17C1)."""

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
    TokenDDCodebook,
    TransmitterConfig,
    build_transmitter_pilot_masks,
    sample_normalized_sparse_multipath_scenario_bank,
    sparse_multipath_operator_separation_scores,
)


def _complex_randn(*shape, dtype=torch.complex64):
    rd = {torch.complex64: torch.float32, torch.complex128: torch.float64}[dtype]
    return torch.randn(*shape, dtype=rd) + 1j * torch.randn(*shape, dtype=rd)


def _tx_config(**kw):
    v = dict(M=8, N=10, vocab_size=6, pilot_delay=4, pilot_doppler=5,
             pilot_guard_delay=3, pilot_guard_doppler=3,
             pilot_obs_delay_radius=1, pilot_obs_doppler_radius=1,
             max_channel_delay=2, max_channel_doppler=2,
             data_power=1.5, pilot_value_real=2.0, pilot_value_imag=-1.0,
             complex_dtype="complex64")
    v.update(kw)
    return TransmitterConfig(**v)


def _simple_bank(R=2, K=1):
    shifts = torch.zeros(R, K, 2, dtype=torch.long)
    gains = torch.ones(R, K, dtype=torch.complex64) / (K ** 0.5)
    return SparseMultipathScenarioBank(
        channel=SparseMultipathDDChannel(
            path_shifts=shifts, path_gains=gains,
            path_active_mask=torch.ones(R, K, dtype=torch.bool),
        ),
    )


# ============================================================================
# A. Basic output tests
# ============================================================================

class BasicOutputTests(unittest.TestCase):

    def setUp(self):
        self.cw = _complex_randn(6, 8, 10)
        self.pairs = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
        self.bank = _simple_bank(R=3, K=2)
        self.mask = torch.ones(1, 8, 10)

    def test_output_shape_PR(self):
        s = sparse_multipath_operator_separation_scores(
            self.cw, self.pairs, self.bank, self.mask,
        )
        self.assertEqual(s.shape, (2, 3))

    def test_output_real_float32(self):
        s = sparse_multipath_operator_separation_scores(
            self.cw, self.pairs, self.bank, self.mask,
        )
        self.assertEqual(s.dtype, torch.float32)

    def test_complex128_output_float64(self):
        cw = _complex_randn(4, 4, 4, dtype=torch.complex128)
        pairs = torch.tensor([[0, 1]], dtype=torch.long)
        ch = SparseMultipathDDChannel(
            path_shifts=torch.zeros(1, 1, 2, dtype=torch.long),
            path_gains=torch.ones(1, 1, dtype=torch.complex128),
            path_active_mask=torch.ones(1, 1, dtype=torch.bool),
        )
        bank = SparseMultipathScenarioBank(channel=ch)
        s = sparse_multipath_operator_separation_scores(
            cw, pairs, bank, torch.ones(1, 4, 4),
        )
        self.assertEqual(s.dtype, torch.float64)

    def test_output_device_equals_cw_device(self):
        s = sparse_multipath_operator_separation_scores(
            self.cw, self.pairs, self.bank, self.mask,
        )
        self.assertEqual(s.device, self.cw.device)

    def test_scores_finite(self):
        s = sparse_multipath_operator_separation_scores(
            self.cw, self.pairs, self.bank, self.mask,
        )
        self.assertTrue(torch.isfinite(s).all())

    def test_duplicate_pairs_preserved(self):
        pairs = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
        s = sparse_multipath_operator_separation_scores(
            self.cw, pairs, self.bank, self.mask,
        )
        self.assertTrue(torch.allclose(s[0], s[1]))

    def test_explicit_P_only_no_V2(self):
        s = sparse_multipath_operator_separation_scores(
            self.cw, self.pairs, self.bank, self.mask,
        )
        self.assertEqual(s.shape[0], 2)
        self.assertNotEqual(s.shape[0], 6 * 6)

    def test_scenario_weights_do_not_affect_scores(self):
        s1 = sparse_multipath_operator_separation_scores(
            self.cw, self.pairs, self.bank, self.mask,
        )
        weighted_bank = SparseMultipathScenarioBank(
            channel=self.bank.channel,
            scenario_weights=torch.tensor([0.98, 0.01, 0.01]),
        )
        s2 = sparse_multipath_operator_separation_scores(
            self.cw, self.pairs, weighted_bank, self.mask,
        )
        self.assertTrue(torch.allclose(s1, s2))


# ============================================================================
# B. Mathematical correctness tests
# ============================================================================

class MathCorrectnessTests(unittest.TestCase):

    def test_single_path_zero_shift_matches_manual(self):
        cw = _complex_randn(4, 4, 4)
        pairs = torch.tensor([[0, 1]], dtype=torch.long)
        delta = cw[0] - cw[1]
        h = (0.5 + 1j) / abs(0.5 + 1j)  # unit norm for K=1
        gains = torch.tensor([[h]], dtype=torch.complex64)
        ch = SparseMultipathDDChannel(
            path_shifts=torch.zeros(1, 1, 2, dtype=torch.long),
            path_gains=gains,
            path_active_mask=torch.ones(1, 1, dtype=torch.bool),
        )
        bank = SparseMultipathScenarioBank(channel=ch)
        mask = torch.ones(1, 4, 4)
        s = sparse_multipath_operator_separation_scores(
            cw, pairs, bank, mask,
        )
        expected = (mask * h * delta).abs().pow(2).sum() / mask.sum()
        self.assertAlmostEqual(s[0, 0].item(), expected.item(), places=4)

    def test_full_mask_unit_gain_equals_delta_norm_div_MN(self):
        cw = _complex_randn(4, 4, 4)
        pairs = torch.tensor([[0, 1]], dtype=torch.long)
        delta = cw[0] - cw[1]
        ch = SparseMultipathDDChannel(
            path_shifts=torch.zeros(1, 1, 2, dtype=torch.long),
            path_gains=torch.ones(1, 1, dtype=torch.complex64),
            path_active_mask=torch.ones(1, 1, dtype=torch.bool),
        )
        bank = SparseMultipathScenarioBank(channel=ch)
        mask = torch.ones(1, 4, 4)
        s = sparse_multipath_operator_separation_scores(
            cw, pairs, bank, mask,
        )
        expected = delta.abs().pow(2).sum() / 16
        self.assertAlmostEqual(s[0, 0].item(), expected.item(), places=4)

    def test_mask_after_operator_energy_moves_into_active(self):
        cw = torch.zeros(3, 2, 2, dtype=torch.complex64)
        cw[0, 0, 0] = 1.0
        pairs = torch.tensor([[0, 1]], dtype=torch.long)
        ch = SparseMultipathDDChannel(
            path_shifts=torch.tensor([[[0, 1]]], dtype=torch.long),
            path_gains=torch.ones(1, 1, dtype=torch.complex64),
            path_active_mask=torch.ones(1, 1, dtype=torch.bool),
        )
        bank = SparseMultipathScenarioBank(channel=ch)
        mask = torch.tensor([[[0.0, 1.0], [0.0, 0.0]]])
        s = sparse_multipath_operator_separation_scores(
            cw, pairs, bank, mask,
        )
        self.assertAlmostEqual(s[0, 0].item(), 1.0)

    def test_two_path_coherent_addition_matches_manual(self):
        cw = _complex_randn(3, 4, 4)
        pairs = torch.tensor([[0, 1]], dtype=torch.long)
        delta = cw[0] - cw[1]
        h1_raw, h2_raw = 1.0 + 0j, -0.5 + 0.5j
        norm = (abs(h1_raw)**2 + abs(h2_raw)**2) ** 0.5
        h1, h2 = h1_raw / norm, h2_raw / norm  # K=2, unit total energy
        ch = SparseMultipathDDChannel(
            path_shifts=torch.tensor([[[0, 0], [2, 0]]], dtype=torch.long),
            path_gains=torch.tensor([[h1, h2]], dtype=torch.complex64),
            path_active_mask=torch.ones(1, 2, dtype=torch.bool),
        )
        bank = SparseMultipathScenarioBank(channel=ch)
        mask = torch.ones(1, 4, 4)
        s = sparse_multipath_operator_separation_scores(
            cw, pairs, bank, mask,
        )
        from transmitter.dd_shifts import dd_circular_shift
        y_manual = (h1 * delta
                    + h2 * dd_circular_shift(delta.unsqueeze(0), 2, 0)[0])
        expected = (mask[0] * y_manual).abs().pow(2).sum() / mask.sum()
        self.assertAlmostEqual(s[0, 0].item(), expected.item(), places=4)

    def test_duplicate_shift_coherent_addition(self):
        cw = _complex_randn(3, 4, 4)
        pairs = torch.tensor([[0, 1]], dtype=torch.long)
        delta = cw[0] - cw[1]
        h1_raw, h2_raw = 0.3 + 0.4j, -0.2 + 0.1j
        n = (abs(h1_raw)**2 + abs(h2_raw)**2) ** 0.5
        h1, h2 = h1_raw / n, h2_raw / n
        ch = SparseMultipathDDChannel(
            path_shifts=torch.tensor([[[2, 0], [2, 0]]], dtype=torch.long),
            path_gains=torch.tensor([[h1, h2]], dtype=torch.complex64),
            path_active_mask=torch.ones(1, 2, dtype=torch.bool),
        )
        bank = SparseMultipathScenarioBank(channel=ch)
        mask = torch.ones(1, 4, 4)
        s = sparse_multipath_operator_separation_scores(
            cw, pairs, bank, mask,
        )
        from transmitter.dd_shifts import dd_circular_shift
        shifted = dd_circular_shift(delta.unsqueeze(0), 2, 0)[0]
        expected = ((h1 + h2) * shifted).abs().pow(2).sum() / 16
        self.assertAlmostEqual(s[0, 0].item(), expected.item(), places=4)

    def test_destructive_duplicate_shifts_score_zero(self):
        cw = _complex_randn(3, 4, 4)
        pairs = torch.tensor([[0, 1]], dtype=torch.long)
        a = 1.0 / (2 ** 0.5)
        ch = SparseMultipathDDChannel(
            path_shifts=torch.tensor([[[2, 0], [2, 0]]], dtype=torch.long),
            path_gains=torch.tensor([[a + 0j, -a + 0j]],
                                    dtype=torch.complex64),
            path_active_mask=torch.ones(1, 2, dtype=torch.bool),
        )
        bank = SparseMultipathScenarioBank(channel=ch)
        mask = torch.ones(1, 4, 4)
        s = sparse_multipath_operator_separation_scores(
            cw, pairs, bank, mask,
        )
        self.assertAlmostEqual(s[0, 0].item(), 0.0, places=4)

    def test_different_scenarios_different_scores(self):
        cw = _complex_randn(4, 4, 4)
        pairs = torch.tensor([[0, 1]], dtype=torch.long)
        # Different shifts with partial mask -> different scores.
        ch = SparseMultipathDDChannel(
            path_shifts=torch.tensor([[[0, 0]], [[1, 0]]], dtype=torch.long),
            path_gains=torch.ones(2, 1, dtype=torch.complex64),
            path_active_mask=torch.ones(2, 1, dtype=torch.bool),
        )
        bank = SparseMultipathScenarioBank(channel=ch)
        mask = torch.zeros(1, 4, 4)
        mask[0, 0, 0] = 1.0  # only (0,0) active
        s = sparse_multipath_operator_separation_scores(
            cw, pairs, bank, mask,
        )
        self.assertNotAlmostEqual(s[0, 0].item(), s[0, 1].item(), places=4)

    def test_zero_shift_scenario_legal(self):
        cw = _complex_randn(4, 4, 4)
        ch = SparseMultipathDDChannel(
            path_shifts=torch.zeros(1, 1, 2, dtype=torch.long),
            path_gains=torch.ones(1, 1, dtype=torch.complex64),
            path_active_mask=torch.ones(1, 1, dtype=torch.bool),
        )
        bank = SparseMultipathScenarioBank(channel=ch)
        s = sparse_multipath_operator_separation_scores(
            cw, torch.tensor([[0, 1]], dtype=torch.long),
            bank, torch.ones(1, 4, 4),
        )
        self.assertTrue(torch.isfinite(s).all())


# ============================================================================
# C. Mask tests
# ============================================================================

class MaskTests(unittest.TestCase):

    def setUp(self):
        self.cw = _complex_randn(4, 4, 4)
        self.pairs = torch.tensor([[0, 1]], dtype=torch.long)
        self.bank = _simple_bank(R=1, K=1)

    def test_bool_mask_accepted(self):
        s = sparse_multipath_operator_separation_scores(
            self.cw, self.pairs, self.bank,
            torch.ones(1, 4, 4, dtype=torch.bool),
        )
        self.assertEqual(s.shape, (1, 1))

    def test_real_binary_mask_accepted(self):
        s = sparse_multipath_operator_separation_scores(
            self.cw, self.pairs, self.bank, torch.ones(1, 4, 4),
        )
        self.assertEqual(s.shape, (1, 1))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cpu_mask_cuda_cw(self):
        cw = _complex_randn(4, 4, 4).to("cuda")
        s = sparse_multipath_operator_separation_scores(
            cw, self.pairs, self.bank, torch.ones(1, 4, 4),
        )
        self.assertTrue(s.is_cuda)

    def test_active_bin_count_is_denominator(self):
        mask = torch.ones(1, 4, 4)
        mask[0, 0, 0] = 0.0
        s = sparse_multipath_operator_separation_scores(
            self.cw, self.pairs, self.bank, mask,
        )
        self.assertGreater(s[0, 0].item(), 0)

    # -- reject ----------------------------------------------------------------
    def test_rejects_wrong_ndim(self):
        with self.assertRaises(ValueError):
            sparse_multipath_operator_separation_scores(
                self.cw, self.pairs, self.bank, torch.ones(4, 4),
            )

    def test_rejects_wrong_shape(self):
        with self.assertRaises(ValueError):
            sparse_multipath_operator_separation_scores(
                self.cw, self.pairs, self.bank, torch.ones(2, 4, 4),
            )

    def test_rejects_complex_mask(self):
        with self.assertRaises(TypeError):
            sparse_multipath_operator_separation_scores(
                self.cw, self.pairs, self.bank,
                torch.ones(1, 4, 4, dtype=torch.complex64),
            )

    def test_rejects_fractional_mask(self):
        with self.assertRaises(ValueError):
            sparse_multipath_operator_separation_scores(
                self.cw, self.pairs, self.bank,
                torch.full((1, 4, 4), 0.5),
            )

    def test_rejects_float64_fractional_mask_lost_by_float32_rounding(self):
        mask = torch.ones(1, 4, 4, dtype=torch.float64)
        mask[0, 0, 0] = 1.0 + 1e-8
        with self.assertRaises(ValueError):
            sparse_multipath_operator_separation_scores(
                self.cw, self.pairs, self.bank, mask,
            )

    def test_rejects_nonfinite_mask(self):
        mask = torch.ones(1, 4, 4)
        mask[0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            sparse_multipath_operator_separation_scores(
                self.cw, self.pairs, self.bank, mask,
            )

    def test_rejects_all_zero_mask(self):
        with self.assertRaises(ValueError):
            sparse_multipath_operator_separation_scores(
                self.cw, self.pairs, self.bank, torch.zeros(1, 4, 4),
            )


# ============================================================================
# D. Input validation tests
# ============================================================================

class ValidationTests(unittest.TestCase):

    def setUp(self):
        self.cw = _complex_randn(4, 4, 4)
        self.pairs = torch.tensor([[0, 1]], dtype=torch.long)
        self.bank = _simple_bank()
        self.mask = torch.ones(1, 4, 4)

    # codeword_book
    def test_rejects_cw_non_tensor(self):
        with self.assertRaises(TypeError):
            sparse_multipath_operator_separation_scores(
                [1.0], self.pairs, self.bank, self.mask,
            )

    def test_rejects_cw_non_complex(self):
        with self.assertRaises(TypeError):
            sparse_multipath_operator_separation_scores(
                torch.randn(4, 4, 4), self.pairs, self.bank, self.mask,
            )

    def test_rejects_cw_V_zero(self):
        with self.assertRaises(ValueError):
            sparse_multipath_operator_separation_scores(
                _complex_randn(0, 4, 4), self.pairs, self.bank, self.mask,
            )

    def test_rejects_cw_M_zero(self):
        with self.assertRaises(ValueError):
            sparse_multipath_operator_separation_scores(
                _complex_randn(4, 0, 4), self.pairs, self.bank, self.mask,
            )

    def test_rejects_cw_N_zero(self):
        with self.assertRaises(ValueError):
            sparse_multipath_operator_separation_scores(
                _complex_randn(4, 4, 0), self.pairs, self.bank, self.mask,
            )

    def test_rejects_cw_nonfinite(self):
        cw = self.cw.clone()
        cw[0, 0, 0] = complex(float("nan"), 0.0)
        with self.assertRaises(ValueError):
            sparse_multipath_operator_separation_scores(
                cw, self.pairs, self.bank, self.mask,
            )

    # token_pairs
    def test_rejects_pairs_non_tensor(self):
        with self.assertRaises(TypeError):
            sparse_multipath_operator_separation_scores(
                self.cw, [[0, 1]], self.bank, self.mask,
            )

    def test_rejects_pairs_non_long(self):
        with self.assertRaises(TypeError):
            sparse_multipath_operator_separation_scores(
                self.cw, torch.tensor([[0, 1]], dtype=torch.int32),
                self.bank, self.mask,
            )

    def test_rejects_pairs_wrong_shape(self):
        with self.assertRaises(ValueError):
            sparse_multipath_operator_separation_scores(
                self.cw, torch.tensor([0, 1], dtype=torch.long),
                self.bank, self.mask,
            )

    def test_rejects_pairs_P_zero(self):
        with self.assertRaises(ValueError):
            sparse_multipath_operator_separation_scores(
                self.cw, torch.empty(0, 2, dtype=torch.long),
                self.bank, self.mask,
            )

    def test_rejects_self_pair(self):
        with self.assertRaises(ValueError):
            sparse_multipath_operator_separation_scores(
                self.cw, torch.tensor([[0, 0]], dtype=torch.long),
                self.bank, self.mask,
            )

    def test_rejects_out_of_range(self):
        with self.assertRaises(ValueError):
            sparse_multipath_operator_separation_scores(
                self.cw, torch.tensor([[0, 4]], dtype=torch.long),
                self.bank, self.mask,
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cpu_pairs_cuda_cw(self):
        cw = _complex_randn(4, 4, 4).to("cuda")
        s = sparse_multipath_operator_separation_scores(
            cw, self.pairs, self.bank, self.mask,
        )
        self.assertTrue(s.is_cuda)

    # scenario_bank
    def test_rejects_non_bank(self):
        with self.assertRaises(TypeError):
            sparse_multipath_operator_separation_scores(
                self.cw, self.pairs, "bad", self.mask,
            )

    def test_source_bank_not_modified(self):
        orig_gains = self.bank.channel.path_gains.clone()
        sparse_multipath_operator_separation_scores(
            self.cw, self.pairs, self.bank, self.mask,
        )
        self.assertTrue(torch.equal(self.bank.channel.path_gains, orig_gains))


# ============================================================================
# E. Codebook integration + autograd
# ============================================================================

class IntegrationAndAutogradTests(unittest.TestCase):

    def test_codebook_integration(self):
        cfg = _tx_config()
        cb = TokenDDCodebook(cfg)
        masks = build_transmitter_pilot_masks(cfg)
        cw = cb.forward(data_mask=masks.data_mask)
        pairs = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
        bank = _simple_bank(R=2, K=1)
        s = sparse_multipath_operator_separation_scores(
            cw, pairs, bank, masks.data_mask,
        )
        self.assertEqual(s.shape, (2, 2))
        self.assertTrue(torch.isfinite(s).all())

    def test_autograd_codeword_book(self):
        cw = _complex_randn(4, 4, 4).requires_grad_(True)
        pairs = torch.tensor([[0, 1]], dtype=torch.long)
        bank = _simple_bank(R=2, K=1)
        s = sparse_multipath_operator_separation_scores(
            cw, pairs, bank, torch.ones(1, 4, 4),
        )
        loss = s.sum()
        loss.backward()
        self.assertIsNotNone(cw.grad)
        self.assertTrue(torch.isfinite(cw.grad).all())

    def test_codebook_autograd_raw_params(self):
        cfg = _tx_config(M=4, N=4, vocab_size=4, pilot_delay=2,
                        pilot_doppler=2, pilot_guard_delay=1,
                        pilot_guard_doppler=1, pilot_obs_delay_radius=0,
                        pilot_obs_doppler_radius=0,
                        max_channel_delay=1, max_channel_doppler=1)
        cb = TokenDDCodebook(cfg)
        masks = build_transmitter_pilot_masks(cfg)
        cw = cb.forward(data_mask=masks.data_mask)
        pairs = torch.tensor([[0, 1]], dtype=torch.long)
        bank = _simple_bank(R=1, K=1)
        s = sparse_multipath_operator_separation_scores(
            cw, pairs, bank, masks.data_mask,
        )
        loss = s.sum()
        loss.backward()
        self.assertIsNotNone(cb.raw_real.grad)
        self.assertIsNotNone(cb.raw_imag.grad)
        self.assertTrue(torch.isfinite(cb.raw_real.grad).all())


# ============================================================================
# F. CUDA tests
# ============================================================================

@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class CUDATests(unittest.TestCase):

    def test_cuda_fwd_and_backward(self):
        cw = _complex_randn(4, 4, 4).to("cuda").requires_grad_(True)
        pairs = torch.tensor([[0, 1]], dtype=torch.long)
        bank = _simple_bank(R=2, K=1)
        mask = torch.ones(1, 4, 4)
        s = sparse_multipath_operator_separation_scores(
            cw, pairs, bank, mask,
        )
        self.assertTrue(s.is_cuda)
        loss = s.sum()
        loss.backward()
        self.assertTrue(torch.isfinite(cw.grad).all())


# ============================================================================
# G. Quality tests
# ============================================================================

class QualityTests(unittest.TestCase):

    def test_multipath_margin_py_ascii_only(self):
        path = TRANSMITTER_ROOT / "transmitter" / "multipath_margin.py"
        content = path.read_text(encoding="utf-8")
        for i, ch in enumerate(content):
            self.assertTrue(ord(ch) < 128,
                            f"Non-ASCII U+{ord(ch):04X} at offset {i}")

    def test_docstring_exists(self):
        self.assertIsNotNone(
            sparse_multipath_operator_separation_scores.__doc__,
        )
        self.assertTrue(len(
            sparse_multipath_operator_separation_scores.__doc__.strip()
        ) > 0)

    def test_docstring_contains_shapes(self):
        doc = sparse_multipath_operator_separation_scores.__doc__
        self.assertIn("[V, M, N]", doc)
        self.assertIn("[P, 2]", doc)
        self.assertIn("[P, R]", doc)

    def test_docstring_contains_formula(self):
        doc = sparse_multipath_operator_separation_scores.__doc__
        self.assertIn("||", doc)

    def test_docstring_contains_complexity(self):
        doc = sparse_multipath_operator_separation_scores.__doc__
        self.assertIn("O(P * R * K * M * N)", doc)

    def test_docstring_mask_after_operator(self):
        doc = sparse_multipath_operator_separation_scores.__doc__
        self.assertTrue("after" in doc.lower()
                        and ("operator" in doc.lower()
                             or "H_r" in doc))

    def test_docstring_destructive_cancellation(self):
        doc = sparse_multipath_operator_separation_scores.__doc__
        self.assertTrue("d == 0" in doc or "destruct" in doc.lower())

    def test_docstring_proxy_claim(self):
        doc = sparse_multipath_operator_separation_scores.__doc__
        self.assertIn("proxy", doc.lower())

    def test_exports_importable(self):
        from transmitter import (
            sparse_multipath_operator_separation_scores as _sm,
        )
        self.assertTrue(callable(_sm))


if __name__ == "__main__":
    unittest.main()
