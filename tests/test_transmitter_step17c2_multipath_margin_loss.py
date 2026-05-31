"""Tests for multipath operator margin loss (Step 17C2)."""

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
    TokenDDCodebook,
    TransmitterConfig,
    build_transmitter_pilot_masks,
    sparse_multipath_operator_margin_loss,
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
        self.cw = _complex_randn(4, 4, 4)
        self.pairs = torch.tensor([[0, 1]], dtype=torch.long)
        self.bank = _simple_bank(R=2, K=1)
        self.mask = torch.ones(1, 4, 4)

    def test_output_scalar(self):
        loss = sparse_multipath_operator_margin_loss(
            self.cw, self.pairs, self.bank, self.mask,
        )
        self.assertEqual(loss.ndim, 0)

    def test_output_float32(self):
        loss = sparse_multipath_operator_margin_loss(
            self.cw, self.pairs, self.bank, self.mask,
        )
        self.assertEqual(loss.dtype, torch.float32)

    def test_output_float64_for_complex128(self):
        loss = sparse_multipath_operator_margin_loss(
            self.cw.to(dtype=torch.complex128),
            self.pairs,
            self.bank,
            self.mask,
        )
        self.assertEqual(loss.dtype, torch.float64)

    def test_output_device_equals_cw(self):
        loss = sparse_multipath_operator_margin_loss(
            self.cw, self.pairs, self.bank, self.mask,
        )
        self.assertEqual(loss.device, self.cw.device)

    def test_loss_finite(self):
        loss = sparse_multipath_operator_margin_loss(
            self.cw, self.pairs, self.bank, self.mask,
        )
        self.assertTrue(torch.isfinite(loss))

    def test_loss_non_negative(self):
        loss = sparse_multipath_operator_margin_loss(
            self.cw, self.pairs, self.bank, self.mask,
        )
        self.assertGreaterEqual(loss.item(), -1e-6)

    def test_scores_called_once(self):
        from unittest.mock import patch
        target = ("transmitter.multipath_margin."
                  "sparse_multipath_operator_separation_scores")
        with patch(target, wraps=sparse_multipath_operator_separation_scores
                   ) as mock_s:
            sparse_multipath_operator_margin_loss(
                self.cw, self.pairs, self.bank, self.mask,
            )
            self.assertEqual(mock_s.call_count, 1)


# ============================================================================
# B. Manual formula tests
# ============================================================================

class ManualFormulaTests(unittest.TestCase):

    def setUp(self):
        self.cw = _complex_randn(4, 4, 4)
        self.pairs = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
        self.bank = _simple_bank(R=3, K=1)
        self.mask = torch.ones(1, 4, 4)

    def test_matches_manual_formula(self):
        scores = sparse_multipath_operator_separation_scores(
            self.cw, self.pairs, self.bank, self.mask,
        )
        w = self.bank.normalized_weights(
            device=scores.device, dtype=scores.dtype,
        )
        tm = 1.5
        expected = (torch.relu(tm - scores).pow(2) * w.view(1, -1)
                    ).sum(dim=1).mean()
        loss = sparse_multipath_operator_margin_loss(
            self.cw, self.pairs, self.bank, self.mask, target_margin=tm,
        )
        self.assertTrue(torch.allclose(loss, expected))

    def test_uniform_weights_when_none(self):
        loss = sparse_multipath_operator_margin_loss(
            self.cw, self.pairs, self.bank, self.mask,
        )
        scores = sparse_multipath_operator_separation_scores(
            self.cw, self.pairs, self.bank, self.mask,
        )
        w_uniform = torch.full((3,), 1.0 / 3, device=scores.device,
                               dtype=scores.dtype)
        expected = (torch.relu(1.0 - scores).pow(2) * w_uniform.view(1, -1)
                    ).sum(dim=1).mean()
        self.assertTrue(torch.allclose(loss, expected))

    def test_explicit_weights_weighted_correctly(self):
        ch = self.bank.channel
        bank_w = SparseMultipathScenarioBank(
            channel=ch,
            scenario_weights=torch.tensor([1.0, 2.0, 3.0]),
        )
        scores = sparse_multipath_operator_separation_scores(
            self.cw, self.pairs, bank_w, self.mask,
        )
        w_norm = torch.tensor([1.0, 2.0, 3.0]) / 6.0
        w_norm = w_norm.to(device=scores.device, dtype=scores.dtype)
        expected = (torch.relu(1.0 - scores).pow(2) * w_norm.view(1, -1)
                    ).sum(dim=1).mean()
        loss = sparse_multipath_operator_margin_loss(
            self.cw, self.pairs, bank_w, self.mask,
        )
        self.assertTrue(torch.allclose(loss, expected))


# ============================================================================
# C. Hinge behavior tests
# ============================================================================

class HingeBehaviorTests(unittest.TestCase):

    def setUp(self):
        self.cw = _complex_randn(4, 4, 4)
        self.pairs = torch.tensor([[0, 1]], dtype=torch.long)
        self.bank = _simple_bank(R=1, K=1)
        self.mask = torch.ones(1, 4, 4)

    def test_score_above_margin_penalty_zero(self):
        loss = sparse_multipath_operator_margin_loss(
            self.cw, self.pairs, self.bank, self.mask, target_margin=0.0,
        )
        self.assertAlmostEqual(loss.item(), 0.0, places=4)

    def test_score_below_margin_penalty_correct(self):
        cw = torch.zeros(2, 4, 4, dtype=torch.complex64)
        cw[0, 0, 0] = 1.0
        cw[1, 0, 1] = 1.0
        pairs = torch.tensor([[0, 1]], dtype=torch.long)
        bank = _simple_bank(R=1, K=1)
        loss = sparse_multipath_operator_margin_loss(
            cw, pairs, bank, torch.ones(1, 4, 4), target_margin=10.0,
        )
        self.assertGreater(loss.item(), 0)

    def test_destructive_zero_score_full_penalty(self):
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
        tm = 3.0
        loss = sparse_multipath_operator_margin_loss(
            cw, pairs, bank, torch.ones(1, 4, 4), target_margin=tm,
        )
        self.assertAlmostEqual(loss.item(), tm ** 2, places=4)

    def test_target_margin_zero_loss_zero(self):
        loss = sparse_multipath_operator_margin_loss(
            self.cw, self.pairs, self.bank, self.mask, target_margin=0.0,
        )
        self.assertAlmostEqual(loss.item(), 0.0, places=4)

    def test_target_margin_zero_backward(self):
        cw = _complex_randn(4, 4, 4).requires_grad_(True)
        loss = sparse_multipath_operator_margin_loss(
            cw, self.pairs, self.bank, self.mask, target_margin=0.0,
        )
        loss.backward()
        self.assertIsNotNone(cw.grad)
        self.assertTrue(torch.isfinite(cw.grad).all())


# ============================================================================
# D. Fail-fast tests
# ============================================================================

class FailFastTests(unittest.TestCase):

    def setUp(self):
        self.cw = _complex_randn(4, 4, 4)
        self.pairs = torch.tensor([[0, 1]], dtype=torch.long)
        self.bank = _simple_bank()
        self.mask = torch.ones(1, 4, 4)

    def _mock_scores_and_assert_not_called(self, **kw):
        from unittest.mock import patch
        target = ("transmitter.multipath_margin."
                  "sparse_multipath_operator_separation_scores")
        with patch(target) as mock_s:
            with self.assertRaises((TypeError, ValueError)):
                sparse_multipath_operator_margin_loss(
                    self.cw, kw.get("pairs", self.pairs),
                    kw.get("bank", self.bank),
                    self.mask,
                    target_margin=kw.get("tm", 1.0),
                )
            mock_s.assert_not_called()

    def test_tm_bool_fail_fast(self):
        self._mock_scores_and_assert_not_called(tm=True)

    def test_tm_nan_fail_fast(self):
        self._mock_scores_and_assert_not_called(tm=float("nan"))

    def test_tm_inf_fail_fast(self):
        self._mock_scores_and_assert_not_called(tm=float("inf"))

    def test_tm_negative_fail_fast(self):
        self._mock_scores_and_assert_not_called(tm=-0.5)

    def test_tm_unrepresentable_for_score_dtype_fail_fast(self):
        self._mock_scores_and_assert_not_called(tm=1e100)

    def test_invalid_bank_fail_fast(self):
        self._mock_scores_and_assert_not_called(bank="bad")


# ============================================================================
# E. Raw weights corruption tests
# ============================================================================

class WeightsCorruptionTests(unittest.TestCase):

    def setUp(self):
        self.cw = _complex_randn(4, 4, 4)
        self.pairs = torch.tensor([[0, 1]], dtype=torch.long)
        self.mask = torch.ones(1, 4, 4)

    def _make_weighted_bank(self, R=2):
        ch = _simple_bank(R=R).channel
        return SparseMultipathScenarioBank(
            channel=ch,
            scenario_weights=torch.tensor([1.0, 2.0][:R]),
        )

    def _check_corruption_raises(self, bank, exc_type=ValueError):
        from unittest.mock import patch
        target = ("transmitter.multipath_margin."
                  "sparse_multipath_operator_separation_scores")
        with patch(target) as mock_s:
            with self.assertRaises(exc_type):
                sparse_multipath_operator_margin_loss(
                    self.cw, self.pairs, bank, self.mask,
                )
            mock_s.assert_not_called()

    def test_weights_negative_corruption(self):
        bank = self._make_weighted_bank()
        bank.scenario_weights.fill_(-1.0)
        self._check_corruption_raises(bank)

    def test_weights_nan_corruption(self):
        bank = self._make_weighted_bank()
        bank.scenario_weights[0] = float("nan")
        self._check_corruption_raises(bank)

    def test_weights_inf_corruption(self):
        bank = self._make_weighted_bank()
        bank.scenario_weights[0] = float("inf")
        self._check_corruption_raises(bank)

    def test_weights_all_zero_corruption(self):
        bank = self._make_weighted_bank()
        bank.scenario_weights.zero_()
        self._check_corruption_raises(bank)

    def test_weights_requires_grad_corruption(self):
        bank = self._make_weighted_bank()
        bank.scenario_weights.requires_grad_(True)
        self._check_corruption_raises(bank)


# ============================================================================
# F. Normalized weights defense-in-depth tests
# ============================================================================

class NormalizedWeightsValidationTests(unittest.TestCase):

    def test_rejects_post_score_normalized_weights_with_wrong_sum(self):
        from unittest.mock import patch
        cw = _complex_randn(4, 4, 4)
        pairs = torch.tensor([[0, 1]], dtype=torch.long)
        bank = _simple_bank(R=2, K=1)
        mask = torch.ones(1, 4, 4)
        invalid = torch.tensor([0.2, 0.2], dtype=torch.float32)
        with patch.object(
            SparseMultipathScenarioBank,
            "normalized_weights",
            return_value=invalid,
        ):
            with self.assertRaises(ValueError):
                sparse_multipath_operator_margin_loss(
                    cw, pairs, bank, mask,
                )


# ============================================================================
# G. Codebook integration + autograd
# ============================================================================

class IntegrationAutogradTests(unittest.TestCase):

    def test_codebook_autograd(self):
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
        loss = sparse_multipath_operator_margin_loss(
            cw, pairs, bank, masks.data_mask, target_margin=0.5,
        )
        self.assertTrue(loss.isfinite())
        loss.backward()
        self.assertIsNotNone(cb.raw_real.grad)
        self.assertIsNotNone(cb.raw_imag.grad)
        self.assertTrue(torch.isfinite(cb.raw_real.grad).all())

    def test_direct_cw_autograd(self):
        cw = _complex_randn(4, 4, 4).requires_grad_(True)
        loss = sparse_multipath_operator_margin_loss(
            cw, torch.tensor([[0, 1]], dtype=torch.long),
            _simple_bank(), torch.ones(1, 4, 4),
        )
        loss.backward()
        self.assertIsNotNone(cw.grad)
        self.assertTrue(torch.isfinite(cw.grad).all())


# ============================================================================
# H. CUDA tests
# ============================================================================

@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class CUDATests(unittest.TestCase):

    def test_cuda_loss_and_backward(self):
        cw = _complex_randn(4, 4, 4).to("cuda").requires_grad_(True)
        loss = sparse_multipath_operator_margin_loss(
            cw, torch.tensor([[0, 1]], dtype=torch.long),
            _simple_bank(), torch.ones(1, 4, 4),
        )
        self.assertTrue(loss.is_cuda)
        self.assertEqual(loss.dtype, torch.float32)
        loss.backward()
        self.assertTrue(torch.isfinite(cw.grad).all())


# ============================================================================
# I. Quality tests
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
            sparse_multipath_operator_margin_loss.__doc__,
        )
        self.assertTrue(len(
            sparse_multipath_operator_margin_loss.__doc__.strip()
        ) > 0)

    def test_docstring_contains_formula(self):
        doc = sparse_multipath_operator_margin_loss.__doc__
        self.assertIn("relu", doc)

    def test_docstring_contains_scalar(self):
        doc = sparse_multipath_operator_margin_loss.__doc__
        self.assertIn("[]", doc)

    def test_docstring_contains_PR(self):
        doc = sparse_multipath_operator_margin_loss.__doc__
        self.assertIn("[P, R]", doc)

    def test_docstring_global_margin(self):
        doc = sparse_multipath_operator_margin_loss.__doc__
        self.assertIn("global", doc.lower())

    def test_docstring_no_per_token(self):
        doc = sparse_multipath_operator_margin_loss.__doc__
        self.assertTrue("no per-token" in doc.lower()
                        or "no semantic" in doc.lower())

    def test_docstring_proxy_claim(self):
        doc = sparse_multipath_operator_margin_loss.__doc__
        self.assertIn("proxy", doc.lower())

    def test_docstring_tm_zero_autograd(self):
        doc = sparse_multipath_operator_margin_loss.__doc__
        self.assertTrue("target_margin == 0" in doc
                        or "autograd" in doc.lower())

    def test_exports_importable(self):
        from transmitter import (
            sparse_multipath_operator_margin_loss as _sm,
        )
        self.assertTrue(callable(_sm))


if __name__ == "__main__":
    unittest.main()
