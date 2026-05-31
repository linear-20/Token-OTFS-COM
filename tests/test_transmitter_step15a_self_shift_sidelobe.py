"""Tests for nonzero self-shift orbit sidelobe regularizer (Step 15A)."""

import math
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
TRANSMITTER_ROOT = ROOT / "Learnable_Mapping_Tokens-to-DD-Signals"
sys.path.insert(0, str(TRANSMITTER_ROOT))

from transmitter import (
    SparseShiftSet,
    TokenDDCodebook,
    TransmitterConfig,
    build_transmitter_pilot_masks,
)
from transmitter.ablations import (
    masked_shift_orbit_correlation,
    self_shift_orbit_sidelobe_loss,
    self_shift_orbit_sidelobe_scores,
)


def _complex_randn(*shape, dtype=torch.complex64):
    rd = {torch.complex64: torch.float32, torch.complex128: torch.float64}[dtype]
    return torch.randn(*shape, dtype=rd) + 1j * torch.randn(*shape, dtype=rd)


def _tx_config(**kwargs):
    values = dict(
        M=8, N=10, vocab_size=6,
        pilot_delay=4, pilot_doppler=5,
        pilot_guard_delay=3, pilot_guard_doppler=3,
        pilot_obs_delay_radius=1, pilot_obs_doppler_radius=1,
        max_channel_delay=2, max_channel_doppler=2,
        data_power=1.5,
        pilot_value_real=2.0, pilot_value_imag=-1.0,
        complex_dtype="complex64",
    )
    values.update(kwargs)
    return TransmitterConfig(**values)


def _nonzero_shift_set():
    return SparseShiftSet(
        shifts=torch.tensor([[1, 0], [0, 1], [-1, 2]], dtype=torch.long),
        weights=None,
        name="nonzero_shifts",
    )


# ============================================================================
# A. Scores tests
# ============================================================================

class ScoresTests(unittest.TestCase):

    def setUp(self):
        self.cw = _complex_randn(4, 8, 10)
        self.ss = _nonzero_shift_set()
        self.mask = torch.ones(1, 8, 10)

    def test_output_shape_VS(self):
        s = self_shift_orbit_sidelobe_scores(self.cw, self.ss, self.mask)
        self.assertEqual(s.shape, (4, 3))

    def test_output_dtype(self):
        s = self_shift_orbit_sidelobe_scores(self.cw, self.ss, self.mask)
        self.assertEqual(s.dtype, torch.float32)

    def test_output_device(self):
        s = self_shift_orbit_sidelobe_scores(self.cw, self.ss, self.mask)
        self.assertEqual(s.device, self.cw.device)

    def test_equals_direct_masked_shift_orbit_correlation(self):
        s = self_shift_orbit_sidelobe_scores(self.cw, self.ss, self.mask)
        expected = masked_shift_orbit_correlation(
            self.cw, self.cw, self.ss.shifts, self.mask,
        )
        self.assertTrue(torch.allclose(s, expected))

    def test_one_hot_codeword_nonzero_shift_sidelobe_zero(self):
        cw = torch.zeros(2, 2, 2, dtype=torch.complex64)
        cw[0, 0, 0] = 1.0 + 0j
        cw[1, 0, 1] = 1.0 + 0j
        ss = SparseShiftSet(
            shifts=torch.tensor([[1, 0]], dtype=torch.long),
        )
        mask = torch.ones(1, 2, 2)
        s = self_shift_orbit_sidelobe_scores(cw, ss, mask)
        # Shift moves a one-hot off itself -> orthogonal -> score ~0.
        self.assertLess(s.abs().max().item(), 1e-4)

    def test_periodic_pattern_matching_shift_score_one(self):
        cw = torch.zeros(2, 2, 2, dtype=torch.complex64)
        cw[0, 0, 0] = 1.0 + 0j
        cw[0, 1, 0] = 1.0 + 0j  # periodic with period 1 in delay
        cw[1, 0, 0] = 1.0 + 0j
        ss = SparseShiftSet(
            shifts=torch.tensor([[1, 0]], dtype=torch.long),
        )
        mask = torch.ones(1, 2, 2)
        s = self_shift_orbit_sidelobe_scores(cw, ss, mask)
        # Shift (1,0) of token 0 aligns with original -> score ~1.
        self.assertGreater(s[0, 0].item(), 0.99)

    def test_integration_with_codebook_and_data_mask(self):
        cfg = _tx_config()
        cb = TokenDDCodebook(cfg)
        masks = build_transmitter_pilot_masks(cfg)
        cw = cb.forward(data_mask=masks.data_mask)
        ss = _nonzero_shift_set()
        s = self_shift_orbit_sidelobe_scores(cw, ss, masks.data_mask)
        self.assertEqual(s.shape, (cfg.vocab_size, 3))

    def test_output_is_VS_not_VV(self):
        s = self_shift_orbit_sidelobe_scores(self.cw, self.ss, self.mask)
        self.assertEqual(s.ndim, 2)
        self.assertEqual(s.shape, (4, 3))


# ============================================================================
# B. Nonzero shift contract tests
# ============================================================================

class NonzeroShiftTests(unittest.TestCase):

    def setUp(self):
        self.cw = _complex_randn(4, 8, 10)
        self.mask = torch.ones(1, 8, 10)

    def test_zero_shift_in_scores_raises_value_error(self):
        ss = SparseShiftSet(
            shifts=torch.tensor([[0, 0], [1, 0]], dtype=torch.long),
        )
        with self.assertRaises(ValueError):
            self_shift_orbit_sidelobe_scores(self.cw, ss, self.mask)

    def test_zero_shift_in_loss_raises_value_error(self):
        ss = SparseShiftSet(
            shifts=torch.tensor([[0, 0], [1, 0]], dtype=torch.long),
        )
        with self.assertRaises(ValueError):
            self_shift_orbit_sidelobe_loss(self.cw, ss, self.mask)

    def test_single_zero_shift_rejected(self):
        ss = SparseShiftSet(
            shifts=torch.tensor([[1, 0], [0, 0], [0, 1]], dtype=torch.long),
        )
        with self.assertRaises(ValueError):
            self_shift_orbit_sidelobe_scores(self.cw, ss, self.mask)

    def test_non_sparseshiftset_raises_type_error(self):
        with self.assertRaises(TypeError):
            self_shift_orbit_sidelobe_scores(self.cw, "bad", self.mask)

    def test_corrupted_shifts_non_tensor_raises_type_error(self):
        ss = _nonzero_shift_set()
        object.__setattr__(ss, "shifts", [[1, 0]])
        with self.assertRaises(TypeError):
            self_shift_orbit_sidelobe_scores(self.cw, ss, self.mask)

    def test_corrupted_shifts_wrong_dtype_raises_type_error(self):
        ss = _nonzero_shift_set()
        object.__setattr__(ss, "shifts",
                           torch.tensor([[1, 0]], dtype=torch.int32))
        with self.assertRaises(TypeError):
            self_shift_orbit_sidelobe_scores(self.cw, ss, self.mask)

    def test_corrupted_shifts_wrong_ndim_raises_value_error(self):
        ss = _nonzero_shift_set()
        object.__setattr__(ss, "shifts",
                           torch.tensor([1, 0], dtype=torch.long))
        with self.assertRaises(ValueError):
            self_shift_orbit_sidelobe_scores(self.cw, ss, self.mask)

    def test_corrupted_shifts_last_dim_not_2_raises_value_error(self):
        ss = _nonzero_shift_set()
        object.__setattr__(ss, "shifts",
                           torch.tensor([[1, 0, 2]], dtype=torch.long))
        with self.assertRaises(ValueError):
            self_shift_orbit_sidelobe_scores(self.cw, ss, self.mask)

    def test_corrupted_shifts_S_zero_raises_value_error(self):
        ss = _nonzero_shift_set()
        object.__setattr__(ss, "shifts",
                           torch.empty(0, 2, dtype=torch.long))
        with self.assertRaises(ValueError):
            self_shift_orbit_sidelobe_scores(self.cw, ss, self.mask)


# ============================================================================
# C. Loss tests
# ============================================================================

class LossTests(unittest.TestCase):

    def setUp(self):
        self.cw = _complex_randn(4, 8, 10)
        self.ss = _nonzero_shift_set()
        self.mask = torch.ones(1, 8, 10)

    def _scores(self):
        return self_shift_orbit_sidelobe_scores(self.cw, self.ss, self.mask)

    def test_weighted_isl_matches_manual(self):
        scores = self._scores()
        q = self.ss.normalized_weights(
            device=scores.device, dtype=scores.dtype,
        )
        expected = (scores.pow(2) * q.view(1, -1)).sum(dim=1).mean()
        loss = self_shift_orbit_sidelobe_loss(
            self.cw, self.ss, self.mask, mode="weighted_isl",
        )
        self.assertTrue(torch.allclose(loss, expected))

    def test_peak_equals_scores_max(self):
        scores = self._scores()
        loss = self_shift_orbit_sidelobe_loss(
            self.cw, self.ss, self.mask, mode="peak",
        )
        self.assertTrue(torch.allclose(loss, scores.max()))

    def test_peak_ignores_q_values_still_validates(self):
        shifts = torch.tensor([[1, 0], [0, 1], [-1, 2]], dtype=torch.long)
        ss_weighted = SparseShiftSet(
            shifts=shifts,
            weights=torch.tensor([100.0, 0.01, 0.0]),
        )
        ss_uniform = SparseShiftSet(
            shifts=shifts,
            weights=None,
        )
        loss_w = self_shift_orbit_sidelobe_loss(
            self.cw, ss_weighted, self.mask, mode="peak",
        )
        loss_u = self_shift_orbit_sidelobe_loss(
            self.cw, ss_uniform, self.mask, mode="peak",
        )
        self.assertTrue(torch.allclose(loss_w, loss_u),
                        "peak must ignore shift weights values")

    def test_smooth_peak_zero_weight_excludes_shift(self):
        cw = torch.zeros(2, 1, 4, dtype=torch.complex64)
        cw[:, 0, 0] = 1.0 + 0j
        cw[:, 0, 2] = 1.0 + 0j
        shifts = torch.tensor([[0, 1], [0, 2]], dtype=torch.long)
        ss = SparseShiftSet(
            shifts=shifts,
            weights=torch.tensor([1.0, 0.0]),
        )
        mask = torch.ones(1, 1, 4)
        scores = self_shift_orbit_sidelobe_scores(cw, ss, mask)
        self.assertTrue(torch.allclose(
            scores,
            torch.tensor([[0.0, 1.0], [0.0, 1.0]]),
        ))

        loss = self_shift_orbit_sidelobe_loss(
            cw, ss, mask, mode="smooth_peak", risk_temperature=0.2,
        )
        self.assertTrue(torch.allclose(loss, torch.zeros_like(loss)))

    def test_smooth_peak_matches_manual_logsumexp(self):
        scores = self._scores()
        q = self.ss.normalized_weights(
            device=scores.device, dtype=scores.dtype,
        )
        tau = 0.5
        V = scores.shape[0]
        log_q = torch.log(q)
        log_terms = scores / tau + log_q.view(1, -1)
        expected = tau * (torch.logsumexp(log_terms.reshape(-1), dim=0)
                          - math.log(V))
        loss = self_shift_orbit_sidelobe_loss(
            self.cw, self.ss, self.mask,
            mode="smooth_peak", risk_temperature=tau,
        )
        self.assertTrue(torch.allclose(loss, expected))

    def test_output_is_scalar(self):
        for mode in ("weighted_isl", "peak", "smooth_peak"):
            loss = self_shift_orbit_sidelobe_loss(
                self.cw, self.ss, self.mask, mode=mode,
            )
            self.assertEqual(loss.ndim, 0)

    def test_no_epsilon_no_clamp(self):
        loss = self_shift_orbit_sidelobe_loss(
            self.cw, self.ss, self.mask, mode="peak",
        )
        self.assertTrue(torch.isfinite(loss))


# ============================================================================
# D. Zero-energy tests
# ============================================================================

class ZeroEnergyTests(unittest.TestCase):

    def test_zero_shifted_candidate_energy_raises_in_scores(self):
        cw = torch.zeros(2, 2, 2, dtype=torch.complex64)
        cw[0, 0, 0] = 1.0 + 0j
        cw[1, 0, 0] = 1.0 + 0j
        # Mask: only bin (0,0) active. Shift (1,0) moves to (1,0)=blocked.
        mask = torch.zeros(1, 2, 2)
        mask[0, 0, 0] = 1.0
        ss = SparseShiftSet(
            shifts=torch.tensor([[1, 0]], dtype=torch.long),
        )
        with self.assertRaises(ValueError):
            self_shift_orbit_sidelobe_scores(cw, ss, mask)

    def test_zero_shifted_energy_raises_in_loss(self):
        cw = torch.zeros(2, 2, 2, dtype=torch.complex64)
        cw[0, 0, 0] = 1.0 + 0j
        cw[1, 0, 0] = 1.0 + 0j
        mask = torch.zeros(1, 2, 2)
        mask[0, 0, 0] = 1.0
        ss = SparseShiftSet(
            shifts=torch.tensor([[1, 0]], dtype=torch.long),
        )
        with self.assertRaises(ValueError):
            self_shift_orbit_sidelobe_loss(cw, ss, mask)


# ============================================================================
# E. Autograd tests
# ============================================================================

class AutogradTests(unittest.TestCase):

    def setUp(self):
        self.cw = _complex_randn(4, 8, 10)
        self.cw.requires_grad_(True)
        self.ss = _nonzero_shift_set()
        self.mask = torch.ones(1, 8, 10)

    def _check_grad(self, loss):
        loss.backward()
        self.assertIsNotNone(self.cw.grad)
        self.assertTrue(torch.isfinite(self.cw.grad).all())

    def test_weighted_isl_backward(self):
        self.cw.grad = None
        loss = self_shift_orbit_sidelobe_loss(
            self.cw, self.ss, self.mask, mode="weighted_isl",
        )
        self._check_grad(loss)

    def test_peak_backward(self):
        self.cw.grad = None
        loss = self_shift_orbit_sidelobe_loss(
            self.cw, self.ss, self.mask, mode="peak",
        )
        self._check_grad(loss)

    def test_smooth_peak_backward(self):
        self.cw.grad = None
        loss = self_shift_orbit_sidelobe_loss(
            self.cw, self.ss, self.mask, mode="smooth_peak",
        )
        self._check_grad(loss)


# ============================================================================
# F. Fail-fast tests
# ============================================================================

class FailFastTests(unittest.TestCase):

    def setUp(self):
        self.cw = _complex_randn(4, 8, 10)
        self.ss = _nonzero_shift_set()
        self.mask = torch.ones(1, 8, 10)

    def _patch_and_assert_not_called(self, **kwargs):
        from unittest.mock import patch
        target = "transmitter.regularizers.self_shift_orbit_sidelobe_scores"
        with patch(target) as mock_scores:
            with self.assertRaises((TypeError, ValueError)):
                self_shift_orbit_sidelobe_loss(
                    self.cw, kwargs.get("ss", self.ss),
                    self.mask, mode=kwargs.get("mode", "peak"),
                    risk_temperature=kwargs.get("tau", 1.0),
                )
            mock_scores.assert_not_called()

    def test_invalid_mode_before_scores(self):
        self._patch_and_assert_not_called(mode="invalid")

    def test_bool_risk_temperature_before_scores(self):
        self._patch_and_assert_not_called(tau=True)

    def test_nonfinite_risk_temperature_before_scores(self):
        self._patch_and_assert_not_called(tau=float("inf"))

    def test_nonpositive_risk_temperature_before_scores(self):
        self._patch_and_assert_not_called(tau=0.0)

    def test_zero_shift_set_before_scores(self):
        ss_bad = SparseShiftSet(
            shifts=torch.tensor([[0, 0], [1, 0]], dtype=torch.long),
        )
        self._patch_and_assert_not_called(ss=ss_bad)

    def test_invalid_shift_set_before_scores(self):
        self._patch_and_assert_not_called(ss="bad")

    def test_invalid_shift_weights_before_scores(self):
        ss_bad = SparseShiftSet(
            shifts=torch.tensor([[1, 0], [0, 1]], dtype=torch.long),
            weights=None,
        )
        object.__setattr__(ss_bad, "weights",
                           torch.tensor([-1.0, 2.0]))
        self._patch_and_assert_not_called(ss=ss_bad)


# ============================================================================
# G. Validation tests
# ============================================================================

class ValidationTests(unittest.TestCase):

    def setUp(self):
        self.cw = _complex_randn(4, 8, 10)
        self.ss = _nonzero_shift_set()
        self.mask = torch.ones(1, 8, 10)

    # codeword_book
    def test_cw_non_tensor_raises_type_error(self):
        with self.assertRaises(TypeError):
            self_shift_orbit_sidelobe_scores([1.0], self.ss, self.mask)

    def test_cw_non_complex_raises_type_error(self):
        with self.assertRaises(TypeError):
            self_shift_orbit_sidelobe_scores(
                torch.randn(4, 8, 10), self.ss, self.mask,
            )

    def test_cw_wrong_ndim_raises_value_error(self):
        with self.assertRaises(ValueError):
            self_shift_orbit_sidelobe_scores(
                _complex_randn(8, 10), self.ss, self.mask,
            )

    def test_cw_V_zero_raises_value_error(self):
        with self.assertRaises(ValueError):
            self_shift_orbit_sidelobe_scores(
                _complex_randn(0, 8, 10), self.ss, self.mask,
            )

    def test_cw_M_zero_raises_value_error(self):
        with self.assertRaises(ValueError):
            self_shift_orbit_sidelobe_scores(
                _complex_randn(4, 0, 10), self.ss, torch.ones(1, 0, 10),
            )

    def test_cw_N_zero_raises_value_error(self):
        with self.assertRaises(ValueError):
            self_shift_orbit_sidelobe_scores(
                _complex_randn(4, 8, 0), self.ss, torch.ones(1, 8, 0),
            )

    def test_cw_nonfinite_raises_value_error(self):
        cw = self.cw.clone()
        cw[0, 0, 0] = complex(float("nan"), 0.0)
        with self.assertRaises(ValueError):
            self_shift_orbit_sidelobe_scores(cw, self.ss, self.mask)

    # evidence_mask
    def test_mask_fractional_raises_value_error(self):
        with self.assertRaises(ValueError):
            self_shift_orbit_sidelobe_scores(
                self.cw, self.ss, torch.full((1, 8, 10), 0.5),
            )

    # corrupted shift weights
    def test_weights_nan_raises_value_error(self):
        ss_bad = _nonzero_shift_set()
        S = ss_bad.shifts.shape[0]
        object.__setattr__(ss_bad, "weights",
                           torch.full((S,), float("nan")))
        with self.assertRaises(ValueError):
            self_shift_orbit_sidelobe_loss(self.cw, ss_bad, self.mask)

    def test_weights_negative_raises_value_error(self):
        ss_bad = _nonzero_shift_set()
        S = ss_bad.shifts.shape[0]
        object.__setattr__(ss_bad, "weights",
                           torch.full((S,), -0.5))
        with self.assertRaises(ValueError):
            self_shift_orbit_sidelobe_loss(self.cw, ss_bad, self.mask)

    def test_weights_zero_sum_raises_value_error(self):
        ss_bad = _nonzero_shift_set()
        S = ss_bad.shifts.shape[0]
        object.__setattr__(ss_bad, "weights",
                           torch.zeros(S))
        with self.assertRaises(ValueError):
            self_shift_orbit_sidelobe_loss(self.cw, ss_bad, self.mask)


# ============================================================================
# H. Quality tests
# ============================================================================

class QualityTests(unittest.TestCase):

    def test_regularizers_py_is_ascii_only(self):
        path = TRANSMITTER_ROOT / "transmitter" / "regularizers.py"
        content = path.read_text(encoding="utf-8")
        for i, ch in enumerate(content):
            self.assertTrue(ord(ch) < 128,
                            f"Non-ASCII U+{ord(ch):04X} at offset {i}")

    def test_scores_docstring_exists(self):
        doc = self_shift_orbit_sidelobe_scores.__doc__
        self.assertIsNotNone(doc)
        self.assertTrue(len(doc.strip()) > 0)

    def test_loss_docstring_exists(self):
        doc = self_shift_orbit_sidelobe_loss.__doc__
        self.assertIsNotNone(doc)
        self.assertTrue(len(doc.strip()) > 0)

    def test_scores_docstring_mentions_shapes(self):
        doc = self_shift_orbit_sidelobe_scores.__doc__
        self.assertIn("[V, M, N]", doc)
        self.assertIn("[S, 2]", doc)
        self.assertIn("[V, S]", doc)

    def test_scores_docstring_mentions_zero_shift_rejected(self):
        doc = self_shift_orbit_sidelobe_scores.__doc__
        self.assertIn("zero shift", doc.lower())

    def test_exports_importable(self):
        from transmitter.ablations import (
            self_shift_orbit_sidelobe_loss as _sl,
            self_shift_orbit_sidelobe_scores as _ss,
        )
        self.assertTrue(callable(_ss))
        self.assertTrue(callable(_sl))


# ============================================================================
# I. CUDA tests
# ============================================================================

@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class CUDATests(unittest.TestCase):

    def test_cuda_scores_and_loss(self):
        cw = _complex_randn(4, 8, 10).to("cuda")
        ss = _nonzero_shift_set()
        mask = torch.ones(1, 8, 10)

        s = self_shift_orbit_sidelobe_scores(cw, ss, mask)
        self.assertTrue(s.is_cuda)

        loss = self_shift_orbit_sidelobe_loss(
            cw, ss, mask, mode="smooth_peak",
        )
        self.assertTrue(loss.is_cuda)
        self.assertEqual(loss.ndim, 0)

    def test_cuda_backward(self):
        cw = _complex_randn(4, 8, 10).to("cuda")
        cw.requires_grad_(True)
        ss = _nonzero_shift_set()
        mask = torch.ones(1, 8, 10)

        loss = self_shift_orbit_sidelobe_loss(
            cw, ss, mask, mode="weighted_isl",
        )
        loss.backward()
        self.assertIsNotNone(cw.grad)
        self.assertTrue(torch.isfinite(cw.grad).all())
        self.assertTrue(cw.grad.is_cuda)


if __name__ == "__main__":
    unittest.main()
