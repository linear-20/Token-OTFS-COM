"""Tests for cross-token sparse orbit regularizer (Step 14)."""

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
    cross_token_orbit_confusion_loss,
    cross_token_orbit_confusion_scores,
    masked_shift_orbit_correlation,
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


def _default_shift_set():
    return SparseShiftSet(
        shifts=torch.tensor([[0, 0], [1, 0], [0, 1]], dtype=torch.long),
        weights=None,
        name="test_shifts",
    )


# ============================================================================
# A. Scores tests
# ============================================================================

class ScoresTests(unittest.TestCase):

    def setUp(self):
        self.cw = _complex_randn(6, 8, 10)
        self.pairs = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.long)
        self.ss = _default_shift_set()
        self.mask = torch.ones(1, 8, 10)

    def test_output_shape_PS(self):
        s = cross_token_orbit_confusion_scores(
            self.cw, self.pairs, self.ss, self.mask,
        )
        self.assertEqual(s.shape, (3, 3))

    def test_output_dtype(self):
        s = cross_token_orbit_confusion_scores(
            self.cw, self.pairs, self.ss, self.mask,
        )
        self.assertEqual(s.dtype, torch.float32)

    def test_output_device(self):
        s = cross_token_orbit_confusion_scores(
            self.cw, self.pairs, self.ss, self.mask,
        )
        self.assertEqual(s.device, self.cw.device)

    def test_equals_direct_masked_shift_orbit_correlation(self):
        s = cross_token_orbit_confusion_scores(
            self.cw, self.pairs, self.ss, self.mask,
        )
        ref = self.cw[self.pairs[:, 0]]
        cand = self.cw[self.pairs[:, 1]]
        expected = masked_shift_orbit_correlation(
            ref, cand, self.ss.shifts, self.mask,
        )
        self.assertTrue(torch.allclose(s, expected))

    def test_different_tokens_same_pattern_zero_shift_score_one(self):
        cw = torch.zeros(4, 4, 4, dtype=torch.complex64)
        cw[:] = _complex_randn(1, 4, 4).expand(4, -1, -1)
        pairs = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
        ss = SparseShiftSet(
            shifts=torch.tensor([[0, 0]], dtype=torch.long),
        )
        mask = torch.ones(1, 4, 4)
        s = cross_token_orbit_confusion_scores(cw, pairs, ss, mask)
        self.assertTrue(torch.allclose(s, torch.ones(2, 1), atol=1e-5))

    def test_orthogonal_patterns_zero_shift_score_zero(self):
        cw = torch.zeros(2, 2, 2, dtype=torch.complex64)
        cw[0, 0, 0] = 1.0 + 0j
        cw[1, 0, 1] = 1.0 + 0j
        pairs = torch.tensor([[0, 1]], dtype=torch.long)
        ss = SparseShiftSet(
            shifts=torch.tensor([[0, 0]], dtype=torch.long),
        )
        mask = torch.ones(1, 2, 2)
        s = cross_token_orbit_confusion_scores(cw, pairs, ss, mask)
        self.assertAlmostEqual(s[0, 0].item(), 0.0, places=5)

    def test_duplicate_pairs_kept(self):
        pairs = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
        s = cross_token_orbit_confusion_scores(
            self.cw, pairs, self.ss, self.mask,
        )
        self.assertTrue(torch.allclose(s[0], s[1]))

    def test_reverse_pairs_kept(self):
        pairs_ab = torch.tensor([[0, 1]], dtype=torch.long)
        pairs_ba = torch.tensor([[1, 0]], dtype=torch.long)
        s_ab = cross_token_orbit_confusion_scores(
            self.cw, pairs_ab, self.ss, self.mask,
        )
        s_ba = cross_token_orbit_confusion_scores(
            self.cw, pairs_ba, self.ss, self.mask,
        )
        self.assertEqual(s_ab.shape, (1, 3))
        self.assertEqual(s_ba.shape, (1, 3))

    def test_integration_with_codebook_and_data_mask(self):
        cfg = _tx_config()
        cb = TokenDDCodebook(cfg)
        masks = build_transmitter_pilot_masks(cfg)
        cw = cb.forward(data_mask=masks.data_mask)
        pairs = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.long)
        ss = _default_shift_set()
        s = cross_token_orbit_confusion_scores(
            cw, pairs, ss, masks.data_mask,
        )
        self.assertEqual(s.shape, (3, 3))

    def test_P_less_than_V2_output_only_PS(self):
        """Explicit cross-token pairs -> output is [P, S], not [V, V, S]."""
        cw = _complex_randn(6, 8, 10)
        pairs = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
        s = cross_token_orbit_confusion_scores(
            cw, pairs, self.ss, self.mask,
        )
        self.assertEqual(s.shape, (2, 3))
        self.assertNotEqual(s.shape[0], 6 * 6)


# ============================================================================
# B. Loss tests
# ============================================================================

class LossTests(unittest.TestCase):

    def setUp(self):
        self.cw = _complex_randn(6, 8, 10)
        self.pairs = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
        self.ss = _default_shift_set()
        self.mask = torch.ones(1, 8, 10)

    def _scores(self):
        return cross_token_orbit_confusion_scores(
            self.cw, self.pairs, self.ss, self.mask,
        )

    def test_weighted_isl_matches_manual(self):
        scores = self._scores()
        q = self.ss.normalized_weights(
            device=scores.device, dtype=scores.dtype,
        )
        expected = (scores.pow(2) * q.view(1, -1)).sum(dim=1).mean()
        loss = cross_token_orbit_confusion_loss(
            self.cw, self.pairs, self.ss, self.mask, mode="weighted_isl",
        )
        self.assertTrue(torch.allclose(loss, expected))

    def test_peak_equals_scores_max(self):
        scores = self._scores()
        loss = cross_token_orbit_confusion_loss(
            self.cw, self.pairs, self.ss, self.mask, mode="peak",
        )
        self.assertTrue(torch.allclose(loss, scores.max()))

    def test_peak_unaffected_by_shift_weights(self):
        ss_weighted = SparseShiftSet(
            shifts=torch.tensor([[0, 0], [1, 0], [0, 1]], dtype=torch.long),
            weights=torch.tensor([100.0, 0.01, 0.01]),
        )
        loss_w = cross_token_orbit_confusion_loss(
            self.cw, self.pairs, ss_weighted, self.mask, mode="peak",
        )
        loss_u = cross_token_orbit_confusion_loss(
            self.cw, self.pairs, self.ss, self.mask, mode="peak",
        )
        self.assertTrue(torch.allclose(loss_w, loss_u),
                        "peak must ignore shift weights values")

    def test_smooth_peak_matches_manual_logsumexp(self):
        scores = self._scores()
        q = self.ss.normalized_weights(
            device=scores.device, dtype=scores.dtype,
        )
        tau = 0.5
        P = scores.shape[0]
        log_q = torch.log(q)
        log_terms = scores / tau + log_q.view(1, -1)
        expected = tau * (torch.logsumexp(log_terms.reshape(-1), dim=0)
                          - math.log(P))
        loss = cross_token_orbit_confusion_loss(
            self.cw, self.pairs, self.ss, self.mask,
            mode="smooth_peak", risk_temperature=tau,
        )
        self.assertTrue(torch.allclose(loss, expected))

    def test_all_equal_scores_smooth_peak_equals_that_score(self):
        r"""Single pair, single shift, controlled rho = 0.7.
        ref = [1, 0], cand = [r, \sqrt{1-r^2}] after mask of ones."""
        r = 0.7
        cw = torch.zeros(2, 1, 2, dtype=torch.complex64)
        cw[0, 0, 0] = 1.0 + 0j       # ref = [1, 0]
        cw[1, 0, 0] = r + 0j
        cw[1, 0, 1] = math.sqrt(1 - r * r) + 0j  # cand = [r, sqrt(1-r^2)]
        pairs = torch.tensor([[0, 1]], dtype=torch.long)
        ss = SparseShiftSet(
            shifts=torch.tensor([[0, 0]], dtype=torch.long),
        )
        mask = torch.ones(1, 1, 2)
        scores = cross_token_orbit_confusion_scores(cw, pairs, ss, mask)
        self.assertAlmostEqual(scores[0, 0].item(), r, places=4)

        loss = cross_token_orbit_confusion_loss(
            cw, pairs, ss, mask, mode="smooth_peak",
        )
        self.assertAlmostEqual(loss.item(), r, places=4)

    def test_smooth_peak_zero_weight_excludes_shift(self):
        """smooth_peak excludes a configured shift whose weight is zero."""
        cw = torch.zeros(2, 1, 2, dtype=torch.complex64)
        cw[0, 0, 0] = 1.0 + 0j
        cw[1, 0, 1] = 1.0 + 0j
        pairs = torch.tensor([[0, 1]], dtype=torch.long)
        ss = SparseShiftSet(
            shifts=torch.tensor([[0, 0], [0, -1]], dtype=torch.long),
            weights=torch.tensor([1.0, 0.0]),
        )
        mask = torch.ones(1, 1, 2)

        scores = cross_token_orbit_confusion_scores(cw, pairs, ss, mask)
        self.assertTrue(torch.allclose(
            scores, torch.tensor([[0.0, 1.0]]), atol=1e-5,
        ))

        loss_smooth = cross_token_orbit_confusion_loss(
            cw, pairs, ss, mask, mode="smooth_peak",
        )
        loss_peak = cross_token_orbit_confusion_loss(
            cw, pairs, ss, mask, mode="peak",
        )
        self.assertAlmostEqual(loss_smooth.item(), 0.0, places=5)
        self.assertAlmostEqual(loss_peak.item(), 1.0, places=5)

    def test_smaller_tau_smooth_peak_closer_to_peak(self):
        loss_small_tau = cross_token_orbit_confusion_loss(
            self.cw, self.pairs, self.ss, self.mask,
            mode="smooth_peak", risk_temperature=0.01,
        )
        loss_large_tau = cross_token_orbit_confusion_loss(
            self.cw, self.pairs, self.ss, self.mask,
            mode="smooth_peak", risk_temperature=10.0,
        )
        loss_peak = cross_token_orbit_confusion_loss(
            self.cw, self.pairs, self.ss, self.mask, mode="peak",
        )
        # small tau closer to peak; large tau further.
        self.assertLess(
            abs(loss_small_tau.item() - loss_peak.item()),
            abs(loss_large_tau.item() - loss_peak.item()) + 1e-6,
        )

    def test_output_is_scalar(self):
        for mode in ("weighted_isl", "peak", "smooth_peak"):
            loss = cross_token_orbit_confusion_loss(
                self.cw, self.pairs, self.ss, self.mask, mode=mode,
            )
            self.assertEqual(loss.ndim, 0)

    def test_no_epsilon_no_clamp(self):
        cw = _complex_randn(6, 8, 10) * 0.01
        loss = cross_token_orbit_confusion_loss(
            cw, self.pairs, self.ss, self.mask, mode="peak",
        )
        self.assertTrue(torch.isfinite(loss))


# ============================================================================
# C. Validation tests
# ============================================================================

class ValidationTests(unittest.TestCase):

    def setUp(self):
        self.cw = _complex_randn(6, 8, 10)
        self.pairs = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
        self.ss = _default_shift_set()
        self.mask = torch.ones(1, 8, 10)

    def _call(self, **kw):
        return cross_token_orbit_confusion_scores(
            kw.get("cw", self.cw),
            kw.get("pairs", self.pairs),
            kw.get("ss", self.ss),
            kw.get("mask", self.mask),
        )

    # codeword_book
    def test_cw_non_tensor_raises_type_error(self):
        with self.assertRaises(TypeError):
            self._call(cw=[1.0])

    def test_cw_non_complex_raises_type_error(self):
        with self.assertRaises(TypeError):
            self._call(cw=torch.randn(6, 8, 10))

    def test_cw_wrong_ndim_raises_value_error(self):
        with self.assertRaises(ValueError):
            self._call(cw=_complex_randn(8, 10))

    def test_cw_V_zero_raises_value_error(self):
        with self.assertRaises(ValueError):
            self._call(cw=_complex_randn(0, 8, 10))

    def test_cw_M_zero_raises_value_error(self):
        with self.assertRaises(ValueError):
            self._call(cw=_complex_randn(6, 0, 10))

    def test_cw_N_zero_raises_value_error(self):
        with self.assertRaises(ValueError):
            self._call(cw=_complex_randn(6, 8, 0))

    def test_cw_nonfinite_raises_value_error(self):
        cw = self.cw.clone()
        cw[0, 0, 0] = complex(float("nan"), 0.0)
        with self.assertRaises(ValueError):
            self._call(cw=cw)

    # token_pairs
    def test_pairs_non_tensor_raises_type_error(self):
        with self.assertRaises(TypeError):
            self._call(pairs=[[0, 1]])

    def test_pairs_wrong_dtype_raises_type_error(self):
        with self.assertRaises(TypeError):
            self._call(pairs=torch.tensor([[0, 1]], dtype=torch.int32))

    def test_pairs_wrong_ndim_raises_value_error(self):
        with self.assertRaises(ValueError):
            self._call(pairs=torch.tensor([0, 1], dtype=torch.long))

    def test_pairs_last_dim_not_2_raises_value_error(self):
        with self.assertRaises(ValueError):
            self._call(pairs=torch.tensor([[0, 1, 2]], dtype=torch.long))

    def test_pairs_P_zero_raises_value_error(self):
        with self.assertRaises(ValueError):
            self._call(pairs=torch.empty(0, 2, dtype=torch.long))

    def test_pairs_negative_index_raises_value_error(self):
        with self.assertRaises(ValueError):
            self._call(pairs=torch.tensor([[-1, 1]], dtype=torch.long))

    def test_pairs_out_of_range_raises_value_error(self):
        with self.assertRaises(ValueError):
            self._call(pairs=torch.tensor([[0, 6]], dtype=torch.long))

    def test_self_pair_raises_value_error(self):
        with self.assertRaises(ValueError):
            self._call(pairs=torch.tensor([[0, 0]], dtype=torch.long))

    # shift_set
    def test_shift_set_not_sparseshiftset_raises_type_error(self):
        with self.assertRaises(TypeError):
            self._call(ss="not_a_shift_set")

    # mode / temperature
    def test_mode_non_str_raises_type_error(self):
        with self.assertRaises(TypeError):
            cross_token_orbit_confusion_loss(
                self.cw, self.pairs, self.ss, self.mask, mode=1,
            )

    def test_unknown_mode_raises_value_error(self):
        with self.assertRaises(ValueError):
            cross_token_orbit_confusion_loss(
                self.cw, self.pairs, self.ss, self.mask, mode="invalid",
            )

    def test_risk_temperature_bool_raises_type_error(self):
        with self.assertRaises(TypeError):
            cross_token_orbit_confusion_loss(
                self.cw, self.pairs, self.ss, self.mask,
                risk_temperature=True,
            )

    def test_risk_temperature_nonfinite_raises_value_error(self):
        with self.assertRaises(ValueError):
            cross_token_orbit_confusion_loss(
                self.cw, self.pairs, self.ss, self.mask,
                risk_temperature=float("inf"),
            )

    def test_risk_temperature_nonpositive_raises_value_error(self):
        with self.assertRaises(ValueError):
            cross_token_orbit_confusion_loss(
                self.cw, self.pairs, self.ss, self.mask,
                risk_temperature=0.0,
            )

    # corrupted shift weights (via modifying SparseShiftSet after creation)
    def test_corrupted_weights_nan_raises_value_error(self):
        ss_bad = SparseShiftSet(
            shifts=torch.tensor([[0, 0], [1, 0]], dtype=torch.long),
            weights=None,  # uniform -> normalized -> 0.5, 0.5
        )
        # Post-init corruption: inject NaN into weights
        object.__setattr__(ss_bad, "weights",
                           torch.tensor([float("nan"), 0.5]))
        with self.assertRaises(ValueError):
            cross_token_orbit_confusion_loss(
                self.cw, self.pairs, ss_bad, self.mask,
            )

    def test_corrupted_weights_negative_raises_value_error(self):
        ss_bad = SparseShiftSet(
            shifts=torch.tensor([[0, 0], [1, 0]], dtype=torch.long),
            weights=None,
        )
        object.__setattr__(ss_bad, "weights",
                           torch.tensor([-0.5, 1.5]))
        with self.assertRaises(ValueError):
            cross_token_orbit_confusion_loss(
                self.cw, self.pairs, ss_bad, self.mask,
            )

    def test_corrupted_weights_zero_sum_raises_value_error(self):
        ss_bad = SparseShiftSet(
            shifts=torch.tensor([[0, 0], [1, 0]], dtype=torch.long),
            weights=None,
        )
        object.__setattr__(ss_bad, "weights",
                           torch.tensor([0.0, 0.0]))
        with self.assertRaises(ValueError):
            cross_token_orbit_confusion_loss(
                self.cw, self.pairs, ss_bad, self.mask,
            )

    def test_corrupted_weights_all_negative_raises_value_error(self):
        """Raw weights [-1, -2] must be rejected before normalization."""
        ss_bad = SparseShiftSet(
            shifts=torch.tensor([[0, 0], [1, 0]], dtype=torch.long),
            weights=None,
        )
        object.__setattr__(ss_bad, "weights",
                           torch.tensor([-1.0, -2.0]))
        with self.assertRaises(ValueError):
            cross_token_orbit_confusion_loss(
                self.cw, self.pairs, ss_bad, self.mask,
            )

    def test_fail_fast_invalid_mode_before_scores(self):
        from unittest.mock import patch
        with patch(
            "transmitter.regularizers.cross_token_orbit_confusion_scores",
        ) as mock_scores:
            with self.assertRaises(ValueError):
                cross_token_orbit_confusion_loss(
                    self.cw, self.pairs, self.ss, self.mask,
                    mode="invalid",
                )
            mock_scores.assert_not_called()

    def test_fail_fast_bool_risk_temperature_before_scores(self):
        from unittest.mock import patch
        with patch(
            "transmitter.regularizers.cross_token_orbit_confusion_scores",
        ) as mock_scores:
            with self.assertRaises(TypeError):
                cross_token_orbit_confusion_loss(
                    self.cw, self.pairs, self.ss, self.mask,
                    risk_temperature=True,
                )
            mock_scores.assert_not_called()

    def test_fail_fast_invalid_shift_set_before_scores(self):
        from unittest.mock import patch
        with patch(
            "transmitter.regularizers.cross_token_orbit_confusion_scores",
        ) as mock_scores:
            with self.assertRaises(TypeError):
                cross_token_orbit_confusion_loss(
                    self.cw, self.pairs, "not_a_shift_set", self.mask,
                )
            mock_scores.assert_not_called()


# ============================================================================
# D. Autograd tests
# ============================================================================

class AutogradTests(unittest.TestCase):

    def setUp(self):
        self.cw = _complex_randn(6, 8, 10)
        self.cw.requires_grad_(True)
        self.pairs = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
        self.ss = _default_shift_set()
        self.mask = torch.ones(1, 8, 10)

    def _check_grad(self, loss):
        loss.backward()
        self.assertIsNotNone(self.cw.grad)
        self.assertTrue(torch.isfinite(self.cw.grad).all())

    def test_weighted_isl_backward(self):
        self.cw.grad = None
        loss = cross_token_orbit_confusion_loss(
            self.cw, self.pairs, self.ss, self.mask, mode="weighted_isl",
        )
        self._check_grad(loss)

    def test_peak_backward(self):
        self.cw.grad = None
        loss = cross_token_orbit_confusion_loss(
            self.cw, self.pairs, self.ss, self.mask, mode="peak",
        )
        self._check_grad(loss)

    def test_smooth_peak_backward(self):
        self.cw.grad = None
        loss = cross_token_orbit_confusion_loss(
            self.cw, self.pairs, self.ss, self.mask, mode="smooth_peak",
        )
        self._check_grad(loss)


# ============================================================================
# E. Quality tests
# ============================================================================

class QualityTests(unittest.TestCase):

    def test_regularizers_py_is_ascii_only(self):
        path = TRANSMITTER_ROOT / "transmitter" / "regularizers.py"
        content = path.read_text(encoding="utf-8")
        for i, ch in enumerate(content):
            self.assertTrue(ord(ch) < 128,
                            f"Non-ASCII U+{ord(ch):04X} at offset {i}")

    def test_scores_docstring_exists(self):
        doc = cross_token_orbit_confusion_scores.__doc__
        self.assertIsNotNone(doc)
        self.assertTrue(len(doc.strip()) > 0)

    def test_loss_docstring_exists(self):
        doc = cross_token_orbit_confusion_loss.__doc__
        self.assertIsNotNone(doc)
        self.assertTrue(len(doc.strip()) > 0)

    def test_scores_docstring_mentions_shapes(self):
        doc = cross_token_orbit_confusion_scores.__doc__
        self.assertIn("[V, M, N]", doc)
        self.assertIn("[P, 2]", doc)
        self.assertIn("[P, S]", doc)

    def test_exports_importable(self):
        from transmitter.ablations import (
            cross_token_orbit_confusion_loss as _cl,
            cross_token_orbit_confusion_scores as _cs,
        )
        self.assertTrue(callable(_cs))
        self.assertTrue(callable(_cl))


# ============================================================================
# F. CUDA tests
# ============================================================================

@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class CUDATests(unittest.TestCase):

    def test_cuda_scores_and_loss(self):
        cw = _complex_randn(6, 8, 10).to("cuda")
        pairs = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
        ss = _default_shift_set()
        mask = torch.ones(1, 8, 10)

        s = cross_token_orbit_confusion_scores(cw, pairs, ss, mask)
        self.assertTrue(s.is_cuda)

        loss = cross_token_orbit_confusion_loss(
            cw, pairs, ss, mask, mode="smooth_peak",
        )
        self.assertTrue(loss.is_cuda)
        self.assertEqual(loss.ndim, 0)

    def test_cuda_backward(self):
        cw = _complex_randn(6, 8, 10).to("cuda")
        cw.requires_grad_(True)
        pairs = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
        ss = _default_shift_set()
        mask = torch.ones(1, 8, 10)

        loss = cross_token_orbit_confusion_loss(
            cw, pairs, ss, mask, mode="weighted_isl",
        )
        loss.backward()
        self.assertIsNotNone(cw.grad)
        self.assertTrue(torch.isfinite(cw.grad).all())
        self.assertTrue(cw.grad.is_cuda)


if __name__ == "__main__":
    unittest.main()
