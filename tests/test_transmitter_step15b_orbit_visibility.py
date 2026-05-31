"""Tests for sparse DD shift-orbit visibility kernel and regularizer."""

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
    dd_circular_shift_bank,
    masked_shift_orbit_visibility,
    shift_orbit_visibility_loss,
    shift_orbit_visibility_scores,
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


# ============================================================================
# A. Kernel tests
# ============================================================================

class VisibilityKernelTests(unittest.TestCase):

    def setUp(self):
        self.cw = _complex_randn(3, 8, 10)
        self.shifts = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)
        self.mask = torch.ones(1, 8, 10)

    def test_output_shape_VS(self):
        eta = masked_shift_orbit_visibility(self.cw, self.shifts, self.mask)
        self.assertEqual(eta.shape, (3, 2))

    def test_output_dtype_complex64(self):
        eta = masked_shift_orbit_visibility(self.cw, self.shifts, self.mask)
        self.assertEqual(eta.dtype, torch.float32)

    def test_output_dtype_complex128(self):
        cw = _complex_randn(2, 4, 4, dtype=torch.complex128)
        eta = masked_shift_orbit_visibility(
            cw,
            torch.tensor([[0, 0]], dtype=torch.long),
            torch.ones(1, 4, 4),
        )
        self.assertEqual(eta.dtype, torch.float64)

    def test_output_device(self):
        eta = masked_shift_orbit_visibility(self.cw, self.shifts, self.mask)
        self.assertEqual(eta.device, self.cw.device)

    def test_matches_manual_formula(self):
        eta = masked_shift_orbit_visibility(self.cw, self.shifts, self.mask)
        shifted = dd_circular_shift_bank(self.cw, self.shifts)
        mask_dev = self.mask.to(device=self.cw.device,
                                dtype=torch.float32).reshape(1, 1, 8, 10)
        masked = shifted * mask_dev
        expected = (masked.abs().pow(2).sum(dim=(-2, -1))
                    / self.cw.abs().pow(2).sum(dim=(-2, -1)).unsqueeze(1))
        self.assertTrue(torch.allclose(eta, expected))

    def test_denominator_is_full_energy_not_masked_original(self):
        mask_partial = torch.zeros(1, 2, 2)
        mask_partial[0, 0, 0] = 1.0
        cw = torch.ones(2, 2, 2, dtype=torch.complex64)
        shifts = torch.tensor([[0, 0]], dtype=torch.long)
        eta = masked_shift_orbit_visibility(cw, shifts, mask_partial)
        # Full energy = 4, masked energy at shift 0 = 1 -> eta = 0.25
        self.assertAlmostEqual(eta[0, 0].item(), 0.25, places=4)

    def test_mask_after_shift_energy_moves_into_active_bin(self):
        cw = torch.zeros(1, 2, 3, dtype=torch.complex64)
        cw[0, 0, 0] = 1.0 + 0j
        # Mask: only bin (0,1) active. (0,0) is excluded.
        mask = torch.zeros(1, 2, 3)
        mask[0, 0, 1] = 1.0
        # Shift (0,1) moves energy from (0,0) to (0,1) = active.
        shifts = torch.tensor([[0, 1]], dtype=torch.long)
        eta = masked_shift_orbit_visibility(cw, shifts, mask)
        self.assertAlmostEqual(eta[0, 0].item(), 1.0, places=4)

    def test_eta_zero_when_shifted_out_of_evidence(self):
        cw = torch.zeros(1, 2, 2, dtype=torch.complex64)
        cw[0, 0, 0] = 1.0 + 0j
        mask = torch.zeros(1, 2, 2)
        mask[0, 0, 0] = 1.0  # only origin active
        shifts = torch.tensor([[0, 1]], dtype=torch.long)  # moves away
        eta = masked_shift_orbit_visibility(cw, shifts, mask)
        self.assertAlmostEqual(eta[0, 0].item(), 0.0, places=4)

    def test_zero_shift_legal(self):
        shifts = torch.tensor([[0, 0]], dtype=torch.long)
        eta = masked_shift_orbit_visibility(self.cw, shifts, self.mask)
        self.assertEqual(eta.shape, (3, 1))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cpu_mask_cuda_cw(self):
        cw = _complex_randn(2, 4, 4).to("cuda")
        eta = masked_shift_orbit_visibility(
            cw,
            torch.tensor([[0, 0]], dtype=torch.long),
            torch.ones(1, 4, 4),
        )
        self.assertTrue(eta.is_cuda)

    # -- reject paths ----------------------------------------------------------

    def test_cw_non_tensor_raises_type_error(self):
        with self.assertRaises(TypeError):
            masked_shift_orbit_visibility([1.0], self.shifts, self.mask)

    def test_cw_non_complex_raises_type_error(self):
        with self.assertRaises(TypeError):
            masked_shift_orbit_visibility(
                torch.randn(3, 8, 10), self.shifts, self.mask,
            )

    def test_cw_wrong_ndim_raises_value_error(self):
        with self.assertRaises(ValueError):
            masked_shift_orbit_visibility(
                _complex_randn(8, 10), self.shifts, self.mask,
            )

    def test_cw_V_zero_raises_value_error(self):
        with self.assertRaises(ValueError):
            masked_shift_orbit_visibility(
                _complex_randn(0, 8, 10), self.shifts, self.mask,
            )

    def test_cw_M_zero_raises_value_error(self):
        with self.assertRaises(ValueError):
            masked_shift_orbit_visibility(
                _complex_randn(3, 0, 10), self.shifts, torch.ones(1, 0, 10),
            )

    def test_cw_N_zero_raises_value_error(self):
        with self.assertRaises(ValueError):
            masked_shift_orbit_visibility(
                _complex_randn(3, 8, 0), self.shifts, torch.ones(1, 8, 0),
            )

    def test_cw_nonfinite_raises_value_error(self):
        cw = self.cw.clone()
        cw[0, 0, 0] = complex(float("nan"), 0.0)
        with self.assertRaises(ValueError):
            masked_shift_orbit_visibility(cw, self.shifts, self.mask)

    def test_shifts_non_tensor_raises_type_error(self):
        with self.assertRaises(TypeError):
            masked_shift_orbit_visibility(self.cw, [[0, 0]], self.mask)

    def test_shifts_wrong_dtype_raises_type_error(self):
        with self.assertRaises(TypeError):
            masked_shift_orbit_visibility(
                self.cw,
                torch.tensor([[0, 0]], dtype=torch.int32),
                self.mask,
            )

    def test_shifts_wrong_ndim_raises_value_error(self):
        with self.assertRaises(ValueError):
            masked_shift_orbit_visibility(
                self.cw, torch.tensor([0, 0], dtype=torch.long), self.mask,
            )

    def test_shifts_last_dim_not_2_raises_value_error(self):
        with self.assertRaises(ValueError):
            masked_shift_orbit_visibility(
                self.cw,
                torch.tensor([[0, 0, 0]], dtype=torch.long),
                self.mask,
            )

    def test_shifts_S_zero_raises_value_error(self):
        with self.assertRaises(ValueError):
            masked_shift_orbit_visibility(
                self.cw, torch.empty(0, 2, dtype=torch.long), self.mask,
            )

    def test_mask_fractional_raises_value_error(self):
        with self.assertRaises(ValueError):
            masked_shift_orbit_visibility(
                self.cw, self.shifts, torch.full((1, 8, 10), 0.5),
            )

    def test_mask_non_tensor_raises_type_error(self):
        with self.assertRaises(TypeError):
            masked_shift_orbit_visibility(self.cw, self.shifts, [[[1.0]]])

    def test_mask_complex_raises_type_error(self):
        with self.assertRaises(TypeError):
            masked_shift_orbit_visibility(
                self.cw, self.shifts,
                torch.ones(1, 8, 10, dtype=torch.complex64),
            )

    def test_mask_wrong_shape_raises_value_error(self):
        with self.assertRaises(ValueError):
            masked_shift_orbit_visibility(
                self.cw, self.shifts, torch.ones(1, 7, 10),
            )

    def test_mask_nonfinite_raises_value_error(self):
        mask = self.mask.clone()
        mask[0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            masked_shift_orbit_visibility(self.cw, self.shifts, mask)

    def test_zero_full_energy_raises_value_error(self):
        cw = torch.zeros(2, 4, 4, dtype=torch.complex64)
        with self.assertRaises(ValueError):
            masked_shift_orbit_visibility(
                cw,
                torch.tensor([[0, 0]], dtype=torch.long),
                torch.ones(1, 4, 4),
            )

    def test_autograd_preserved(self):
        cw = _complex_randn(2, 4, 4)
        cw.requires_grad_(True)
        eta = masked_shift_orbit_visibility(
            cw,
            torch.tensor([[0, 0]], dtype=torch.long),
            torch.ones(1, 4, 4),
        )
        loss = eta.sum()
        loss.backward()
        self.assertIsNotNone(cw.grad)
        self.assertTrue(torch.isfinite(cw.grad).all())


# ============================================================================
# B. Regularizer tests
# ============================================================================

class RegularizerTests(unittest.TestCase):

    def setUp(self):
        self.cw = _complex_randn(3, 8, 10)
        self.ss = SparseShiftSet(
            shifts=torch.tensor([[0, 0], [1, 0], [0, 1]], dtype=torch.long),
            weights=None,
        )
        self.mask = torch.ones(1, 8, 10)

    def test_scores_match_kernel(self):
        s = shift_orbit_visibility_scores(self.cw, self.ss, self.mask)
        expected = masked_shift_orbit_visibility(
            self.cw, self.ss.shifts, self.mask,
        )
        self.assertTrue(torch.allclose(s, expected))

    def test_weighted_hinge_matches_manual(self):
        eta = shift_orbit_visibility_scores(self.cw, self.ss, self.mask)
        q = self.ss.normalized_weights(device=eta.device, dtype=eta.dtype)
        floor = 0.3
        expected = (
            torch.relu(floor - eta).pow(2) * q.view(1, -1)
        ).sum(dim=1).mean()
        loss = shift_orbit_visibility_loss(
            self.cw, self.ss, self.mask, minimum_visibility=floor,
        )
        self.assertTrue(torch.allclose(loss, expected))

    def test_minimum_visibility_zero_loss_is_zero(self):
        loss = shift_orbit_visibility_loss(
            self.cw, self.ss, self.mask, minimum_visibility=0.0,
        )
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_minimum_visibility_zero_preserves_autograd_graph(self):
        cw = _complex_randn(2, 2, 2)
        cw.requires_grad_(True)
        ss = SparseShiftSet(
            shifts=torch.tensor([[0, 0]], dtype=torch.long),
        )
        loss = shift_orbit_visibility_loss(
            cw, ss, torch.ones(1, 2, 2), minimum_visibility=0.0,
        )
        self.assertTrue(loss.requires_grad)
        loss.backward()
        self.assertIsNotNone(cw.grad)
        self.assertTrue(torch.isfinite(cw.grad).all())

    def test_eta_above_floor_no_penalty(self):
        cw = torch.full((2, 2, 2), 1.0 + 0j, dtype=torch.complex64)
        ss = SparseShiftSet(
            shifts=torch.tensor([[0, 0]], dtype=torch.long),
        )
        mask = torch.ones(1, 2, 2)
        loss = shift_orbit_visibility_loss(
            cw, ss, mask, minimum_visibility=0.5,
        )
        # eta = 1.0 > 0.5 -> penalty = 0
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_eta_zero_produces_penalty(self):
        cw = torch.zeros(2, 2, 2, dtype=torch.complex64)
        cw[:, 0, 0] = 1.0 + 0j
        ss = SparseShiftSet(
            shifts=torch.tensor([[0, 1]], dtype=torch.long),
        )
        mask = torch.zeros(1, 2, 2)
        mask[0, 0, 0] = 1.0  # only origin active; shift moves energy away
        loss = shift_orbit_visibility_loss(
            cw, ss, mask, minimum_visibility=0.5,
        )
        self.assertAlmostEqual(loss.item(), 0.25, places=6)

    def test_zero_weight_shift_excluded(self):
        cw = torch.zeros(2, 2, 2, dtype=torch.complex64)
        cw[:, 0, 0] = 1.0 + 0j  # energy only at origin
        mask = torch.zeros(1, 2, 2)
        mask[0, 0, 0] = 1.0  # only origin active
        # Shift 0: eta = 1/1 = 1.0. Shift 1: energy moves to (0,1)=blocked,
        # eta = 0. With floor=0.5:
        #   q_excl=[1,0]: loss = 1*relu(0.5-1)^2 + 0*relu(0.5-0)^2 = 0
        #   q_both=[0.5,0.5]: loss = 0.5*0 + 0.5*0.25 = 0.125
        ss_excl = SparseShiftSet(
            shifts=torch.tensor([[0, 0], [0, 1]], dtype=torch.long),
            weights=torch.tensor([10.0, 0.0]),
        )
        loss_excl = shift_orbit_visibility_loss(
            cw, ss_excl, mask, minimum_visibility=0.5,
        )
        ss_both = SparseShiftSet(
            shifts=torch.tensor([[0, 0], [0, 1]], dtype=torch.long),
            weights=torch.tensor([1.0, 1.0]),
        )
        loss_both = shift_orbit_visibility_loss(
            cw, ss_both, mask, minimum_visibility=0.5,
        )
        self.assertAlmostEqual(loss_excl.item(), 0.0, places=6)
        self.assertAlmostEqual(loss_both.item(), 0.125, places=6)

    def test_zero_shift_legal_in_scores_and_loss(self):
        ss = SparseShiftSet(
            shifts=torch.tensor([[0, 0]], dtype=torch.long),
        )
        s = shift_orbit_visibility_scores(self.cw, ss, self.mask)
        self.assertEqual(s.shape, (3, 1))
        loss = shift_orbit_visibility_loss(
            self.cw, ss, self.mask, minimum_visibility=0.5,
        )
        self.assertEqual(loss.ndim, 0)

    # -- minimum_visibility validation -----------------------------------------

    def test_min_vis_bool_raises_type_error(self):
        with self.assertRaises(TypeError):
            shift_orbit_visibility_loss(
                self.cw, self.ss, self.mask, minimum_visibility=True,
            )

    def test_min_vis_non_number_raises_type_error(self):
        with self.assertRaises(TypeError):
            shift_orbit_visibility_loss(
                self.cw, self.ss, self.mask, minimum_visibility="0.5",
            )

    def test_min_vis_nan_raises_value_error(self):
        with self.assertRaises(ValueError):
            shift_orbit_visibility_loss(
                self.cw, self.ss, self.mask,
                minimum_visibility=float("nan"),
            )

    def test_min_vis_inf_raises_value_error(self):
        with self.assertRaises(ValueError):
            shift_orbit_visibility_loss(
                self.cw, self.ss, self.mask,
                minimum_visibility=float("inf"),
            )

    def test_min_vis_negative_raises_value_error(self):
        with self.assertRaises(ValueError):
            shift_orbit_visibility_loss(
                self.cw, self.ss, self.mask, minimum_visibility=-0.1,
            )

    def test_min_vis_above_one_raises_value_error(self):
        with self.assertRaises(ValueError):
            shift_orbit_visibility_loss(
                self.cw, self.ss, self.mask, minimum_visibility=1.1,
            )

    # -- corrupted weights -----------------------------------------------------

    def test_corrupted_weights_nan_raises(self):
        ss = SparseShiftSet(
            shifts=torch.tensor([[0, 0]], dtype=torch.long), weights=None,
        )
        object.__setattr__(ss, "weights", torch.tensor([float("nan")]))
        with self.assertRaises(ValueError):
            shift_orbit_visibility_loss(self.cw, ss, self.mask)

    def test_corrupted_weights_negative_raises(self):
        ss = SparseShiftSet(
            shifts=torch.tensor([[0, 0]], dtype=torch.long), weights=None,
        )
        object.__setattr__(ss, "weights", torch.tensor([-1.0]))
        with self.assertRaises(ValueError):
            shift_orbit_visibility_loss(self.cw, ss, self.mask)

    def test_corrupted_weights_zero_sum_raises(self):
        ss = SparseShiftSet(
            shifts=torch.tensor([[0, 0]], dtype=torch.long), weights=None,
        )
        object.__setattr__(ss, "weights", torch.tensor([0.0]))
        with self.assertRaises(ValueError):
            shift_orbit_visibility_loss(self.cw, ss, self.mask)

    # -- fail-fast -------------------------------------------------------------

    def test_invalid_min_vis_fail_fast_before_scores(self):
        from unittest.mock import patch
        target = "transmitter.regularizers.shift_orbit_visibility_scores"
        with patch(target) as mock_scores:
            with self.assertRaises(ValueError):
                shift_orbit_visibility_loss(
                    self.cw, self.ss, self.mask, minimum_visibility=-0.5,
                )
            mock_scores.assert_not_called()

    def test_invalid_weights_fail_fast_before_scores(self):
        from unittest.mock import patch
        target = "transmitter.regularizers.shift_orbit_visibility_scores"
        ss = SparseShiftSet(
            shifts=torch.tensor([[0, 0]], dtype=torch.long), weights=None,
        )
        object.__setattr__(ss, "weights", torch.tensor([-1.0]))
        with patch(target) as mock_scores:
            with self.assertRaises(ValueError):
                shift_orbit_visibility_loss(self.cw, ss, self.mask)
            mock_scores.assert_not_called()

    # -- integration -----------------------------------------------------------

    def test_integration_with_codebook(self):
        cfg = _tx_config()
        cb = TokenDDCodebook(cfg)
        masks = build_transmitter_pilot_masks(cfg)
        cw = cb.forward(data_mask=masks.data_mask)
        ss = SparseShiftSet(
            shifts=torch.tensor([[0, 0], [1, 0]], dtype=torch.long),
        )
        s = shift_orbit_visibility_scores(cw, ss, masks.data_mask)
        self.assertEqual(s.shape, (cfg.vocab_size, 2))
        loss = shift_orbit_visibility_loss(
            cw, ss, masks.data_mask, minimum_visibility=0.5,
        )
        self.assertEqual(loss.ndim, 0)
        loss.backward()
        self.assertTrue(torch.isfinite(cb.raw_real.grad).all())


# ============================================================================
# C. Quality tests
# ============================================================================

class QualityTests(unittest.TestCase):

    def test_orbit_visibility_py_ascii_only(self):
        path = TRANSMITTER_ROOT / "transmitter" / "orbit_visibility.py"
        content = path.read_text(encoding="utf-8")
        for i, ch in enumerate(content):
            self.assertTrue(ord(ch) < 128,
                            f"Non-ASCII U+{ord(ch):04X} at offset {i}")

    def test_regularizers_py_ascii_only(self):
        path = TRANSMITTER_ROOT / "transmitter" / "regularizers.py"
        content = path.read_text(encoding="utf-8")
        for i, ch in enumerate(content):
            self.assertTrue(ord(ch) < 128,
                            f"Non-ASCII U+{ord(ch):04X} at offset {i}")

    def test_kernel_docstring_exists(self):
        doc = masked_shift_orbit_visibility.__doc__
        self.assertIsNotNone(doc)
        self.assertTrue(len(doc.strip()) > 0)

    def test_scores_docstring_exists(self):
        doc = shift_orbit_visibility_scores.__doc__
        self.assertIsNotNone(doc)
        self.assertTrue(len(doc.strip()) > 0)

    def test_loss_docstring_exists(self):
        doc = shift_orbit_visibility_loss.__doc__
        self.assertIsNotNone(doc)
        self.assertTrue(len(doc.strip()) > 0)

    def test_kernel_docstring_mentions_shapes(self):
        doc = masked_shift_orbit_visibility.__doc__
        self.assertIn("[V, M, N]", doc)
        self.assertIn("[S, 2]", doc)
        self.assertIn("[V, S]", doc)

    def test_kernel_docstring_mask_after_shift(self):
        doc = masked_shift_orbit_visibility.__doc__
        self.assertTrue("after shift" in doc.lower()
                        or "AFTER" in doc)

    def test_kernel_docstring_full_energy_denominator(self):
        doc = masked_shift_orbit_visibility.__doc__
        self.assertIn("full", doc.lower())

    def test_kernel_docstring_eta_zero_valid(self):
        doc = masked_shift_orbit_visibility.__doc__
        self.assertIn("eta == 0", doc.lower()
                       or "eta == 0" in doc)

    def test_exports_importable(self):
        from transmitter.ablations import (
            masked_shift_orbit_visibility as _v,
            shift_orbit_visibility_loss as _vl,
            shift_orbit_visibility_scores as _vs,
        )
        self.assertTrue(callable(_v))
        self.assertTrue(callable(_vs))
        self.assertTrue(callable(_vl))


if __name__ == "__main__":
    unittest.main()
