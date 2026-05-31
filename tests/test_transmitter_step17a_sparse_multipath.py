"""Tests for sparse multipath DD operator (Step 17A)."""

import importlib
import inspect
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
TRANSMITTER_ROOT = ROOT / "Learnable_Mapping_Tokens-to-DD-Signals"
sys.path.insert(0, str(TRANSMITTER_ROOT))

from transmitter import (
    SparseMultipathDDChannel,
    apply_sparse_multipath_dd_operator,
)
from transmitter.dd_shifts import dd_circular_shift


def _complex_randn(*shape, dtype=torch.complex64):
    rd = {torch.complex64: torch.float32, torch.complex128: torch.float64}[dtype]
    return torch.randn(*shape, dtype=rd) + 1j * torch.randn(*shape, dtype=rd)


def _asymmetric_x_dd(B=2, M=4, N=6):
    """Non-symmetric x_dd to catch dim-swap bugs."""
    m = torch.arange(M, dtype=torch.float32).view(1, M, 1).expand(B, M, N)
    n = torch.arange(N, dtype=torch.float32).view(1, 1, N).expand(B, M, N)
    return (10.0 * m + n + 1j * (m - n)).to(torch.complex64)


# ============================================================================
# A. Dataclass tests
# ============================================================================

class DataclassTests(unittest.TestCase):

    def _valid_ch(self, B=2, K=3):
        return SparseMultipathDDChannel(
            path_shifts=torch.zeros(B, K, 2, dtype=torch.long),
            path_gains=_complex_randn(B, K),
            path_active_mask=None,
        )

    def test_valid_construction(self):
        ch = self._valid_ch()
        self.assertIsInstance(ch, SparseMultipathDDChannel)

    def test_shifts_shape_BK2(self):
        ch = self._valid_ch(B=3, K=4)
        self.assertEqual(ch.path_shifts.shape, (3, 4, 2))

    def test_gains_shape_BK(self):
        ch = self._valid_ch(B=3, K=4)
        self.assertEqual(ch.path_gains.shape, (3, 4))

    def test_active_mask_none_legal(self):
        ch = self._valid_ch()
        self.assertIsNone(ch.path_active_mask)

    def test_bool_active_mask_legal(self):
        ch = SparseMultipathDDChannel(
            path_shifts=torch.zeros(2, 3, 2, dtype=torch.long),
            path_gains=_complex_randn(2, 3),
            path_active_mask=torch.ones(2, 3, dtype=torch.bool),
        )
        self.assertIsNotNone(ch.path_active_mask)

    def test_duplicate_shifts_legal(self):
        ch = SparseMultipathDDChannel(
            path_shifts=torch.tensor([[[0, 0], [0, 0], [1, 2]]],
                                     dtype=torch.long).expand(2, -1, -1),
            path_gains=_complex_randn(2, 3),
        )
        self.assertEqual(ch.path_shifts.shape, (2, 3, 2))

    def test_signed_shifts_legal(self):
        ch = SparseMultipathDDChannel(
            path_shifts=torch.tensor([[[-3, 5], [0, -2]]], dtype=torch.long
                                     ).expand(2, -1, -1),
            path_gains=_complex_randn(2, 2),
        )
        self.assertEqual(ch.path_shifts.shape, (2, 2, 2))

    def test_gains_requires_grad_legal(self):
        gains = _complex_randn(2, 3)
        gains.requires_grad_(True)
        ch = SparseMultipathDDChannel(
            path_shifts=torch.zeros(2, 3, 2, dtype=torch.long),
            path_gains=gains,
        )
        self.assertTrue(ch.path_gains.requires_grad)

    # -- reject paths ----------------------------------------------------------

    def test_rejects_shifts_non_tensor(self):
        with self.assertRaises(TypeError):
            SparseMultipathDDChannel(
                path_shifts=[[[0, 0]]],
                path_gains=_complex_randn(1, 1),
            )

    def test_rejects_gains_non_tensor(self):
        with self.assertRaises(TypeError):
            SparseMultipathDDChannel(
                path_shifts=torch.zeros(1, 1, 2, dtype=torch.long),
                path_gains=[[1.0]],
            )

    def test_rejects_shifts_non_long(self):
        with self.assertRaises(TypeError):
            SparseMultipathDDChannel(
                path_shifts=torch.zeros(1, 1, 2, dtype=torch.int32),
                path_gains=_complex_randn(1, 1),
            )

    def test_rejects_gains_non_complex(self):
        with self.assertRaises(TypeError):
            SparseMultipathDDChannel(
                path_shifts=torch.zeros(1, 1, 2, dtype=torch.long),
                path_gains=torch.randn(1, 1),
            )

    def test_rejects_gains_nonfinite(self):
        g = _complex_randn(1, 1)
        g[0, 0] = complex(float("nan"), 0.0)
        with self.assertRaises(ValueError):
            SparseMultipathDDChannel(
                path_shifts=torch.zeros(1, 1, 2, dtype=torch.long),
                path_gains=g,
            )

    def test_rejects_wrong_ndim(self):
        with self.assertRaises(ValueError):
            SparseMultipathDDChannel(
                path_shifts=torch.zeros(1, 2, dtype=torch.long),
                path_gains=_complex_randn(1, 2),
            )

    def test_rejects_B_zero(self):
        with self.assertRaises(ValueError):
            SparseMultipathDDChannel(
                path_shifts=torch.empty(0, 1, 2, dtype=torch.long),
                path_gains=_complex_randn(0, 1),
            )

    def test_rejects_K_zero(self):
        with self.assertRaises(ValueError):
            SparseMultipathDDChannel(
                path_shifts=torch.empty(1, 0, 2, dtype=torch.long),
                path_gains=_complex_randn(1, 0),
            )

    def test_rejects_BK_mismatch(self):
        with self.assertRaises(ValueError):
            SparseMultipathDDChannel(
                path_shifts=torch.zeros(2, 3, 2, dtype=torch.long),
                path_gains=_complex_randn(3, 2),
            )

    def test_rejects_active_mask_non_tensor(self):
        with self.assertRaises(TypeError):
            SparseMultipathDDChannel(
                path_shifts=torch.zeros(1, 2, 2, dtype=torch.long),
                path_gains=_complex_randn(1, 2),
                path_active_mask=[[True, False]],
            )

    def test_rejects_active_mask_non_bool(self):
        with self.assertRaises(TypeError):
            SparseMultipathDDChannel(
                path_shifts=torch.zeros(1, 2, 2, dtype=torch.long),
                path_gains=_complex_randn(1, 2),
                path_active_mask=torch.ones(1, 2, dtype=torch.float32),
            )

    def test_rejects_active_mask_wrong_shape(self):
        with self.assertRaises(ValueError):
            SparseMultipathDDChannel(
                path_shifts=torch.zeros(1, 2, 2, dtype=torch.long),
                path_gains=_complex_randn(1, 2),
                path_active_mask=torch.ones(1, 3, dtype=torch.bool),
            )

    def test_docstring_mentions_shapes(self):
        doc = SparseMultipathDDChannel.__doc__
        self.assertIn("[B, K, 2]", doc)
        self.assertIn("[B, K]", doc)

    def test_docstring_active_mask_bool_gate(self):
        doc = SparseMultipathDDChannel.__doc__
        self.assertTrue("bool gate" in doc or "bool" in doc.lower())


# ============================================================================
# B. Operator math tests
# ============================================================================

class OperatorMathTests(unittest.TestCase):

    def setUp(self):
        self.x = _asymmetric_x_dd(B=2, M=4, N=6)
        self.g1 = 2.0 + 3j
        self.g2 = -1.0 + 1j

    def _ch(self, shifts, gains):
        return SparseMultipathDDChannel(
            path_shifts=shifts,
            path_gains=gains,
        )

    def test_single_zero_shift_path(self):
        ch = self._ch(
            torch.tensor([[[0, 0]]], dtype=torch.long).expand(2, -1, -1),
            torch.full((2, 1), self.g1, dtype=torch.complex64),
        )
        y = apply_sparse_multipath_dd_operator(self.x, ch)
        expected = self.g1 * self.x
        self.assertTrue(torch.allclose(y, expected))

    def test_single_positive_delay_shift(self):
        ch = self._ch(
            torch.tensor([[[2, 0]]], dtype=torch.long).expand(2, -1, -1),
            torch.full((2, 1), 1.0 + 0j, dtype=torch.complex64),
        )
        y = apply_sparse_multipath_dd_operator(self.x, ch)
        expected = dd_circular_shift(self.x, 2, 0)
        self.assertTrue(torch.allclose(y, expected))

    def test_single_negative_doppler_shift(self):
        ch = self._ch(
            torch.tensor([[[0, -3]]], dtype=torch.long).expand(2, -1, -1),
            torch.full((2, 1), 1.0 + 0j, dtype=torch.complex64),
        )
        y = apply_sparse_multipath_dd_operator(self.x, ch)
        expected = dd_circular_shift(self.x, 0, -3)
        self.assertTrue(torch.allclose(y, expected))

    def test_combined_signed_shift(self):
        ch = self._ch(
            torch.tensor([[[3, -2]]], dtype=torch.long).expand(2, -1, -1),
            torch.full((2, 1), self.g1, dtype=torch.complex64),
        )
        y = apply_sparse_multipath_dd_operator(self.x, ch)
        expected = self.g1 * dd_circular_shift(self.x, 3, -2)
        self.assertTrue(torch.allclose(y, expected))

    def test_two_path_superposition(self):
        shifts = torch.tensor([[[2, 0], [0, -3]]], dtype=torch.long
                              ).expand(2, -1, -1)
        gains = torch.tensor([[self.g1, self.g2]], dtype=torch.complex64
                             ).expand(2, -1)
        ch = SparseMultipathDDChannel(path_shifts=shifts, path_gains=gains)
        y = apply_sparse_multipath_dd_operator(self.x, ch)
        expected = (self.g1 * dd_circular_shift(self.x, 2, 0)
                    + self.g2 * dd_circular_shift(self.x, 0, -3))
        self.assertTrue(torch.allclose(y, expected, atol=1e-5))

    def test_duplicate_shifts_coherent_addition(self):
        shifts = torch.tensor([[[2, 0], [2, 0]]], dtype=torch.long
                              ).expand(2, -1, -1)
        gains = torch.tensor([[self.g1, self.g2]], dtype=torch.complex64
                             ).expand(2, -1)
        ch = SparseMultipathDDChannel(path_shifts=shifts, path_gains=gains)
        y = apply_sparse_multipath_dd_operator(self.x, ch)
        expected = (self.g1 + self.g2) * dd_circular_shift(self.x, 2, 0)
        self.assertTrue(torch.allclose(y, expected, atol=1e-5))

    def test_zero_gain_path_no_effect(self):
        shifts = torch.tensor([[[0, 0], [1, 0]]], dtype=torch.long
                              ).expand(2, -1, -1)
        gains = torch.tensor([[1.0 + 0j, 0.0 + 0j]], dtype=torch.complex64
                             ).expand(2, -1)
        ch = SparseMultipathDDChannel(path_shifts=shifts, path_gains=gains)
        y = apply_sparse_multipath_dd_operator(self.x, ch)
        self.assertTrue(torch.allclose(y, self.x, atol=1e-5))

    def test_inactive_path_no_effect(self):
        shifts = torch.tensor([[[0, 0], [1, 0]]], dtype=torch.long
                              ).expand(2, -1, -1)
        gains = torch.tensor([[1.0 + 0j, 10.0 + 0j]], dtype=torch.complex64
                             ).expand(2, -1)
        mask = torch.tensor([[True, False]]).expand(2, -1)
        ch = SparseMultipathDDChannel(
            path_shifts=shifts, path_gains=gains, path_active_mask=mask,
        )
        y = apply_sparse_multipath_dd_operator(self.x, ch)
        self.assertTrue(torch.allclose(y, self.x, atol=1e-5))

    def test_all_inactive_returns_zero_frame(self):
        shifts = torch.tensor([[[0, 0]]], dtype=torch.long).expand(2, -1, -1)
        gains = _complex_randn(2, 1)
        mask = torch.zeros(2, 1, dtype=torch.bool)
        ch = SparseMultipathDDChannel(
            path_shifts=shifts, path_gains=gains, path_active_mask=mask,
        )
        y = apply_sparse_multipath_dd_operator(self.x, ch)
        self.assertTrue(torch.allclose(y, torch.zeros_like(y)))

    def test_batch_specific_paths_independent(self):
        x = torch.stack([_asymmetric_x_dd(B=1, M=4, N=6)[0],
                         _asymmetric_x_dd(B=1, M=4, N=6)[0].clone() * 0])
        shifts = torch.tensor([[[0, 0]], [[0, 0]]], dtype=torch.long)
        gains = torch.tensor([[2.0 + 0j], [3.0 + 0j]], dtype=torch.complex64)
        ch = SparseMultipathDDChannel(path_shifts=shifts, path_gains=gains)
        y = apply_sparse_multipath_dd_operator(x, ch)
        self.assertAlmostEqual(y[0, 0, 0].item(),
                               2.0 * x[0, 0, 0].real.item(), places=4)
        self.assertAlmostEqual(y[1, 0, 0].real.item(), 0.0, places=4)

    def test_output_shape_device_dtype(self):
        ch = self._ch(
            torch.tensor([[[0, 0]]], dtype=torch.long).expand(2, -1, -1),
            torch.ones(2, 1, dtype=torch.complex64),
        )
        y = apply_sparse_multipath_dd_operator(self.x, ch)
        self.assertEqual(y.shape, self.x.shape)
        self.assertEqual(y.device, self.x.device)
        self.assertEqual(y.dtype, self.x.dtype)

    def test_inputs_not_modified(self):
        x_copy = self.x.clone()
        shifts = torch.tensor([[[2, 0]]], dtype=torch.long).expand(2, -1, -1)
        gains = torch.full((2, 1), self.g1, dtype=torch.complex64)
        g_copy = gains.clone()
        ch = SparseMultipathDDChannel(path_shifts=shifts, path_gains=gains)
        apply_sparse_multipath_dd_operator(self.x, ch)
        self.assertTrue(torch.equal(self.x, x_copy))
        self.assertTrue(torch.equal(gains, g_copy))

    def test_vectorized_matches_roll_reference_with_batch_specific_masks(self):
        torch.manual_seed(17)
        x = _complex_randn(5, 4, 6)
        shifts = torch.tensor([
            [[0, 0], [1, -2], [5, 7], [1, -2]],
            [[-1, 0], [0, 3], [2, -7], [0, 0]],
            [[4, 6], [3, -1], [-5, 2], [1, 1]],
            [[2, -3], [0, 0], [2, -3], [-2, 8]],
            [[7, -9], [1, 0], [0, -1], [4, 6]],
        ], dtype=torch.long)
        gains = _complex_randn(5, 4)
        mask = torch.tensor([
            [True, True, False, True],
            [False, True, True, True],
            [True, False, True, True],
            [False, False, False, False],
            [True, True, True, False],
        ])
        ch = SparseMultipathDDChannel(
            path_shifts=shifts, path_gains=gains,
            path_active_mask=mask,
        )
        actual = apply_sparse_multipath_dd_operator(x, ch)

        expected = []
        for batch_idx in range(x.shape[0]):
            y = x[batch_idx] * 0.0 + gains[batch_idx].sum() * 0.0
            for path_idx in range(shifts.shape[1]):
                if bool(mask[batch_idx, path_idx]):
                    delay, doppler = shifts[batch_idx, path_idx].tolist()
                    y = y + gains[batch_idx, path_idx] * torch.roll(
                        x[batch_idx], shifts=(delay, doppler), dims=(0, 1),
                    )
            expected.append(y)
        self.assertTrue(
            torch.allclose(actual, torch.stack(expected), atol=1e-5),
        )


# ============================================================================
# C. Receiver contract tests
# ============================================================================

class ReceiverContractTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        receiver_root = str(
            ROOT / "Learnable_Receiver_DD-Signals-to-Tokens",
        )
        sys.path.insert(0, receiver_root)
        try:
            cls.dd_ops = importlib.import_module("receiver.dd_ops")
        finally:
            if receiver_root in sys.path:
                sys.path.remove(receiver_root)

    def _compare(self, x_dd, channel):
        tx = apply_sparse_multipath_dd_operator(x_dd, channel)
        # Build receiver arguments from channel.
        gains = channel.path_gains
        shifts = channel.path_shifts.to(device=gains.device)
        if channel.path_active_mask is not None:
            conf = channel.path_active_mask.to(
                device=gains.device, dtype=gains.real.dtype,
            )
        else:
            conf = None
        rx = self.dd_ops.dd_circular_convolve_sparse(
            x_dd, path_indices=shifts, path_gains=gains, confidence=conf,
        )
        self.assertTrue(torch.allclose(tx, rx, atol=1e-5))

    def test_single_path(self):
        x = _asymmetric_x_dd(B=2)
        ch = SparseMultipathDDChannel(
            path_shifts=torch.tensor([[[2, 0]]], dtype=torch.long
                                     ).expand(2, -1, -1),
            path_gains=torch.full((2, 1), 1.5 + 2j, dtype=torch.complex64),
        )
        self._compare(x, ch)

    def test_two_paths(self):
        x = _asymmetric_x_dd(B=2)
        ch = SparseMultipathDDChannel(
            path_shifts=torch.tensor([[[2, 0], [0, -3]]], dtype=torch.long
                                     ).expand(2, -1, -1),
            path_gains=torch.tensor([[1.0 + 0j, -1.0 + 1j]],
                                    dtype=torch.complex64).expand(2, -1),
        )
        self._compare(x, ch)

    def test_batch_specific_paths(self):
        x = _asymmetric_x_dd(B=2)
        shifts = torch.tensor([[[2, 0]], [[0, -3]]], dtype=torch.long)
        gains = torch.tensor([[1.0 + 0j], [2.0 + 1j]], dtype=torch.complex64)
        ch = SparseMultipathDDChannel(path_shifts=shifts, path_gains=gains)
        self._compare(x, ch)

    def test_duplicate_paths(self):
        x = _asymmetric_x_dd(B=2)
        ch = SparseMultipathDDChannel(
            path_shifts=torch.tensor([[[2, 0], [2, 0]]], dtype=torch.long
                                     ).expand(2, -1, -1),
            path_gains=torch.tensor([[1.0 + 0j, 2.0 + 0j]],
                                    dtype=torch.complex64).expand(2, -1),
        )
        self._compare(x, ch)

    def test_inactive_padded_path(self):
        x = _asymmetric_x_dd(B=2)
        ch = SparseMultipathDDChannel(
            path_shifts=torch.tensor([[[0, 0], [1, 0]]], dtype=torch.long
                                     ).expand(2, -1, -1),
            path_gains=torch.tensor([[1.0 + 0j, 10.0 + 0j]],
                                    dtype=torch.complex64).expand(2, -1),
            path_active_mask=torch.tensor([[True, False]]).expand(2, -1),
        )
        self._compare(x, ch)

    def test_signed_shift(self):
        x = _asymmetric_x_dd(B=2)
        ch = SparseMultipathDDChannel(
            path_shifts=torch.tensor([[[-1, 3]]], dtype=torch.long
                                     ).expand(2, -1, -1),
            path_gains=torch.full((2, 1), 0.5 + 1j, dtype=torch.complex64),
        )
        self._compare(x, ch)

    def test_zero_shift(self):
        x = _asymmetric_x_dd(B=2)
        ch = SparseMultipathDDChannel(
            path_shifts=torch.tensor([[[0, 0]]], dtype=torch.long
                                     ).expand(2, -1, -1),
            path_gains=torch.full((2, 1), 1.0 + 0j, dtype=torch.complex64),
        )
        self._compare(x, ch)


# ============================================================================
# D. Validation tests
# ============================================================================

class ValidationTests(unittest.TestCase):

    def _good_ch(self):
        return SparseMultipathDDChannel(
            path_shifts=torch.zeros(2, 2, 2, dtype=torch.long),
            path_gains=_complex_randn(2, 2),
        )

    def test_rejects_x_dd_non_tensor(self):
        with self.assertRaises(TypeError):
            apply_sparse_multipath_dd_operator([1.0], self._good_ch())

    def test_rejects_x_dd_non_complex(self):
        with self.assertRaises(TypeError):
            apply_sparse_multipath_dd_operator(
                torch.randn(2, 4, 6), self._good_ch(),
            )

    def test_rejects_x_dd_nonfinite(self):
        x = _asymmetric_x_dd()
        x[0, 0, 0] = complex(float("nan"), 0.0)
        with self.assertRaises(ValueError):
            apply_sparse_multipath_dd_operator(x, self._good_ch())

    def test_rejects_x_dd_wrong_ndim(self):
        with self.assertRaises(ValueError):
            apply_sparse_multipath_dd_operator(
                _complex_randn(4, 6), self._good_ch(),
            )

    def test_rejects_x_dd_B_zero(self):
        with self.assertRaises(ValueError):
            apply_sparse_multipath_dd_operator(
                _complex_randn(0, 4, 6), self._good_ch(),
            )

    def test_rejects_x_dd_M_zero(self):
        with self.assertRaises(ValueError):
            apply_sparse_multipath_dd_operator(
                _complex_randn(2, 0, 6), self._good_ch(),
            )

    def test_rejects_x_dd_N_zero(self):
        with self.assertRaises(ValueError):
            apply_sparse_multipath_dd_operator(
                _complex_randn(2, 4, 0), self._good_ch(),
            )

    def test_rejects_channel_non_type(self):
        with self.assertRaises(TypeError):
            apply_sparse_multipath_dd_operator(_asymmetric_x_dd(), "bad")

    def test_rejects_channel_B_mismatch(self):
        ch = SparseMultipathDDChannel(
            path_shifts=torch.zeros(3, 2, 2, dtype=torch.long),
            path_gains=_complex_randn(3, 2),
        )
        with self.assertRaises(ValueError):
            apply_sparse_multipath_dd_operator(_asymmetric_x_dd(B=2), ch)

    def test_rejects_gains_device_mismatch(self):
        try:
            gains_cuda = _complex_randn(2, 2).to("cuda")
        except (AssertionError, RuntimeError):
            self.skipTest("CUDA not available")
        ch = SparseMultipathDDChannel(
            path_shifts=torch.zeros(2, 2, 2, dtype=torch.long),
            path_gains=gains_cuda,
        )
        with self.assertRaises(ValueError):
            apply_sparse_multipath_dd_operator(_asymmetric_x_dd(B=2), ch)

    def test_rejects_gains_dtype_mismatch(self):
        ch = SparseMultipathDDChannel(
            path_shifts=torch.zeros(2, 2, 2, dtype=torch.long),
            path_gains=_complex_randn(2, 2, dtype=torch.complex128),
        )
        with self.assertRaises(TypeError):
            apply_sparse_multipath_dd_operator(
                _asymmetric_x_dd(B=2), ch,
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cpu_shifts_cuda_x_dd_works(self):
        x = _asymmetric_x_dd(B=2).to("cuda")
        ch = SparseMultipathDDChannel(
            path_shifts=torch.zeros(2, 1, 2, dtype=torch.long),  # CPU
            path_gains=torch.ones(2, 1, dtype=torch.complex64).to("cuda"),
        )
        y = apply_sparse_multipath_dd_operator(x, ch)
        self.assertTrue(y.is_cuda)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cpu_mask_cuda_x_dd_works(self):
        x = _asymmetric_x_dd(B=2).to("cuda")
        ch = SparseMultipathDDChannel(
            path_shifts=torch.zeros(2, 2, 2, dtype=torch.long),
            path_gains=torch.ones(2, 2, dtype=torch.complex64).to("cuda"),
            path_active_mask=torch.ones(2, 2, dtype=torch.bool),  # CPU
        )
        y = apply_sparse_multipath_dd_operator(x, ch)
        self.assertTrue(y.is_cuda)


# ============================================================================
# E. Autograd tests
# ============================================================================

class AutogradTests(unittest.TestCase):

    def test_x_dd_and_gains_grad(self):
        x = _asymmetric_x_dd(B=2).clone().requires_grad_(True)
        gains = _complex_randn(2, 2).requires_grad_(True)
        ch = SparseMultipathDDChannel(
            path_shifts=torch.tensor([[[2, 0], [0, -3]]], dtype=torch.long
                                     ).expand(2, -1, -1),
            path_gains=gains,
        )
        y = apply_sparse_multipath_dd_operator(x, ch)
        loss = y.abs().pow(2).sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertIsNotNone(gains.grad)
        self.assertTrue(torch.isfinite(gains.grad).all())

    def test_inactive_path_gain_grad_is_zero(self):
        x = _asymmetric_x_dd(B=1).clone().requires_grad_(True)
        gains = torch.tensor([[1.0 + 0j, 10.0 + 0j]], dtype=torch.complex64,
                             requires_grad=True)
        mask = torch.tensor([[True, False]])
        ch = SparseMultipathDDChannel(
            path_shifts=torch.tensor([[[0, 0], [1, 0]]], dtype=torch.long),
            path_gains=gains,
            path_active_mask=mask,
        )
        y = apply_sparse_multipath_dd_operator(x, ch)
        loss = y.abs().pow(2).sum()
        loss.backward()
        # Inactive path gain (k=1) must have zero grad.
        self.assertAlmostEqual(gains.grad[0, 1].abs().item(), 0.0, places=5)
        # Active path gain (k=0) must have non-zero grad.
        self.assertGreater(gains.grad[0, 0].abs().item(), 0.0)

    def test_duplicate_shifts_gains_both_have_grad(self):
        x = _asymmetric_x_dd(B=1).clone().requires_grad_(True)
        gains = torch.tensor([[1.0 + 0j, 2.0 + 0j]], dtype=torch.complex64,
                             requires_grad=True)
        ch = SparseMultipathDDChannel(
            path_shifts=torch.tensor([[[2, 0], [2, 0]]], dtype=torch.long),
            path_gains=gains,
        )
        y = apply_sparse_multipath_dd_operator(x, ch)
        loss = y.abs().pow(2).sum()
        loss.backward()
        self.assertIsNotNone(gains.grad)
        self.assertTrue(torch.isfinite(gains.grad).all())

    def test_complex_gain_real_and_imag_grad_paths(self):
        x = torch.ones(1, 1, 1, dtype=torch.complex64, requires_grad=True)
        gains = torch.tensor(
            [[1.0 + 2.0j]], dtype=torch.complex64, requires_grad=True,
        )
        ch = SparseMultipathDDChannel(
            path_shifts=torch.tensor([[[0, 0]]], dtype=torch.long),
            path_gains=gains,
        )
        y = apply_sparse_multipath_dd_operator(x, ch)
        y.abs().pow(2).sum().backward()
        self.assertGreater(gains.grad.real.abs().item(), 0.0)
        self.assertGreater(gains.grad.imag.abs().item(), 0.0)

    def test_all_inactive_preserves_zero_grad_graph(self):
        x = _asymmetric_x_dd(B=1).clone().requires_grad_(True)
        gains = torch.tensor(
            [[1.0 + 2.0j]], dtype=torch.complex64, requires_grad=True,
        )
        ch = SparseMultipathDDChannel(
            path_shifts=torch.tensor([[[0, 0]]], dtype=torch.long),
            path_gains=gains,
            path_active_mask=torch.tensor([[False]]),
        )
        y = apply_sparse_multipath_dd_operator(x, ch)
        self.assertTrue(torch.equal(y, torch.zeros_like(y)))
        y.abs().pow(2).sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.equal(x.grad, torch.zeros_like(x.grad)))
        self.assertIsNotNone(gains.grad)
        self.assertTrue(torch.equal(gains.grad, torch.zeros_like(gains.grad)))

    def test_vectorized_grad_matches_roll_reference(self):
        torch.manual_seed(19)
        shifts = torch.tensor([
            [[0, 0], [1, -2], [5, 7]],
            [[-1, 0], [0, 3], [2, -7]],
        ], dtype=torch.long)
        mask = torch.tensor([
            [True, False, True],
            [True, True, False],
        ])
        x_actual = _complex_randn(2, 4, 6).requires_grad_(True)
        gains_actual = _complex_randn(2, 3).requires_grad_(True)
        ch = SparseMultipathDDChannel(
            path_shifts=shifts, path_gains=gains_actual,
            path_active_mask=mask,
        )
        actual = apply_sparse_multipath_dd_operator(x_actual, ch)
        actual.abs().pow(2).sum().backward()

        x_reference = x_actual.detach().clone().requires_grad_(True)
        gains_reference = gains_actual.detach().clone().requires_grad_(True)
        expected = []
        for batch_idx in range(x_reference.shape[0]):
            y = x_reference[batch_idx] * 0.0
            for path_idx in range(shifts.shape[1]):
                if bool(mask[batch_idx, path_idx]):
                    delay, doppler = shifts[batch_idx, path_idx].tolist()
                    y = y + gains_reference[batch_idx, path_idx] * torch.roll(
                        x_reference[batch_idx],
                        shifts=(delay, doppler), dims=(0, 1),
                    )
            expected.append(y)
        torch.stack(expected).abs().pow(2).sum().backward()

        self.assertTrue(torch.allclose(
            x_actual.grad, x_reference.grad, atol=1e-4,
        ))
        self.assertTrue(torch.allclose(
            gains_actual.grad, gains_reference.grad, atol=1e-4,
        ))


# ============================================================================
# F. Quality tests
# ============================================================================

class QualityTests(unittest.TestCase):

    def test_sparse_multipath_py_ascii_only(self):
        path = TRANSMITTER_ROOT / "transmitter" / "sparse_multipath.py"
        content = path.read_text(encoding="utf-8")
        for i, ch in enumerate(content):
            self.assertTrue(ord(ch) < 128,
                            f"Non-ASCII U+{ord(ch):04X} at offset {i}")

    def test_channel_docstring_exists(self):
        self.assertIsNotNone(SparseMultipathDDChannel.__doc__)
        self.assertTrue(len(SparseMultipathDDChannel.__doc__.strip()) > 0)

    def test_operator_docstring_exists(self):
        self.assertIsNotNone(apply_sparse_multipath_dd_operator.__doc__)
        self.assertTrue(
            len(apply_sparse_multipath_dd_operator.__doc__.strip()) > 0,
        )

    def test_channel_docstring_mentions_shapes(self):
        doc = SparseMultipathDDChannel.__doc__
        self.assertIn("[B, K, 2]", doc)
        self.assertIn("[B, K]", doc)

    def test_operator_docstring_mentions_shapes(self):
        doc = apply_sparse_multipath_dd_operator.__doc__
        self.assertIn("[B, M, N]", doc)

    def test_operator_docstring_not_twisted_convolution(self):
        doc = apply_sparse_multipath_dd_operator.__doc__
        self.assertIn("NOT", doc)

    def test_operator_hot_path_has_no_python_batch_loop_or_item(self):
        source = inspect.getsource(apply_sparse_multipath_dd_operator)
        self.assertNotIn("for b in range", source)
        self.assertNotIn(".item()", source)

    def test_exports_importable(self):
        from transmitter import (
            SparseMultipathDDChannel as _SMDC,
            apply_sparse_multipath_dd_operator as _aso,
        )
        self.assertIsInstance(_SMDC, type)
        self.assertTrue(callable(_aso))


# ============================================================================
# G. CUDA tests
# ============================================================================

@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class CUDATests(unittest.TestCase):

    def test_cuda_operator(self):
        x = _asymmetric_x_dd(B=2).to("cuda")
        ch = SparseMultipathDDChannel(
            path_shifts=torch.tensor([[[2, 0]]], dtype=torch.long
                                     ).expand(2, -1, -1),  # CPU
            path_gains=torch.full((2, 1), 1.0 + 0j,
                                  dtype=torch.complex64).to("cuda"),
            path_active_mask=None,
        )
        y = apply_sparse_multipath_dd_operator(x, ch)
        self.assertTrue(y.is_cuda)
        self.assertEqual(y.dtype, torch.complex64)

    def test_cuda_backward(self):
        x = _asymmetric_x_dd(B=2).to("cuda").requires_grad_(True)
        gains = _complex_randn(2, 2).to("cuda").requires_grad_(True)
        ch = SparseMultipathDDChannel(
            path_shifts=torch.zeros(2, 2, 2, dtype=torch.long),  # CPU
            path_gains=gains,
        )
        y = apply_sparse_multipath_dd_operator(x, ch)
        loss = y.abs().pow(2).sum()
        loss.backward()
        self.assertTrue(x.grad.is_cuda)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(gains.grad.is_cuda)
        self.assertTrue(torch.isfinite(gains.grad).all())


if __name__ == "__main__":
    unittest.main()
