"""Tests for receiver-aligned integer DD circular shift primitives (Step 12).

Covers dd_circular_shift, dd_circular_shift_bank, receiver contract,
mask semantics, autograd, and quality checks.
"""

import importlib.util
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
TRANSMITTER_ROOT = ROOT / "Learnable_Mapping_Tokens-to-DD-Signals"
sys.path.insert(0, str(TRANSMITTER_ROOT))

from transmitter.dd_shifts import dd_circular_shift, dd_circular_shift_bank


# -- helpers -------------------------------------------------------------------

def _complex_randn(*shape, dtype=torch.complex64):
    rd = {torch.complex64: torch.float32, torch.complex128: torch.float64}[dtype]
    return torch.randn(*shape, dtype=rd) + 1j * torch.randn(*shape, dtype=rd)


def _import_receiver_dd_ops():
    """Import receiver.dd_ops by adding the receiver package root to sys.path.

    The receiver package lives under a hyphenated directory name, so
    direct package-relative import is not possible.  We add the parent
    of the receiver package to sys.path and use importlib.import_module.
    """
    import importlib
    receiver_root = str(
        ROOT / "Learnable_Receiver_DD-Signals-to-Tokens"
    )
    sys.path.insert(0, receiver_root)
    try:
        mod = importlib.import_module("receiver.dd_ops")
    finally:
        if receiver_root in sys.path:
            sys.path.remove(receiver_root)
    return mod


# ============================================================================
# A. dd_circular_shift legal paths
# ============================================================================

class DDShiftLegalTests(unittest.TestCase):

    def setUp(self):
        self.x = _complex_randn(2, 8, 10)

    def test_zero_shift_returns_equal(self):
        y = dd_circular_shift(self.x, 0, 0)
        self.assertTrue(torch.equal(y, self.x))

    def test_positive_delay_matches_manual_roll(self):
        y = dd_circular_shift(self.x, 3, 0)
        expected = torch.roll(self.x, shifts=(3, 0), dims=(-2, -1))
        self.assertTrue(torch.equal(y, expected))

    def test_negative_delay_matches_manual_roll(self):
        y = dd_circular_shift(self.x, -2, 0)
        expected = torch.roll(self.x, shifts=(-2, 0), dims=(-2, -1))
        self.assertTrue(torch.equal(y, expected))

    def test_positive_doppler_matches_manual_roll(self):
        y = dd_circular_shift(self.x, 0, 5)
        expected = torch.roll(self.x, shifts=(0, 5), dims=(-2, -1))
        self.assertTrue(torch.equal(y, expected))

    def test_negative_doppler_matches_manual_roll(self):
        y = dd_circular_shift(self.x, 0, -3)
        expected = torch.roll(self.x, shifts=(0, -3), dims=(-2, -1))
        self.assertTrue(torch.equal(y, expected))

    def test_combined_shift_matches_manual_roll(self):
        y = dd_circular_shift(self.x, 3, -5)
        expected = torch.roll(self.x, shifts=(3, -5), dims=(-2, -1))
        self.assertTrue(torch.equal(y, expected))

    def test_large_shift_circular_modulo(self):
        y = dd_circular_shift(self.x, 3 + 8 * 100, -5 + 10 * 50)
        expected = torch.roll(self.x, shifts=(3, -5), dims=(-2, -1))
        self.assertTrue(torch.equal(y, expected))

    def test_inverse_property(self):
        for d in (0, 3, -2, 5):
            for v in (0, 4, -3, 7):
                with self.subTest(delay=d, doppler=v):
                    shifted = dd_circular_shift(self.x, d, v)
                    recovered = dd_circular_shift(shifted, -d, -v)
                    self.assertTrue(torch.equal(recovered, self.x))

    def test_composition_property(self):
        d1, v1 = 3, -2
        d2, v2 = -1, 5
        y1 = dd_circular_shift(
            dd_circular_shift(self.x, d1, v1), d2, v2,
        )
        y2 = dd_circular_shift(self.x, d1 + d2, v1 + v2)
        self.assertTrue(torch.equal(y1, y2))

    def test_norm_preserved(self):
        y = dd_circular_shift(self.x, 3, 4)
        self.assertTrue(torch.allclose(
            self.x.abs().pow(2).sum(dim=(-2, -1)),
            y.abs().pow(2).sum(dim=(-2, -1)),
        ))

    def test_input_not_modified_in_place(self):
        x_copy = self.x.clone()
        dd_circular_shift(self.x, 3, 4)
        self.assertTrue(torch.equal(self.x, x_copy))

    def test_dtype_device_preserved(self):
        y = dd_circular_shift(self.x, 2, 3)
        self.assertEqual(y.dtype, self.x.dtype)
        self.assertEqual(y.device, self.x.device)

    def test_codeword_book_V_M_N_accepted(self):
        cw = _complex_randn(6, 8, 10)
        y = dd_circular_shift(cw, 2, 1)
        self.assertEqual(y.shape, (6, 8, 10))


# ============================================================================
# B. dd_circular_shift reject paths
# ============================================================================

class DDShiftRejectTests(unittest.TestCase):

    def setUp(self):
        self.x = _complex_randn(2, 8, 10)

    def test_non_tensor_raises_type_error(self):
        with self.assertRaises(TypeError):
            dd_circular_shift([1.0], 0, 0)

    def test_non_complex_raises_type_error(self):
        with self.assertRaises(TypeError):
            dd_circular_shift(torch.randn(2, 8, 10), 0, 0)

    def test_wrong_ndim_raises_value_error(self):
        with self.assertRaises(ValueError):
            dd_circular_shift(_complex_randn(8, 10), 0, 0)

    def test_B_zero_raises_value_error(self):
        with self.assertRaises(ValueError):
            dd_circular_shift(_complex_randn(0, 8, 10), 0, 0)

    def test_M_zero_raises_value_error(self):
        with self.assertRaises(ValueError):
            dd_circular_shift(_complex_randn(2, 0, 10), 0, 0)

    def test_N_zero_raises_value_error(self):
        with self.assertRaises(ValueError):
            dd_circular_shift(_complex_randn(2, 8, 0), 0, 0)

    def test_delay_shift_bool_raises_type_error(self):
        with self.assertRaises(TypeError):
            dd_circular_shift(self.x, True, 0)
        with self.assertRaises(TypeError):
            dd_circular_shift(self.x, False, 0)

    def test_delay_shift_float_raises_type_error(self):
        with self.assertRaises(TypeError):
            dd_circular_shift(self.x, 1.0, 0)

    def test_doppler_shift_bool_raises_type_error(self):
        with self.assertRaises(TypeError):
            dd_circular_shift(self.x, 0, True)
        with self.assertRaises(TypeError):
            dd_circular_shift(self.x, 0, False)

    def test_doppler_shift_float_raises_type_error(self):
        with self.assertRaises(TypeError):
            dd_circular_shift(self.x, 0, 1.0)


# ============================================================================
# C. dd_circular_shift_bank legal paths
# ============================================================================

class DDShiftBankLegalTests(unittest.TestCase):

    def setUp(self):
        self.x = _complex_randn(2, 8, 10)

    def test_output_shape_B_S_M_N(self):
        shifts = torch.tensor([[0, 0], [1, 2], [-1, 3]], dtype=torch.long)
        y = dd_circular_shift_bank(self.x, shifts)
        self.assertEqual(y.shape, (2, 3, 8, 10))

    def test_each_output_matches_single_shift(self):
        shifts = torch.tensor([[0, 0], [3, -2], [-1, 5]], dtype=torch.long)
        y_bank = dd_circular_shift_bank(self.x, shifts)
        for s in range(shifts.shape[0]):
            d = int(shifts[s, 0].item())
            v = int(shifts[s, 1].item())
            y_single = dd_circular_shift(self.x, d, v)
            self.assertTrue(torch.equal(y_bank[:, s], y_single),
                            f"mismatch at s={s}, shift=({d},{v})")

    def test_shift_order_preserved(self):
        s1 = torch.tensor([[1, 0], [0, 0], [2, 0]], dtype=torch.long)
        y1 = dd_circular_shift_bank(self.x, s1)
        s2 = torch.tensor([[0, 0], [1, 0], [2, 0]], dtype=torch.long)
        y2 = dd_circular_shift_bank(self.x, s2)
        self.assertFalse(torch.equal(y1, y2))

    def test_duplicate_shifts_kept(self):
        shifts = torch.tensor([[1, 2], [1, 2]], dtype=torch.long)
        y = dd_circular_shift_bank(self.x, shifts)
        self.assertTrue(torch.equal(y[:, 0], y[:, 1]))

    def test_zero_shift_included(self):
        shifts = torch.tensor([[0, 0], [3, 0]], dtype=torch.long)
        y = dd_circular_shift_bank(self.x, shifts)
        self.assertTrue(torch.equal(y[:, 0], self.x))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cpu_shifts_gpu_x_dd_works(self):
        x_gpu = _complex_randn(2, 8, 10).to("cuda")
        shifts_cpu = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)
        y = dd_circular_shift_bank(x_gpu, shifts_cpu)
        self.assertTrue(y.is_cuda)

    def test_dtype_device_preserved(self):
        shifts = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)
        y = dd_circular_shift_bank(self.x, shifts)
        self.assertEqual(y.dtype, self.x.dtype)
        self.assertEqual(y.device, self.x.device)


# ============================================================================
# D. dd_circular_shift_bank reject paths
# ============================================================================

class DDShiftBankRejectTests(unittest.TestCase):

    def setUp(self):
        self.x = _complex_randn(2, 8, 10)

    def test_shifts_non_tensor_raises_type_error(self):
        with self.assertRaises(TypeError):
            dd_circular_shift_bank(self.x, [[0, 0]])

    def test_shifts_wrong_dtype_raises_type_error(self):
        s = torch.tensor([[0, 0], [1, 2]], dtype=torch.int32)
        with self.assertRaises(TypeError):
            dd_circular_shift_bank(self.x, s)

    def test_shifts_wrong_ndim_raises_value_error(self):
        s = torch.tensor([0, 0], dtype=torch.long)
        with self.assertRaises(ValueError):
            dd_circular_shift_bank(self.x, s)

    def test_shifts_last_dim_not_2_raises_value_error(self):
        s = torch.tensor([[0, 0, 0], [1, 2, 3]], dtype=torch.long)
        with self.assertRaises(ValueError):
            dd_circular_shift_bank(self.x, s)

    def test_S_zero_raises_value_error(self):
        s = torch.empty(0, 2, dtype=torch.long)
        with self.assertRaises(ValueError):
            dd_circular_shift_bank(self.x, s)


# ============================================================================
# E. Receiver contract
# ============================================================================

class ReceiverContractTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.dd_ops = _import_receiver_dd_ops()

    def _make_single_path(self, x_dd, delay, doppler):
        """Build path_indices [B, 1, 2] and path_gains [B, 1] = 1+0j."""
        B = x_dd.shape[0]
        path_indices = torch.full(
            (B, 1, 2), fill_value=0, dtype=torch.long,
            device=x_dd.device,
        )
        path_indices[:, 0, 0] = delay
        path_indices[:, 0, 1] = doppler
        path_gains = torch.ones(
            B, 1, dtype=x_dd.dtype, device=x_dd.device,
        )
        return path_indices, path_gains

    def _check_equivalence(self, x_dd, delay, doppler):
        path_indices, path_gains = self._make_single_path(
            x_dd, delay, doppler,
        )
        tx_result = dd_circular_shift(x_dd, delay, doppler)
        rx_result = self.dd_ops.dd_circular_convolve_sparse(
            x_dd, path_indices=path_indices, path_gains=path_gains,
        )
        self.assertTrue(torch.allclose(tx_result, rx_result),
                        f"mismatch for shift ({delay}, {doppler})")

    def test_zero_shift(self):
        x = _complex_randn(2, 8, 10)
        self._check_equivalence(x, 0, 0)

    def test_positive_delay(self):
        x = _complex_randn(2, 8, 10)
        self._check_equivalence(x, 3, 0)

    def test_negative_delay(self):
        x = _complex_randn(2, 8, 10)
        self._check_equivalence(x, -2, 0)

    def test_positive_doppler(self):
        x = _complex_randn(2, 8, 10)
        self._check_equivalence(x, 0, 5)

    def test_negative_doppler(self):
        x = _complex_randn(2, 8, 10)
        self._check_equivalence(x, 0, -3)

    def test_combined_delay_doppler(self):
        x = _complex_randn(2, 8, 10)
        self._check_equivalence(x, 3, -4)


# ============================================================================
# F. Mask semantics
# ============================================================================

class MaskSemanticsTests(unittest.TestCase):

    def test_no_implicit_data_mask_projection(self):
        """Shifted non-zero energy may enter another region."""
        x = torch.zeros(1, 8, 10, dtype=torch.complex64)
        x[0, 0, 0] = 1.0 + 2j
        y = dd_circular_shift(x, 1, 0)
        # Energy moved from (0,0) to (1,0); no mask suppressed it.
        self.assertEqual(y[0, 1, 0], x[0, 0, 0])
        self.assertEqual(y[0, 0, 0], 0.0 + 0j)

    def test_shift_preserves_total_energy(self):
        x = torch.zeros(1, 8, 10, dtype=torch.complex64)
        x[0, 2, 3] = 3.0 + 4j  # |amplitude|^2 = 25
        y = dd_circular_shift(x, 5, -2)
        self.assertAlmostEqual(
            y.abs().pow(2).sum().item(),
            x.abs().pow(2).sum().item(),
            places=5,
        )


# ============================================================================
# G. Autograd
# ============================================================================

class AutogradTests(unittest.TestCase):

    def test_shift_bank_preserves_grad(self):
        x = _complex_randn(2, 8, 10)
        x.requires_grad_(True)
        shifts = torch.tensor([[0, 0], [1, 2], [3, -1]], dtype=torch.long)
        y = dd_circular_shift_bank(x, shifts)
        loss = y.abs().pow(2).sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_single_shift_preserves_grad(self):
        x = _complex_randn(2, 8, 10)
        x.requires_grad_(True)
        y = dd_circular_shift(x, 3, -2)
        loss = y.abs().pow(2).sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())


# ============================================================================
# H. Quality tests
# ============================================================================

class QualityTests(unittest.TestCase):

    def test_dd_shifts_py_is_ascii_only(self):
        path = TRANSMITTER_ROOT / "transmitter" / "dd_shifts.py"
        content = path.read_text(encoding="utf-8")
        for i, ch in enumerate(content):
            self.assertTrue(ord(ch) < 128,
                            f"Non-ASCII U+{ord(ch):04X} at offset {i}")

    def test_shift_docstring_exists(self):
        doc = dd_circular_shift.__doc__
        self.assertIsNotNone(doc)
        self.assertTrue(len(doc.strip()) > 0)

    def test_shift_bank_docstring_exists(self):
        doc = dd_circular_shift_bank.__doc__
        self.assertIsNotNone(doc)
        self.assertTrue(len(doc.strip()) > 0)

    def test_shift_docstring_mentions_shapes(self):
        doc = dd_circular_shift.__doc__
        self.assertIn("[B, M, N]", doc)
        self.assertIn("dim -2", doc)
        self.assertIn("dim -1", doc)

    def test_shift_bank_docstring_mentions_shapes(self):
        doc = dd_circular_shift_bank.__doc__
        self.assertIn("[B, S, M, N]", doc)
        self.assertIn("[S, 2]", doc)

    def test_exports_importable(self):
        from transmitter.dd_shifts import dd_circular_shift as _dcs
        from transmitter.dd_shifts import dd_circular_shift_bank as _dcsb
        self.assertTrue(callable(_dcs))
        self.assertTrue(callable(_dcsb))


if __name__ == "__main__":
    unittest.main()
