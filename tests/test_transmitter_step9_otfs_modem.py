"""Tests for deterministic OTFS modem API (Step 9a).

Covers constructor validation, ISFFT/SFFT, modulate/demodulate roundtrip,
helper functions (normalized_mse, max_abs_error), and quality checks.
"""

import importlib.util
import os
import sys
import unittest
from pathlib import Path

import torch

# -- path-based import because the folder name contains hyphens ----------------
ROOT = Path(__file__).resolve().parents[1]
_MODEM_PATH = (
    ROOT / "Learnable_Mapping_Tokens-to-DD-Signals" / "otfs_modem.py"
)
_spec = importlib.util.spec_from_file_location(
    "otfs_modem", str(_MODEM_PATH),
)
_otfs = importlib.util.module_from_spec(_spec)
sys.modules["otfs_modem"] = _otfs
_spec.loader.exec_module(_otfs)

OTFSModem = _otfs.OTFSModem
normalized_mse = _otfs.normalized_mse
max_abs_error = _otfs.max_abs_error


# -- helpers -------------------------------------------------------------------

def _complex_rand(*shape, dtype=torch.complex64):
    return torch.randn(*shape, dtype=_real_dtype(dtype)) + \
           1j * torch.randn(*shape, dtype=_real_dtype(dtype))


def _complex_randn(*shape, dtype=torch.complex64):
    return torch.randn(*shape, dtype=_real_dtype(dtype)) + \
           1j * torch.randn(*shape, dtype=_real_dtype(dtype))


def _real_dtype(cdtype):
    return {torch.complex64: torch.float32, torch.complex128: torch.float64}[cdtype]


def _make_modem(M=8, N=6, cp_len=2, dtype=torch.complex64):
    return OTFSModem(dd_shape=(M, N), cp_len=cp_len, dtype=dtype)


# ============================================================================
# A. Constructor tests
# ============================================================================

class OTFSModemConstructorTests(unittest.TestCase):

    def test_valid_init_stores_attributes(self):
        modem = OTFSModem(dd_shape=(8, 6), cp_len=2, device="cpu",
                          dtype=torch.complex64)
        self.assertEqual(modem.M, 8)
        self.assertEqual(modem.N, 6)
        self.assertEqual(modem.cp_len, 2)
        self.assertEqual(modem.dd_shape, (8, 6))
        self.assertEqual(modem.device, torch.device("cpu"))
        self.assertEqual(modem.dtype, torch.complex64)

    def test_default_cp_len_is_zero(self):
        modem = OTFSModem(dd_shape=(4, 4))
        self.assertEqual(modem.cp_len, 0)

    def test_default_dtype_is_complex64(self):
        modem = OTFSModem(dd_shape=(4, 4))
        self.assertEqual(modem.dtype, torch.complex64)

    def test_default_device_is_cpu(self):
        modem = OTFSModem(dd_shape=(4, 4))
        self.assertEqual(modem.device, torch.device("cpu"))

    # -- invalid dd_shape ------------------------------------------------------

    def test_dd_shape_not_tuple_raises_value_error(self):
        with self.assertRaises(ValueError):
            OTFSModem(dd_shape=[8, 6])

    def test_dd_shape_wrong_length_raises_value_error(self):
        with self.assertRaises(ValueError):
            OTFSModem(dd_shape=(8,))

    def test_dd_shape_non_positive_raises_value_error(self):
        with self.assertRaises(ValueError):
            OTFSModem(dd_shape=(0, 6))
        with self.assertRaises(ValueError):
            OTFSModem(dd_shape=(8, -1))

    def test_dd_shape_bool_element_raises_type_error(self):
        with self.assertRaises(TypeError):
            OTFSModem(dd_shape=(True, 6))
        with self.assertRaises(TypeError):
            OTFSModem(dd_shape=(8, False))

    # -- invalid cp_len --------------------------------------------------------

    def test_negative_cp_len_raises_value_error(self):
        with self.assertRaises(ValueError):
            OTFSModem(dd_shape=(8, 6), cp_len=-1)

    def test_bool_cp_len_raises_type_error(self):
        with self.assertRaises(TypeError):
            OTFSModem(dd_shape=(8, 6), cp_len=True)
        with self.assertRaises(TypeError):
            OTFSModem(dd_shape=(8, 6), cp_len=False)

    def test_cp_len_greater_than_M_raises_value_error(self):
        with self.assertRaises(ValueError):
            OTFSModem(dd_shape=(4, 6), cp_len=5)

    def test_cp_len_equal_to_M_allowed(self):
        modem = OTFSModem(dd_shape=(4, 6), cp_len=4)
        self.assertEqual(modem.cp_len, 4)

    def test_cp_len_zero_allowed(self):
        modem = OTFSModem(dd_shape=(4, 6), cp_len=0)
        self.assertEqual(modem.cp_len, 0)

    # -- invalid dtype ---------------------------------------------------------

    def test_non_complex_dtype_raises_type_error(self):
        with self.assertRaises(TypeError):
            OTFSModem(dd_shape=(8, 6), dtype=torch.float32)
        with self.assertRaises(TypeError):
            OTFSModem(dd_shape=(8, 6), dtype=torch.float64)

    def test_complex128_dtype_accepted(self):
        modem = OTFSModem(dd_shape=(8, 6), dtype=torch.complex128)
        self.assertEqual(modem.dtype, torch.complex128)

    # -- device ----------------------------------------------------------------

    def test_device_string_accepted(self):
        modem = OTFSModem(dd_shape=(8, 6), device="cpu")
        self.assertEqual(modem.device, torch.device("cpu"))

    def test_device_none_defaults_to_cpu(self):
        modem = OTFSModem(dd_shape=(8, 6), device=None)
        self.assertEqual(modem.device, torch.device("cpu"))


# ============================================================================
# B. ISFFT / SFFT tests
# ============================================================================

class ISFFTSFFTTests(unittest.TestCase):

    def setUp(self):
        self.modem = _make_modem(M=8, N=6)

    # -- shape tests -----------------------------------------------------------

    def test_isfft_output_shape(self):
        x_dd = _complex_randn(3, 8, 6)
        x_tf = self.modem.isfft(x_dd)
        self.assertEqual(x_tf.shape, (3, 8, 6))

    def test_sfft_output_shape(self):
        x_tf = _complex_randn(3, 8, 6)
        x_dd = self.modem.sfft(x_tf)
        self.assertEqual(x_dd.shape, (3, 8, 6))

    # -- invertibility ---------------------------------------------------------

    def test_sfft_inverts_isfft_complex64(self):
        x_dd = _complex_randn(2, 8, 6, dtype=torch.complex64)
        x_tf = self.modem.isfft(x_dd)
        recovered = self.modem.sfft(x_tf)
        self.assertTrue(torch.allclose(recovered, x_dd, atol=1e-5, rtol=1e-5))

    def test_sfft_inverts_isfft_complex128(self):
        modem = _make_modem(M=8, N=6, dtype=torch.complex128)
        x_dd = _complex_randn(2, 8, 6, dtype=torch.complex128)
        x_tf = modem.isfft(x_dd)
        recovered = modem.sfft(x_tf)
        self.assertTrue(torch.allclose(recovered, x_dd, atol=1e-10, rtol=1e-10))

    def test_batch_size_1_works(self):
        x_dd = _complex_randn(1, 8, 6)
        x_tf = self.modem.isfft(x_dd)
        self.assertEqual(x_tf.shape, (1, 8, 6))
        recovered = self.modem.sfft(x_tf)
        self.assertTrue(torch.allclose(recovered, x_dd, atol=1e-5, rtol=1e-5))

    # -- dtype tests -----------------------------------------------------------

    def test_complex64_works(self):
        x_dd = _complex_randn(2, 8, 6, dtype=torch.complex64)
        x_tf = self.modem.isfft(x_dd)
        self.assertEqual(x_tf.dtype, torch.complex64)

    def test_complex128_works(self):
        modem = _make_modem(M=8, N=6, dtype=torch.complex128)
        x_dd = _complex_randn(2, 8, 6, dtype=torch.complex128)
        x_tf = modem.isfft(x_dd)
        self.assertEqual(x_tf.dtype, torch.complex128)

    # -- input validation ------------------------------------------------------

    def test_isfft_rejects_non_tensor(self):
        with self.assertRaises(TypeError):
            self.modem.isfft([1.0, 2.0])

    def test_isfft_rejects_non_complex(self):
        with self.assertRaises(TypeError):
            self.modem.isfft(torch.randn(3, 8, 6))

    def test_isfft_rejects_wrong_ndim(self):
        with self.assertRaises(ValueError):
            self.modem.isfft(_complex_randn(8, 6))

    def test_isfft_rejects_wrong_spatial_shape(self):
        with self.assertRaises(ValueError):
            self.modem.isfft(_complex_randn(3, 4, 6))
        with self.assertRaises(ValueError):
            self.modem.isfft(_complex_randn(3, 8, 5))

    def test_isfft_rejects_zero_batch(self):
        with self.assertRaises(ValueError):
            self.modem.isfft(_complex_randn(0, 8, 6))

    def test_sfft_rejects_non_tensor(self):
        with self.assertRaises(TypeError):
            self.modem.sfft([1.0, 2.0])

    def test_sfft_rejects_non_complex(self):
        with self.assertRaises(TypeError):
            self.modem.sfft(torch.randn(3, 8, 6))

    def test_sfft_rejects_wrong_ndim(self):
        with self.assertRaises(ValueError):
            self.modem.sfft(_complex_randn(8, 6))

    def test_sfft_rejects_wrong_spatial_shape(self):
        with self.assertRaises(ValueError):
            self.modem.sfft(_complex_randn(3, 4, 6))

    def test_sfft_rejects_zero_batch(self):
        with self.assertRaises(ValueError):
            self.modem.sfft(_complex_randn(0, 8, 6))


# ============================================================================
# C. Modulate / demodulate tests
# ============================================================================

class ModulateDemodulateTests(unittest.TestCase):

    # -- modulate output shape -------------------------------------------------

    def test_modulate_output_shape_cp_zero(self):
        modem = _make_modem(M=8, N=6, cp_len=0)
        x_dd = _complex_randn(3, 8, 6)
        ts = modem.modulate(x_dd)
        self.assertEqual(ts.shape, (3, 6 * 8))

    def test_modulate_output_shape_cp_positive(self):
        modem = _make_modem(M=8, N=6, cp_len=3)
        x_dd = _complex_randn(3, 8, 6)
        ts = modem.modulate(x_dd)
        self.assertEqual(ts.shape, (3, 6 * (8 + 3)))

    def test_modulate_batch_size_1(self):
        modem = _make_modem(M=8, N=6, cp_len=2)
        x_dd = _complex_randn(1, 8, 6)
        ts = modem.modulate(x_dd)
        self.assertEqual(ts.shape, (1, 6 * (8 + 2)))

    # -- roundtrip invertibility -----------------------------------------------

    def test_roundtrip_cp_zero_complex64(self):
        modem = _make_modem(M=8, N=6, cp_len=0, dtype=torch.complex64)
        x_dd = _complex_randn(3, 8, 6, dtype=torch.complex64)
        _, recovered = modem.roundtrip(x_dd)
        self.assertTrue(torch.allclose(recovered, x_dd, atol=1e-5, rtol=1e-5))

    def test_roundtrip_cp_positive_complex64(self):
        modem = _make_modem(M=8, N=6, cp_len=3, dtype=torch.complex64)
        x_dd = _complex_randn(3, 8, 6, dtype=torch.complex64)
        _, recovered = modem.roundtrip(x_dd)
        self.assertTrue(torch.allclose(recovered, x_dd, atol=1e-5, rtol=1e-5))

    def test_roundtrip_cp_zero_complex128(self):
        modem = _make_modem(M=8, N=6, cp_len=0, dtype=torch.complex128)
        x_dd = _complex_randn(2, 8, 6, dtype=torch.complex128)
        _, recovered = modem.roundtrip(x_dd)
        self.assertTrue(torch.allclose(recovered, x_dd, atol=1e-10, rtol=1e-10))

    def test_roundtrip_cp_positive_complex128(self):
        modem = _make_modem(M=8, N=6, cp_len=3, dtype=torch.complex128)
        x_dd = _complex_randn(2, 8, 6, dtype=torch.complex128)
        _, recovered = modem.roundtrip(x_dd)
        self.assertTrue(torch.allclose(recovered, x_dd, atol=1e-10, rtol=1e-10))

    def test_roundtrip_returns_correct_shapes(self):
        modem = _make_modem(M=8, N=6, cp_len=2)
        x_dd = _complex_randn(3, 8, 6)
        time_signal, recovered_dd = modem.roundtrip(x_dd)
        self.assertEqual(time_signal.shape, (3, 6 * (8 + 2)))
        self.assertEqual(recovered_dd.shape, (3, 8, 6))

    # -- demodulate with 1-D input ---------------------------------------------

    def test_demodulate_accepts_1d_and_returns_1_M_N(self):
        modem = _make_modem(M=8, N=6, cp_len=0)
        x_dd = _complex_randn(1, 8, 6)
        ts = modem.modulate(x_dd)
        ts_1d = ts.squeeze(0)
        self.assertEqual(ts_1d.ndim, 1)
        recovered = modem.demodulate(ts_1d)
        self.assertEqual(recovered.shape, (1, 8, 6))

    # -- demodulate input validation -------------------------------------------

    def test_demodulate_rejects_non_tensor(self):
        modem = _make_modem(M=8, N=6)
        with self.assertRaises(TypeError):
            modem.demodulate([1.0, 2.0])

    def test_demodulate_rejects_non_complex(self):
        modem = _make_modem(M=8, N=6, cp_len=0)
        with self.assertRaises(TypeError):
            modem.demodulate(torch.randn(2, 48))

    def test_demodulate_rejects_wrong_ndim(self):
        modem = _make_modem(M=8, N=6, cp_len=0)
        with self.assertRaises(ValueError):
            modem.demodulate(_complex_randn(2, 3, 48))

    def test_demodulate_rejects_wrong_flattened_length(self):
        modem = _make_modem(M=8, N=6, cp_len=0)
        with self.assertRaises(ValueError):
            modem.demodulate(_complex_randn(2, 47))

    def test_demodulate_rejects_zero_batch(self):
        modem = _make_modem(M=8, N=6, cp_len=0)
        with self.assertRaises(ValueError):
            modem.demodulate(_complex_randn(0, 48))


# ============================================================================
# D. Helper function tests
# ============================================================================

class NormalizedMSETests(unittest.TestCase):

    def test_exact_match_returns_zero(self):
        x = _complex_randn(4, 8)
        self.assertAlmostEqual(normalized_mse(x, x), 0.0, places=10)

    def test_positive_mse_for_different_tensors(self):
        a = _complex_randn(4, 8)
        b = a + 0.1 * _complex_randn(4, 8)
        val = normalized_mse(a, b)
        self.assertGreater(val, 0.0)

    def test_rejects_shape_mismatch(self):
        a = _complex_randn(4, 8)
        b = _complex_randn(4, 9)
        with self.assertRaises(ValueError):
            normalized_mse(a, b)

    def test_rejects_zero_power_reference(self):
        a = torch.zeros(4, 8, dtype=torch.complex64)
        b = _complex_randn(4, 8)
        with self.assertRaises(ValueError):
            normalized_mse(a, b)

    def test_rejects_non_tensor(self):
        with self.assertRaises(TypeError):
            normalized_mse([1.0], _complex_randn(4))


class MaxAbsErrorTests(unittest.TestCase):

    def test_exact_match_returns_zero(self):
        x = _complex_randn(4, 8)
        self.assertAlmostEqual(max_abs_error(x, x), 0.0, places=10)

    def test_matches_manual_value(self):
        a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.complex64)
        b = torch.tensor([1.5, 2.5, 2.0], dtype=torch.complex64)
        val = max_abs_error(a, b)
        expected = max(abs(1.0 - 1.5), abs(2.0 - 2.5), abs(3.0 - 2.0))
        self.assertAlmostEqual(val, expected, places=6)

    def test_rejects_shape_mismatch(self):
        a = _complex_randn(4, 8)
        b = _complex_randn(4, 9)
        with self.assertRaises(ValueError):
            max_abs_error(a, b)

    def test_rejects_empty_tensor(self):
        a = torch.empty(0, dtype=torch.complex64)
        b = torch.empty(0, dtype=torch.complex64)
        with self.assertRaises(ValueError):
            max_abs_error(a, b)

    def test_rejects_non_tensor(self):
        with self.assertRaises(TypeError):
            max_abs_error([1.0], _complex_randn(4))


# ============================================================================
# E. Quality tests
# ============================================================================

class QualityTests(unittest.TestCase):

    def test_public_api_docstrings_exist(self):
        """All public API items must have non-empty docstrings."""
        for name in ("normalized_mse", "max_abs_error"):
            self.assertIsNotNone(getattr(_otfs, name).__doc__)
            self.assertTrue(len(getattr(_otfs, name).__doc__.strip()) > 0)

        cls = _otfs.OTFSModem
        self.assertIsNotNone(cls.__doc__)
        self.assertTrue(len(cls.__doc__.strip()) > 0)

        for method_name in ("isfft", "sfft", "modulate", "demodulate", "roundtrip"):
            method = getattr(cls, method_name)
            self.assertIsNotNone(method.__doc__,
                                 f"{method_name} missing docstring")
            self.assertTrue(len(method.__doc__.strip()) > 0,
                            f"{method_name} docstring is empty")

    def test_otfs_modem_py_is_ascii_only(self):
        content = _MODEM_PATH.read_text(encoding="utf-8")
        for i, ch in enumerate(content):
            self.assertTrue(
                ord(ch) < 128,
                f"Non-ASCII character U+{ord(ch):04X} at offset {i} in "
                f"{_MODEM_PATH}",
            )

    def test_this_file_is_ascii_only(self):
        content = Path(__file__).read_text(encoding="utf-8")
        for i, ch in enumerate(content):
            self.assertTrue(
                ord(ch) < 128,
                f"Non-ASCII character U+{ord(ch):04X} at offset {i} in "
                f"{Path(__file__)}",
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_roundtrip(self):
        modem = OTFSModem(dd_shape=(8, 6), cp_len=2, device="cuda",
                          dtype=torch.complex64)
        x_dd = _complex_randn(3, 8, 6, dtype=torch.complex64).to("cuda")
        _, recovered = modem.roundtrip(x_dd)
        self.assertTrue(torch.allclose(recovered.cpu(), x_dd.cpu(),
                                       atol=1e-5, rtol=1e-5))

    def test_roundtrip_preserves_determinism(self):
        modem = _make_modem(M=8, N=6, cp_len=2)
        x_dd = _complex_randn(3, 8, 6)
        _, r1 = modem.roundtrip(x_dd)
        _, r2 = modem.roundtrip(x_dd)
        self.assertTrue(torch.equal(r1, r2))


if __name__ == "__main__":
    unittest.main()
