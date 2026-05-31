"""Tests for time-domain waveform metrics (Step 11).

Covers time_signal_power, peak_to_average_power_ratio,
summarize_time_transmitter_metrics, and quality/integration checks.
"""

import importlib.util
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
TRANSMITTER_ROOT = ROOT / "Learnable_Mapping_Tokens-to-DD-Signals"
sys.path.insert(0, str(TRANSMITTER_ROOT))

from transmitter.diagnostics import (
    peak_to_average_power_ratio,
    summarize_time_transmitter_metrics,
    time_signal_power,
)

# Path-based import for hyphenated directory.
_OTFS_PATH = TRANSMITTER_ROOT / "otfs_modem.py"
_spec = importlib.util.spec_from_file_location("otfs_modem", str(_OTFS_PATH))
_otfs_mod = importlib.util.module_from_spec(_spec)
sys.modules["otfs_modem"] = _otfs_mod
_spec.loader.exec_module(_otfs_mod)
OTFSModem = _otfs_mod.OTFSModem


# -- helpers -------------------------------------------------------------------

def _complex_randn(*shape, dtype=torch.complex64):
    rd = {torch.complex64: torch.float32, torch.complex128: torch.float64}[dtype]
    return torch.randn(*shape, dtype=rd) + 1j * torch.randn(*shape, dtype=rd)


def _real_dtype(cdtype):
    return {torch.complex64: torch.float32, torch.complex128: torch.float64}[cdtype]


# ============================================================================
# A. time_signal_power tests
# ============================================================================

class TimeSignalPowerTests(unittest.TestCase):

    def test_valid_returns_shape_B(self):
        ts = _complex_randn(3, 80)
        p = time_signal_power(ts)
        self.assertEqual(p.shape, (3,))

    def test_matches_manual_mean(self):
        ts = _complex_randn(2, 50)
        p = time_signal_power(ts)
        expected = ts.abs().pow(2).mean(dim=1).to(dtype=torch.float32)
        self.assertTrue(torch.allclose(p, expected))

    def test_constant_envelope_waveform(self):
        ts = torch.ones(2, 100, dtype=torch.complex64)
        p = time_signal_power(ts)
        self.assertTrue(torch.allclose(p, torch.ones(2, dtype=torch.float32)))

    def test_complex64_output_float32(self):
        ts = _complex_randn(2, 80, dtype=torch.complex64)
        p = time_signal_power(ts)
        self.assertEqual(p.dtype, torch.float32)

    def test_complex128_output_float64(self):
        ts = _complex_randn(2, 80, dtype=torch.complex128)
        p = time_signal_power(ts)
        self.assertEqual(p.dtype, torch.float64)

    def test_output_device_preserved(self):
        ts = _complex_randn(2, 80)
        p = time_signal_power(ts)
        self.assertEqual(p.device, ts.device)

    def test_rejects_non_tensor(self):
        with self.assertRaises(TypeError):
            time_signal_power([1.0])

    def test_rejects_non_complex(self):
        with self.assertRaises(TypeError):
            time_signal_power(torch.randn(2, 80))

    def test_rejects_wrong_ndim(self):
        with self.assertRaises(ValueError):
            time_signal_power(_complex_randn(2, 8, 10))

    def test_rejects_B_zero(self):
        with self.assertRaises(ValueError):
            time_signal_power(_complex_randn(0, 80))

    def test_rejects_T_zero(self):
        with self.assertRaises(ValueError):
            time_signal_power(_complex_randn(2, 0))

    def test_rejects_nan_real(self):
        ts = _complex_randn(2, 80)
        ts[0, 0] = complex(float("nan"), 0.0)
        with self.assertRaises(ValueError):
            time_signal_power(ts)

    def test_rejects_inf_real(self):
        ts = _complex_randn(2, 80)
        ts[0, 0] = complex(float("inf"), 0.0)
        with self.assertRaises(ValueError):
            time_signal_power(ts)

    def test_rejects_nan_imag(self):
        ts = _complex_randn(2, 80)
        ts[0, 0] = complex(0.0, float("nan"))
        with self.assertRaises(ValueError):
            time_signal_power(ts)

    def test_rejects_inf_imag(self):
        ts = _complex_randn(2, 80)
        ts[0, 0] = complex(0.0, float("inf"))
        with self.assertRaises(ValueError):
            time_signal_power(ts)


# ============================================================================
# B. peak_to_average_power_ratio tests
# ============================================================================

class PAPRTests(unittest.TestCase):

    def test_valid_returns_shape_B(self):
        ts = _complex_randn(3, 80)
        papr = peak_to_average_power_ratio(ts)
        self.assertEqual(papr.shape, (3,))

    def test_constant_envelope_papr_equals_one(self):
        ts = torch.ones(2, 100, dtype=torch.complex64)
        papr = peak_to_average_power_ratio(ts)
        self.assertTrue(torch.allclose(papr, torch.ones(2, dtype=torch.float32)))

    def test_known_waveform_manual_check(self):
        ts = torch.tensor([[1.0 + 0j, 0.0 + 0j]], dtype=torch.complex64)
        papr = peak_to_average_power_ratio(ts)
        expected = 2.0
        self.assertAlmostEqual(papr[0].item(), expected, places=5)

    def test_batch_wise_independent(self):
        ts = torch.tensor([
            [1.0 + 0j, 0.0 + 0j],
            [1.0 + 0j, 1.0 + 0j],
        ], dtype=torch.complex64)
        papr = peak_to_average_power_ratio(ts)
        self.assertAlmostEqual(papr[0].item(), 2.0, places=5)
        self.assertAlmostEqual(papr[1].item(), 1.0, places=5)

    def test_complex64_output_float32(self):
        ts = _complex_randn(2, 80, dtype=torch.complex64)
        papr = peak_to_average_power_ratio(ts)
        self.assertEqual(papr.dtype, torch.float32)

    def test_complex128_output_float64(self):
        ts = _complex_randn(2, 80, dtype=torch.complex128)
        papr = peak_to_average_power_ratio(ts)
        self.assertEqual(papr.dtype, torch.float64)

    def test_output_device_preserved(self):
        ts = _complex_randn(2, 80)
        papr = peak_to_average_power_ratio(ts)
        self.assertEqual(papr.device, ts.device)

    def test_rejects_zero_power_waveform(self):
        ts = torch.zeros(2, 80, dtype=torch.complex64)
        with self.assertRaises(ValueError):
            peak_to_average_power_ratio(ts)

    def test_rejects_one_zero_power_batch_element(self):
        ts = _complex_randn(2, 80)
        ts[0, :] = 0.0
        with self.assertRaises(ValueError):
            peak_to_average_power_ratio(ts)

    def test_rejects_non_tensor(self):
        with self.assertRaises(TypeError):
            peak_to_average_power_ratio([1.0])

    def test_rejects_non_complex(self):
        with self.assertRaises(TypeError):
            peak_to_average_power_ratio(torch.randn(2, 80))

    def test_reuses_time_signal_power(self):
        from unittest.mock import patch
        ts = _complex_randn(3, 80)
        mock_power = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float32)
        with patch(
            "transmitter.metrics.time_signal_power",
            return_value=mock_power,
        ) as mock_tsp:
            papr = peak_to_average_power_ratio(ts)
            mock_tsp.assert_called_once()
            # Verify the argument is the same time_signal object.
            self.assertIs(mock_tsp.call_args[0][0], ts)
        self.assertEqual(papr.shape, (3,))
        self.assertTrue((papr > 0).all())


# ============================================================================
# C. summarize_time_transmitter_metrics tests
# ============================================================================

class SummarizeTimeMetricsTests(unittest.TestCase):

    def test_returns_exact_keys(self):
        ts = _complex_randn(3, 80)
        result = summarize_time_transmitter_metrics(ts)
        self.assertEqual(set(result.keys()), {"time_signal_power", "papr"})

    def test_both_values_have_shape_B(self):
        ts = _complex_randn(3, 80)
        result = summarize_time_transmitter_metrics(ts)
        self.assertEqual(result["time_signal_power"].shape, (3,))
        self.assertEqual(result["papr"].shape, (3,))

    def test_time_signal_power_matches_individual(self):
        ts = _complex_randn(3, 80)
        result = summarize_time_transmitter_metrics(ts)
        direct = time_signal_power(ts)
        self.assertTrue(torch.equal(result["time_signal_power"], direct))

    def test_papr_matches_individual(self):
        ts = _complex_randn(3, 80)
        result = summarize_time_transmitter_metrics(ts)
        direct = peak_to_average_power_ratio(ts)
        self.assertTrue(torch.equal(result["papr"], direct))

    def test_does_not_include_dd_metrics_keys(self):
        ts = _complex_randn(3, 80)
        result = summarize_time_transmitter_metrics(ts)
        for dd_key in ("data_codeword_power", "pilot_power",
                        "total_dd_frame_power"):
            self.assertNotIn(dd_key, result)

    def test_does_not_include_extra_keys(self):
        ts = _complex_randn(3, 80)
        result = summarize_time_transmitter_metrics(ts)
        self.assertEqual(len(result), 2)
        self.assertNotIn("time_signal_power_mean", result)
        self.assertNotIn("papr_mean", result)


# ============================================================================
# D. Real modem linkage tests
# ============================================================================

class RealModemLinkageTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.modem_cp0 = OTFSModem(dd_shape=(8, 6), cp_len=0)
        cls.modem_cp2 = OTFSModem(dd_shape=(8, 6), cp_len=2)

    def _x_dd(self, B=3):
        return _complex_randn(B, 8, 6)

    def test_time_signal_power_on_real_modem_output(self):
        x_dd = self._x_dd()
        ts = self.modem_cp0.modulate(x_dd)
        p = time_signal_power(ts)
        self.assertEqual(p.shape, (3,))
        self.assertTrue((p > 0).all())

    def test_papr_on_real_modem_output(self):
        x_dd = self._x_dd()
        ts = self.modem_cp0.modulate(x_dd)
        papr = peak_to_average_power_ratio(ts)
        self.assertEqual(papr.shape, (3,))
        self.assertTrue((papr >= 1.0).all())

    def test_summarize_on_real_modem_output(self):
        x_dd = self._x_dd()
        ts = self.modem_cp0.modulate(x_dd)
        result = summarize_time_transmitter_metrics(ts)
        self.assertEqual(result["time_signal_power"].shape, (3,))
        self.assertEqual(result["papr"].shape, (3,))

    def test_cp_positive_path_works(self):
        x_dd = self._x_dd()
        ts = self.modem_cp2.modulate(x_dd)
        expected_T = 6 * (8 + 2)
        self.assertEqual(ts.shape[1], expected_T)
        p = time_signal_power(ts)
        self.assertEqual(p.shape, (3,))
        papr = peak_to_average_power_ratio(ts)
        self.assertEqual(papr.shape, (3,))


# ============================================================================
# E. Quality tests
# ============================================================================

class QualityTests(unittest.TestCase):

    def test_metrics_py_is_ascii_only(self):
        path = TRANSMITTER_ROOT / "transmitter" / "metrics.py"
        content = path.read_text(encoding="utf-8")
        for i, ch in enumerate(content):
            self.assertTrue(ord(ch) < 128,
                            f"Non-ASCII U+{ord(ch):04X} at offset {i}")

    def test_time_signal_power_docstring_exists(self):
        doc = time_signal_power.__doc__
        self.assertIsNotNone(doc)
        self.assertTrue(len(doc.strip()) > 0)

    def test_papr_docstring_exists(self):
        doc = peak_to_average_power_ratio.__doc__
        self.assertIsNotNone(doc)
        self.assertTrue(len(doc.strip()) > 0)

    def test_summarize_docstring_exists(self):
        doc = summarize_time_transmitter_metrics.__doc__
        self.assertIsNotNone(doc)
        self.assertTrue(len(doc.strip()) > 0)

    def test_time_signal_power_docstring_mentions_B_T(self):
        doc = time_signal_power.__doc__
        self.assertIn("[B, T]", doc)
        self.assertIn("[B]", doc)

    def test_papr_docstring_mentions_B_T(self):
        doc = peak_to_average_power_ratio.__doc__
        self.assertIn("[B, T]", doc)
        self.assertIn("[B]", doc)

    def test_summarize_docstring_mentions_B_T(self):
        doc = summarize_time_transmitter_metrics.__doc__
        self.assertIn("[B, T]", doc)
        self.assertIn("[B]", doc)

    def test_exports_importable(self):
        from transmitter.diagnostics import (
            peak_to_average_power_ratio as _papr,
            summarize_time_transmitter_metrics as _stm,
            time_signal_power as _tsp,
        )
        self.assertTrue(callable(_tsp))
        self.assertTrue(callable(_papr))
        self.assertTrue(callable(_stm))


# ============================================================================
# F. CUDA tests
# ============================================================================

@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class CUDATests(unittest.TestCase):

    def test_time_signal_power_cuda(self):
        ts = _complex_randn(3, 80).to("cuda")
        p = time_signal_power(ts)
        self.assertTrue(p.is_cuda)
        self.assertEqual(p.device.type, "cuda")

    def test_papr_cuda(self):
        ts = _complex_randn(3, 80).to("cuda")
        papr = peak_to_average_power_ratio(ts)
        self.assertTrue(papr.is_cuda)

    def test_real_modem_cuda_path(self):
        modem = OTFSModem(dd_shape=(8, 6), cp_len=2, device="cuda",
                          dtype=torch.complex64)
        x_dd = _complex_randn(3, 8, 6, dtype=torch.complex64).to("cuda")
        ts = modem.modulate(x_dd)
        p = time_signal_power(ts)
        papr = peak_to_average_power_ratio(ts)
        self.assertTrue(p.is_cuda)
        self.assertTrue(papr.is_cuda)


if __name__ == "__main__":
    unittest.main()
