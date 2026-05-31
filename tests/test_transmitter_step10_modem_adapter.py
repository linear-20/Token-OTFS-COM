"""Tests for OTFS modem adapter and LearnableTokenDDTransmitter.forward wiring.

Covers modulate_otfs_dd_frame validation, modem contract enforcement,
model integration, autograd preservation, CUDA, and quality checks.
"""

import importlib.util
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
TRANSMITTER_ROOT = ROOT / "Learnable_Mapping_Tokens-to-DD-Signals"
sys.path.insert(0, str(TRANSMITTER_ROOT))

# Path-based import for hyphenated directory name.
_OTFS_PATH = TRANSMITTER_ROOT / "otfs_modem.py"
_spec = importlib.util.spec_from_file_location("otfs_modem", str(_OTFS_PATH))
_otfs_mod = importlib.util.module_from_spec(_spec)
sys.modules["otfs_modem"] = _otfs_mod
_spec.loader.exec_module(_otfs_mod)
OTFSModem = _otfs_mod.OTFSModem

from transmitter import (
    LearnableTokenDDTransmitter,
    TransmitterConfig,
    modulate_otfs_dd_frame,
)


# -- helpers -------------------------------------------------------------------

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


def _complex_randn(*shape, dtype=torch.complex64):
    rd = {torch.complex64: torch.float32, torch.complex128: torch.float64}[dtype]
    return torch.randn(*shape, dtype=rd) + 1j * torch.randn(*shape, dtype=rd)


def _token_indices(batch, V=6):
    return torch.arange(batch, dtype=torch.long) % V


# -- dummy / spy modems --------------------------------------------------------

class DummyModem:
    """Minimal modem for happy-path adapter tests."""

    def __init__(self, M, N, cp_len=0):
        self.dd_shape = (M, N)
        self.cp_len = cp_len

    def modulate(self, x_dd):
        B, M, N = x_dd.shape
        T = N * (M + self.cp_len)
        return _complex_randn(B, T, dtype=x_dd.dtype).to(x_dd.device)


class SpyModem:
    """Records whether received x_dd is the same Python object."""

    def __init__(self, M, N, cp_len=0):
        self.dd_shape = (M, N)
        self.cp_len = cp_len
        self.received_x_dd = None

    def modulate(self, x_dd):
        self.received_x_dd = x_dd
        B, M, N = x_dd.shape
        T = N * (M + self.cp_len)
        return _complex_randn(B, T, dtype=x_dd.dtype).to(x_dd.device)


class MissingModulateModem:
    def __init__(self):
        self.dd_shape = (8, 10)
        self.cp_len = 0


class NonCallableModulateModem:
    def __init__(self):
        self.dd_shape = (8, 10)
        self.cp_len = 0
        self.modulate = "not_callable"


class MissingDDShapeModem:
    def __init__(self):
        self.cp_len = 0

    def modulate(self, x_dd):
        B, _, _ = x_dd.shape
        return _complex_randn(B, 80, dtype=x_dd.dtype)


class BadOutputModem:
    """Returns non-tensor, non-complex, wrong shape, wrong device, etc."""

    def __init__(self, M, N, cp_len=0, *, output_factory):
        self.dd_shape = (M, N)
        self.cp_len = cp_len
        self._factory = output_factory

    def modulate(self, x_dd):
        return self._factory(x_dd)


# ============================================================================
# A. Happy-path adapter tests
# ============================================================================

class AdapterHappyPathTests(unittest.TestCase):

    def test_dummy_modem_returns_complex_time_signal(self):
        x_dd = _complex_randn(3, 8, 10)
        modem = DummyModem(8, 10, cp_len=0)
        ts = modulate_otfs_dd_frame(x_dd, modem)
        self.assertTrue(torch.is_complex(ts))

    def test_output_shape_cp_zero(self):
        x_dd = _complex_randn(3, 8, 10)
        modem = DummyModem(8, 10, cp_len=0)
        ts = modulate_otfs_dd_frame(x_dd, modem)
        self.assertEqual(ts.shape, (3, 10 * 8))

    def test_output_shape_cp_positive(self):
        x_dd = _complex_randn(3, 8, 10)
        modem = DummyModem(8, 10, cp_len=4)
        ts = modulate_otfs_dd_frame(x_dd, modem)
        self.assertEqual(ts.shape, (3, 10 * (8 + 4)))

    def test_output_device_matches_x_dd(self):
        x_dd = _complex_randn(3, 8, 10)
        modem = DummyModem(8, 10, cp_len=0)
        ts = modulate_otfs_dd_frame(x_dd, modem)
        self.assertEqual(ts.device, x_dd.device)

    def test_output_dtype_matches_x_dd(self):
        x_dd = _complex_randn(3, 8, 10, dtype=torch.complex64)
        modem = DummyModem(8, 10, cp_len=0)
        ts = modulate_otfs_dd_frame(x_dd, modem)
        self.assertEqual(ts.dtype, x_dd.dtype)

    def test_real_otfs_modem_matches_direct_call(self):
        modem = OTFSModem(dd_shape=(8, 10), cp_len=2)
        x_dd = _complex_randn(3, 8, 10)
        ts_adapter = modulate_otfs_dd_frame(x_dd, modem)
        ts_direct = modem.modulate(x_dd)
        self.assertTrue(torch.equal(ts_adapter, ts_direct))

    def test_spy_modem_receives_same_object(self):
        x_dd = _complex_randn(3, 8, 10)
        modem = SpyModem(8, 10, cp_len=0)
        modulate_otfs_dd_frame(x_dd, modem)
        self.assertIs(modem.received_x_dd, x_dd,
                      "adapter must not clone/detach/to x_dd")


# ============================================================================
# B. Reject-path: modem contract
# ============================================================================

class AdapterModemRejectTests(unittest.TestCase):

    def setUp(self):
        self.x_dd = _complex_randn(3, 8, 10)

    def test_modem_is_none_raises_type_error(self):
        with self.assertRaises(TypeError):
            modulate_otfs_dd_frame(self.x_dd, None)

    def test_modem_missing_modulate_raises_type_error(self):
        with self.assertRaises(TypeError):
            modulate_otfs_dd_frame(self.x_dd, MissingModulateModem())

    def test_modem_modulate_not_callable_raises_type_error(self):
        with self.assertRaises(TypeError):
            modulate_otfs_dd_frame(self.x_dd, NonCallableModulateModem())

    def test_modem_missing_dd_shape_raises_type_error(self):
        with self.assertRaises(TypeError):
            modulate_otfs_dd_frame(self.x_dd, MissingDDShapeModem())

    def test_dd_shape_not_tuple_raises_type_error(self):
        modem = DummyModem(8, 10)
        modem.dd_shape = [8, 10]
        with self.assertRaises(TypeError):
            modulate_otfs_dd_frame(self.x_dd, modem)

    def test_dd_shape_wrong_length_raises_value_error(self):
        modem = DummyModem(8, 10)
        modem.dd_shape = (8,)
        with self.assertRaises(ValueError):
            modulate_otfs_dd_frame(self.x_dd, modem)

    def test_dd_shape_element_not_int_raises_value_error(self):
        modem = DummyModem(8, 10)
        modem.dd_shape = (8.0, 10.0)
        with self.assertRaises(ValueError):
            modulate_otfs_dd_frame(self.x_dd, modem)

    def test_dd_shape_element_bool_raises_type_error(self):
        modem = DummyModem(8, 10)
        modem.dd_shape = (True, 10)
        with self.assertRaises(TypeError):
            modulate_otfs_dd_frame(self.x_dd, modem)

    def test_dd_shape_element_non_positive_raises_value_error(self):
        modem = DummyModem(8, 10)
        modem.dd_shape = (0, 10)
        with self.assertRaises(ValueError):
            modulate_otfs_dd_frame(self.x_dd, modem)

    def test_dd_shape_mismatch_raises_value_error(self):
        modem = DummyModem(4, 10)
        with self.assertRaises(ValueError):
            modulate_otfs_dd_frame(self.x_dd, modem)

    def test_modem_missing_cp_len_raises_type_error(self):
        modem = DummyModem(8, 10)
        del modem.cp_len
        with self.assertRaises(TypeError):
            modulate_otfs_dd_frame(self.x_dd, modem)

    def test_cp_len_not_int_raises_type_error(self):
        modem = DummyModem(8, 10)
        modem.cp_len = 0.0
        with self.assertRaises(TypeError):
            modulate_otfs_dd_frame(self.x_dd, modem)

    def test_cp_len_bool_raises_type_error(self):
        modem = DummyModem(8, 10)
        modem.cp_len = False
        with self.assertRaises(TypeError):
            modulate_otfs_dd_frame(self.x_dd, modem)

    def test_cp_len_negative_raises_value_error(self):
        modem = DummyModem(8, 10)
        modem.cp_len = -1
        with self.assertRaises(ValueError):
            modulate_otfs_dd_frame(self.x_dd, modem)

    def test_cp_len_exceeds_M_raises_value_error(self):
        modem = DummyModem(8, 10)
        modem.cp_len = 9
        with self.assertRaises(ValueError):
            modulate_otfs_dd_frame(self.x_dd, modem)


# ============================================================================
# C. Reject-path: x_dd
# ============================================================================

class AdapterXDDRejectTests(unittest.TestCase):

    def test_x_dd_non_tensor_raises_type_error(self):
        modem = DummyModem(8, 10)
        with self.assertRaises(TypeError):
            modulate_otfs_dd_frame([1.0], modem)

    def test_x_dd_non_complex_raises_type_error(self):
        modem = DummyModem(8, 10)
        with self.assertRaises(TypeError):
            modulate_otfs_dd_frame(torch.randn(3, 8, 10), modem)

    def test_x_dd_wrong_ndim_raises_value_error(self):
        modem = DummyModem(8, 10)
        with self.assertRaises(ValueError):
            modulate_otfs_dd_frame(_complex_randn(8, 10), modem)

    def test_x_dd_zero_batch_raises_value_error(self):
        modem = DummyModem(8, 10)
        with self.assertRaises(ValueError):
            modulate_otfs_dd_frame(_complex_randn(0, 8, 10), modem)

    def test_x_dd_zero_M_raises_value_error(self):
        modem = DummyModem(0, 10)
        with self.assertRaises(ValueError):
            modulate_otfs_dd_frame(_complex_randn(3, 0, 10), modem)

    def test_x_dd_zero_N_raises_value_error(self):
        modem = DummyModem(8, 0)
        with self.assertRaises(ValueError):
            modulate_otfs_dd_frame(_complex_randn(3, 8, 0), modem)


# ============================================================================
# D. Reject-path: modem output validation
# ============================================================================

class AdapterOutputRejectTests(unittest.TestCase):

    def setUp(self):
        self.x_dd = _complex_randn(3, 8, 10)

    def test_output_non_tensor_raises_type_error(self):
        modem = BadOutputModem(8, 10, output_factory=lambda x: "string")
        with self.assertRaises(TypeError):
            modulate_otfs_dd_frame(self.x_dd, modem)

    def test_output_non_complex_raises_type_error(self):
        modem = BadOutputModem(8, 10,
            output_factory=lambda x: torch.randn(3, 80))
        with self.assertRaises(TypeError):
            modulate_otfs_dd_frame(self.x_dd, modem)

    def test_output_wrong_ndim_raises_value_error(self):
        modem = BadOutputModem(8, 10,
            output_factory=lambda x: _complex_randn(3, 8, 10))
        with self.assertRaises(ValueError):
            modulate_otfs_dd_frame(self.x_dd, modem)

    def test_output_batch_mismatch_raises_value_error(self):
        modem = BadOutputModem(8, 10,
            output_factory=lambda x: _complex_randn(5, 80))
        with self.assertRaises(ValueError):
            modulate_otfs_dd_frame(self.x_dd, modem)

    def test_output_time_length_mismatch_raises_value_error(self):
        modem = BadOutputModem(8, 10,
            output_factory=lambda x: _complex_randn(3, 79))
        with self.assertRaises(ValueError):
            modulate_otfs_dd_frame(self.x_dd, modem)

    def test_output_device_mismatch_raises_value_error(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available for device mismatch test")
        x_dd_gpu = _complex_randn(3, 8, 10).to("cuda")
        modem = BadOutputModem(8, 10,
            output_factory=lambda x: _complex_randn(3, 80))
        with self.assertRaises(ValueError):
            modulate_otfs_dd_frame(x_dd_gpu, modem)

    def test_output_dtype_mismatch_raises_type_error(self):
        modem = BadOutputModem(8, 10,
            output_factory=lambda x: _complex_randn(
                3, 80, dtype=torch.complex128,
            ).to(x.device))
        with self.assertRaises(TypeError):
            modulate_otfs_dd_frame(self.x_dd, modem)


# ============================================================================
# E. Model integration tests
# ============================================================================

class ModelIntegrationTests(unittest.TestCase):

    def setUp(self):
        self.cfg = _tx_config()
        self.tx = LearnableTokenDDTransmitter(self.cfg)
        self.indices = _token_indices(3)

    # -- default behavior preserved --------------------------------------------

    def test_default_forward_returns_time_signal_none(self):
        out = self.tx.forward(self.indices)
        self.assertIsNone(out.time_signal)

    def test_default_tx_trace_time_modulation_enabled_false(self):
        out = self.tx.forward(self.indices)
        self.assertFalse(out.tx_trace["time_modulation_enabled"])

    # -- error paths -----------------------------------------------------------

    def test_return_time_true_modem_none_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.tx.forward(self.indices, return_time=True, modem=None)

    def test_return_time_false_modem_not_none_raises_value_error(self):
        modem = OTFSModem(dd_shape=(8, 10), cp_len=0)
        with self.assertRaises(ValueError):
            self.tx.forward(self.indices, return_time=False, modem=modem)

    def test_return_time_not_bool_1_raises_type_error(self):
        with self.assertRaises(TypeError):
            self.tx.forward(self.indices, return_time=1)

    def test_return_time_not_bool_0_raises_type_error(self):
        with self.assertRaises(TypeError):
            self.tx.forward(self.indices, return_time=0)

    # -- happy path with real OTFSModem ----------------------------------------

    def test_real_otfs_modem_cp_zero(self):
        modem = OTFSModem(dd_shape=(8, 10), cp_len=0)
        out = self.tx.forward(self.indices, return_time=True, modem=modem)
        self.assertIsNotNone(out.time_signal)
        self.assertEqual(out.time_signal.shape, (3, 10 * 8))

    def test_real_otfs_modem_cp_positive(self):
        modem = OTFSModem(dd_shape=(8, 10), cp_len=2)
        out = self.tx.forward(self.indices, return_time=True, modem=modem)
        self.assertEqual(out.time_signal.shape, (3, 10 * (8 + 2)))

    def test_time_signal_shape_correct(self):
        modem = OTFSModem(dd_shape=(8, 10), cp_len=3)
        out = self.tx.forward(self.indices, return_time=True, modem=modem)
        B, T = out.time_signal.shape
        self.assertEqual(B, 3)
        self.assertEqual(T, 10 * (8 + 3))

    def test_x_dd_unchanged_when_return_time_true(self):
        """x_dd must be identical regardless of return_time flag."""
        modem = OTFSModem(dd_shape=(8, 10), cp_len=0)
        out_no_time = self.tx.forward(self.indices, return_time=False)
        out_with_time = self.tx.forward(
            self.indices, return_time=True, modem=modem,
        )
        self.assertTrue(torch.equal(out_no_time.x_dd, out_with_time.x_dd))

    def test_dd_metrics_keys_unchanged(self):
        modem = OTFSModem(dd_shape=(8, 10), cp_len=2)
        out = self.tx.forward(self.indices, return_time=True, modem=modem)
        expected_keys = {"data_codeword_power", "pilot_power",
                         "total_dd_frame_power"}
        self.assertEqual(set(out.metrics.keys()), expected_keys)

    def test_tx_trace_time_modulation_enabled_true(self):
        modem = OTFSModem(dd_shape=(8, 10), cp_len=2)
        out = self.tx.forward(self.indices, return_time=True, modem=modem)
        self.assertTrue(out.tx_trace["time_modulation_enabled"])


# ============================================================================
# F. Autograd tests
# ============================================================================

class AutogradTests(unittest.TestCase):

    def test_gradients_flow_through_time_signal(self):
        cfg = _tx_config()
        tx = LearnableTokenDDTransmitter(cfg)
        indices = _token_indices(3)
        modem = OTFSModem(dd_shape=(8, 10), cp_len=2)

        out = tx.forward(indices, return_time=True, modem=modem)
        loss = out.time_signal.abs().pow(2).mean()
        loss.backward()

        self.assertIsNotNone(tx.codebook.raw_real.grad,
                             "raw_real.grad must not be None")
        self.assertIsNotNone(tx.codebook.raw_imag.grad,
                             "raw_imag.grad must not be None")
        self.assertTrue(torch.isfinite(tx.codebook.raw_real.grad).all(),
                        "raw_real.grad must be finite")
        self.assertTrue(torch.isfinite(tx.codebook.raw_imag.grad).all(),
                        "raw_imag.grad must be finite")


# ============================================================================
# G. CUDA tests
# ============================================================================

@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class CUDATests(unittest.TestCase):

    def test_cuda_roundtrip(self):
        cfg = _tx_config(complex_dtype="complex64")
        tx = LearnableTokenDDTransmitter(cfg).to("cuda")
        indices = _token_indices(3).to("cuda")
        modem = OTFSModem(dd_shape=(8, 10), cp_len=2, device="cuda",
                          dtype=torch.complex64)

        out = tx.forward(indices, return_time=True, modem=modem)
        self.assertTrue(out.time_signal.is_cuda)
        self.assertEqual(out.time_signal.dtype, torch.complex64)
        self.assertEqual(out.x_dd.device.type, "cuda")


# ============================================================================
# H. Quality tests
# ============================================================================

class QualityTests(unittest.TestCase):

    def test_modem_adapter_py_is_ascii_only(self):
        path = TRANSMITTER_ROOT / "transmitter" / "modem_adapter.py"
        content = path.read_text(encoding="utf-8")
        for i, ch in enumerate(content):
            self.assertTrue(ord(ch) < 128,
                            f"Non-ASCII U+{ord(ch):04X} at offset {i}")

    def test_model_py_is_ascii_only(self):
        path = TRANSMITTER_ROOT / "transmitter" / "model.py"
        content = path.read_text(encoding="utf-8")
        for i, ch in enumerate(content):
            self.assertTrue(ord(ch) < 128,
                            f"Non-ASCII U+{ord(ch):04X} at offset {i}")

    def test_modulate_otfs_dd_frame_docstring_exists(self):
        doc = modulate_otfs_dd_frame.__doc__
        self.assertIsNotNone(doc)
        self.assertTrue(len(doc.strip()) > 0)

    def test_docstring_mentions_shapes(self):
        doc = modulate_otfs_dd_frame.__doc__
        self.assertIn("[B, M, N]", doc)
        self.assertIn("[B, N * (M + cp_len)]", doc)


if __name__ == "__main__":
    unittest.main()
