from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
TRANSMITTER_ROOT = ROOT / "Learnable_Mapping_Tokens-to-DD-Signals"
sys.path.insert(0, str(TRANSMITTER_ROOT))

from transmitter import (
    LearnableTokenDDTransmitter,
    TransmitterConfig,
    build_transmitter_pilot_masks,
)
from transmitter.diagnostics import (
    data_codeword_power,
    dd_frame_power,
    pilot_power,
    summarize_dd_transmitter_metrics,
)


def _tx_cfg(**kwargs) -> TransmitterConfig:
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


def _complex_cw(cfg: TransmitterConfig) -> torch.Tensor:
    V, M, N = cfg.vocab_size, cfg.M, cfg.N
    real = torch.randn(V, M, N)
    imag = torch.randn(V, M, N)
    return torch.complex(real, imag)


def _complex_xdd(batch: int, cfg: TransmitterConfig) -> torch.Tensor:
    real = torch.randn(batch, cfg.M, cfg.N)
    imag = torch.randn(batch, cfg.M, cfg.N)
    return torch.complex(real, imag)


class DataCodewordPowerTests(unittest.TestCase):
    """Tests for data_codeword_power()."""

    def test_valid_1_M_N_mask_returns_V(self):
        cfg = _tx_cfg()
        cw = _complex_cw(cfg)
        mask = build_transmitter_pilot_masks(cfg).data_mask
        result = data_codeword_power(cw, mask)
        self.assertEqual(result.shape, (cfg.vocab_size,))

    def test_valid_V_M_N_mask_returns_V(self):
        cfg = _tx_cfg()
        cw = _complex_cw(cfg)
        mask_1 = build_transmitter_pilot_masks(cfg).data_mask
        mask = mask_1.expand(cfg.vocab_size, -1, -1)
        result = data_codeword_power(cw, mask)
        self.assertEqual(result.shape, (cfg.vocab_size,))

    def test_matches_manual_mean_over_active_region(self):
        cfg = _tx_cfg()
        cw = _complex_cw(cfg)
        mask = build_transmitter_pilot_masks(cfg).data_mask
        result = data_codeword_power(cw, mask)
        data_2d = mask[0]
        for v in range(cfg.vocab_size):
            manual = cw[v, data_2d].abs().pow(2).mean()
            self.assertAlmostEqual(result[v].item(), manual.item(), places=5)

    def test_output_dtype_complex64_is_float32(self):
        cfg = _tx_cfg()
        cw = _complex_cw(cfg)
        mask = build_transmitter_pilot_masks(cfg).data_mask
        self.assertEqual(data_codeword_power(cw, mask).dtype, torch.float32)

    def test_output_dtype_complex128_is_float64(self):
        cfg = _tx_cfg(complex_dtype="complex128")
        cw = torch.complex(
            torch.randn(cfg.vocab_size, cfg.M, cfg.N, dtype=torch.float64),
            torch.randn(cfg.vocab_size, cfg.M, cfg.N, dtype=torch.float64),
        )
        mask = build_transmitter_pilot_masks(cfg).data_mask
        self.assertEqual(data_codeword_power(cw, mask).dtype, torch.float64)

    def test_output_device_matches_codeword_book(self):
        cfg = _tx_cfg()
        cw = _complex_cw(cfg)
        mask = build_transmitter_pilot_masks(cfg).data_mask
        self.assertEqual(
            data_codeword_power(cw, mask).device, cw.device,
        )

    def test_non_tensor_codeword_book_raises(self):
        mask = build_transmitter_pilot_masks(_tx_cfg()).data_mask
        with self.assertRaises(TypeError):
            data_codeword_power([1.0], mask)  # type: ignore

    def test_non_complex_codeword_book_raises(self):
        cfg = _tx_cfg()
        mask = build_transmitter_pilot_masks(cfg).data_mask
        with self.assertRaises(TypeError):
            data_codeword_power(torch.randn(3, 8, 10), mask)

    def test_bad_codeword_book_shape_raises(self):
        cfg = _tx_cfg()
        mask = build_transmitter_pilot_masks(cfg).data_mask
        cw = torch.complex(torch.randn(3, 2), torch.randn(3, 2))
        with self.assertRaises(ValueError):
            data_codeword_power(cw, mask)

    def test_non_tensor_data_mask_raises(self):
        cfg = _tx_cfg()
        cw = _complex_cw(cfg)
        with self.assertRaises(TypeError):
            data_codeword_power(cw, [1.0])  # type: ignore

    def test_complex_data_mask_raises(self):
        cfg = _tx_cfg()
        cw = _complex_cw(cfg)
        mask = torch.ones(1, cfg.M, cfg.N, dtype=torch.complex64)
        with self.assertRaises(TypeError):
            data_codeword_power(cw, mask)

    def test_bad_data_mask_shape_raises(self):
        cfg = _tx_cfg()
        cw = _complex_cw(cfg)
        mask = torch.ones(2, cfg.M, cfg.N)
        with self.assertRaises(ValueError):
            data_codeword_power(cw, mask)

    def test_spatial_shape_mismatch_raises(self):
        cfg = _tx_cfg()
        cw = _complex_cw(cfg)
        mask = torch.ones(1, 5, 5)
        with self.assertRaises(ValueError):
            data_codeword_power(cw, mask)

    def test_nonfinite_data_mask_raises(self):
        cfg = _tx_cfg()
        cw = _complex_cw(cfg)
        mask = torch.full((1, cfg.M, cfg.N), float("inf"))
        with self.assertRaises(ValueError):
            data_codeword_power(cw, mask)

    def test_fractional_data_mask_raises(self):
        cfg = _tx_cfg()
        cw = _complex_cw(cfg)
        mask = torch.full((1, cfg.M, cfg.N), 0.5)
        with self.assertRaises(ValueError):
            data_codeword_power(cw, mask)

    def test_empty_active_region_raises(self):
        cfg = _tx_cfg()
        cw = _complex_cw(cfg)
        mask = torch.zeros(1, cfg.M, cfg.N)
        with self.assertRaises(ValueError):
            data_codeword_power(cw, mask)

    def test_per_token_empty_active_raises(self):
        cfg = _tx_cfg()
        cw = _complex_cw(cfg)
        mask = torch.ones(cfg.vocab_size, cfg.M, cfg.N)
        mask[3] = 0.0
        with self.assertRaises(ValueError):
            data_codeword_power(cw, mask)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cpu_mask_cuda_codeword_book_output_stays_cuda(self):
        cfg = _tx_cfg()
        cw = _complex_cw(cfg).cuda()
        mask = build_transmitter_pilot_masks(cfg).data_mask  # CPU mask
        result = data_codeword_power(cw, mask)
        self.assertEqual(result.device.type, "cuda")
        self.assertEqual(result.shape, (cfg.vocab_size,))

    def test_zero_vocab_dimension_raises(self):
        cw = torch.zeros(0, 8, 10, dtype=torch.complex64)
        mask = torch.ones(1, 8, 10)
        with self.assertRaises(ValueError):
            data_codeword_power(cw, mask)

    def test_zero_M_dimension_raises(self):
        cw = torch.zeros(3, 0, 10, dtype=torch.complex64)
        mask = torch.ones(1, 0, 10)
        with self.assertRaises(ValueError):
            data_codeword_power(cw, mask)

    def test_zero_N_dimension_raises(self):
        cw = torch.zeros(3, 8, 0, dtype=torch.complex64)
        mask = torch.ones(1, 8, 0)
        with self.assertRaises(ValueError):
            data_codeword_power(cw, mask)


class PilotPowerTests(unittest.TestCase):
    """Tests for pilot_power()."""

    def test_python_complex_returns_scalar(self):
        result = pilot_power(2.0 + 1.0j)
        self.assertEqual(result.shape, ())
        self.assertAlmostEqual(result.item(), 5.0, places=5)

    def test_python_real_returns_scalar(self):
        result = pilot_power(3.0)
        self.assertEqual(result.shape, ())
        self.assertAlmostEqual(result.item(), 9.0, places=5)

    def test_complex_tensor_scalar_returns_scalar(self):
        pv = torch.as_tensor(2.0 + 1.0j)
        result = pilot_power(pv)
        self.assertEqual(result.shape, ())
        self.assertAlmostEqual(result.item(), 5.0, places=4)

    def test_real_tensor_scalar_returns_scalar(self):
        pv = torch.tensor(3.0)
        result = pilot_power(pv)
        self.assertEqual(result.shape, ())
        self.assertAlmostEqual(result.item(), 9.0, places=5)

    def test_output_shape_is_scalar(self):
        self.assertEqual(pilot_power(1.0 + 0.0j).shape, ())

    def test_output_dtype_equals_requested(self):
        result = pilot_power(1.0, dtype=torch.float64)
        self.assertEqual(result.dtype, torch.float64)

    def test_output_device_from_tensor_pilot_value(self):
        pv = torch.tensor(2.0, device="cpu")
        result = pilot_power(pv)
        self.assertEqual(result.device, pv.device)

    def test_output_device_explicit(self):
        result = pilot_power(1.0, device=torch.device("cpu"))
        self.assertEqual(result.device.type, "cpu")

    def test_non_floating_dtype_raises(self):
        with self.assertRaises(TypeError):
            pilot_power(1.0, dtype=torch.int32)  # type: ignore

    def test_non_scalar_tensor_raises(self):
        with self.assertRaises(ValueError):
            pilot_power(torch.tensor([1.0, 2.0]))

    def test_unsupported_object_raises_type_error(self):
        with self.assertRaises(TypeError):
            pilot_power("bad")  # type: ignore

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_device_preserved(self):
        pv = torch.tensor(2.0, device="cuda")
        result = pilot_power(pv)
        self.assertEqual(result.device.type, "cuda")


class DDFramePowerTests(unittest.TestCase):
    """Tests for dd_frame_power()."""

    def test_valid_returns_B(self):
        cfg = _tx_cfg()
        x_dd = _complex_xdd(4, cfg)
        result = dd_frame_power(x_dd)
        self.assertEqual(result.shape, (4,))

    def test_matches_manual_mean(self):
        cfg = _tx_cfg()
        x_dd = _complex_xdd(3, cfg)
        manual = x_dd.abs().pow(2).mean(dim=(-2, -1))
        result = dd_frame_power(x_dd)
        self.assertTrue(torch.allclose(result, manual.to(dtype=torch.float32)))

    def test_output_dtype_complex64_is_float32(self):
        cfg = _tx_cfg()
        x_dd = _complex_xdd(2, cfg)
        self.assertEqual(dd_frame_power(x_dd).dtype, torch.float32)

    def test_output_dtype_complex128_is_float64(self):
        cfg = _tx_cfg(complex_dtype="complex128")
        x_dd = torch.complex(
            torch.randn(2, cfg.M, cfg.N, dtype=torch.float64),
            torch.randn(2, cfg.M, cfg.N, dtype=torch.float64),
        )
        self.assertEqual(dd_frame_power(x_dd).dtype, torch.float64)

    def test_output_device_matches_x_dd(self):
        x_dd = _complex_xdd(2, _tx_cfg())
        self.assertEqual(dd_frame_power(x_dd).device, x_dd.device)

    def test_non_tensor_raises(self):
        with self.assertRaises(TypeError):
            dd_frame_power([1.0])  # type: ignore

    def test_non_complex_raises(self):
        with self.assertRaises(TypeError):
            dd_frame_power(torch.randn(2, 8, 10))

    def test_wrong_shape_raises(self):
        with self.assertRaises(ValueError):
            dd_frame_power(torch.complex(torch.randn(2, 3), torch.randn(2, 3)))

    def test_zero_batch_raises(self):
        x_dd = torch.zeros(0, 8, 10, dtype=torch.complex64)
        with self.assertRaises(ValueError):
            dd_frame_power(x_dd)

    def test_zero_M_dimension_raises(self):
        x_dd = torch.zeros(2, 0, 10, dtype=torch.complex64)
        with self.assertRaises(ValueError):
            dd_frame_power(x_dd)

    def test_zero_N_dimension_raises(self):
        x_dd = torch.zeros(2, 8, 0, dtype=torch.complex64)
        with self.assertRaises(ValueError):
            dd_frame_power(x_dd)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_device_preserved(self):
        cfg = _tx_cfg()
        x_dd = _complex_xdd(2, cfg).cuda()
        result = dd_frame_power(x_dd)
        self.assertEqual(result.device.type, "cuda")


class SummarizeMetricsTests(unittest.TestCase):
    """Tests for summarize_dd_transmitter_metrics()."""

    def test_returns_exactly_required_keys(self):
        cfg = _tx_cfg()
        cw = _complex_cw(cfg)
        x_dd = _complex_xdd(3, cfg)
        mask = build_transmitter_pilot_masks(cfg).data_mask
        result = summarize_dd_transmitter_metrics(cw, x_dd, mask, 2.0 - 1.0j)
        self.assertSetEqual(
            set(result.keys()),
            {"data_codeword_power", "pilot_power", "total_dd_frame_power"},
        )

    def test_data_codeword_power_shape_V(self):
        cfg = _tx_cfg()
        cw = _complex_cw(cfg)
        x_dd = _complex_xdd(3, cfg)
        mask = build_transmitter_pilot_masks(cfg).data_mask
        result = summarize_dd_transmitter_metrics(cw, x_dd, mask, 1.0)
        self.assertEqual(result["data_codeword_power"].shape, (cfg.vocab_size,))

    def test_pilot_power_shape_scalar(self):
        cfg = _tx_cfg()
        cw = _complex_cw(cfg)
        x_dd = _complex_xdd(3, cfg)
        mask = build_transmitter_pilot_masks(cfg).data_mask
        result = summarize_dd_transmitter_metrics(cw, x_dd, mask, 1.0)
        self.assertEqual(result["pilot_power"].shape, ())

    def test_total_dd_frame_power_shape_B(self):
        cfg = _tx_cfg()
        cw = _complex_cw(cfg)
        x_dd = _complex_xdd(3, cfg)
        mask = build_transmitter_pilot_masks(cfg).data_mask
        result = summarize_dd_transmitter_metrics(cw, x_dd, mask, 1.0)
        self.assertEqual(result["total_dd_frame_power"].shape, (3,))

    def test_values_match_individual_functions(self):
        cfg = _tx_cfg()
        cw = _complex_cw(cfg)
        x_dd = _complex_xdd(3, cfg)
        mask = build_transmitter_pilot_masks(cfg).data_mask
        pv = 2.0 - 1.0j
        result = summarize_dd_transmitter_metrics(cw, x_dd, mask, pv)
        self.assertTrue(torch.equal(
            result["data_codeword_power"], data_codeword_power(cw, mask),
        ))
        real_dtype = torch.float32
        self.assertAlmostEqual(
            result["pilot_power"].item(),
            pilot_power(pv, dtype=real_dtype).item(),
            places=5,
        )
        self.assertTrue(torch.allclose(
            result["total_dd_frame_power"], dd_frame_power(x_dd),
        ))


class ModelIntegrationTests(unittest.TestCase):
    """Tests that model.py correctly uses the new metrics functions."""

    def test_forward_metrics_keys_unchanged(self):
        cfg = _tx_cfg()
        tx = LearnableTokenDDTransmitter(cfg)
        out = tx.forward(torch.tensor([0, 1, 2], dtype=torch.long))
        self.assertSetEqual(
            set(out.metrics.keys()),
            {"data_codeword_power", "pilot_power", "total_dd_frame_power"},
        )

    def test_data_codeword_power_shape_V(self):
        cfg = _tx_cfg()
        tx = LearnableTokenDDTransmitter(cfg)
        out = tx.forward(torch.tensor([0, 1, 2], dtype=torch.long))
        self.assertEqual(out.metrics["data_codeword_power"].shape, (cfg.vocab_size,))

    def test_pilot_power_shape_scalar(self):
        cfg = _tx_cfg()
        tx = LearnableTokenDDTransmitter(cfg)
        out = tx.forward(torch.tensor([0, 1, 2], dtype=torch.long))
        self.assertEqual(out.metrics["pilot_power"].shape, ())

    def test_total_dd_frame_power_shape_B(self):
        cfg = _tx_cfg()
        tx = LearnableTokenDDTransmitter(cfg)
        out = tx.forward(torch.tensor([0, 1, 2], dtype=torch.long))
        self.assertEqual(out.metrics["total_dd_frame_power"].shape, (3,))

    def test_data_codeword_power_approx_data_power(self):
        cfg = _tx_cfg()
        tx = LearnableTokenDDTransmitter(cfg)
        out = tx.forward(torch.tensor([0, 1, 2], dtype=torch.long))
        dcp = out.metrics["data_codeword_power"]
        for v in range(cfg.vocab_size):
            self.assertTrue(
                torch.allclose(
                    dcp[v], torch.tensor(cfg.data_power, device=dcp.device, dtype=dcp.dtype),
                    atol=0.01,
                ),
            )

    def test_output_semantics_unchanged(self):
        cfg = _tx_cfg()
        tx = LearnableTokenDDTransmitter(cfg)
        out = tx.forward(torch.tensor([0, 1], dtype=torch.long))
        self.assertTrue(torch.is_complex(out.codeword_book))
        self.assertTrue(torch.is_complex(out.selected_codewords))
        self.assertTrue(torch.is_complex(out.x_dd))
        self.assertIsNone(out.time_signal)


class MetricsFileEncodingTests(unittest.TestCase):
    """Tests that metrics.py contains only ASCII characters."""

    def test_metrics_py_is_ascii_only(self):
        metrics_path = Path(TRANSMITTER_ROOT) / "transmitter" / "metrics.py"
        content = metrics_path.read_text(encoding="utf-8")
        for i, ch in enumerate(content):
            self.assertLess(ord(ch), 128, f"non-ASCII char at position {i}: {ch!r}")


if __name__ == "__main__":
    unittest.main()
