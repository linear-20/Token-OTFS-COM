from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
TRANSMITTER_ROOT = ROOT / "Learnable_Mapping_Tokens-to-DD-Signals"
sys.path.insert(0, str(TRANSMITTER_ROOT))

from transmitter import (
    LearnableTokenDDTransmitter,
    TokenDDCodebook,
    TokenDDTransmitterOutput,
    TransmitterConfig,
    build_transmitter_pilot_masks,
    initialize_token_codebook_,
)


def _tx_config(**kwargs) -> TransmitterConfig:
    values = dict(
        M=8,
        N=10,
        vocab_size=6,
        pilot_delay=4,
        pilot_doppler=5,
        pilot_guard_delay=3,
        pilot_guard_doppler=3,
        pilot_obs_delay_radius=1,
        pilot_obs_doppler_radius=1,
        max_channel_delay=2,
        max_channel_doppler=2,
        data_power=1.5,
        pilot_value_real=2.0,
        pilot_value_imag=-1.0,
        complex_dtype="complex64",
    )
    values.update(kwargs)
    return TransmitterConfig(**values)


def _token_indices(batch: int, V: int = 6) -> torch.Tensor:
    return torch.arange(batch, dtype=torch.long) % V


class LearnableTokenDDTransmitterInitTests(unittest.TestCase):
    """Tests for LearnableTokenDDTransmitter.__init__."""

    def test_init_creates_codebook_when_none(self):
        cfg = _tx_config()
        tx = LearnableTokenDDTransmitter(cfg)
        self.assertIsInstance(tx.codebook, TokenDDCodebook)

    def test_init_accepts_compatible_codebook(self):
        cfg = _tx_config()
        cb = TokenDDCodebook(cfg)
        tx = LearnableTokenDDTransmitter(cfg, codebook=cb)
        self.assertIs(tx.codebook, cb)

    def test_incompatible_codebook_raises_value_error(self):
        cfg = _tx_config()
        other_cfg = _tx_config(data_power=999.0)
        cb = TokenDDCodebook(other_cfg)
        with self.assertRaises(ValueError):
            LearnableTokenDDTransmitter(cfg, codebook=cb)

    def test_non_codebook_raises_type_error(self):
        cfg = _tx_config()
        with self.assertRaises(TypeError):
            LearnableTokenDDTransmitter(cfg, codebook="not_a_codebook")  # type: ignore


class LearnableTokenDDTransmitterForwardTests(unittest.TestCase):
    """Tests for forward() output structure."""

    def setUp(self):
        self.cfg = _tx_config()
        self.tx = LearnableTokenDDTransmitter(self.cfg)
        self.indices = _token_indices(3)

    def test_forward_returns_token_dd_transmitter_output(self):
        out = self.tx.forward(self.indices)
        self.assertIsInstance(out, TokenDDTransmitterOutput)

    def test_output_token_indices_shape_B(self):
        out = self.tx.forward(self.indices)
        self.assertEqual(out.token_indices.shape, (3,))
        self.assertEqual(out.token_indices.dtype, torch.long)

    def test_output_codeword_book_complex_V_M_N(self):
        out = self.tx.forward(self.indices)
        self.assertTrue(torch.is_complex(out.codeword_book))
        self.assertEqual(out.codeword_book.shape, (6, 8, 10))

    def test_output_selected_codewords_complex_B_M_N(self):
        out = self.tx.forward(self.indices)
        self.assertTrue(torch.is_complex(out.selected_codewords))
        self.assertEqual(out.selected_codewords.shape, (3, 8, 10))

    def test_output_x_dd_complex_B_M_N(self):
        out = self.tx.forward(self.indices)
        self.assertTrue(torch.is_complex(out.x_dd))
        self.assertEqual(out.x_dd.shape, (3, 8, 10))

    def test_output_time_signal_is_none(self):
        out = self.tx.forward(self.indices)
        self.assertIsNone(out.time_signal)

    def test_output_masks_shape_1_M_N_bool(self):
        out = self.tx.forward(self.indices)
        for name in ["data_mask", "pilot_mask", "guard_mask"]:
            mask = getattr(out, name)
            self.assertEqual(mask.shape, (1, 8, 10), f"{name} shape")
            self.assertIs(mask.dtype, torch.bool, f"{name} dtype")

    def test_codeword_book_zero_in_pilot_guard(self):
        out = self.tx.forward(self.indices)
        guard_2d = out.guard_mask[0]
        pilot_2d = out.pilot_mask[0]
        for v in range(self.cfg.vocab_size):
            self.assertTrue(torch.all(out.codeword_book[v, guard_2d].abs() < 1e-5))
            self.assertTrue(torch.all(out.codeword_book[v, pilot_2d].abs() < 1e-5))

    def test_x_dd_pilot_bin_equals_pilot_value(self):
        out = self.tx.forward(self.indices)
        expected = torch.as_tensor(
            self.cfg.pilot_value, device=out.x_dd.device, dtype=out.x_dd.dtype,
        )
        actual = out.x_dd[:, self.cfg.pilot_delay, self.cfg.pilot_doppler]
        self.assertTrue(torch.allclose(actual, expected.expand_as(actual)))

    def test_x_dd_guard_bins_are_zero(self):
        out = self.tx.forward(self.indices)
        guard_2d = out.guard_mask[0].to(out.x_dd.device)
        self.assertTrue(torch.all(out.x_dd[:, guard_2d] == 0.0))

    def test_x_dd_data_bins_match_selected_codewords(self):
        out = self.tx.forward(self.indices)
        data_2d = out.data_mask[0].to(out.x_dd.device)
        self.assertTrue(
            torch.allclose(
                out.x_dd[:, data_2d],
                out.selected_codewords[:, data_2d],
            )
        )


class LearnableTokenDDTransmitterMetricsTests(unittest.TestCase):
    """Tests for metrics dict in forward output."""

    def setUp(self):
        self.cfg = _tx_config()
        self.tx = LearnableTokenDDTransmitter(self.cfg)
        self.indices = _token_indices(3)

    def test_metrics_contains_required_keys(self):
        out = self.tx.forward(self.indices)
        for key in ["data_codeword_power", "pilot_power", "total_dd_frame_power"]:
            self.assertIn(key, out.metrics)

    def test_data_codeword_power_shape_V(self):
        out = self.tx.forward(self.indices)
        dcp = out.metrics["data_codeword_power"]
        self.assertEqual(dcp.shape, (self.cfg.vocab_size,))

    def test_data_codeword_power_approx_data_power(self):
        out = self.tx.forward(self.indices)
        dcp = out.metrics["data_codeword_power"]
        for v in range(self.cfg.vocab_size):
            self.assertTrue(
                torch.allclose(
                    dcp[v],
                    torch.tensor(self.cfg.data_power, device=dcp.device, dtype=dcp.dtype),
                    atol=0.01,
                ),
                f"token {v}: {dcp[v].item():.4f}",
            )

    def test_pilot_power_is_scalar(self):
        out = self.tx.forward(self.indices)
        pp = out.metrics["pilot_power"]
        self.assertEqual(pp.ndim, 0)
        expected = abs(self.cfg.pilot_value) ** 2
        self.assertAlmostEqual(pp.item(), expected, places=4)

    def test_total_dd_frame_power_shape_B(self):
        out = self.tx.forward(self.indices)
        tfp = out.metrics["total_dd_frame_power"]
        self.assertEqual(tfp.shape, (3,))


class LearnableTokenDDTransmitterTraceTests(unittest.TestCase):
    """Tests for tx_trace dict."""

    def test_tx_trace_flags_correct(self):
        cfg = _tx_config()
        tx = LearnableTokenDDTransmitter(cfg)
        out = tx.forward(_token_indices(2))
        self.assertTrue(out.tx_trace["physical_codeword_book_generated"])
        self.assertFalse(out.tx_trace["physical_codeword_book_exported"])
        self.assertTrue(out.tx_trace["hard_data_mask_projection"])
        self.assertTrue(out.tx_trace["pilot_guard_hard_constraints"])
        self.assertFalse(out.tx_trace["time_modulation_enabled"])
        self.assertFalse(out.tx_trace["shaping_algorithm_enabled"])


class LearnableTokenDDTransmitterErrorTests(unittest.TestCase):
    """Tests for error handling in forward()."""

    def setUp(self):
        self.cfg = _tx_config()
        self.tx = LearnableTokenDDTransmitter(self.cfg)

    def test_return_time_true_with_modem_none_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.tx.forward(_token_indices(2), return_time=True, modem=None)

    def test_modem_provided_with_return_time_false_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.tx.forward(_token_indices(2), return_time=False, modem=object())

    def test_non_tensor_token_indices_raises_type_error(self):
        with self.assertRaises(TypeError):
            self.tx.forward([0, 1])  # type: ignore

    def test_non_long_token_indices_raises_type_error(self):
        with self.assertRaises(TypeError):
            self.tx.forward(torch.tensor([0, 1], dtype=torch.int32))

    def test_non_1d_token_indices_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.tx.forward(torch.tensor([[0], [1]], dtype=torch.long))

    def test_empty_token_indices_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.tx.forward(torch.empty(0, dtype=torch.long))

    def test_out_of_range_token_indices_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.tx.forward(torch.tensor([0, 6], dtype=torch.long))


class LearnableTokenDDTransmitterReceiverCompatTests(unittest.TestCase):
    """Tests that codeword_book meets receiver prior contract."""

    def test_codeword_book_shape_dtype_contract(self):
        cfg = _tx_config()
        tx = LearnableTokenDDTransmitter(cfg)
        out = tx.forward(_token_indices(2))
        book = out.codeword_book
        self.assertTrue(torch.is_complex(book))
        self.assertEqual(book.ndim, 3)
        self.assertEqual(book.shape, (cfg.vocab_size, cfg.M, cfg.N))

    def test_receiver_token_codeword_prior_compatible(self):
        cfg = _tx_config()
        tx = LearnableTokenDDTransmitter(cfg)
        out = tx.forward(_token_indices(1))
        try:
            receiver_root = ROOT / "Learnable_Receiver_DD-Signals-to-Tokens"
            sys.path.insert(0, str(receiver_root))
            try:
                from receiver.token_prior import TokenCodewordPrior
            except ImportError:
                self.skipTest("TokenCodewordPrior not importable")
            finally:
                if str(receiver_root) in sys.path:
                    sys.path.remove(str(receiver_root))
            prior = TokenCodewordPrior(codeword_book=out.codeword_book)
            self.assertIsNotNone(prior)
        except ImportError:
            self.skipTest("Receiver package not available")


if __name__ == "__main__":
    unittest.main()
