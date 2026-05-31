from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
TRANSMITTER_ROOT = ROOT / "Learnable_Mapping_Tokens-to-DD-Signals"
sys.path.insert(0, str(TRANSMITTER_ROOT))

from transmitter import (
    TokenDDCodebook,
    TransmitterConfig,
    build_transmitter_pilot_masks,
)


def _codebook_config(**kwargs) -> TransmitterConfig:
    values = dict(
        M=8,
        N=10,
        vocab_size=8,
        pilot_delay=4,
        pilot_doppler=5,
        pilot_guard_delay=3,
        pilot_guard_doppler=3,
        pilot_obs_delay_radius=1,
        pilot_obs_doppler_radius=1,
        max_channel_delay=2,
        max_channel_doppler=2,
        complex_dtype="complex64",
    )
    values.update(kwargs)
    return TransmitterConfig(**values)


class TokenDDCodebookInitTests(unittest.TestCase):
    """Tests for TokenDDCodebook.__init__."""

    def test_raw_params_shape(self):
        cfg = _codebook_config()
        cb = TokenDDCodebook(cfg)
        V, M, N = cfg.vocab_size, cfg.M, cfg.N
        self.assertEqual(cb.raw_real.shape, (V, M, N))
        self.assertEqual(cb.raw_imag.shape, (V, M, N))

    def test_raw_params_dtype(self):
        cfg = _codebook_config(complex_dtype="complex64")
        cb = TokenDDCodebook(cfg)
        self.assertEqual(cb.raw_real.dtype, torch.float32)
        self.assertEqual(cb.raw_imag.dtype, torch.float32)

    def test_raw_params_dtype_complex128(self):
        cfg = _codebook_config(complex_dtype="complex128")
        cb = TokenDDCodebook(cfg)
        self.assertEqual(cb.raw_real.dtype, torch.float64)
        self.assertEqual(cb.raw_imag.dtype, torch.float64)

    def test_invalid_init_scale_zero_raises(self):
        cfg = _codebook_config()
        with self.assertRaises(ValueError):
            TokenDDCodebook(cfg, init_scale=0.0)

    def test_invalid_init_scale_negative_raises(self):
        cfg = _codebook_config()
        with self.assertRaises(ValueError):
            TokenDDCodebook(cfg, init_scale=-0.02)

    def test_invalid_init_scale_nonfinite_raises(self):
        cfg = _codebook_config()
        with self.assertRaises(ValueError):
            TokenDDCodebook(cfg, init_scale=float("inf"))
        with self.assertRaises(ValueError):
            TokenDDCodebook(cfg, init_scale=float("nan"))

    def test_invalid_init_scale_non_number_raises(self):
        cfg = _codebook_config()
        with self.assertRaises(TypeError):
            TokenDDCodebook(cfg, init_scale="0.02")  # type: ignore

    def test_invalid_init_scale_bool_raises(self):
        cfg = _codebook_config()
        with self.assertRaises(TypeError):
            TokenDDCodebook(cfg, init_scale=True)


class TokenDDCodebookForwardBasicTests(unittest.TestCase):
    """Basic forward() behaviour tests."""

    def test_forward_returns_complex_V_M_N(self):
        cfg = _codebook_config()
        cb = TokenDDCodebook(cfg)
        book = cb.forward()
        self.assertTrue(torch.is_complex(book))
        self.assertEqual(book.shape, (cfg.vocab_size, cfg.M, cfg.N))

    def test_forward_output_dtype_matches_config(self):
        cfg = _codebook_config(complex_dtype="complex64")
        cb = TokenDDCodebook(cfg)
        book = cb.forward()
        self.assertEqual(book.dtype, torch.complex64)

    def test_forward_output_device_matches_raw_params(self):
        cfg = _codebook_config()
        cb = TokenDDCodebook(cfg)
        book = cb.forward()
        self.assertEqual(book.device, cb.raw_real.device)

    def test_forward_data_mask_none_uses_full_grid(self):
        cfg = _codebook_config(data_power=1.0)
        cb = TokenDDCodebook(cfg)
        book = cb.forward()
        avg_power = book.abs().pow(2).mean()
        self.assertTrue(torch.allclose(avg_power, torch.tensor(1.0), atol=0.01))

    def test_forward_pilot_guard_are_zero_with_data_mask(self):
        cfg = _codebook_config()
        masks = build_transmitter_pilot_masks(cfg)
        cb = TokenDDCodebook(cfg)
        book = cb.forward(data_mask=masks.data_mask)
        guard_2d = masks.guard_mask[0]
        pilot_2d = masks.pilot_mask[0]
        for v in range(cfg.vocab_size):
            self.assertTrue(torch.all(book[v, guard_2d].abs() < 1e-6))
            self.assertTrue(torch.all(book[v, pilot_2d].abs() < 1e-6))

    def test_forward_data_region_average_power_equals_data_power(self):
        cfg = _codebook_config(data_power=2.0)
        masks = build_transmitter_pilot_masks(cfg)
        cb = TokenDDCodebook(cfg)
        book = cb.forward(data_mask=masks.data_mask)
        data_2d = masks.data_mask[0]
        for v in range(cfg.vocab_size):
            avg_power = book[v, data_2d].abs().pow(2).mean()
            self.assertTrue(
                torch.allclose(avg_power, torch.tensor(cfg.data_power), atol=0.01)
            )

    def test_forward_data_mask_1_M_N_broadcasts(self):
        cfg = _codebook_config()
        mask = torch.ones(1, cfg.M, cfg.N)
        cb = TokenDDCodebook(cfg)
        book = cb.forward(data_mask=mask)
        self.assertEqual(book.shape, (cfg.vocab_size, cfg.M, cfg.N))
        self.assertTrue(torch.is_complex(book))

    def test_forward_data_mask_V_M_N_per_token(self):
        cfg = _codebook_config()
        V = cfg.vocab_size
        mask = torch.ones(V, cfg.M, cfg.N)
        cb = TokenDDCodebook(cfg)
        book = cb.forward(data_mask=mask)
        self.assertEqual(book.shape, (V, cfg.M, cfg.N))

    def test_forward_per_token_mask_independent_normalization(self):
        cfg = _codebook_config(data_power=1.0, vocab_size=3)
        V = cfg.vocab_size
        M, N = cfg.M, cfg.N
        mask = torch.ones(V, M, N)
        mask[0, :, M // 2 :] = 0.0
        mask[1, :M // 2, :] = 0.0
        cb = TokenDDCodebook(cfg)
        book = cb.forward(data_mask=mask)
        for v in range(V):
            active = mask[v] > 0
            if active.any():
                avg_power = book[v, active].abs().pow(2).mean()
                self.assertTrue(
                    torch.allclose(avg_power, torch.tensor(1.0), atol=0.01),
                    f"token {v}: avg_power={avg_power.item():.4f}",
                )

    def test_forward_bool_data_mask_supported(self):
        cfg = _codebook_config()
        mask = torch.ones(1, cfg.M, cfg.N, dtype=torch.bool)
        cb = TokenDDCodebook(cfg)
        book = cb.forward(data_mask=mask)
        self.assertEqual(book.shape, (cfg.vocab_size, cfg.M, cfg.N))

    def test_forward_real_binary_mask_supported(self):
        cfg = _codebook_config()
        mask = torch.ones(1, cfg.M, cfg.N)  # real 1.0
        cb = TokenDDCodebook(cfg)
        book = cb.forward(data_mask=mask)
        self.assertEqual(book.shape, (cfg.vocab_size, cfg.M, cfg.N))

    def test_forward_fractional_data_mask_raises_value_error(self):
        cfg = _codebook_config()
        mask = torch.full((1, cfg.M, cfg.N), 0.5)
        cb = TokenDDCodebook(cfg)
        with self.assertRaises(ValueError):
            cb.forward(data_mask=mask)

    def test_forward_negative_data_mask_raises_value_error(self):
        cfg = _codebook_config()
        mask = torch.full((1, cfg.M, cfg.N), -1.0)
        cb = TokenDDCodebook(cfg)
        with self.assertRaises(ValueError):
            cb.forward(data_mask=mask)

    def test_forward_out_of_binary_data_mask_raises_value_error(self):
        cfg = _codebook_config()
        mask = torch.full((1, cfg.M, cfg.N), 2.0)
        cb = TokenDDCodebook(cfg)
        with self.assertRaises(ValueError):
            cb.forward(data_mask=mask)

    def test_forward_binary_real_mask_active_power_equals_data_power(self):
        cfg = _codebook_config(data_power=1.5)
        masks = build_transmitter_pilot_masks(cfg)
        real_data_mask = masks.data_mask.to(dtype=cfg.torch_real_dtype)
        cb = TokenDDCodebook(cfg)
        book = cb.forward(data_mask=real_data_mask)
        data_2d = masks.data_mask[0]
        for v in range(cfg.vocab_size):
            avg_power = book[v, data_2d].abs().pow(2).mean()
            self.assertTrue(
                torch.allclose(avg_power, torch.tensor(cfg.data_power), atol=0.01)
            )

    def test_forward_complex_data_mask_raises_type_error(self):
        cfg = _codebook_config()
        mask = torch.ones(1, cfg.M, cfg.N, dtype=torch.complex64)
        cb = TokenDDCodebook(cfg)
        with self.assertRaises(TypeError):
            cb.forward(data_mask=mask)

    def test_forward_bad_shape_raises_value_error(self):
        cfg = _codebook_config()
        mask = torch.ones(2, cfg.M, cfg.N)
        cb = TokenDDCodebook(cfg)
        with self.assertRaises(ValueError):
            cb.forward(data_mask=mask)

    def test_forward_spatial_shape_mismatch_raises_value_error(self):
        cfg = _codebook_config()
        mask = torch.ones(1, 5, 5)
        cb = TokenDDCodebook(cfg)
        with self.assertRaises(ValueError):
            cb.forward(data_mask=mask)

    def test_forward_nonfinite_data_mask_raises_value_error(self):
        cfg = _codebook_config()
        mask = torch.full((1, cfg.M, cfg.N), float("inf"))
        cb = TokenDDCodebook(cfg)
        with self.assertRaises(ValueError):
            cb.forward(data_mask=mask)

    def test_forward_empty_data_mask_raises_value_error(self):
        cfg = _codebook_config()
        mask = torch.zeros(1, cfg.M, cfg.N)
        cb = TokenDDCodebook(cfg)
        with self.assertRaises(ValueError):
            cb.forward(data_mask=mask)

    def test_forward_per_token_empty_active_raises_value_error(self):
        cfg = _codebook_config(vocab_size=2)
        V, M, N = cfg.vocab_size, cfg.M, cfg.N
        mask = torch.ones(V, M, N)
        mask[1] = 0.0
        cb = TokenDDCodebook(cfg)
        with self.assertRaises(ValueError):
            cb.forward(data_mask=mask)

    def test_forward_zero_raw_params_raises_value_error(self):
        cfg = _codebook_config()
        cb = TokenDDCodebook(cfg)
        cb.raw_real.data.zero_()
        cb.raw_imag.data.zero_()
        mask = torch.ones(1, cfg.M, cfg.N)
        with self.assertRaises(ValueError):
            cb.forward(data_mask=mask)

    def test_forward_cuda_output_device(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        cfg = _codebook_config()
        cb = TokenDDCodebook(cfg).cuda()
        book = cb.forward()
        self.assertEqual(book.device.type, "cuda")


class TokenDDCodebookSelectedCodewordsTests(unittest.TestCase):
    """Tests for selected_codewords()."""

    def test_returns_complex_B_M_N(self):
        cfg = _codebook_config()
        cb = TokenDDCodebook(cfg)
        indices = torch.tensor([0, 1, 2], dtype=torch.long)
        sel = cb.selected_codewords(indices)
        self.assertTrue(torch.is_complex(sel))
        self.assertEqual(sel.shape, (3, cfg.M, cfg.N))

    def test_preserves_token_order(self):
        cfg = _codebook_config()
        cb = TokenDDCodebook(cfg)
        book = cb.forward()
        indices = torch.tensor([3, 0, 5], dtype=torch.long)
        sel = cb.selected_codewords(indices)
        self.assertTrue(torch.equal(sel, book[indices]))

    def test_equals_forward_indexed(self):
        cfg = _codebook_config()
        cb = TokenDDCodebook(cfg)
        book = cb.forward()
        indices = torch.tensor([2, 4], dtype=torch.long)
        sel = cb.selected_codewords(indices, codeword_book=book)
        self.assertTrue(torch.equal(sel, book[indices]))

    def test_non_tensor_raises_type_error(self):
        cfg = _codebook_config()
        cb = TokenDDCodebook(cfg)
        with self.assertRaises(TypeError):
            cb.selected_codewords([0, 1])  # type: ignore

    def test_non_long_dtype_raises_type_error(self):
        cfg = _codebook_config()
        cb = TokenDDCodebook(cfg)
        with self.assertRaises(TypeError):
            cb.selected_codewords(torch.tensor([0, 1], dtype=torch.int32))

    def test_wrong_ndim_raises_value_error(self):
        cfg = _codebook_config()
        cb = TokenDDCodebook(cfg)
        with self.assertRaises(ValueError):
            cb.selected_codewords(torch.tensor([[0, 1]], dtype=torch.long))

    def test_zero_batch_raises_value_error(self):
        cfg = _codebook_config()
        cb = TokenDDCodebook(cfg)
        with self.assertRaises(ValueError):
            cb.selected_codewords(torch.empty(0, dtype=torch.long))

    def test_out_of_range_raises_value_error(self):
        cfg = _codebook_config(vocab_size=4)
        cb = TokenDDCodebook(cfg)
        with self.assertRaises(ValueError):
            cb.selected_codewords(torch.tensor([0, 4], dtype=torch.long))

    def test_provided_codeword_book_valid_accepted(self):
        cfg = _codebook_config()
        cb = TokenDDCodebook(cfg)
        book = cb.forward()
        indices = torch.tensor([0, 1], dtype=torch.long)
        sel = cb.selected_codewords(indices, codeword_book=book)
        self.assertTrue(torch.equal(sel, book[indices]))

    def test_provided_codeword_book_non_tensor_raises_type_error(self):
        cfg = _codebook_config()
        cb = TokenDDCodebook(cfg)
        indices = torch.tensor([0], dtype=torch.long)
        with self.assertRaises(TypeError):
            cb.selected_codewords(indices, codeword_book=[1.0])  # type: ignore

    def test_provided_codeword_book_non_complex_raises_type_error(self):
        cfg = _codebook_config()
        cb = TokenDDCodebook(cfg)
        indices = torch.tensor([0], dtype=torch.long)
        bad_book = torch.ones(cfg.vocab_size, cfg.M, cfg.N)
        with self.assertRaises(TypeError):
            cb.selected_codewords(indices, codeword_book=bad_book)

    def test_provided_codeword_book_wrong_shape_raises_value_error(self):
        cfg = _codebook_config()
        cb = TokenDDCodebook(cfg)
        indices = torch.tensor([0], dtype=torch.long)
        bad_book = torch.ones(2, cfg.M, cfg.N, dtype=torch.complex64)
        with self.assertRaises(ValueError):
            cb.selected_codewords(indices, codeword_book=bad_book)


class TokenDDCodebookReceiverCompatibilityTests(unittest.TestCase):
    """Tests that forward() output meets receiver TokenCodewordPrior contract."""

    def test_forward_output_is_complex_V_M_N(self):
        cfg = _codebook_config()
        cb = TokenDDCodebook(cfg)
        book = cb.forward()
        self.assertTrue(torch.is_complex(book))
        self.assertEqual(book.ndim, 3)
        V, M, N = cfg.vocab_size, cfg.M, cfg.N
        self.assertEqual(book.shape, (V, M, N))

    def test_receiver_token_codeword_prior_import_and_create(self):
        cfg = _codebook_config()
        cb = TokenDDCodebook(cfg)
        book = cb.forward()
        try:
            receiver_root = ROOT / "Learnable_Receiver_DD-Signals-to-Tokens"
            sys.path.insert(0, str(receiver_root))
            try:
                from receiver.token_prior import TokenCodewordPrior
            except ImportError:
                self.skipTest("TokenCodewordPrior not importable from receiver")
            finally:
                if str(receiver_root) in sys.path:
                    sys.path.remove(str(receiver_root))
            prior = TokenCodewordPrior(codeword_book=book)
            self.assertIsNotNone(prior)
        except ImportError:
            self.skipTest("Receiver package not available")


if __name__ == "__main__":
    unittest.main()
