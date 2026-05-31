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
    initialize_token_codebook_,
)


def _codebook_config(**kwargs) -> TransmitterConfig:
    values = dict(
        M=8,
        N=10,
        vocab_size=7,
        pilot_delay=4,
        pilot_doppler=5,
        pilot_guard_delay=3,
        pilot_guard_doppler=3,
        pilot_obs_delay_radius=1,
        pilot_obs_doppler_radius=1,
        max_channel_delay=2,
        max_channel_doppler=2,
        data_power=1.5,
        complex_dtype="complex64",
    )
    values.update(kwargs)
    return TransmitterConfig(**values)


def _raw_magnitude(cb: TokenDDCodebook) -> torch.Tensor:
    return (cb.raw_real.pow(2) + cb.raw_imag.pow(2)).sqrt()


class InitializeTokenCodebookBasicTests(unittest.TestCase):
    """Basic API tests for initialize_token_codebook_."""

    def test_returns_same_module(self):
        cfg = _codebook_config()
        cb = TokenDDCodebook(cfg)
        result = initialize_token_codebook_(cb, mode="random_phase")
        self.assertIs(result, cb)

    def test_invalid_module_raises_type_error(self):
        with self.assertRaises(TypeError):
            initialize_token_codebook_(123, mode="random_phase")  # type: ignore

    def test_invalid_mode_raises_value_error(self):
        cfg = _codebook_config()
        cb = TokenDDCodebook(cfg)
        with self.assertRaises(ValueError):
            initialize_token_codebook_(cb, mode="bad_mode")

    def test_invalid_init_scale_zero_raises(self):
        cfg = _codebook_config()
        cb = TokenDDCodebook(cfg)
        with self.assertRaises(ValueError):
            initialize_token_codebook_(cb, mode="random_phase", init_scale=0.0)

    def test_invalid_init_scale_negative_raises(self):
        cfg = _codebook_config()
        cb = TokenDDCodebook(cfg)
        with self.assertRaises(ValueError):
            initialize_token_codebook_(cb, mode="random_phase", init_scale=-1.0)

    def test_invalid_init_scale_nonfinite_raises(self):
        cfg = _codebook_config()
        cb = TokenDDCodebook(cfg)
        with self.assertRaises(ValueError):
            initialize_token_codebook_(cb, mode="random_phase", init_scale=float("inf"))
        with self.assertRaises(ValueError):
            initialize_token_codebook_(cb, mode="random_phase", init_scale=float("nan"))

    def test_invalid_init_scale_bool_raises(self):
        cfg = _codebook_config()
        cb = TokenDDCodebook(cfg)
        with self.assertRaises(TypeError):
            initialize_token_codebook_(cb, mode="random_phase", init_scale=True)

    def test_invalid_init_scale_non_number_raises(self):
        cfg = _codebook_config()
        cb = TokenDDCodebook(cfg)
        with self.assertRaises(TypeError):
            initialize_token_codebook_(cb, mode="random_phase", init_scale="1.0")  # type: ignore


class RandomPhaseTests(unittest.TestCase):
    """Tests for random_phase initializer."""

    def test_raw_params_shape_preserved(self):
        cfg = _codebook_config()
        cb = TokenDDCodebook(cfg)
        initialize_token_codebook_(cb, mode="random_phase")
        V, M, N = cfg.vocab_size, cfg.M, cfg.N
        self.assertEqual(cb.raw_real.shape, (V, M, N))
        self.assertEqual(cb.raw_imag.shape, (V, M, N))

    def test_raw_magnitude_approx_init_scale(self):
        cfg = _codebook_config()
        cb = TokenDDCodebook(cfg)
        initialize_token_codebook_(cb, mode="random_phase", init_scale=2.0)
        mag = _raw_magnitude(cb)
        self.assertTrue(torch.allclose(mag.mean(), torch.tensor(2.0), atol=0.1))

    def test_generator_supported(self):
        cfg = _codebook_config()
        cb = TokenDDCodebook(cfg)
        gen = torch.Generator(device=cb.raw_real.device)
        initialize_token_codebook_(cb, mode="random_phase", generator=gen)
        self.assertEqual(cb.raw_real.shape, (cfg.vocab_size, cfg.M, cfg.N))

    def test_fixed_seed_reproducible(self):
        cfg = _codebook_config()
        cb1 = TokenDDCodebook(cfg)
        cb2 = TokenDDCodebook(cfg)
        gen1 = torch.Generator(device=cb1.raw_real.device).manual_seed(42)
        gen2 = torch.Generator(device=cb2.raw_real.device).manual_seed(42)
        initialize_token_codebook_(cb1, mode="random_phase", generator=gen1)
        initialize_token_codebook_(cb2, mode="random_phase", generator=gen2)
        self.assertTrue(torch.equal(cb1.raw_real, cb2.raw_real))
        self.assertTrue(torch.equal(cb1.raw_imag, cb2.raw_imag))

    def test_different_seed_different_results(self):
        cfg = _codebook_config()
        cb1 = TokenDDCodebook(cfg)
        cb2 = TokenDDCodebook(cfg)
        gen1 = torch.Generator(device=cb1.raw_real.device).manual_seed(42)
        gen2 = torch.Generator(device=cb2.raw_real.device).manual_seed(99)
        initialize_token_codebook_(cb1, mode="random_phase", generator=gen1)
        initialize_token_codebook_(cb2, mode="random_phase", generator=gen2)
        self.assertFalse(torch.equal(cb1.raw_real, cb2.raw_real))


class SeparableDFTPhaseTests(unittest.TestCase):
    """Tests for separable_dft_phase initializer."""

    def test_deterministic_same_result_twice(self):
        cfg = _codebook_config()
        cb1 = TokenDDCodebook(cfg)
        cb2 = TokenDDCodebook(cfg)
        initialize_token_codebook_(cb1, mode="separable_dft_phase")
        initialize_token_codebook_(cb2, mode="separable_dft_phase")
        self.assertTrue(torch.equal(cb1.raw_real, cb2.raw_real))
        self.assertTrue(torch.equal(cb1.raw_imag, cb2.raw_imag))

    def test_raw_magnitude_approx_init_scale(self):
        cfg = _codebook_config()
        cb = TokenDDCodebook(cfg)
        initialize_token_codebook_(cb, mode="separable_dft_phase", init_scale=2.0)
        mag = _raw_magnitude(cb)
        self.assertTrue(torch.allclose(mag.mean(), torch.tensor(2.0), atol=0.01))

    def test_generator_ignored(self):
        cfg = _codebook_config()
        cb1 = TokenDDCodebook(cfg)
        cb2 = TokenDDCodebook(cfg)
        gen = torch.Generator(device=cb1.raw_real.device).manual_seed(0)
        initialize_token_codebook_(cb1, mode="separable_dft_phase")
        initialize_token_codebook_(cb2, mode="separable_dft_phase", generator=gen)
        self.assertTrue(torch.equal(cb1.raw_real, cb2.raw_real))

    def test_config_unchanged(self):
        cfg = _codebook_config()
        cb = TokenDDCodebook(cfg)
        initialize_token_codebook_(cb, mode="separable_dft_phase")
        self.assertIs(cb.config, cfg)


class ChirpPhaseTests(unittest.TestCase):
    """Tests for chirp_phase initializer."""

    def test_deterministic_same_result_twice(self):
        cfg = _codebook_config()
        cb1 = TokenDDCodebook(cfg)
        cb2 = TokenDDCodebook(cfg)
        initialize_token_codebook_(cb1, mode="chirp_phase")
        initialize_token_codebook_(cb2, mode="chirp_phase")
        self.assertTrue(torch.equal(cb1.raw_real, cb2.raw_real))
        self.assertTrue(torch.equal(cb1.raw_imag, cb2.raw_imag))

    def test_raw_magnitude_approx_init_scale(self):
        cfg = _codebook_config()
        cb = TokenDDCodebook(cfg)
        initialize_token_codebook_(cb, mode="chirp_phase", init_scale=2.0)
        mag = _raw_magnitude(cb)
        self.assertTrue(torch.allclose(mag.mean(), torch.tensor(2.0), atol=0.01))

    def test_generator_ignored(self):
        cfg = _codebook_config()
        cb1 = TokenDDCodebook(cfg)
        cb2 = TokenDDCodebook(cfg)
        gen = torch.Generator(device=cb1.raw_real.device).manual_seed(0)
        initialize_token_codebook_(cb1, mode="chirp_phase")
        initialize_token_codebook_(cb2, mode="chirp_phase", generator=gen)
        self.assertTrue(torch.equal(cb1.raw_real, cb2.raw_real))

    def test_config_unchanged(self):
        cfg = _codebook_config()
        cb = TokenDDCodebook(cfg)
        initialize_token_codebook_(cb, mode="chirp_phase")
        self.assertIs(cb.config, cfg)


class ForwardContractTests(unittest.TestCase):
    """Tests forward contract is preserved after initialization."""

    def _init_and_test(self, mode: str):
        cfg = _codebook_config()
        cb = TokenDDCodebook(cfg)
        initialize_token_codebook_(cb, mode=mode)
        return cb, cfg

    def test_forward_returns_complex_V_M_N(self):
        for mode in ["random_phase", "separable_dft_phase", "chirp_phase"]:
            with self.subTest(mode=mode):
                cb, cfg = self._init_and_test(mode)
                book = cb.forward()
                self.assertTrue(torch.is_complex(book))
                self.assertEqual(book.shape, (cfg.vocab_size, cfg.M, cfg.N))

    def test_forward_output_is_complex_dtype(self):
        for mode in ["random_phase", "separable_dft_phase", "chirp_phase"]:
            with self.subTest(mode=mode):
                cb, cfg = self._init_and_test(mode)
                book = cb.forward()
                self.assertEqual(book.dtype, cfg.torch_complex_dtype)

    def test_data_region_power_equals_data_power(self):
        for mode in ["random_phase", "separable_dft_phase", "chirp_phase"]:
            with self.subTest(mode=mode):
                cb, cfg = self._init_and_test(mode)
                masks = build_transmitter_pilot_masks(cfg)
                book = cb.forward(data_mask=masks.data_mask)
                data_2d = masks.data_mask[0]
                for v in range(cfg.vocab_size):
                    avg_power = book[v, data_2d].abs().pow(2).mean()
                    self.assertTrue(
                        torch.allclose(avg_power, torch.tensor(cfg.data_power), atol=0.01),
                        f"mode={mode}, v={v}, power={avg_power.item():.4f}",
                    )

    def test_pilot_guard_are_zero(self):
        for mode in ["random_phase", "separable_dft_phase", "chirp_phase"]:
            with self.subTest(mode=mode):
                cb, cfg = self._init_and_test(mode)
                masks = build_transmitter_pilot_masks(cfg)
                book = cb.forward(data_mask=masks.data_mask)
                guard_2d = masks.guard_mask[0]
                pilot_2d = masks.pilot_mask[0]
                for v in range(cfg.vocab_size):
                    self.assertTrue(torch.all(book[v, guard_2d].abs() < 1e-5))
                    self.assertTrue(torch.all(book[v, pilot_2d].abs() < 1e-5))

    def test_selected_codewords_returns_B_M_N(self):
        for mode in ["random_phase", "separable_dft_phase", "chirp_phase"]:
            with self.subTest(mode=mode):
                cb, cfg = self._init_and_test(mode)
                masks = build_transmitter_pilot_masks(cfg)
                indices = torch.tensor([0, 2, 4], dtype=torch.long)
                sel = cb.selected_codewords(indices, data_mask=masks.data_mask)
                self.assertTrue(torch.is_complex(sel))
                self.assertEqual(sel.shape, (3, cfg.M, cfg.N))

    def test_selected_codewords_equals_forward_indexed(self):
        for mode in ["random_phase", "separable_dft_phase", "chirp_phase"]:
            with self.subTest(mode=mode):
                cb, cfg = self._init_and_test(mode)
                masks = build_transmitter_pilot_masks(cfg)
                book = cb.forward(data_mask=masks.data_mask)
                indices = torch.tensor([1, 3], dtype=torch.long)
                sel = cb.selected_codewords(indices, codeword_book=book)
                self.assertTrue(torch.equal(sel, book[indices]))


if __name__ == "__main__":
    unittest.main()
