from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
TRANSMITTER_ROOT = ROOT / "Learnable_Mapping_Tokens-to-DD-Signals"
sys.path.insert(0, str(TRANSMITTER_ROOT))

from transmitter import TransmitterConfig, TransmitterPilotMasks, build_transmitter_pilot_masks


def _valid_config(**kwargs) -> TransmitterConfig:
    values = dict(
        M=8,
        N=10,
        vocab_size=16,
        pilot_delay=4,
        pilot_doppler=5,
        pilot_guard_delay=3,
        pilot_guard_doppler=3,
        pilot_obs_delay_radius=1,
        pilot_obs_doppler_radius=1,
        data_power=1.0,
        pilot_value_real=1.0,
        pilot_value_imag=0.0,
        max_channel_delay=2,
        max_channel_doppler=2,
        complex_dtype="complex64",
    )
    values.update(kwargs)
    return TransmitterConfig(**values)


class TransmitterConfigValidationTests(unittest.TestCase):
    """Tests for TransmitterConfig __post_init__ validation."""

    def test_valid_config_creates_successfully(self):
        cfg = _valid_config()
        self.assertEqual(cfg.M, 8)
        self.assertEqual(cfg.N, 10)
        self.assertEqual(cfg.vocab_size, 16)

    def test_invalid_M_raises_value_error(self):
        with self.assertRaises(ValueError):
            _valid_config(M=0)
        with self.assertRaises(ValueError):
            _valid_config(M=-1)

    def test_invalid_N_raises_value_error(self):
        with self.assertRaises(ValueError):
            _valid_config(N=0)
        with self.assertRaises(ValueError):
            _valid_config(N=-5)

    def test_invalid_vocab_size_raises_value_error(self):
        with self.assertRaises(ValueError):
            _valid_config(vocab_size=0)
        with self.assertRaises(ValueError):
            _valid_config(vocab_size=-1)

    def test_pilot_delay_out_of_bounds_raises_value_error(self):
        with self.assertRaises(ValueError):
            _valid_config(pilot_delay=-1)
        with self.assertRaises(ValueError):
            _valid_config(pilot_delay=8)  # M=8, must be < 8
        with self.assertRaises(ValueError):
            _valid_config(pilot_delay=100)

    def test_pilot_doppler_out_of_bounds_raises_value_error(self):
        with self.assertRaises(ValueError):
            _valid_config(pilot_doppler=-1)
        with self.assertRaises(ValueError):
            _valid_config(pilot_doppler=10)  # N=10, must be < 10
        with self.assertRaises(ValueError):
            _valid_config(pilot_doppler=100)

    def test_invalid_data_power_raises_value_error(self):
        with self.assertRaises(ValueError):
            _valid_config(data_power=0)
        with self.assertRaises(ValueError):
            _valid_config(data_power=-1.5)
        with self.assertRaises(ValueError):
            _valid_config(data_power=float("nan"))
        with self.assertRaises(ValueError):
            _valid_config(data_power=float("inf"))
        with self.assertRaises(ValueError):
            _valid_config(data_power=True)

    def test_nonfinite_or_bool_pilot_value_raises_value_error(self):
        for field in ["pilot_value_real", "pilot_value_imag"]:
            with self.subTest(field=field, value="nan"):
                with self.assertRaises(ValueError):
                    _valid_config(**{field: float("nan")})
            with self.subTest(field=field, value="inf"):
                with self.assertRaises(ValueError):
                    _valid_config(**{field: float("inf")})
            with self.subTest(field=field, value="bool"):
                with self.assertRaises(ValueError):
                    _valid_config(**{field: True})

    def test_invalid_complex_dtype_raises_value_error(self):
        with self.assertRaises(ValueError):
            _valid_config(complex_dtype="complex32")
        with self.assertRaises(ValueError):
            _valid_config(complex_dtype="float32")
        with self.assertRaises(ValueError):
            _valid_config(complex_dtype="")

    def test_non_negative_int_fields_raise_value_error(self):
        for field in [
            "pilot_guard_delay",
            "pilot_guard_doppler",
            "pilot_obs_delay_radius",
            "pilot_obs_doppler_radius",
            "max_channel_delay",
            "max_channel_doppler",
        ]:
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    _valid_config(**{field: -1})
                with self.assertRaises(ValueError):
                    _valid_config(**{field: 1.5})
                with self.assertRaises(ValueError):
                    _valid_config(**{field: True})

    def test_bool_integer_fields_raise_value_error(self):
        for field in ["M", "N", "vocab_size", "pilot_delay",
                      "pilot_doppler"]:
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    _valid_config(**{field: True})

    def test_guard_delay_less_than_channel_plus_obs_raises_value_error(self):
        with self.assertRaises(ValueError):
            _valid_config(pilot_guard_delay=2, max_channel_delay=1, pilot_obs_delay_radius=2)
        # 2 < 1 + 2 = 3 -> should fail

    def test_guard_doppler_less_than_channel_plus_obs_raises_value_error(self):
        with self.assertRaises(ValueError):
            _valid_config(pilot_guard_doppler=1, max_channel_doppler=1, pilot_obs_doppler_radius=1)
        # 1 < 1 + 1 = 2 -> should fail

    def test_no_wrap_boundary_safety_delay_low_raises_value_error(self):
        # pilot_delay - (max_channel_delay + pilot_obs_delay_radius) < 0
        with self.assertRaises(ValueError):
            _valid_config(pilot_delay=1, max_channel_delay=1, pilot_obs_delay_radius=1)
        # 1 - (1+1) = -1 < 0 -> should fail

    def test_no_wrap_boundary_safety_delay_high_raises_value_error(self):
        # pilot_delay + (max_channel_delay + pilot_obs_delay_radius) >= M
        with self.assertRaises(ValueError):
            _valid_config(M=8, pilot_delay=6, max_channel_delay=1, pilot_obs_delay_radius=1)
        # 6 + (1+1) = 8 >= 8 -> should fail

    def test_no_wrap_boundary_safety_doppler_low_raises_value_error(self):
        # pilot_doppler - (max_channel_doppler + pilot_obs_doppler_radius) < 0
        with self.assertRaises(ValueError):
            _valid_config(pilot_doppler=0, max_channel_doppler=1, pilot_obs_doppler_radius=1)
        # 0 - (1+1) = -2 < 0 -> should fail

    def test_no_wrap_boundary_safety_doppler_high_raises_value_error(self):
        # pilot_doppler + (max_channel_doppler + pilot_obs_doppler_radius) >= N
        with self.assertRaises(ValueError):
            _valid_config(N=10, pilot_doppler=8, max_channel_doppler=1, pilot_obs_doppler_radius=1)
        # 8 + (1+1) = 10 >= 10 -> should fail


class TransmitterConfigPropertiesTests(unittest.TestCase):
    """Tests for TransmitterConfig properties."""

    def test_pilot_value_property(self):
        cfg = _valid_config(pilot_value_real=2.5, pilot_value_imag=-1.0)
        self.assertEqual(cfg.pilot_value, complex(2.5, -1.0))

    def test_pilot_value_default_is_one_plus_zero_j(self):
        cfg = TransmitterConfig(
            M=8, N=10, vocab_size=16,
            pilot_delay=4, pilot_doppler=5,
            pilot_guard_delay=3, pilot_guard_doppler=3,
            pilot_obs_delay_radius=1, pilot_obs_doppler_radius=1,
            max_channel_delay=2, max_channel_doppler=2,
        )
        self.assertEqual(cfg.pilot_value, 1.0 + 0.0j)

    def test_torch_complex_dtype_complex64(self):
        cfg = _valid_config(complex_dtype="complex64")
        self.assertIs(cfg.torch_complex_dtype, torch.complex64)

    def test_torch_complex_dtype_complex128(self):
        cfg = _valid_config(complex_dtype="complex128")
        self.assertIs(cfg.torch_complex_dtype, torch.complex128)

    def test_torch_real_dtype_complex64(self):
        cfg = _valid_config(complex_dtype="complex64")
        self.assertIs(cfg.torch_real_dtype, torch.float32)

    def test_torch_real_dtype_complex128(self):
        cfg = _valid_config(complex_dtype="complex128")
        self.assertIs(cfg.torch_real_dtype, torch.float64)


class TransmitterPilotMasksTests(unittest.TestCase):
    """Tests for build_transmitter_pilot_masks and TransmitterPilotMasks."""

    def test_masks_have_shape_1_M_N(self):
        cfg = _valid_config(M=8, N=10)
        masks = build_transmitter_pilot_masks(cfg)
        self.assertEqual(masks.pilot_mask.shape, (1, 8, 10))
        self.assertEqual(masks.guard_mask.shape, (1, 8, 10))
        self.assertEqual(masks.data_mask.shape, (1, 8, 10))

    def test_pilot_mask_exactly_one_true(self):
        cfg = _valid_config()
        masks = build_transmitter_pilot_masks(cfg)
        self.assertEqual(int(masks.pilot_mask.sum().item()), 1)
        self.assertTrue(masks.pilot_mask[0, cfg.pilot_delay, cfg.pilot_doppler].item())

    def test_masks_are_mutually_exclusive(self):
        cfg = _valid_config()
        masks = build_transmitter_pilot_masks(cfg)
        self.assertFalse(bool((masks.pilot_mask & masks.guard_mask).any()))
        self.assertFalse(bool((masks.pilot_mask & masks.data_mask).any()))
        self.assertFalse(bool((masks.guard_mask & masks.data_mask).any()))

    def test_data_mask_equals_not_pilot_or_guard(self):
        cfg = _valid_config()
        masks = build_transmitter_pilot_masks(cfg)
        expected_data = ~(masks.pilot_mask | masks.guard_mask)
        self.assertTrue(torch.equal(masks.data_mask, expected_data))

    def test_data_mask_has_at_least_one_true(self):
        cfg = _valid_config(M=8, N=10)
        masks = build_transmitter_pilot_masks(cfg)
        self.assertGreater(int(masks.data_mask.sum().item()), 0)

    def test_data_mask_empty_raises_value_error(self):
        cfg = _valid_config(
            M=3, N=3, pilot_delay=1, pilot_doppler=1,
            pilot_guard_delay=2, pilot_guard_doppler=2,
            pilot_obs_delay_radius=0, pilot_obs_doppler_radius=0,
            max_channel_delay=0, max_channel_doppler=0,
        )
        # Guard 2 covers all bins in 3x3 grid -> no data bins
        with self.assertRaises(ValueError):
            build_transmitter_pilot_masks(cfg)

    def test_guard_boundary_clip_no_wrap_around_pilot_at_origin(self):
        # Pilot at (0,0), guard radii 2 -> guard only covers [0:3, 0:3], no wrap
        cfg = _valid_config(
            M=8, N=10,
            pilot_delay=0, pilot_doppler=0,
            pilot_guard_delay=2, pilot_guard_doppler=2,
            pilot_obs_delay_radius=0, pilot_obs_doppler_radius=0,
            max_channel_delay=0, max_channel_doppler=0,
        )
        masks = build_transmitter_pilot_masks(cfg)
        guard = masks.guard_mask[0]
        # Guard should cover delay [0, 2] and Doppler [0, 2], excluding pilot at (0,0)
        self.assertTrue(guard[0, 1].item())   # delay=0, doppler=1
        self.assertTrue(guard[1, 0].item())   # delay=1, doppler=0
        self.assertTrue(guard[2, 2].item())   # delay=2, doppler=2
        # Should NOT wrap to M-1 or N-1
        self.assertFalse(guard[7, 0].item())  # delay=7 is outside guard
        self.assertFalse(guard[0, 9].item())  # doppler=9 is outside guard
        self.assertFalse(guard[7, 9].item())  # far corner

    def test_guard_boundary_clip_at_edges(self):
        # Pilot near bottom-right corner, guard should clip at M-1, N-1
        cfg = _valid_config(
            M=8, N=10,
            pilot_delay=7, pilot_doppler=9,
            pilot_guard_delay=2, pilot_guard_doppler=2,
            pilot_obs_delay_radius=0, pilot_obs_doppler_radius=0,
            max_channel_delay=0, max_channel_doppler=0,
        )
        masks = build_transmitter_pilot_masks(cfg)
        guard = masks.guard_mask[0]
        # Guard should cover delay [5, 7] and Doppler [7, 9], excluding pilot
        self.assertTrue(guard[6, 9].item())   # delay=6, doppler=9
        self.assertTrue(guard[7, 8].item())   # delay=7, doppler=8
        self.assertTrue(guard[5, 7].item())   # delay=5, doppler=7
        # Should NOT wrap to 0
        self.assertFalse(guard[0, 0].item())
        self.assertFalse(guard[7, 0].item())

    def test_guard_mask_excludes_pilot(self):
        cfg = _valid_config()
        masks = build_transmitter_pilot_masks(cfg)
        self.assertFalse(masks.guard_mask[0, cfg.pilot_delay, cfg.pilot_doppler].item())

    def test_all_masks_are_bool_dtype(self):
        cfg = _valid_config()
        masks = build_transmitter_pilot_masks(cfg)
        self.assertIs(masks.pilot_mask.dtype, torch.bool)
        self.assertIs(masks.guard_mask.dtype, torch.bool)
        self.assertIs(masks.data_mask.dtype, torch.bool)

    def test_full_grid_coverage(self):
        cfg = _valid_config()
        masks = build_transmitter_pilot_masks(cfg)
        full = masks.pilot_mask | masks.guard_mask | masks.data_mask
        self.assertTrue(full.all().item())


class TransmitterConfigEdgeCaseTests(unittest.TestCase):
    """Edge-case tests for boundary-valid configs."""

    def test_minimal_valid_config(self):
        cfg = TransmitterConfig(
            M=3, N=3, vocab_size=2,
            pilot_delay=1, pilot_doppler=1,
            pilot_guard_delay=0, pilot_guard_doppler=0,
            pilot_obs_delay_radius=0, pilot_obs_doppler_radius=0,
            max_channel_delay=0, max_channel_doppler=0,
        )
        self.assertEqual(cfg.M, 3)
        masks = build_transmitter_pilot_masks(cfg)
        self.assertEqual(int(masks.pilot_mask.sum().item()), 1)
        self.assertGreater(int(masks.data_mask.sum().item()), 0)

    def test_zero_guard_with_zero_channel_and_obs(self):
        cfg = TransmitterConfig(
            M=4, N=4, vocab_size=4,
            pilot_delay=2, pilot_doppler=2,
            pilot_guard_delay=0, pilot_guard_doppler=0,
            pilot_obs_delay_radius=0, pilot_obs_doppler_radius=0,
            max_channel_delay=0, max_channel_doppler=0,
        )
        masks = build_transmitter_pilot_masks(cfg)
        self.assertEqual(int(masks.guard_mask.sum().item()), 0)

    def test_tight_boundary_safety_exact_fit(self):
        # pilot at 2, max_channel_delay + obs = 2 -> 2-2=0 >= 0, 2+2=4 < 5 (M=5)
        cfg = TransmitterConfig(
            M=5, N=5, vocab_size=4,
            pilot_delay=2, pilot_doppler=2,
            pilot_guard_delay=2, pilot_guard_doppler=2,
            pilot_obs_delay_radius=1, pilot_obs_doppler_radius=1,
            max_channel_delay=1, max_channel_doppler=1,
        )
        self.assertEqual(cfg.M, 5)


if __name__ == "__main__":
    unittest.main()
