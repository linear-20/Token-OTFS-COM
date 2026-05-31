from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
TRANSMITTER_ROOT = ROOT / "Learnable_Mapping_Tokens-to-DD-Signals"
sys.path.insert(0, str(TRANSMITTER_ROOT))

from transmitter import (
    TransmitterConfig,
    TransmitterPilotMasks,
    build_transmitter_pilot_masks,
    insert_transmitter_pilot,
)


def _config(**kwargs) -> TransmitterConfig:
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
        pilot_value_real=2.0,
        pilot_value_imag=-1.0,
        max_channel_delay=2,
        max_channel_doppler=2,
        complex_dtype="complex64",
    )
    values.update(kwargs)
    return TransmitterConfig(**values)


def _codewords(batch: int, config: TransmitterConfig) -> torch.Tensor:
    real = torch.randn(batch, config.M, config.N)
    imag = torch.randn(batch, config.M, config.N)
    return torch.complex(real, imag)


class InsertTransmitterPilotBasicTests(unittest.TestCase):
    """Basic functionality tests for insert_transmitter_pilot."""

    def test_returns_correct_shape(self):
        cfg = _config()
        cw = _codewords(3, cfg)
        x_dd = insert_transmitter_pilot(cw, cfg)
        self.assertEqual(x_dd.shape, (3, cfg.M, cfg.N))

    def test_output_is_complex(self):
        cfg = _config()
        cw = _codewords(2, cfg)
        x_dd = insert_transmitter_pilot(cw, cfg)
        self.assertTrue(torch.is_complex(x_dd))

    def test_output_device_matches_input(self):
        cfg = _config()
        cw = _codewords(2, cfg)
        x_dd = insert_transmitter_pilot(cw, cfg)
        self.assertEqual(x_dd.device, cw.device)

    def test_output_dtype_matches_input(self):
        cfg = _config()
        cw = _codewords(2, cfg)
        x_dd = insert_transmitter_pilot(cw, cfg)
        self.assertEqual(x_dd.dtype, cw.dtype)

    def test_input_codewords_not_modified_in_place(self):
        cfg = _config()
        cw = _codewords(2, cfg)
        cw_before = cw.clone()
        _ = insert_transmitter_pilot(cw, cfg)
        self.assertTrue(torch.equal(cw, cw_before))

    def test_batch_size_one_works(self):
        cfg = _config()
        cw = _codewords(1, cfg)
        x_dd = insert_transmitter_pilot(cw, cfg)
        self.assertEqual(x_dd.shape, (1, cfg.M, cfg.N))


class InsertTransmitterPilotSemanticTests(unittest.TestCase):
    """Pilot/guard/data semantic tests for insert_transmitter_pilot."""

    def test_pilot_bin_set_to_pilot_value(self):
        cfg = _config()
        cw = _codewords(2, cfg)
        x_dd = insert_transmitter_pilot(cw, cfg)
        expected = torch.as_tensor(cfg.pilot_value, device=x_dd.device, dtype=x_dd.dtype)
        actual = x_dd[:, cfg.pilot_delay, cfg.pilot_doppler]
        self.assertTrue(torch.allclose(actual, expected.expand_as(actual)))

    def test_guard_bins_are_zero(self):
        cfg = _config()
        cw = _codewords(2, cfg)
        masks = build_transmitter_pilot_masks(cfg)
        x_dd = insert_transmitter_pilot(cw, cfg, masks=masks)
        guard_2d = masks.guard_mask[0].to(device=x_dd.device)
        self.assertTrue(torch.all(x_dd[:, guard_2d] == 0.0))

    def test_data_bins_preserve_codewords(self):
        cfg = _config()
        cw = _codewords(2, cfg)
        masks = build_transmitter_pilot_masks(cfg)
        x_dd = insert_transmitter_pilot(cw, cfg, masks=masks)
        data_2d = masks.data_mask[0].to(device=x_dd.device)
        self.assertTrue(torch.allclose(x_dd[:, data_2d], cw[:, data_2d]))

    def test_masks_none_auto_builds(self):
        cfg = _config()
        cw = _codewords(2, cfg)
        x_dd_none = insert_transmitter_pilot(cw, cfg, masks=None)
        x_dd_explicit = insert_transmitter_pilot(
            cw, cfg, masks=build_transmitter_pilot_masks(cfg)
        )
        self.assertTrue(torch.equal(x_dd_none, x_dd_explicit))

    def test_explicit_masks_same_result_as_none(self):
        cfg = _config()
        cw = _codewords(3, cfg)
        masks = build_transmitter_pilot_masks(cfg)
        x1 = insert_transmitter_pilot(cw, cfg, masks=None)
        x2 = insert_transmitter_pilot(cw, cfg, masks=masks)
        self.assertTrue(torch.equal(x1, x2))

    def test_no_wrap_around_guard_used(self):
        cfg = _config(
            M=8, N=10,
            pilot_delay=0, pilot_doppler=0,
            pilot_guard_delay=2, pilot_guard_doppler=2,
            pilot_obs_delay_radius=0, pilot_obs_doppler_radius=0,
            max_channel_delay=0, max_channel_doppler=0,
        )
        cw = torch.full((2, cfg.M, cfg.N), 3.0 + 4.0j, dtype=torch.complex64)
        x_dd = insert_transmitter_pilot(cw, cfg)
        # 1. pilot bin set to cfg.pilot_value
        expected_pilot = torch.as_tensor(cfg.pilot_value, device=x_dd.device, dtype=x_dd.dtype)
        actual_pilot = x_dd[:, 0, 0]
        self.assertTrue(torch.allclose(actual_pilot, expected_pilot.expand_as(actual_pilot)))
        # 2. non-pilot guard bins zeroed (guard covers delay [0,3), doppler [0,3) excl pilot)
        masks = build_transmitter_pilot_masks(cfg)
        self.assertTrue(torch.all(x_dd[:, masks.guard_mask[0].to(x_dd.device)] == 0.0))
        # 3. wrap-around positions keep original codeword value
        for pos in [(7, 0), (0, 9), (7, 9)]:
            self.assertTrue(
                torch.allclose(x_dd[:, pos[0], pos[1]], cw[:, pos[0], pos[1]]),
                f"wrap-around position {pos} should keep original value",
            )


class InsertTransmitterPilotCodewordValidationTests(unittest.TestCase):
    """Input validation tests for codewords argument."""

    def test_non_tensor_raises_type_error(self):
        cfg = _config()
        with self.assertRaises(TypeError):
            insert_transmitter_pilot([1.0, 2.0], cfg)  # type: ignore

    def test_non_complex_raises_type_error(self):
        cfg = _config()
        real_tensor = torch.randn(2, cfg.M, cfg.N)
        with self.assertRaises(TypeError):
            insert_transmitter_pilot(real_tensor, cfg)

    def test_wrong_ndim_raises_value_error(self):
        cfg = _config()
        cw2d = torch.complex(
            torch.randn(cfg.M, cfg.N),
            torch.randn(cfg.M, cfg.N),
        )
        with self.assertRaises(ValueError):
            insert_transmitter_pilot(cw2d, cfg)

    def test_zero_batch_raises_value_error(self):
        cfg = _config()
        cw = torch.zeros(0, cfg.M, cfg.N, dtype=torch.complex64)
        with self.assertRaises(ValueError):
            insert_transmitter_pilot(cw, cfg)

    def test_spatial_shape_mismatch_raises_value_error(self):
        cfg = _config()
        cw = torch.complex(
            torch.randn(2, 5, 7),
            torch.randn(2, 5, 7),
        )
        with self.assertRaises(ValueError):
            insert_transmitter_pilot(cw, cfg)


class InsertTransmitterPilotMaskValidationTests(unittest.TestCase):
    """Mask validation tests for externally supplied masks."""

    def test_bad_mask_shape_raises_value_error(self):
        cfg = _config()
        cw = _codewords(2, cfg)
        bad_masks = TransmitterPilotMasks(
            pilot_mask=torch.zeros(2, cfg.M, cfg.N, dtype=torch.bool),
            guard_mask=torch.zeros(1, cfg.M, cfg.N, dtype=torch.bool),
            data_mask=torch.zeros(1, cfg.M, cfg.N, dtype=torch.bool),
        )
        with self.assertRaises(ValueError):
            insert_transmitter_pilot(cw, cfg, masks=bad_masks)

    def test_bad_mask_dtype_raises_type_error(self):
        cfg = _config()
        cw = _codewords(2, cfg)
        bad_masks = TransmitterPilotMasks(
            pilot_mask=torch.zeros(1, cfg.M, cfg.N),
            guard_mask=torch.zeros(1, cfg.M, cfg.N, dtype=torch.bool),
            data_mask=torch.zeros(1, cfg.M, cfg.N, dtype=torch.bool),
        )
        with self.assertRaises(TypeError):
            insert_transmitter_pilot(cw, cfg, masks=bad_masks)

    def test_non_disjoint_pilot_and_guard_raises_value_error(self):
        cfg = _config()
        cw = _codewords(2, cfg)
        pilot = torch.zeros(1, cfg.M, cfg.N, dtype=torch.bool)
        pilot[0, cfg.pilot_delay, cfg.pilot_doppler] = True
        guard = pilot.clone()  # same as pilot
        data = ~(pilot | guard)
        bad_masks = TransmitterPilotMasks(
            pilot_mask=pilot, guard_mask=guard, data_mask=data,
        )
        with self.assertRaises(ValueError):
            insert_transmitter_pilot(cw, cfg, masks=bad_masks)

    def test_non_disjoint_pilot_and_data_raises_value_error(self):
        cfg = _config()
        cw = _codewords(2, cfg)
        pilot = torch.zeros(1, cfg.M, cfg.N, dtype=torch.bool)
        pilot[0, cfg.pilot_delay, cfg.pilot_doppler] = True
        guard = torch.zeros(1, cfg.M, cfg.N, dtype=torch.bool)
        data = pilot.clone()  # same as pilot
        bad_masks = TransmitterPilotMasks(
            pilot_mask=pilot, guard_mask=guard, data_mask=data,
        )
        with self.assertRaises(ValueError):
            insert_transmitter_pilot(cw, cfg, masks=bad_masks)

    def test_data_mask_not_complement_raises_value_error(self):
        cfg = _config()
        cw = _codewords(2, cfg)
        masks = build_transmitter_pilot_masks(cfg)
        # Flip one bit in data_mask
        bad_data = masks.data_mask.clone()
        bad_data[0, 0, 0] = ~bad_data[0, 0, 0]
        bad_masks = TransmitterPilotMasks(
            pilot_mask=masks.pilot_mask,
            guard_mask=masks.guard_mask,
            data_mask=bad_data,
        )
        with self.assertRaises(ValueError):
            insert_transmitter_pilot(cw, cfg, masks=bad_masks)

    def test_data_mask_empty_raises_value_error(self):
        cfg = _config(
            M=3, N=3,
            pilot_delay=1, pilot_doppler=1,
            pilot_guard_delay=0, pilot_guard_doppler=0,
            pilot_obs_delay_radius=0, pilot_obs_doppler_radius=0,
            max_channel_delay=0, max_channel_doppler=0,
        )
        cw = _codewords(2, cfg)
        # Manually construct masks with empty data (guard covers everything but pilot)
        pilot = torch.zeros(1, 3, 3, dtype=torch.bool)
        pilot[0, 1, 1] = True
        guard = ~pilot  # everything except pilot
        data = torch.zeros(1, 3, 3, dtype=torch.bool)  # empty
        bad_masks = TransmitterPilotMasks(
            pilot_mask=pilot, guard_mask=guard, data_mask=data,
        )
        self.assertEqual(int(bad_masks.data_mask.sum().item()), 0)
        with self.assertRaises(ValueError):
            insert_transmitter_pilot(cw, cfg, masks=bad_masks)


class InsertTransmitterPilotStrictMaskValidationTests(unittest.TestCase):
    """Tests for strengthened explicit-mask validation in insert_transmitter_pilot."""

    def test_masks_not_transmitter_pilot_masks_instance_raises_type_error(self):
        cfg = _config()
        cw = _codewords(2, cfg)
        with self.assertRaises(TypeError):
            insert_transmitter_pilot(cw, cfg, masks=(1, 2, 3))  # type: ignore

    def test_explicit_masks_equal_to_build_transmitter_pilot_masks_accepted(self):
        cfg = _config()
        cw = _codewords(2, cfg)
        expected = build_transmitter_pilot_masks(cfg)
        x_dd = insert_transmitter_pilot(cw, cfg, masks=expected)
        self.assertEqual(x_dd.shape, (2, cfg.M, cfg.N))

    def test_pilot_at_wrong_location_raises_value_error(self):
        cfg = _config()
        cw = _codewords(2, cfg)
        expected = build_transmitter_pilot_masks(cfg)
        pilot = expected.pilot_mask.clone()
        guard = expected.guard_mask.clone()
        data = expected.data_mask.clone()
        # Move pilot from correct location to a data-region location
        pilot[0, cfg.pilot_delay, cfg.pilot_doppler] = False
        new_delay = (cfg.pilot_delay + 1) % cfg.M
        new_doppler = (cfg.pilot_doppler + 1) % cfg.N
        # Ensure new location is not in guard
        if guard[0, new_delay, new_doppler].item():
            new_delay = (new_delay + 1) % cfg.M
        pilot[0, new_delay, new_doppler] = True
        data[0, new_delay, new_doppler] = False
        data[0, cfg.pilot_delay, cfg.pilot_doppler] = True
        bad_masks = TransmitterPilotMasks(
            pilot_mask=pilot, guard_mask=guard, data_mask=data,
        )
        with self.assertRaises(ValueError):
            insert_transmitter_pilot(cw, cfg, masks=bad_masks)

    def test_extra_pilot_true_raises_value_error(self):
        cfg = _config()
        cw = _codewords(2, cfg)
        expected = build_transmitter_pilot_masks(cfg)
        pilot = expected.pilot_mask.clone()
        guard = expected.guard_mask.clone()
        data = expected.data_mask.clone()
        # Add a second True to pilot at a data-region location
        data_indices = torch.nonzero(data[0])
        extra_idx = data_indices[0]
        pilot[0, extra_idx[0], extra_idx[1]] = True
        data[0, extra_idx[0], extra_idx[1]] = False
        bad_masks = TransmitterPilotMasks(
            pilot_mask=pilot, guard_mask=guard, data_mask=data,
        )
        with self.assertRaises(ValueError):
            insert_transmitter_pilot(cw, cfg, masks=bad_masks)

    def test_wrap_around_like_guard_raises_value_error(self):
        cfg = _config(
            M=6, N=6,
            pilot_delay=0, pilot_doppler=0,
            pilot_guard_delay=1, pilot_guard_doppler=1,
            pilot_obs_delay_radius=0, pilot_obs_doppler_radius=0,
            max_channel_delay=0, max_channel_doppler=0,
        )
        cw = _codewords(2, cfg)
        expected = build_transmitter_pilot_masks(cfg)
        pilot = expected.pilot_mask.clone()
        guard = expected.guard_mask.clone()
        data = expected.data_mask.clone()
        # Add a guard entry at (M-1, 0) which is a "wrap-around" from delay side
        guard[0, cfg.M - 1, 0] = True
        data[0, cfg.M - 1, 0] = False
        bad_masks = TransmitterPilotMasks(
            pilot_mask=pilot, guard_mask=guard, data_mask=data,
        )
        with self.assertRaises(ValueError):
            insert_transmitter_pilot(cw, cfg, masks=bad_masks)

    def test_wrong_data_mask_same_pilot_guard_raises_value_error(self):
        cfg = _config()
        cw = _codewords(2, cfg)
        expected = build_transmitter_pilot_masks(cfg)
        bad_data = expected.data_mask.clone()
        bad_data[0, 0, 0] = ~bad_data[0, 0, 0]
        bad_masks = TransmitterPilotMasks(
            pilot_mask=expected.pilot_mask.clone(),
            guard_mask=expected.guard_mask.clone(),
            data_mask=bad_data,
        )
        with self.assertRaises(ValueError):
            insert_transmitter_pilot(cw, cfg, masks=bad_masks)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_explicit_masks_passes_strict_validation(self):
        cfg = _config()
        cw = _codewords(2, cfg).cuda()
        expected = build_transmitter_pilot_masks(cfg)
        masks_cuda = TransmitterPilotMasks(
            pilot_mask=expected.pilot_mask.cuda(),
            guard_mask=expected.guard_mask.cuda(),
            data_mask=expected.data_mask.cuda(),
        )
        x_dd = insert_transmitter_pilot(cw, cfg, masks=masks_cuda)
        self.assertEqual(x_dd.device.type, "cuda")
        self.assertTrue(torch.is_complex(x_dd))


class InsertTransmitterPilotCUDATests(unittest.TestCase):
    """CUDA device tests for insert_transmitter_pilot."""

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_codewords_cpu_masks_produces_cuda_output(self):
        cfg = _config()
        cw = _codewords(2, cfg).cuda()
        x_dd = insert_transmitter_pilot(cw, cfg)
        self.assertEqual(x_dd.device.type, "cuda")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_codewords_cpu_masks_correct_semantics(self):
        cfg = _config()
        cw = _codewords(2, cfg).cuda()
        x_dd = insert_transmitter_pilot(cw, cfg)
        masks = build_transmitter_pilot_masks(cfg)
        guard_2d = masks.guard_mask[0].to(device=x_dd.device)
        data_2d = masks.data_mask[0].to(device=x_dd.device)
        self.assertTrue(torch.all(x_dd[:, guard_2d] == 0.0))
        pilot_val = torch.as_tensor(cfg.pilot_value, device=x_dd.device, dtype=x_dd.dtype)
        actual_pilot = x_dd[:, cfg.pilot_delay, cfg.pilot_doppler]
        self.assertTrue(torch.allclose(actual_pilot, pilot_val.expand_as(actual_pilot)))
        self.assertTrue(torch.allclose(x_dd[:, data_2d], cw[:, data_2d]))


if __name__ == "__main__":
    unittest.main()
