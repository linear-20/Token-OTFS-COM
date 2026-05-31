from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
TRANSMITTER_ROOT = ROOT / "Learnable_Mapping_Tokens-to-DD-Signals"
sys.path.insert(0, str(TRANSMITTER_ROOT))

from transmitter import (
    SparseShiftSet,
    default_integer_shift_set,
)


class SparseShiftSetTests(unittest.TestCase):
    """Tests for SparseShiftSet creation and normalized_weights."""

    def test_valid_shifts_creates_successfully(self):
        shifts = torch.tensor([[0, 1], [1, 0], [-1, 0]], dtype=torch.long)
        ss = SparseShiftSet(shifts=shifts)
        self.assertEqual(ss.shifts.shape, (3, 2))
        self.assertIsNone(ss.weights)

    def test_shifts_bad_dtype_raises(self):
        shifts = torch.tensor([[0, 1], [1, 0]], dtype=torch.float32)
        with self.assertRaises(TypeError):
            SparseShiftSet(shifts=shifts)

    def test_shifts_bad_ndim_raises(self):
        shifts = torch.tensor([0, 1, 2], dtype=torch.long)
        with self.assertRaises(ValueError):
            SparseShiftSet(shifts=shifts)

    def test_shifts_bad_last_dim_raises(self):
        shifts = torch.tensor([[0, 1, 2]], dtype=torch.long)
        with self.assertRaises(ValueError):
            SparseShiftSet(shifts=shifts)

    def test_shifts_empty_raises(self):
        shifts = torch.empty(0, 2, dtype=torch.long)
        with self.assertRaises(ValueError):
            SparseShiftSet(shifts=shifts)

    def test_name_empty_raises(self):
        shifts = torch.tensor([[0, 0]], dtype=torch.long)
        with self.assertRaises(ValueError):
            SparseShiftSet(shifts=shifts, name="")
        with self.assertRaises(ValueError):
            SparseShiftSet(shifts=shifts, name=123)  # type: ignore

    def test_valid_weights_normalized_sum_is_one(self):
        shifts = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        weights = torch.tensor([1.0, 3.0])
        ss = SparseShiftSet(shifts=shifts, weights=weights)
        nw = ss.normalized_weights()
        self.assertTrue(torch.allclose(nw.sum(), torch.tensor(1.0)))

    def test_weights_none_returns_uniform(self):
        shifts = torch.tensor([[0, 1], [1, 0], [2, 0], [0, 2]], dtype=torch.long)
        ss = SparseShiftSet(shifts=shifts)
        nw = ss.normalized_weights()
        self.assertTrue(torch.allclose(nw.sum(), torch.tensor(1.0)))
        self.assertTrue(torch.allclose(nw, torch.full((4,), 0.25)))

    def test_normalized_weights_default_device_and_dtype(self):
        shifts = torch.tensor([[0, 1]], dtype=torch.long)
        weights = torch.tensor([2.0], dtype=torch.float64)
        ss = SparseShiftSet(shifts=shifts, weights=weights)
        nw = ss.normalized_weights()
        self.assertEqual(nw.device, weights.device)
        self.assertEqual(nw.dtype, weights.dtype)

    def test_normalized_weights_uniform_default_device_and_dtype(self):
        shifts = torch.tensor([[0, 1]], dtype=torch.long)
        ss = SparseShiftSet(shifts=shifts)
        nw = ss.normalized_weights()
        self.assertEqual(nw.device, shifts.device)
        self.assertEqual(nw.dtype, torch.float32)

    def test_normalized_weights_explicit_dtype(self):
        shifts = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        ss = SparseShiftSet(shifts=shifts)
        nw = ss.normalized_weights(dtype=torch.float64)
        self.assertEqual(nw.dtype, torch.float64)
        self.assertTrue(torch.allclose(nw.sum(), torch.tensor(1.0, dtype=torch.float64)))

    def test_weights_negative_raises(self):
        shifts = torch.tensor([[0, 0]], dtype=torch.long)
        weights = torch.tensor([-0.5])
        with self.assertRaises(ValueError):
            SparseShiftSet(shifts=shifts, weights=weights)

    def test_weights_nonfinite_raises(self):
        shifts = torch.tensor([[0, 0]], dtype=torch.long)
        weights = torch.tensor([float("inf")])
        with self.assertRaises(ValueError):
            SparseShiftSet(shifts=shifts, weights=weights)
        weights_nan = torch.tensor([float("nan")])
        with self.assertRaises(ValueError):
            SparseShiftSet(shifts=shifts, weights=weights_nan)

    def test_weights_wrong_shape_raises(self):
        shifts = torch.tensor([[0, 1], [1, 0], [2, 0]], dtype=torch.long)
        weights = torch.tensor([1.0, 2.0])  # length 2 != S=3
        with self.assertRaises(ValueError):
            SparseShiftSet(shifts=shifts, weights=weights)

    def test_weights_sum_zero_raises(self):
        shifts = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        weights = torch.tensor([0.0, 0.0])
        with self.assertRaises(ValueError):
            SparseShiftSet(shifts=shifts, weights=weights)

    def test_weights_integer_dtype_raises(self):
        shifts = torch.tensor([[0, 1]], dtype=torch.long)
        weights = torch.tensor([1], dtype=torch.int32)
        with self.assertRaises(TypeError):
            SparseShiftSet(shifts=shifts, weights=weights)

    def test_normalized_weights_does_not_modify_input(self):
        shifts = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        weights = torch.tensor([1.0, 3.0])
        ss = SparseShiftSet(shifts=shifts, weights=weights)
        weights_before = weights.clone()
        _ = ss.normalized_weights()
        self.assertTrue(torch.equal(weights, weights_before))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_normalized_weights_cuda_device(self):
        shifts = torch.tensor([[0, 1], [1, 0]], dtype=torch.long, device="cuda")
        ss = SparseShiftSet(shifts=shifts)
        nw = ss.normalized_weights()
        self.assertEqual(nw.device.type, "cuda")


class DefaultIntegerShiftSetTests(unittest.TestCase):
    """Tests for default_integer_shift_set function."""

    def test_max1_include_zero_false_returns_8_shifts(self):
        ss = default_integer_shift_set(max_delay=1, max_doppler=1, include_zero=False)
        self.assertEqual(ss.shifts.shape, (8, 2))
        self.assertFalse(
            ((ss.shifts[:, 0] == 0) & (ss.shifts[:, 1] == 0)).any().item()
        )

    def test_include_zero_true_includes_origin(self):
        ss = default_integer_shift_set(max_delay=1, max_doppler=1, include_zero=True)
        self.assertEqual(ss.shifts.shape, (9, 2))
        self.assertTrue(
            ((ss.shifts[:, 0] == 0) & (ss.shifts[:, 1] == 0)).any().item()
        )

    def test_max0_include_zero_false_raises(self):
        with self.assertRaises(ValueError):
            default_integer_shift_set(max_delay=0, max_doppler=0, include_zero=False)

    def test_max0_include_zero_true_returns_one_shift(self):
        ss = default_integer_shift_set(max_delay=0, max_doppler=0, include_zero=True)
        self.assertEqual(ss.shifts.shape, (1, 2))
        self.assertTrue(
            ((ss.shifts[:, 0] == 0) & (ss.shifts[:, 1] == 0)).all().item()
        )

    def test_max_delay_negative_raises(self):
        with self.assertRaises(ValueError):
            default_integer_shift_set(max_delay=-1, max_doppler=1)

    def test_max_doppler_negative_raises(self):
        with self.assertRaises(ValueError):
            default_integer_shift_set(max_delay=1, max_doppler=-1)

    def test_include_zero_not_bool_raises(self):
        with self.assertRaises(ValueError):
            default_integer_shift_set(max_delay=1, max_doppler=1, include_zero=1)  # type: ignore

    def test_name_empty_raises(self):
        with self.assertRaises(ValueError):
            default_integer_shift_set(max_delay=1, max_doppler=1, name="")
        with self.assertRaises(ValueError):
            default_integer_shift_set(max_delay=1, max_doppler=1, name=123)  # type: ignore

    def test_output_shifts_dtype_is_long(self):
        ss = default_integer_shift_set(max_delay=2, max_doppler=3)
        self.assertIn(ss.shifts.dtype, (torch.long, torch.int64))

    def test_output_shifts_shape_S_2(self):
        ss = default_integer_shift_set(max_delay=2, max_doppler=1)
        self.assertEqual(ss.shifts.ndim, 2)
        self.assertEqual(ss.shifts.shape[-1], 2)
        self.assertGreater(ss.shifts.shape[0], 0)


class ExportTests(unittest.TestCase):
    """Tests that only scenario-construction shaping symbols remain public."""

    def test_exported_symbols_importable(self):
        from transmitter import (
            SparseShiftSet as S,
            default_integer_shift_set as F,
        )
        self.assertIs(S, SparseShiftSet)
        self.assertIs(F, default_integer_shift_set)

    def test_removed_legacy_placeholders_absent(self):
        import transmitter
        import transmitter.shaping as shaping

        for name in ("SparseAwareShapingConfig", "SparseShapingDiagnostics"):
            self.assertFalse(hasattr(transmitter, name))
            self.assertFalse(hasattr(shaping, name))

    def test_package_root_excludes_optional_ablation_and_diagnostic_symbols(self):
        import transmitter

        excluded = (
            "cross_token_orbit_confusion_loss",
            "mine_cross_token_hard_negatives",
            "peak_to_average_power_ratio",
            "self_shift_orbit_sidelobe_loss",
            "shift_orbit_visibility_loss",
        )
        for name in excluded:
            self.assertNotIn(name, transmitter.__all__)
            self.assertFalse(hasattr(transmitter, name))


if __name__ == "__main__":
    unittest.main()
