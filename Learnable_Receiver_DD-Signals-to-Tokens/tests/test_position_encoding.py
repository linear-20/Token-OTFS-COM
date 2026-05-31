from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from receiver.position_encoding import build_dd_position_encoding


class PositionEncodingTests(unittest.TestCase):
    def test_position_encoding_shape_device_dtype(self):
        encoding = build_dd_position_encoding(
            6,
            5,
            7,
            device=torch.device("cpu"),
            dtype=torch.float64,
        )

        self.assertEqual(encoding.shape, (1, 7, 6, 5))
        self.assertEqual(encoding.device.type, "cpu")
        self.assertEqual(encoding.dtype, torch.float64)

    def test_different_dd_positions_have_different_encoding(self):
        encoding = build_dd_position_encoding(6, 5, 6)

        first = encoding[0, :, 0, 0]
        other = encoding[0, :, 3, 4]

        self.assertFalse(torch.allclose(first, other))


if __name__ == "__main__":
    unittest.main()
