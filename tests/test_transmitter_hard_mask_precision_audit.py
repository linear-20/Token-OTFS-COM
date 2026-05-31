"""Regression tests for hard-mask validation before dtype conversion."""

import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
TRANSMITTER_ROOT = ROOT / "Learnable_Mapping_Tokens-to-DD-Signals"
sys.path.insert(0, str(TRANSMITTER_ROOT))

from transmitter import (
    TokenDDCodebook,
    TransmitterConfig,
    build_transmitter_pilot_masks,
    export_physical_codeword_book,
)
from transmitter.ablations import masked_normalized_dd_correlation
from transmitter.diagnostics import data_codeword_power


def _config():
    return TransmitterConfig(
        M=5,
        N=5,
        vocab_size=2,
        pilot_delay=2,
        pilot_doppler=2,
        pilot_guard_delay=1,
        pilot_guard_doppler=1,
        pilot_obs_delay_radius=0,
        pilot_obs_doppler_radius=0,
        max_channel_delay=1,
        max_channel_doppler=1,
    )


def _fractional_data_mask(config):
    mask = build_transmitter_pilot_masks(config).data_mask.to(torch.float64)
    active_pos = mask[0].nonzero()[0]
    mask[0, active_pos[0], active_pos[1]] = 1e-50
    return mask


class HardMaskPrecisionAuditTests(unittest.TestCase):

    def setUp(self):
        self.config = _config()
        self.mask = _fractional_data_mask(self.config)

    def test_codebook_rejects_float64_fraction_lost_by_float32_rounding(self):
        with self.assertRaises(ValueError):
            TokenDDCodebook(self.config).forward(data_mask=self.mask)

    def test_metrics_reject_float64_fraction_lost_by_float32_rounding(self):
        codeword_book = torch.ones(2, 5, 5, dtype=torch.complex64)
        with self.assertRaises(ValueError):
            data_codeword_power(codeword_book, self.mask)

    def test_orbit_corr_rejects_float64_fraction_lost_by_float32_rounding(self):
        dd = torch.ones(1, 5, 5, dtype=torch.complex64)
        with self.assertRaises(ValueError):
            masked_normalized_dd_correlation(dd, dd, self.mask)

    def test_export_rejects_float64_fraction_lost_by_float32_rounding(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "physical.pt"
            with self.assertRaises(ValueError):
                export_physical_codeword_book(
                    TokenDDCodebook(self.config),
                    path,
                    data_mask=self.mask,
                )


if __name__ == "__main__":
    unittest.main()
