"""Tests for paired Tier 1 waveform validation of frozen TX artifacts."""

import json
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
TX_ROOT = ROOT / "Learnable_Mapping_Tokens-to-DD-Signals"
sys.path.insert(0, str(TX_ROOT))

from experiments.tx_waveform_tier1 import (
    Tier1WaveformConfig,
    load_tier1_waveform_config,
    oracle_ml_template_argmin,
    run_tx_waveform_tier1,
)
from transmitter import (
    TokenDDCodebook,
    TransmitterConfig,
    export_physical_codeword_book,
    initialize_token_codebook_,
)


def _tiny_tx_config():
    return TransmitterConfig(
        M=4,
        N=4,
        vocab_size=4,
        pilot_delay=2,
        pilot_doppler=2,
        pilot_guard_delay=1,
        pilot_guard_doppler=1,
        pilot_obs_delay_radius=0,
        pilot_obs_doppler_radius=0,
        max_channel_delay=1,
        max_channel_doppler=1,
        data_power=1.0,
        complex_dtype="complex64",
    )


def _tiny_tier1_config(**updates):
    values = {
        "seed": 91,
        "cp_len": 1,
        "sample_rate": 16000.0,
        "num_scenarios": 2,
        "num_tokens_per_scenario": 8,
        "num_separation_pairs": 8,
        "snr_db_values": (20.0, 30.0),
        "num_paths": 1,
        "max_delay": 1,
        "max_doppler": 1,
        "device": "cpu",
    }
    values.update(updates)
    return Tier1WaveformConfig(**values)


def _write_artifact_run(root, *, shaped_seed=None):
    cfg = _tiny_tx_config()
    baseline_codebook = TokenDDCodebook(cfg)
    initialize_token_codebook_(
        baseline_codebook,
        mode="random_phase",
        generator=torch.Generator(device="cpu").manual_seed(7),
    )
    shaped_codebook = baseline_codebook
    if shaped_seed is not None:
        shaped_codebook = TokenDDCodebook(cfg)
        generator = torch.Generator(device="cpu").manual_seed(shaped_seed)
        with torch.no_grad():
            shaped_codebook.raw_real.copy_(
                torch.randn(shaped_codebook.raw_real.shape, generator=generator),
            )
            shaped_codebook.raw_imag.copy_(
                torch.randn(shaped_codebook.raw_imag.shape, generator=generator),
            )
    baseline = root / "baseline_physical_codeword_book.pt"
    shaped = root / "shaped_physical_codeword_book.pt"
    export_physical_codeword_book(baseline_codebook, baseline)
    export_physical_codeword_book(shaped_codebook, shaped)
    manifest = {
        "schema_version": 1,
        "complete": True,
        "run_config": {"target_margin": 0.5},
        "artifact_files": {
            "baseline_physical_codeword_book": baseline.name,
            "shaped_physical_codeword_book": shaped.name,
        },
    }
    with open(root / "manifest.json", "w", encoding="ascii") as handle:
        json.dump(manifest, handle)
    return cfg


def _write_identical_artifact_run(root):
    return _write_artifact_run(root)


def _run_silently(*args, **kwargs):
    with redirect_stdout(io.StringIO()):
        return run_tx_waveform_tier1(*args, **kwargs)


class Tier1ConfigTests(unittest.TestCase):

    def test_loads_json_and_rejects_unknown_field(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tier1.json"
            payload = {
                "schema_version": 1,
                "snr_db_values": [10.0, 20.0],
                "device": "cpu",
            }
            path.write_text(json.dumps(payload), encoding="ascii")
            config = load_tier1_waveform_config(path)
            self.assertEqual(config.snr_db_values, (10.0, 20.0))

            payload["not_a_parameter"] = 1
            path.write_text(json.dumps(payload), encoding="ascii")
            with self.assertRaises(ValueError):
                load_tier1_waveform_config(path)

    def test_rejects_insufficient_cp(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_identical_artifact_run(root)
            with self.assertRaises(ValueError):
                _run_silently(
                    root,
                    root / "tier1",
                    config=_tiny_tier1_config(cp_len=0),
                )


class OracleMLTests(unittest.TestCase):

    def test_exact_templates_recover_token_ids(self):
        templates = torch.tensor([
            [[1.0 + 0.0j, 0.0 + 0.0j]],
            [[0.0 + 0.0j, 1.0 + 0.0j]],
            [[1.0 + 0.0j, 1.0 + 0.0j]],
        ], dtype=torch.complex64)
        ids = torch.tensor([2, 0, 1], dtype=torch.long)
        prediction = oracle_ml_template_argmin(templates[ids], templates)
        self.assertTrue(torch.equal(prediction, ids))


class Tier1TinyRunTests(unittest.TestCase):

    def test_cp_waveform_power_difference_does_not_reject_paired_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_artifact_run(root, shaped_seed=13)
            manifest = _run_silently(
                root,
                root / "tier1",
                config=_tiny_tier1_config(num_scenarios=1),
            )
            artifacts = manifest["artifacts"]
            self.assertAlmostEqual(
                artifacts["baseline"]["average_useful_dd_reference_power"],
                artifacts["shaped"]["average_useful_dd_reference_power"],
                places=6,
            )
            self.assertNotAlmostEqual(
                artifacts["baseline"][
                    "average_transmit_waveform_power_including_cp"
                ],
                artifacts["shaped"][
                    "average_transmit_waveform_power_including_cp"
                ],
                places=6,
            )
            self.assertEqual(
                manifest["noise_convention"]["reference_domain"],
                "pre-CP DD frame / useful OFDM samples",
            )

    def test_identical_artifacts_produce_exact_paired_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_identical_artifact_run(root)
            output = root / "tier1"
            manifest = _run_silently(
                root,
                output,
                config=_tiny_tier1_config(),
            )
            self.assertTrue((output / "manifest.json").exists())
            self.assertTrue((output / "metrics.jsonl").exists())
            self.assertTrue((output / "scenario_bank.pt").exists())
            self.assertTrue(
                manifest["flags"]["paired_baseline_shaped_comparison"],
            )
            self.assertTrue(manifest["flags"]["cp_sufficient"])
            self.assertFalse(
                manifest["flags"]["engineering_receiver_ter_evaluated"],
            )
            self.assertEqual(
                manifest["scenario_audit"]["scenarios_with_duplicate_dd_taps"],
                0,
            )
            for row in manifest["aggregate_metrics"]:
                self.assertEqual(
                    row["artifacts"]["baseline"],
                    row["artifacts"]["shaped"],
                )
                for value in row["paired_delta_shaped_minus_baseline"].values():
                    self.assertEqual(value, 0.0)

    def test_existing_output_manifest_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_identical_artifact_run(root)
            output = root / "tier1"
            _run_silently(
                root,
                output,
                config=_tiny_tier1_config(num_scenarios=1),
            )
            with self.assertRaises(ValueError):
                _run_silently(
                    root,
                    output,
                    config=_tiny_tier1_config(num_scenarios=1),
                )


if __name__ == "__main__":
    unittest.main()
