"""Tests for Step 19 standalone L_core surrogate shaping harness."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
TRANSMITTER_ROOT = ROOT / "Learnable_Mapping_Tokens-to-DD-Signals"
sys.path.insert(0, str(TRANSMITTER_ROOT))

# Import harness under experiments namespace.
import experiments.step19_lcore_harness as harness
from experiments.step19_lcore_harness import (
    Step19LCoreRunConfig,
    run_step19_lcore_harness,
    summarize_separation_scores,
)

from transmitter import (
    TokenDDCodebook,
    TransmitterConfig,
    build_transmitter_pilot_masks,
    default_integer_shift_set,
    sample_normalized_sparse_multipath_scenario_bank,
    sample_uniform_cross_token_pairs,
)
from transmitter.metrics import data_codeword_power


def _tiny_config():
    return TransmitterConfig(
        M=4, N=4, vocab_size=4,
        pilot_delay=2, pilot_doppler=2,
        pilot_guard_delay=1, pilot_guard_doppler=1,
        pilot_obs_delay_radius=0, pilot_obs_doppler_radius=0,
        max_channel_delay=1, max_channel_doppler=1,
        data_power=1.0,
        complex_dtype="complex64",
    )


def _tiny_shift_set():
    return default_integer_shift_set(max_delay=1, max_doppler=1,
                                     include_zero=True)


def _tiny_run_config(**kw):
    d = dict(seed=42, train_steps=10, train_pairs_per_step=4,
             eval_pairs=16, eval_every=5, num_scenarios=2,
             num_paths=2, target_margin=0.5, learning_rate=0.01,
             device="cpu")
    d.update(kw)
    return Step19LCoreRunConfig(**d)


# ==============================================================================
# A. summarize_separation_scores tests
# ==============================================================================

class SummarizeSeparationTests(unittest.TestCase):

    def test_matches_manual(self):
        s = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        diag = summarize_separation_scores(s, target_margin=0.3)
        flat = s.reshape(-1)
        self.assertAlmostEqual(diag["separation_mean"], flat.mean().item())
        self.assertAlmostEqual(diag["separation_min"], flat.min().item())
        self.assertAlmostEqual(
            diag["separation_p01"],
            torch.quantile(flat, 0.01, interpolation="linear").item(),
        )
        expected_outage = (flat < 0.3).float().mean().item()
        self.assertAlmostEqual(diag["outage_probability"], expected_outage)

    def test_rejects_non_tensor(self):
        with self.assertRaises(TypeError):
            summarize_separation_scores([[0.1]], target_margin=0.5)

    def test_rejects_nonfinite(self):
        s = torch.tensor([[float("nan"), 0.1]])
        with self.assertRaises(ValueError):
            summarize_separation_scores(s, target_margin=0.5)

    def test_rejects_wrong_ndim(self):
        with self.assertRaises(ValueError):
            summarize_separation_scores(torch.tensor([0.1, 0.2]),
                                        target_margin=0.5)

    def test_rejects_complex_and_empty_scores(self):
        with self.assertRaises(TypeError):
            summarize_separation_scores(
                torch.ones(1, 1, dtype=torch.complex64),
                target_margin=0.5,
            )
        with self.assertRaises(ValueError):
            summarize_separation_scores(
                torch.empty(0, 1),
                target_margin=0.5,
            )

    def test_rejects_invalid_target_margin(self):
        with self.assertRaises(TypeError):
            summarize_separation_scores(torch.ones(1, 1),
                                        target_margin=True)
        with self.assertRaises(ValueError):
            summarize_separation_scores(torch.ones(1, 1),
                                        target_margin=float("nan"))
        with self.assertRaises(ValueError):
            summarize_separation_scores(torch.ones(1, 1),
                                        target_margin=-0.1)


# ==============================================================================
# B. RunConfig validation tests
# ==============================================================================

class RunConfigValidationTests(unittest.TestCase):

    def test_valid_default(self):
        cfg = Step19LCoreRunConfig()
        self.assertIsInstance(cfg, Step19LCoreRunConfig)

    def test_rejects_bool_values(self):
        with self.assertRaises(TypeError):
            Step19LCoreRunConfig(train_steps=True)
        with self.assertRaises(TypeError):
            Step19LCoreRunConfig(target_margin=True)
        with self.assertRaises(TypeError):
            Step19LCoreRunConfig(learning_rate=True)

    def test_rejects_negative_target_margin(self):
        with self.assertRaises(ValueError):
            Step19LCoreRunConfig(target_margin=-0.1)

    def test_rejects_zero_learning_rate(self):
        with self.assertRaises(ValueError):
            Step19LCoreRunConfig(learning_rate=0.0)

    def test_rejects_nonfinite_target_margin(self):
        with self.assertRaises(ValueError):
            Step19LCoreRunConfig(target_margin=float("inf"))

    def test_rejects_bad_device(self):
        with self.assertRaises(ValueError):
            Step19LCoreRunConfig(device="not_a_device")

    def test_seed_zero_allowed_and_negative_rejected(self):
        self.assertEqual(Step19LCoreRunConfig(seed=0).seed, 0)
        with self.assertRaises(ValueError):
            Step19LCoreRunConfig(seed=-1)

    def test_rejects_non_string_device(self):
        with self.assertRaises(TypeError):
            Step19LCoreRunConfig(device=0)


# ==============================================================================
# C. Tiny CPU run tests
# ==============================================================================

class TinyCPURunTests(unittest.TestCase):

    def setUp(self):
        self.tx_cfg = _tiny_config()
        self.ss = _tiny_shift_set()
        self.run_cfg = _tiny_run_config()

    def test_creates_all_seven_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            run_step19_lcore_harness(
                self.tx_cfg, self.ss, out, run_config=self.run_cfg,
            )
            expected = [
                "baseline_raw_state.pt",
                "baseline_physical_codeword_book.pt",
                "shaped_raw_state.pt",
                "shaped_physical_codeword_book.pt",
                "scenario_bank.pt",
                "metrics.jsonl",
                "manifest.json",
            ]
            for name in expected:
                self.assertTrue((out / name).exists(),
                                f"missing {name}")

    def test_baseline_and_shaped_hard_pilot_guard_zeros(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            run_step19_lcore_harness(
                self.tx_cfg, self.ss, out, run_config=self.run_cfg,
            )
            masks = build_transmitter_pilot_masks(self.tx_cfg)
            for fname in ("baseline_physical_codeword_book.pt",
                          "shaped_physical_codeword_book.pt"):
                payload = torch.load(str(out / fname), map_location="cpu")
                cw = payload["codeword_book"]
                for v in range(cw.shape[0]):
                    self.assertTrue(
                        torch.all(cw[v, masks.pilot_mask[0]].abs() < 1e-5),
                    )
                    self.assertTrue(
                        torch.all(cw[v, masks.guard_mask[0]].abs() < 1e-5),
                    )

    def test_equal_per_token_data_region_power(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            run_step19_lcore_harness(
                self.tx_cfg, self.ss, out, run_config=self.run_cfg,
            )
            masks = build_transmitter_pilot_masks(self.tx_cfg)
            for fname in ("baseline_physical_codeword_book.pt",
                          "shaped_physical_codeword_book.pt"):
                payload = torch.load(str(out / fname), map_location="cpu")
                cw = payload["codeword_book"]
                dcp = data_codeword_power(cw, masks.data_mask)
                for v in range(cw.shape[0]):
                    self.assertTrue(
                        torch.allclose(
                            dcp[v],
                            torch.tensor(self.tx_cfg.data_power,
                                         device=dcp.device,
                                         dtype=dcp.dtype),
                            atol=0.01,
                        ),
                    )

    def test_metrics_jsonl_has_required_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            run_step19_lcore_harness(
                self.tx_cfg, self.ss, out, run_config=self.run_cfg,
            )
            with open(out / "metrics.jsonl") as f:
                records = [json.loads(line) for line in f]
            self.assertGreaterEqual(len(records), 3)
            self.assertEqual(records[0]["phase"], "baseline")
            self.assertEqual(records[-1]["phase"], "final")
            for rec in records:
                for key in ("eval_l_core", "separation_mean",
                            "separation_min", "separation_p01",
                            "outage_probability", "data_power_min",
                            "data_power_max", "elapsed_seconds"):
                    self.assertIn(key, rec, f"missing {key}")

    def test_manifest_flags(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            run_step19_lcore_harness(
                self.tx_cfg, self.ss, out, run_config=self.run_cfg,
            )
            with open(out / "manifest.json") as f:
                manifest = json.load(f)
            flags = manifest["flags"]
            self.assertTrue(flags["surrogate_only"])
            self.assertFalse(flags["waveform_evaluation_performed"])
            self.assertFalse(flags["ter_evaluated"])
            self.assertFalse(flags["transmitter_architecture_changed"])

    def test_scenario_bank_weights_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            run_step19_lcore_harness(
                self.tx_cfg, self.ss, out, run_config=self.run_cfg,
            )
            sb = torch.load(str(out / "scenario_bank.pt"), map_location="cpu")
            self.assertTrue(sb["scenario_weights_is_none"])
            self.assertIsNone(sb["scenario_weights"])

    def test_manifest_final_metrics_match_final_record_when_schedule_is_uneven(self):
        run_cfg = _tiny_run_config(train_steps=6, eval_every=5)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            manifest = run_step19_lcore_harness(
                self.tx_cfg, self.ss, out, run_config=run_cfg,
            )
            with open(out / "metrics.jsonl") as f:
                records = [json.loads(line) for line in f]
            self.assertEqual(records[-1]["phase"], "final")
            self.assertEqual(records[-1]["step"], 6)
            for key in ("eval_l_core", "separation_mean",
                        "separation_min", "separation_p01",
                        "outage_probability"):
                self.assertEqual(manifest["final_metrics"][key],
                                 records[-1][key])

    def test_deterministic_same_seed(self):
        with tempfile.TemporaryDirectory() as d1, \
             tempfile.TemporaryDirectory() as d2:
            out1 = Path(d1)
            out2 = Path(d2)
            m1 = run_step19_lcore_harness(
                self.tx_cfg, self.ss, out1, run_config=self.run_cfg,
            )
            m2 = run_step19_lcore_harness(
                self.tx_cfg, self.ss, out2, run_config=self.run_cfg,
            )
            for fname in ("baseline_physical_codeword_book.pt",
                          "shaped_physical_codeword_book.pt"):
                p1 = torch.load(str(out1 / fname), map_location="cpu")
                p2 = torch.load(str(out2 / fname), map_location="cpu")
                self.assertTrue(
                    torch.equal(p1["codeword_book"], p2["codeword_book"]),
                    f"{fname} differs",
                )
            # Metrics excluding elapsed_seconds
            for key in ("eval_l_core", "separation_mean",
                        "separation_min", "separation_p01",
                        "outage_probability"):
                self.assertEqual(
                    m1["baseline_metrics"][key],
                    m2["baseline_metrics"][key],
                )
                self.assertEqual(
                    m1["final_metrics"][key],
                    m2["final_metrics"][key],
                )
            # Manifest flags match
            for k in m1["flags"]:
                self.assertEqual(m1["flags"][k], m2["flags"][k])

    def test_scenario_bank_sampler_called_once(self):
        from unittest.mock import patch
        target = ("experiments.step19_lcore_harness."
                  "sample_normalized_sparse_multipath_scenario_bank")
        with patch(target, wraps=sample_normalized_sparse_multipath_scenario_bank
                   ) as mock_sb:
            with tempfile.TemporaryDirectory() as tmpdir:
                run_step19_lcore_harness(
                    self.tx_cfg, self.ss, Path(tmpdir),
                    run_config=self.run_cfg,
                )
            self.assertEqual(mock_sb.call_count, 1)

    def test_training_pair_sampling_called_per_step(self):
        from unittest.mock import patch
        target = ("experiments.step19_lcore_harness."
                  "sample_uniform_cross_token_pairs")
        with patch(target, wraps=sample_uniform_cross_token_pairs
                   ) as mock_sp:
            with tempfile.TemporaryDirectory() as tmpdir:
                run_step19_lcore_harness(
                    self.tx_cfg, self.ss, Path(tmpdir),
                    run_config=self.run_cfg,
                )
            # Called per training step + once for eval pairs = 10+1 = 11
            self.assertEqual(mock_sp.call_count,
                             self.run_cfg.train_steps + 1)

    def test_training_updates_codebook_parameters(self):
        """Verify gradients flow and parameters change through training."""
        tx_cfg = TransmitterConfig(
            M=4, N=4, vocab_size=4,
            pilot_delay=2, pilot_doppler=2,
            pilot_guard_delay=1, pilot_guard_doppler=1,
            pilot_obs_delay_radius=0, pilot_obs_doppler_radius=0,
            max_channel_delay=1, max_channel_doppler=1,
            data_power=1.0, complex_dtype="complex64",
        )
        ss = default_integer_shift_set(max_delay=1, max_doppler=1,
                                       include_zero=True)
        run_cfg = _tiny_run_config(target_margin=10.0, train_steps=5)
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            run_step19_lcore_harness(tx_cfg, ss, out, run_config=run_cfg)
            baseline = torch.load(
                str(out / "baseline_raw_state.pt"), map_location="cpu",
            )
            shaped = torch.load(
                str(out / "shaped_raw_state.pt"), map_location="cpu",
            )
            self.assertFalse(
                torch.equal(baseline["state_dict"]["raw_real"],
                            shaped["state_dict"]["raw_real"]),
                "raw_real should change through training",
            )

    def test_ascii_only_and_docstrings(self):
        path = (TRANSMITTER_ROOT / "experiments"
                / "step19_lcore_harness.py")
        content = path.read_text(encoding="utf-8")
        for i, ch in enumerate(content):
            self.assertTrue(ord(ch) < 128,
                            f"Non-ASCII U+{ord(ch):04X} at offset {i}")
        self.assertIsNotNone(harness.__doc__)
        self.assertIsNotNone(Step19LCoreRunConfig.__doc__)
        self.assertIsNotNone(summarize_separation_scores.__doc__)
        self.assertIsNotNone(run_step19_lcore_harness.__doc__)

    def test_run_rejects_invalid_typed_inputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(TypeError):
                run_step19_lcore_harness(
                    "bad", self.ss, Path(tmpdir), run_config=self.run_cfg,
                )
            with self.assertRaises(TypeError):
                run_step19_lcore_harness(
                    self.tx_cfg, "bad", Path(tmpdir),
                    run_config=self.run_cfg,
                )
            with self.assertRaises(TypeError):
                run_step19_lcore_harness(
                    self.tx_cfg, self.ss, Path(tmpdir), run_config="bad",
                )

    def test_codebook_setup_preserves_global_cpu_rng_state(self):
        torch.manual_seed(123)
        state_before = torch.random.get_rng_state()
        with tempfile.TemporaryDirectory() as tmpdir:
            run_step19_lcore_harness(
                self.tx_cfg, self.ss, Path(tmpdir), run_config=self.run_cfg,
            )
        state_after = torch.random.get_rng_state()
        self.assertTrue(torch.equal(state_before, state_after))


if __name__ == "__main__":
    unittest.main()
