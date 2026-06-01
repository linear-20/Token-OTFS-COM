"""Tests for the unified TX-only cloud training entry."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRANSMITTER_ROOT = ROOT / "Learnable_Mapping_Tokens-to-DD-Signals"
sys.path.insert(0, str(TRANSMITTER_ROOT))

import experiments.tx_train_entry as entry
from experiments.tx_train_entry import load_tx_study, run_tx_study


def _write_json(path, payload):
    with open(path, "w", encoding="ascii") as handle:
        json.dump(payload, handle)


def _write_tiny_inputs(root):
    tx_path = root / "tx.json"
    curriculum_path = root / "curriculum.json"
    _write_json(tx_path, {
        "M": 4,
        "N": 4,
        "vocab_size": 4,
        "pilot_delay": 2,
        "pilot_doppler": 2,
        "pilot_guard_delay": 1,
        "pilot_guard_doppler": 1,
        "pilot_obs_delay_radius": 0,
        "pilot_obs_doppler_radius": 0,
        "data_power": 1.0,
        "max_channel_delay": 1,
        "max_channel_doppler": 1,
        "complex_dtype": "complex64",
    })
    _write_json(curriculum_path, [
        {"until_step": 2, "max_delay": 1, "max_doppler": 1},
    ])
    return tx_path, curriculum_path


def _study_payload(root, **updates):
    tx_path, curriculum_path = _write_tiny_inputs(root)
    payload = {
        "schema_version": 1,
        "study_name": "tiny_study",
        "output_root": str(root / "artifacts"),
        "tx_profile_json": str(tx_path),
        "curriculum_json": str(curriculum_path),
        "base_run_config": {
            "device": "cpu",
            "train_steps": 2,
            "train_pairs_per_step": 3,
            "validation_pairs": 4,
            "test_pairs": 4,
            "eval_every": 1,
            "checkpoint_every": 1,
            "progress_every": 1,
            "train_bank_refresh_every": 1,
            "num_train_scenarios": 2,
            "num_validation_scenarios": 2,
            "num_test_scenarios": 2,
            "num_paths": 1,
            "unique_dd_taps_per_scenario": True,
            "target_margin": 0.5,
            "training_risk_aggregation": "mean",
            "tail_cvar_fraction": 0.5,
            "learning_rate": 0.01,
        },
        "runs": [
            {
                "name": "mean_seed7",
                "description": "mean baseline",
                "tags": ["main"],
                "overrides": {"seed": 7},
            },
            {
                "name": "tail_seed8",
                "description": "tail ablation",
                "tags": ["tail"],
                "overrides": {
                    "seed": 8,
                    "training_risk_aggregation": "tail_cvar",
                },
            },
            {
                "name": "no_curriculum",
                "description": "full-support ablation",
                "curriculum_json": None,
                "overrides": {"seed": 9},
            },
        ],
    }
    payload.update(updates)
    path = root / "study.json"
    _write_json(path, payload)
    return path


class StudyLoadTests(unittest.TestCase):

    def test_resolves_explicit_runs_and_overrides(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            study = load_tx_study(_study_payload(Path(tmpdir)))
            self.assertEqual(study.name, "tiny_study")
            self.assertEqual(len(study.runs), 3)
            self.assertEqual(study.runs[0].run_config.seed, 7)
            self.assertEqual(
                study.runs[1].run_config.training_risk_aggregation,
                "tail_cvar",
            )
            self.assertEqual(study.runs[2].run_config.curriculum, ())

    def test_rejects_unknown_override_field(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path = _study_payload(root)
            payload = json.loads(path.read_text(encoding="ascii"))
            payload["runs"][0]["overrides"]["not_a_parameter"] = 1
            _write_json(path, payload)
            with self.assertRaises(ValueError):
                load_tx_study(path)

    def test_rejects_duplicate_run_names(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path = _study_payload(root)
            payload = json.loads(path.read_text(encoding="ascii"))
            payload["runs"][1]["name"] = "mean_seed7"
            _write_json(path, payload)
            with self.assertRaises(ValueError):
                load_tx_study(path)

    def test_rejects_unsafe_study_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path = _study_payload(root)
            payload = json.loads(path.read_text(encoding="ascii"))
            payload["study_name"] = "../bad"
            _write_json(path, payload)
            with self.assertRaises(ValueError):
                load_tx_study(path)


class StudyRunTests(unittest.TestCase):

    def test_dry_run_does_not_create_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            study = load_tx_study(_study_payload(Path(tmpdir)))
            plan = run_tx_study(
                study, only_runs=("tail_seed8",), dry_run=True,
            )
            self.assertEqual(plan["selected_runs"], ["tail_seed8"])
            self.assertFalse(study.output_dir.exists())

    def test_executes_only_selected_run_and_emits_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            study = load_tx_study(_study_payload(Path(tmpdir)))
            result = run_tx_study(
                study, only_runs=("mean_seed7",), emit_progress=False,
            )
            states = {
                item["name"]: item["status"] for item in result["runs"]
            }
            self.assertEqual(states["mean_seed7"], "completed")
            self.assertEqual(states["tail_seed8"], "pending")
            self.assertEqual(states["no_curriculum"], "pending")
            run_dir = study.output_dir / "mean_seed7"
            self.assertTrue((run_dir / "manifest.json").exists())
            self.assertTrue((run_dir / "entry_resolved_config.json").exists())
            self.assertTrue((study.output_dir / "study_plan.json").exists())
            self.assertTrue((study.output_dir / "study_manifest.json").exists())
            self.assertFalse((study.output_dir / "tail_seed8").exists())

    def test_rejects_unknown_selected_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            study = load_tx_study(_study_payload(Path(tmpdir)))
            with self.assertRaises(ValueError):
                run_tx_study(study, only_runs=("missing",), dry_run=True)

    def test_sequential_only_run_invocations_preserve_study_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            study = load_tx_study(_study_payload(Path(tmpdir)))
            first = run_tx_study(
                study, only_runs=("mean_seed7",), emit_progress=False,
            )
            second = run_tx_study(
                study, only_runs=("tail_seed8",), emit_progress=False,
            )
            first_states = {
                item["name"]: item["status"] for item in first["runs"]
            }
            second_states = {
                item["name"]: item["status"] for item in second["runs"]
            }
            self.assertEqual(first_states["mean_seed7"], "completed")
            self.assertEqual(second_states["mean_seed7"], "completed")
            self.assertEqual(second_states["tail_seed8"], "completed")
            self.assertEqual(second_states["no_curriculum"], "pending")

    def test_ascii_only_and_docstrings(self):
        path = TRANSMITTER_ROOT / "experiments" / "tx_train_entry.py"
        content = path.read_text(encoding="utf-8")
        for index, character in enumerate(content):
            self.assertLess(
                ord(character), 128,
                f"Non-ASCII U+{ord(character):04X} at offset {index}",
            )
        self.assertIsNotNone(entry.__doc__)
        self.assertIsNotNone(entry.ResolvedTXRun.__doc__)
        self.assertIsNotNone(entry.ResolvedTXStudy.__doc__)
        self.assertIsNotNone(load_tx_study.__doc__)
        self.assertIsNotNone(run_tx_study.__doc__)


if __name__ == "__main__":
    unittest.main()
