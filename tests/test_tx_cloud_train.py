"""Tests for cloud-ready TX-only L_core training orchestration."""

import json
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
TRANSMITTER_ROOT = ROOT / "Learnable_Mapping_Tokens-to-DD-Signals"
sys.path.insert(0, str(TRANSMITTER_ROOT))

import experiments.tx_cloud_train as cloud
from experiments.tx_cloud_train import (
    TXCloudTrainConfig,
    TXCurriculumStage,
    build_causal_integer_shift_set,
    run_tx_cloud_training,
)
from transmitter import (
    TransmitterConfig,
    load_exported_physical_dd_artifact,
)


def _tiny_tx_config():
    return TransmitterConfig(
        M=4, N=4, vocab_size=4,
        pilot_delay=2, pilot_doppler=2,
        pilot_guard_delay=1, pilot_guard_doppler=1,
        pilot_obs_delay_radius=0, pilot_obs_doppler_radius=0,
        max_channel_delay=1, max_channel_doppler=1,
        data_power=1.0,
        complex_dtype="complex64",
    )


def _tiny_run_config(**overrides):
    values = dict(
        seed=7,
        train_steps=6,
        train_pairs_per_step=4,
        validation_pairs=8,
        test_pairs=8,
        eval_every=2,
        checkpoint_every=2,
        train_bank_refresh_every=2,
        num_train_scenarios=2,
        num_validation_scenarios=3,
        num_test_scenarios=3,
        num_paths=2,
        unique_dd_taps_per_scenario=False,
        target_margin=0.5,
        learning_rate=0.01,
        device="cpu",
        curriculum=(
            TXCurriculumStage(until_step=2, max_delay=0, max_doppler=0),
            TXCurriculumStage(until_step=6, max_delay=1, max_doppler=1),
        ),
    )
    values.update(overrides)
    return TXCloudTrainConfig(**values)


class CausalSupportTests(unittest.TestCase):

    def test_causal_delay_and_signed_doppler_support(self):
        support = build_causal_integer_shift_set(
            max_delay=2, max_doppler=1,
        )
        self.assertEqual(support.shifts.shape, (9, 2))
        self.assertTrue((support.shifts[:, 0] >= 0).all())
        self.assertEqual(set(support.shifts[:, 1].tolist()), {-1, 0, 1})
        self.assertTrue(
            ((support.shifts[:, 0] == 0)
             & (support.shifts[:, 1] == 0)).any(),
        )

    def test_rejects_invalid_support(self):
        with self.assertRaises(ValueError):
            build_causal_integer_shift_set(max_delay=-1, max_doppler=1)
        with self.assertRaises(ValueError):
            build_causal_integer_shift_set(
                max_delay=1, max_doppler=1, name="",
            )


class CloudConfigTests(unittest.TestCase):

    def test_rejects_non_monotonic_curriculum(self):
        cfg = _tiny_run_config(curriculum=(
            TXCurriculumStage(until_step=2, max_delay=1, max_doppler=1),
            TXCurriculumStage(until_step=6, max_delay=0, max_doppler=1),
        ))
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                run_tx_cloud_training(
                    _tiny_tx_config(), tmpdir, run_config=cfg,
                )

    def test_rejects_curriculum_beyond_tx_contract(self):
        cfg = _tiny_run_config(curriculum=(
            TXCurriculumStage(until_step=6, max_delay=2, max_doppler=1),
        ))
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(ValueError):
                run_tx_cloud_training(
                    _tiny_tx_config(), tmpdir, run_config=cfg,
                )

    def test_rejects_invalid_scalar_controls(self):
        with self.assertRaises(ValueError):
            TXCloudTrainConfig(train_steps=0)
        with self.assertRaises(ValueError):
            TXCloudTrainConfig(target_margin=float("nan"))
        with self.assertRaises(TypeError):
            TXCloudTrainConfig(curriculum=[])

    def test_loads_locked_tx_profile_and_curriculum_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            tx_path = root / "tx_config.json"
            curriculum_path = root / "curriculum.json"
            with open(tx_path, "w", encoding="ascii") as handle:
                json.dump({
                    "M": 4, "N": 4, "vocab_size": 4,
                    "pilot_delay": 2, "pilot_doppler": 2,
                    "pilot_guard_delay": 1, "pilot_guard_doppler": 1,
                    "pilot_obs_delay_radius": 0,
                    "pilot_obs_doppler_radius": 0,
                    "max_channel_delay": 1, "max_channel_doppler": 1,
                    "data_power": 1.0,
                    "complex_dtype": "complex64",
                }, handle)
            with open(curriculum_path, "w", encoding="ascii") as handle:
                json.dump([
                    {"until_step": 2, "max_delay": 0, "max_doppler": 0},
                    {"until_step": 6, "max_delay": 1, "max_doppler": 1},
                ], handle)
            tx_config = cloud._load_tx_config_json(tx_path)
            curriculum = cloud._load_curriculum_json(curriculum_path)
            self.assertEqual(tx_config, _tiny_tx_config())
            self.assertEqual(curriculum, _tiny_run_config().curriculum)


class TinyCloudRunTests(unittest.TestCase):

    def setUp(self):
        self.tx_config = _tiny_tx_config()
        self.run_config = _tiny_run_config()

    def test_run_emits_artifacts_and_boundary_flags(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            manifest = run_tx_cloud_training(
                self.tx_config, out, run_config=self.run_config,
            )
            expected = (
                "baseline_raw_state.pt",
                "baseline_physical_codeword_book.pt",
                "shaped_raw_state.pt",
                "shaped_physical_codeword_book.pt",
                "validation_scenario_bank.pt",
                "test_scenario_bank.pt",
                "checkpoint_latest.pt",
                "metrics.jsonl",
                "manifest.json",
            )
            for name in expected:
                self.assertTrue((out / name).exists(), f"missing {name}")
            self.assertTrue(manifest["complete"])
            self.assertEqual(manifest["completed_step"], 6)
            flags = manifest["flags"]
            self.assertTrue(flags["surrogate_only"])
            self.assertTrue(flags["causal_integer_delay_support"])
            self.assertFalse(flags["snr_training_applied"])
            self.assertFalse(flags["waveform_evaluation_performed"])
            self.assertFalse(flags["receiver_changed"])
            self.assertFalse(flags["transmitter_architecture_changed"])
            environment = manifest["runtime_environment"]
            self.assertEqual(environment["requested_device"], "cpu")
            self.assertIsNone(environment["gpu_name"])
            self.assertIsNone(environment["gpu_total_memory_bytes"])
            self.assertIn("torch_version", environment)

    def test_final_physical_artifact_passes_strict_loader(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            run_tx_cloud_training(
                self.tx_config, out, run_config=self.run_config,
            )
            payload = load_exported_physical_dd_artifact(
                out / "shaped_physical_codeword_book.pt",
            )
            self.assertEqual(payload["codeword_book"].shape, (4, 4, 4))

    def test_training_banks_refresh_and_held_out_banks_are_independent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            manifest = run_tx_cloud_training(
                self.tx_config, out, run_config=self.run_config,
            )
            self.assertEqual(manifest["train_bank_draw_count"], 3)
            validation = torch.load(
                out / "validation_scenario_bank.pt",
                map_location="cpu", weights_only=True,
            )
            test = torch.load(
                out / "test_scenario_bank.pt",
                map_location="cpu", weights_only=True,
            )
            self.assertTrue((validation["path_shifts"][:, :, 0] >= 0).all())
            self.assertTrue((test["path_shifts"][:, :, 0] >= 0).all())
            self.assertFalse(
                torch.equal(validation["path_gains"], test["path_gains"]),
            )

    def test_resume_matches_uninterrupted_training_exactly(self):
        with tempfile.TemporaryDirectory() as full_tmp, \
             tempfile.TemporaryDirectory() as resumed_tmp:
            full_out = Path(full_tmp)
            resumed_out = Path(resumed_tmp)
            run_tx_cloud_training(
                self.tx_config, full_out, run_config=self.run_config,
            )
            paused = run_tx_cloud_training(
                self.tx_config, resumed_out, run_config=self.run_config,
                stop_after_step=3,
            )
            self.assertFalse(paused["complete"])
            resumed = run_tx_cloud_training(
                self.tx_config, resumed_out, run_config=self.run_config,
                resume_checkpoint=resumed_out / "checkpoint_latest.pt",
            )
            self.assertTrue(resumed["complete"])
            full = load_exported_physical_dd_artifact(
                full_out / "shaped_physical_codeword_book.pt",
            )
            resumed_payload = load_exported_physical_dd_artifact(
                resumed_out / "shaped_physical_codeword_book.pt",
            )
            self.assertTrue(torch.equal(
                full["codeword_book"], resumed_payload["codeword_book"],
            ))
            self.assertEqual(
                full["metadata"]["artifact_sha256"],
                resumed_payload["metadata"]["artifact_sha256"],
            )

    def test_resume_rejects_wrong_output_directory(self):
        with tempfile.TemporaryDirectory() as first_tmp, \
             tempfile.TemporaryDirectory() as second_tmp:
            first_out = Path(first_tmp)
            run_tx_cloud_training(
                self.tx_config, first_out, run_config=self.run_config,
                stop_after_step=2,
            )
            with self.assertRaises(ValueError):
                run_tx_cloud_training(
                    self.tx_config, Path(second_tmp),
                    run_config=self.run_config,
                    resume_checkpoint=first_out / "checkpoint_latest.pt",
                )
            self.assertFalse(
                (Path(second_tmp) / "validation_scenario_bank.pt").exists(),
            )

    def test_existing_run_requires_explicit_resume(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_tx_cloud_training(
                self.tx_config, tmpdir, run_config=self.run_config,
            )
            with self.assertRaises(ValueError):
                run_tx_cloud_training(
                    self.tx_config, tmpdir, run_config=self.run_config,
                )

    def test_metrics_keep_test_split_for_final_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            run_tx_cloud_training(
                self.tx_config, out, run_config=self.run_config,
            )
            with open(out / "metrics.jsonl", encoding="ascii") as handle:
                records = [json.loads(line) for line in handle]
            test_records = [
                record for record in records if record["split"] == "test"
            ]
            self.assertEqual(len(test_records), 1)
            self.assertEqual(test_records[0]["phase"], "final")

    def test_codebook_setup_preserves_global_cpu_rng_state(self):
        torch.manual_seed(123)
        state_before = torch.random.get_rng_state()
        with tempfile.TemporaryDirectory() as tmpdir:
            run_tx_cloud_training(
                self.tx_config, tmpdir, run_config=self.run_config,
            )
        state_after = torch.random.get_rng_state()
        self.assertTrue(torch.equal(state_before, state_after))

    def test_emit_progress_prints_lightweight_training_status(self):
        run_config = _tiny_run_config(
            train_steps=2, progress_every=1,
        )
        stream = io.StringIO()
        with tempfile.TemporaryDirectory() as tmpdir, redirect_stdout(stream):
            run_tx_cloud_training(
                self.tx_config, tmpdir, run_config=run_config,
                emit_progress=True,
            )
        records = [
            json.loads(line) for line in stream.getvalue().splitlines()
        ]
        events = [record["event"] for record in records]
        self.assertIn("training_start", events)
        self.assertIn("baseline_validation_start", events)
        progress = [
            record for record in records
            if record["event"] == "train_progress"
        ]
        self.assertEqual([record["step"] for record in progress], [1, 2])
        for record in progress:
            self.assertIn("latest_train_l_core", record)
            self.assertIn("estimated_remaining_seconds", record)

    def test_ascii_only_and_docstrings(self):
        path = TRANSMITTER_ROOT / "experiments" / "tx_cloud_train.py"
        content = path.read_text(encoding="utf-8")
        for index, character in enumerate(content):
            self.assertLess(
                ord(character), 128,
                f"Non-ASCII U+{ord(character):04X} at offset {index}",
            )
        self.assertIsNotNone(cloud.__doc__)
        self.assertIsNotNone(TXCurriculumStage.__doc__)
        self.assertIsNotNone(TXCloudTrainConfig.__doc__)
        self.assertIsNotNone(run_tx_cloud_training.__doc__)


if __name__ == "__main__":
    unittest.main()
