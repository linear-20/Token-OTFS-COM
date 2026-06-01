"""Tests for TX-only empirical CVaR aggregation and tail audit."""

import math
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
TRANSMITTER_ROOT = ROOT / "Learnable_Mapping_Tokens-to-DD-Signals"
sys.path.insert(0, str(TRANSMITTER_ROOT))

from experiments.tail_risk import (
    empirical_tail_cvar_margin_loss,
    summarize_tail_separation_scores,
)
from experiments.tx_cloud_train import (
    TXCloudTrainConfig,
    TXCurriculumStage,
    run_tx_cloud_training,
)
from experiments.tx_tail_audit import audit_tx_tail
from transmitter import TransmitterConfig


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
        seed=9,
        train_steps=2,
        train_pairs_per_step=4,
        validation_pairs=5,
        test_pairs=6,
        eval_every=1,
        checkpoint_every=1,
        progress_every=1,
        train_bank_refresh_every=1,
        num_train_scenarios=2,
        num_validation_scenarios=2,
        num_test_scenarios=3,
        num_paths=2,
        target_margin=0.5,
        training_risk_aggregation="tail_cvar",
        tail_cvar_fraction=0.5,
        learning_rate=0.01,
        device="cpu",
        curriculum=(
            TXCurriculumStage(until_step=2, max_delay=1, max_doppler=1),
        ),
    )
    values.update(overrides)
    return TXCloudTrainConfig(**values)


class TailCVaRTests(unittest.TestCase):

    def test_matches_worst_fraction_mean(self):
        scores = torch.tensor([[0.0, 0.5], [1.0, 2.0]])
        loss = empirical_tail_cvar_margin_loss(
            scores, target_margin=1.0, tail_fraction=0.5,
        )
        self.assertAlmostEqual(loss.item(), (1.0 + 0.25) / 2)

    def test_ceil_retained_count_and_autograd(self):
        scores = torch.tensor(
            [[0.0, 0.5], [0.75, 2.0]], requires_grad=True,
        )
        loss = empirical_tail_cvar_margin_loss(
            scores, target_margin=1.0, tail_fraction=0.26,
        )
        loss.backward()
        self.assertAlmostEqual(loss.item(), (1.0 + 0.25) / 2)
        self.assertIsNotNone(scores.grad)
        self.assertTrue(torch.isfinite(scores.grad).all())

    def test_zero_penalty_keeps_autograd_graph(self):
        scores = torch.tensor([[2.0, 3.0]], requires_grad=True)
        loss = empirical_tail_cvar_margin_loss(
            scores, target_margin=1.0, tail_fraction=0.5,
        )
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        self.assertIsNotNone(scores.grad)

    def test_summary_adds_tail_quantiles_and_cvar(self):
        scores = torch.tensor([[0.0, 0.5], [1.0, 2.0]])
        summary = summarize_tail_separation_scores(
            scores, target_margin=1.0, tail_fraction=0.5,
        )
        self.assertIn("separation_p001", summary)
        self.assertIn("separation_p005", summary)
        self.assertIn("separation_p05", summary)
        self.assertAlmostEqual(summary["eval_tail_cvar_l_core"], 0.625)

    def test_rejects_invalid_tail_fraction(self):
        scores = torch.ones(1, 1)
        for value in (True, 0.0, -0.1, 1.1, float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                empirical_tail_cvar_margin_loss(
                    scores, target_margin=1.0, tail_fraction=value,
                )


class TailAuditTests(unittest.TestCase):

    def test_tail_cvar_training_run_and_audit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            manifest = run_tx_cloud_training(
                _tiny_tx_config(), run_dir,
                run_config=_tiny_run_config(),
            )
            self.assertEqual(
                manifest["flags"]["training_risk_aggregation"],
                "tail_cvar",
            )
            self.assertTrue(
                manifest["flags"]["tail_risk_training_applied"],
            )
            self.assertIn(
                "eval_tail_cvar_l_core", manifest["test_metrics"],
            )
            report = audit_tx_tail(
                run_dir, split="test", artifact="both", top_k=2,
                tail_cvar_fraction=0.5,
            )
            self.assertEqual(report["pair_count"], 6)
            self.assertEqual(report["scenario_count"], 3)
            self.assertEqual(report["scenarios_with_duplicate_dd_taps"], 0)
            self.assertEqual(set(report["artifacts"]), {"baseline", "shaped"})
            for summary in report["artifacts"].values():
                self.assertLessEqual(len(summary["worst_pairs"]), 2)
                self.assertLessEqual(len(summary["worst_scenarios"]), 2)
                self.assertTrue(math.isfinite(summary["pair_outage_max"]))
                self.assertTrue(math.isfinite(summary["scenario_outage_max"]))

    def test_cloud_config_rejects_invalid_risk_controls(self):
        with self.assertRaises(ValueError):
            _tiny_run_config(training_risk_aggregation="bad")
        with self.assertRaises(TypeError):
            _tiny_run_config(training_risk_aggregation=[])
        with self.assertRaises(ValueError):
            _tiny_run_config(tail_cvar_fraction=0.0)

    def test_cloud_config_defaults_to_unique_dd_taps(self):
        self.assertTrue(TXCloudTrainConfig().unique_dd_taps_per_scenario)


if __name__ == "__main__":
    unittest.main()
