from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from receiver import (
    LearnableOTFSReceiver,
    ReceiverComplexityReport,
    ReceiverConfig,
    ReceiverPaperTrace,
    build_receiver_paper_trace,
    count_receiver_parameters,
    estimate_receiver_ops,
    named_receiver_paper_ablation,
    receiver_paper_summary,
    summarize_receiver_config,
)
from receiver.token_prior import TokenCodewordPrior


def _config(**kwargs) -> ReceiverConfig:
    values = dict(
        M=5, N=4, vocab_size=11, token_embedding_dim=7,
        num_unfolded_layers=2, topk_paths=3, hidden_channels=6,
        noise_var=0.1, use_refinement_net=True, use_offgrid_refinement=True,
    )
    values.update(kwargs)
    return ReceiverConfig(**values)


def _complex_dd(batch, config):
    real = torch.randn(batch, config.M, config.N)
    imag = torch.randn(batch, config.M, config.N)
    return torch.complex(real, imag)


class SummarizeConfigTests(unittest.TestCase):
    def test_returns_dict_with_basic_keys(self):
        config = _config()
        s = summarize_receiver_config(config)
        self.assertIsInstance(s, dict)
        for k in ("M", "N", "vocab_size", "token_embedding_dim", "topk_paths"):
            self.assertIn(k, s)

    def test_includes_classifier_heads_and_fusion(self):
        config = _config(classifier_num_evidence_heads=3, use_token_logit_fusion=True,
                         token_candidate_pruning_mode="sketch")
        s = summarize_receiver_config(config)
        self.assertEqual(s["classifier_num_evidence_heads"], 3)
        self.assertTrue(s["use_token_logit_fusion"])
        self.assertEqual(s["token_candidate_pruning_mode"], "sketch")


class CountParametersTests(unittest.TestCase):
    def test_total_gt_zero(self):
        config = _config(use_offgrid_refinement=False)
        model = LearnableOTFSReceiver(config)
        report = count_receiver_parameters(model)
        self.assertIsInstance(report, ReceiverComplexityReport)
        self.assertGreater(report.total_parameters, 0)
        self.assertGreater(report.trainable_parameters, 0)

    def test_classifier_parameters_gt_zero(self):
        config = _config()
        model = LearnableOTFSReceiver(config)
        report = count_receiver_parameters(model)
        self.assertGreater(report.classifier_parameters, 0)

    def test_fusion_disabled_zero_params(self):
        config = _config(use_token_logit_fusion=False)
        model = LearnableOTFSReceiver(config)
        report = count_receiver_parameters(model)
        self.assertEqual(report.fusion_parameters, 0)

    def test_fusion_enabled_nonzero_params(self):
        config = _config(use_token_logit_fusion=True)
        model = LearnableOTFSReceiver(config)
        report = count_receiver_parameters(model)
        self.assertGreater(report.fusion_parameters, 0)

    def test_offgrid_refinement_has_params(self):
        config = _config(use_offgrid_refinement=True)
        model = LearnableOTFSReceiver(config)
        report = count_receiver_parameters(model)
        self.assertGreater(report.offgrid_parameters, 0)


class EstimateOpsTests(unittest.TestCase):
    def test_returns_dict_with_keys(self):
        config = _config()
        ops = estimate_receiver_ops(config)
        for k in ("sparse_operator_per_pass", "detector", "classifier", "token_prox"):
            self.assertIn(k, ops, f"missing ops key: {k}")

    def test_candidate_pruning_has_different_ops(self):
        config_full = _config(token_candidate_pruning_mode="none")
        config_prune = _config(token_candidate_pruning_mode="sketch",
                               token_candidate_count=5,
                               token_candidate_sketch_size=10)
        ops_full = estimate_receiver_ops(config_full)
        ops_prune = estimate_receiver_ops(config_prune)
        self.assertNotEqual(ops_full["token_prox"], ops_prune["token_prox"])

    def test_l1_detector_has_zero_token_prox_ops(self):
        config = _config(detector_prox_mode="l1")
        ops = estimate_receiver_ops(config)
        self.assertEqual(ops["token_prox"], 0)

    def test_all_ops_non_negative(self):
        config = _config()
        ops = estimate_receiver_ops(config)
        for k, v in ops.items():
            self.assertGreaterEqual(v, 0, f"ops[{k}] = {v} < 0")


class BuildPaperTraceTests(unittest.TestCase):
    def test_builds_with_full_output(self):
        config = _config(use_offgrid_refinement=False, use_data_consistency_correction=True)
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)
        output = model(y_dd, return_details=True, return_paper_trace=True)

        trace = build_receiver_paper_trace(output, config, ablation_name="test")
        self.assertIsInstance(trace, ReceiverPaperTrace)
        self.assertIn("x_equalized", trace.tensor_shapes)
        self.assertIn("token_logits", trace.tensor_shapes)
        self.assertGreater(len(trace.enabled_modules), 0)
        self.assertGreater(len(trace.algorithm_flags), 0)
        self.assertEqual(trace.ablation_name, "test")

    def test_trace_with_fusion_enabled(self):
        config = _config(
            use_token_logit_fusion=True,
            detector_prox_mode="token_posterior",
            use_offgrid_refinement=False,
        )
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)
        codeword = _complex_dd(config.vocab_size, config)
        prior = TokenCodewordPrior(codeword_book=codeword)
        output = model(y_dd, token_prior=prior, return_details=True)

        trace = build_receiver_paper_trace(output, config)
        self.assertIn("token_logit_fusion_gate_mean", trace.scalar_metrics)


class PaperAblationTests(unittest.TestCase):
    def test_full_unchanged(self):
        config = _config()
        result = named_receiver_paper_ablation("full", config)
        self.assertEqual(result.M, config.M)

    def test_no_offgrid_disables(self):
        config = _config(use_offgrid_refinement=True, dd_operator_mode="auto")
        result = named_receiver_paper_ablation("no_offgrid", config)
        self.assertFalse(result.use_offgrid_refinement)
        self.assertEqual(result.dd_operator_mode, "ongrid")

    def test_no_denoiser_disables(self):
        config = _config(use_denoiser=True)
        result = named_receiver_paper_ablation("no_denoiser", config)
        self.assertFalse(result.use_denoiser)

    def test_single_head_classifier(self):
        config = _config(classifier_num_evidence_heads=4)
        result = named_receiver_paper_ablation("single_head_classifier", config)
        self.assertEqual(result.classifier_num_evidence_heads, 1)

    def test_no_logit_fusion(self):
        config = _config(use_token_logit_fusion=True)
        result = named_receiver_paper_ablation("no_logit_fusion", config)
        self.assertFalse(result.use_token_logit_fusion)

    def test_classifier_only(self):
        config = _config(
            detector_prox_mode="token_posterior",
            use_denoiser=True,
            use_data_consistency_correction=True,
            use_token_logit_fusion=True,
        )
        result = named_receiver_paper_ablation("classifier_only", config)
        self.assertEqual(result.detector_prox_mode, "l1")
        self.assertFalse(result.use_denoiser)
        self.assertFalse(result.use_data_consistency_correction)
        self.assertFalse(result.use_token_logit_fusion)

    def test_no_physics_gain_refinement(self):
        config = _config(use_physics_guided_gain_refinement=True)
        result = named_receiver_paper_ablation("no_physics_gain_refinement", config)
        self.assertFalse(result.use_physics_guided_gain_refinement)

    def test_no_candidate_pruning(self):
        config = _config(token_candidate_pruning_mode="sketch")
        result = named_receiver_paper_ablation("no_candidate_pruning", config)
        self.assertEqual(result.token_candidate_pruning_mode, "none")

    def test_no_reliability_calibration(self):
        config = _config(dc_use_reliability=True, dc_use_uncertainty=True,
                         classifier_use_uncertainty=True, classifier_num_evidence_heads=4)
        result = named_receiver_paper_ablation("no_reliability_calibration", config)
        self.assertTrue(result.dc_use_reliability)
        self.assertTrue(result.classifier_use_uncertainty)
        self.assertEqual(result.classifier_num_evidence_heads, 4)

    def test_unknown_ablation_raises(self):
        with self.assertRaises(ValueError):
            named_receiver_paper_ablation("not_an_ablation", _config())


class PaperSummaryTests(unittest.TestCase):
    def test_returns_dict_with_config_and_complexity(self):
        config = _config(use_offgrid_refinement=False)
        model = LearnableOTFSReceiver(config)
        summary = receiver_paper_summary(model, config)
        self.assertIsInstance(summary, dict)
        self.assertIn("config", summary)
        self.assertIn("complexity", summary)
        self.assertIn("ops", summary)

    def test_with_output_has_trace(self):
        config = _config(use_offgrid_refinement=False)
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)
        output = model(y_dd, return_details=True)
        summary = receiver_paper_summary(model, config, output=output)
        self.assertIn("trace", summary)


class BackwardCompatibilityTests(unittest.TestCase):
    def test_default_model_returns_logits(self):
        config = _config(use_offgrid_refinement=False)
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)
        logits = model(y_dd)
        self.assertEqual(logits.shape, (2, config.vocab_size))

    def test_no_bit_qam_no_transformer_gnn_vae(self):
        config = _config()
        model = LearnableOTFSReceiver(config)
        names = {m.__class__.__name__.lower() for m in model.modules()}
        self.assertFalse(any("transformer" in n for n in names))
        self.assertFalse(any("graph" in n for n in names))
        self.assertFalse(any("vae" in n for n in names))
        self.assertFalse(hasattr(model, "bit_head"))
        self.assertFalse(hasattr(model, "qam_head"))


if __name__ == "__main__":
    unittest.main()
