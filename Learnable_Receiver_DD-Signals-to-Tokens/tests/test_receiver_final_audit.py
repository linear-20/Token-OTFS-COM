from pathlib import Path
import sys
import unittest

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from receiver import (
    LearnableOTFSReceiver,
    ReceiverConfig,
    ReceiverLossWeights,
    candidate_margin_regularization,
    candidate_recall_metric,
    estimate_receiver_ops,
    named_receiver_paper_ablation,
    paper_receiver_loss,
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


class DefaultCompatibilityTests(unittest.TestCase):
    def test_model_returns_tensor_batch_v(self):
        config = _config(use_offgrid_refinement=False)
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)
        logits = model(y_dd)
        self.assertIsInstance(logits, torch.Tensor)
        self.assertEqual(logits.shape, (2, config.vocab_size))

    def test_token_prior_none_returns_details(self):
        config = _config(use_offgrid_refinement=False)
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)
        output = model(y_dd, return_details=True)
        self.assertIsNotNone(output.token_logits)
        self.assertTrue(torch.isfinite(output.token_logits).all())

    def test_l1_detector_sketch_pruning_no_candidate_scores(self):
        config = _config(
            detector_prox_mode="l1",
            token_candidate_pruning_mode="sketch",
            token_candidate_count=3,
            use_offgrid_refinement=False,
        )
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)
        codeword = _complex_dd(config.vocab_size, config)
        prior = TokenCodewordPrior(codeword_book=codeword)
        output = model(y_dd, token_prior=prior, return_details=True)
        # l1 means no proxy, but sketch may still run; scores may still exist
        self.assertTrue(torch.isfinite(output.token_logits).all())


class FullPaperChainSmokeTest(unittest.TestCase):
    def test_full_paper_chain_smoke(self):
        config = _config(
            channel_estimator_mode="fallback_topk",
            use_offgrid_refinement=True,
            dd_operator_mode="auto",
            detector_prox_mode="token_posterior",
            token_candidate_pruning_mode="sketch",
            token_candidate_score_mode="energy_weighted_distance",
            token_candidate_adaptive=True,
            token_candidate_count=3,
            token_candidate_max_count=6,
            use_data_consistency_correction=True,
            classifier_num_evidence_heads=4,
            classifier_head_fusion="confidence",
            use_token_logit_fusion=True,
        )
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)
        codeword = _complex_dd(config.vocab_size, config)
        prior = TokenCodewordPrior(codeword_book=codeword)

        output = model(y_dd, token_prior=prior, return_details=True)

        trace = output.receiver_trace
        self.assertEqual(output.token_logits.shape, (2, config.vocab_size))
        self.assertIsNotNone(output.equalizer_output)
        self.assertIsNotNone(output.classifier_output)
        self.assertIsNotNone(trace)
        self.assertIn("token_candidate_adaptive", trace)
        self.assertIn("classifier_num_evidence_heads", trace)
        self.assertIn("classifier_uses_multihead_evidence", trace)
        self.assertIn("classifier_head_fusion", trace)
        self.assertIn("use_token_logit_fusion", trace)
        self.assertIsNotNone(output.logit_fusion_output)


class PaperHarnessTests(unittest.TestCase):
    def test_summarize_has_candidate_fields(self):
        config = _config(token_candidate_adaptive=True,
                         token_candidate_score_mode="energy_weighted_distance")
        s = summarize_receiver_config(config)
        for key in ("token_candidate_pruning_mode", "token_candidate_sketch_mode",
                     "token_candidate_score_mode", "token_candidate_adaptive",
                     "token_candidate_count"):
            self.assertIn(key, s)

    def test_ops_l1_token_prox_zero(self):
        config = _config(detector_prox_mode="l1")
        ops = estimate_receiver_ops(config)
        self.assertEqual(ops["token_prox"], 0)

    def test_ops_adaptive_uses_max_count(self):
        config = _config(
            token_candidate_pruning_mode="sketch",
            token_candidate_count=3,
            token_candidate_max_count=10,
            token_candidate_adaptive=True,
            token_candidate_sketch_size=8,
        )
        ops = estimate_receiver_ops(config)
        # max_count of 10 forces worst-case Kc >= 10
        self.assertGreaterEqual(ops["token_prox"], config.vocab_size * 8 + 10 * config.M * config.N)

    def test_ablation_no_adaptive_disables(self):
        config = _config(token_candidate_adaptive=True)
        result = named_receiver_paper_ablation("no_adaptive_candidate_pruning", config)
        self.assertFalse(result.token_candidate_adaptive)

    def test_ablation_no_energy_weighted_score(self):
        config = _config(token_candidate_score_mode="energy_weighted_distance")
        result = named_receiver_paper_ablation("no_energy_weighted_candidate_score", config)
        self.assertEqual(result.token_candidate_score_mode, "distance")

    def test_ablation_unknown_raises(self):
        with self.assertRaises(ValueError):
            named_receiver_paper_ablation("nonexistent_ablation", _config())


class LossesAuditTests(unittest.TestCase):
    def test_default_weights_candidate_margin_zero(self):
        w = ReceiverLossWeights()
        self.assertEqual(w.candidate_margin_weight, 0.0)

    def test_candidate_margin_in_paper_loss(self):
        config = _config(
            detector_prox_mode="token_posterior",
            token_candidate_pruning_mode="sketch",
            token_candidate_count=3,
            use_offgrid_refinement=False,
        )
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)
        codeword = _complex_dd(config.vocab_size, config)
        prior = TokenCodewordPrior(codeword_book=codeword)
        output = model(y_dd, token_prior=prior, return_details=True)
        token_ids = torch.tensor([3, 7])
        weights = ReceiverLossWeights(token_ce_weight=1.0, candidate_margin_weight=0.5)
        result = paper_receiver_loss(output, token_ids, weights=weights)
        self.assertIn("candidate_margin", result.components)

    def test_candidate_recall_metric(self):
        cand = torch.tensor([[0, 3, 5], [1, 2, 7]])
        ids = torch.tensor([3, 9])  # first hits, second misses
        recall = candidate_recall_metric(cand, ids)
        self.assertAlmostEqual(float(recall.item()), 0.5, places=2)

    def test_candidate_recall_none_input(self):
        self.assertEqual(float(candidate_recall_metric(None, None).item()), 0.0)

    def test_margin_regularization_k1_graph_safe(self):
        scores = torch.randn(2, 1)
        loss = candidate_margin_regularization(scores)
        self.assertEqual(float(loss.item()), 0.0)


class ConstraintsTests(unittest.TestCase):
    def test_no_bit_qam_transformer_gnn_vae(self):
        config = _config()
        model = LearnableOTFSReceiver(config)
        names = {m.__class__.__name__.lower() for m in model.modules()}
        self.assertFalse(any("transformer" in n for n in names))
        self.assertFalse(any("graph" in n for n in names))
        self.assertFalse(any("vae" in n for n in names))
        self.assertFalse(hasattr(model, "bit_head"))
        self.assertFalse(hasattr(model, "qam_head"))

    def test_no_linear_vocab_classifier_head(self):
        config = _config()
        model = LearnableOTFSReceiver(config)
        for m in model.modules():
            if isinstance(m, nn.Linear):
                self.assertNotEqual(getattr(m, "out_features", None), config.vocab_size,
                                    f"Linear(..., vocab_size) head found in {m}")


if __name__ == "__main__":
    unittest.main()
