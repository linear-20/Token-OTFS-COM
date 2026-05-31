from pathlib import Path
import sys
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from receiver import ReceiverConfig, TokenClassifierOutput
from receiver.token_classifier import TokenEmbeddingClassifier, evidence_head_diversity_loss


def _config(**kwargs) -> ReceiverConfig:
    values = dict(
        M=6, N=5, vocab_size=13, token_embedding_dim=8,
        num_unfolded_layers=2, topk_paths=3, hidden_channels=7,
        noise_var=0.1, use_refinement_net=True,
    )
    values.update(kwargs)
    return ReceiverConfig(**values)


def _complex_dd(batch: int, config: ReceiverConfig) -> torch.Tensor:
    real = torch.randn(batch, config.M, config.N)
    imag = torch.randn(batch, config.M, config.N)
    return torch.complex(real, imag)


class TokenClassifierTests(unittest.TestCase):
    def test_logits_shape(self):
        config = _config()
        classifier = TokenEmbeddingClassifier(config)
        x_dd = _complex_dd(4, config)

        logits = classifier(x_dd)

        self.assertEqual(logits.shape, (4, config.vocab_size))
        self.assertTrue(logits.dtype.is_floating_point)

    def test_embedding_and_codebook_shapes(self):
        config = _config()
        classifier = TokenEmbeddingClassifier(config)
        x_dd = _complex_dd(3, config)

        output = classifier(x_dd, return_embedding=True)

        self.assertIsInstance(output, TokenClassifierOutput)
        self.assertEqual(output.rx_embedding.shape, (3, config.token_embedding_dim))
        self.assertEqual(classifier.token_codebook.shape, (config.vocab_size, config.token_embedding_dim))
        self.assertEqual(output.token_logits.shape, (3, config.vocab_size))
        self.assertEqual(output.evidence_weights.shape, (3, config.M, config.N))

    def test_evidence_weights_sum_to_one(self):
        config = _config()
        classifier = TokenEmbeddingClassifier(config)
        x_dd = _complex_dd(4, config)

        output = classifier(x_dd, return_embedding=True)

        self.assertTrue(torch.allclose(output.evidence_weights.sum(dim=(-2, -1)), torch.ones(4), atol=1e-6, rtol=1e-6))

    def test_confidence_map_changes_evidence_weights(self):
        config = _config()
        classifier = TokenEmbeddingClassifier(config)
        x_dd = _complex_dd(1, config)
        support = torch.ones(1, config.M, config.N)
        confidence_uniform = torch.ones(1, config.M, config.N)
        confidence_focused = torch.full((1, config.M, config.N), 1e-3)
        confidence_focused[:, 0, 0] = 1.0

        uniform = classifier(
            x_dd,
            return_embedding=True,
            support_mask=support,
            confidence_map=confidence_uniform,
        )
        focused = classifier(
            x_dd,
            return_embedding=True,
            support_mask=support,
            confidence_map=confidence_focused,
        )

        self.assertFalse(torch.allclose(uniform.evidence_weights, focused.evidence_weights))

    def test_return_local_includes_local_embeddings(self):
        config = _config()
        classifier = TokenEmbeddingClassifier(config)
        x_dd = _complex_dd(2, config)

        output = classifier(x_dd, return_embedding=True, return_local=True)

        self.assertEqual(output.local_embeddings.shape, (2, config.token_embedding_dim, config.M, config.N))

    def test_logits_are_similarity_not_linear_vocab_head(self):
        config = _config()
        classifier = TokenEmbeddingClassifier(config)
        x_dd = _complex_dd(2, config)

        output = classifier(x_dd, return_embedding=True)
        manual_logits = (
            F.normalize(output.rx_embedding, dim=-1)
            @ F.normalize(classifier.token_codebook, dim=-1).transpose(0, 1)
            / classifier.temperature
        )
        linear_vocab_heads = [
            module
            for module in classifier.modules()
            if isinstance(module, nn.Linear) and module.out_features == config.vocab_size
        ]
        adaptive_pooling = [
            module
            for module in classifier.modules()
            if isinstance(module, nn.AdaptiveAvgPool2d)
        ]

        self.assertFalse(hasattr(classifier, "classifier"))
        self.assertEqual(len(linear_vocab_heads), 0)
        self.assertEqual(len(adaptive_pooling), 0)
        self.assertTrue(torch.allclose(output.token_logits, manual_logits, atol=1e-6, rtol=1e-6))

    def test_topk_output_shape(self):
        config = _config()
        classifier = TokenEmbeddingClassifier(config)
        x_dd = _complex_dd(2, config)

        output = classifier(x_dd, return_embedding=True, topk=4)

        self.assertEqual(output.topk_indices.shape, (2, 4))
        self.assertEqual(output.topk_logits.shape, (2, 4))
        self.assertEqual(output.topk_indices.dtype, torch.long)

    def test_temperature_is_positive(self):
        config = _config()
        classifier = TokenEmbeddingClassifier(config)

        self.assertGreater(float(classifier.temperature.item()), 0.0)

    def test_backward_updates_codebook_and_feature_net(self):
        config = _config()
        classifier = TokenEmbeddingClassifier(config)
        x_dd = _complex_dd(3, config)
        token_ids = torch.randint(0, config.vocab_size, (3,))

        logits = classifier(x_dd)
        loss = F.cross_entropy(logits, token_ids)
        loss.backward()

        self.assertIsNotNone(classifier.token_codebook.grad)
        self.assertGreater(float(classifier.token_codebook.grad.abs().sum().item()), 0.0)
        first_conv = classifier.feature_net[0]
        self.assertIsNotNone(first_conv.weight.grad)
        self.assertGreater(float(first_conv.weight.grad.abs().sum().item()), 0.0)


class MultiHeadClassifierTests(unittest.TestCase):
    def test_h1_backward_compat_logits_shape(self):
        config = _config(classifier_num_evidence_heads=1)
        classifier = TokenEmbeddingClassifier(config)
        x_dd = _complex_dd(3, config)
        logits = classifier(x_dd)
        self.assertEqual(logits.shape, (3, config.vocab_size))

    def test_h4_head_shapes(self):
        config = _config(
            classifier_num_evidence_heads=4,
            classifier_return_head_details=True,
        )
        classifier = TokenEmbeddingClassifier(config)
        x_dd = _complex_dd(2, config)
        output = classifier(x_dd, return_embedding=True)

        self.assertEqual(output.token_logits.shape, (2, config.vocab_size))
        self.assertIsNotNone(output.head_embeddings)
        self.assertIsNotNone(output.head_logits)
        self.assertIsNotNone(output.head_evidence_weights)
        self.assertIsNotNone(output.head_fusion_weights)
        self.assertIsNotNone(output.head_evidence_entropy)
        self.assertEqual(output.head_embeddings.shape, (2, 4, config.token_embedding_dim))
        self.assertEqual(output.head_logits.shape, (2, 4, config.vocab_size))
        self.assertEqual(output.head_evidence_weights.shape, (2, 4, config.M, config.N))
        self.assertEqual(output.head_fusion_weights.shape, (2, 4))

    def test_return_head_details_override(self):
        config = _config(
            classifier_num_evidence_heads=3,
            classifier_return_head_details=False,
        )
        classifier = TokenEmbeddingClassifier(config)
        x_dd = _complex_dd(2, config)

        output = classifier(x_dd, return_embedding=True, return_head_details=True)

        self.assertIsNotNone(output.head_embeddings)
        self.assertEqual(output.head_embeddings.shape, (2, 3, config.token_embedding_dim))

    def test_evidence_weights_sum_to_one_per_head(self):
        config = _config(
            classifier_num_evidence_heads=3,
            classifier_return_head_details=True,
        )
        classifier = TokenEmbeddingClassifier(config)
        x_dd = _complex_dd(2, config)
        output = classifier(x_dd, return_embedding=True)

        head_w = output.head_evidence_weights  # [2, 3, M, N]
        sums = head_w.sum(dim=(-2, -1))
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-5))

    def test_data_mask_zero_region_has_near_zero_weight(self):
        config = _config(
            classifier_num_evidence_heads=2,
            classifier_return_head_details=True,
        )
        classifier = TokenEmbeddingClassifier(config)
        x_dd = _complex_dd(1, config)
        data_mask = torch.ones(1, config.M, config.N)
        data_mask[:, 0, :] = 0.0
        output = classifier(x_dd, return_embedding=True, support_mask=data_mask)
        head_w = output.head_evidence_weights  # [1, 2, M, N]
        self.assertLess(float(head_w[:, :, 0, :].abs().sum().item()), 1e-4)

    def test_all_zero_mask_no_nan(self):
        config = _config(
            classifier_num_evidence_heads=2,
            classifier_return_head_details=True,
        )
        classifier = TokenEmbeddingClassifier(config)
        x_dd = _complex_dd(1, config)
        data_mask = torch.zeros(1, config.M, config.N)
        output = classifier(x_dd, return_embedding=True, support_mask=data_mask)
        self.assertTrue(torch.isfinite(output.token_logits).all())

    def test_mean_fusion(self):
        config = _config(classifier_num_evidence_heads=3, classifier_head_fusion="mean",
                         classifier_return_head_details=True)
        classifier = TokenEmbeddingClassifier(config)
        x_dd = _complex_dd(2, config)
        output = classifier(x_dd, return_embedding=True)
        self.assertTrue(torch.allclose(output.head_fusion_weights[0],
                         torch.ones(3) / 3, atol=1e-5))

    def test_learned_static_fusion(self):
        config = _config(classifier_num_evidence_heads=3, classifier_head_fusion="learned_static",
                         classifier_return_head_details=True)
        classifier = TokenEmbeddingClassifier(config)
        x_dd = _complex_dd(2, config)
        output = classifier(x_dd, return_embedding=True)
        sums = output.head_fusion_weights.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones(2), atol=1e-5))

    def test_confidence_fusion(self):
        config = _config(classifier_num_evidence_heads=3, classifier_head_fusion="confidence",
                         classifier_return_head_details=True)
        classifier = TokenEmbeddingClassifier(config)
        x_dd = _complex_dd(2, config)
        output = classifier(x_dd, return_embedding=True)
        self.assertEqual(output.head_fusion_weights.shape, (2, 3))
        self.assertTrue(torch.isfinite(output.head_fusion_weights).all())

    def test_backward_multihead(self):
        config = _config(classifier_num_evidence_heads=3)
        classifier = TokenEmbeddingClassifier(config)
        x_dd = _complex_dd(2, config)
        x_dd.requires_grad_(True)
        logits = classifier(x_dd)
        loss = logits.sum()
        loss.backward()
        self.assertIsNotNone(x_dd.grad)
        self.assertGreater(float(x_dd.grad.abs().sum().item()), 0.0)

    def test_no_linear_vocab_head(self):
        config = _config(classifier_num_evidence_heads=4)
        classifier = TokenEmbeddingClassifier(config)
        linear_heads = [m for m in classifier.modules()
                        if isinstance(m, nn.Linear) and getattr(m, "out_features", None) == config.vocab_size]
        self.assertEqual(len(linear_heads), 0)

    def test_diversity_loss_h1_keeps_graph_safe_zero(self):
        weights = torch.rand(2, 1, 4, 3, requires_grad=True)

        loss = evidence_head_diversity_loss(weights)
        loss.backward()

        self.assertLessEqual(float(loss.item()), 1e-8)
        self.assertIsNotNone(weights.grad)
        self.assertTrue(torch.allclose(weights.grad, torch.zeros_like(weights.grad)))

    def test_diversity_loss_accepts_batched_data_mask(self):
        config = _config(classifier_num_evidence_heads=3)
        weights = torch.rand(2, 3, config.M, config.N)
        weights = weights / weights.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        data_mask = torch.ones(2, config.M, config.N)
        data_mask[:, 0, :] = 0.0

        loss = evidence_head_diversity_loss(weights, data_mask=data_mask)

        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss).item())


if __name__ == "__main__":
    unittest.main()
