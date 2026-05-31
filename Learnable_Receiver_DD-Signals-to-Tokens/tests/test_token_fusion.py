from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from receiver import (
    DetectorClassifierLogitFusion,
    LearnableOTFSReceiver,
    ReceiverConfig,
    ReceiverLossWeights,
    TokenLogitFusionOutput,
    expand_candidate_logits_to_vocab,
    paper_receiver_loss,
    token_logit_fusion_consistency_loss,
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


class ExpandCandidateLogitsTests(unittest.TestCase):
    def test_topk_to_full_vocab_shape(self):
        cand = torch.tensor([[0.5, 0.3, 0.1], [0.2, 0.4, 0.6]])
        indices = torch.tensor([[0, 3, 7], [1, 5, 9]])
        full = expand_candidate_logits_to_vocab(cand, indices, 10)
        self.assertEqual(full.shape, (2, 10))

    def test_unselected_positions_are_zero(self):
        cand = torch.tensor([[1.0, 2.0]])
        indices = torch.tensor([[0, 5]])
        full = expand_candidate_logits_to_vocab(cand, indices, 7)
        self.assertEqual(float(full[0, 0].item()), 1.0)
        self.assertEqual(float(full[0, 5].item()), 2.0)
        self.assertEqual(float(full[0, 1].item()), 0.0)

    def test_no_inf_values(self):
        cand = torch.randn(2, 4)
        indices = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])
        full = expand_candidate_logits_to_vocab(cand, indices, 10)
        self.assertTrue(torch.isfinite(full).all())

    def test_invalid_candidate_index_raises(self):
        cand = torch.randn(1, 2)
        indices = torch.tensor([[0, 99]])
        with self.assertRaises(ValueError):
            expand_candidate_logits_to_vocab(cand, indices, 10)


class FusionModuleTests(unittest.TestCase):
    def test_disabled_mode_returns_classifier_logits(self):
        module = DetectorClassifierLogitFusion(fusion_mode="disabled")
        c_logits = torch.randn(2, 10)
        det = torch.randn(2, 10)
        out = module(c_logits, detector_logits=det)
        self.assertTrue(torch.equal(out.fused_logits, c_logits))

    def test_static_mode_shape(self):
        module = DetectorClassifierLogitFusion(fusion_mode="static", init_gate=0.1)
        c_logits = torch.randn(2, 10)
        det = torch.randn(2, 10)
        out = module(c_logits, detector_logits=det)
        self.assertEqual(out.fused_logits.shape, (2, 10))
        self.assertIsNotNone(out.fusion_gate)
        self.assertEqual(out.fusion_gate.shape, (2, 1))

    def test_zero_init_gate_is_nearly_zero(self):
        module = DetectorClassifierLogitFusion(fusion_mode="static", init_gate=0.0)
        self.assertLess(float(module.base_gate.item()), 1e-5)

    def test_static_init_gate_near_zero_yields_approx_classifier(self):
        module = DetectorClassifierLogitFusion(fusion_mode="static", init_gate=0.0, scale=0.01)
        c_logits = torch.randn(2, 10)
        det = torch.randn(2, 10)
        out = module(c_logits, detector_logits=det)
        diff = (out.fused_logits - c_logits).abs().mean()
        self.assertLess(float(diff.item()), 0.1)

    def test_reliability_mode_runs(self):
        module = DetectorClassifierLogitFusion(fusion_mode="reliability")
        c_logits = torch.randn(2, 10)
        det = torch.randn(2, 10)
        c_conf = torch.rand(2)
        c_ent = torch.rand(2)
        p_conf = torch.rand(2)
        p_ent = torch.rand(2)
        rel = torch.rand(2)
        out = module(c_logits, detector_logits=det,
                     classifier_confidence=c_conf, classifier_entropy=c_ent,
                     posterior_confidence=p_conf, posterior_entropy=p_ent,
                     reliability_scalar=rel)
        self.assertEqual(out.fused_logits.shape, (2, 10))
        self.assertTrue(torch.isfinite(out.fused_logits).all())
        self.assertTrue(torch.isfinite(out.fusion_gate).all())

    def test_no_detector_logits_returns_classifier(self):
        module = DetectorClassifierLogitFusion(fusion_mode="static")
        c_logits = torch.randn(2, 10)
        out = module(c_logits)
        self.assertTrue(torch.equal(out.fused_logits, c_logits))

    def test_topk_detector_expands(self):
        module = DetectorClassifierLogitFusion(fusion_mode="static", init_gate=1.0, scale=1.0,
                                               center_detector=False)
        c_logits = torch.zeros(1, 10)
        det_logits = torch.tensor([[5.0, 3.0, 1.0]])
        det_indices = torch.tensor([[0, 3, 7]])
        out = module(c_logits, detector_logits=det_logits, detector_candidate_indices=det_indices)
        self.assertEqual(out.detector_logits.shape, (1, 10))
        self.assertTrue(torch.isfinite(out.fused_logits).all())

    def test_backward_to_gate(self):
        module = DetectorClassifierLogitFusion(fusion_mode="static", init_gate=0.5)
        c_logits = torch.randn(2, 10)
        det = torch.randn(2, 10)
        out = module(c_logits, detector_logits=det)
        loss = out.fused_logits.sum()
        loss.backward()
        self.assertIsNotNone(module.raw_gate.grad)


class FusionConsistencyLossTests(unittest.TestCase):
    def test_scalar_finite(self):
        c_logits = torch.randn(4, 10)
        d_logits = torch.randn(4, 10)
        loss = token_logit_fusion_consistency_loss(c_logits, d_logits)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))

    def test_identical_logits_near_zero(self):
        logits = torch.randn(4, 10)
        loss = token_logit_fusion_consistency_loss(logits, logits.clone())
        self.assertLess(float(loss.item()), 1e-4)


class ModelIntegrationTests(unittest.TestCase):
    def test_default_no_fusion_logits_shape(self):
        config = _config(use_token_logit_fusion=False)
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)
        logits = model(y_dd)
        self.assertEqual(logits.shape, (2, config.vocab_size))

    def test_fusion_enabled_no_prior_safe_fallback(self):
        config = _config(use_token_logit_fusion=True, token_logit_fusion_mode="static",
                         token_logit_fusion_init_gate=0.0,
                         use_offgrid_refinement=False)
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)
        logits = model(y_dd)
        self.assertTrue(torch.isfinite(logits).all())

    def test_fusion_with_detector_posterior(self):
        config = _config(
            use_token_logit_fusion=True, token_logit_fusion_mode="static",
            token_logit_fusion_init_gate=0.1,
            detector_prox_mode="token_posterior",
            use_offgrid_refinement=False,
        )
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)
        codeword = _complex_dd(config.vocab_size, config)
        prior = TokenCodewordPrior(codeword_book=codeword)

        output = model(y_dd, token_prior=prior, return_details=True)

        self.assertIsNotNone(output.logit_fusion_output)
        self.assertIsNotNone(output.classifier_token_logits)
        self.assertTrue(output.receiver_trace["use_token_logit_fusion"])
        self.assertEqual(output.token_logits.shape, (2, config.vocab_size))

    def test_fusion_with_topk_detector_posterior(self):
        config = _config(
            use_token_logit_fusion=True, token_logit_fusion_mode="reliability",
            detector_prox_mode="token_posterior",
            token_candidate_pruning_mode="sketch",
            token_candidate_count=5,
            token_candidate_sketch_size=10,
            use_offgrid_refinement=False,
        )
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)
        codeword = _complex_dd(config.vocab_size, config)
        prior = TokenCodewordPrior(codeword_book=codeword)

        output = model(y_dd, token_prior=prior, return_details=True)

        fusion = output.logit_fusion_output
        self.assertIsNotNone(fusion)
        self.assertIsNotNone(fusion.detector_logits)
        self.assertEqual(fusion.detector_logits.shape, (2, config.vocab_size))
        self.assertTrue(torch.isfinite(output.token_logits).all())

    def test_reliability_fusion_uses_detector_even_without_residual_denoiser_path(self):
        config = _config(
            use_token_logit_fusion=True,
            token_logit_fusion_mode="reliability",
            detector_prox_mode="token_posterior",
            use_equalizer_residual_for_denoiser=False,
            use_offgrid_refinement=False,
        )
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)
        prior = TokenCodewordPrior(codeword_book=_complex_dd(config.vocab_size, config))

        output = model(y_dd, token_prior=prior, return_details=True)

        self.assertIsNotNone(output.logit_fusion_output)
        self.assertIsNotNone(output.logit_fusion_output.detector_logits)
        self.assertIsNotNone(output.logit_fusion_output.fusion_gate)
        self.assertGreater(float(output.logit_fusion_output.fusion_gate.mean().item()), 1e-4)

    def test_receiver_trace_includes_fusion_fields(self):
        config = _config(
            use_token_logit_fusion=True, token_logit_fusion_mode="static",
            detector_prox_mode="token_posterior",
            use_offgrid_refinement=False,
        )
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)
        codeword = _complex_dd(config.vocab_size, config)
        prior = TokenCodewordPrior(codeword_book=codeword)

        output = model(y_dd, token_prior=prior, return_details=True)

        trace = output.receiver_trace
        self.assertTrue(trace["use_token_logit_fusion"])
        self.assertEqual(trace["token_logit_fusion_mode"], "static")
        self.assertTrue(trace["token_logit_fusion_has_detector_logits"])
        self.assertIsNotNone(trace["token_logit_fusion_gate_mean"])

    def test_fusion_consistency_loss_in_paper_loss(self):
        config = _config(
            use_token_logit_fusion=True, token_logit_fusion_mode="static",
            detector_prox_mode="token_posterior",
            use_offgrid_refinement=False,
        )
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)
        codeword = _complex_dd(config.vocab_size, config)
        prior = TokenCodewordPrior(codeword_book=codeword)
        output = model(y_dd, token_prior=prior, return_details=True)
        token_ids = torch.tensor([3, 7])
        weights = ReceiverLossWeights(
            token_ce_weight=1.0, token_logit_fusion_consistency_weight=0.5,
        )
        result = paper_receiver_loss(output, token_ids, weights=weights)
        self.assertIn("token_logit_fusion_consistency", result.components)
        self.assertTrue(torch.isfinite(result.total))


class ConstraintsTests(unittest.TestCase):
    def test_no_bit_qam_no_transformer_gnn_vae(self):
        config = _config(use_token_logit_fusion=True)
        model = LearnableOTFSReceiver(config)
        names = {m.__class__.__name__.lower() for m in model.modules()}
        self.assertFalse(any("transformer" in n for n in names))
        self.assertFalse(any("graph" in n for n in names))
        self.assertFalse(any("vae" in n for n in names))
        self.assertFalse(hasattr(model, "bit_head"))
        self.assertFalse(hasattr(model, "qam_head"))


if __name__ == "__main__":
    unittest.main()
