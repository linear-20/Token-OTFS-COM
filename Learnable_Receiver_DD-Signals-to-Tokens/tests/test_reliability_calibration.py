from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from receiver import (
    LearnableOTFSReceiver,
    ReceiverConfig,
    ReceiverLossOutput,
    ReceiverLossWeights,
    ReliabilityDiagnostics,
    build_reliability_diagnostics,
    confidence_nll_regularization,
    paper_receiver_loss,
    posterior_confidence_entropy,
    posterior_entropy_regularization,
    reliability_error_alignment_loss,
    softmax_confidence_and_entropy,
    token_confidence_calibration_loss,
    token_margin,
    total_receiver_loss,
)
from receiver.token_prior import TokenCodewordPrior


def _config(**kwargs) -> ReceiverConfig:
    values = dict(
        M=5,
        N=4,
        vocab_size=11,
        token_embedding_dim=7,
        num_unfolded_layers=2,
        topk_paths=3,
        hidden_channels=6,
        noise_var=0.1,
        use_refinement_net=True,
        use_offgrid_refinement=True,
    )
    values.update(kwargs)
    return ReceiverConfig(**values)


def _complex_dd(batch, config):
    real = torch.randn(batch, config.M, config.N)
    imag = torch.randn(batch, config.M, config.N)
    return torch.complex(real, imag)


class DiagnosticsHelperTests(unittest.TestCase):
    def test_softmax_confidence_entropy_shapes(self):
        logits = torch.randn(4, 10)
        conf, ent = softmax_confidence_and_entropy(logits)
        self.assertEqual(conf.shape, (4,))
        self.assertEqual(ent.shape, (4,))
        self.assertTrue(torch.all(conf >= 0) and torch.all(conf <= 1))
        self.assertTrue(torch.all(ent >= 0) and torch.all(ent <= 1))

    def test_uniform_logits_high_entropy(self):
        logits = torch.zeros(2, 100)
        _, ent = softmax_confidence_and_entropy(logits)
        self.assertGreater(float(ent.mean().item()), 0.95)

    def test_peaky_logits_low_entropy(self):
        logits = torch.zeros(2, 10)
        logits[:, 0] = 100.0
        _, ent = softmax_confidence_and_entropy(logits)
        self.assertLess(float(ent.mean().item()), 0.05)

    def test_token_margin_shape_range(self):
        logits = torch.randn(4, 10)
        margin = token_margin(logits)
        self.assertEqual(margin.shape, (4,))
        self.assertTrue(torch.all(margin >= 0) and torch.all(margin <= 1))

    def test_posterior_confidence_entropy_kshape(self):
        logits = torch.randn(4, 5)
        conf, ent = posterior_confidence_entropy(logits)
        self.assertEqual(conf.shape, (4,))
        self.assertEqual(ent.shape, (4,))

    def test_build_diagnostics_all_fields(self):
        config = _config()
        rel_map = torch.rand(2, config.M, config.N)
        logits = torch.randn(2, config.vocab_size)
        post_logits = torch.randn(2, 5)
        res_energy = torch.rand(2, config.M, config.N)
        unc_map = torch.rand(2, config.M, config.N)
        data_mask = torch.ones(2, config.M, config.N)

        diag = build_reliability_diagnostics(
            reliability_map=rel_map,
            token_logits=logits,
            posterior_logits=post_logits,
            residual_energy=res_energy,
            uncertainty_map=unc_map,
            data_mask=data_mask,
        )

        self.assertIsInstance(diag, ReliabilityDiagnostics)
        self.assertIsNotNone(diag.classifier_confidence)
        self.assertIsNotNone(diag.classifier_entropy)
        self.assertIsNotNone(diag.token_margin)
        self.assertIsNotNone(diag.posterior_confidence)
        self.assertIsNotNone(diag.posterior_entropy)
        self.assertIsNotNone(diag.expected_error_proxy)
        self.assertIsNotNone(diag.reliability_scalar)

    def test_diagnostics_all_zero_mask_no_nan(self):
        config = _config()
        data_mask = torch.zeros(1, config.M, config.N)
        diag = build_reliability_diagnostics(
            token_logits=torch.randn(1, config.vocab_size),
            residual_energy=torch.ones(1, config.M, config.N),
            data_mask=data_mask,
        )
        self.assertIsNotNone(diag.classifier_confidence)
        if diag.residual_energy_mean is not None:
            self.assertTrue(torch.isfinite(diag.residual_energy_mean).all())

    def test_no_token_ids_used_in_diagnostics(self):
        diag = build_reliability_diagnostics(
            token_logits=torch.randn(2, 10),
        )
        self.assertIsNotNone(diag.classifier_confidence)
        # diag never references token_ids


class ModelIntegrationTests(unittest.TestCase):
    def test_model_returns_diagnostics_with_details(self):
        config = _config(use_offgrid_refinement=False)
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)

        output = model(y_dd, return_details=True)

        self.assertIsNotNone(output.reliability_diagnostics)
        self.assertIsNotNone(output.reliability_diagnostics.classifier_confidence)

    def test_model_returns_diagnostics_with_flag(self):
        config = _config(use_offgrid_refinement=False)
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)

        output = model(y_dd, return_reliability_diagnostics=True)

        self.assertIsNotNone(output.reliability_diagnostics)

    def test_default_model_returns_logits_tensor(self):
        config = _config(use_offgrid_refinement=False)
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)

        logits = model(y_dd)

        self.assertIsInstance(logits, torch.Tensor)
        self.assertEqual(logits.shape, (2, config.vocab_size))

    def test_trace_includes_reliability_fields(self):
        config = _config(use_offgrid_refinement=False)
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)

        output = model(y_dd, return_details=True)

        trace = output.receiver_trace
        self.assertTrue(trace["has_reliability_diagnostics"])
        self.assertIsNotNone(trace["classifier_confidence_mean"])
        self.assertIsNotNone(trace["classifier_entropy_mean"])


class CalibrationLossTests(unittest.TestCase):
    def test_calibration_loss_scalar_finite(self):
        logits = torch.randn(4, 10)
        token_ids = torch.tensor([2, 5, 1, 8])
        loss = token_confidence_calibration_loss(logits, token_ids)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))

    def test_calibration_perfect_logits_has_low_cal_loss(self):
        logits = torch.zeros(4, 10)
        for i in range(4):
            logits[i, i] = 100.0
        token_ids = torch.arange(4)
        loss = token_confidence_calibration_loss(logits, token_ids)
        self.assertLess(float(loss.item()), 0.02)

    def test_reliability_alignment_loss_finite(self):
        rel_scalar = torch.rand(4)
        logits = torch.randn(4, 10)
        token_ids = torch.tensor([3, 1, 7, 2])
        loss = reliability_error_alignment_loss(rel_scalar, logits, token_ids)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))

    def test_posterior_entropy_reg_scalar_finite(self):
        logits = torch.randn(4, 5)
        loss = posterior_entropy_regularization(logits)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))

    def test_posterior_entropy_with_target(self):
        logits = torch.randn(4, 5)
        loss = posterior_entropy_regularization(logits, target_entropy=0.5)
        self.assertTrue(torch.isfinite(loss))

    def test_confidence_nll_reg_finite(self):
        logits = torch.randn(4, 10)
        token_ids = torch.tensor([2, 5, 1, 8])
        loss = confidence_nll_regularization(logits, token_ids)
        self.assertTrue(torch.isfinite(loss))


class PaperReceiverLossCalibrationTests(unittest.TestCase):
    def test_calibration_component_in_paper_loss(self):
        config = _config(use_offgrid_refinement=False)
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)
        output = model(y_dd, return_details=True)
        token_ids = torch.tensor([3, 7])
        weights = ReceiverLossWeights(token_ce_weight=1.0, calibration_weight=0.5)

        result = paper_receiver_loss(output, token_ids, weights=weights)

        self.assertIn("calibration", result.components)
        self.assertTrue(torch.isfinite(result.total))

    def test_reliability_alignment_component(self):
        config = _config(use_offgrid_refinement=False)
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)
        output = model(y_dd, return_details=True)
        token_ids = torch.tensor([3, 7])
        weights = ReceiverLossWeights(token_ce_weight=1.0, reliability_alignment_weight=0.3)

        result = paper_receiver_loss(output, token_ids, weights=weights)

        self.assertIn("reliability_alignment", result.components)
        self.assertTrue(torch.isfinite(result.total))

    def test_posterior_entropy_component(self):
        config = _config(
            detector_prox_mode="token_posterior",
            use_offgrid_refinement=False,
        )
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)
        codeword = _complex_dd(config.vocab_size, config)
        prior = TokenCodewordPrior(codeword_book=codeword)
        output = model(y_dd, token_prior=prior, return_details=True)
        token_ids = torch.tensor([2, 5])
        weights = ReceiverLossWeights(token_ce_weight=1.0, posterior_entropy_weight=0.2)

        result = paper_receiver_loss(output, token_ids, token_prior=prior, weights=weights)

        self.assertIn("posterior_entropy_reg", result.components)
        self.assertTrue(torch.isfinite(result.total))

    def test_old_total_receiver_loss_still_works(self):
        config = _config()
        logits = torch.randn(4, config.vocab_size)
        token_ids = torch.tensor([2, 5, 1, 8])
        loss = total_receiver_loss(logits, token_ids)
        self.assertTrue(torch.isfinite(loss))


class ConstraintsTests(unittest.TestCase):
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
