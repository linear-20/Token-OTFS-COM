from pathlib import Path
import sys
import unittest

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from receiver import (
    DataConsistencyOutput,
    LearnableOTFSReceiver,
    ReceiverConfig,
    ReceiverOutput,
    named_receiver_ablation,
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


def _complex_dd(batch: int, config: ReceiverConfig) -> torch.Tensor:
    real = torch.randn(batch, config.M, config.N)
    imag = torch.randn(batch, config.M, config.N)
    return torch.complex(real, imag)


class ReceiverPaperChainTests(unittest.TestCase):
    def test_default_model_returns_logits_tensor(self):
        config = _config()
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)

        logits = model(y_dd)

        self.assertIsInstance(logits, torch.Tensor)
        self.assertEqual(logits.shape, (2, config.vocab_size))

    def test_return_details_contains_paper_chain_outputs(self):
        config = _config()
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)

        output = model(y_dd, return_details=True)

        self.assertIsInstance(output, ReceiverOutput)
        self.assertIsNotNone(output.sparse_estimate)
        self.assertIsNotNone(output.operator_state)
        self.assertIsNotNone(output.equalizer_output)
        self.assertIsNotNone(output.denoiser_output)
        self.assertIsNotNone(output.classifier_output)
        self.assertIsNotNone(output.receiver_trace)
        self.assertIsNotNone(output.equalizer_output.residual_variances)
        self.assertIsNotNone(output.equalizer_output.estimate_variances)
        self.assertIsNotNone(output.equalizer_output.posterior_temperatures)
        self.assertEqual(output.token_logits.shape, (2, config.vocab_size))

    def test_return_details_without_return_aux_has_no_aux_logits(self):
        config = _config()
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)

        output = model(y_dd, return_details=True, return_aux=False)

        self.assertIsNone(output.aux_logits)
        self.assertIsNotNone(output.equalizer_output)

    def test_residual_energy_available_without_return_aux(self):
        config = _config(use_equalizer_residual_for_denoiser=True)
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)

        output = model(y_dd, return_details=True, return_aux=False)

        self.assertIsNotNone(output.residual_energy)
        self.assertEqual(output.residual_energy.shape, (2, config.M, config.N))

    def test_classifier_output_evidence_weights_shape(self):
        config = _config()
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)

        output = model(y_dd, return_details=True)

        self.assertIsNotNone(output.classifier_output.evidence_weights)
        self.assertEqual(output.classifier_output.evidence_weights.shape, (2, config.M, config.N))

    def test_return_details_exposes_multihead_classifier_fields(self):
        config = _config(classifier_num_evidence_heads=3)
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)

        output = model(y_dd, return_details=True)

        self.assertIsNotNone(output.classifier_output.head_embeddings)
        self.assertIsNotNone(output.classifier_output.head_logits)
        self.assertIsNotNone(output.classifier_output.head_evidence_weights)
        self.assertIsNotNone(output.classifier_output.head_fusion_weights)
        self.assertEqual(output.classifier_output.head_embeddings.shape, (2, 3, config.token_embedding_dim))
        self.assertEqual(output.classifier_output.head_logits.shape, (2, 3, config.vocab_size))
        self.assertEqual(output.classifier_output.head_evidence_weights.shape, (2, 3, config.M, config.N))
        self.assertEqual(output.classifier_output.head_fusion_weights.shape, (2, 3))
        self.assertEqual(output.receiver_trace["classifier_num_evidence_heads"], 3)
        self.assertTrue(output.receiver_trace["classifier_uses_multihead_evidence"])
        self.assertIn("classifier_head_entropy_mean", output.receiver_trace)
        self.assertIn("classifier_head_fusion_entropy_mean", output.receiver_trace)

    def test_return_classifier_details_exposes_multihead_fields_without_trace(self):
        config = _config(classifier_num_evidence_heads=2)
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)

        output = model(y_dd, return_classifier_details=True)

        self.assertIsNotNone(output.classifier_output.head_evidence_weights)
        self.assertEqual(output.classifier_output.head_evidence_weights.shape, (2, 2, config.M, config.N))
        self.assertIsNone(output.receiver_trace)

    def test_token_prior_return_details_exposes_logits_and_weights(self):
        config = _config()
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)
        codeword_book = _complex_dd(config.vocab_size, config)
        prior = TokenCodewordPrior(codeword_book=codeword_book, prior_strength=1.0)

        output = model(y_dd, token_prior=prior, return_details=True)

        self.assertIsNotNone(output.token_prior_logits)
        self.assertIsNotNone(output.token_prior_weights)
        self.assertIsNotNone(output.token_posterior_variance)
        self.assertEqual(output.token_posterior_variance.shape, (2, config.M, config.N))
        self.assertTrue(output.receiver_trace["uses_token_posterior_variance_for_reliability"])
        self.assertEqual(len(output.token_prior_logits), config.num_unfolded_layers)
        self.assertEqual(len(output.token_prior_weights), config.num_unfolded_layers)
        for logits, weights in zip(output.token_prior_logits, output.token_prior_weights):
            self.assertEqual(logits.shape, (2, config.vocab_size))
            self.assertEqual(weights.shape, (2, config.vocab_size))

    def test_no_token_prior_has_no_posterior_variance(self):
        config = _config()
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)

        output = model(y_dd, return_details=True)

        self.assertIsNone(output.token_posterior_variance)
        self.assertFalse(output.receiver_trace["uses_token_posterior_variance_for_reliability"])

    def test_l1_detector_has_no_token_posterior_variance(self):
        config = _config(detector_prox_mode="l1")
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)
        prior = TokenCodewordPrior(codeword_book=_complex_dd(config.vocab_size, config), prior_strength=1.0)

        output = model(y_dd, token_prior=prior, return_details=True)

        self.assertIsNone(output.token_posterior_variance)
        self.assertFalse(output.receiver_trace["uses_token_posterior_variance_for_reliability"])
        self.assertEqual(output.token_logits.shape, (2, config.vocab_size))

    def test_data_mask_return_details_runs_and_trace_records_token_posterior_settings(self):
        config = _config(token_posterior_topk=3)
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)
        data_mask = torch.ones(1, config.M, config.N)
        data_mask[:, 0, 0] = 0.0
        prior = TokenCodewordPrior(codeword_book=_complex_dd(config.vocab_size, config), prior_strength=1.0)

        output = model(y_dd, token_prior=prior, data_mask=data_mask, return_details=True)

        self.assertEqual(output.token_logits.shape, (2, config.vocab_size))
        self.assertEqual(output.receiver_trace["detector_prox_mode"], config.detector_prox_mode)
        self.assertEqual(output.receiver_trace["token_posterior_topk"], 3)
        self.assertTrue(output.receiver_trace["use_data_region_mask_for_token_prox"])
        self.assertTrue(output.receiver_trace["has_data_mask"])
        self.assertFalse(output.receiver_trace["has_pilot_guard_mask"])
        self.assertTrue(output.receiver_trace["uses_symbol_reliability_map"])
        self.assertTrue(output.receiver_trace["classifier_uses_data_mask_not_channel_support"])
        self.assertTrue(output.receiver_trace["use_variance_tracking"])
        self.assertTrue(output.receiver_trace["posterior_temperature_from_variance"])
        self.assertTrue(output.receiver_trace["confidence_weighted_variance"])
        self.assertIsNotNone(output.equalizer_output.token_posterior_weights)
        self.assertEqual(output.equalizer_output.token_posterior_weights[0].shape, (2, 3))

    def test_return_details_trace_includes_variance_flags(self):
        config = _config(
            use_variance_tracking=True,
            posterior_temperature_from_variance=True,
            confidence_weighted_variance=True,
        )
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)

        output = model(y_dd, return_details=True)

        self.assertTrue(output.receiver_trace["use_variance_tracking"])
        self.assertTrue(output.receiver_trace["posterior_temperature_from_variance"])
        self.assertTrue(output.receiver_trace["confidence_weighted_variance"])
        self.assertEqual(len(output.equalizer_output.residual_variances), config.num_unfolded_layers)
        self.assertEqual(output.equalizer_output.residual_variances[0].shape, (2, 1, 1))

    def test_return_details_trace_includes_physics_refinement_flags(self):
        config = _config(use_physics_guided_gain_refinement=True,
                         physics_refinement_allow_full_grid_debug=True)
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)

        output = model(y_dd, return_details=True)

        self.assertTrue(output.receiver_trace["use_physics_guided_gain_refinement"])
        self.assertTrue(output.receiver_trace["physics_refined"])
        self.assertIsNotNone(output.receiver_trace["physics_residual_power_mean"])
        self.assertIsNotNone(output.receiver_trace["physics_fit_residual_power_mean"])
        self.assertIsNotNone(output.receiver_trace["physics_full_residual_power_mean"])
        self.assertIsNotNone(output.sparse_estimate.physics_residual_power)
        self.assertIsNotNone(output.sparse_estimate.physics_fit_residual_power)
        self.assertIsNotNone(output.sparse_estimate.physics_full_residual_power)
        self.assertEqual(output.receiver_trace["physics_fit_mask_source"], "full_grid_debug")

    def test_classifier_reliability_not_sparse_channel_support_only(self):
        config = _config()
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(1, config)
        h_dd = _complex_dd(1, config)
        support_mask = torch.zeros(1, config.M, config.N)
        support_mask[:, 0, 0] = 1.0
        data_mask = torch.ones(1, config.M, config.N)

        output = model(
            y_dd,
            h_dd=h_dd,
            support_mask=support_mask,
            data_mask=data_mask,
            return_details=True,
        )

        channel_support = output.sparse_estimate.support_mask.bool()
        classifier_confidence = output.classifier_output.confidence_map
        self.assertIsNotNone(classifier_confidence)
        self.assertGreater(float(classifier_confidence[~channel_support].sum().item()), 0.0)
        self.assertTrue(torch.all(output.classifier_output.confidence_map >= 0.0))

    def test_pilot_guard_masks_derive_data_mask_for_trace(self):
        config = _config()
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)
        pilot_mask = torch.zeros(1, config.M, config.N)
        guard_mask = torch.zeros(1, config.M, config.N)
        pilot_mask[:, 1, 1] = 1.0
        guard_mask[:, 1:3, 1:3] = 1.0

        output = model(y_dd, pilot_mask=pilot_mask, guard_mask=guard_mask, return_details=True)

        self.assertTrue(output.receiver_trace["has_data_mask"])
        self.assertTrue(output.receiver_trace["has_pilot_guard_mask"])

    def test_return_local_classifier_exposes_local_embeddings(self):
        config = _config()
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)

        output = model(y_dd, return_details=True, return_local_classifier=True)

        self.assertIsNotNone(output.classifier_output.local_embeddings)
        self.assertEqual(
            output.classifier_output.local_embeddings.shape,
            (2, config.token_embedding_dim, config.M, config.N),
        )

    def test_named_ablation_no_offgrid(self):
        config = _config(use_offgrid_refinement=True, dd_operator_mode="auto")

        ablated = named_receiver_ablation("no_offgrid", config)

        self.assertFalse(ablated.use_offgrid_refinement)
        self.assertEqual(ablated.dd_operator_mode, "ongrid")

    def test_named_ablation_fallback_ce_baseline(self):
        config = _config(channel_estimator_mode="pilot", use_offgrid_refinement=True, dd_operator_mode="auto")

        ablated = named_receiver_ablation("fallback_ce_baseline", config)

        self.assertEqual(ablated.channel_estimator_mode, "fallback_topk")
        self.assertFalse(ablated.use_offgrid_refinement)
        self.assertEqual(ablated.dd_operator_mode, "ongrid")

    def test_physics_trace_includes_fit_mask_source(self):
        config = _config(use_physics_guided_gain_refinement=True, physics_refinement_ridge=1e-4)
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)

        output = model(y_dd, return_details=True)

        self.assertIn("physics_fit_mask_source", output.receiver_trace)
        self.assertIn("physics_probe_position", output.receiver_trace)
        self.assertIn("physics_uses_known_pilot_probe", output.receiver_trace)

    def test_ce_data_mask_priority_over_plain_data_mask(self):
        config = _config(use_physics_guided_gain_refinement=True, physics_refinement_ridge=1e-4)
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)
        ce_mask = torch.ones(1, config.M, config.N)
        data_mask = torch.zeros(1, config.M, config.N)

        output = model(y_dd, return_details=True, ce_data_mask=ce_mask, data_mask=data_mask)

        self.assertIsNotNone(output.receiver_trace["physics_fit_mask_source"])

    def test_default_logits_output_still_works(self):
        config = _config(use_physics_guided_gain_refinement=True)
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)

        logits = model(y_dd)

        self.assertIsInstance(logits, torch.Tensor)
        self.assertEqual(logits.shape, (2, config.vocab_size))

    def test_dc_correction_enabled_returns_data_consistency_output(self):
        config = _config(use_data_consistency_correction=True, dc_step_init=0.0)
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)

        output = model(y_dd, return_details=True)

        self.assertIsNotNone(output.data_consistency_output)
        self.assertIsNotNone(output.x_consistent)
        self.assertTrue(output.receiver_trace["use_data_consistency_correction"])
        self.assertTrue(output.receiver_trace["data_consistency_corrected"])
        self.assertIsNotNone(output.receiver_trace["dc_step_size"])
        self.assertIsNotNone(output.receiver_trace["dc_residual_energy_mean"])
        self.assertIsNotNone(output.receiver_trace["dc_observation_weight_mean"])
        self.assertIsNotNone(output.receiver_trace["dc_update_gate_mean"])

    def test_dc_correction_disabled_is_none(self):
        config = _config(use_data_consistency_correction=False)
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)

        output = model(y_dd, return_details=True)

        self.assertIsNone(output.data_consistency_output)
        self.assertIsNone(output.x_consistent)

    def test_dc_default_logits_shape(self):
        config = _config(use_data_consistency_correction=True, dc_step_init=0.0)
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)

        logits = model(y_dd)

        self.assertIsInstance(logits, torch.Tensor)
        self.assertEqual(logits.shape, (2, config.vocab_size))

    def test_no_bit_qam_head_no_linear_vocab_head_no_forbidden_architecture(self):
        config = _config()
        model = LearnableOTFSReceiver(config)
        module_names = {module.__class__.__name__.lower() for module in model.modules()}
        linear_vocab_heads = [
            module
            for module in model.token_classifier.modules()
            if isinstance(module, nn.Linear) and module.out_features == config.vocab_size
        ]

        self.assertEqual(len(linear_vocab_heads), 0)
        self.assertFalse(hasattr(model, "bit_head"))
        self.assertFalse(hasattr(model, "qam_head"))
        self.assertFalse(any("transformer" in name for name in module_names))
        self.assertFalse(any("graph" in name for name in module_names))
        self.assertFalse(any("vae" in name for name in module_names))


if __name__ == "__main__":
    unittest.main()
