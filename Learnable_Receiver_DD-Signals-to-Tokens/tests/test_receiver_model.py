from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from receiver import LearnableOTFSReceiver, ReceiverConfig, ReceiverOutput
from receiver.pilot_ce import EmbeddedPilotConfig
from receiver.token_prior import TokenCodewordPrior


def _config() -> ReceiverConfig:
    return ReceiverConfig(
        M=5,
        N=4,
        vocab_size=11,
        token_embedding_dim=7,
        num_unfolded_layers=2,
        topk_paths=3,
        hidden_channels=6,
        noise_var=0.1,
        use_refinement_net=True,
    )


def _complex_dd(batch: int, config: ReceiverConfig) -> torch.Tensor:
    real = torch.randn(batch, config.M, config.N)
    imag = torch.randn(batch, config.M, config.N)
    return torch.complex(real, imag)


class ReceiverModelTests(unittest.TestCase):
    def test_default_forward_returns_logits_tensor(self):
        config = _config()
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)

        logits = model(y_dd)

        self.assertIsInstance(logits, torch.Tensor)
        self.assertEqual(logits.shape, (2, config.vocab_size))

    def test_return_details_returns_receiver_output(self):
        config = _config()
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)

        output = model(y_dd, return_details=True)

        self.assertIsInstance(output, ReceiverOutput)
        self.assertEqual(output.token_logits.shape, (2, config.vocab_size))
        self.assertEqual(output.rx_embedding.shape, (2, config.token_embedding_dim))

    def test_receiver_output_field_shapes(self):
        config = _config()
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(3, config)
        h_dd = _complex_dd(3, config)
        support_mask = torch.zeros(3, config.M, config.N)
        support_mask[:, 0, 0] = 1.0
        support_mask[:, 1, 2] = 1.0

        output = model(y_dd, h_dd=h_dd, support_mask=support_mask, snr_db=12.0, return_details=True, topk=2)

        self.assertEqual(output.sparse_estimate.h_dd.shape, (3, config.M, config.N))
        self.assertEqual(output.sparse_estimate.path_indices.shape, (3, config.topk_paths, 2))
        self.assertEqual(output.operator_state.path_gains.shape, (3, config.topk_paths))
        self.assertEqual(output.x_equalized.shape, (3, config.M, config.N))
        self.assertEqual(output.x_refined.shape, (3, config.M, config.N))
        self.assertTrue(torch.is_complex(output.x_equalized))
        self.assertTrue(torch.is_complex(output.x_refined))

    def test_return_aux_details_passes_confidence_info_to_denoiser(self):
        config = _config()
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)
        h_dd = _complex_dd(2, config)
        support_mask = torch.zeros(2, config.M, config.N)
        support_mask[:, 0, 0] = 1.0
        support_mask[:, 1, 2] = 1.0

        output = model(
            y_dd,
            h_dd=h_dd,
            support_mask=support_mask,
            return_aux=True,
            return_details=True,
        )

        self.assertEqual(output.token_logits.shape, (2, config.vocab_size))
        self.assertEqual(output.x_refined.shape, (2, config.M, config.N))
        self.assertIsNotNone(output.equalizer_output)

    def test_return_aux_logits_match_equalizer_layer_count(self):
        config = _config()
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)

        output = model(y_dd, return_aux=True)

        self.assertIsInstance(output, ReceiverOutput)
        self.assertIsNotNone(output.equalizer_output)
        self.assertIsNotNone(output.aux_logits)
        self.assertEqual(len(output.aux_logits), len(output.equalizer_output.layer_estimates))
        for logits in output.aux_logits:
            self.assertEqual(logits.shape, (2, config.vocab_size))

    def test_return_aux_with_token_prior_exposes_prior_logits(self):
        config = _config()
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)
        codeword_book = torch.randn(config.vocab_size, config.M, config.N, dtype=torch.complex64)
        prior = TokenCodewordPrior(codeword_book=codeword_book, prior_strength=1.0)

        output = model(y_dd, token_prior=prior, return_aux=True)

        self.assertIsInstance(output, ReceiverOutput)
        self.assertIsNotNone(output.token_prior_logits)
        self.assertEqual(len(output.token_prior_logits), config.num_unfolded_layers)
        for logits in output.token_prior_logits:
            self.assertEqual(logits.shape, (2, config.vocab_size))

    def test_return_details_with_topk_preserves_topk_metadata(self):
        config = _config()
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)

        output = model(y_dd, return_details=True, topk=3)

        self.assertEqual(output.topk_indices.shape, (2, 3))
        self.assertEqual(output.topk_logits.shape, (2, 3))
        self.assertEqual(output.topk_indices.dtype, torch.long)

    def test_classifier_receives_sparse_support_confidence_forward_runs(self):
        config = _config()
        model = LearnableOTFSReceiver(config)
        y_dd = torch.zeros(2, config.M, config.N, dtype=torch.complex64)
        y_dd[:, 2, 2] = 1.0 + 0.1j
        y_dd[:, 3, 3] = 0.8 - 0.2j
        pilot_config = EmbeddedPilotConfig(
            pilot_delay=2,
            pilot_doppler=2,
            guard_delay=1,
            guard_doppler=1,
            obs_delay_radius=1,
            obs_doppler_radius=1,
        )

        output = model(
            y_dd,
            channel_estimator_mode="pilot",
            pilot_config=pilot_config,
            return_details=True,
            topk=2,
        )

        self.assertEqual(output.token_logits.shape, (2, config.vocab_size))
        self.assertIsNotNone(output.sparse_estimate.path_confidence_map)
        self.assertEqual(output.topk_indices.shape, (2, 2))

    def test_snr_scalar_and_batch_vector_both_run(self):
        config = _config()
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)

        scalar_output = model(y_dd, snr_db=10.0, return_details=True)
        vector_output = model(y_dd, snr_db=torch.tensor([10.0, 15.0]), return_details=True)

        self.assertEqual(scalar_output.token_logits.shape, (2, config.vocab_size))
        self.assertEqual(vector_output.token_logits.shape, (2, config.vocab_size))

    def test_use_denoiser_flag_preserves_legacy_default_and_allows_override(self):
        legacy_config = _config()
        override_config = ReceiverConfig(
            M=5,
            N=4,
            vocab_size=11,
            token_embedding_dim=7,
            num_unfolded_layers=2,
            topk_paths=3,
            hidden_channels=6,
            noise_var=0.1,
            use_refinement_net=True,
            use_denoiser=False,
        )

        legacy_model = LearnableOTFSReceiver(legacy_config)
        override_model = LearnableOTFSReceiver(override_config)

        self.assertTrue(legacy_model.denoiser.config.use_refinement_net)
        self.assertFalse(override_model.denoiser.config.use_refinement_net)


if __name__ == "__main__":
    unittest.main()
