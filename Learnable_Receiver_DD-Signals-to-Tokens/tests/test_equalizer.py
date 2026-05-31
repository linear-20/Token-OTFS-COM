from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from receiver import ReceiverConfig
from receiver.dd_ops import SparseDDOperator, SparseDDOperatorState
from receiver.equalizer import (
    EqualizerOutput,
    UnfoldedPICProxEqualizer,
    complex_soft_threshold,
)
from receiver.token_prior import TokenCodewordPrior


def _config(noise_var: float = 0.1, **kwargs) -> ReceiverConfig:
    values = dict(
        M=4,
        N=5,
        vocab_size=9,
        token_embedding_dim=8,
        num_unfolded_layers=3,
        topk_paths=2,
        hidden_channels=6,
        noise_var=noise_var,
        use_refinement_net=False,
    )
    values.update(kwargs)
    return ReceiverConfig(**values)


def _complex_dd(batch: int, config: ReceiverConfig) -> torch.Tensor:
    real = torch.randn(batch, config.M, config.N)
    imag = torch.randn(batch, config.M, config.N)
    return torch.complex(real, imag)


def _token_prior(config: ReceiverConfig, prior_strength: float = 1.0) -> TokenCodewordPrior:
    codeword_book = _complex_dd(config.vocab_size, config)
    return TokenCodewordPrior(
        codeword_book=codeword_book,
        prior_strength=prior_strength,
        temperature=0.5,
        similarity="real",
    )


def _identity_state(config: ReceiverConfig, batch: int, path_gains: torch.Tensor | None = None) -> SparseDDOperatorState:
    if path_gains is None:
        path_gains = torch.ones(batch, 1, dtype=torch.complex64)
    h_dd = torch.zeros(batch, config.M, config.N, dtype=path_gains.dtype)
    h_dd[:, 0, 0] = path_gains[:, 0].detach()
    support_mask = torch.zeros(batch, config.M, config.N)
    support_mask[:, 0, 0] = 1.0
    path_indices = torch.zeros(batch, 1, 2, dtype=torch.long)
    confidence = torch.ones(batch, 1)
    return SparseDDOperatorState(
        h_dd=h_dd,
        support_mask=support_mask,
        path_indices=path_indices,
        path_gains=path_gains,
        confidence=confidence,
    )


class EqualizerTests(unittest.TestCase):
    def test_complex_soft_threshold_zero_threshold_matches_input(self):
        config = _config()
        z = _complex_dd(2, config)

        out = complex_soft_threshold(z, 0.0)

        self.assertTrue(torch.allclose(out, z, atol=1e-6, rtol=1e-6))

    def test_complex_soft_threshold_large_threshold_goes_to_zero(self):
        config = _config()
        z = _complex_dd(2, config)

        out = complex_soft_threshold(z, 1e6)

        self.assertTrue(torch.allclose(out, torch.zeros_like(z)))

    def test_equalizer_forward_shape_and_dtype(self):
        config = _config()
        operator = SparseDDOperator(config)
        equalizer = UnfoldedPICProxEqualizer(config, operator)
        y_dd = _complex_dd(2, config)
        state = _identity_state(config, batch=2)

        x_hat = equalizer(y_dd, state)

        self.assertEqual(x_hat.shape, (2, config.M, config.N))
        self.assertTrue(torch.is_complex(x_hat))
        self.assertEqual(x_hat.dtype, y_dd.dtype)

    def test_equalizer_without_token_prior_preserves_old_behavior(self):
        config = _config()
        operator = SparseDDOperator(config)
        equalizer = UnfoldedPICProxEqualizer(config, operator)
        y_dd = _complex_dd(2, config)
        state = _identity_state(config, batch=2)

        x_hat = equalizer(y_dd, state, token_prior=None)

        self.assertEqual(x_hat.shape, (2, config.M, config.N))
        self.assertEqual(x_hat.dtype, y_dd.dtype)

    def test_identity_channel_noise_free_is_close_to_input(self):
        config = _config(noise_var=0.0)
        operator = SparseDDOperator(config)
        equalizer = UnfoldedPICProxEqualizer(config, operator)
        y_dd = torch.full((1, config.M, config.N), 1.0 + 0.5j, dtype=torch.complex64)
        state = _identity_state(config, batch=1)

        x_hat = equalizer(y_dd, state)

        self.assertTrue(torch.allclose(x_hat, y_dd, atol=5e-3, rtol=5e-3))

    def test_return_all_contains_initial_and_per_layer_estimates(self):
        config = _config()
        operator = SparseDDOperator(config)
        equalizer = UnfoldedPICProxEqualizer(config, operator)
        y_dd = _complex_dd(1, config)
        state = _identity_state(config, batch=1)

        output = equalizer(y_dd, state, return_all=True)

        self.assertIsInstance(output, EqualizerOutput)
        self.assertEqual(len(output.layer_estimates), config.num_unfolded_layers + 1)
        self.assertEqual(len(output.residuals), config.num_unfolded_layers)
        self.assertEqual(output.x_hat.shape, (1, config.M, config.N))

    def test_equalizer_with_token_prior_returns_prior_logits(self):
        config = _config()
        operator = SparseDDOperator(config)
        equalizer = UnfoldedPICProxEqualizer(config, operator)
        y_dd = _complex_dd(2, config)
        state = _identity_state(config, batch=2)
        prior = _token_prior(config)

        output = equalizer(y_dd, state, return_all=True, token_prior=prior)

        self.assertIsInstance(output, EqualizerOutput)
        self.assertIsNotNone(output.token_prior_logits)
        self.assertIsNotNone(output.token_prior_weights)
        self.assertEqual(len(output.token_prior_logits), config.num_unfolded_layers)
        self.assertEqual(len(output.token_prior_weights), config.num_unfolded_layers)
        for logits, weights in zip(output.token_prior_logits, output.token_prior_weights):
            self.assertEqual(logits.shape, (2, config.vocab_size))
            self.assertEqual(weights.shape, (2, config.vocab_size))

    def test_alpha_and_threshold_are_trainable_parameters(self):
        config = _config()
        operator = SparseDDOperator(config)
        equalizer = UnfoldedPICProxEqualizer(config, operator)

        for block in equalizer.blocks:
            self.assertTrue(block.alpha_raw.requires_grad)
            self.assertTrue(block.threshold_raw.requires_grad)

    def test_beta_is_trainable(self):
        config = _config()
        operator = SparseDDOperator(config)
        equalizer = UnfoldedPICProxEqualizer(config, operator)

        for block in equalizer.blocks:
            self.assertTrue(block.beta_raw.requires_grad)
            self.assertTrue(block.variance_damping_raw.requires_grad)

    def test_token_prior_changes_output_when_strength_nonzero(self):
        config = _config()
        operator = SparseDDOperator(config)
        equalizer = UnfoldedPICProxEqualizer(config, operator)
        y_dd = _complex_dd(1, config)
        state = _identity_state(config, batch=1)
        codeword_book = _complex_dd(config.vocab_size, config)

        no_prior = equalizer(y_dd, state)
        zero_prior = equalizer(
            y_dd,
            state,
            token_prior=TokenCodewordPrior(codeword_book=codeword_book, prior_strength=0.0),
        )
        full_prior = equalizer(
            y_dd,
            state,
            token_prior=TokenCodewordPrior(codeword_book=codeword_book, prior_strength=1.0),
        )

        self.assertTrue(torch.allclose(no_prior, zero_prior, atol=1e-6, rtol=1e-6))
        self.assertFalse(torch.allclose(no_prior, full_prior, atol=1e-5, rtol=1e-5))

    def test_backward_flows_to_y_dd_and_path_gains(self):
        config = _config()
        operator = SparseDDOperator(config)
        equalizer = UnfoldedPICProxEqualizer(config, operator)
        y_dd = _complex_dd(1, config).requires_grad_()
        path_gains = torch.tensor([[1.0 + 0.2j]], dtype=torch.complex64, requires_grad=True)
        state = _identity_state(config, batch=1, path_gains=path_gains)

        x_hat = equalizer(y_dd, state)
        loss = x_hat.abs().pow(2).mean()
        loss.backward()

        self.assertIsNotNone(y_dd.grad)
        self.assertIsNotNone(path_gains.grad)
        self.assertGreater(float(y_dd.grad.abs().sum().item()), 0.0)
        self.assertGreater(float(path_gains.grad.abs().sum().item()), 0.0)

    def test_backward_flows_to_codeword_book(self):
        config = _config()
        operator = SparseDDOperator(config)
        equalizer = UnfoldedPICProxEqualizer(config, operator)
        y_dd = _complex_dd(1, config)
        state = _identity_state(config, batch=1)
        codeword_book = _complex_dd(config.vocab_size, config).requires_grad_()
        prior = TokenCodewordPrior(codeword_book=codeword_book, prior_strength=1.0)

        output = equalizer(y_dd, state, return_all=True, token_prior=prior)
        prior_logit_loss = sum(logits.mean() for logits in output.token_prior_logits)
        loss = output.x_hat.abs().mean() + prior_logit_loss
        loss.backward()

        self.assertIsNotNone(codeword_book.grad)
        self.assertGreater(float(codeword_book.grad.abs().sum().item()), 0.0)

    def test_detector_prox_mode_l1_ignores_token_prior(self):
        config = _config(detector_prox_mode="l1")
        operator = SparseDDOperator(config)
        equalizer = UnfoldedPICProxEqualizer(config, operator)
        y_dd = _complex_dd(1, config)
        state = _identity_state(config, batch=1)
        prior = _token_prior(config)

        no_prior = equalizer(y_dd, state, token_prior=None)
        with_prior = equalizer(y_dd, state, token_prior=prior)

        self.assertTrue(torch.allclose(no_prior, with_prior, atol=1e-6, rtol=1e-6))

    def test_token_posterior_mode_returns_posterior_fields(self):
        config = _config(detector_prox_mode="token_posterior", token_posterior_topk=4)
        operator = SparseDDOperator(config)
        equalizer = UnfoldedPICProxEqualizer(config, operator)
        y_dd = _complex_dd(2, config)
        state = _identity_state(config, batch=2)
        prior = _token_prior(config)

        output = equalizer(y_dd, state, return_all=True, token_prior=prior)

        self.assertIsNotNone(output.token_posterior_logits)
        self.assertIsNotNone(output.token_posterior_weights)
        self.assertIsNotNone(output.token_posterior_indices)
        self.assertIsNotNone(output.token_posterior_variance)
        self.assertEqual(len(output.token_posterior_logits), config.num_unfolded_layers)
        for logits, weights, indices, variance in zip(
            output.token_posterior_logits,
            output.token_posterior_weights,
            output.token_posterior_indices,
            output.token_posterior_variance,
        ):
            self.assertEqual(logits.shape, (2, 4))
            self.assertEqual(weights.shape, (2, 4))
            self.assertEqual(indices.shape, (2, 4))
            self.assertEqual(variance.shape, (2, config.M, config.N))

    def test_token_posterior_prox_changes_output_compared_to_no_prior(self):
        config = _config(detector_prox_mode="token_posterior")
        operator = SparseDDOperator(config)
        equalizer = UnfoldedPICProxEqualizer(config, operator)
        y_dd = _complex_dd(1, config)
        state = _identity_state(config, batch=1)
        prior = _token_prior(config)

        no_prior = equalizer(y_dd, state, token_prior=None)
        with_prior = equalizer(y_dd, state, token_prior=prior)

        self.assertFalse(torch.allclose(no_prior, with_prior, atol=1e-5, rtol=1e-5))

    def test_token_posterior_requires_no_ground_truth_token_id(self):
        config = _config(detector_prox_mode="token_posterior")
        operator = SparseDDOperator(config)
        equalizer = UnfoldedPICProxEqualizer(config, operator)
        y_dd = _complex_dd(2, config)
        state = _identity_state(config, batch=2)

        output = equalizer(y_dd, state, return_all=True, token_prior=_token_prior(config))

        self.assertEqual(output.x_hat.shape, (2, config.M, config.N))
        self.assertIsNotNone(output.token_posterior_weights)

    def test_return_all_contains_posterior_metadata(self):
        config = _config(detector_prox_mode="hybrid", token_posterior_topk=2)
        operator = SparseDDOperator(config)
        equalizer = UnfoldedPICProxEqualizer(config, operator)
        y_dd = _complex_dd(1, config)
        state = _identity_state(config, batch=1)

        output = equalizer(y_dd, state, return_all=True, token_prior=_token_prior(config))

        self.assertIsNotNone(output.token_posterior_logits)
        self.assertIsNotNone(output.token_posterior_weights)
        self.assertEqual(output.token_posterior_logits[0].shape, (1, 2))
        self.assertEqual(output.token_posterior_weights[0].shape, (1, 2))

    def test_return_all_contains_variance_metadata(self):
        config = _config()
        operator = SparseDDOperator(config)
        equalizer = UnfoldedPICProxEqualizer(config, operator)
        y_dd = _complex_dd(2, config)
        state = _identity_state(config, batch=2)

        output = equalizer(y_dd, state, return_all=True)

        self.assertIsNotNone(output.residual_variances)
        self.assertIsNotNone(output.estimate_variances)
        self.assertIsNotNone(output.posterior_temperatures)
        self.assertIsNotNone(output.detector_damping)
        self.assertEqual(len(output.residual_variances), config.num_unfolded_layers)
        self.assertEqual(len(output.estimate_variances), config.num_unfolded_layers)
        self.assertEqual(len(output.posterior_temperatures), config.num_unfolded_layers)
        self.assertEqual(len(output.detector_damping), config.num_unfolded_layers)
        for residual_var, estimate_var, tau, damping in zip(
            output.residual_variances,
            output.estimate_variances,
            output.posterior_temperatures,
            output.detector_damping,
        ):
            self.assertEqual(residual_var.shape, (2, 1, 1))
            self.assertEqual(estimate_var.shape, (2, 1, 1))
            self.assertEqual(tau.shape, (2, 1, 1))
            self.assertEqual(damping.shape, (2, 1, 1))

    def test_variances_are_finite_and_non_negative(self):
        config = _config()
        operator = SparseDDOperator(config)
        equalizer = UnfoldedPICProxEqualizer(config, operator)
        y_dd = _complex_dd(2, config)
        state = _identity_state(config, batch=2)

        output = equalizer(y_dd, state, return_all=True)

        for tensors in (
            output.residual_variances,
            output.estimate_variances,
            output.posterior_temperatures,
            output.detector_damping,
        ):
            for tensor in tensors:
                self.assertFalse(torch.isnan(tensor).any())
                self.assertFalse(torch.isinf(tensor).any())
                self.assertTrue(torch.all(tensor >= 0.0))

    def test_posterior_temperature_from_variance_changes_with_residual_power(self):
        config = _config(noise_var=0.1, posterior_temperature_from_variance=True)
        operator = SparseDDOperator(config)
        equalizer = UnfoldedPICProxEqualizer(config, operator)
        y_dd = torch.zeros(2, config.M, config.N, dtype=torch.complex64)
        y_dd[1] = 10.0 + 0.0j
        state = _identity_state(config, batch=2)

        output = equalizer(y_dd, state, return_all=True)
        tau = output.posterior_temperatures[0].reshape(2)

        self.assertGreater(float(tau[1].item()), float(tau[0].item()))

    def test_detector_prox_mode_l1_still_returns_variance_metadata(self):
        config = _config(detector_prox_mode="l1")
        operator = SparseDDOperator(config)
        equalizer = UnfoldedPICProxEqualizer(config, operator)
        y_dd = _complex_dd(2, config)
        state = _identity_state(config, batch=2)

        output = equalizer(y_dd, state, return_all=True, token_prior=_token_prior(config))

        self.assertIsNone(output.token_posterior_logits)
        self.assertEqual(len(output.residual_variances), config.num_unfolded_layers)
        self.assertEqual(output.residual_variances[0].shape, (2, 1, 1))


if __name__ == "__main__":
    unittest.main()
