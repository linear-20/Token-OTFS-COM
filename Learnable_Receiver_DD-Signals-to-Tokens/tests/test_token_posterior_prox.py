from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from receiver.token_prior import (
    TokenCodewordPrior,
    TokenPosteriorProxOutput,
    token_codeword_posterior_prox,
)


def _complex_dd(*shape: int) -> torch.Tensor:
    real = torch.randn(*shape)
    imag = torch.randn(*shape)
    return torch.complex(real, imag)


class TokenPosteriorProxTests(unittest.TestCase):
    def test_posterior_prox_output_shapes(self):
        batch, vocab, delay, doppler = 2, 5, 3, 4
        z_dd = _complex_dd(batch, delay, doppler)
        codeword_book = _complex_dd(vocab, delay, doppler)
        prior = TokenCodewordPrior(codeword_book=codeword_book, temperature=0.7)

        output = token_codeword_posterior_prox(z_dd, prior)

        self.assertIsInstance(output, TokenPosteriorProxOutput)
        self.assertEqual(output.projected_dd.shape, (batch, delay, doppler))
        self.assertEqual(output.posterior_logits.shape, (batch, vocab))
        self.assertEqual(output.posterior_weights.shape, (batch, vocab))
        self.assertIsNone(output.candidate_indices)
        self.assertEqual(output.posterior_variance.shape, (batch, delay, doppler))

    def test_posterior_weights_sum_to_one(self):
        z_dd = _complex_dd(2, 3, 4)
        prior = TokenCodewordPrior(codeword_book=_complex_dd(6, 3, 4), temperature=1.0)

        output = token_codeword_posterior_prox(z_dd, prior)

        self.assertTrue(torch.allclose(output.posterior_weights.sum(dim=-1), torch.ones(2), atol=1e-6))

    def test_exact_matching_codeword_gets_highest_posterior(self):
        vocab, delay, doppler = 4, 3, 4
        codeword_book = _complex_dd(vocab, delay, doppler)
        z_dd = codeword_book[2:3].clone()
        prior = TokenCodewordPrior(codeword_book=codeword_book, temperature=0.01)

        output = token_codeword_posterior_prox(z_dd, prior, temperature=0.01)

        self.assertEqual(int(output.posterior_weights.argmax(dim=-1).item()), 2)

    def test_data_mask_suppresses_masked_region_distance(self):
        delay, doppler = 2, 2
        codeword_book = torch.zeros(2, delay, doppler, dtype=torch.complex64)
        codeword_book[1, 0, 0] = 10.0 + 0.0j
        z_dd = torch.zeros(1, delay, doppler, dtype=torch.complex64)
        data_mask = torch.ones(1, delay, doppler)
        data_mask[:, 0, 0] = 0.0
        prior = TokenCodewordPrior(codeword_book=codeword_book, temperature=0.5)

        output = token_codeword_posterior_prox(z_dd, prior, data_mask=data_mask)

        self.assertTrue(torch.allclose(output.posterior_logits[:, 0], output.posterior_logits[:, 1]))
        self.assertTrue(torch.allclose(output.posterior_weights, torch.full((1, 2), 0.5), atol=1e-6))

    def test_topk_candidates_returns_indices_shape(self):
        batch, vocab, delay, doppler = 2, 7, 3, 4
        z_dd = _complex_dd(batch, delay, doppler)
        prior = TokenCodewordPrior(codeword_book=_complex_dd(vocab, delay, doppler), temperature=1.0)

        output = token_codeword_posterior_prox(z_dd, prior, topk_candidates=3)

        self.assertEqual(output.posterior_logits.shape, (batch, 3))
        self.assertEqual(output.posterior_weights.shape, (batch, 3))
        self.assertEqual(output.candidate_indices.shape, (batch, 3))
        self.assertEqual(output.candidate_indices.dtype, torch.long)

    def test_posterior_variance_shape(self):
        z_dd = _complex_dd(2, 3, 4)
        prior = TokenCodewordPrior(codeword_book=_complex_dd(5, 3, 4), temperature=1.0)

        output = token_codeword_posterior_prox(z_dd, prior)

        self.assertEqual(output.posterior_variance.shape, (2, 3, 4))
        self.assertFalse(torch.isnan(output.posterior_variance).any())

    def test_backward_flows_to_z_dd_and_codeword_book(self):
        z_dd = _complex_dd(2, 3, 4).requires_grad_()
        codeword_book = _complex_dd(5, 3, 4).requires_grad_()
        prior = TokenCodewordPrior(codeword_book=codeword_book, temperature=1.0)

        output = token_codeword_posterior_prox(z_dd, prior)
        loss = output.projected_dd.abs().mean() + output.posterior_logits.mean()
        loss.backward()

        self.assertIsNotNone(z_dd.grad)
        self.assertIsNotNone(codeword_book.grad)
        self.assertGreater(float(z_dd.grad.abs().sum().item()), 0.0)
        self.assertGreater(float(codeword_book.grad.abs().sum().item()), 0.0)

    def test_tensor_temperature_batch_shape_is_supported(self):
        batch, vocab, delay, doppler = 2, 5, 3, 4
        z_dd = _complex_dd(batch, delay, doppler)
        prior = TokenCodewordPrior(codeword_book=_complex_dd(vocab, delay, doppler), temperature=1.0)
        temperature = torch.tensor([0.5, 2.0])

        output = token_codeword_posterior_prox(z_dd, prior, temperature=temperature)

        self.assertEqual(output.posterior_logits.shape, (batch, vocab))
        self.assertEqual(output.posterior_weights.shape, (batch, vocab))
        self.assertTrue(torch.allclose(output.posterior_weights.sum(dim=-1), torch.ones(batch), atol=1e-6))

    def test_higher_temperature_makes_posterior_more_uniform(self):
        delay, doppler = 2, 2
        codeword_book = torch.zeros(3, delay, doppler, dtype=torch.complex64)
        codeword_book[0, 0, 0] = 0.0 + 0.0j
        codeword_book[1, 0, 0] = 1.0 + 0.0j
        codeword_book[2, 0, 0] = 2.0 + 0.0j
        z_dd = codeword_book[0:1].clone()
        prior = TokenCodewordPrior(codeword_book=codeword_book, temperature=1.0)

        low = token_codeword_posterior_prox(z_dd, prior, temperature=torch.tensor([0.05]))
        high = token_codeword_posterior_prox(z_dd, prior, temperature=torch.tensor([10.0]))

        uniform = torch.full_like(high.posterior_weights, 1.0 / codeword_book.shape[0])
        low_deviation = (low.posterior_weights - uniform).abs().sum()
        high_deviation = (high.posterior_weights - uniform).abs().sum()
        self.assertLess(float(high_deviation.item()), float(low_deviation.item()))


if __name__ == "__main__":
    unittest.main()
