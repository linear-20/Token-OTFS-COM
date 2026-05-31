from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from receiver.token_prior import TokenCodewordPrior, TokenPriorProjection, token_codeword_projection


def _complex_dd(batch: int, M: int, N: int) -> torch.Tensor:
    real = torch.randn(batch, M, N)
    imag = torch.randn(batch, M, N)
    return torch.complex(real, imag)


class TokenPriorTests(unittest.TestCase):
    def test_token_codeword_projection_shapes(self):
        codeword_book = _complex_dd(7, 4, 5)
        z_dd = _complex_dd(3, 4, 5)
        prior = TokenCodewordPrior(codeword_book=codeword_book)

        projection = token_codeword_projection(z_dd, prior)

        self.assertIsInstance(projection, TokenPriorProjection)
        self.assertEqual(projection.projected_dd.shape, (3, 4, 5))
        self.assertEqual(projection.token_prior_logits.shape, (3, 7))
        self.assertEqual(projection.token_weights.shape, (3, 7))
        self.assertTrue(torch.is_complex(projection.projected_dd))

    def test_token_weights_sum_to_one(self):
        codeword_book = _complex_dd(5, 4, 5)
        z_dd = _complex_dd(2, 4, 5)
        prior = TokenCodewordPrior(codeword_book=codeword_book)

        projection = token_codeword_projection(z_dd, prior)

        self.assertTrue(torch.allclose(projection.token_weights.sum(dim=-1), torch.ones(2), atol=1e-6, rtol=1e-6))

    def test_projection_is_differentiable(self):
        codeword_book = _complex_dd(6, 4, 5).requires_grad_()
        z_dd = _complex_dd(2, 4, 5).requires_grad_()
        prior = TokenCodewordPrior(codeword_book=codeword_book)

        projection = token_codeword_projection(z_dd, prior)
        loss = projection.projected_dd.abs().mean() + projection.token_prior_logits.mean()
        loss.backward()

        self.assertIsNotNone(z_dd.grad)
        self.assertIsNotNone(codeword_book.grad)
        self.assertGreater(float(z_dd.grad.abs().sum().item()), 0.0)
        self.assertGreater(float(codeword_book.grad.abs().sum().item()), 0.0)

    def test_similarity_modes(self):
        codeword_book = _complex_dd(5, 4, 5)
        z_dd = _complex_dd(2, 4, 5)

        real_projection = token_codeword_projection(
            z_dd,
            TokenCodewordPrior(codeword_book=codeword_book, similarity="real"),
        )
        abs_projection = token_codeword_projection(
            z_dd,
            TokenCodewordPrior(codeword_book=codeword_book, similarity="abs"),
        )

        self.assertEqual(real_projection.token_prior_logits.shape, (2, 5))
        self.assertEqual(abs_projection.token_prior_logits.shape, (2, 5))
        with self.assertRaises(ValueError):
            TokenCodewordPrior(codeword_book=codeword_book, similarity="bad")


if __name__ == "__main__":
    unittest.main()
