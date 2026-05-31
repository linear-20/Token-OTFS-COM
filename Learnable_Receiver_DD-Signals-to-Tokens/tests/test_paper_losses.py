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
    contrastive_token_loss,
    dd_data_consistency_loss,
    denoiser_delta_regularization,
    evidence_head_diversity_loss,
    paper_receiver_loss,
    physics_refinement_loss,
    token_ce_loss,
    token_cross_entropy,
    token_dd_codeword_loss,
    token_posterior_nll_loss,
    total_receiver_loss,
    weighted_complex_mse,
)
from receiver.dd_ops import SparseDDOperator, SparseDDOperatorState
from receiver.sparse_channel import SparseChannelEstimate
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


def _complex_dd(batch, config, **kwargs):
    real = torch.randn(batch, config.M, config.N, **kwargs)
    imag = torch.randn(batch, config.M, config.N, **kwargs)
    return torch.complex(real, imag)


class WeightedComplexMSETests(unittest.TestCase):
    def test_scalar_finite(self):
        config = _config()
        pred = _complex_dd(2, config)
        target = _complex_dd(2, config)
        loss = weighted_complex_mse(pred, target)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss).item())

    def test_zero_mask_yields_zero(self):
        config = _config()
        pred = _complex_dd(2, config)
        target = _complex_dd(2, config)
        mask = torch.zeros(1, config.M, config.N)
        loss = weighted_complex_mse(pred, target, mask=mask)
        self.assertLessEqual(float(loss.item()), 1e-6)

    def test_noise_var_scale_matters(self):
        config = _config()
        pred = _complex_dd(1, config)
        target = pred + 0.1 * _complex_dd(1, config)
        loss_no_nv = weighted_complex_mse(pred, target)
        loss_with_nv = weighted_complex_mse(pred, target, noise_var=100.0)
        self.assertGreater(float(loss_no_nv.item()), float(loss_with_nv.item()))

    def test_perfect_match_is_zero(self):
        config = _config()
        pred = _complex_dd(1, config)
        loss = weighted_complex_mse(pred, pred.clone())
        self.assertLess(float(loss.item()), 1e-8)


class DDDataConsistencyTests(unittest.TestCase):
    def test_calls_sparse_operator_apply_and_returns_finite(self):
        config = _config()
        dd_op = SparseDDOperator(config)
        x_dd = _complex_dd(2, config)
        y_dd = _complex_dd(2, config)
        path_indices = torch.tensor([[[0, 0], [1, 2]], [[0, 1], [2, 2]]])
        path_gains = torch.ones(2, 2, dtype=torch.complex64)
        state = SparseDDOperatorState(
            h_dd=y_dd,
            support_mask=torch.ones_like(y_dd.real),
            path_indices=path_indices,
            path_gains=path_gains,
            operator_mode="ongrid",
            kernel_radius=0,
            kernel_type="linear",
            normalize_kernel=True,
        )
        loss = dd_data_consistency_loss(x_dd, y_dd, dd_op, state)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss).item())

    def test_data_mask_restricts_region(self):
        config = _config()
        dd_op = SparseDDOperator(config)
        x_dd = torch.zeros(1, config.M, config.N, dtype=torch.complex64)
        y_dd = torch.zeros(1, config.M, config.N, dtype=torch.complex64)
        y_dd[:, 0, 0] = 1.0 + 0.0j
        data_mask = torch.zeros(1, config.M, config.N)
        data_mask[:, 0, 0] = 1.0
        path_indices = torch.tensor([[[0, 0]]])
        path_gains = torch.ones(1, 1, dtype=torch.complex64)
        state = SparseDDOperatorState(
            h_dd=y_dd,
            support_mask=torch.ones_like(y_dd.real),
            path_indices=path_indices,
            path_gains=path_gains,
            operator_mode="ongrid",
            kernel_radius=0,
            kernel_type="linear",
            normalize_kernel=True,
        )
        loss_full = dd_data_consistency_loss(x_dd, y_dd, dd_op, state)
        loss_masked = dd_data_consistency_loss(x_dd, y_dd, dd_op, state, data_mask=data_mask)
        self.assertFalse(torch.allclose(loss_full, loss_masked))


class TokenDDCodewordLossTests(unittest.TestCase):
    def test_uses_complex_codeword_not_embedding(self):
        config = _config()
        x_dd = _complex_dd(2, config)
        token_ids = torch.tensor([0, 1])
        # complex [V, M, N] — the DD codeword book
        dd_codeword = _complex_dd(config.vocab_size, config)

        loss = token_dd_codeword_loss(x_dd, token_ids, dd_codeword)

        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss).item())

    def test_perfect_match_is_zero(self):
        config = _config()
        dd_codeword = _complex_dd(config.vocab_size, config)
        token_ids = torch.tensor([3, 7])
        x_dd = dd_codeword[token_ids].clone()

        loss = token_dd_codeword_loss(x_dd, token_ids, dd_codeword)

        self.assertLess(float(loss.item()), 1e-6)

    def test_data_mask_reduces_supervision_region(self):
        config = _config()
        x_dd = _complex_dd(1, config)
        token_ids = torch.tensor([5])
        dd_codeword = _complex_dd(config.vocab_size, config)
        data_mask = torch.ones(1, config.M, config.N)
        data_mask[:, 0, 0] = 0.0

        loss = token_dd_codeword_loss(x_dd, token_ids, dd_codeword, data_mask=data_mask)

        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss).item())


class TokenPosteriorNLLTests(unittest.TestCase):
    def test_full_vocab_ce(self):
        logits = torch.randn(4, 10)
        token_ids = torch.tensor([2, 5, 1, 8])
        loss = token_posterior_nll_loss(logits, token_ids)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss).item())

    def test_topk_with_hits(self):
        logits = torch.tensor([[0.1, 0.2, 0.5], [0.3, 0.6, 0.1]])
        token_ids = torch.tensor([2, 1])
        candidates = torch.tensor([[5, 3, 2], [1, 4, 7]])
        loss = token_posterior_nll_loss(logits, token_ids, candidate_indices=candidates)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss).item())

    def test_topk_miss_adds_penalty(self):
        logits = torch.tensor([[0.5, 0.3, 0.2]])
        token_ids = torch.tensor([9])  # 9 is not in candidates
        candidates = torch.tensor([[0, 1, 2]])
        loss_no_penalty = token_posterior_nll_loss(
            logits, token_ids, candidate_indices=candidates, missing_penalty=0.0,
        )
        loss_with_penalty = token_posterior_nll_loss(
            logits, token_ids, candidate_indices=candidates, missing_penalty=10.0,
        )
        self.assertGreater(float(loss_with_penalty.item()), float(loss_no_penalty.item()))

    def test_topk_miss_no_incorrect_ce_gradient(self):
        logits = torch.tensor([[0.5, 0.3, 0.2]], requires_grad=True)
        token_ids = torch.tensor([9])
        candidates = torch.tensor([[0, 1, 2]])
        loss = token_posterior_nll_loss(
            logits, token_ids, candidate_indices=candidates, missing_penalty=10.0,
        )
        loss.backward()
        # miss sample: gradient should be zero (no CE against candidate 0)
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.allclose(logits.grad, torch.zeros_like(logits.grad)))

    def test_topk_mixed_hit_miss_batch(self):
        logits = torch.tensor([[0.5, 0.3, 0.2], [0.1, 0.8, 0.1]])
        token_ids = torch.tensor([2, 9])   # first hits, second misses
        candidates = torch.tensor([[5, 3, 2], [0, 1, 2]])
        loss = token_posterior_nll_loss(logits, token_ids, candidate_indices=candidates)
        self.assertTrue(torch.isfinite(loss).item())
        # loss should be > penalty/2 since first sample has CE > 0
        self.assertGreater(float(loss.item()), 5.0)

    def test_topk_all_miss_returns_finite(self):
        logits = torch.tensor([[0.5, 0.3, 0.2], [0.1, 0.6, 0.2]])
        token_ids = torch.tensor([9, 8])
        candidates = torch.tensor([[0, 1, 2], [3, 4, 5]])
        loss = token_posterior_nll_loss(
            logits, token_ids, candidate_indices=candidates, missing_penalty=10.0,
        )
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss).item())
        self.assertAlmostEqual(float(loss.item()), 10.0, places=1)


class DenoiserDeltaRegularizationTests(unittest.TestCase):
    def test_high_reliability_gives_higher_weight(self):
        config = _config()
        # delta concentrated at (0,0) in a grid with M=5,N=4
        delta = torch.zeros(1, config.M, config.N, dtype=torch.complex64)
        delta[:, 0, 0] = 1.0 + 0.0j
        # reliability high at (0,0) → large penalty
        rel_high_spot = torch.zeros(1, config.M, config.N)
        rel_high_spot[:, 0, 0] = 1.0
        rel_high_spot[:, 0, 1] = 0.01  # small weight elsewhere for denom stability
        # reliability low at (0,0) → small penalty
        rel_low_spot = torch.zeros(1, config.M, config.N)
        rel_low_spot[:, 0, 0] = 0.01
        rel_low_spot[:, 0, 1] = 1.0

        loss_high = denoiser_delta_regularization(delta, reliability_map=rel_high_spot)
        loss_low = denoiser_delta_regularization(delta, reliability_map=rel_low_spot)

        self.assertGreater(float(loss_high.item()), float(loss_low.item()))

    def test_zero_delta_gives_zero_loss(self):
        config = _config()
        delta = torch.zeros(1, config.M, config.N, dtype=torch.complex64)
        loss = denoiser_delta_regularization(delta)
        self.assertLess(float(loss.item()), 1e-8)


class PhysicsRefinementLossTests(unittest.TestCase):
    def test_uses_fit_residual_not_full_residual(self):
        config = _config()
        estimate = SparseChannelEstimate(
            h_dd=_complex_dd(1, config),
            support_mask=torch.ones(1, config.M, config.N),
            path_indices=torch.zeros(1, 2, 2, dtype=torch.long),
            path_gains=torch.ones(1, 2, dtype=torch.complex64),
            pilot_residual_power=torch.tensor([0.01]),
            physics_fit_residual_power=torch.tensor([0.02]),
            physics_full_residual_power=torch.tensor([5.0]),
            gain_covariance_diag=torch.tensor([[0.1, 0.2]]),
            physics_refined=True,
        )
        loss = physics_refinement_loss(estimate)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss).item())
        val = float(loss.item())
        # loss averages {pilot, fit, cov} = (0.01 + 0.02 + 0.15) / 3 ≈ 0.06
        # full_residual_power = 5.0 was excluded
        self.assertLess(val, 1.0)

    def test_no_physics_fields_returns_zero(self):
        config = _config()
        estimate = SparseChannelEstimate(
            h_dd=_complex_dd(1, config),
            support_mask=torch.ones(1, config.M, config.N),
            path_indices=torch.zeros(1, 2, 2, dtype=torch.long),
            path_gains=torch.ones(1, 2, dtype=torch.complex64),
        )
        loss = physics_refinement_loss(estimate)
        self.assertLessEqual(float(loss.item()), 1e-8)


class PaperReceiverLossTests(unittest.TestCase):
    def test_runs_on_receiver_output_with_return_details(self):
        config = _config()
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)
        output = model(y_dd, return_details=True)
        token_ids = torch.tensor([3, 7])

        result = paper_receiver_loss(output, token_ids)

        self.assertIsInstance(result, ReceiverLossOutput)
        self.assertEqual(result.total.ndim, 0)
        self.assertTrue(torch.isfinite(result.total).item())
        self.assertIn("token_ce", result.components)

    def test_return_components_false_returns_scalar(self):
        config = _config()
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)
        output = model(y_dd, return_details=True)
        token_ids = torch.tensor([0, 1])

        total = paper_receiver_loss(output, token_ids, return_components=False)

        self.assertEqual(total.ndim, 0)
        self.assertIsInstance(total, torch.Tensor)

    def test_dd_codeword_from_token_prior(self):
        config = _config()
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)
        dd_codeword = _complex_dd(config.vocab_size, config)
        prior = TokenCodewordPrior(codeword_book=dd_codeword, prior_strength=1.0)
        output = model(y_dd, token_prior=prior, return_details=True)
        token_ids = torch.tensor([2, 5])
        weights = ReceiverLossWeights(
            token_ce_weight=1.0,
            dd_codeword_weight=0.5,
        )

        result = paper_receiver_loss(
            output, token_ids, token_prior=prior, weights=weights,
        )

        self.assertIn("dd_codeword", result.components)
        self.assertTrue(torch.isfinite(result.total).item())

    def test_data_consistency_uses_sparse_operator(self):
        config = _config()
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)
        output = model(y_dd, return_details=True)
        token_ids = torch.tensor([0, 1])
        dd_op = SparseDDOperator(config)
        weights = ReceiverLossWeights(
            token_ce_weight=1.0,
            data_consistency_weight=0.5,
        )

        result = paper_receiver_loss(
            output, token_ids, y_dd=y_dd, dd_operator=dd_op, weights=weights,
        )

        self.assertIn("data_consistency", result.components)
        self.assertTrue(torch.isfinite(result.total).item())

    def test_posterior_nll_from_detector(self):
        config = _config(token_posterior_topk=5)
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)
        dd_codeword = _complex_dd(config.vocab_size, config)
        prior = TokenCodewordPrior(codeword_book=dd_codeword, prior_strength=1.0)
        output = model(y_dd, token_prior=prior, return_details=True)
        token_ids = torch.tensor([3, 7])
        weights = ReceiverLossWeights(
            token_ce_weight=1.0,
            posterior_nll_weight=0.3,
        )

        result = paper_receiver_loss(
            output, token_ids, token_prior=prior, weights=weights,
        )

        self.assertIn("posterior_nll", result.components)
        self.assertTrue(torch.isfinite(result.total).item())

    def test_denoiser_delta_regularization_runs(self):
        config = _config()
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)
        output = model(y_dd, return_details=True)
        token_ids = torch.tensor([1, 2])
        weights = ReceiverLossWeights(
            token_ce_weight=1.0,
            denoiser_delta_weight=0.1,
        )

        result = paper_receiver_loss(output, token_ids, weights=weights)

        self.assertIn("denoiser_delta", result.components)
        self.assertTrue(torch.isfinite(result.total).item())

    def test_physics_refinement_loss_runs(self):
        config = _config(
            use_physics_guided_gain_refinement=True,
            physics_refinement_ridge=1e-4,
            physics_refinement_allow_full_grid_debug=True,
        )
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)
        output = model(y_dd, return_details=True)
        token_ids = torch.tensor([3, 7])
        weights = ReceiverLossWeights(
            token_ce_weight=1.0,
            physics_ce_weight=0.2,
        )

        result = paper_receiver_loss(output, token_ids, weights=weights)

        self.assertIn("physics_ce", result.components)
        self.assertTrue(torch.isfinite(result.total).item())

    def test_channel_supervised_nmse_runs(self):
        config = _config()
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)
        h_true = _complex_dd(2, config)
        output = model(y_dd, h_dd=h_true, return_details=True)
        token_ids = torch.tensor([0, 1])
        weights = ReceiverLossWeights(
            token_ce_weight=1.0,
            channel_supervised_weight=0.2,
        )

        result = paper_receiver_loss(output, token_ids, h_true=h_true, weights=weights)

        self.assertIn("channel_supervised", result.components)
        self.assertTrue(torch.isfinite(result.total).item())

    def test_embedding_contrastive_with_explicit_codebook(self):
        config = _config()
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)
        output = model(y_dd, return_details=True)
        token_ids = torch.tensor([2, 5])
        emb_codebook = torch.randn(config.vocab_size, config.token_embedding_dim)
        weights = ReceiverLossWeights(
            token_ce_weight=1.0,
            embedding_contrastive_weight=0.5,
        )
        result = paper_receiver_loss(
            output, token_ids, embedding_codebook=emb_codebook, weights=weights,
        )
        self.assertIn("embedding_contrastive", result.components)
        self.assertTrue(torch.isfinite(result.total).item())

    def test_embedding_contrastive_skipped_without_explicit_codebook(self):
        config = _config()
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)
        output = model(y_dd, return_details=True)
        token_ids = torch.tensor([3, 7])
        weights = ReceiverLossWeights(
            token_ce_weight=1.0,
            embedding_contrastive_weight=0.5,
        )
        result = paper_receiver_loss(output, token_ids, weights=weights)
        self.assertNotIn("embedding_contrastive", result.components)
        self.assertTrue(torch.isfinite(result.total).item())

    def test_dd_codeword_book_not_used_as_embedding_codebook(self):
        config = _config()
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)
        dd_codeword = _complex_dd(config.vocab_size, config)  # [V, M, N] complex
        prior = TokenCodewordPrior(codeword_book=dd_codeword, prior_strength=1.0)
        output = model(y_dd, token_prior=prior, return_details=True)
        token_ids = torch.tensor([0, 1])
        weights = ReceiverLossWeights(
            token_ce_weight=1.0,
            embedding_contrastive_weight=0.5,
        )
        # No embedding_codebook passed — DD codeword must not be silently reused
        result = paper_receiver_loss(
            output, token_ids, token_prior=prior, weights=weights,
        )
        self.assertNotIn("embedding_contrastive", result.components)
        # DD codeword loss still works via its own path
        self.assertIn("token_ce", result.components)

    def test_evidence_head_diversity_loss_h1_zero(self):
        weights = torch.rand(2, 1, 5, 4, requires_grad=True)

        loss = evidence_head_diversity_loss(weights)
        loss.backward()

        self.assertLessEqual(float(loss.item()), 1e-8)
        self.assertIsNotNone(weights.grad)

    def test_evidence_head_diversity_loss_h4_finite_with_batched_mask(self):
        weights = torch.rand(2, 4, 5, 4)
        weights = weights / weights.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        data_mask = torch.ones(2, 5, 4)
        data_mask[:, 0, :] = 0.0

        loss = evidence_head_diversity_loss(weights, data_mask=data_mask)

        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss).item())

    def test_paper_receiver_loss_includes_classifier_head_diversity(self):
        config = _config(classifier_num_evidence_heads=3)
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)
        output = model(y_dd, return_details=True)
        token_ids = torch.tensor([0, 1])
        data_mask = torch.ones(2, config.M, config.N)
        weights = ReceiverLossWeights(
            token_ce_weight=1.0,
            classifier_head_diversity_weight=0.2,
        )

        result = paper_receiver_loss(
            output,
            token_ids,
            data_mask=data_mask,
            weights=weights,
        )

        self.assertIn("classifier_head_diversity", result.components)
        self.assertTrue(torch.isfinite(result.total).item())


class BackwardCompatibilityTests(unittest.TestCase):
    def test_default_model_returns_logits_tensor(self):
        config = _config()
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)
        logits = model(y_dd)
        self.assertIsInstance(logits, torch.Tensor)
        self.assertEqual(logits.shape, (2, config.vocab_size))

    def test_old_total_receiver_loss_still_works(self):
        config = _config()
        logits = torch.randn(4, config.vocab_size)
        token_ids = torch.tensor([2, 5, 1, 8])
        loss = total_receiver_loss(logits, token_ids)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss).item())

    def test_old_contrastive_token_loss_still_works(self):
        config = _config()
        rx_embedding = torch.randn(4, config.token_embedding_dim)
        codebook = torch.randn(config.vocab_size, config.token_embedding_dim)
        token_ids = torch.tensor([3, 1, 7, 2])
        loss = contrastive_token_loss(rx_embedding, codebook, token_ids)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss).item())

    def test_aux_ce_runs_with_old_and_new_api(self):
        config = _config()
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)
        output = model(y_dd, return_details=True, return_aux=True)
        token_ids = torch.tensor([4, 9])
        # old API
        old_loss = total_receiver_loss(
            output.token_logits, token_ids,
            aux_logits=output.aux_logits, aux_ce_weight=0.5,
        )
        self.assertTrue(torch.isfinite(old_loss).item())
        # new API
        weights = ReceiverLossWeights(token_ce_weight=1.0, aux_ce_weight=0.5)
        new_result = paper_receiver_loss(output, token_ids, weights=weights)
        self.assertIn("aux_ce", new_result.components)

    def test_receiver_loss_weights_default_only_token_ce(self):
        w = ReceiverLossWeights()
        self.assertEqual(w.token_ce_weight, 1.0)
        self.assertEqual(w.aux_ce_weight, 0.0)
        self.assertEqual(w.data_consistency_weight, 0.0)

    def test_weighted_complex_mse_handles_batch1_mask(self):
        config = _config()
        pred = _complex_dd(2, config)
        target = _complex_dd(2, config)
        mask = torch.ones(1, config.M, config.N)
        loss = weighted_complex_mse(pred, target, mask=mask)
        self.assertTrue(torch.isfinite(loss).item())

    def test_losses_file_has_no_garbled_unicode(self):
        losses_path = Path(__file__).resolve().parents[0].parent / "receiver" / "losses.py"
        content = losses_path.read_text(encoding="utf-8")
        # Any non-ASCII character causes mojibake in CJK terminals.
        # This covers box-drawing, em-dash, arrows, math symbols, and
        # literal CJK codepoints that appear as renderings of UTF-8 bytes
        # misinterpreted as GBK/Shift-JIS/etc.
        for i, ch in enumerate(content):
            if ord(ch) > 127:
                line = content[:i].count("\n") + 1
                self.fail(
                    f"losses.py line {line}: non-ASCII char U+{ord(ch):04X} "
                    f"({repr(ch)}) causes mojibake rendering"
                )


if __name__ == "__main__":
    unittest.main()
