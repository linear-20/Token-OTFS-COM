from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from receiver import (
    LearnableOTFSReceiver,
    ReceiverConfig,
    ReceiverLossWeights,
    TokenCandidateSelection,
    candidate_margin_regularization,
    candidate_recall_metric,
    paper_receiver_loss,
    select_token_candidates_by_sketch,
    token_codeword_posterior_prox,
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


def _complex_dd(batch, config, **kwargs):
    real = torch.randn(batch, config.M, config.N, **kwargs)
    imag = torch.randn(batch, config.M, config.N, **kwargs)
    return torch.complex(real, imag)


class CandidateSelectionTests(unittest.TestCase):
    def test_sketch_selector_shape(self):
        config = _config()
        z_dd = _complex_dd(2, config)
        codeword = _complex_dd(config.vocab_size, config)
        prior = TokenCodewordPrior(codeword_book=codeword)

        sel = select_token_candidates_by_sketch(z_dd, prior, num_candidates=5)

        self.assertIsInstance(sel, TokenCandidateSelection)
        self.assertEqual(sel.candidate_indices.shape, (2, 5))
        self.assertEqual(sel.candidate_scores.shape, (2, 5))

    def test_sketch_scores_finite(self):
        config = _config()
        z_dd = _complex_dd(1, config)
        codeword = _complex_dd(config.vocab_size, config)
        prior = TokenCodewordPrior(codeword_book=codeword)

        sel = select_token_candidates_by_sketch(z_dd, prior, num_candidates=3)

        self.assertTrue(torch.isfinite(sel.candidate_scores).all())

    def test_num_candidates_exceeds_vocab_size_raises(self):
        config = _config()
        z_dd = _complex_dd(1, config)
        codeword = _complex_dd(config.vocab_size, config)
        prior = TokenCodewordPrior(codeword_book=codeword)

        with self.assertRaises(ValueError):
            select_token_candidates_by_sketch(
                z_dd, prior, num_candidates=config.vocab_size + 1,
            )

    def test_sketch_uses_data_mask(self):
        config = _config()
        z_dd = _complex_dd(1, config)
        codeword = _complex_dd(config.vocab_size, config)
        prior = TokenCodewordPrior(codeword_book=codeword)
        mask = torch.ones(1, config.M, config.N)
        mask[:, 0, 0] = 0.0

        sel_full = select_token_candidates_by_sketch(z_dd, prior, num_candidates=3)
        sel_masked = select_token_candidates_by_sketch(
            z_dd, prior, num_candidates=3, data_mask=mask,
        )

        self.assertFalse(torch.equal(sel_full.candidate_scores, sel_masked.candidate_scores))

    def test_energy_topk_selector_shape(self):
        config = _config()
        z_dd = _complex_dd(2, config)
        codeword = _complex_dd(config.vocab_size, config)
        prior = TokenCodewordPrior(codeword_book=codeword)

        sel = select_token_candidates_by_sketch(
            z_dd, prior, num_candidates=5, sketch_mode="energy_topk",
        )

        self.assertEqual(sel.candidate_indices.shape, (2, 5))
        self.assertIsNotNone(sel.sketch_indices)
        self.assertEqual(sel.sketch_indices.shape, (2, sel.sketch_size))

    def test_hybrid_selector_shape(self):
        config = _config()
        z_dd = _complex_dd(2, config)
        codeword = _complex_dd(config.vocab_size, config)
        prior = TokenCodewordPrior(codeword_book=codeword)

        sel = select_token_candidates_by_sketch(
            z_dd, prior, num_candidates=4, sketch_mode="hybrid",
        )

        self.assertEqual(sel.mode, "hybrid")
        self.assertTrue(torch.isfinite(sel.candidate_scores).all())

    def test_sketch_indices_field_present(self):
        config = _config()
        z_dd = _complex_dd(1, config)
        codeword = _complex_dd(config.vocab_size, config)
        prior = TokenCodewordPrior(codeword_book=codeword)

        for sm in ("strided", "energy_topk", "hybrid"):
            sel = select_token_candidates_by_sketch(
                z_dd, prior, num_candidates=3, sketch_mode=sm,
            )
            self.assertIsNotNone(sel.sketch_indices, f"missing sketch_indices for {sm}")
            self.assertEqual(sel.mode, sm)

    def test_sketch_size_exceeds_MN_clips(self):
        config = _config()
        z_dd = _complex_dd(1, config)
        codeword = _complex_dd(config.vocab_size, config)
        prior = TokenCodewordPrior(codeword_book=codeword)
        huge = config.M * config.N + 100

        sel = select_token_candidates_by_sketch(
            z_dd, prior, num_candidates=3, sketch_size=huge,
        )

        self.assertLessEqual(sel.sketch_size, config.M * config.N)

    def test_corr_score_mode_returns_finite(self):
        config = _config()
        z_dd = _complex_dd(1, config)
        codeword = _complex_dd(config.vocab_size, config)
        prior = TokenCodewordPrior(codeword_book=codeword)

        sel = select_token_candidates_by_sketch(
            z_dd, prior, num_candidates=4, sketch_mode="hybrid", score_mode="corr",
        )

        self.assertTrue(torch.isfinite(sel.candidate_scores).all())

    def test_energy_weighted_distance_score_mode_returns_diagnostics(self):
        config = _config()
        z_dd = _complex_dd(2, config)
        codeword = _complex_dd(config.vocab_size, config)
        prior = TokenCodewordPrior(codeword_book=codeword)
        confidence = torch.rand(2, config.M, config.N)
        uncertainty = torch.rand(2, config.M, config.N)

        sel = select_token_candidates_by_sketch(
            z_dd,
            prior,
            num_candidates=4,
            sketch_mode="hybrid",
            score_mode="energy_weighted_distance",
            confidence_map=confidence,
            uncertainty_map=uncertainty,
        )

        self.assertEqual(sel.score_mode, "energy_weighted_distance")
        self.assertEqual(sel.candidate_entropy.shape, (2,))
        self.assertEqual(sel.candidate_margin.shape, (2,))
        self.assertTrue(torch.isfinite(sel.candidate_scores).all())
        self.assertTrue(((sel.candidate_entropy >= 0.0) & (sel.candidate_entropy <= 1.0)).all())
        self.assertTrue(((sel.candidate_margin >= 0.0) & (sel.candidate_margin <= 1.0)).all())

    def test_energy_weighted_distance_all_zero_weight_is_finite(self):
        config = _config()
        z_dd = _complex_dd(1, config)
        codeword = _complex_dd(config.vocab_size, config)
        prior = TokenCodewordPrior(codeword_book=codeword)
        zero_mask = torch.zeros(1, config.M, config.N)

        sel = select_token_candidates_by_sketch(
            z_dd,
            prior,
            num_candidates=3,
            data_mask=zero_mask,
            confidence_map=zero_mask,
            sketch_mode="energy_topk",
            score_mode="energy_weighted_distance",
        )

        self.assertEqual(sel.candidate_indices.shape, (1, 3))
        self.assertTrue(torch.isfinite(sel.candidate_scores).all())

    def test_adaptive_selector_expands_uncertain_candidates(self):
        config = _config()
        z_dd = torch.zeros(2, config.M, config.N, dtype=torch.complex64)
        codeword = torch.zeros(config.vocab_size, config.M, config.N, dtype=torch.complex64)
        prior = TokenCodewordPrior(codeword_book=codeword)

        sel = select_token_candidates_by_sketch(
            z_dd,
            prior,
            num_candidates=2,
            adaptive=True,
            max_candidates=5,
            entropy_threshold=0.0,
            margin_threshold=0.9,
            expand_factor=3.0,
        )

        self.assertEqual(sel.candidate_indices.shape, (2, 5))
        self.assertEqual(sel.effective_candidate_count, 5)
        self.assertTrue(sel.fallback_used.all())
        self.assertTrue(torch.allclose(sel.candidate_margin, torch.zeros_like(sel.candidate_margin)))

    def test_adaptive_selector_can_expand_to_full_vocab(self):
        config = _config()
        z_dd = torch.zeros(1, config.M, config.N, dtype=torch.complex64)
        codeword = torch.zeros(config.vocab_size, config.M, config.N, dtype=torch.complex64)
        prior = TokenCodewordPrior(codeword_book=codeword)

        sel = select_token_candidates_by_sketch(
            z_dd,
            prior,
            num_candidates=3,
            adaptive=True,
            max_candidates=config.vocab_size,
            entropy_threshold=0.0,
            margin_threshold=0.9,
            expand_factor=10.0,
            allow_full_fallback=True,
        )

        self.assertEqual(sel.candidate_indices.shape, (1, config.vocab_size))
        self.assertEqual(sel.effective_candidate_count, config.vocab_size)
        self.assertTrue(bool(sel.fallback_used.item()))

    def test_energy_topk_beats_strided_on_sparse_dd(self):
        """Sparse DD: non-zero positions outside strided grid.
        energy_topk or hybrid must score the true codeword higher."""
        config = _config()
        # only position (M-1, N-1) = (4, 3) has signal; strided grid may skip it
        z_dd = torch.zeros(1, config.M, config.N, dtype=torch.complex64)
        z_dd[0, 4, 3] = 2.0 + 1.0j
        # codeword 0 matches this pattern; others are random noise
        codeword = torch.randn(config.vocab_size, config.M, config.N, dtype=torch.complex64) * 0.1
        codeword[0, 4, 3] = 2.0 + 1.0j  # token 0 matches
        prior = TokenCodewordPrior(codeword_book=codeword)

        sel_strided = select_token_candidates_by_sketch(
            z_dd, prior, num_candidates=5, sketch_size=10, sketch_mode="strided",
        )
        sel_energy = select_token_candidates_by_sketch(
            z_dd, prior, num_candidates=5, sketch_size=10, sketch_mode="energy_topk",
        )
        sel_hybrid = select_token_candidates_by_sketch(
            z_dd, prior, num_candidates=5, sketch_size=10, sketch_mode="hybrid",
        )

        # energy_topk and hybrid should rank token 0 higher
        rank_strided = (sel_strided.candidate_indices[0] == 0).nonzero(as_tuple=True)
        rank_energy = (sel_energy.candidate_indices[0] == 0).nonzero(as_tuple=True)
        rank_hybrid = (sel_hybrid.candidate_indices[0] == 0).nonzero(as_tuple=True)

        self.assertGreater(len(rank_energy[0]), 0, "energy_topk missed matching token")
        self.assertGreater(len(rank_hybrid[0]), 0, "hybrid missed matching token")


class CandidateProxTests(unittest.TestCase):
    def test_candidate_indices_prox_only_computes_K_distances(self):
        # verify prox runs without touching full vocab
        config = _config()
        z_dd = _complex_dd(2, config)
        codeword = _complex_dd(config.vocab_size, config)
        prior = TokenCodewordPrior(codeword_book=codeword)
        candidates = torch.tensor([[0, 3, 5], [2, 7, 1]])

        output = token_codeword_posterior_prox(
            z_dd, prior, candidate_indices=candidates,
        )

        self.assertEqual(output.posterior_logits.shape, (2, 3))
        self.assertEqual(output.posterior_weights.shape, (2, 3))
        self.assertEqual(output.candidate_source, "explicit")

    def test_token_prior_rejects_nan_before_candidate_prox(self):
        # Reject a corrupt artifact at the prior boundary, even if unselected.
        config = _config()
        codeword = _complex_dd(config.vocab_size, config)
        codeword[9] = torch.nan  # corrupt token 9
        with self.assertRaises(ValueError):
            TokenCodewordPrior(codeword_book=codeword)

    def test_candidate_prox_output_shapes(self):
        config = _config()
        z_dd = _complex_dd(1, config)
        codeword = _complex_dd(config.vocab_size, config)
        prior = TokenCodewordPrior(codeword_book=codeword)
        K = 4
        candidates = torch.tensor([[0, 2, 4, 6]])

        output = token_codeword_posterior_prox(
            z_dd, prior, candidate_indices=candidates,
        )

        self.assertEqual(output.posterior_logits.shape, (1, K))
        self.assertEqual(output.posterior_weights.shape, (1, K))
        self.assertEqual(output.projected_dd.shape, z_dd.shape)
        self.assertIsNotNone(output.posterior_variance)

    def test_full_topk_old_behaviour_still_works(self):
        config = _config()
        z_dd = _complex_dd(1, config)
        codeword = _complex_dd(config.vocab_size, config)
        prior = TokenCodewordPrior(codeword_book=codeword)

        output = token_codeword_posterior_prox(z_dd, prior, topk_candidates=5)

        self.assertEqual(output.posterior_logits.shape, (1, 5))
        self.assertEqual(output.candidate_source, "full_topk")

    def test_full_vocab_old_behaviour_still_works(self):
        config = _config()
        z_dd = _complex_dd(1, config)
        codeword = _complex_dd(config.vocab_size, config)
        prior = TokenCodewordPrior(codeword_book=codeword)

        output = token_codeword_posterior_prox(z_dd, prior)

        self.assertEqual(output.posterior_logits.shape, (1, config.vocab_size))
        self.assertEqual(output.candidate_source, "full_vocab")

    def test_candidate_diagnostics_pass_through_prox_output(self):
        config = _config()
        z_dd = _complex_dd(2, config)
        codeword = _complex_dd(config.vocab_size, config)
        prior = TokenCodewordPrior(codeword_book=codeword)
        candidates = torch.tensor([[0, 3, 5], [2, 7, 1]])
        entropy = torch.tensor([0.1, 0.2])
        margin = torch.tensor([0.8, 0.7])
        fallback = torch.tensor([False, True])

        output = token_codeword_posterior_prox(
            z_dd,
            prior,
            candidate_indices=candidates,
            candidate_entropy=entropy,
            candidate_margin=margin,
            fallback_used=fallback,
            effective_candidate_count=3,
            candidate_source="sketch",
        )

        self.assertTrue(torch.equal(output.candidate_entropy, entropy))
        self.assertTrue(torch.equal(output.candidate_margin, margin))
        self.assertTrue(torch.equal(output.fallback_used, fallback))
        self.assertEqual(output.effective_candidate_count, 3)


class EqualizerPruningTests(unittest.TestCase):
    def test_sketch_mode_equalizer_returns_posterior_indices(self):
        config = _config(
            detector_prox_mode="token_posterior",
            token_candidate_pruning_mode="sketch",
            token_candidate_count=4,
            token_candidate_sketch_size=10,
            use_offgrid_refinement=False,
        )
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)
        codeword = _complex_dd(config.vocab_size, config)
        prior = TokenCodewordPrior(codeword_book=codeword)

        output = model(y_dd, token_prior=prior, return_details=True)

        self.assertIsNotNone(output.equalizer_output)
        self.assertIsNotNone(output.equalizer_output.token_posterior_indices)
        self.assertEqual(
            output.equalizer_output.token_posterior_indices[0].shape, (2, 4),
        )
        self.assertTrue(output.receiver_trace["uses_token_candidate_pruning"])

    def test_explicit_mode_without_indices_raises(self):
        config = _config(
            detector_prox_mode="token_posterior",
            token_candidate_pruning_mode="explicit",
            use_offgrid_refinement=False,
        )
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)
        codeword = _complex_dd(config.vocab_size, config)
        prior = TokenCodewordPrior(codeword_book=codeword)

        with self.assertRaises(ValueError):
            model(y_dd, token_prior=prior)

    def test_sketch_mode_default_logits_shape(self):
        config = _config(
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

        logits = model(y_dd, token_prior=prior)

        self.assertIsInstance(logits, torch.Tensor)
        self.assertEqual(logits.shape, (2, config.vocab_size))

    def test_default_no_pruning_old_behaviour(self):
        config = _config(detector_prox_mode="token_posterior",
                         use_offgrid_refinement=False)
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)
        codeword = _complex_dd(config.vocab_size, config)
        prior = TokenCodewordPrior(codeword_book=codeword)

        logits = model(y_dd, token_prior=prior)

        self.assertEqual(logits.shape, (2, config.vocab_size))

    def test_no_bit_qam_no_transformer_no_gnn_in_model(self):
        config = _config(token_candidate_pruning_mode="sketch",
                         token_candidate_count=4)
        model = LearnableOTFSReceiver(config)
        module_names = {m.__class__.__name__.lower() for m in model.modules()}
        self.assertFalse(any("transformer" in n for n in module_names))
        self.assertFalse(any("graph" in n for n in module_names))
        self.assertFalse(hasattr(model, "bit_head"))
        self.assertFalse(hasattr(model, "qam_head"))

    def test_energy_topk_sketch_mode_runs(self):
        config = _config(
            detector_prox_mode="token_posterior",
            token_candidate_pruning_mode="sketch",
            token_candidate_sketch_mode="energy_topk",
            token_candidate_count=4,
            token_candidate_sketch_size=10,
            use_offgrid_refinement=False,
        )
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)
        codeword = _complex_dd(config.vocab_size, config)
        prior = TokenCodewordPrior(codeword_book=codeword)

        output = model(y_dd, token_prior=prior, return_details=True)

        indices = output.equalizer_output.token_posterior_indices
        self.assertIsNotNone(indices)
        self.assertEqual(indices[0].shape, (2, 4))

    def test_corr_score_mode_runs(self):
        config = _config(
            detector_prox_mode="token_posterior",
            token_candidate_pruning_mode="sketch",
            token_candidate_sketch_mode="hybrid",
            token_candidate_score_mode="corr",
            token_candidate_count=5,
            token_candidate_sketch_size=12,
            use_offgrid_refinement=False,
        )
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)
        codeword = _complex_dd(config.vocab_size, config)
        prior = TokenCodewordPrior(codeword_book=codeword)

        logits = model(y_dd, token_prior=prior)

        self.assertEqual(logits.shape, (2, config.vocab_size))

    def test_energy_weighted_distance_score_mode_runs(self):
        config = _config(
            detector_prox_mode="token_posterior",
            token_candidate_pruning_mode="sketch",
            token_candidate_sketch_mode="hybrid",
            token_candidate_score_mode="energy_weighted_distance",
            token_candidate_count=5,
            token_candidate_sketch_size=12,
            use_offgrid_refinement=False,
        )
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)
        codeword = _complex_dd(config.vocab_size, config)
        prior = TokenCodewordPrior(codeword_book=codeword)

        output = model(y_dd, token_prior=prior, return_details=True)

        self.assertEqual(output.token_logits.shape, (2, config.vocab_size))
        self.assertEqual(output.equalizer_output.token_candidate_scores[-1].shape, (2, 5))

    def test_invalid_sketch_mode_raises(self):
        with self.assertRaises(ValueError):
            _config(token_candidate_sketch_mode="invalid")

    def test_candidate_scores_propagate_in_equalizer_output(self):
        config = _config(
            detector_prox_mode="token_posterior",
            token_candidate_pruning_mode="sketch",
            token_candidate_sketch_mode="hybrid",
            token_candidate_count=4,
            token_candidate_sketch_size=10,
            use_offgrid_refinement=False,
        )
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)
        codeword = _complex_dd(config.vocab_size, config)
        prior = TokenCodewordPrior(codeword_book=codeword)

        output = model(y_dd, token_prior=prior, return_details=True)
        eq_out = output.equalizer_output

        self.assertIsNotNone(eq_out.token_candidate_scores)
        self.assertEqual(eq_out.token_candidate_scores[0].shape, (2, 4))
        self.assertTrue(torch.isfinite(eq_out.token_candidate_scores[0]).all())

    def test_candidate_source_is_sketch(self):
        config = _config(
            detector_prox_mode="token_posterior",
            token_candidate_pruning_mode="sketch",
            token_candidate_sketch_mode="hybrid",
            token_candidate_count=4,
            token_candidate_sketch_size=10,
            use_offgrid_refinement=False,
        )
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)
        codeword = _complex_dd(config.vocab_size, config)
        prior = TokenCodewordPrior(codeword_book=codeword)

        output = model(y_dd, token_prior=prior, return_details=True)
        eq_out = output.equalizer_output

        self.assertIsNotNone(eq_out.token_candidate_sources)
        for src in eq_out.token_candidate_sources:
            self.assertEqual(src, "sketch")

    def test_sketch_indices_propagate(self):
        config = _config(
            detector_prox_mode="token_posterior",
            token_candidate_pruning_mode="sketch",
            token_candidate_sketch_mode="strided",
            token_candidate_score_mode="distance",
            token_candidate_count=4,
            token_candidate_sketch_size=8,
            use_offgrid_refinement=False,
        )
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)
        codeword = _complex_dd(config.vocab_size, config)
        prior = TokenCodewordPrior(codeword_book=codeword)

        output = model(y_dd, token_prior=prior, return_details=True)
        eq_out = output.equalizer_output

        self.assertIsNotNone(eq_out.token_candidate_sketch_indices)
        idx = eq_out.token_candidate_sketch_indices[0]
        # strided mode: all batch same indices
        self.assertIn(idx.ndim, (1, 2))

    def test_selection_based_on_z_not_stale_x_dd(self):
        """Block-internal sketch selection receives z, not the block input x_dd."""
        import receiver.token_prior as tp_module
        original_fn = tp_module.select_token_candidates_by_sketch
        captured_first_args = []

        def spy(*args, **kwargs):
            captured_first_args.append(args[0])  # first positional = z_dd
            return original_fn(*args, **kwargs)

        tp_module.select_token_candidates_by_sketch = spy
        try:
            config = _config(
                detector_prox_mode="token_posterior",
                token_candidate_pruning_mode="sketch",
                token_candidate_sketch_mode="strided",
                token_candidate_count=3,
                token_candidate_sketch_size=8,
                use_offgrid_refinement=False,
            )
            model = LearnableOTFSReceiver(config)
            y_dd = _complex_dd(2, config)
            codeword = _complex_dd(config.vocab_size, config)
            prior = TokenCodewordPrior(codeword_book=codeword)

            output = model(y_dd, token_prior=prior)

            self.assertGreater(len(captured_first_args), 0,
                               "select_token_candidates_by_sketch was never called")
            z_passed = captured_first_args[0]
            self.assertTrue(torch.is_complex(z_passed))
            self.assertEqual(z_passed.shape, (2, config.M, config.N))
        finally:
            tp_module.select_token_candidates_by_sketch = original_fn

    def test_trace_includes_candidate_source_and_count(self):
        config = _config(
            detector_prox_mode="token_posterior",
            token_candidate_pruning_mode="sketch",
            token_candidate_sketch_mode="hybrid",
            token_candidate_count=5,
            use_offgrid_refinement=False,
        )
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)
        codeword = _complex_dd(config.vocab_size, config)
        prior = TokenCodewordPrior(codeword_book=codeword)

        output = model(y_dd, token_prior=prior, return_details=True)

        self.assertEqual(output.receiver_trace["token_candidate_source_last"], "sketch")
        self.assertEqual(output.receiver_trace["token_candidate_effective_count"], 5)

    def test_adaptive_candidate_diagnostics_propagate_to_equalizer_and_trace(self):
        config = _config(
            detector_prox_mode="token_posterior",
            token_candidate_pruning_mode="sketch",
            token_candidate_count=2,
            token_candidate_max_count=5,
            token_candidate_adaptive=True,
            token_candidate_entropy_threshold=0.0,
            token_candidate_margin_threshold=0.9,
            token_candidate_expand_factor=3.0,
            use_offgrid_refinement=False,
        )
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)
        codeword = torch.zeros(config.vocab_size, config.M, config.N, dtype=torch.complex64)
        prior = TokenCodewordPrior(codeword_book=codeword)

        output = model(y_dd, token_prior=prior, return_details=True)
        eq_out = output.equalizer_output

        self.assertIsNotNone(eq_out.token_candidate_entropy)
        self.assertIsNotNone(eq_out.token_candidate_margin)
        self.assertIsNotNone(eq_out.token_candidate_fallback_used)
        self.assertIsNotNone(eq_out.token_candidate_effective_count)
        self.assertEqual(eq_out.token_posterior_indices[-1].shape, (2, 5))
        self.assertEqual(eq_out.token_candidate_effective_count[-1], 5)
        self.assertTrue(eq_out.token_candidate_fallback_used[-1].all())
        self.assertIn("token_candidate_entropy_last_mean", output.receiver_trace)
        self.assertIn("token_candidate_margin_last_mean", output.receiver_trace)
        self.assertIn("token_candidate_fallback_rate_last", output.receiver_trace)
        self.assertEqual(output.receiver_trace["token_candidate_effective_count_last"], 5)


class CandidateMetricTests(unittest.TestCase):
    def test_candidate_recall_metric(self):
        candidates = torch.tensor([[1, 4, 7], [2, 5, 8], [0, 3, 6]])
        token_ids = torch.tensor([4, 9, 0])

        recall = candidate_recall_metric(candidates, token_ids)

        self.assertAlmostEqual(float(recall.item()), 2.0 / 3.0, places=6)

    def test_candidate_margin_regularization_finite_and_graph_safe(self):
        scores = torch.tensor([[4.0, 0.0, -1.0], [0.1, 0.0, -0.1]], requires_grad=True)

        loss = candidate_margin_regularization(scores, target_margin=0.5)
        loss.backward()

        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss).item())
        self.assertIsNotNone(scores.grad)

    def test_candidate_margin_regularization_single_candidate_returns_zero(self):
        scores = torch.tensor([[1.0], [2.0]], requires_grad=True)

        loss = candidate_margin_regularization(scores)
        loss.backward()

        self.assertAlmostEqual(float(loss.item()), 0.0, places=6)
        self.assertIsNotNone(scores.grad)

    def test_paper_loss_includes_candidate_margin_component(self):
        config = _config(
            detector_prox_mode="token_posterior",
            token_candidate_pruning_mode="sketch",
            token_candidate_count=4,
            use_offgrid_refinement=False,
        )
        model = LearnableOTFSReceiver(config)
        y_dd = _complex_dd(2, config)
        codeword = _complex_dd(config.vocab_size, config)
        prior = TokenCodewordPrior(codeword_book=codeword)
        output = model(y_dd, token_prior=prior, return_details=True)
        token_ids = torch.tensor([1, 2])
        weights = ReceiverLossWeights(token_ce_weight=1.0, candidate_margin_weight=0.1)

        result = paper_receiver_loss(output, token_ids, weights=weights)

        self.assertIn("candidate_margin", result.components)
        self.assertTrue(torch.isfinite(result.total).item())


if __name__ == "__main__":
    unittest.main()
