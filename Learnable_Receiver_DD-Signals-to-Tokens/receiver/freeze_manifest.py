"""Receiver freeze manifest -- rx_v1.

Lightweight module (no heavy torch imports).  Provides canonical lists of
the frozen receiver chain, public APIs, TX/RX contract, and baseline
ablations for downstream code and tests to reference.
"""

RECEIVER_FREEZE_VERSION: str = "rx_v1"

FROZEN_RECEIVER_CHAIN: tuple[str, ...] = (
    "Y_DD [B,M,N] complex",
    "Embedded Pilot Sparse CE",
    "Off-grid Sparse Path Refinement",
    "Physics-Guided Gain Re-estimation",
    "Parametric Sparse DD Operator H_theta / H_theta^H",
    "Variance-Aware Token Posterior Unfolded Detector",
    "Adaptive Candidate-Pruned Token Posterior Prox",
    "Confidence / Symbol-Reliability-Gated Residual Denoiser",
    "Data-Consistency Correction",
    "Multi-Head Evidence Token Classifier",
    "Detector-Classifier Logit Fusion",
    "token_logits [B,V]",
)

FROZEN_PUBLIC_APIS: tuple[str, ...] = (
    "LearnableOTFSReceiver.forward",
    "ReceiverConfig",
    "ReceiverOutput",
    "SparseChannelEstimate",
    "SparseDDOperatorState",
    "EqualizerOutput",
    "DenoiserOutput",
    "DataConsistencyOutput",
    "TokenClassifierOutput",
    "TokenLogitFusionOutput",
    "ReliabilityDiagnostics",
    "PhysicsValidationReport",
    "ReceiverComplexityReport",
    "ReceiverPaperTrace",
    "ReceiverLossWeights",
    "ReceiverLossOutput",
    "TokenCodewordPrior",
    "EmbeddedPilotConfig",
    "paper_receiver_loss",
    "named_receiver_paper_ablation",
    "validate_receiver_physics",
    "summarize_receiver_config",
    "estimate_receiver_ops",
    "count_receiver_parameters",
    "build_receiver_paper_trace",
    "receiver_paper_summary",
    "select_token_candidates_by_sketch",
    "token_codeword_posterior_prox",
    "build_symbol_reliability_map",
    "build_reliability_diagnostics",
    "token_ce_loss",
    "token_cross_entropy",
    "total_receiver_loss",
)

FROZEN_TX_RX_CONTRACT: dict[str, object] = {
    "description": (
        "TX must produce X_DD [B,M,N] complex from token indices. "
        "RX consumes Y_DD [B,M,N] complex and produces token_logits [B,V]. "
        "Token codeword book is shared via TokenCodewordPrior.codeword_book "
        "with shape [V,M,N] complex."
    ),
    "shared": {
        "codeword_book_shape": "[V, M, N]",
        "codeword_book_dtype": "complex",
        "token_vocab_size": "V (ReceiverConfig.vocab_size)",
        "dd_grid_shape": "[M, N] (ReceiverConfig.M, N)",
    },
    "tx_output": "X_DD [B, M, N] complex (token-dependent DD signal)",
    "rx_input": "Y_DD [B, M, N] complex (received DD signal after channel)",
    "rx_output": "token_logits [B, V] float (receiver decision)",
    "data_mask_semantics": (
        "Real [B,M,N] or [1,M,N] with 1.0 on data-bearing DD positions. "
        "Derived from 1 - pilot_mask - guard_mask when not explicitly passed."
    ),
    "pilot_mask_semantics": "Real [B,M,N] or [1,M,N]. Exactly one position = 1.0.",
    "guard_mask_semantics": "Real [B,M,N] or [1,M,N]. Positions surrounding pilot = 1.0.",
    "forbidden": [
        "bit/QAM head",
        "Linear(..., vocab_size) classifier head",
        "Transformer / self-attention",
        "GNN",
        "VAE",
        "dense MN x MN channel matrix",
    ],
}

FROZEN_BASELINE_ABLATIONS: tuple[str, ...] = (
    "full",
    "no_offgrid",
    "no_physics_gain_refinement",
    "ongrid_operator",
    "no_token_posterior_prox",
    "no_candidate_pruning",
    "no_adaptive_candidate_pruning",
    "no_energy_weighted_candidate_score",
    "no_denoiser",
    "no_data_consistency",
    "single_head_classifier",
    "no_logit_fusion",
    "classifier_only",
    "no_reliability_calibration",
)
