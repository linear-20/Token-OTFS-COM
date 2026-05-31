"""Configuration for the fixed sparse OTFS receiver.

All modules use DD tensors with shape [B, M, N].
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReceiverConfig:
    """Receiver hyperparameters.

    Attributes:
        M: Delay/subcarrier bins in y_dd with shape [B, M, N].
        N: Doppler/timeslot bins in y_dd with shape [B, M, N].
        vocab_size: Number of token classes in token_logits [B, vocab_size].
        token_embedding_dim: Hidden token embedding width [B, token_embedding_dim].
        num_unfolded_layers: Number of unfolded PIC/prox layers.
        topk_paths: Maximum sparse DD support entries per batch item.
        hidden_channels: CNN channel width for DD denoising/classification.
        noise_var: Scalar noise variance used by the placeholder equalizer.
        use_refinement_net: Enables the residual DD denoiser when true.
        channel_estimator_mode: Sparse CE mode for y_dd [B, M, N]. Supported
            values are "oracle", "pilot", and "fallback_topk".
        pilot_delay: Optional embedded pilot delay index in y_dd [B, M, N].
        pilot_doppler: Optional embedded pilot Doppler index in y_dd [B, M, N].
        pilot_guard_delay: Non-negative guard radius along delay.
        pilot_guard_doppler: Non-negative guard radius along Doppler.
        pilot_obs_delay_radius: Non-negative pilot observation radius along delay.
        pilot_obs_doppler_radius: Non-negative pilot observation radius along
            Doppler.
        pilot_threshold: Optional non-negative magnitude threshold for pilot CE.
        pilot_topk_paths: Optional maximum pilot CE paths [B, K].
        pilot_value_real: Real part of the embedded pilot scalar.
        pilot_value_imag: Imaginary part of the embedded pilot scalar.
        confidence_temperature: Positive scalar shaping normalized confidence
            tensors [B, K] and [B, M, N].
        use_channel_refinement: Enables SparseRefinementNet for channel
            estimates h_dd [B, M, N].
        use_denoiser: Optional denoiser enable flag. If None, legacy
            use_refinement_net controls the denoiser.
        pilot_cfar_scale: Positive scale for sqrt(noise_var) CFAR-style pilot
            CE thresholding.
        pilot_min_confidence: Minimum normalized pilot confidence in [0, 1]
            for valid paths [B, K].
        require_obs_within_guard: If true, pilot observation windows must fit
            inside the pilot+guard rectangle.
        use_offgrid_refinement: Enables local off-grid path refinement after
            on-grid sparse CE. It produces fractional_offsets [B, K, 2].
        offgrid_patch_delay_radius: Non-negative local patch radius along delay.
        offgrid_patch_doppler_radius: Non-negative local patch radius along
            Doppler.
        offgrid_max_offset: Maximum absolute fractional path offset in delay
            and Doppler, constrained to (0, 0.5].
        offgrid_confidence_floor: Minimum input confidence in [0, 1] required
            for an off-grid path update.
        offgrid_gain_correction_scale: Non-negative scale for complex gain
            correction [B, K].
        dd_operator_mode: Sparse DD operator mode: "auto", "ongrid", or
            "offgrid".
        offgrid_kernel_radius: Non-negative leakage kernel radius for off-grid
            DD paths.
        offgrid_kernel_type: Approximate leakage kernel type: "linear" or
            "sinc".
        offgrid_normalize_kernel: If true, normalize leakage kernel weights to
            avoid gain blow-up.
        classifier_position_channels: Positive DD position encoding channels
            used by token classifier inputs [B, C, M, N].
        classifier_evidence_temperature: Positive evidence softmax temperature.
        classifier_confidence_floor: Positive floor used in log confidence
            evidence prior.
        classifier_use_uncertainty: If true, uncertainty suppresses classifier
            evidence prior.
        classifier_num_evidence_heads: Number of evidence heads H.  H=1 is the
            original single-head classifier; H>1 enables multi-head evidence
            pooling with per-head to-codebook similarity and head fusion.
        classifier_head_fusion: Head fusion mode.  "mean" averages heads
            equally; "learned_static" learns scalar head weights; "confidence"
            weights heads by inverse evidence entropy.
        classifier_head_diversity_eps: Small positive value for numerical
            safety in evidence softmax and head diversity loss.
        classifier_return_head_details: If true, TokenClassifierOutput
            includes per-head fields on return_embedding.
        use_equalizer_residual_for_denoiser: If true, the last unfolded
            equalizer residual energy [B, M, N] is passed to the residual
            denoiser.
        return_paper_trace_by_default: If true, receiver forward returns
            details with an interpretability trace unless overridden.
        expose_token_prior_weights: If true, ReceiverOutput may expose token
            prior weights [B, V] per unfolded detector layer.
        expose_classifier_evidence: If true, ReceiverOutput may expose
            classifier evidence weights [B, M, N] through classifier_output.
        detector_prox_mode: Detector proximal mode: "l1", "token_posterior",
            or "hybrid".
        token_posterior_topk: Optional positive top-k candidate count for
            token posterior prox weights [B, K].
        token_posterior_temperature: Positive posterior temperature for
            token-codeword prox.
        use_data_region_mask_for_token_prox: If true, data region masks are
            used in token posterior distance [B, M, N].
        use_variance_tracking: If true, unfolded detector layers track
            residual/estimate variance [B, 1, 1].
        variance_floor: Positive lower bound for variance tensors [B, 1, 1].
        variance_damping_init: Initial detector damping scalar in (0, 1].
        posterior_temperature_from_variance: If true, token posterior
            temperature is scaled by estimate variance [B, 1, 1].
        confidence_weighted_variance: If true, path confidence [B, K] adjusts
            residual variance estimation.
        use_physics_guided_gain_refinement: If true, apply approximate
            physics-guided sparse gain LS/MMSE re-estimation after CE.
        physics_refinement_ridge: Positive ridge/MMSE prior weight for sparse
            path gain re-estimation.
        physics_refinement_min_confidence: Minimum confidence in [0, 1] for a
            path to participate in physics-guided gain re-estimation.
        physics_refinement_use_data_mask: If true, CE data masks may restrict
            the fitting region [B, M, N].
        physics_refinement_require_fit_mask: If true (default), physics-guided
            gain refinement requires an explicit CE fitting mask. When no mask
            is available the refiner is skipped unless
            physics_refinement_allow_full_grid_debug is enabled.
        physics_refinement_allow_full_grid_debug: If true, physics refinement
            may fall back to a full-grid fitting mask when no CE observation
            mask is supplied. This is a debug-only switch; the paper receiver
            must not fit channel gains against unknown token data regions.
        use_data_consistency_correction: If true, a data-consistency gradient
            correction block runs between denoiser and classifier.
        dc_step_init: Initial step size for the correction block. 0.0 means
            identity at initialisation.
        dc_step_scale: Positive scalar multiplier on top of the learnable step.
        dc_use_reliability: If true, reliability map gates the correction.
        dc_use_uncertainty: If true, uncertainty map suppresses the correction.
        dc_gate_floor: Minimum gate value in [0, 1] so the block is not fully
            disabled after training.
        dc_gate_uses_reliability: If true, the scalar update gate is multiplied
            by the reliability map.  This is a heuristic x-domain gating knob
            separate from the observation-domain LS gradient weighting.  The
            gate is never multiplied by data_mask or uncertainty.
    """

    M: int
    N: int
    vocab_size: int
    token_embedding_dim: int
    num_unfolded_layers: int
    topk_paths: int
    hidden_channels: int
    noise_var: float
    use_refinement_net: bool
    channel_estimator_mode: str = "fallback_topk"
    pilot_delay: int | None = None
    pilot_doppler: int | None = None
    pilot_guard_delay: int = 0
    pilot_guard_doppler: int = 0
    pilot_obs_delay_radius: int = 0
    pilot_obs_doppler_radius: int = 0
    pilot_threshold: float | None = None
    pilot_topk_paths: int | None = None
    pilot_value_real: float = 1.0
    pilot_value_imag: float = 0.0
    confidence_temperature: float = 1.0
    use_channel_refinement: bool = False
    use_denoiser: bool | None = None
    pilot_cfar_scale: float = 3.0
    pilot_min_confidence: float = 0.0
    require_obs_within_guard: bool = True
    use_offgrid_refinement: bool = False
    offgrid_patch_delay_radius: int = 1
    offgrid_patch_doppler_radius: int = 1
    offgrid_max_offset: float = 0.5
    offgrid_confidence_floor: float = 0.0
    offgrid_gain_correction_scale: float = 0.1
    dd_operator_mode: str = "auto"
    offgrid_kernel_radius: int = 1
    offgrid_kernel_type: str = "linear"
    offgrid_normalize_kernel: bool = True
    classifier_position_channels: int = 6
    classifier_evidence_temperature: float = 1.0
    classifier_confidence_floor: float = 1e-3
    classifier_use_uncertainty: bool = True
    classifier_num_evidence_heads: int = 1
    classifier_head_fusion: str = "mean"
    classifier_head_diversity_eps: float = 1e-8
    classifier_return_head_details: bool = False
    use_token_logit_fusion: bool = False
    token_logit_fusion_mode: str = "reliability"
    token_logit_fusion_init_gate: float = 0.0
    token_logit_fusion_scale: float = 1.0
    token_logit_fusion_center_detector: bool = True
    use_equalizer_residual_for_denoiser: bool = True
    return_paper_trace_by_default: bool = False
    expose_token_prior_weights: bool = True
    expose_classifier_evidence: bool = True
    detector_prox_mode: str = "hybrid"
    token_posterior_topk: int | None = None
    token_posterior_temperature: float = 1.0
    use_data_region_mask_for_token_prox: bool = True
    use_variance_tracking: bool = True
    variance_floor: float = 1e-8
    variance_damping_init: float = 0.5
    posterior_temperature_from_variance: bool = True
    confidence_weighted_variance: bool = True
    use_physics_guided_gain_refinement: bool = False
    physics_refinement_ridge: float = 1e-3
    physics_refinement_min_confidence: float = 0.0
    physics_refinement_use_data_mask: bool = False
    physics_refinement_require_fit_mask: bool = True
    physics_refinement_allow_full_grid_debug: bool = False
    use_data_consistency_correction: bool = False
    dc_step_init: float = 0.0
    dc_step_scale: float = 1.0
    dc_use_reliability: bool = True
    dc_use_uncertainty: bool = True
    dc_gate_floor: float = 0.0
    dc_gate_uses_reliability: bool = False
    token_candidate_pruning_mode: str = "none"
    token_candidate_sketch_mode: str = "hybrid"
    token_candidate_score_mode: str = "distance"
    token_candidate_sketch_size: int | None = None
    token_candidate_count: int | None = None
    token_candidate_adaptive: bool = False
    token_candidate_min_count: int | None = None
    token_candidate_max_count: int | None = None
    token_candidate_entropy_threshold: float = 0.85
    token_candidate_margin_threshold: float = 0.05
    token_candidate_expand_factor: float = 2.0
    token_candidate_allow_full_fallback: bool = False
    token_candidate_reliability_floor: float = 1e-4

    def __post_init__(self) -> None:
        """Validate scalar config values for DD tensors [B, M, N]."""

        positive_int_fields = {
            "M": self.M,
            "N": self.N,
            "vocab_size": self.vocab_size,
            "token_embedding_dim": self.token_embedding_dim,
            "num_unfolded_layers": self.num_unfolded_layers,
            "topk_paths": self.topk_paths,
            "hidden_channels": self.hidden_channels,
        }
        for name, value in positive_int_fields.items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if not isinstance(self.noise_var, (int, float)) or self.noise_var < 0:
            raise ValueError("noise_var must be a non-negative scalar.")
        if not isinstance(self.use_refinement_net, bool):
            raise ValueError("use_refinement_net must be a bool.")
        if self.channel_estimator_mode not in {"oracle", "pilot", "fallback_topk"}:
            raise ValueError('channel_estimator_mode must be one of "oracle", "pilot", or "fallback_topk".')

        optional_indices = {
            "pilot_delay": (self.pilot_delay, self.M),
            "pilot_doppler": (self.pilot_doppler, self.N),
        }
        for name, (value, limit) in optional_indices.items():
            if value is None:
                continue
            if not isinstance(value, int) or value < 0 or value >= limit:
                raise ValueError(f"{name} must be None or an integer in [0, {limit}).")

        non_negative_int_fields = {
            "pilot_guard_delay": self.pilot_guard_delay,
            "pilot_guard_doppler": self.pilot_guard_doppler,
            "pilot_obs_delay_radius": self.pilot_obs_delay_radius,
            "pilot_obs_doppler_radius": self.pilot_obs_doppler_radius,
        }
        for name, value in non_negative_int_fields.items():
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")

        if self.pilot_threshold is not None:
            if not isinstance(self.pilot_threshold, (int, float)) or self.pilot_threshold < 0:
                raise ValueError("pilot_threshold must be None or a non-negative scalar.")
        if self.pilot_topk_paths is not None:
            if not isinstance(self.pilot_topk_paths, int) or self.pilot_topk_paths <= 0:
                raise ValueError("pilot_topk_paths must be None or a positive integer.")
        if not isinstance(self.pilot_value_real, (int, float)):
            raise ValueError("pilot_value_real must be a scalar.")
        if not isinstance(self.pilot_value_imag, (int, float)):
            raise ValueError("pilot_value_imag must be a scalar.")
        if not isinstance(self.confidence_temperature, (int, float)) or self.confidence_temperature <= 0:
            raise ValueError("confidence_temperature must be positive.")
        if not isinstance(self.use_channel_refinement, bool):
            raise ValueError("use_channel_refinement must be a bool.")
        if self.use_denoiser is not None and not isinstance(self.use_denoiser, bool):
            raise ValueError("use_denoiser must be None or a bool.")
        if not isinstance(self.pilot_cfar_scale, (int, float)) or self.pilot_cfar_scale <= 0:
            raise ValueError("pilot_cfar_scale must be positive.")
        if not isinstance(self.pilot_min_confidence, (int, float)) or not 0.0 <= self.pilot_min_confidence <= 1.0:
            raise ValueError("pilot_min_confidence must be in [0, 1].")
        if not isinstance(self.require_obs_within_guard, bool):
            raise ValueError("require_obs_within_guard must be a bool.")
        if not isinstance(self.use_offgrid_refinement, bool):
            raise ValueError("use_offgrid_refinement must be a bool.")
        offgrid_radii = {
            "offgrid_patch_delay_radius": self.offgrid_patch_delay_radius,
            "offgrid_patch_doppler_radius": self.offgrid_patch_doppler_radius,
        }
        for name, value in offgrid_radii.items():
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        if not isinstance(self.offgrid_max_offset, (int, float)) or not 0.0 < self.offgrid_max_offset <= 0.5:
            raise ValueError("offgrid_max_offset must be in (0, 0.5].")
        if (
            not isinstance(self.offgrid_confidence_floor, (int, float))
            or not 0.0 <= self.offgrid_confidence_floor <= 1.0
        ):
            raise ValueError("offgrid_confidence_floor must be in [0, 1].")
        if (
            not isinstance(self.offgrid_gain_correction_scale, (int, float))
            or self.offgrid_gain_correction_scale < 0
        ):
            raise ValueError("offgrid_gain_correction_scale must be non-negative.")
        if self.dd_operator_mode not in {"auto", "ongrid", "offgrid"}:
            raise ValueError('dd_operator_mode must be one of "auto", "ongrid", or "offgrid".')
        if not isinstance(self.offgrid_kernel_radius, int) or self.offgrid_kernel_radius < 0:
            raise ValueError("offgrid_kernel_radius must be a non-negative integer.")
        if self.offgrid_kernel_type not in {"linear", "sinc"}:
            raise ValueError('offgrid_kernel_type must be "linear" or "sinc".')
        if not isinstance(self.offgrid_normalize_kernel, bool):
            raise ValueError("offgrid_normalize_kernel must be a bool.")
        if not isinstance(self.classifier_position_channels, int) or self.classifier_position_channels <= 0:
            raise ValueError("classifier_position_channels must be a positive integer.")
        if (
            not isinstance(self.classifier_evidence_temperature, (int, float))
            or self.classifier_evidence_temperature <= 0
        ):
            raise ValueError("classifier_evidence_temperature must be positive.")
        if (
            not isinstance(self.classifier_confidence_floor, (int, float))
            or self.classifier_confidence_floor <= 0
        ):
            raise ValueError("classifier_confidence_floor must be positive.")
        if not isinstance(self.classifier_use_uncertainty, bool):
            raise ValueError("classifier_use_uncertainty must be a bool.")
        if not isinstance(self.classifier_num_evidence_heads, int) or self.classifier_num_evidence_heads < 1:
            raise ValueError("classifier_num_evidence_heads must be >= 1.")
        if self.classifier_head_fusion not in {"mean", "learned_static", "confidence"}:
            raise ValueError('classifier_head_fusion must be "mean", "learned_static", or "confidence".')
        if not isinstance(self.classifier_head_diversity_eps, (int, float)) or self.classifier_head_diversity_eps <= 0:
            raise ValueError("classifier_head_diversity_eps must be positive.")
        if not isinstance(self.classifier_return_head_details, bool):
            raise ValueError("classifier_return_head_details must be a bool.")
        if not isinstance(self.use_token_logit_fusion, bool):
            raise ValueError("use_token_logit_fusion must be a bool.")
        if self.token_logit_fusion_mode not in {"disabled", "static", "reliability"}:
            raise ValueError('token_logit_fusion_mode must be "disabled", "static", or "reliability".')
        if not isinstance(self.token_logit_fusion_init_gate, (int, float)):
            raise ValueError("token_logit_fusion_init_gate must be a float.")
        if not isinstance(self.token_logit_fusion_scale, (int, float)) or self.token_logit_fusion_scale < 0:
            raise ValueError("token_logit_fusion_scale must be non-negative.")
        if not isinstance(self.token_logit_fusion_center_detector, bool):
            raise ValueError("token_logit_fusion_center_detector must be a bool.")
        bool_fields = {
            "use_equalizer_residual_for_denoiser": self.use_equalizer_residual_for_denoiser,
            "return_paper_trace_by_default": self.return_paper_trace_by_default,
            "expose_token_prior_weights": self.expose_token_prior_weights,
            "expose_classifier_evidence": self.expose_classifier_evidence,
        }
        for name, value in bool_fields.items():
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a bool.")
        if self.detector_prox_mode not in {"l1", "token_posterior", "hybrid"}:
            raise ValueError('detector_prox_mode must be one of "l1", "token_posterior", or "hybrid".')
        if self.token_posterior_topk is not None:
            if not isinstance(self.token_posterior_topk, int) or self.token_posterior_topk <= 0:
                raise ValueError("token_posterior_topk must be None or a positive integer.")
        if (
            not isinstance(self.token_posterior_temperature, (int, float))
            or self.token_posterior_temperature <= 0
        ):
            raise ValueError("token_posterior_temperature must be positive.")
        if not isinstance(self.use_data_region_mask_for_token_prox, bool):
            raise ValueError("use_data_region_mask_for_token_prox must be a bool.")
        variance_bool_fields = {
            "use_variance_tracking": self.use_variance_tracking,
            "posterior_temperature_from_variance": self.posterior_temperature_from_variance,
            "confidence_weighted_variance": self.confidence_weighted_variance,
        }
        for name, value in variance_bool_fields.items():
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a bool.")
        if not isinstance(self.variance_floor, (int, float)) or self.variance_floor <= 0:
            raise ValueError("variance_floor must be positive.")
        if (
            not isinstance(self.variance_damping_init, (int, float))
            or not 0.0 < self.variance_damping_init <= 1.0
        ):
            raise ValueError("variance_damping_init must be in (0, 1].")
        physics_bool_fields = {
            "use_physics_guided_gain_refinement": self.use_physics_guided_gain_refinement,
            "physics_refinement_use_data_mask": self.physics_refinement_use_data_mask,
            "physics_refinement_require_fit_mask": self.physics_refinement_require_fit_mask,
            "physics_refinement_allow_full_grid_debug": self.physics_refinement_allow_full_grid_debug,
        }
        for name, value in physics_bool_fields.items():
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a bool.")
        if not isinstance(self.physics_refinement_ridge, (int, float)) or self.physics_refinement_ridge <= 0:
            raise ValueError("physics_refinement_ridge must be positive.")
        if (
            not isinstance(self.physics_refinement_min_confidence, (int, float))
            or not 0.0 <= self.physics_refinement_min_confidence <= 1.0
        ):
            raise ValueError("physics_refinement_min_confidence must be in [0, 1].")
        dc_bool_fields = {
            "use_data_consistency_correction": self.use_data_consistency_correction,
            "dc_use_reliability": self.dc_use_reliability,
            "dc_use_uncertainty": self.dc_use_uncertainty,
            "dc_gate_uses_reliability": self.dc_gate_uses_reliability,
        }
        for name, value in dc_bool_fields.items():
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a bool.")
        if not isinstance(self.dc_step_init, (int, float)):
            raise ValueError("dc_step_init must be a float.")
        if not isinstance(self.dc_step_scale, (int, float)) or self.dc_step_scale < 0:
            raise ValueError("dc_step_scale must be non-negative.")
        if not isinstance(self.dc_gate_floor, (int, float)) or not 0.0 <= self.dc_gate_floor <= 1.0:
            raise ValueError("dc_gate_floor must be in [0, 1].")
        if self.token_candidate_pruning_mode not in {"none", "sketch", "explicit"}:
            raise ValueError('token_candidate_pruning_mode must be "none", "sketch", or "explicit".')
        if self.token_candidate_sketch_mode not in {"strided", "energy_topk", "hybrid"}:
            raise ValueError('token_candidate_sketch_mode must be "strided", "energy_topk", or "hybrid".')
        if self.token_candidate_score_mode not in {"distance", "corr", "energy_weighted_distance"}:
            raise ValueError('token_candidate_score_mode must be "distance", "corr", or "energy_weighted_distance".')
        if self.token_candidate_sketch_size is not None:
            if not isinstance(self.token_candidate_sketch_size, int) or self.token_candidate_sketch_size <= 0:
                raise ValueError("token_candidate_sketch_size must be a positive integer or None.")
        if self.token_candidate_count is not None:
            if not isinstance(self.token_candidate_count, int) or self.token_candidate_count <= 0:
                raise ValueError("token_candidate_count must be a positive integer or None.")
        if not isinstance(self.token_candidate_adaptive, bool):
            raise ValueError("token_candidate_adaptive must be a bool.")
        if self.token_candidate_min_count is not None:
            if not isinstance(self.token_candidate_min_count, int) or self.token_candidate_min_count <= 0:
                raise ValueError("token_candidate_min_count must be a positive integer or None.")
        if self.token_candidate_max_count is not None:
            if not isinstance(self.token_candidate_max_count, int) or self.token_candidate_max_count <= 0:
                raise ValueError("token_candidate_max_count must be a positive integer or None.")
        if self.token_candidate_min_count is not None and self.token_candidate_max_count is not None:
            if self.token_candidate_max_count < self.token_candidate_min_count:
                raise ValueError("token_candidate_max_count must be >= token_candidate_min_count.")
        if not 0.0 <= self.token_candidate_entropy_threshold <= 1.0:
            raise ValueError("token_candidate_entropy_threshold must be in [0, 1].")
        if self.token_candidate_margin_threshold < 0:
            raise ValueError("token_candidate_margin_threshold must be >= 0.")
        if self.token_candidate_expand_factor < 1.0:
            raise ValueError("token_candidate_expand_factor must be >= 1.")
        if not isinstance(self.token_candidate_allow_full_fallback, bool):
            raise ValueError("token_candidate_allow_full_fallback must be a bool.")
        if self.token_candidate_reliability_floor <= 0:
            raise ValueError("token_candidate_reliability_floor must be > 0.")
