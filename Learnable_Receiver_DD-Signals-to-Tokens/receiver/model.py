"""Fixed sparse OTFS receiver model.

The forward path is fixed:
Y_DD -> Sparse CE/support -> Sparse DD operator -> Unfolded PIC/Prox Equalizer
-> Residual DD Denoiser -> Token Embedding Classifier -> token_logits.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.nn as nn

from .complex_utils import validate_complex_dd, validate_optional_complex_dd, validate_support_mask
from .config import ReceiverConfig
from .data_consistency import DataConsistencyCorrection, DataConsistencyOutput
from .dd_ops import SparseDDOperator, SparseDDOperatorState
from .denoiser import DenoiserOutput, ResidualDDDenoiser
from .equalizer import EqualizerOutput, UnfoldedPICProxEqualizer
from .pilot_ce import EmbeddedPilotConfig
from .reliability import (
    ReliabilityDiagnostics,
    build_reliability_diagnostics,
    build_symbol_reliability_map,
)
from .sparse_channel import SparseChannelEstimate, SparseChannelEstimator
from .token_classifier import TokenEmbeddingClassifier, TokenClassifierOutput
from .token_fusion import (
    DetectorClassifierLogitFusion,
    TokenLogitFusionOutput,
)
from .token_prior import TokenCodewordPrior


@dataclass
class ReceiverOutput:
    """Detailed receiver output for analysis and deep supervision.

    Attributes:
        token_logits: Float token logits with shape [B, vocab_size].
        rx_embedding: Float receiver embedding with shape [B, D].
        sparse_estimate: SparseChannelEstimate with h_dd [B, M, N] and paths
            [B, K, 2].
        operator_state: SparseDDOperatorState with sparse DD paths [B, K, 2].
        x_equalized: Complex equalizer output with shape [B, M, N].
        x_refined: Complex denoiser output with shape [B, M, N].
        equalizer_output: Optional EqualizerOutput containing layer estimates
            [B, M, N].
        aux_logits: Optional list of float logits, each with shape
            [B, vocab_size].
        token_prior_logits: Optional list of float token prior logits, each
            with shape [B, V], one per unfolded detector layer.
        denoiser_output: Optional DenoiserOutput carrying x_refined [B, M, N],
            delta [B, M, N], and confidence/uncertainty maps [B, M, N].
        classifier_output: Optional TokenClassifierOutput carrying token logits
            [B, vocab_size], evidence weights [B, M, N], and optional local
            embeddings [B, D, M, N].
        token_prior_weights: Optional list of float token prior weights, each
            with shape [B, V], one per unfolded detector layer.
        residual_energy: Optional equalizer residual energy with shape
            [B, M, N].
        token_posterior_variance: Optional final unfolded detector posterior
            variance with shape [B, M, N].
        receiver_trace: Optional dict with paper-chain configuration metadata.
        topk_indices: Optional long top-k token indices with shape [B, K].
        topk_logits: Optional float top-k token logits with shape [B, K].
        x_consistent: Optional complex DD tensor after data-consistency
            correction with shape [B, M, N].
        data_consistency_output: Optional DataConsistencyOutput with correction
            details.
        reliability_diagnostics: Optional ReliabilityDiagnostics computed
            without token_ids.
        logit_fusion_output: Optional TokenLogitFusionOutput with fused logits
            and gate.
        classifier_token_logits: Optional float pre-fusion classifier logits
            [B, V].
        detector_token_logits: Optional float expanded detector logits [B, V].
    """

    token_logits: torch.Tensor
    rx_embedding: torch.Tensor
    sparse_estimate: SparseChannelEstimate
    operator_state: SparseDDOperatorState
    x_equalized: torch.Tensor
    x_refined: torch.Tensor
    equalizer_output: EqualizerOutput | None = None
    aux_logits: list[torch.Tensor] | None = None
    token_prior_logits: list[torch.Tensor] | None = None
    denoiser_output: DenoiserOutput | None = None
    classifier_output: TokenClassifierOutput | None = None
    token_prior_weights: list[torch.Tensor] | None = None
    residual_energy: torch.Tensor | None = None
    token_posterior_variance: torch.Tensor | None = None
    receiver_trace: dict | None = None
    topk_indices: torch.Tensor | None = None
    topk_logits: torch.Tensor | None = None
    x_consistent: torch.Tensor | None = None
    data_consistency_output: DataConsistencyOutput | None = None
    reliability_diagnostics: ReliabilityDiagnostics | None = None
    logit_fusion_output: TokenLogitFusionOutput | None = None
    classifier_token_logits: torch.Tensor | None = None
    detector_token_logits: torch.Tensor | None = None


class LearnableOTFSReceiver(nn.Module):
    """Learnable sparse DD receiver.

    Inputs:
        y_dd: Complex received DD grid with shape [B, M, N].
        h_dd: Optional complex DD channel with shape [B, M, N].
        support_mask: Optional bool/float support mask with shape [B, M, N].
        snr_db: Optional scalar or [B] tensor for sparse CE/denoiser features.
        channel_estimator_mode: Optional CE mode: "oracle", "pilot", or
            "fallback_topk".
        pilot_config: Optional EmbeddedPilotConfig for pilot CE on y_dd
            [B, M, N].
        pilot_layout: Optional alias for pilot_config.
        pilot_threshold: Optional pilot CE threshold.
        token_prior: Optional TokenCodewordPrior with codeword_book [V, M, N]
            used by the unfolded detector.
        return_paper_trace: Optional bool controlling receiver_trace output.
        return_denoiser_details: If true, expose DenoiserOutput.
        return_classifier_details: If true, expose TokenClassifierOutput.
        return_local_classifier: If true, classifier_output includes local
            embeddings [B, D, M, N].
        data_mask: Optional data-region mask with shape [B, M, N] or
            [1, M, N] for token posterior prox.
        pilot_mask: Optional pilot mask with shape [B, M, N] or [1, M, N].
        guard_mask: Optional guard mask with shape [B, M, N] or [1, M, N].
        ce_data_mask: Optional CE fitting mask with shape [B, M, N] or
            [1, M, N] for physics-guided gain refinement.

    Output:
        By default, token_logits [B, vocab_size]. With return_details=True or
        return_aux=True, ReceiverOutput with intermediate DD tensors [B, M, N].
    """

    def __init__(self, config: ReceiverConfig):
        super().__init__()
        self.config = config
        self.sparse_ce = SparseChannelEstimator(config)
        self.dd_operator = SparseDDOperator(config)
        self.equalizer = UnfoldedPICProxEqualizer(config, self.dd_operator)
        denoiser_config = config
        if getattr(config, "use_denoiser", None) is not None:
            denoiser_config = replace(config, use_refinement_net=bool(config.use_denoiser))
        self.denoiser = ResidualDDDenoiser(denoiser_config)
        self.data_consistency_correction = (
            DataConsistencyCorrection(config, self.dd_operator)
            if config.use_data_consistency_correction
            else None
        )
        self.token_classifier = TokenEmbeddingClassifier(config)
        self.logit_fusion = (
            DetectorClassifierLogitFusion(
                fusion_mode=config.token_logit_fusion_mode,
                init_gate=config.token_logit_fusion_init_gate,
                scale=config.token_logit_fusion_scale,
                center_detector=config.token_logit_fusion_center_detector,
            )
            if config.use_token_logit_fusion
            else None
        )

    def forward(
        self,
        y_dd: torch.Tensor,
        h_dd: torch.Tensor | None = None,
        support_mask: torch.Tensor | None = None,
        snr_db: torch.Tensor | float | None = None,
        return_details: bool = False,
        return_aux: bool = False,
        topk: int | None = None,
        channel_estimator_mode: str | None = None,
        pilot_config: EmbeddedPilotConfig | None = None,
        pilot_layout: EmbeddedPilotConfig | None = None,
        pilot_threshold: float | None = None,
        token_prior: TokenCodewordPrior | None = None,
        return_paper_trace: bool | None = None,
        return_denoiser_details: bool = False,
        return_classifier_details: bool = False,
        return_local_classifier: bool = False,
        return_reliability_diagnostics: bool = False,
        data_mask: torch.Tensor | None = None,
        pilot_mask: torch.Tensor | None = None,
        guard_mask: torch.Tensor | None = None,
        ce_data_mask: torch.Tensor | None = None,
        token_candidate_indices: torch.Tensor | None = None,
    ) -> torch.Tensor | ReceiverOutput:
        """Run fixed receiver chain from y_dd [B, M, N].

        Args:
            y_dd: Complex received DD grid with shape [B, M, N].
            h_dd: Optional complex DD channel with shape [B, M, N].
            support_mask: Optional bool/float support mask with shape [B, M, N].
            snr_db: Optional scalar or [B] tensor.
            return_details: If false and return_aux is false, return
                token_logits [B, vocab_size]. If true, return ReceiverOutput.
            return_aux: If true, equalizer layer estimates [B, M, N] are
                classified into aux logits [B, vocab_size] without denoising.
            topk: Optional K for final classifier top-k metadata [B, K] when
                details are returned.
            channel_estimator_mode: Optional CE mode: "oracle", "pilot", or
                "fallback_topk". Defaults to config.channel_estimator_mode.
            pilot_config: Optional EmbeddedPilotConfig for pilot CE.
            pilot_layout: Optional alias for pilot_config.
            pilot_threshold: Optional pilot CE magnitude threshold.
            token_prior: Optional TokenCodewordPrior with codeword_book
                [V, M, N] for equalizer-side proximal projection.
            return_paper_trace: Optional bool. If true, return ReceiverOutput
                with receiver_trace metadata.
            return_denoiser_details: If true, expose denoiser_output.
            return_classifier_details: If true, expose classifier_output.
            return_local_classifier: If true, classifier_output includes local
                embeddings [B, D, M, N].
            data_mask: Optional real/bool data-region mask with shape
                [B, M, N] or [1, M, N].
            pilot_mask: Optional real/bool pilot mask with shape [B, M, N] or
                [1, M, N].
            guard_mask: Optional real/bool guard mask with shape [B, M, N] or
                [1, M, N].
            ce_data_mask: Optional real/bool CE fitting mask with shape
                [B, M, N] or [1, M, N]. If omitted, data_mask is used when
                physics_refinement_use_data_mask is enabled.

        Returns:
            token_logits [B, vocab_size] or ReceiverOutput.
        """

        validate_complex_dd(y_dd, self.config, "y_dd")
        validate_optional_complex_dd(h_dd, self.config, "h_dd")
        if h_dd is not None and h_dd.shape[0] != y_dd.shape[0]:
            raise ValueError("h_dd batch size must match y_dd.")
        if support_mask is not None:
            validate_support_mask(support_mask, self.config, "support_mask")
            if support_mask.shape[0] != y_dd.shape[0]:
                raise ValueError("support_mask batch size must match y_dd.")
        resolved_data_mask = _resolve_data_mask(y_dd, data_mask, pilot_mask, guard_mask)
        resolved_ce_data_mask = (
            _optional_batch_map(ce_data_mask, y_dd, "ce_data_mask").clamp(0.0, 1.0)
            if ce_data_mask is not None
            else None
        )
        has_data_mask = resolved_data_mask is not None
        has_pilot_guard_mask = pilot_mask is not None or guard_mask is not None
        resolved_return_trace = (
            self.config.return_paper_trace_by_default
            if return_paper_trace is None
            else bool(return_paper_trace)
        )
        need_receiver_output = (
            return_details
            or return_aux
            or resolved_return_trace
            or return_denoiser_details
            or return_classifier_details
            or return_local_classifier
            or return_reliability_diagnostics
        )
        need_equalizer_output = (
            return_details
            or return_aux
            or self.config.use_equalizer_residual_for_denoiser
            or self.config.use_token_logit_fusion
            or (token_prior is not None and (return_details or resolved_return_trace))
        )

        sparse_estimate = self.sparse_ce(
            y_dd,
            h_dd=h_dd,
            support_mask=support_mask,
            snr_db=snr_db,
            mode=channel_estimator_mode,
            pilot_config=pilot_config,
            pilot_layout=pilot_layout,
            pilot_threshold=pilot_threshold,
            ce_fit_mask=resolved_ce_data_mask,
        )
        operator_state = self.dd_operator(sparse_estimate)
        token_data_mask = _data_mask_or_ones(y_dd, resolved_data_mask)
        detector_reliability_map = build_symbol_reliability_map(
            data_mask=token_data_mask,
            path_confidence=sparse_estimate.confidence,
            path_confidence_map=sparse_estimate.path_confidence_map,
        )
        detector_data_mask = (
            token_data_mask
            if self.config.use_data_region_mask_for_token_prox
            else None
        )
        eq_candidate_indices = (
            token_candidate_indices
            if self.config.token_candidate_pruning_mode == "explicit"
            else None
        )
        equalizer_result = self.equalizer(
            y_dd,
            operator_state,
            return_all=need_equalizer_output,
            token_prior=token_prior,
            data_mask=detector_data_mask,
            confidence_map=detector_reliability_map,
            uncertainty_map=None,
            token_candidate_indices=eq_candidate_indices,
        )
        if need_equalizer_output:
            equalizer_output = _as_equalizer_output(equalizer_result)
            x_equalized = equalizer_output.x_hat
        else:
            equalizer_output = None
            x_equalized = equalizer_result
        residual_energy = None
        if (
            self.config.use_equalizer_residual_for_denoiser
            and equalizer_output is not None
            and equalizer_output.residuals
        ):
            residual_energy = equalizer_output.residuals[-1].abs().pow(2)
        posterior_variance = None
        if (
            equalizer_output is not None
            and equalizer_output.token_posterior_variance is not None
            and equalizer_output.token_posterior_variance
        ):
            posterior_variance = equalizer_output.token_posterior_variance[-1]

        denoiser_result = self.denoiser(
            x_equalized,
            y_dd=y_dd,
            h_dd=sparse_estimate.h_dd,
            support_mask=sparse_estimate.support_mask,
            snr_db=snr_db,
            confidence_map=sparse_estimate.path_confidence_map,
            path_confidence=sparse_estimate.confidence,
            path_indices=sparse_estimate.path_indices,
            residual_energy=residual_energy,
            pilot_residual_power=sparse_estimate.pilot_residual_power,
            return_delta=True,
        )
        denoiser_output = _as_denoiser_output(denoiser_result)
        x_refined = denoiser_output.x_refined
        classifier_reliability_map = build_symbol_reliability_map(
            data_mask=token_data_mask,
            path_confidence=sparse_estimate.confidence,
            path_confidence_map=sparse_estimate.path_confidence_map,
            residual_energy=residual_energy,
            posterior_variance=posterior_variance,
            uncertainty_map=denoiser_output.uncertainty_map,
        )

        # ---- Step 12: Data-Consistency Correction ----
        dc_output = None
        x_for_classifier = x_refined
        if self.data_consistency_correction is not None:
            dc_needs_details = need_receiver_output or return_denoiser_details
            dc_result = self.data_consistency_correction(
                x_refined,
                y_dd,
                operator_state,
                data_mask=token_data_mask,
                reliability_map=classifier_reliability_map,
                uncertainty_map=denoiser_output.uncertainty_map,
                return_details=dc_needs_details,
            )
            if dc_needs_details:
                dc_output = dc_result
                x_for_classifier = dc_output.x_corrected
            else:
                x_for_classifier = dc_result

        classifier_result = self.token_classifier(
            x_for_classifier,
            return_embedding=need_receiver_output,
            topk=topk if need_receiver_output else None,
            support_mask=token_data_mask,
            confidence_map=classifier_reliability_map,
            uncertainty_map=denoiser_output.uncertainty_map,
            return_local=return_local_classifier,
            return_head_details=return_details or return_classifier_details,
        )

        # ---- Step 16: detector-classifier logit fusion ----
        fusion_output = None
        final_logits = _token_logits(classifier_result)
        if self.logit_fusion is not None:
            det_logits = None
            det_indices = None
            if (equalizer_output is not None
                    and equalizer_output.token_posterior_logits is not None
                    and equalizer_output.token_posterior_logits):
                det_logits = equalizer_output.token_posterior_logits[-1]
                if (equalizer_output.token_posterior_indices is not None
                        and equalizer_output.token_posterior_indices):
                    det_indices = equalizer_output.token_posterior_indices[-1]

            classifier_logits = _token_logits(classifier_result)
            fusion_diag = build_reliability_diagnostics(
                reliability_map=classifier_reliability_map,
                token_logits=classifier_logits,
                posterior_logits=det_logits,
                residual_energy=residual_energy,
                uncertainty_map=denoiser_output.uncertainty_map,
                data_mask=token_data_mask,
            )
            fusion_output = self.logit_fusion(
                classifier_logits,
                detector_logits=det_logits,
                detector_candidate_indices=det_indices,
                classifier_confidence=fusion_diag.classifier_confidence,
                classifier_entropy=fusion_diag.classifier_entropy,
                posterior_confidence=fusion_diag.posterior_confidence,
                posterior_entropy=fusion_diag.posterior_entropy,
                reliability_scalar=fusion_diag.reliability_scalar,
            )
            final_logits = fusion_output.fused_logits

        if not need_receiver_output:
            return final_logits

        classifier_output = _as_classifier_output(classifier_result)
        aux_logits = None
        if return_aux and equalizer_output is not None:
            aux_logits = [
                self.token_classifier(
                    layer,
                    support_mask=token_data_mask,
                    confidence_map=classifier_reliability_map,
                    uncertainty_map=denoiser_output.uncertainty_map,
                    return_head_details=False,
                )
                for layer in equalizer_output.layer_estimates
            ]
        token_prior_weights = None
        if (
            self.config.expose_token_prior_weights
            and equalizer_output is not None
            and equalizer_output.token_prior_weights is not None
        ):
            token_prior_weights = equalizer_output.token_prior_weights
        exposed_equalizer_output = equalizer_output if (return_details or return_aux) else None
        exposed_denoiser_output = denoiser_output if (return_details or return_denoiser_details) else None
        exposed_classifier_output = (
            classifier_output
            if (return_details or return_classifier_details or return_local_classifier)
            else None
        )
        if exposed_classifier_output is not None and not self.config.expose_classifier_evidence:
            exposed_classifier_output = _without_classifier_evidence(exposed_classifier_output)
        receiver_trace = (
            self._receiver_trace(
                sparse_estimate=sparse_estimate,
                operator_state=operator_state,
                token_prior=token_prior,
                pilot_config=pilot_config,
                pilot_layout=pilot_layout,
                has_data_mask=has_data_mask,
                has_pilot_guard_mask=has_pilot_guard_mask,
                has_token_posterior_variance=posterior_variance is not None,
                dc_output=dc_output,
            )
            if (return_details or resolved_return_trace)
            else None
        )
        if receiver_trace is not None and equalizer_output is not None:
            if equalizer_output.token_candidate_sources:
                receiver_trace["token_candidate_source_last"] = equalizer_output.token_candidate_sources[-1]
            if equalizer_output.token_posterior_indices:
                receiver_trace["token_candidate_effective_count"] = equalizer_output.token_posterior_indices[-1].shape[-1]
                receiver_trace["token_candidate_effective_count_last"] = equalizer_output.token_posterior_indices[-1].shape[-1]
            if equalizer_output.token_candidate_effective_count:
                last_count = equalizer_output.token_candidate_effective_count[-1]
                receiver_trace["token_candidate_effective_count"] = _trace_scalar(last_count)
                receiver_trace["token_candidate_effective_count_last"] = _trace_scalar(last_count)
            if equalizer_output.token_candidate_entropy:
                receiver_trace["token_candidate_entropy_last_mean"] = float(
                    equalizer_output.token_candidate_entropy[-1].detach().mean().cpu().item()
                )
            if equalizer_output.token_candidate_margin:
                receiver_trace["token_candidate_margin_last_mean"] = float(
                    equalizer_output.token_candidate_margin[-1].detach().mean().cpu().item()
                )
            if equalizer_output.token_candidate_fallback_used:
                receiver_trace["token_candidate_fallback_rate_last"] = float(
                    equalizer_output.token_candidate_fallback_used[-1].to(dtype=torch.float32).mean().cpu().item()
                )

        # reliability diagnostics (token-free)
        reliability_diag = None
        if return_reliability_diagnostics or return_details:
            post_logits = None
            if equalizer_output is not None and equalizer_output.token_posterior_logits is not None and equalizer_output.token_posterior_logits:
                post_logits = equalizer_output.token_posterior_logits[-1]
            reliability_diag = build_reliability_diagnostics(
                reliability_map=classifier_reliability_map,
                token_logits=final_logits,
                posterior_logits=post_logits,
                residual_energy=residual_energy,
                uncertainty_map=denoiser_output.uncertainty_map,
                data_mask=token_data_mask,
            )
            if receiver_trace is not None:
                receiver_trace["has_reliability_diagnostics"] = reliability_diag is not None
                if reliability_diag.classifier_confidence is not None:
                    receiver_trace["classifier_confidence_mean"] = float(
                        reliability_diag.classifier_confidence.mean().cpu().item()
                    )
                if reliability_diag.classifier_entropy is not None:
                    receiver_trace["classifier_entropy_mean"] = float(
                        reliability_diag.classifier_entropy.mean().cpu().item()
                    )
                if reliability_diag.posterior_entropy is not None:
                    receiver_trace["posterior_entropy_mean"] = float(
                        reliability_diag.posterior_entropy.mean().cpu().item()
                    )
                if reliability_diag.reliability_scalar is not None:
                    receiver_trace["reliability_scalar_mean"] = float(
                        reliability_diag.reliability_scalar.mean().cpu().item()
                    )
                if reliability_diag.expected_error_proxy is not None:
                    receiver_trace["expected_error_proxy_mean"] = float(
                        reliability_diag.expected_error_proxy.mean().cpu().item()
                    )

        # classifier head trace
        if receiver_trace is not None and classifier_output.head_evidence_entropy is not None:
            receiver_trace["classifier_head_entropy_mean"] = float(
                classifier_output.head_evidence_entropy.mean().cpu().item()
            )
        if receiver_trace is not None and classifier_output.head_fusion_weights is not None:
            fw = classifier_output.head_fusion_weights
            fw_entropy = -(fw * torch.log(fw.clamp_min(1e-8))).sum(dim=-1)
            receiver_trace["classifier_head_fusion_entropy_mean"] = float(fw_entropy.mean().cpu().item())

        # fusion trace
        if receiver_trace is not None:
            receiver_trace["use_token_logit_fusion"] = self.config.use_token_logit_fusion
            receiver_trace["token_logit_fusion_mode"] = self.config.token_logit_fusion_mode
            if fusion_output is not None:
                receiver_trace["token_logit_fusion_has_detector_logits"] = fusion_output.detector_logits is not None
                if fusion_output.fusion_gate is not None:
                    receiver_trace["token_logit_fusion_gate_mean"] = float(
                        fusion_output.fusion_gate.mean().cpu().item()
                    )

        return ReceiverOutput(
            token_logits=final_logits,
            rx_embedding=classifier_output.rx_embedding,
            sparse_estimate=sparse_estimate,
            operator_state=operator_state,
            x_equalized=x_equalized,
            x_refined=x_refined,
            equalizer_output=exposed_equalizer_output,
            aux_logits=aux_logits,
            token_prior_logits=equalizer_output.token_prior_logits if equalizer_output is not None else None,
            denoiser_output=exposed_denoiser_output,
            classifier_output=exposed_classifier_output,
            token_prior_weights=token_prior_weights,
            residual_energy=residual_energy,
            token_posterior_variance=posterior_variance,
            receiver_trace=receiver_trace,
            topk_indices=classifier_output.topk_indices,
            topk_logits=classifier_output.topk_logits,
            x_consistent=x_for_classifier if dc_output is not None else None,
            data_consistency_output=dc_output,
            reliability_diagnostics=reliability_diag,
            logit_fusion_output=fusion_output,
            classifier_token_logits=(
                fusion_output.classifier_logits if fusion_output is not None else None
            ),
            detector_token_logits=(
                fusion_output.detector_logits if fusion_output is not None else None
            ),
        )

    def _receiver_trace(
        self,
        sparse_estimate: SparseChannelEstimate,
        operator_state: SparseDDOperatorState,
        token_prior: TokenCodewordPrior | None,
        pilot_config: EmbeddedPilotConfig | None,
        pilot_layout: EmbeddedPilotConfig | None,
        has_data_mask: bool,
        has_pilot_guard_mask: bool,
        has_token_posterior_variance: bool,
        dc_output: DataConsistencyOutput | None = None,
    ) -> dict:
        """Build paper receiver trace for tensors [B, M, N] and paths [B, K, 2]."""

        layout = pilot_config if pilot_config is not None else pilot_layout
        return {
            "channel_estimator_mode": sparse_estimate.estimator_mode,
            "use_offgrid_refinement": self.config.use_offgrid_refinement,
            "operator_mode": operator_state.operator_mode,
            "use_token_prior": token_prior is not None,
            "use_denoiser": self.denoiser.config.use_refinement_net,
            "use_equalizer_residual_for_denoiser": self.config.use_equalizer_residual_for_denoiser,
            "classifier_is_position_aware": hasattr(self.token_classifier, "local_embedding_head")
            and hasattr(self.token_classifier, "evidence_head"),
            "classifier_num_evidence_heads": self.config.classifier_num_evidence_heads,
            "classifier_head_fusion": self.config.classifier_head_fusion,
            "classifier_uses_multihead_evidence": self.config.classifier_num_evidence_heads > 1,
            "pilot_delay": layout.pilot_delay if layout is not None else self.config.pilot_delay,
            "pilot_doppler": layout.pilot_doppler if layout is not None else self.config.pilot_doppler,
            "pilot_guard_delay": layout.guard_delay if layout is not None else self.config.pilot_guard_delay,
            "pilot_guard_doppler": layout.guard_doppler if layout is not None else self.config.pilot_guard_doppler,
            "pilot_obs_delay_radius": (
                layout.obs_delay_radius if layout is not None else self.config.pilot_obs_delay_radius
            ),
            "pilot_obs_doppler_radius": (
                layout.obs_doppler_radius if layout is not None else self.config.pilot_obs_doppler_radius
            ),
            "dd_operator_mode": self.config.dd_operator_mode,
            "offgrid_kernel_type": self.config.offgrid_kernel_type,
            "offgrid_kernel_radius": self.config.offgrid_kernel_radius,
            "detector_prox_mode": self.config.detector_prox_mode,
            "token_posterior_topk": self.config.token_posterior_topk,
            "use_data_region_mask_for_token_prox": self.config.use_data_region_mask_for_token_prox,
            "has_data_mask": bool(has_data_mask),
            "has_pilot_guard_mask": bool(has_pilot_guard_mask),
            "uses_symbol_reliability_map": True,
            "classifier_uses_data_mask_not_channel_support": True,
            "use_variance_tracking": self.config.use_variance_tracking,
            "posterior_temperature_from_variance": self.config.posterior_temperature_from_variance,
            "confidence_weighted_variance": self.config.confidence_weighted_variance,
            "uses_token_posterior_variance_for_reliability": bool(has_token_posterior_variance),
            "use_physics_guided_gain_refinement": self.config.use_physics_guided_gain_refinement,
            "physics_refined": bool(sparse_estimate.physics_refined),
            "physics_residual_power_mean": (
                float(sparse_estimate.physics_residual_power.mean().detach().cpu().item())
                if sparse_estimate.physics_residual_power is not None
                else None
            ),
            "physics_fit_residual_power_mean": (
                float(sparse_estimate.physics_fit_residual_power.mean().detach().cpu().item())
                if sparse_estimate.physics_fit_residual_power is not None
                else None
            ),
            "physics_full_residual_power_mean": (
                float(sparse_estimate.physics_full_residual_power.mean().detach().cpu().item())
                if sparse_estimate.physics_full_residual_power is not None
                else None
            ),
            "physics_fit_mask_source": sparse_estimate.ce_fit_mask_source,
            "physics_probe_position": sparse_estimate.physics_probe_position,
            "physics_uses_known_pilot_probe": (
                sparse_estimate.physics_probe_position is not None
                and sparse_estimate.ce_fit_mask_source in ("pilot_observation", "explicit_ce_data_mask")
            ),
            "use_data_consistency_correction": self.config.use_data_consistency_correction,
            "data_consistency_corrected": dc_output is not None,
            "dc_step_size": (
                float(dc_output.step_size.detach().cpu().item())
                if dc_output is not None and dc_output.step_size is not None
                else None
            ),
            "dc_residual_energy_mean": (
                float(dc_output.residual_energy.mean().detach().cpu().item())
                if dc_output is not None and dc_output.residual_energy is not None
                else None
            ),
            "dc_observation_weight_mean": (
                float(dc_output.observation_weight.mean().detach().cpu().item())
                if dc_output is not None and dc_output.observation_weight is not None
                else None
            ),
            "dc_update_gate_mean": (
                float(dc_output.update_gate.mean().detach().cpu().item())
                if dc_output is not None and dc_output.update_gate is not None
                else None
            ),
            "token_candidate_pruning_mode": self.config.token_candidate_pruning_mode,
            "token_candidate_sketch_mode": self.config.token_candidate_sketch_mode,
            "token_candidate_score_mode": self.config.token_candidate_score_mode,
            "token_candidate_count": self.config.token_candidate_count,
            "token_candidate_sketch_size": self.config.token_candidate_sketch_size,
            "uses_token_candidate_pruning": self.config.token_candidate_pruning_mode != "none",
            "token_candidate_adaptive": self.config.token_candidate_adaptive,
        }


def _token_logits(value: torch.Tensor | TokenClassifierOutput) -> torch.Tensor:
    if isinstance(value, TokenClassifierOutput):
        return value.token_logits
    return value


def _trace_scalar(value: int | float | torch.Tensor) -> int | float:
    if torch.is_tensor(value):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return float(value.detach().to(dtype=torch.float32).mean().cpu().item())
    return value


def _as_classifier_output(value: torch.Tensor | TokenClassifierOutput) -> TokenClassifierOutput:
    if not isinstance(value, TokenClassifierOutput):
        raise TypeError("TokenClassifierOutput is required when return_details=True.")
    return value


def _as_equalizer_output(value: torch.Tensor | EqualizerOutput) -> EqualizerOutput:
    if not isinstance(value, EqualizerOutput):
        raise TypeError("EqualizerOutput is required when return_aux=True.")
    return value


def _as_denoiser_output(value: torch.Tensor | DenoiserOutput) -> DenoiserOutput:
    if not isinstance(value, DenoiserOutput):
        raise TypeError("DenoiserOutput is required when return_delta=True.")
    return value


def _without_classifier_evidence(value: TokenClassifierOutput) -> TokenClassifierOutput:
    return TokenClassifierOutput(
        token_logits=value.token_logits,
        rx_embedding=value.rx_embedding,
        topk_indices=value.topk_indices,
        topk_logits=value.topk_logits,
        evidence_weights=None,
        local_embeddings=value.local_embeddings,
        confidence_map=None,
        uncertainty_map=None,
    )


def _resolve_data_mask(
    y_dd: torch.Tensor,
    data_mask: torch.Tensor | None,
    pilot_mask: torch.Tensor | None,
    guard_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    if data_mask is not None:
        return _optional_batch_map(data_mask, y_dd, "data_mask").clamp(0.0, 1.0)
    if pilot_mask is None and guard_mask is None:
        return None
    pilot = _optional_batch_map(pilot_mask, y_dd, "pilot_mask") if pilot_mask is not None else torch.zeros_like(y_dd.real)
    guard = _optional_batch_map(guard_mask, y_dd, "guard_mask") if guard_mask is not None else torch.zeros_like(y_dd.real)
    return (1.0 - pilot.clamp(0.0, 1.0) - guard.clamp(0.0, 1.0)).clamp(0.0, 1.0)


def _optional_batch_map(value: torch.Tensor, y_dd: torch.Tensor, name: str) -> torch.Tensor:
    if not torch.is_tensor(value) or value.ndim != 3 or torch.is_complex(value):
        raise ValueError(f"{name} must have real shape [B, M, N] or [1, M, N].")
    if value.shape[-2:] != y_dd.shape[-2:] or value.shape[0] not in {1, y_dd.shape[0]}:
        raise ValueError(f"{name} must have real shape [B, M, N] or [1, M, N].")
    resolved = value.to(device=y_dd.device, dtype=y_dd.real.dtype)
    if resolved.shape[0] == 1 and y_dd.shape[0] != 1:
        resolved = resolved.expand(y_dd.shape[0], -1, -1)
    return resolved


def _data_mask_or_ones(y_dd: torch.Tensor, data_mask: torch.Tensor | None) -> torch.Tensor:
    if data_mask is None:
        return torch.ones_like(y_dd.real)
    return data_mask.to(device=y_dd.device, dtype=y_dd.real.dtype)
