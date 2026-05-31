"""Unfolded sparse PIC/prox equalizer for DD tensors.

This is a UAMP/BPIC-inspired sparse PIC/proximal-gradient unfolding stage, not
a complete reproduction of UAMP or BPIC. It uses the sparse DD operator only
through apply() and matched_filter(); no dense MN x MN channel matrix is built.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .complex_utils import validate_complex_dd
from .config import ReceiverConfig
from .dd_ops import SparseDDOperator, SparseDDOperatorState
from .token_prior import TokenCodewordPrior, TokenPosteriorProxOutput, token_codeword_posterior_prox


@dataclass
class EqualizerOutput:
    """Detailed unfolded equalizer output.

    Attributes:
        x_hat: Complex equalized DD estimate with shape [B, M, N].
        layer_estimates: List of complex DD estimates, each with shape
            [B, M, N]. The list contains x0 followed by one estimate per
            unfolded layer, so its length is num_unfolded_layers + 1.
        residuals: Optional list of complex residuals, each with shape
            [B, M, N], one per unfolded layer.
        token_prior_logits: Optional list of float token prior logits, each
            with shape [B, V], one per unfolded layer.
        token_prior_weights: Optional list of float token prior weights, each
            with shape [B, V], one per unfolded layer.
        token_posterior_logits: Optional list of float posterior logits, each
            with shape [B, K] or [B, V].
        token_posterior_weights: Optional list of float posterior weights, each
            with shape [B, K] or [B, V].
        token_posterior_indices: Optional list of long posterior candidate
            indices, each with shape [B, K].
        token_candidate_scores: Optional list of float candidate scores, each
            with shape [B, K].
        token_candidate_sources: Optional list of str candidate source labels.
        token_candidate_sketch_indices: Optional list of long sketch indices.
        token_candidate_entropy: Optional list of float normalized selector
            entropy tensors, each with shape [B].
        token_candidate_margin: Optional list of float selector top1-top2
            margin tensors, each with shape [B].
        token_candidate_fallback_used: Optional list of bool adaptive fallback
            tensors, each with shape [B].
        token_candidate_effective_count: Optional list of candidate counts
            after adaptive expansion.
        token_posterior_variance: Optional list of float posterior variance
            maps, each with shape [B, M, N].
        residual_variances: Optional list of float residual variances, each
            with shape [B, 1, 1].
        estimate_variances: Optional list of float estimate variances, each
            with shape [B, 1, 1].
        posterior_temperatures: Optional list of float posterior temperatures,
            each with shape [B, 1, 1].
        detector_damping: Optional list of float damping values, each with
            shape [B, 1, 1].
    """

    x_hat: torch.Tensor
    layer_estimates: list[torch.Tensor]
    residuals: list[torch.Tensor] | None = None
    token_prior_logits: list[torch.Tensor] | None = None
    token_prior_weights: list[torch.Tensor] | None = None
    token_posterior_logits: list[torch.Tensor] | None = None
    token_posterior_weights: list[torch.Tensor] | None = None
    token_posterior_indices: list[torch.Tensor] | None = None
    token_candidate_scores: list[torch.Tensor] | None = None
    token_candidate_sources: list[str] | None = None
    token_candidate_sketch_indices: list[torch.Tensor] | None = None
    token_candidate_entropy: list[torch.Tensor] | None = None
    token_candidate_margin: list[torch.Tensor] | None = None
    token_candidate_fallback_used: list[torch.Tensor] | None = None
    token_candidate_effective_count: list[int | torch.Tensor] | None = None
    token_posterior_variance: list[torch.Tensor] | None = None
    residual_variances: list[torch.Tensor] | None = None
    estimate_variances: list[torch.Tensor] | None = None
    posterior_temperatures: list[torch.Tensor] | None = None
    detector_damping: list[torch.Tensor] | None = None


@dataclass
class _LayerVarianceState:
    residual_variance: torch.Tensor
    estimate_variance: torch.Tensor
    posterior_temperature: torch.Tensor
    detector_damping: torch.Tensor


def complex_soft_threshold(z: torch.Tensor, threshold: torch.Tensor | float, eps: float = 1e-8) -> torch.Tensor:
    """Apply complex soft-thresholding to z [B, M, N], returning [B, M, N]."""

    if not torch.is_tensor(z) or not torch.is_complex(z) or z.ndim != 3:
        raise TypeError("z must be a complex tensor with shape [B, M, N].")
    threshold_tensor = torch.as_tensor(threshold, device=z.device, dtype=z.real.dtype)
    magnitude = z.abs()
    shrink = torch.relu(magnitude - threshold_tensor) / (magnitude + eps)
    return z * shrink


class UnfoldedPICProxBlock(nn.Module):
    """One sparse PIC/proximal-gradient layer for DD tensors [B, M, N].

    The block computes residual [B, M, N], matched-filter gradient [B, M, N],
    a damped update, and complex soft-threshold prox output [B, M, N].
    If token_prior is supplied and detector_prox_mode is token_posterior or
    hybrid, the block applies a token-codeword posterior prox in DD space.
    """

    def __init__(
        self,
        dd_operator: SparseDDOperator,
        alpha_init: float = 0.5,
        threshold_init: float = 1e-3,
        variance_damping_init: float = 0.5,
    ):
        super().__init__()
        if not 0.0 < alpha_init < 1.0:
            raise ValueError("alpha_init must be in (0, 1).")
        if threshold_init <= 0:
            raise ValueError("threshold_init must be positive.")
        if not 0.0 < variance_damping_init <= 1.0:
            raise ValueError("variance_damping_init must be in (0, 1].")
        self.dd_operator = dd_operator
        self.alpha_raw = nn.Parameter(torch.tensor(_logit(alpha_init), dtype=torch.float32))
        self.threshold_raw = nn.Parameter(torch.tensor(_inverse_softplus(threshold_init), dtype=torch.float32))
        self.beta_raw = nn.Parameter(torch.tensor(_logit(0.1), dtype=torch.float32))
        self.variance_damping_raw = nn.Parameter(torch.tensor(_bounded_logit(variance_damping_init), dtype=torch.float32))

    def forward(
        self,
        y_dd: torch.Tensor,
        x_dd: torch.Tensor,
        state: SparseDDOperatorState,
        denom: torch.Tensor,
        token_prior: TokenCodewordPrior | None = None,
        data_mask: torch.Tensor | None = None,
        confidence_map: torch.Tensor | None = None,
        uncertainty_map: torch.Tensor | None = None,
        token_posterior_topk: int | None = None,
        token_candidate_indices: torch.Tensor | None = None,
        pruning_mode: str = "none",
        num_candidates: int | None = None,
        sketch_size: int | None = None,
        sketch_mode: str = "hybrid",
        score_mode: str = "distance",
        adaptive: bool = False,
        min_candidates: int | None = None,
        max_candidates: int | None = None,
        entropy_threshold: float = 0.85,
        margin_threshold: float = 0.05,
        expand_factor: float = 2.0,
        allow_full_fallback: bool = False,
        reliability_floor: float = 1e-4,
        detector_prox_mode: str = "hybrid",
        posterior_temperature: float | torch.Tensor | None = None,
        use_variance_tracking: bool = True,
        variance_floor: float = 1e-8,
        posterior_temperature_from_variance: bool = True,
        confidence_weighted_variance: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, TokenPosteriorProxOutput | None, _LayerVarianceState]:
        """Run one block.

        Candidate selection (sketch) is done inside the block on ``z``,
        not on the stale input ``x_dd``.

        Args:
            y_dd: Complex received DD grid with shape [B, M, N].
            x_dd: Complex current DD estimate with shape [B, M, N].
            state: SparseDDOperatorState carrying paths [B, K, 2] and gains
                [B, K].
            denom: Real normalization tensor with shape [B, 1, 1].
            token_prior: Optional TokenCodewordPrior with codeword_book
                [V, M, N].
            data_mask: Optional data region mask with shape [B, M, N] or
                [1, M, N].
            confidence_map: Optional confidence map with shape [B, M, N].
            uncertainty_map: Optional uncertainty map with shape [B, M, N].
            token_posterior_topk: Optional posterior candidate K for
                full-vocab top-K (used only when candidate pruning is off).
            token_candidate_indices: Optional long [B, K] for explicit mode.
            pruning_mode: "none", "sketch", or "explicit".
            num_candidates: K for sketch pruning.
            sketch_size: S for sketch positions.
            sketch_mode: "strided", "energy_topk", or "hybrid".
            score_mode: "distance" or "corr".
            detector_prox_mode: "l1", "token_posterior", or "hybrid".
            posterior_temperature: Optional positive posterior temperature.
            use_variance_tracking: If true, variance adapts posterior
                temperature and damping metadata [B, 1, 1].
            variance_floor: Positive lower bound for variance tensors
                [B, 1, 1].
            posterior_temperature_from_variance: If true, temperature is
                scaled by estimate variance [B, 1, 1].
            confidence_weighted_variance: If true, path confidence [B, K]
                adjusts residual variance.

        Returns:
            x_next: Complex next DD estimate with shape [B, M, N].
            residual: Complex residual y_dd - H_S x_dd with shape [B, M, N].
            posterior: Optional TokenPosteriorProxOutput with projected_dd
            [B, M, N] and posterior weights [B, K] or [B, V].
            variance_state: Residual/estimate variance, posterior temperature,
            and damping tensors, each with shape [B, 1, 1].
        """

        residual = y_dd - self.dd_operator.apply(x_dd, state)
        gradient = self.dd_operator.matched_filter(residual, state) / denom
        alpha = torch.sigmoid(self.alpha_raw).to(device=y_dd.device, dtype=y_dd.real.dtype)
        z = x_dd + alpha * gradient
        if detector_prox_mode not in {"l1", "token_posterior", "hybrid"}:
            raise ValueError('detector_prox_mode must be "l1", "token_posterior", or "hybrid".')
        variance_state = self._variance_state(
            residual,
            state,
            denom,
            posterior_temperature,
            use_variance_tracking,
            variance_floor,
            posterior_temperature_from_variance,
            confidence_weighted_variance,
        )
        threshold = F.softplus(self.threshold_raw).to(device=y_dd.device, dtype=y_dd.real.dtype)
        if token_prior is None or detector_prox_mode == "l1" or float(token_prior.prior_strength) <= 0:
            return complex_soft_threshold(z, threshold), residual, None, variance_state

        # ---- candidate selection on z (not stale x_dd) ----
        prox_candidate_indices = token_candidate_indices
        prox_candidate_scores = None
        prox_candidate_source = "full_vocab"
        prox_sketch_indices = None
        prox_candidate_entropy = None
        prox_candidate_margin = None
        prox_fallback_used = None
        prox_eff_count = 0

        if pruning_mode == "explicit":
            if prox_candidate_indices is None:
                raise ValueError(
                    "token_candidate_indices is required when pruning_mode='explicit'."
                )
            prox_candidate_source = "explicit"
        elif pruning_mode == "sketch" and num_candidates is not None:
            from .token_prior import select_token_candidates_by_sketch
            sel = select_token_candidates_by_sketch(
                z,
                token_prior,
                num_candidates=num_candidates,
                data_mask=data_mask,
                confidence_map=confidence_map,
                uncertainty_map=uncertainty_map,
                sketch_size=sketch_size,
                sketch_mode=sketch_mode,
                score_mode=score_mode,
                adaptive=adaptive,
                min_candidates=min_candidates,
                max_candidates=max_candidates,
                entropy_threshold=entropy_threshold,
                margin_threshold=margin_threshold,
                expand_factor=expand_factor,
                allow_full_fallback=allow_full_fallback,
                reliability_floor=reliability_floor,
            )
            prox_candidate_indices = sel.candidate_indices
            prox_candidate_scores = sel.candidate_scores
            prox_candidate_source = "sketch"
            prox_sketch_indices = sel.sketch_indices
            prox_candidate_entropy = sel.candidate_entropy
            prox_candidate_margin = sel.candidate_margin
            prox_fallback_used = sel.fallback_used
            prox_eff_count = sel.effective_candidate_count

        posterior = token_codeword_posterior_prox(
            z,
            token_prior,
            data_mask=data_mask,
            confidence_map=confidence_map,
            uncertainty_map=uncertainty_map,
            topk_candidates=token_posterior_topk if prox_candidate_indices is None else None,
            candidate_indices=prox_candidate_indices,
            candidate_scores=prox_candidate_scores,
            candidate_source=prox_candidate_source,
            sketch_indices=prox_sketch_indices,
            candidate_entropy=prox_candidate_entropy,
            candidate_margin=prox_candidate_margin,
            fallback_used=prox_fallback_used,
            effective_candidate_count=prox_eff_count,
            temperature=variance_state.posterior_temperature,
        )
        beta = torch.sigmoid(self.beta_raw).to(device=y_dd.device, dtype=y_dd.real.dtype)
        beta = (beta * float(token_prior.prior_strength)).clamp(0.0, 1.0)
        if use_variance_tracking:
            beta = beta * variance_state.detector_damping
        x_next = (1.0 - beta) * z + beta.to(dtype=z.dtype) * posterior.projected_dd
        return x_next, residual, posterior, variance_state

    def _variance_state(
        self,
        residual: torch.Tensor,
        state: SparseDDOperatorState,
        denom: torch.Tensor,
        posterior_temperature: float | torch.Tensor | None,
        use_variance_tracking: bool,
        variance_floor: float,
        posterior_temperature_from_variance: bool,
        confidence_weighted_variance: bool,
    ) -> _LayerVarianceState:
        residual_power = residual.abs().pow(2).mean(dim=(-2, -1), keepdim=True)
        denom_real = denom.to(device=residual.device, dtype=residual.real.dtype).clamp_min(float(variance_floor))
        residual_variance = residual_power
        if use_variance_tracking and confidence_weighted_variance and state.confidence is not None:
            confidence_scalar = _active_confidence_scalar(
                state.confidence,
                residual.shape[0],
                residual.device,
                residual.real.dtype,
                variance_floor,
            )
            residual_variance = residual_variance / confidence_scalar.clamp_min(float(variance_floor))
        residual_variance = torch.nan_to_num(residual_variance, nan=0.0, posinf=1e6, neginf=0.0).clamp_min(float(variance_floor))
        estimate_variance = (residual_variance / denom_real).clamp_min(float(variance_floor))
        base_temperature = _temperature_tensor(
            posterior_temperature,
            default=1.0,
            batch_size=residual.shape[0],
            device=residual.device,
            dtype=residual.real.dtype,
        )
        if use_variance_tracking and posterior_temperature_from_variance:
            posterior_tau = base_temperature * estimate_variance.detach().clamp_min(float(variance_floor))
        else:
            posterior_tau = base_temperature.expand_as(estimate_variance)
        damping = torch.sigmoid(self.variance_damping_raw).to(device=residual.device, dtype=residual.real.dtype)
        damping = damping.reshape(1, 1, 1).expand_as(estimate_variance).clamp(0.0, 1.0)
        return _LayerVarianceState(
            residual_variance=residual_variance,
            estimate_variance=estimate_variance,
            posterior_temperature=posterior_tau.clamp_min(float(variance_floor)),
            detector_damping=damping,
        )


class UnfoldedPICProxEqualizer(nn.Module):
    """Unfolded sparse PIC/prox equalizer for y_dd [B, M, N].

    This module initializes x0 = H_S^H y / (sum_i |h_i|^2 + noise_var + eps)
    and applies num_unfolded_layers sparse PIC/prox blocks. It is not a full
    UAMP/BPIC implementation.
    A TokenCodewordPrior can optionally be supplied as a proximal prior after
    each sparse PIC/prox layer; without it, legacy behavior is unchanged.
    """

    def __init__(self, config: ReceiverConfig, dd_operator: SparseDDOperator):
        super().__init__()
        self.config = config
        self.dd_operator = dd_operator
        self.eps = 1e-8
        self.blocks = nn.ModuleList(
            UnfoldedPICProxBlock(
                dd_operator,
                variance_damping_init=config.variance_damping_init,
            )
            for _ in range(config.num_unfolded_layers)
        )

    def forward(
        self,
        y_dd: torch.Tensor,
        state: SparseDDOperatorState,
        return_all: bool = False,
        token_prior: TokenCodewordPrior | None = None,
        data_mask: torch.Tensor | None = None,
        confidence_map: torch.Tensor | None = None,
        uncertainty_map: torch.Tensor | None = None,
        token_posterior_topk: int | None = None,
        token_candidate_indices: torch.Tensor | None = None,
    ) -> torch.Tensor | EqualizerOutput:
        """Equalize y_dd [B, M, N].

        Args:
            y_dd: Complex received DD grid with shape [B, M, N].
            state: SparseDDOperatorState with path_indices [B, K, 2] and
                path_gains [B, K].
            return_all: If false, return x_hat [B, M, N]. If true, return
                EqualizerOutput containing x0 and per-layer estimates.
            token_prior: Optional TokenCodewordPrior with codeword_book
                [V, M, N]. If supplied, each unfolded layer applies a soft
                token-codeword posterior prox after the H^H residual update.
            data_mask: Optional data region mask with shape [B, M, N] or
                [1, M, N] for token posterior distance.
            confidence_map: Optional confidence map with shape [B, M, N].
            uncertainty_map: Optional uncertainty map with shape [B, M, N].
            token_posterior_topk: Optional posterior candidate K. Defaults to
                config.token_posterior_topk.  Ignored when pruning is active.
            token_candidate_indices: Optional long [B, K] pre-selected
                candidates for candidate-only prox.  Takes priority over
                sketch selection and top-K.

        Returns:
            Complex x_hat [B, M, N] or EqualizerOutput.
        """

        validate_complex_dd(y_dd, self.config, "y_dd")
        denom = self._denominator(state)
        x_dd = self.dd_operator.matched_filter(y_dd, state) / denom

        # resolve candidate pruning config (passed to each block)
        pruning_mode = self.config.token_candidate_pruning_mode
        num_candidates = self.config.token_candidate_count
        if num_candidates is None:
            num_candidates = self.config.token_posterior_topk
        sketch_size = self.config.token_candidate_sketch_size
        sketch_mode = self.config.token_candidate_sketch_mode
        score_mode = self.config.token_candidate_score_mode

        layer_estimates = [x_dd]
        residuals: list[torch.Tensor] = []
        token_prior_logits: list[torch.Tensor] = []
        token_prior_weights: list[torch.Tensor] = []
        token_posterior_logits: list[torch.Tensor] = []
        token_posterior_weights: list[torch.Tensor] = []
        token_posterior_indices: list[torch.Tensor] = []
        token_posterior_variance: list[torch.Tensor] = []
        token_candidate_scores: list[torch.Tensor] = []
        token_candidate_sources: list[str] = []
        token_candidate_sketch_indices: list[torch.Tensor] = []
        token_candidate_entropy: list[torch.Tensor] = []
        token_candidate_margin: list[torch.Tensor] = []
        token_candidate_fallback_used: list[torch.Tensor] = []
        token_candidate_effective_count: list[int | torch.Tensor] = []
        residual_variances: list[torch.Tensor] = []
        estimate_variances: list[torch.Tensor] = []
        posterior_temperatures: list[torch.Tensor] = []
        detector_damping: list[torch.Tensor] = []
        posterior_topk = self.config.token_posterior_topk if token_posterior_topk is None else token_posterior_topk
        for block in self.blocks:
            x_dd, residual, posterior, variance_state = block(
                y_dd,
                x_dd,
                state,
                denom,
                token_prior=token_prior,
                data_mask=data_mask,
                confidence_map=confidence_map,
                uncertainty_map=uncertainty_map,
                token_posterior_topk=posterior_topk,
                token_candidate_indices=(
                    token_candidate_indices if pruning_mode == "explicit" else None
                ),
                pruning_mode=pruning_mode,
                num_candidates=num_candidates,
                sketch_size=sketch_size,
                sketch_mode=sketch_mode,
                score_mode=score_mode,
                adaptive=self.config.token_candidate_adaptive,
                min_candidates=self.config.token_candidate_min_count,
                max_candidates=self.config.token_candidate_max_count,
                entropy_threshold=self.config.token_candidate_entropy_threshold,
                margin_threshold=self.config.token_candidate_margin_threshold,
                expand_factor=self.config.token_candidate_expand_factor,
                allow_full_fallback=self.config.token_candidate_allow_full_fallback,
                reliability_floor=self.config.token_candidate_reliability_floor,
                detector_prox_mode=self.config.detector_prox_mode,
                posterior_temperature=self.config.token_posterior_temperature,
                use_variance_tracking=self.config.use_variance_tracking,
                variance_floor=self.config.variance_floor,
                posterior_temperature_from_variance=self.config.posterior_temperature_from_variance,
                confidence_weighted_variance=self.config.confidence_weighted_variance,
            )
            layer_estimates.append(x_dd)
            residuals.append(residual)
            residual_variances.append(variance_state.residual_variance)
            estimate_variances.append(variance_state.estimate_variance)
            posterior_temperatures.append(variance_state.posterior_temperature)
            detector_damping.append(variance_state.detector_damping)
            if posterior is not None:
                token_prior_logits.append(posterior.posterior_logits)
                token_prior_weights.append(posterior.posterior_weights)
                token_posterior_logits.append(posterior.posterior_logits)
                token_posterior_weights.append(posterior.posterior_weights)
                if posterior.candidate_indices is not None:
                    token_posterior_indices.append(posterior.candidate_indices)
                if posterior.candidate_scores is not None:
                    token_candidate_scores.append(posterior.candidate_scores)
                token_candidate_sources.append(posterior.candidate_source)
                if posterior.sketch_indices is not None:
                    token_candidate_sketch_indices.append(posterior.sketch_indices)
                if posterior.candidate_entropy is not None:
                    token_candidate_entropy.append(posterior.candidate_entropy)
                if posterior.candidate_margin is not None:
                    token_candidate_margin.append(posterior.candidate_margin)
                if posterior.fallback_used is not None:
                    token_candidate_fallback_used.append(posterior.fallback_used)
                token_candidate_effective_count.append(posterior.effective_candidate_count)
                if posterior.posterior_variance is not None:
                    token_posterior_variance.append(posterior.posterior_variance)

        if not return_all:
            return x_dd
        has_posterior = len(token_posterior_logits) > 0
        return EqualizerOutput(
            x_hat=x_dd,
            layer_estimates=layer_estimates,
            residuals=residuals,
            token_prior_logits=token_prior_logits if has_posterior else None,
            token_prior_weights=token_prior_weights if has_posterior else None,
            token_posterior_logits=token_posterior_logits if has_posterior else None,
            token_posterior_weights=token_posterior_weights if has_posterior else None,
            token_posterior_indices=token_posterior_indices if token_posterior_indices else None,
            token_candidate_scores=token_candidate_scores if token_candidate_scores else None,
            token_candidate_sources=token_candidate_sources if token_candidate_sources else None,
            token_candidate_sketch_indices=token_candidate_sketch_indices if token_candidate_sketch_indices else None,
            token_candidate_entropy=token_candidate_entropy if token_candidate_entropy else None,
            token_candidate_margin=token_candidate_margin if token_candidate_margin else None,
            token_candidate_fallback_used=token_candidate_fallback_used if token_candidate_fallback_used else None,
            token_candidate_effective_count=token_candidate_effective_count if token_candidate_effective_count else None,
            token_posterior_variance=token_posterior_variance if has_posterior else None,
            residual_variances=residual_variances,
            estimate_variances=estimate_variances,
            posterior_temperatures=posterior_temperatures,
            detector_damping=detector_damping,
        )

    def _denominator(self, state: SparseDDOperatorState) -> torch.Tensor:
        """Compute real denominator [B, 1, 1] from path_gains [B, K]."""

        path_power = state.path_gains.abs().pow(2)
        if state.confidence is not None:
            path_power = path_power * (state.confidence > 0).to(device=path_power.device, dtype=path_power.dtype)
        return path_power.sum(dim=1).reshape(-1, 1, 1) + float(self.config.noise_var) + self.eps


def _logit(value: float) -> float:
    return math.log(value / (1.0 - value))


def _bounded_logit(value: float, eps: float = 1e-6) -> float:
    return _logit(min(max(value, eps), 1.0 - eps))


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


def _active_confidence_scalar(
    confidence: torch.Tensor,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    eps: float,
) -> torch.Tensor:
    resolved = confidence.to(device=device, dtype=dtype)
    if resolved.shape[0] == 1 and batch_size != 1:
        resolved = resolved.expand(batch_size, -1)
    active = resolved > eps
    count = active.sum(dim=1).clamp_min(1)
    mean = (resolved * active.to(dtype=dtype)).sum(dim=1) / count.to(dtype=dtype)
    mean = torch.where(active.any(dim=1), mean, torch.ones_like(mean))
    return mean.reshape(batch_size, 1, 1)


def _temperature_tensor(
    temperature: float | torch.Tensor | None,
    default: float,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    value = default if temperature is None else temperature
    if isinstance(value, (int, float)):
        if value <= 0:
            raise ValueError("posterior temperature must be positive.")
        return torch.full((batch_size, 1, 1), float(value), device=device, dtype=dtype)
    if not torch.is_tensor(value) or torch.is_complex(value):
        raise ValueError("posterior temperature must be a positive float or real tensor.")
    tensor = value.to(device=device, dtype=dtype)
    if tensor.ndim == 0:
        tensor = tensor.reshape(1, 1, 1).expand(batch_size, 1, 1)
    elif tensor.ndim == 1:
        if tensor.shape[0] not in {1, batch_size}:
            raise ValueError("posterior temperature tensor must have shape [B] or [1].")
        tensor = tensor.reshape(-1, 1, 1)
        if tensor.shape[0] == 1 and batch_size != 1:
            tensor = tensor.expand(batch_size, 1, 1)
    elif tensor.ndim == 2:
        if tensor.shape not in {(batch_size, 1), (1, 1)}:
            raise ValueError("posterior temperature tensor must have shape [B, 1] or [1, 1].")
        if tensor.shape[0] == 1 and batch_size != 1:
            tensor = tensor.expand(batch_size, 1)
        tensor = tensor.reshape(batch_size, 1, 1)
    elif tensor.ndim == 3:
        if tensor.shape not in {(batch_size, 1, 1), (1, 1, 1)}:
            raise ValueError("posterior temperature tensor must have shape [B, 1, 1] or [1, 1, 1].")
        if tensor.shape[0] == 1 and batch_size != 1:
            tensor = tensor.expand(batch_size, 1, 1)
    else:
        raise ValueError("posterior temperature tensor must be scalar, [B], [B, 1], or [B, 1, 1].")
    if bool((tensor <= 0).any()):
        raise ValueError("posterior temperature must be positive.")
    return tensor
