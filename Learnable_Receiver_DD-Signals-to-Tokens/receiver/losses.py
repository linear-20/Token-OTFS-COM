"""Loss functions for receiver training.

The losses operate on token logits [B, vocab_size], receiver embeddings [B, D],
and optional DD tensors [B, M, N]. No semantic-model loss is included.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


# --- dataclass helpers ------------------------------------------------

@dataclass
class ReceiverLossWeights:
    """Scalar weights for the paper-level receiver loss components.

    All weights default to 0.0 except token_ce_weight = 1.0 so a bare call
    to ``paper_receiver_loss`` gives at least the primary token CE loss.
    """

    token_ce_weight: float = 1.0
    aux_ce_weight: float = 0.0
    embedding_contrastive_weight: float = 0.0
    dd_codeword_weight: float = 0.0
    data_consistency_weight: float = 0.0
    posterior_nll_weight: float = 0.0
    denoiser_delta_weight: float = 0.0
    physics_ce_weight: float = 0.0
    channel_supervised_weight: float = 0.0
    support_bce_weight: float = 0.0
    calibration_weight: float = 0.0
    reliability_alignment_weight: float = 0.0
    posterior_entropy_weight: float = 0.0
    classifier_head_diversity_weight: float = 0.0
    token_logit_fusion_consistency_weight: float = 0.0
    candidate_margin_weight: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "token_ce_weight", "aux_ce_weight", "embedding_contrastive_weight",
            "dd_codeword_weight", "data_consistency_weight", "posterior_nll_weight",
            "denoiser_delta_weight", "physics_ce_weight", "channel_supervised_weight",
            "support_bce_weight", "calibration_weight", "reliability_alignment_weight",
            "posterior_entropy_weight", "classifier_head_diversity_weight",
            "token_logit_fusion_consistency_weight", "candidate_margin_weight",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a float.")
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative.")


@dataclass
class ReceiverLossOutput:
    """Named loss components for logging and monitoring.

    Attributes:
        total: Scalar total loss tensor (sum of enabled components).
        components: Dict mapping component names to scalar tensors.
    """

    total: torch.Tensor
    components: dict[str, torch.Tensor] = field(default_factory=dict)


# --- existing API (preserved unchanged) --------------------------------

def token_ce_loss(token_logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """Compute CE loss from token_logits [B, vocab_size] and token_ids [B]."""

    if token_logits.ndim != 2:
        raise ValueError(f"token_logits must have shape [B, vocab_size]; got {list(token_logits.shape)}.")
    if token_ids.ndim != 1:
        raise ValueError(f"token_ids must have shape [B]; got {list(token_ids.shape)}.")
    if token_ids.shape[0] != token_logits.shape[0]:
        raise ValueError("token_ids batch size must match token_logits.")
    return F.cross_entropy(token_logits, token_ids.long())


def token_cross_entropy(token_logits: torch.Tensor, target_tokens: torch.Tensor) -> torch.Tensor:
    """Compatibility alias for CE from token_logits [B, vocab_size] and targets [B]."""

    return token_ce_loss(token_logits, target_tokens)


def contrastive_token_loss(
    rx_embedding: torch.Tensor,
    token_codebook: torch.Tensor,
    token_ids: torch.Tensor,
    temperature: float | torch.Tensor = 0.1,
) -> torch.Tensor:
    """Compute normalized embedding CE loss.

    Args:
        rx_embedding: Float receiver embeddings with shape [B, D].
        token_codebook: Float token codebook with shape [vocab_size, D].
        token_ids: Long token ids with shape [B].
        temperature: Positive scalar temperature.

    Returns:
        Scalar contrastive CE loss.
    """

    if rx_embedding.ndim != 2:
        raise ValueError("rx_embedding must have shape [B, D].")
    if token_codebook.ndim != 2:
        raise ValueError("token_codebook must have shape [vocab_size, D].")
    if rx_embedding.shape[1] != token_codebook.shape[1]:
        raise ValueError("rx_embedding and token_codebook embedding dimensions must match.")
    if token_ids.ndim != 1 or token_ids.shape[0] != rx_embedding.shape[0]:
        raise ValueError("token_ids must have shape [B].")
    temp = torch.as_tensor(temperature, device=rx_embedding.device, dtype=rx_embedding.dtype).clamp_min(1e-6)
    logits = F.normalize(rx_embedding, dim=-1) @ F.normalize(token_codebook, dim=-1).transpose(0, 1)
    return F.cross_entropy(logits / temp, token_ids.long())


def total_receiver_loss(
    token_logits: torch.Tensor,
    token_ids: torch.Tensor,
    aux_logits: list[torch.Tensor] | None = None,
    h_hat: torch.Tensor | None = None,
    h_true: torch.Tensor | None = None,
    support_logits: torch.Tensor | None = None,
    support_true: torch.Tensor | None = None,
    h_sparse: torch.Tensor | None = None,
    rx_embedding: torch.Tensor | None = None,
    token_codebook: torch.Tensor | None = None,
    ce_weight: float = 1.0,
    aux_ce_weight: float = 0.0,
    channel_nmse_weight: float = 0.0,
    support_bce_weight: float = 0.0,
    sparsity_l1_weight: float = 0.0,
    contrastive_weight: float = 0.0,
    contrastive_temperature: float = 0.1,
) -> torch.Tensor:
    """Combine receiver losses.

    Required:
        token_logits: Float logits with shape [B, vocab_size].
        token_ids: Long token ids with shape [B].

    Optional:
        aux_logits: List of float logits, each with shape [B, vocab_size].
        h_hat/h_true: Complex or real DD tensors with shape [B, M, N] for NMSE.
        support_logits/support_true: Float DD tensors with shape [B, M, N].
        h_sparse: Complex sparse DD tensor with shape [B, M, N] for L1.
        rx_embedding/token_codebook: Float tensors [B, D] and [vocab_size, D].

    Returns:
        Scalar weighted receiver loss.
    """

    loss = token_logits.new_tensor(0.0)
    loss = loss + float(ce_weight) * token_ce_loss(token_logits, token_ids)

    if aux_logits is not None and aux_ce_weight != 0.0:
        if len(aux_logits) > 0:
            aux_loss = sum(token_ce_loss(logits, token_ids) for logits in aux_logits) / len(aux_logits)
            loss = loss + float(aux_ce_weight) * aux_loss

    if h_hat is not None and h_true is not None and channel_nmse_weight != 0.0:
        loss = loss + float(channel_nmse_weight) * _channel_nmse(h_hat, h_true).to(loss.device)

    if support_logits is not None and support_true is not None and support_bce_weight != 0.0:
        loss = loss + float(support_bce_weight) * F.binary_cross_entropy_with_logits(
            support_logits,
            support_true.to(device=support_logits.device, dtype=support_logits.dtype),
        ).to(loss.device)

    if h_sparse is not None and sparsity_l1_weight != 0.0:
        loss = loss + float(sparsity_l1_weight) * h_sparse.abs().mean().to(loss.device)

    if (
        rx_embedding is not None
        and token_codebook is not None
        and contrastive_weight != 0.0
    ):
        loss = loss + float(contrastive_weight) * contrastive_token_loss(
            rx_embedding,
            token_codebook,
            token_ids,
            temperature=contrastive_temperature,
        ).to(loss.device)

    return loss


# --- new helpers -------------------------------------------------------

def weighted_complex_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    weight: torch.Tensor | None = None,
    noise_var: torch.Tensor | float | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Weighted complex MSE for pred/target with shape [B, M, N].

    loss = mean(w * m * |pred - target|^2) / max(mean(w * m), eps)

    If *noise_var* is supplied the result is divided by noise_var so the
    loss approximates a Gaussian NLL scale.

    Returns a scalar tensor.  When mask is everywhere zero the numerator
    is zero and the denominator is clamped at ``eps``, yielding zero.
    """
    if not torch.is_tensor(pred) or pred.ndim != 3 or not torch.is_complex(pred):
        raise ValueError("pred must be complex with shape [B, M, N].")
    if not torch.is_tensor(target) or target.shape != pred.shape or not torch.is_complex(target):
        raise ValueError("target must be complex with shape [B, M, N] matching pred.")

    resolved_mask = _resolve_real_map(mask, pred, "mask", default=1.0)
    resolved_weight = _resolve_real_map(weight, pred, "weight", default=1.0).clamp_min(0.0)
    combined = (resolved_mask * resolved_weight).clamp_min(0.0)

    sq_err = (pred - target).abs().pow(2)
    weighted_sum = (combined * sq_err).sum()
    weight_sum = combined.sum().clamp_min(float(eps))
    loss = weighted_sum / weight_sum

    if noise_var is not None:
        nv = torch.as_tensor(noise_var, device=pred.device, dtype=pred.real.dtype)
        if nv.ndim == 0 or (nv.ndim == 1 and nv.shape[0] == pred.shape[0]):
            nv = nv.reshape(-1, 1, 1).expand_as(pred.real)
        loss = loss / nv.mean().clamp_min(float(eps))

    return torch.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=0.0)


# --- new paper-level loss components ------------------------------------

def dd_data_consistency_loss(
    x_dd: torch.Tensor,
    y_dd: torch.Tensor,
    dd_operator: "SparseDDOperator",
    operator_state: "SparseDDOperatorState",
    data_mask: torch.Tensor | None = None,
    reliability_map: torch.Tensor | None = None,
    noise_var: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Penalise inconsistency between *x_dd* and *y_dd* via the sparse operator.

    y_hat = H(x_dd),  loss = weighted MSE(y_hat, y_dd).

    Only ``dd_operator.apply`` is called; no dense matrix is built.
    """
    y_hat = dd_operator.apply(x_dd, operator_state)
    return weighted_complex_mse(
        y_hat, y_dd,
        mask=data_mask,
        weight=reliability_map,
        noise_var=noise_var,
    )


def token_dd_codeword_loss(
    x_dd: torch.Tensor,
    token_ids: torch.Tensor,
    dd_codeword_book: torch.Tensor,
    data_mask: torch.Tensor | None = None,
    reliability_map: torch.Tensor | None = None,
) -> torch.Tensor:
    """Supervise *x_dd* to match the ground-truth DD codeword.

    Args:
        x_dd: Complex [B, M, N] -- typically x_equalized or x_refined.
        token_ids: Long [B] ground-truth token indices.
        dd_codeword_book: Complex [V, M, N] DD token codewords.
        data_mask: Optional real [B, M, N] or [1, M, N].
        reliability_map: Optional real [B, M, N] or [1, M, N].

    Returns:
        Scalar weighted complex MSE between x_dd and dd_codeword_book[token_ids].
    """
    if not torch.is_tensor(dd_codeword_book) or dd_codeword_book.ndim != 3:
        raise ValueError("dd_codeword_book must have shape [V, M, N].")
    if not torch.is_complex(dd_codeword_book):
        raise TypeError("dd_codeword_book must be complex [V, M, N].")
    if dd_codeword_book.shape[-2:] != x_dd.shape[-2:]:
        raise ValueError("dd_codeword_book spatial dims must match x_dd [B, M, N].")
    if token_ids.ndim != 1 or token_ids.shape[0] != x_dd.shape[0]:
        raise ValueError("token_ids must have shape [B].")
    target = dd_codeword_book[token_ids.long()].to(device=x_dd.device, dtype=x_dd.dtype)
    return weighted_complex_mse(x_dd, target, mask=data_mask, weight=reliability_map)


def token_posterior_nll_loss(
    posterior_logits: torch.Tensor,
    token_ids: torch.Tensor,
    candidate_indices: torch.Tensor | None = None,
    missing_penalty: float = 10.0,
) -> torch.Tensor:
    """Posterior NLL from unfolded detector token-posterior logits.

    Args:
        posterior_logits: Float [B, V] (full vocab) or [B, K] (top-K).
        token_ids: Long [B] ground-truth token indices.
        candidate_indices: Optional long [B, K] -- required when posterior_logits
            is [B, K].
        missing_penalty: Scalar penalty added when the true token is not among
            the top-K candidates.

    Returns:
        Scalar CE loss.  When working in top-K mode and the true token is
        absent, missing_penalty is added to the total.
    """
    if posterior_logits.ndim != 2:
        raise ValueError("posterior_logits must have shape [B, V] or [B, K].")
    if token_ids.ndim != 1 or token_ids.shape[0] != posterior_logits.shape[0]:
        raise ValueError("token_ids must have shape [B].")

    if candidate_indices is None:
        return F.cross_entropy(posterior_logits, token_ids.long())

    # top-K mode
    if candidate_indices.ndim != 2 or candidate_indices.shape[0] != posterior_logits.shape[0]:
        raise ValueError("candidate_indices must have shape [B, K].")
    if candidate_indices.shape[1] != posterior_logits.shape[1]:
        raise ValueError("candidate_indices K must match posterior_logits K.")

    token_ids_expanded = token_ids.long().unsqueeze(1)
    match_mask = (candidate_indices == token_ids_expanded)
    hit = match_mask.any(dim=1)
    miss = ~hit
    hit_count = hit.sum()

    if hit_count == 0:
        # all miss: flat penalty only, but keep graph connectivity
        zero_term = (posterior_logits * 0.0).sum()
        return zero_term + float(missing_penalty)

    # hit samples: CE at the correct candidate position
    hit_targets = match_mask.float().argmax(dim=1)
    ce_all = F.cross_entropy(posterior_logits, hit_targets, reduction="none")
    hit_mask_float = hit.float().to(ce_all.device)
    ce_hit = (ce_all * hit_mask_float).sum() / hit_count.clamp_min(1)

    loss = ce_hit
    miss_count = miss.sum()
    if miss_count > 0:
        loss = loss + float(missing_penalty) * (miss_count.float() / posterior_logits.shape[0])
    return loss


def denoiser_delta_regularization(
    delta: torch.Tensor,
    reliability_map: torch.Tensor | None = None,
    confidence_map: torch.Tensor | None = None,
) -> torch.Tensor:
    """Regularise denoiser correction *delta* [B, M, N].

    High reliability/confidence positions are penalised more heavily so the
    denoiser does not corrupt high-quality detector outputs.

    Args:
        delta: Complex [B, M, N].
        reliability_map: Optional real [B, M, N] or [1, M, N].
        confidence_map: Optional real [B, M, N] or [1, M, N].

    Returns:
        Scalar weighted L2 of delta.
    """
    if not torch.is_tensor(delta) or delta.ndim != 3 or not torch.is_complex(delta):
        raise ValueError("delta must be complex with shape [B, M, N].")

    weight = _resolve_real_map(reliability_map, delta, "reliability_map", default=1.0)
    conf = _resolve_real_map(confidence_map, delta, "confidence_map", default=1.0).clamp(0.0, 1.0)
    combined = (weight * conf).clamp_min(0.0)

    sq = delta.abs().pow(2)
    num = (combined * sq).sum()
    den = combined.sum().clamp_min(1e-8)
    return torch.nan_to_num(num / den, nan=0.0, posinf=1e6, neginf=0.0)


def physics_refinement_loss(
    sparse_estimate: "SparseChannelEstimate",
) -> torch.Tensor:
    """Regularise physics-guided gain refinement quality.

    Uses only CE-fit-region diagnostics (pilot / fit residual).  Full-grid
    residual power is deliberately excluded from optimisation.

    Returns a scalar (0.0 when no physics fields are available).
    """
    loss = torch.tensor(0.0, device=_estimate_device(sparse_estimate))
    count = 0

    # pilot CE residual -- the original pilot-window fit quality
    if getattr(sparse_estimate, "pilot_residual_power", None) is not None:
        loss = loss + sparse_estimate.pilot_residual_power.mean()
        count += 1

    # fit-masked residual from physics gain refinement
    fit_pow = getattr(sparse_estimate, "physics_fit_residual_power", None)
    if fit_pow is not None:
        loss = loss + fit_pow.mean()
        count += 1

    # gain covariance (parameter uncertainty)
    cov = getattr(sparse_estimate, "gain_covariance_diag", None)
    if cov is not None:
        loss = loss + cov.mean()
        count += 1

    if count == 0:
        return torch.tensor(0.0, device=loss.device)
    return loss / float(count)


# --- calibration / reliability losses -----------------------------------

def token_confidence_calibration_loss(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    num_bins: int = 10,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Differentiable expected calibration error proxy.

    Bins predictions by confidence and computes a soft histogram of
    (confidence - accuracy) per bin.  Not a strict ECE but a smooth,
    differentiable analogue suitable for gradient-based optimisation.

    Args:
        logits: Float [B, V].
        token_ids: Long [B].
        num_bins: Number of confidence bins.
        eps: Numerical stability.

    Returns:
        Scalar calibration loss.
    """
    probs = F.softmax(logits, dim=-1)
    confidence, preds = probs.max(dim=-1)
    correct = preds.eq(token_ids).float()

    bin_edges = torch.linspace(0.0, 1.0, num_bins + 1, device=logits.device)
    loss = logits.new_tensor(0.0)
    count = 0
    for i in range(num_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = ((confidence >= lo) & (confidence < hi + eps)).float()
        weight = in_bin.sum()
        if weight < eps:
            continue
        conf_bin = (confidence * in_bin).sum() / weight.clamp_min(eps)
        acc_bin = (correct * in_bin).sum() / weight.clamp_min(eps)
        loss = loss + (conf_bin - acc_bin).abs()
        count += 1
    if count == 0:
        return logits.new_tensor(0.0)
    return loss / float(count)


def reliability_error_alignment_loss(
    reliability_scalar: torch.Tensor,
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """MSE between reliability_scalar [B] and softmax confidence [B].

    reliability should correlate with correctness probability, so the
    target is p_true.detach() (the softmax probability of the true class).
    """
    probs = F.softmax(logits, dim=-1)
    target = probs.gather(1, token_ids.long().unsqueeze(1)).squeeze(1).detach()
    rel = reliability_scalar.to(device=target.device, dtype=target.dtype)
    return F.mse_loss(rel, target.clamp(float(eps), 1.0 - float(eps)))


def posterior_entropy_regularization(
    posterior_logits: torch.Tensor,
    target_entropy: float | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Penalise posterior entropy deviating from a target level.

    If target_entropy is None, penalises both extremes equally
    (returns mean normalised entropy, encouraging moderate spread).
    If target_entropy is a float, returns MSE(entropy, target).
    """
    probs = F.softmax(posterior_logits, dim=-1)
    log_probs = torch.log(probs.clamp_min(float(eps)))
    entropy_raw = -(probs * log_probs).sum(dim=-1)
    K = posterior_logits.shape[-1]
    max_ent = torch.log(torch.tensor(K, device=posterior_logits.device, dtype=posterior_logits.dtype))
    entropy = entropy_raw / max_ent.clamp_min(float(eps))

    if target_entropy is None:
        return entropy.mean()
    target = torch.full_like(entropy, float(target_entropy))
    return F.mse_loss(entropy.clamp(0.0, 1.0), target.clamp(0.0, 1.0))


def confidence_nll_regularization(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """CE with temperature -- simple NLL baseline for calibration."""
    return F.cross_entropy(logits / max(float(temperature), 1e-6), token_ids.long())


def candidate_recall_metric(
    candidate_indices: torch.Tensor | None,
    token_ids: torch.Tensor | None,
) -> torch.Tensor:
    """Fraction of samples where the true token is in the candidate set.

    FOR EVAL/LOGGING ONLY -- must not appear in forward selection.
    Returns 0.0 when candidate_indices is None.
    """
    if candidate_indices is None or token_ids is None:
        device = token_ids.device if torch.is_tensor(token_ids) else torch.device("cpu")
        return torch.tensor(0.0, device=device)
    if candidate_indices.ndim != 2:
        raise ValueError("candidate_indices must have shape [B, K].")
    if token_ids.ndim != 1 or token_ids.shape[0] != candidate_indices.shape[0]:
        raise ValueError("token_ids must have shape [B].")
    match = (candidate_indices == token_ids.unsqueeze(1)).any(dim=1)
    return match.float().mean()


def candidate_margin_regularization(
    candidate_scores: torch.Tensor,
    target_margin: float = 0.2,
) -> torch.Tensor:
    """Encourage candidate selector confidence.
    margin = softmax(top1) - softmax(top2), penalise below target.
    """
    if candidate_scores.ndim != 2:
        raise ValueError("candidate_scores must have shape [B, K].")
    probs = torch.softmax(candidate_scores, dim=-1)
    K = candidate_scores.shape[-1]
    if K < 2:
        return candidate_scores.sum() * 0.0
    top2 = probs.topk(k=2, dim=-1).values
    margin = (top2[:, 0] - top2[:, 1]).clamp(0.0, 1.0)
    return torch.relu(float(target_margin) - margin).mean()


def token_logit_fusion_consistency_loss(
    classifier_logits: torch.Tensor,
    detector_logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Symmetric KL divergence between classifier and detector softmax.

    Encourages agreement between the two token-level evidence sources
    without using ground-truth token ids.
    """
    temp = max(float(temperature), 1e-6)
    c_probs = F.softmax(classifier_logits / temp, dim=-1)
    d_probs = F.softmax(detector_logits / temp, dim=-1)
    log_c = torch.log(c_probs.clamp_min(1e-8))
    log_d = torch.log(d_probs.clamp_min(1e-8))
    kl_cd = (c_probs * (log_c - log_d)).sum(dim=-1)
    kl_dc = (d_probs * (log_d - log_c)).sum(dim=-1)
    return (kl_cd + kl_dc).mean()


# --- paper-level composite loss -----------------------------------------

def paper_receiver_loss(
    output: "ReceiverOutput",
    token_ids: torch.Tensor,
    y_dd: torch.Tensor | None = None,
    dd_operator: "SparseDDOperator | None" = None,
    dd_codeword_book: torch.Tensor | None = None,
    embedding_codebook: torch.Tensor | None = None,
    token_prior: "TokenCodewordPrior | None" = None,
    data_mask: torch.Tensor | None = None,
    reliability_map: torch.Tensor | None = None,
    h_true: torch.Tensor | None = None,
    support_true: torch.Tensor | None = None,
    reliability_diagnostics: "ReliabilityDiagnostics | None" = None,
    calibration_num_bins: int = 10,
    posterior_entropy_target: float | None = None,
    weights: ReceiverLossWeights | None = None,
    return_components: bool = True,
) -> torch.Tensor | ReceiverLossOutput:
    """Paper-level composite receiver loss.

    Args:
        output: ``ReceiverOutput`` from ``LearnableOTFSReceiver.forward`` with
            ``return_details=True``.
        token_ids: Long [B] ground-truth token indices.
        y_dd: Optional complex [B, M, N] received DD grid for data consistency.
        dd_operator: Optional ``SparseDDOperator`` for data consistency.
        dd_codeword_book: Optional complex [V, M, N] DD token codewords.  If
            omitted and ``token_prior`` is given, ``token_prior.codeword_book``
            is used.  Must NOT be used as an embedding codebook [V, D].
        embedding_codebook: Optional float [V, D] token embedding codebook
            for contrastive loss.  Only used when
            ``weights.embedding_contrastive_weight != 0`` and an explicit
            codebook is supplied.  The DD codeword book is never used here.
        token_prior: Optional ``TokenCodewordPrior`` carrying a DD codeword
            book [V, M, N] and used by the unfolded detector prox.
        data_mask: Optional real [B, M, N] or [1, M, N].
        reliability_map: Optional real [B, M, N] or [1, M, N].
        h_true: Optional complex/real [B, M, N] ground-truth channel.
        support_true: Optional real [B, M, N] ground-truth support mask.
        reliability_diagnostics: Optional ``ReliabilityDiagnostics`` from
            receiver forward.  Used preferentially for reliability alignment
            when available.
        calibration_num_bins: Number of bins for calibration loss.
        posterior_entropy_target: Optional float target normalised entropy.
        weights: Optional ``ReceiverLossWeights``.  Defaults to all zeros
            except ``token_ce_weight=1.0``.
        return_components: If False, return total scalar only.

    Returns:
        Scalar total loss or ``ReceiverLossOutput(total, components)``.
    """
    w = ReceiverLossWeights() if weights is None else weights
    resolved_dd_codeword = dd_codeword_book
    if resolved_dd_codeword is None and token_prior is not None:
        resolved_dd_codeword = token_prior.codeword_book

    total = output.token_logits.new_tensor(0.0)
    components: dict[str, torch.Tensor] = {}

    # token CE (always computed when weight > 0)
    if w.token_ce_weight != 0.0:
        ce = token_ce_loss(output.token_logits, token_ids)
        components["token_ce"] = ce.detach().clone()
        total = total + float(w.token_ce_weight) * ce

    # aux CE from intermediate equalizer layers
    if w.aux_ce_weight != 0.0 and output.aux_logits is not None and len(output.aux_logits) > 0:
        aux = sum(token_ce_loss(logits, token_ids) for logits in output.aux_logits) / len(output.aux_logits)
        components["aux_ce"] = aux.detach().clone()
        total = total + float(w.aux_ce_weight) * aux

    # embedding contrastive -- requires explicit embedding_codebook [V, D]
    if (
        w.embedding_contrastive_weight != 0.0
        and embedding_codebook is not None
        and output.rx_embedding is not None
    ):
        emb = contrastive_token_loss(output.rx_embedding, embedding_codebook, token_ids)
        components["embedding_contrastive"] = emb.detach().clone()
        total = total + float(w.embedding_contrastive_weight) * emb

    # DD codeword supervision (uses complex [V, M, N], not [V, D])
    x_for_dd = output.x_refined if output.x_refined is not None else output.x_equalized
    if w.dd_codeword_weight != 0.0 and resolved_dd_codeword is not None and x_for_dd is not None:
        dd = token_dd_codeword_loss(x_for_dd, token_ids, resolved_dd_codeword,
                                    data_mask=data_mask, reliability_map=reliability_map)
        components["dd_codeword"] = dd.detach().clone()
        total = total + float(w.dd_codeword_weight) * dd

    # data consistency via sparse DD operator
    if (
        w.data_consistency_weight != 0.0
        and y_dd is not None
        and dd_operator is not None
        and x_for_dd is not None
    ):
        operator_state = output.operator_state
        dc = dd_data_consistency_loss(
            x_for_dd, y_dd, dd_operator, operator_state,
            data_mask=data_mask, reliability_map=reliability_map,
        )
        components["data_consistency"] = dc.detach().clone()
        total = total + float(w.data_consistency_weight) * dc

    # posterior NLL from unfolded detector
    if (
        w.posterior_nll_weight != 0.0
        and output.equalizer_output is not None
        and output.equalizer_output.token_posterior_logits is not None
        and output.equalizer_output.token_posterior_logits
    ):
        # use last layer posterior
        post_logits = output.equalizer_output.token_posterior_logits[-1]
        post_indices = None
        if output.equalizer_output.token_posterior_indices is not None and output.equalizer_output.token_posterior_indices:
            post_indices = output.equalizer_output.token_posterior_indices[-1]
        pnll = token_posterior_nll_loss(post_logits, token_ids, candidate_indices=post_indices)
        components["posterior_nll"] = pnll.detach().clone()
        total = total + float(w.posterior_nll_weight) * pnll

    # denoiser delta regularization
    if (
        w.denoiser_delta_weight != 0.0
        and output.denoiser_output is not None
        and output.denoiser_output.delta is not None
    ):
        den_conf = None
        if output.denoiser_output.confidence_map is not None:
            den_conf = output.denoiser_output.confidence_map
        dr = denoiser_delta_regularization(
            output.denoiser_output.delta,
            reliability_map=reliability_map,
            confidence_map=den_conf,
        )
        components["denoiser_delta"] = dr.detach().clone()
        total = total + float(w.denoiser_delta_weight) * dr

    # physics CE refinement
    if w.physics_ce_weight != 0.0:
        pr = physics_refinement_loss(output.sparse_estimate)
        components["physics_ce"] = pr.detach().clone()
        total = total + float(w.physics_ce_weight) * pr

    # supervised channel NMSE
    if w.channel_supervised_weight != 0.0 and h_true is not None:
        ch = _channel_nmse(output.sparse_estimate.h_dd, h_true).to(total.device)
        components["channel_supervised"] = ch.detach().clone()
        total = total + float(w.channel_supervised_weight) * ch

    # support BCE
    if (
        w.support_bce_weight != 0.0
        and support_true is not None
        and output.sparse_estimate.support_logits is not None
    ):
        sb = F.binary_cross_entropy_with_logits(
            output.sparse_estimate.support_logits,
            support_true.to(device=output.sparse_estimate.support_logits.device,
                            dtype=output.sparse_estimate.support_logits.dtype),
        ).to(total.device)
        components["support_bce"] = sb.detach().clone()
        total = total + float(w.support_bce_weight) * sb

    # calibration loss
    if w.calibration_weight != 0.0:
        cal = token_confidence_calibration_loss(
            output.token_logits, token_ids, num_bins=calibration_num_bins,
        )
        components["calibration"] = cal.detach().clone()
        total = total + float(w.calibration_weight) * cal

    # reliability alignment
    if w.reliability_alignment_weight != 0.0:
        diag = reliability_diagnostics
        if diag is None and hasattr(output, "reliability_diagnostics"):
            diag = output.reliability_diagnostics
        if diag is not None and diag.reliability_scalar is not None:
            ral = reliability_error_alignment_loss(
                diag.reliability_scalar, output.token_logits, token_ids,
            )
            components["reliability_alignment"] = ral.detach().clone()
            total = total + float(w.reliability_alignment_weight) * ral

    # posterior entropy regularization
    if w.posterior_entropy_weight != 0.0:
        post_logits = None
        if output.equalizer_output is not None and output.equalizer_output.token_posterior_logits is not None and output.equalizer_output.token_posterior_logits:
            post_logits = output.equalizer_output.token_posterior_logits[-1]
        if post_logits is not None:
            per = posterior_entropy_regularization(
                post_logits, target_entropy=posterior_entropy_target,
            )
            components["posterior_entropy_reg"] = per.detach().clone()
            total = total + float(w.posterior_entropy_weight) * per

    # classifier head diversity
    if w.classifier_head_diversity_weight != 0.0:
        head_w = None
        if (output.classifier_output is not None
                and output.classifier_output.head_evidence_weights is not None):
            head_w = output.classifier_output.head_evidence_weights
        if head_w is not None:
            from .token_classifier import evidence_head_diversity_loss
            hdiv = evidence_head_diversity_loss(head_w, data_mask=data_mask)
            components["classifier_head_diversity"] = hdiv.detach().clone()
            total = total + float(w.classifier_head_diversity_weight) * hdiv

    # candidate margin
    if w.candidate_margin_weight != 0.0:
        cand_scores = None
        if (output.equalizer_output is not None
                and output.equalizer_output.token_candidate_scores is not None
                and output.equalizer_output.token_candidate_scores):
            cand_scores = output.equalizer_output.token_candidate_scores[-1]
        if cand_scores is not None:
            cm = candidate_margin_regularization(cand_scores)
            components["candidate_margin"] = cm.detach().clone()
            total = total + float(w.candidate_margin_weight) * cm

    # fusion consistency
    if w.token_logit_fusion_consistency_weight != 0.0:
        c_logits = None
        d_logits = None
        if output.logit_fusion_output is not None:
            c_logits = output.logit_fusion_output.classifier_logits
            d_logits = output.logit_fusion_output.detector_logits
        if c_logits is not None and d_logits is not None:
            fc = token_logit_fusion_consistency_loss(c_logits, d_logits)
            components["token_logit_fusion_consistency"] = fc.detach().clone()
            total = total + float(w.token_logit_fusion_consistency_weight) * fc

    if not return_components:
        return total
    return ReceiverLossOutput(total=total, components=components)


# --- internal helpers ---------------------------------------------------

def _channel_nmse(h_hat: torch.Tensor, h_true: torch.Tensor) -> torch.Tensor:
    """Return NMSE for DD tensors h_hat/h_true [B, M, N]."""

    if h_hat.shape != h_true.shape:
        raise ValueError("h_hat and h_true must have the same shape [B, M, N].")
    return (h_hat - h_true).abs().pow(2).sum() / h_true.abs().pow(2).sum().clamp_min(1e-12)


def _resolve_real_map(
    value: torch.Tensor | None,
    reference: torch.Tensor,
    name: str,
    default: float,
) -> torch.Tensor:
    """Resolve optional [B, M, N] or [1, M, N] real map to [B, M, N]."""
    if value is None:
        return torch.full_like(reference.real, float(default))
    if not torch.is_tensor(value) or value.ndim != 3 or torch.is_complex(value):
        raise ValueError(f"{name} must have real shape [B, M, N] or [1, M, N].")
    if value.shape[-2:] != reference.shape[-2:] or value.shape[0] not in {1, reference.shape[0]}:
        raise ValueError(f"{name} must have real shape [B, M, N] or [1, M, N].")
    resolved = value.to(device=reference.device, dtype=reference.real.dtype)
    if resolved.shape[0] == 1 and reference.shape[0] != 1:
        resolved = resolved.expand(reference.shape[0], -1, -1)
    return resolved


def _estimate_device(sparse_estimate) -> torch.device:
    try:
        return sparse_estimate.h_dd.device
    except Exception:
        return torch.device("cpu")
