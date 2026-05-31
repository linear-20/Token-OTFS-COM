"""Token DD codeword prior for unfolded sparse detection.

The prior projects intermediate DD estimates toward a soft mixture of token
DD codewords. It is an optional proximal prior inside the model-driven
equalizer and does not output bits or QAM symbols.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class TokenCodewordPrior:
    """Token DD codeword prior.

    Attributes:
        codeword_book: Complex token DD codewords with shape [V, M, N].
        prior_strength: Non-negative scalar controlling projection strength.
        temperature: Positive scalar for token prior logits [B, V].
        similarity: Similarity mode, either "real" or "abs".
    """

    codeword_book: torch.Tensor
    prior_strength: float = 1.0
    temperature: float = 0.1
    similarity: str = "real"

    def __post_init__(self) -> None:
        """Validate codeword_book [V, M, N] and scalar prior settings."""

        if not torch.is_tensor(self.codeword_book):
            raise TypeError(
                "codeword_book must be a torch.Tensor, "
                f"got {type(self.codeword_book).__name__}."
            )
        if not torch.is_complex(self.codeword_book):
            raise TypeError(
                "codeword_book must be a complex tensor, "
                f"got dtype {self.codeword_book.dtype}."
            )
        if self.codeword_book.ndim != 3:
            raise ValueError(
                "codeword_book must have 3 dimensions [V, M, N], "
                f"got ndim={self.codeword_book.ndim}."
            )
        V, M, N = self.codeword_book.shape
        if V <= 0:
            raise ValueError(f"codeword_book V must be > 0, got V={V}.")
        if M <= 0:
            raise ValueError(f"codeword_book M must be > 0, got M={M}.")
        if N <= 0:
            raise ValueError(f"codeword_book N must be > 0, got N={N}.")
        if not torch.isfinite(self.codeword_book).all():
            raise ValueError(
                "codeword_book must contain only finite values."
            )

        # prior_strength: number, not bool, finite, >= 0
        if isinstance(self.prior_strength, bool):
            raise TypeError(
                "prior_strength must be a number, not bool."
            )
        if not isinstance(self.prior_strength, (int, float)):
            raise TypeError(
                f"prior_strength must be a number, "
                f"got {type(self.prior_strength).__name__}."
            )
        if not math.isfinite(self.prior_strength):
            raise ValueError("prior_strength must be finite.")
        if self.prior_strength < 0:
            raise ValueError("prior_strength must be non-negative.")

        # temperature: number, not bool, finite, > 0
        if isinstance(self.temperature, bool):
            raise TypeError(
                "temperature must be a number, not bool."
            )
        if not isinstance(self.temperature, (int, float)):
            raise TypeError(
                f"temperature must be a number, "
                f"got {type(self.temperature).__name__}."
            )
        if not math.isfinite(self.temperature):
            raise ValueError("temperature must be finite.")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive.")

        # similarity: str, in {"real", "abs"}
        if not isinstance(self.similarity, str):
            raise TypeError(
                f"similarity must be str, "
                f"got {type(self.similarity).__name__}."
            )
        if self.similarity not in {"real", "abs"}:
            raise ValueError(
                f"similarity must be 'real' or 'abs', "
                f"got '{self.similarity}'."
            )


@dataclass
class TokenPriorProjection:
    """Token prior projection output.

    Attributes:
        projected_dd: Complex projected DD tensor with shape [B, M, N].
        token_prior_logits: Float token-prior logits with shape [B, V].
        token_weights: Float token mixture weights with shape [B, V].
    """

    projected_dd: torch.Tensor
    token_prior_logits: torch.Tensor
    token_weights: torch.Tensor


@dataclass
class TokenPosteriorProxOutput:
    """Token posterior proximal output for DD codewords.

    Attributes:
        projected_dd: Complex posterior mean DD tensor with shape [B, M, N].
        posterior_logits: Float posterior logits with shape [B, K] when
            top-k candidates are used, otherwise [B, V].
        posterior_weights: Float posterior weights with shape [B, K] when
            top-k candidates are used, otherwise [B, V].
        candidate_indices: Optional long candidate token indices with shape
            [B, K] when top-k candidates are used.
        candidate_scores: Optional float sketch/candidate scores [B, K].
        candidate_source: Source label for the candidate set.
        sketch_indices: Optional long sketch DD positions [S] or [B, S].
        candidate_entropy: Optional float normalised candidate entropy [B].
        candidate_margin: Optional float top1-top2 candidate margin [B].
        fallback_used: Optional bool [B] whether adaptive fallback triggered.
        effective_candidate_count: Int or [B] actual K used.
        posterior_variance: Optional float posterior variance map with shape
            [B, M, N].
    """

    projected_dd: torch.Tensor
    posterior_logits: torch.Tensor
    posterior_weights: torch.Tensor
    candidate_indices: torch.Tensor | None = None
    candidate_scores: torch.Tensor | None = None
    candidate_source: str = "full_vocab"
    sketch_indices: torch.Tensor | None = None
    candidate_entropy: torch.Tensor | None = None
    candidate_margin: torch.Tensor | None = None
    fallback_used: torch.Tensor | None = None
    effective_candidate_count: int | torch.Tensor = 0
    posterior_variance: torch.Tensor | None = None


@dataclass
class TokenCandidateSelection:
    """Sketch-based token candidate selection output.

    Attributes:
        candidate_indices: Long candidate token indices [B, K].
        candidate_scores: Float sketch scores [B, K].
        sketch_indices: Long flat DD grid positions used [S] or [B, S].
        sketch_size: Int number of DD grid positions used.
        mode: Selection mode string.
        effective_candidate_count: Int or [B] actual K used after expansion.
        candidate_entropy: Optional float normalised entropy [B].
        candidate_margin: Optional float top1-top2 margin [B].
        fallback_used: Optional bool [B] whether adaptive fallback triggered.
        score_mode: str score mode used.
    """

    candidate_indices: torch.Tensor
    candidate_scores: torch.Tensor | None = None
    sketch_indices: torch.Tensor | None = None
    sketch_size: int = 0
    mode: str = ""
    effective_candidate_count: int | torch.Tensor = 0
    candidate_entropy: torch.Tensor | None = None
    candidate_margin: torch.Tensor | None = None
    fallback_used: torch.Tensor | None = None
    score_mode: str = "distance"


def token_codeword_projection(z_dd: torch.Tensor, prior: TokenCodewordPrior) -> TokenPriorProjection:
    """Project z_dd [B, M, N] onto token DD codewords [V, M, N].

    Args:
        z_dd: Complex DD tensor with shape [B, M, N].
        prior: TokenCodewordPrior with complex codeword_book [V, M, N].

    Returns:
        TokenPriorProjection with projected_dd [B, M, N],
        token_prior_logits [B, V], and token_weights [B, V]. The operation is
        differentiable with respect to z_dd and prior.codeword_book.
    """

    _validate_projection_inputs(z_dd, prior)
    codeword_book = prior.codeword_book.to(device=z_dd.device, dtype=z_dd.dtype)
    batch_size, delay_bins, doppler_bins = z_dd.shape
    vocab_size = codeword_book.shape[0]
    z_flat = z_dd.reshape(batch_size, -1)
    codeword_flat = codeword_book.reshape(vocab_size, -1)

    numerator = z_flat @ codeword_flat.conj().transpose(0, 1)
    z_norm = z_flat.abs().pow(2).sum(dim=-1, keepdim=True).sqrt()
    codeword_norm = codeword_flat.abs().pow(2).sum(dim=-1).sqrt().reshape(1, vocab_size)
    corr = numerator / (z_norm * codeword_norm).clamp_min(1e-12)

    if prior.similarity == "real":
        logits = corr.real / float(prior.temperature)
    else:
        logits = corr.abs() / float(prior.temperature)
    token_weights = F.softmax(logits, dim=-1)
    projected_flat = token_weights.to(dtype=codeword_flat.dtype) @ codeword_flat
    projected_dd = projected_flat.reshape(batch_size, delay_bins, doppler_bins)
    return TokenPriorProjection(
        projected_dd=projected_dd,
        token_prior_logits=logits,
        token_weights=token_weights,
    )


def select_token_candidates_by_sketch(
    z_dd: torch.Tensor,
    token_prior: TokenCodewordPrior,
    num_candidates: int,
    data_mask: torch.Tensor | None = None,
    confidence_map: torch.Tensor | None = None,
    uncertainty_map: torch.Tensor | None = None,
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
) -> TokenCandidateSelection:
    """Select top-K token candidates via data-adaptive DD sketch.

    Three sketch modes: "strided", "energy_topk", "hybrid".
    Three score modes: "distance", "corr", "energy_weighted_distance".

    When ``adaptive=True``, entropy/margin diagnostics may expand the
    candidate set if the selector is uncertain (no token_ids used).
    """
    _validate_projection_inputs(z_dd, token_prior)
    vocab_size = token_prior.codeword_book.shape[0]
    if not isinstance(num_candidates, int) or num_candidates <= 0:
        raise ValueError("num_candidates must be a positive integer.")
    if num_candidates > vocab_size:
        raise ValueError("num_candidates must be <= vocab size.")
    if sketch_mode not in {"strided", "energy_topk", "hybrid"}:
        raise ValueError('sketch_mode must be "strided", "energy_topk", or "hybrid".')
    if score_mode not in {"distance", "corr", "energy_weighted_distance"}:
        raise ValueError('score_mode must be "distance", "corr", or "energy_weighted_distance".')

    batch_size, M, N = z_dd.shape
    total_positions = M * N
    default_sketch = min(total_positions, 64)
    S = min(default_sketch if sketch_size is None else int(sketch_size), total_positions)
    if S <= 0:
        raise ValueError(f"sketch_size must be positive.")

    # per-position energy weight
    dm = _optional_map(data_mask, z_dd, "data_mask", default=1.0)
    cf = _optional_map(confidence_map, z_dd, "confidence_map", default=1.0)
    uc = _optional_map(uncertainty_map, z_dd, "uncertainty_map", default=0.0).clamp_min(0.0)
    pos_weight = dm * cf / (1.0 + uc)    # [B, M, N]
    pos_energy = z_dd.abs().pow(2) * pos_weight  # [B, M, N]

    z_flat = z_dd.reshape(batch_size, total_positions)
    energy_flat = pos_energy.reshape(batch_size, total_positions)
    weight_flat = pos_weight.reshape(batch_size, total_positions)

    # ---- build per-sample sketch indices [B, S] ----
    if sketch_mode == "strided":
        stride = max(1, total_positions // S)
        base = torch.arange(0, total_positions, stride, device=z_dd.device)[:S]  # [S]
        sketch_idx = base.unsqueeze(0).expand(batch_size, -1)                     # [B, S]

    elif sketch_mode == "energy_topk":
        _, top_pos = torch.topk(energy_flat, k=S, dim=-1)                         # [B, S]
        sketch_idx = top_pos

    else:  # hybrid
        half = max(1, S // 2)
        stride = max(1, total_positions // half)
        base = torch.arange(0, total_positions, stride, device=z_dd.device)[:half]  # [half]
        _, top_energy = torch.topk(energy_flat, k=half, dim=-1)                     # [B, half]
        # deduplicate and pad
        combined_lists: list[torch.Tensor] = []
        for b in range(batch_size):
            merged = torch.cat([base, top_energy[b]])
            unique = torch.unique(merged)
            if unique.numel() < S:
                # pad from strided positions not already included
                full_range = torch.arange(0, total_positions, device=z_dd.device)
                mask = torch.ones(total_positions, dtype=torch.bool, device=z_dd.device)
                mask[unique] = False
                remaining = full_range[mask]
                pad_needed = S - unique.numel()
                take = min(pad_needed, remaining.numel())
                if take > 0:
                    extra = remaining[:take]
                    unique = torch.cat([unique, extra])
            else:
                unique = unique[:S]
            combined_lists.append(unique)
        sketch_idx = torch.stack(combined_lists, dim=0)  # [B, S]

    actual_S = sketch_idx.shape[1]

    codeword_flat = token_prior.codeword_book.to(device=z_dd.device, dtype=z_dd.dtype).reshape(vocab_size, total_positions)

    # ---- score on sketch (per-batch loop when indices vary) ----
    z_sketch = torch.gather(z_flat, dim=-1, index=sketch_idx.to(torch.long))      # [B, S]
    w_sketch = torch.gather(weight_flat, dim=-1, index=sketch_idx.to(torch.long))  # [B, S]

    all_scores = torch.zeros(batch_size, vocab_size, device=z_dd.device, dtype=z_dd.real.dtype)
    for b in range(batch_size):
        idx_b = sketch_idx[b].to(torch.long)     # [S]
        cw_b = codeword_flat[:, idx_b]            # [V, S]
        z_b = z_sketch[b:b + 1]                   # [1, S]
        w_b = w_sketch[b:b + 1]                   # [1, S]

        if score_mode == "energy_weighted_distance":
            if w_b.clamp_min(0.0).sum() < 1e-12:
                w_clamped = torch.ones_like(w_b)
            else:
                w_clamped = w_b.clamp(float(reliability_floor), 1.0)
            w_sum = w_clamped.sum(dim=-1).clamp_min(1e-12)  # [1]
            diff = z_b - cw_b                     # [V, S]
            w_sq = (w_clamped * diff.abs().pow(2)).sum(dim=-1)  # [V]
            all_scores[b] = -w_sq / w_sum
        elif score_mode == "distance":
            w_clamped = w_b.clamp_min(0.0)
            # if all weights are zero, fallback to uniform
            if w_clamped.sum() < 1e-12:
                w_clamped = torch.ones_like(w_clamped)
            diff = z_b - cw_b                     # [V, S]
            sq_dist = (w_clamped * diff.abs().pow(2)).sum(dim=-1)  # [V]
            all_scores[b] = -sq_dist
        else:  # corr
            z_norm_b = z_b.abs().pow(2).sum(dim=-1).sqrt().clamp_min(1e-12)         # [1]
            cw_norm_b = cw_b.abs().pow(2).sum(dim=-1).sqrt().clamp_min(1e-12)       # [V]
            numerator = (z_b * cw_b.conj()).sum(dim=-1).real                         # [V]
            all_scores[b] = numerator / (z_norm_b * cw_norm_b + 1e-12)               # [V]

    topk_scores, topk_indices = torch.topk(all_scores, k=num_candidates, dim=-1)
    cand_entropy_norm, cand_margin = _candidate_score_diagnostics(topk_scores)

    # ---- adaptive fallback ----
    effective_K = num_candidates
    fallback_used = torch.zeros(batch_size, dtype=torch.bool, device=z_dd.device)
    if adaptive:
        triggers = (cand_entropy_norm > float(entropy_threshold)) | (cand_margin < float(margin_threshold))
        fallback_used = triggers
        if bool(triggers.any()):
            min_k = int(min_candidates) if min_candidates is not None else num_candidates
            max_k_val = int(max_candidates) if max_candidates is not None else vocab_size
            expanded_k = min(max_k_val, max(min_k, int(float(num_candidates) * float(expand_factor))))
            if allow_full_fallback and expanded_k >= vocab_size:
                expanded_k = vocab_size
            if expanded_k > num_candidates:
                effective_K = expanded_k
                topk_scores, topk_indices = torch.topk(all_scores, k=effective_K, dim=-1)
                cand_entropy_norm, cand_margin = _candidate_score_diagnostics(topk_scores)

    return TokenCandidateSelection(
        candidate_indices=topk_indices,
        candidate_scores=topk_scores,
        sketch_indices=sketch_idx,
        sketch_size=actual_S,
        mode=sketch_mode,
        effective_candidate_count=effective_K,
        candidate_entropy=cand_entropy_norm,
        candidate_margin=cand_margin,
        fallback_used=fallback_used,
        score_mode=score_mode,
    )


def _candidate_score_diagnostics(candidate_scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized entropy [B] and top1-top2 margin [B] from scores [B, K]."""
    probs = torch.softmax(candidate_scores, dim=-1)
    log_probs = torch.log(probs.clamp_min(1e-12))
    entropy_raw = -(probs * log_probs).sum(dim=-1)
    K = candidate_scores.shape[-1]
    if K <= 1:
        entropy = torch.zeros(candidate_scores.shape[0], device=candidate_scores.device, dtype=candidate_scores.dtype)
        margin = torch.ones_like(entropy)
        return entropy, margin
    max_entropy = torch.log(torch.tensor(float(K), device=candidate_scores.device, dtype=candidate_scores.dtype))
    entropy = (entropy_raw / max_entropy.clamp_min(1e-12)).clamp(0.0, 1.0)
    top2 = probs.topk(k=2, dim=-1).values
    margin = (top2[:, 0] - top2[:, 1]).clamp(0.0, 1.0)
    return entropy, margin


def token_codeword_posterior_prox(
    z_dd: torch.Tensor,
    token_prior: TokenCodewordPrior,
    data_mask: torch.Tensor | None = None,
    confidence_map: torch.Tensor | None = None,
    uncertainty_map: torch.Tensor | None = None,
    topk_candidates: int | None = None,
    candidate_indices: torch.Tensor | None = None,
    candidate_scores: torch.Tensor | None = None,
    candidate_source: str | None = None,
    sketch_indices: torch.Tensor | None = None,
    candidate_entropy: torch.Tensor | None = None,
    candidate_margin: torch.Tensor | None = None,
    fallback_used: torch.Tensor | None = None,
    effective_candidate_count: int | torch.Tensor = 0,
    temperature: float | torch.Tensor | None = None,
    eps: float = 1e-8,
) -> TokenPosteriorProxOutput:
    """Apply token-codeword posterior prox to z_dd [B, M, N].

    Args:
        z_dd: Complex DD tensor with shape [B, M, N].
        token_prior: TokenCodewordPrior with codeword_book [V, M, N].
        data_mask: Optional real/bool data-region mask with shape [B, M, N]
            or [1, M, N]. If omitted, all positions are used.
        confidence_map: Optional real confidence map with shape [B, M, N] or
            [1, M, N]. If omitted, all positions use confidence 1.
        uncertainty_map: Optional real uncertainty map with shape [B, M, N] or
            [1, M, N]. If omitted, uncertainty is zero.
        topk_candidates: Optional positive K.  If supplied AND
            ``candidate_indices`` is None, full-vocab distances are computed
            and top-K selected (backward-compatible baseline).
        candidate_indices: Optional long [B, K].  When provided, only the
            K codewords at these indices are evaluated, avoiding full
            [B, V, M, N] distance computation.  Takes priority over
            ``topk_candidates``.
        candidate_scores: Optional float [B, K] sketch scores passed through
            to the output metadata.
        temperature: Optional positive posterior temperature. If omitted,
            token_prior.temperature is used.
        eps: Positive numerical stability scalar.

    Returns:
        TokenPosteriorProxOutput.
    """

    _validate_projection_inputs(z_dd, token_prior)
    if not isinstance(eps, (int, float)) or eps <= 0:
        raise ValueError("eps must be positive.")
    codeword_book = token_prior.codeword_book.to(device=z_dd.device, dtype=z_dd.dtype)
    batch_size, delay_bins, doppler_bins = z_dd.shape
    vocab_size = codeword_book.shape[0]
    tau = _posterior_temperature(temperature, token_prior.temperature, z_dd, batch_size, eps)

    data = _optional_map(data_mask, z_dd, "data_mask", default=1.0).clamp(0.0, 1.0)
    confidence = _optional_map(confidence_map, z_dd, "confidence_map", default=1.0).clamp(0.0, 1.0)
    uncertainty = _optional_map(uncertainty_map, z_dd, "uncertainty_map", default=0.0).clamp_min(0.0)
    effective_weight = data * confidence / (1.0 + uncertainty)
    effective_weight = effective_weight.clamp_min(0.0)

    out_candidate_source = candidate_source if candidate_source is not None else "full_vocab"
    out_candidate_scores = candidate_scores
    out_sketch_indices = sketch_indices

    # --- explicit candidate mode: only compute K distances, no full-vocab ---
    if candidate_indices is not None:
        if candidate_indices.ndim != 2 or candidate_indices.shape[0] != batch_size:
            raise ValueError("candidate_indices must have shape [B, K].")
        K = candidate_indices.shape[1]
        if K <= 0 or K > vocab_size:
            raise ValueError("candidate_indices K must be in [1, V].")
        cw_candidates = codeword_book[candidate_indices]                    # [B, K, M, N]
        diff = z_dd.unsqueeze(1) - cw_candidates                           # [B, K, M, N]
        distance = (effective_weight.unsqueeze(1) * diff.abs().pow(2)).sum(dim=(-2, -1))  # [B, K]
        posterior_logits = -distance / tau
        out_candidate_indices = candidate_indices
        if out_candidate_scores is None:
            out_candidate_source = "explicit"
        else:
            out_candidate_source = "sketch"
        candidate_codewords = cw_candidates

    # --- backward-compatible full-vocab path ---
    else:
        if topk_candidates is not None:
            if not isinstance(topk_candidates, int) or topk_candidates <= 0:
                raise ValueError("topk_candidates must be None or a positive integer.")
            if topk_candidates > vocab_size:
                raise ValueError("topk_candidates must be <= vocab size.")

        diff = z_dd.unsqueeze(1) - codeword_book.unsqueeze(0)              # [B, V, M, N]
        distance = (effective_weight.unsqueeze(1) * diff.abs().pow(2)).sum(dim=(-2, -1))  # [B, V]
        logits = -distance / tau

        if topk_candidates is None:
            posterior_logits = logits
            out_candidate_indices = None
            candidate_codewords = codeword_book.unsqueeze(0).expand(batch_size, -1, -1, -1)
            if out_candidate_source == "full_vocab":
                pass  # keep default
        else:
            posterior_logits, out_candidate_indices = torch.topk(logits, k=topk_candidates, dim=-1)
            candidate_codewords = codeword_book[out_candidate_indices]
            if candidate_source is None:
                out_candidate_source = "full_topk"

    posterior_weights = F.softmax(posterior_logits, dim=-1)
    projected_dd = (posterior_weights.to(dtype=candidate_codewords.dtype).unsqueeze(-1).unsqueeze(-1) * candidate_codewords).sum(dim=1)
    variance = (
        posterior_weights.unsqueeze(-1).unsqueeze(-1)
        * (candidate_codewords - projected_dd.unsqueeze(1)).abs().pow(2)
    ).sum(dim=1)
    return TokenPosteriorProxOutput(
        projected_dd=projected_dd,
        posterior_logits=posterior_logits,
        posterior_weights=posterior_weights,
        candidate_indices=out_candidate_indices,
        candidate_scores=out_candidate_scores,
        candidate_source=out_candidate_source,
        sketch_indices=out_sketch_indices,
        candidate_entropy=candidate_entropy,
        candidate_margin=candidate_margin,
        fallback_used=fallback_used,
        effective_candidate_count=effective_candidate_count,
        posterior_variance=torch.nan_to_num(variance, nan=0.0, posinf=1e6, neginf=0.0),
    )


def _validate_projection_inputs(z_dd: torch.Tensor, prior: TokenCodewordPrior) -> None:
    if not torch.is_tensor(z_dd) or z_dd.ndim != 3 or not torch.is_complex(z_dd):
        raise TypeError("z_dd must be a complex tensor with shape [B, M, N].")
    if prior.codeword_book.shape[-2:] != z_dd.shape[-2:]:
        raise ValueError("codeword_book must have shape [V, M, N] matching z_dd [B, M, N].")


def _optional_map(
    value: torch.Tensor | None,
    z_dd: torch.Tensor,
    name: str,
    default: float,
) -> torch.Tensor:
    if value is None:
        return torch.full_like(z_dd.real, float(default))
    if not torch.is_tensor(value) or value.ndim != 3 or torch.is_complex(value):
        raise ValueError(f"{name} must have real shape [B, M, N] or [1, M, N].")
    if value.shape[-2:] != z_dd.shape[-2:] or value.shape[0] not in {1, z_dd.shape[0]}:
        raise ValueError(f"{name} must have real shape [B, M, N] or [1, M, N].")
    resolved = value.to(device=z_dd.device, dtype=z_dd.real.dtype)
    if resolved.shape[0] == 1 and z_dd.shape[0] != 1:
        resolved = resolved.expand(z_dd.shape[0], -1, -1)
    return resolved


def _posterior_temperature(
    temperature: float | torch.Tensor | None,
    default: float,
    z_dd: torch.Tensor,
    batch_size: int,
    eps: float,
) -> torch.Tensor:
    if temperature is None:
        return torch.full((batch_size, 1), float(default), device=z_dd.device, dtype=z_dd.real.dtype)
    if isinstance(temperature, (int, float)):
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        return torch.full((batch_size, 1), float(temperature), device=z_dd.device, dtype=z_dd.real.dtype)
    if not torch.is_tensor(temperature) or torch.is_complex(temperature):
        raise ValueError("temperature must be a positive float or real tensor.")
    tau = temperature.to(device=z_dd.device, dtype=z_dd.real.dtype)
    if tau.ndim == 0:
        tau = tau.reshape(1, 1).expand(batch_size, 1)
    elif tau.ndim == 1:
        if tau.shape[0] not in {1, batch_size}:
            raise ValueError("temperature tensor must have shape [B] or [1].")
        tau = tau.reshape(-1, 1)
        if tau.shape[0] == 1 and batch_size != 1:
            tau = tau.expand(batch_size, 1)
    elif tau.ndim == 2:
        if tau.shape not in {(batch_size, 1), (1, 1)}:
            raise ValueError("temperature tensor must have shape [B, 1] or [1, 1].")
        if tau.shape[0] == 1 and batch_size != 1:
            tau = tau.expand(batch_size, 1)
    elif tau.ndim == 3:
        if tau.shape not in {(batch_size, 1, 1), (1, 1, 1)}:
            raise ValueError("temperature tensor must have shape [B, 1, 1] or [1, 1, 1].")
        if tau.shape[0] == 1 and batch_size != 1:
            tau = tau.expand(batch_size, 1, 1)
        tau = tau.reshape(batch_size, 1)
    else:
        raise ValueError("temperature tensor must be scalar, [B], [B, 1], or [B, 1, 1].")
    if bool((tau <= 0).any()):
        raise ValueError("temperature must be positive.")
    return tau.clamp_min(float(eps))
