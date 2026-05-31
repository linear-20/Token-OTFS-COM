"""Symbol reliability maps and calibration diagnostics.

Token-free diagnostics (no token_ids used) for monitoring and calibration
losses.  Calibration losses that require token_ids live in ``losses.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


@dataclass
class ReliabilityDiagnostics:
    """Token-free reliability and calibration diagnostics.

    All fields are computed from forward outputs only -- no token_ids.

    Attributes:
        reliability_map: Float [B, M, N] input reliability map.
        reliability_scalar: Float [B] per-sample mean reliability.
        classifier_confidence: Float [B] max softmax prob.
        classifier_entropy: Float [B] normalised entropy in [0, 1].
        token_margin: Float [B] top1 - top2 softmax probability.
        posterior_confidence: Optional float [B] posterior max weight.
        posterior_entropy: Optional float [B] normalised posterior entropy.
        residual_energy_mean: Optional float [B].
        uncertainty_mean: Optional float [B] mean uncertainty.
        expected_error_proxy: Float [B] heuristic error proxy in [0, 1].
        components: Dict of named scalar diagnostic values.
    """

    reliability_map: torch.Tensor | None = None
    reliability_scalar: torch.Tensor | None = None
    classifier_confidence: torch.Tensor | None = None
    classifier_entropy: torch.Tensor | None = None
    token_margin: torch.Tensor | None = None
    posterior_confidence: torch.Tensor | None = None
    posterior_entropy: torch.Tensor | None = None
    residual_energy_mean: torch.Tensor | None = None
    uncertainty_mean: torch.Tensor | None = None
    expected_error_proxy: torch.Tensor | None = None
    components: dict[str, torch.Tensor] = field(default_factory=dict)


def softmax_confidence_and_entropy(
    logits: torch.Tensor, eps: float = 1e-8
) -> tuple[torch.Tensor, torch.Tensor]:
    """Softmax confidence and normalised entropy from logits [B, V].

    Returns:
        confidence [B] = max softmax probability.
        entropy [B] = normalised entropy in [0, 1] (0 = peaky, 1 = uniform).
    """
    probs = F.softmax(logits, dim=-1)
    confidence, _ = probs.max(dim=-1)
    log_probs = torch.log(probs.clamp_min(float(eps)))
    entropy_raw = -(probs * log_probs).sum(dim=-1)
    max_entropy = torch.log(torch.tensor(logits.shape[-1], device=logits.device, dtype=logits.dtype))
    entropy = entropy_raw / max_entropy.clamp_min(float(eps))
    return confidence, entropy.clamp(0.0, 1.0)


def posterior_confidence_entropy(
    posterior_logits: torch.Tensor, eps: float = 1e-8
) -> tuple[torch.Tensor, torch.Tensor]:
    """Confidence and normalised entropy from posterior logits [B, K] or [B, V]."""
    return softmax_confidence_and_entropy(posterior_logits, eps=eps)


def token_margin(logits: torch.Tensor) -> torch.Tensor:
    """Top1 - top2 softmax probability margin [B]."""
    probs = F.softmax(logits, dim=-1)
    top2 = probs.topk(k=min(2, logits.shape[-1]), dim=-1).values
    if top2.shape[-1] < 2:
        return top2[:, 0].clone()
    return (top2[:, 0] - top2[:, 1]).clamp(0.0, 1.0)


def build_reliability_diagnostics(
    reliability_map: torch.Tensor | None = None,
    token_logits: torch.Tensor | None = None,
    posterior_logits: torch.Tensor | None = None,
    residual_energy: torch.Tensor | None = None,
    uncertainty_map: torch.Tensor | None = None,
    data_mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> ReliabilityDiagnostics:
    """Build token-free calibration diagnostics from receiver outputs.

    All inputs are optional; missing fields are left as None.  No token_ids
    are used -- this is purely a forward diagnostic.

    expected_error_proxy combines several uncertainty signals and is intended
    as a heuristic correlate of per-sample error probability.
    """
    batch_size = 1
    device = torch.device("cpu")
    dtype = torch.float32

    for t in (reliability_map, token_logits, posterior_logits, residual_energy, uncertainty_map):
        if t is not None:
            batch_size = t.shape[0]
            device = t.device
            dtype = t.dtype if t.dtype.is_floating_point else torch.float32
            break

    diag = ReliabilityDiagnostics()
    comps: dict[str, torch.Tensor] = {}

    # reliability spatial stats
    if reliability_map is not None:
        rm = reliability_map.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
        diag.reliability_map = rm
        diag.reliability_scalar = rm.reshape(batch_size, -1).mean(dim=-1)
        comps["reliability_mean"] = diag.reliability_scalar

    # classifier stats
    if token_logits is not None:
        conf, ent = softmax_confidence_and_entropy(token_logits, eps=eps)
        diag.classifier_confidence = conf
        diag.classifier_entropy = ent
        diag.token_margin = token_margin(token_logits)
        comps["classifier_confidence"] = conf
        comps["classifier_entropy"] = ent
        comps["token_margin"] = diag.token_margin

    # posterior stats
    if posterior_logits is not None:
        pconf, pent = posterior_confidence_entropy(posterior_logits, eps=eps)
        diag.posterior_confidence = pconf
        diag.posterior_entropy = pent
        comps["posterior_confidence"] = pconf
        comps["posterior_entropy"] = pent

    # residual energy
    if residual_energy is not None:
        re = residual_energy.to(device=device, dtype=torch.float32)
        if data_mask is not None:
            dm = data_mask.to(device=device, dtype=torch.float32).clamp(0.0, 1.0)
            if dm.shape[0] == 1 and batch_size != 1:
                dm = dm.expand(batch_size, -1, -1)
            num = (dm * re).reshape(batch_size, -1).sum(dim=-1)
            den = dm.reshape(batch_size, -1).sum(dim=-1).clamp_min(float(eps))
            diag.residual_energy_mean = num / den
        else:
            diag.residual_energy_mean = re.reshape(batch_size, -1).mean(dim=-1)
        comps["residual_energy_mean"] = diag.residual_energy_mean

    # uncertainty
    if uncertainty_map is not None:
        um = uncertainty_map.to(device=device, dtype=torch.float32).clamp_min(0.0)
        diag.uncertainty_mean = um.reshape(batch_size, -1).mean(dim=-1)
        comps["uncertainty_mean"] = diag.uncertainty_mean

    # expected error proxy (heuristic, no token_ids)
    parts: list[torch.Tensor] = []
    if diag.classifier_entropy is not None:
        parts.append(diag.classifier_entropy.to(device=device, dtype=torch.float32))
    if diag.posterior_entropy is not None:
        parts.append(diag.posterior_entropy.to(device=device, dtype=torch.float32))
    if diag.residual_energy_mean is not None:
        # sigmoid-like: normalise and scale
        re_scaled = diag.residual_energy_mean / (diag.residual_energy_mean + 1.0)
        parts.append(re_scaled.to(torch.float32))
    if diag.uncertainty_mean is not None:
        uc_scaled = diag.uncertainty_mean / (diag.uncertainty_mean + 1.0)
        parts.append(uc_scaled.to(torch.float32))
    if diag.reliability_scalar is not None:
        parts.append(-diag.reliability_scalar.to(torch.float32))

    if parts:
        stacked = torch.stack([p.to(device=device) for p in parts], dim=0)
        proxy = stacked.mean(dim=0)
        diag.expected_error_proxy = torch.sigmoid(proxy).to(dtype=torch.float32)

    diag.components = comps
    return diag


def build_symbol_reliability_map(
    data_mask: torch.Tensor | None,
    path_confidence: torch.Tensor | None = None,
    path_confidence_map: torch.Tensor | None = None,
    residual_energy: torch.Tensor | None = None,
    posterior_variance: torch.Tensor | None = None,
    uncertainty_map: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Build a data-region symbol reliability map with shape [B, M, N].

    Args:
        data_mask: Optional real/bool data-region mask with shape [B, M, N] or
            [1, M, N]. If omitted, an all-ones mask is inferred from another
            spatial input tensor.
        path_confidence: Optional path confidence tensor with shape [B, K].
            Active path confidences are reduced to a batch scalar [B, 1, 1]
            and broadcast over the data region.
        path_confidence_map: Optional sparse path confidence map with shape
            [B, M, N] or [1, M, N]. It is never used as a sparse symbol map;
            nonzero entries are reduced to a batch scalar [B, 1, 1].
        residual_energy: Optional equalizer residual energy with shape
            [B, M, N] or [1, M, N]. Larger values lower reliability.
        posterior_variance: Optional token posterior variance with shape
            [B, M, N] or [1, M, N]. Larger values lower reliability.
        uncertainty_map: Optional denoiser uncertainty with shape [B, M, N] or
            [1, M, N]. Larger values lower reliability.
        eps: Positive scalar for numerical stability.

    Returns:
        Real symbol reliability map with shape [B, M, N], clamped to [0, 1].
        Channel path confidence is represented only as a batch/global scalar,
        while data_mask determines the token evidence region.
    """

    if not isinstance(eps, (int, float)) or eps <= 0:
        raise ValueError("eps must be positive.")
    batch_size, delay_bins, doppler_bins, device, dtype = _infer_reference(
        data_mask,
        path_confidence,
        path_confidence_map,
        residual_energy,
        posterior_variance,
        uncertainty_map,
    )
    data = (
        torch.ones(batch_size, delay_bins, doppler_bins, device=device, dtype=dtype)
        if data_mask is None
        else _resolve_spatial_map(data_mask, batch_size, delay_bins, doppler_bins, device, dtype, "data_mask")
    ).clamp(0.0, 1.0)

    scalar_terms: list[torch.Tensor] = []
    if path_confidence is not None:
        scalar_terms.append(_path_confidence_scalar(path_confidence, batch_size, device, dtype, eps))
    if path_confidence_map is not None:
        path_map = _resolve_spatial_map(
            path_confidence_map,
            batch_size,
            delay_bins,
            doppler_bins,
            device,
            dtype,
            "path_confidence_map",
        ).clamp(0.0, 1.0)
        scalar_terms.append(_nonzero_spatial_mean(path_map, eps))

    if scalar_terms:
        path_scalar = torch.stack(scalar_terms, dim=0).mean(dim=0).reshape(batch_size, 1, 1).clamp(0.0, 1.0)
    else:
        path_scalar = torch.ones(batch_size, 1, 1, device=device, dtype=dtype)

    reliability = data * path_scalar
    for name, value in (
        ("residual_energy", residual_energy),
        ("posterior_variance", posterior_variance),
        ("uncertainty_map", uncertainty_map),
    ):
        if value is None:
            continue
        resolved = _resolve_spatial_map(value, batch_size, delay_bins, doppler_bins, device, dtype, name)
        resolved = torch.nan_to_num(resolved, nan=0.0, posinf=1e6, neginf=0.0).clamp_min(0.0)
        reliability = reliability / (1.0 + resolved)

    reliability = torch.nan_to_num(reliability, nan=0.0, posinf=1.0, neginf=0.0)
    return (reliability * data).clamp(0.0, 1.0)


def _infer_reference(
    data_mask: torch.Tensor | None,
    path_confidence: torch.Tensor | None,
    path_confidence_map: torch.Tensor | None,
    residual_energy: torch.Tensor | None,
    posterior_variance: torch.Tensor | None,
    uncertainty_map: torch.Tensor | None,
) -> tuple[int, int, int, torch.device, torch.dtype]:
    spatial_values = [
        value
        for value in (data_mask, path_confidence_map, residual_energy, posterior_variance, uncertainty_map)
        if value is not None
    ]
    if not spatial_values:
        raise ValueError("At least one spatial [B, M, N] tensor is required when data_mask is None.")
    reference = spatial_values[0]
    if not torch.is_tensor(reference) or reference.ndim != 3 or torch.is_complex(reference):
        raise ValueError("Spatial reliability inputs must be real tensors with shape [B, M, N] or [1, M, N].")
    delay_bins, doppler_bins = reference.shape[-2:]
    device = reference.device
    dtype = reference.dtype if torch.is_floating_point(reference) else torch.float32
    batch_size = 1
    for value in spatial_values:
        if not torch.is_tensor(value) or value.ndim != 3 or torch.is_complex(value):
            raise ValueError("Spatial reliability inputs must be real tensors with shape [B, M, N] or [1, M, N].")
        if value.shape[-2:] != (delay_bins, doppler_bins):
            raise ValueError("Spatial reliability inputs must share M, N dimensions.")
        if value.shape[0] != 1:
            if batch_size not in {1, value.shape[0]}:
                raise ValueError("Spatial reliability inputs must share batch size or use batch 1.")
            batch_size = value.shape[0]
        if torch.is_floating_point(value):
            dtype = value.dtype
            device = value.device
    if path_confidence is not None:
        if not torch.is_tensor(path_confidence) or path_confidence.ndim != 2 or torch.is_complex(path_confidence):
            raise ValueError("path_confidence must have real shape [B, K].")
        if path_confidence.shape[0] != 1:
            if batch_size not in {1, path_confidence.shape[0]}:
                raise ValueError("path_confidence batch size must match spatial maps or use batch 1.")
            batch_size = path_confidence.shape[0]
        if torch.is_floating_point(path_confidence):
            dtype = path_confidence.dtype
            device = path_confidence.device
    return batch_size, delay_bins, doppler_bins, device, dtype


def _resolve_spatial_map(
    value: torch.Tensor,
    batch_size: int,
    delay_bins: int,
    doppler_bins: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> torch.Tensor:
    if not torch.is_tensor(value) or value.ndim != 3 or torch.is_complex(value):
        raise ValueError(f"{name} must have real shape [B, M, N] or [1, M, N].")
    if value.shape[-2:] != (delay_bins, doppler_bins) or value.shape[0] not in {1, batch_size}:
        raise ValueError(f"{name} must have real shape [B, M, N] or [1, M, N].")
    resolved = value.to(device=device, dtype=dtype)
    if resolved.shape[0] == 1 and batch_size != 1:
        resolved = resolved.expand(batch_size, -1, -1)
    return resolved


def _path_confidence_scalar(
    path_confidence: torch.Tensor,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    eps: float,
) -> torch.Tensor:
    if not torch.is_tensor(path_confidence) or path_confidence.ndim != 2 or torch.is_complex(path_confidence):
        raise ValueError("path_confidence must have real shape [B, K].")
    if path_confidence.shape[0] not in {1, batch_size}:
        raise ValueError("path_confidence must have real shape [B, K] or [1, K].")
    confidence = path_confidence.to(device=device, dtype=dtype).clamp(0.0, 1.0)
    if confidence.shape[0] == 1 and batch_size != 1:
        confidence = confidence.expand(batch_size, -1)
    active = confidence > eps
    count = active.sum(dim=1).clamp_min(1)
    mean = (confidence * active.to(dtype=dtype)).sum(dim=1) / count.to(dtype=dtype)
    return torch.where(active.any(dim=1), mean, torch.zeros_like(mean))


def _nonzero_spatial_mean(value: torch.Tensor, eps: float) -> torch.Tensor:
    active = value > eps
    count = active.flatten(1).sum(dim=1).clamp_min(1)
    mean = (value * active.to(dtype=value.dtype)).flatten(1).sum(dim=1) / count.to(dtype=value.dtype)
    return torch.where(active.flatten(1).any(dim=1), mean, torch.zeros_like(mean))
