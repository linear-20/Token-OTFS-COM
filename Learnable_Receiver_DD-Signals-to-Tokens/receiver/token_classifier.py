"""Multi-head evidence token embedding classifier.

The classifier consumes a complex DD tensor [B, M, N], maps it to H heads
of evidence-weighted embeddings [B, H, D], computes per-head cosine similarity
against a learnable token codebook [V, D], and fuses the H head logits into
final token_logits [B, V].

No Linear(vocab_size) head, no self-attention, no GNN, no VAE.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .complex_utils import validate_complex_dd
from .config import ReceiverConfig
from .position_encoding import build_dd_position_encoding


@dataclass
class TokenClassifierOutput:
    """Detailed multi-head token classifier output.

    Attributes:
        token_logits: Float token logits [B, vocab_size].
        rx_embedding: Float fused receiver embedding [B, D].
        topk_indices: Optional long top-k indices [B, K].
        topk_logits: Optional float top-k logits [B, K].
        evidence_weights: Optional float evidence weights [B, M, N] (head 0
            or fused for backward compat).
        local_embeddings: Optional float local embeddings [B, D, M, N].
        head_embeddings: Optional float per-head embeddings [B, H, D].
        head_logits: Optional float per-head logits [B, H, V].
        head_evidence_weights: Optional float per-head evidence weights
            [B, H, M, N].
        head_fusion_weights: Optional float fusion weights [B, H].
        head_evidence_entropy: Optional float per-head entropy [B, H].
        confidence_map: Optional float confidence map [B, M, N].
        uncertainty_map: Optional float uncertainty map [B, M, N].
    """

    token_logits: torch.Tensor
    rx_embedding: torch.Tensor
    topk_indices: torch.Tensor | None = None
    topk_logits: torch.Tensor | None = None
    evidence_weights: torch.Tensor | None = None
    local_embeddings: torch.Tensor | None = None
    head_embeddings: torch.Tensor | None = None
    head_logits: torch.Tensor | None = None
    head_evidence_weights: torch.Tensor | None = None
    head_fusion_weights: torch.Tensor | None = None
    head_evidence_entropy: torch.Tensor | None = None
    confidence_map: torch.Tensor | None = None
    uncertainty_map: torch.Tensor | None = None


class TokenEmbeddingClassifier(nn.Module):
    """Multi-head evidence token embedding classifier for x_dd [B, M, N]."""

    def __init__(self, config: ReceiverConfig, temperature_init: float = 0.07):
        super().__init__()
        if temperature_init <= 0:
            raise ValueError("temperature_init must be positive.")
        self.config = config
        H = int(config.classifier_num_evidence_heads)
        D = int(config.token_embedding_dim)
        hidden = int(config.hidden_channels)

        input_channels = 3 + config.classifier_position_channels + 3
        self.local_encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.feature_net = self.local_encoder
        self.local_embedding_head = nn.Conv2d(hidden, D, kernel_size=1)
        self.evidence_head = nn.Conv2d(hidden, H, kernel_size=1)
        self.token_codebook = nn.Parameter(torch.randn(config.vocab_size, D) * 0.02)
        self.log_temperature = nn.Parameter(torch.tensor(math.log(temperature_init), dtype=torch.float32))

        # head fusion
        if config.classifier_head_fusion == "learned_static":
            self.head_fusion_logits = nn.Parameter(torch.zeros(H, dtype=torch.float32))
        else:
            self.head_fusion_logits = None

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp().clamp_min(1e-6)

    def forward(
        self,
        x_dd: torch.Tensor,
        return_embedding: bool = False,
        topk: int | None = None,
        support_mask: torch.Tensor | None = None,
        confidence_map: torch.Tensor | None = None,
        uncertainty_map: torch.Tensor | None = None,
        return_local: bool = False,
        return_head_details: bool | None = None,
    ) -> torch.Tensor | TokenClassifierOutput:
        """Classify x_dd [B, M, N].

        Args:
            x_dd: Complex DD tensor [B, M, N].
            return_embedding: If false and topk is None, return token_logits.
            topk: Optional K for top-k logits/indices [B, K].
            support_mask: Optional real [B, M, N].
            confidence_map: Optional real [B, M, N].
            uncertainty_map: Optional real [B, M, N].
            return_local: If true, include local_embeddings [B, D, M, N].
            return_head_details: Optional override for exposing per-head
                embeddings/logits/evidence fields.

        Returns:
            token_logits [B, vocab_size] or TokenClassifierOutput.
        """
        validate_complex_dd(x_dd, self.config, "x_dd")
        if topk is not None and (not isinstance(topk, int) or topk <= 0 or topk > self.config.vocab_size):
            raise ValueError("topk must be a positive integer <= vocab_size.")

        support, confidence, uncertainty = self._maps(x_dd, support_mask, confidence_map, uncertainty_map)
        features = self.local_encoder(self._channels(x_dd, support, confidence, uncertainty))
        local_embeddings = self.local_embedding_head(features)  # [B, D, M, N]

        H = int(self.config.classifier_num_evidence_heads)
        B = x_dd.shape[0]
        D = int(self.config.token_embedding_dim)

        # per-head evidence logits [B, H, M, N]
        evidence_raw = self.evidence_head(features)  # [B, H, M, N]
        evidence_temp = float(self.config.classifier_evidence_temperature)
        conf_floor = float(self.config.classifier_confidence_floor)
        eps = float(self.config.classifier_head_diversity_eps)

        confidence_prior = confidence.clamp_min(conf_floor).unsqueeze(1)  # [B, 1, M, N]
        if self.config.classifier_use_uncertainty:
            confidence_prior = confidence_prior / (1.0 + uncertainty.unsqueeze(1).clamp_min(0.0))
        confidence_prior = confidence_prior.clamp_min(conf_floor)

        log_data_mask = torch.log(support.unsqueeze(1).clamp_min(eps))
        log_confidence = torch.log(confidence_prior)
        evidence_score = (evidence_raw + log_data_mask + log_confidence - uncertainty.unsqueeze(1)) / evidence_temp

        # Per-head softmax over DD grid.  If a sample has any valid data
        # positions, zero-mask positions are excluded exactly; if all positions
        # are masked, keep the finite score tensor as a safe fallback.
        flat_score = evidence_score.reshape(B, H, -1)
        valid = support.unsqueeze(1).expand(B, H, -1, -1).reshape(B, H, -1) > 0.0
        has_valid = valid.any(dim=-1, keepdim=True)
        flat_score = torch.where(
            valid | ~has_valid,
            flat_score,
            flat_score.new_full((), -1e9),
        )
        head_evidence_weights = torch.softmax(flat_score, dim=-1).reshape(B, H, evidence_score.shape[-2], evidence_score.shape[-1])
        # [B, H, M, N]

        # per-head entropy
        log_w = torch.log(head_evidence_weights.clamp_min(eps))
        head_entropy = -(head_evidence_weights * log_w).sum(dim=(-2, -1))  # [B, H]
        max_ent = math.log(evidence_score.shape[-2] * evidence_score.shape[-1])
        head_entropy = head_entropy / max(max_ent, 1e-6)

        # per-head weighted embedding [B, H, D]
        head_emb = (head_evidence_weights.unsqueeze(2) * local_embeddings.unsqueeze(1)).sum(dim=(-2, -1))

        # per-head cosine similarity [B, H, V]
        codebook_norm = F.normalize(self.token_codebook, dim=-1)  # [V, D]
        head_emb_norm = F.normalize(head_emb, dim=-1)             # [B, H, D]
        head_logits = torch.einsum("bhd,vd->bhv", head_emb_norm, codebook_norm) / self.temperature.to(x_dd.device)

        # head fusion
        fusion_mode = self.config.classifier_head_fusion
        if fusion_mode == "learned_static" and self.head_fusion_logits is not None:
            fusion_raw = self.head_fusion_logits  # [H]
            fusion_weights = F.softmax(fusion_raw, dim=0).unsqueeze(0).expand(B, -1)  # [B, H]
        elif fusion_mode == "confidence":
            inv_entropy = 1.0 / (head_entropy + eps)
            fusion_weights = inv_entropy / inv_entropy.sum(dim=-1, keepdim=True).clamp_min(eps)  # [B, H]
        else:
            fusion_weights = torch.full((B, H), 1.0 / H, device=x_dd.device, dtype=head_logits.dtype)

        token_logits = (fusion_weights.unsqueeze(-1) * head_logits).sum(dim=1)  # [B, V]

        # fused rx_embedding for compatibility
        rx_embedding = F.normalize((fusion_weights.unsqueeze(-1) * head_emb).sum(dim=1), dim=-1)

        # backward-compat evidence_weights (head 0 or mean)
        evidence_weights = head_evidence_weights[:, 0] if H == 1 else head_evidence_weights.mean(dim=1)

        need_output = return_embedding or topk is not None
        if not need_output:
            return token_logits

        topk_logits = None
        topk_indices = None
        if topk is not None:
            topk_logits, topk_indices = torch.topk(token_logits, k=topk, dim=-1)

        resolved_return_head_details = (
            bool(self.config.classifier_return_head_details)
            if return_head_details is None
            else bool(return_head_details)
        )
        return TokenClassifierOutput(
            token_logits=token_logits,
            rx_embedding=rx_embedding,
            topk_indices=topk_indices,
            topk_logits=topk_logits,
            evidence_weights=evidence_weights,
            local_embeddings=local_embeddings if return_local else None,
            head_embeddings=head_emb if resolved_return_head_details else None,
            head_logits=head_logits if resolved_return_head_details else None,
            head_evidence_weights=head_evidence_weights if resolved_return_head_details else None,
            head_fusion_weights=fusion_weights if resolved_return_head_details else None,
            head_evidence_entropy=head_entropy if resolved_return_head_details else None,
            confidence_map=confidence,
            uncertainty_map=uncertainty,
        )

    def _channels(self, x_dd, support, confidence, uncertainty):
        dtype = self.token_codebook.dtype
        batch_size = x_dd.shape[0]
        position = build_dd_position_encoding(
            self.config.M, self.config.N,
            self.config.classifier_position_channels,
            device=x_dd.device, dtype=dtype,
        ).expand(batch_size, -1, -1, -1)
        dd_channels = torch.stack(
            (x_dd.real.to(dtype=dtype), x_dd.imag.to(dtype=dtype), x_dd.abs().to(dtype=dtype)), dim=1,
        )
        map_channels = torch.stack(
            (support.to(device=x_dd.device, dtype=dtype),
             confidence.to(device=x_dd.device, dtype=dtype),
             uncertainty.to(device=x_dd.device, dtype=dtype)), dim=1,
        )
        return torch.cat((dd_channels, position, map_channels), dim=1)

    def _maps(self, x_dd, support_mask, confidence_map, uncertainty_map):
        support = _optional_real_map(support_mask, x_dd, "support_mask")
        if support is None:
            support = torch.ones_like(x_dd.real)
        else:
            support = support.clamp(0.0, 1.0)
        confidence = _optional_real_map(confidence_map, x_dd, "confidence_map")
        if confidence is None:
            confidence = support.clone()
        confidence = confidence.clamp(0.0, 1.0)
        uncertainty = _optional_real_map(uncertainty_map, x_dd, "uncertainty_map")
        if uncertainty is None:
            uncertainty = torch.zeros_like(x_dd.real)
        uncertainty = uncertainty.clamp_min(0.0)
        return support, confidence, uncertainty


class EvidenceHeadDiversityLoss(nn.Module):
    """Optional regularizer preventing evidence head collapse."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(
        self,
        head_evidence_weights: torch.Tensor,
        data_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Penalize pairwise overlap between head evidence distributions.

        Args:
            head_evidence_weights: Float [B, H, M, N].
            data_mask: Optional real [B, M, N] or [1, M, N].

        Returns:
            Scalar loss.  Returns 0 when H == 1.
        """
        B, H, M, N = head_evidence_weights.shape
        if H <= 1:
            return head_evidence_weights.sum() * 0.0

        flat = head_evidence_weights.reshape(B, H, -1)  # [B, H, MN]

        if data_mask is not None:
            if not torch.is_tensor(data_mask) or data_mask.ndim != 3 or torch.is_complex(data_mask):
                raise ValueError("data_mask must have real shape [B, M, N] or [1, M, N].")
            if data_mask.shape[-2:] != (M, N) or data_mask.shape[0] not in {1, B}:
                raise ValueError("data_mask must have real shape [B, M, N] or [1, M, N].")
            dm = data_mask.to(device=flat.device, dtype=flat.dtype).clamp(0.0, 1.0)
            if dm.shape[0] == 1 and B != 1:
                dm = dm.expand(B, -1, -1)
            dm = dm.reshape(B, 1, -1)
            flat = flat * dm

        # normalize each head distribution
        denom = flat.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        flat_norm = flat / denom

        # pairwise cosine overlap
        overlap = torch.einsum("bim,bjm->bij", flat_norm, flat_norm)  # [B, H, H]
        # off-diagonal mean
        mask = 1.0 - torch.eye(H, device=overlap.device, dtype=overlap.dtype).unsqueeze(0)
        off_diag = (overlap * mask).sum(dim=(-2, -1)) / max(H * (H - 1), 1)
        return off_diag.mean()


def evidence_head_diversity_loss(
    head_evidence_weights: torch.Tensor,
    data_mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Functional interface to EvidenceHeadDiversityLoss."""
    module = EvidenceHeadDiversityLoss(eps=eps)
    return module(head_evidence_weights, data_mask=data_mask)


def _optional_real_map(value, x_dd, name):
    if value is None:
        return None
    if not torch.is_tensor(value) or value.shape != x_dd.shape or torch.is_complex(value):
        raise ValueError(f"{name} must have real shape [B, M, N].")
    return value.to(device=x_dd.device, dtype=x_dd.real.dtype)
