"""Detector-classifier token logit fusion (Step 16).

Fuses classifier token_logits [B,V] with unfolded detector posterior
logits [B,V] or [B,K] at the token-logit level using a learnable gate.

No bit/QAM, no Transformer/GNN/VAE, no linear classifier head.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


def expand_candidate_logits_to_vocab(
    candidate_logits: torch.Tensor,
    candidate_indices: torch.Tensor,
    vocab_size: int,
    fill_value: float = 0.0,
) -> torch.Tensor:
    """Expand top-K posterior logits [B, K] to full-vocab [B, V].

    Positions not in ``candidate_indices`` are set to ``fill_value`` (0.0
    by default, not -inf, to avoid suppressing un-candidate tokens).

    Args:
        candidate_logits: Float [B, K].
        candidate_indices: Long [B, K].
        vocab_size: V.
        fill_value: Value for positions not in candidate set.

    Returns:
        Float [B, V].
    """
    if not torch.is_tensor(candidate_logits) or candidate_logits.ndim != 2:
        raise ValueError("candidate_logits must have shape [B, K].")
    if not torch.is_tensor(candidate_indices) or candidate_indices.shape != candidate_logits.shape:
        raise ValueError("candidate_indices must have shape [B, K] matching candidate_logits.")
    if not isinstance(vocab_size, int) or vocab_size <= 0:
        raise ValueError("vocab_size must be a positive integer.")
    if candidate_indices.numel() > 0:
        if int(candidate_indices.min().item()) < 0 or int(candidate_indices.max().item()) >= vocab_size:
            raise ValueError("candidate_indices must be in [0, vocab_size).")
    B, K = candidate_logits.shape
    device = candidate_logits.device
    dtype = candidate_logits.dtype
    expanded = torch.full((B, vocab_size), float(fill_value), device=device, dtype=dtype)
    expanded.scatter_(dim=1, index=candidate_indices.to(torch.long), src=candidate_logits)
    return expanded


@dataclass
class TokenLogitFusionOutput:
    """Output of detector-classifier logit fusion.

    Attributes:
        fused_logits: Float fused token logits [B, V].
        classifier_logits: Float classifier logits [B, V] (pre-fusion).
        detector_logits: Float detector logits [B, V] (expanded if top-K).
        fusion_gate: Float gate values [B, 1].
        fusion_mode: str mode used.
    """

    fused_logits: torch.Tensor
    classifier_logits: torch.Tensor
    detector_logits: torch.Tensor | None = None
    fusion_gate: torch.Tensor | None = None
    fusion_mode: str = "disabled"


class DetectorClassifierLogitFusion(nn.Module):
    """Learnable reliability-gated fusion of classifier and detector logits.

    fused = classifier_logits + gate * detector_logits_centered

    where gate is a per-sample scalar, optionally gated by reliability
    diagnostics, initialised near zero so the block starts as identity.
    """

    def __init__(
        self,
        fusion_mode: str = "reliability",
        init_gate: float = 0.0,
        scale: float = 1.0,
        center_detector: bool = True,
    ):
        super().__init__()
        if fusion_mode not in {"disabled", "static", "reliability"}:
            raise ValueError('fusion_mode must be "disabled", "static", or "reliability".')
        self.fusion_mode = fusion_mode
        self.scale = float(scale)
        self.center_detector = bool(center_detector)

        # static gate: scalar learnable
        import math
        safe_gate = max(min(float(init_gate), 1.0 - 1e-6), 1e-6)
        raw_init = math.log(safe_gate / (1.0 - safe_gate))
        self.raw_gate = nn.Parameter(torch.tensor(raw_init, dtype=torch.float32))

        # reliability-gate params (only used in "reliability" mode)
        self.raw_a = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.raw_b = nn.Parameter(torch.tensor(-3.0, dtype=torch.float32))

    @property
    def base_gate(self) -> torch.Tensor:
        return torch.sigmoid(self.raw_gate) * self.scale

    def forward(
        self,
        classifier_logits: torch.Tensor,
        detector_logits: torch.Tensor | None = None,
        detector_candidate_indices: torch.Tensor | None = None,
        classifier_confidence: torch.Tensor | None = None,
        classifier_entropy: torch.Tensor | None = None,
        posterior_confidence: torch.Tensor | None = None,
        posterior_entropy: torch.Tensor | None = None,
        reliability_scalar: torch.Tensor | None = None,
    ) -> TokenLogitFusionOutput:
        """Fuse classifier and detector logits.

        Args:
            classifier_logits: Float [B, V].
            detector_logits: Optional float [B, V] or [B, K].
            detector_candidate_indices: Optional long [B, K].
            classifier_confidence: Optional float [B].
            classifier_entropy: Optional float [B].
            posterior_confidence: Optional float [B].
            posterior_entropy: Optional float [B].
            reliability_scalar: Optional float [B].

        Returns:
            TokenLogitFusionOutput.
        """
        B, V = classifier_logits.shape
        device = classifier_logits.device
        dtype = classifier_logits.dtype

        # handle missing detector logits
        if detector_logits is None or self.fusion_mode == "disabled":
            return TokenLogitFusionOutput(
                fused_logits=classifier_logits,
                classifier_logits=classifier_logits,
                detector_logits=None,
                fusion_gate=None,
                fusion_mode=self.fusion_mode,
            )

        # expand top-K to full vocab
        if detector_logits.shape[1] < V:
            if detector_candidate_indices is None:
                raise ValueError("detector_candidate_indices required for top-K detector_logits.")
            det_full = expand_candidate_logits_to_vocab(
                detector_logits, detector_candidate_indices, V, fill_value=0.0,
            )
        elif detector_logits.shape[1] == V:
            det_full = detector_logits.to(device=device, dtype=dtype)
        else:
            raise ValueError("detector_logits vocab dimension cannot exceed classifier vocab size.")

        det_full = torch.nan_to_num(det_full.to(device=device, dtype=dtype), nan=0.0, posinf=1e6, neginf=-1e6)

        # center detector logits
        if self.center_detector:
            det_full = det_full - det_full.mean(dim=-1, keepdim=True)

        # gate
        gate = self._compute_gate(
            B, V, device, dtype,
            classifier_confidence, classifier_entropy,
            posterior_confidence, posterior_entropy,
            reliability_scalar,
        )

        fused = classifier_logits + gate * det_full

        return TokenLogitFusionOutput(
            fused_logits=fused,
            classifier_logits=classifier_logits,
            detector_logits=det_full,
            fusion_gate=gate,
            fusion_mode=self.fusion_mode,
        )

    def _compute_gate(
        self, B, V, device, dtype,
        classifier_confidence, classifier_entropy,
        posterior_confidence, posterior_entropy,
        reliability_scalar,
    ) -> torch.Tensor:
        if self.fusion_mode == "static":
            g = self.base_gate.to(device=device, dtype=dtype)
            return g.expand(B, 1)

        # reliability mode
        score = torch.zeros(B, device=device, dtype=dtype)
        count = 0
        if posterior_confidence is not None:
            score = score + posterior_confidence.to(device=device, dtype=dtype)
            count += 1
        if posterior_entropy is not None:
            score = score - posterior_entropy.to(device=device, dtype=dtype)
            count += 1
        if reliability_scalar is not None:
            score = score + reliability_scalar.to(device=device, dtype=dtype)
            count += 1
        if classifier_confidence is not None:
            score = score - classifier_confidence.to(device=device, dtype=dtype)
            count += 1
        if classifier_entropy is not None:
            score = score + classifier_entropy.to(device=device, dtype=dtype)
            count += 1

        if count == 0:
            g = self.base_gate.to(device=device, dtype=dtype)
            return g.expand(B, 1)

        a = torch.nn.functional.softplus(self.raw_a).to(device=device, dtype=dtype)
        b = self.raw_b.to(device=device, dtype=dtype)
        raw = a * score + b
        g = torch.sigmoid(raw) * self.scale

        # clamp and guard against NaN
        return torch.nan_to_num(g, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0).unsqueeze(1)
