"""Minimal LearnableTokenDDTransmitter: codebook -> mask -> pilot insertion.

This module chains TokenDDCodebook, TransmitterPilotMasks,
insert_transmitter_pilot, and optional OTFS modulation into a single
forward pass. It does NOT perform PAPR computation, shaping loss, or
bit/QAM mapping.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .codebook import TokenDDCodebook
from .config import TransmitterConfig
from .metrics import summarize_dd_transmitter_metrics
from .modem_adapter import modulate_otfs_dd_frame
from .pilot_frame import (
    TransmitterPilotMasks,
    build_transmitter_pilot_masks,
    insert_transmitter_pilot,
)


@dataclass
class TokenDDTransmitterOutput:
    """Output of LearnableTokenDDTransmitter.forward().

    Attributes:
        token_indices: Long tensor [B] of input token indices (detached clone).
        codeword_book: Complex tensor [V, M, N], physical normalized
            codeword book for receiver TokenCodewordPrior consumption.
        selected_codewords: Complex tensor [B, M, N], codeword book rows
            indexed by token_indices, before pilot insertion.
        x_dd: Complex tensor [B, M, N], DD frame after hard pilot/guard
            insertion.
        data_mask: Boolean tensor [1, M, N], data-region support mask.
        pilot_mask: Boolean tensor [1, M, N], pilot bin mask.
        guard_mask: Boolean tensor [1, M, N], guard region mask.
        metrics: Dict of scalar / tensor-valued diagnostics.
        tx_trace: Dict of boolean and string trace flags.
        time_signal: None when return_time is False; complex [B, N*(M+cp_len)]
            when return_time is True and a valid modem is provided.
    """

    token_indices: torch.Tensor
    codeword_book: torch.Tensor
    selected_codewords: torch.Tensor
    x_dd: torch.Tensor
    data_mask: torch.Tensor
    pilot_mask: torch.Tensor
    guard_mask: torch.Tensor
    metrics: dict
    tx_trace: dict
    time_signal: torch.Tensor | None = None


class LearnableTokenDDTransmitter(nn.Module):
    """Transmitter: codebook -> mask -> pilot insertion -> optional OTFS.

    Forward signature:
        input:  token_indices [B]
        kwargs: return_time (bool), modem (optional OTFS modem)
        output: TokenDDTransmitterOutput with:
            codeword_book       complex [V, M, N]
            selected_codewords  complex [B, M, N]
            x_dd                complex [B, M, N]
            data_mask / pilot_mask / guard_mask  bool [1, M, N]
            time_signal         None or complex [B, N*(M+cp_len)]

    When return_time is True and a valid modem is provided, the transmitter
    calls modulate_otfs_dd_frame(x_dd, modem) to produce the time_signal.
    The autograd graph through x_dd is preserved. This step does NOT perform
    PAPR, time-domain metrics, or shaping loss.
    """

    def __init__(
        self,
        config: TransmitterConfig,
        codebook: TokenDDCodebook | None = None,
    ) -> None:
        super().__init__()
        self.config = config

        if codebook is None:
            self.codebook = TokenDDCodebook(config)
        elif not isinstance(codebook, TokenDDCodebook):
            raise TypeError(
                f"codebook must be a TokenDDCodebook, got {type(codebook).__name__}."
            )
        else:
            _validate_compatible_codebook(codebook, config)
            self.codebook = codebook

    def forward(
        self,
        token_indices: torch.Tensor,
        return_time: bool = False,
        modem: object | None = None,
    ) -> TokenDDTransmitterOutput:
        """Run codebook lookup, mask projection, pilot insertion, and
        optional OTFS modulation.

        Args:
            token_indices: Long tensor [B] of token indices in [0, vocab_size).
            return_time: If True, perform OTFS modulation via the provided
                modem and return the time_signal. If False, time_signal is
                None and modem must also be None.
            modem: OTFS modem satisfying the modulate_otfs_dd_frame contract
                (modulate, dd_shape, cp_len). Must be None when return_time
                is False. Required when return_time is True.

        Returns:
            TokenDDTransmitterOutput with codeword_book, selected_codewords,
            x_dd, masks, metrics, tx_trace, and optionally time_signal.

        Raises:
            TypeError: If return_time is not bool, or token_indices are
                invalid.
            ValueError: If modem is provided but return_time is False, or
                return_time is True but modem is None.
        """
        if not isinstance(return_time, bool):
            raise TypeError(
                f"return_time must be bool, got {type(return_time).__name__}."
            )
        if return_time is False and modem is not None:
            raise ValueError(
                "modem must be None when return_time is False."
            )
        if return_time is True and modem is None:
            raise ValueError(
                "modem must not be None when return_time is True."
            )

        _validate_forward_token_indices(token_indices, self.config.vocab_size)
        token_indices_out = token_indices.detach().clone()

        masks = build_transmitter_pilot_masks(self.config)

        codeword_book = self.codebook.forward(data_mask=masks.data_mask)
        selected_codewords = self.codebook.selected_codewords(
            token_indices, codeword_book=codeword_book,
        )
        x_dd = insert_transmitter_pilot(selected_codewords, self.config, masks=masks)

        if return_time:
            time_signal = modulate_otfs_dd_frame(x_dd, modem)
        else:
            time_signal = None

        metrics = summarize_dd_transmitter_metrics(
            codeword_book=codeword_book,
            x_dd=x_dd,
            data_mask=masks.data_mask,
            pilot_value=self.config.pilot_value,
        )

        tx_trace = {
            "physical_codeword_book_generated": True,
            "physical_codeword_book_exported": False,
            "hard_data_mask_projection": True,
            "pilot_guard_hard_constraints": True,
            "time_modulation_enabled": return_time,
            "shaping_algorithm_enabled": False,
        }

        return TokenDDTransmitterOutput(
            token_indices=token_indices_out,
            codeword_book=codeword_book,
            selected_codewords=selected_codewords,
            x_dd=x_dd,
            data_mask=masks.data_mask,
            pilot_mask=masks.pilot_mask,
            guard_mask=masks.guard_mask,
            metrics=metrics,
            tx_trace=tx_trace,
            time_signal=time_signal,
        )


def _validate_compatible_codebook(
    codebook: TokenDDCodebook,
    config: TransmitterConfig,
) -> None:
    """Raise ValueError if codebook.config is incompatible with config."""
    cb_cfg = codebook.config
    checks = {
        "M": (cb_cfg.M, config.M),
        "N": (cb_cfg.N, config.N),
        "vocab_size": (cb_cfg.vocab_size, config.vocab_size),
        "data_power": (cb_cfg.data_power, config.data_power),
        "complex_dtype": (cb_cfg.complex_dtype, config.complex_dtype),
    }
    for name, (cb_val, cfg_val) in checks.items():
        if cb_val != cfg_val:
            raise ValueError(
                f"codebook.config.{name} ({cb_val}) must match "
                f"config.{name} ({cfg_val})."
            )


def _validate_forward_token_indices(
    token_indices: torch.Tensor,
    V: int,
) -> None:
    if not torch.is_tensor(token_indices):
        raise TypeError(
            f"token_indices must be a torch.Tensor, got {type(token_indices).__name__}."
        )
    if token_indices.dtype not in (torch.long, torch.int64):
        raise TypeError(
            f"token_indices dtype must be torch.long, got {token_indices.dtype}."
        )
    if token_indices.ndim != 1:
        raise ValueError(
            f"token_indices must be 1-D [B], got ndim={token_indices.ndim}."
        )
    if token_indices.shape[0] <= 0:
        raise ValueError("token_indices batch size must be positive.")
    if (token_indices < 0).any() or (token_indices >= V).any():
        raise ValueError(f"token_indices must be in [0, V={V}).")

