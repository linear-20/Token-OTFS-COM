"""Minimal OTFS token closed-loop acceptance harness.

Token IDs -> LearnableTokenDDTransmitter -> OTFSModem.modulate
-> TimeVaryingMultipathChannel -> OTFSModem.demodulate
-> LearnableOTFSReceiver -> token decision -> TER.

This harness does NOT train, modify TX algorithm, L_core, receiver
network structure, or channel math.  It runs CPU smoke tests only.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Sequence

import torch

# -- TX imports ----------------------------------------------------------------
_R = Path(__file__).resolve().parents[1]
if str(_R) not in sys.path:
    sys.path.insert(0, str(_R))

from transmitter import (
    LearnableTokenDDTransmitter,
    TransmitterConfig,
    build_transmitter_pilot_masks,
    initialize_token_codebook_,
    insert_transmitter_pilot,
    load_exported_physical_dd_config,
    load_exported_physical_dd_artifact,
    TokenDDTransmitterOutput,
)
from transmitter.modem_adapter import modulate_otfs_dd_frame

# -- RX imports (path-based, must precede channel import) --------------------
_RX = str(Path(__file__).resolve().parents[2]
          / "Learnable_Receiver_DD-Signals-to-Tokens")
if _RX not in sys.path:
    sys.path.insert(0, _RX)

# -- OTFS modem ----------------------------------------------------------------
_OTFS = _R / "otfs_modem.py"
import importlib.util as _iu
_spec = _iu.spec_from_file_location("otfs_modem", str(_OTFS))
_otfs = _iu.module_from_spec(_spec)
_spec.loader.exec_module(_otfs)
OTFSModem = _otfs.OTFSModem

# -- Channel -------------------------------------------------------------------
_ch_p = Path(__file__).resolve().parents[2]  # workspace root
if str(_ch_p) not in sys.path:
    sys.path.insert(0, str(_ch_p))
from channel_model import ChannelConfig, TimeVaryingMultipathChannel
_ch_mod = sys.modules["channel_model"]

# -- RX imports ---------------------------------------------------------------
import importlib as _il
_receiver_config = _il.import_module("receiver.config")
_receiver_model = _il.import_module("receiver.model")
_pilot_ce = _il.import_module("receiver.pilot_ce")
_token_prior = _il.import_module("receiver.token_prior")

ReceiverConfig = _receiver_config.ReceiverConfig
LearnableOTFSReceiver = _receiver_model.LearnableOTFSReceiver
EmbeddedPilotConfig = _pilot_ce.EmbeddedPilotConfig
TokenCodewordPrior = _token_prior.TokenCodewordPrior

from experiments.system_contract import validate_system_physical_contract


# ==============================================================================
# 1. Waveform profile
# ==============================================================================

@dataclass
class WaveformProfile:
    M: int
    N: int
    cp_len: int
    sample_rate: float
    doppler_bin_hz: float
    max_delay_samples: float
    cp_sufficient: bool
    realized_max_delay_samples: float | None = None
    cp_sufficient_for_config: bool | None = None
    waveform_variant: str = "CP-OFDM-based OTFS-per-slot-CP"

    def to_dict(self) -> dict:
        return {
            "M": self.M, "N": self.N, "cp_len": self.cp_len,
            "sample_rate": self.sample_rate,
            "doppler_bin_hz": self.doppler_bin_hz,
            "max_delay_samples": self.max_delay_samples,
            "cp_sufficient": self.cp_sufficient,
            "realized_max_delay_samples": self.realized_max_delay_samples,
            "cp_sufficient_for_config": self.cp_sufficient_for_config,
            "waveform_variant": self.waveform_variant,
        }


def _build_waveform_profile(
    tx_cfg: TransmitterConfig,
    modem: OTFSModem,
    channel_cfg: ChannelConfig,
    realized_delays: torch.Tensor | None = None,
) -> WaveformProfile:
    frame_duration_s = (
        modem.N * (modem.M + modem.cp_len) / channel_cfg.sample_rate
    )
    doppler_bin_hz = (
        1.0 / frame_duration_s if frame_duration_s > 0 else float("inf")
    )
    configured_sufficient = (
        modem.cp_len >= math.ceil(channel_cfg.max_delay_samples)
    )
    realized_max_delay = None
    sufficient = configured_sufficient
    if realized_delays is not None:
        realized_max_delay = float(realized_delays.max().item())
        sufficient = modem.cp_len >= math.ceil(realized_max_delay)
    return WaveformProfile(
        M=tx_cfg.M, N=tx_cfg.N, cp_len=modem.cp_len,
        sample_rate=channel_cfg.sample_rate,
        doppler_bin_hz=doppler_bin_hz,
        max_delay_samples=channel_cfg.max_delay_samples,
        cp_sufficient=sufficient,
        realized_max_delay_samples=realized_max_delay,
        cp_sufficient_for_config=configured_sufficient,
    )


# ==============================================================================
# 2. Decision functions
# ==============================================================================

def unequalized_codeword_nn_argmin(
    y_dd: torch.Tensor,
    codeword_book: torch.Tensor,
    data_mask: torch.Tensor,
) -> torch.Tensor:
    """Nearest physical codeword before equalization, for diagnostics only.

    This is a valid token baseline for an identity channel or an explicitly
    inverted channel. It is not a multipath posterior and must not be
    interpreted as a receiver output under an unknown channel.

    Returns long tensor [B] of predicted token indices.
    """
    mask_2d = data_mask.to(device=y_dd.device, dtype=y_dd.real.dtype)
    if mask_2d.ndim == 3 and mask_2d.shape[0] == 1:
        mask_2d = mask_2d[0]
    if mask_2d.ndim != 2 or tuple(mask_2d.shape) != tuple(y_dd.shape[-2:]):
        raise ValueError("data_mask must have shape [M, N] or [1, M, N].")
    diff = y_dd.unsqueeze(1) - codeword_book.unsqueeze(0)  # [B, V, M, N]
    dist = (mask_2d.unsqueeze(0).unsqueeze(0) * diff.abs().pow(2)).sum(
        dim=(-2, -1),
    )  # [B, V]
    return dist.argmin(dim=-1)  # [B]


def receiver_detector_posterior_argmax(
    receiver_output,
) -> torch.Tensor | None:
    """Return token IDs from the final unfolded detector posterior."""

    equalizer_output = receiver_output.equalizer_output
    if (
        equalizer_output is None
        or equalizer_output.token_posterior_logits is None
        or not equalizer_output.token_posterior_logits
    ):
        return None
    logits = equalizer_output.token_posterior_logits[-1]
    local_argmax = logits.argmax(dim=-1)
    candidate_lists = equalizer_output.token_posterior_indices
    if candidate_lists is None or not candidate_lists:
        return local_argmax
    return candidate_lists[-1].gather(
        dim=1,
        index=local_argmax.unsqueeze(-1),
    ).squeeze(-1)


def receiver_classifier_argmax(
    receiver_output,
) -> torch.Tensor:
    """Argmax over receiver classifier token_logits.

    Requires a trained classifier. Marked requires_rx_training=True.
    """
    return receiver_output.token_logits.argmax(dim=-1)


# ==============================================================================
# 3. Scenario runner
# ==============================================================================

@dataclass
class ScenarioResult:
    name: str
    token_ids: torch.Tensor
    predicted_unequalized_nn: torch.Tensor
    predicted_detector: torch.Tensor | None
    predicted_classifier: torch.Tensor | None
    predicted_fused: torch.Tensor | None
    ter_unequalized_nn: float
    ter_detector: float | None
    ter_classifier: float | None
    ter_fused: float | None
    waveform_vs_dd_surrogate_nmse: float
    cp_sufficient: bool
    num_paths: int
    has_fractional: bool
    has_fractional_delay: bool
    has_fractional_doppler: bool
    snr_db: float | None
    seed: int
    oracle_ce: bool
    waveform_evaluation: bool
    waveform_profile: WaveformProfile | None = None
    physical_contract: dict | None = None
    rx_sparse_active_path_count: int | None = None
    pilot_ce_path_count: int | None = None
    metadata: dict = dc_field(default_factory=dict)

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "name": self.name,
            "ter_unequalized_nn": self.ter_unequalized_nn,
            "ter_detector": self.ter_detector,
            "ter_classifier": self.ter_classifier,
            "ter_fused": self.ter_fused,
            "waveform_vs_dd_surrogate_nmse": (
                self.waveform_vs_dd_surrogate_nmse
            ),
            "cp_sufficient": self.cp_sufficient,
            "num_paths": self.num_paths,
            "has_fractional": self.has_fractional,
            "has_fractional_delay": self.has_fractional_delay,
            "has_fractional_doppler": self.has_fractional_doppler,
            "snr_db": self.snr_db,
            "seed": self.seed,
            "oracle_ce": self.oracle_ce,
            "waveform_evaluation": self.waveform_evaluation,
            "waveform_profile": (
                self.waveform_profile.to_dict()
                if self.waveform_profile is not None
                else None
            ),
            "physical_contract": self.physical_contract,
            "rx_sparse_active_path_count": self.rx_sparse_active_path_count,
            "pilot_ce_path_count": self.pilot_ce_path_count,
            "requires_rx_training_for_detector": False,
            "requires_rx_training_for_classifier": True,
            "requires_rx_training_for_fused": self.predicted_fused is not None,
        }
        d.update(self.metadata)
        return d


# ==============================================================================
# 4. Main harness
# ==============================================================================

def _surrogate_path_shifts(
    delays_samples: torch.Tensor,
    dopplers_hz: torch.Tensor,
    modem: OTFSModem,
    sample_rate: float,
) -> torch.Tensor:
    """Map waveform path metadata to nearest on-grid DD surrogate shifts."""

    frame_duration_s = modem.N * (modem.M + modem.cp_len) / sample_rate
    delay_bins = torch.round(delays_samples).long().remainder(modem.M)
    doppler_bins = torch.round(dopplers_hz * frame_duration_s).long()
    doppler_bins = doppler_bins.remainder(modem.N)
    return torch.stack([delay_bins, doppler_bins], dim=-1)


def _build_dd_surrogate(
    x_dd: torch.Tensor,
    delays_samples: torch.Tensor,
    dopplers_hz: torch.Tensor,
    path_gains: torch.Tensor,
    modem: OTFSModem,
    sample_rate: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return nearest-bin DD surrogate output and its path shifts."""

    from transmitter.sparse_multipath import (
        SparseMultipathDDChannel,
        apply_sparse_multipath_dd_operator,
    )

    path_shifts = _surrogate_path_shifts(
        delays_samples,
        dopplers_hz,
        modem,
        sample_rate,
    )
    channel = SparseMultipathDDChannel(
        path_shifts=path_shifts,
        path_gains=path_gains,
    )
    return apply_sparse_multipath_dd_operator(x_dd, channel), path_shifts


def _build_oracle_dd_surrogate(
    x_dd: torch.Tensor,
    path_shifts: torch.Tensor,
    path_gains: torch.Tensor,
) -> torch.Tensor:
    """Scatter nearest-bin path gains into an oracle DD surrogate grid."""

    batch_size, M, N = x_dd.shape
    h_dd = torch.zeros_like(x_dd)
    for batch_idx in range(batch_size):
        for path_idx in range(path_shifts.shape[1]):
            delay = int(path_shifts[batch_idx, path_idx, 0].item())
            doppler = int(path_shifts[batch_idx, path_idx, 1].item())
            h_dd[batch_idx, delay % M, doppler % N] += path_gains[
                batch_idx, path_idx
            ]
    return h_dd


def _max_active_path_count(sparse_estimate) -> int | None:
    """Return the maximum non-padded sparse path count across a batch."""

    if sparse_estimate is None or sparse_estimate.path_gains is None:
        return None
    active = sparse_estimate.path_gains.abs() > 0
    if sparse_estimate.confidence is not None:
        active = active & (sparse_estimate.confidence > 0)
    return int(active.sum(dim=-1).max().item())


def _fractional_path_flags(
    delays_samples: torch.Tensor,
    dopplers_hz: torch.Tensor,
    modem: OTFSModem,
    sample_rate: float,
    tol: float = 1e-6,
) -> tuple[bool, bool]:
    """Report actual nearest-grid mismatch for delay and Doppler paths."""

    frame_duration_s = modem.N * (modem.M + modem.cp_len) / sample_rate
    delay_coordinates = delays_samples
    doppler_coordinates = dopplers_hz * frame_duration_s
    has_fractional_delay = bool(
        (delay_coordinates - torch.round(delay_coordinates)).abs().max().item()
        > tol
    )
    has_fractional_doppler = bool(
        (doppler_coordinates - torch.round(doppler_coordinates)).abs().max().item()
        > tol
    )
    return has_fractional_delay, has_fractional_doppler


def _ter(prediction: torch.Tensor | None, token_ids: torch.Tensor) -> float | None:
    if prediction is None:
        return None
    return float((prediction != token_ids).float().mean().item())


def _artifact_tx_frame(
    token_ids: torch.Tensor,
    codeword_book: torch.Tensor,
    tx_cfg: TransmitterConfig,
    masks,
    modem: OTFSModem,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build an OTFS waveform directly from an exported physical codebook."""

    selected_codewords = codeword_book[token_ids]
    x_dd = insert_transmitter_pilot(
        selected_codewords,
        tx_cfg,
        masks=masks,
    )
    return x_dd, modulate_otfs_dd_frame(x_dd, modem)


def run_system_closeloop(
    tx_cfg: TransmitterConfig,
    modem: OTFSModem,
    rx_cfg,
    pilot_cfg,
    channel_cfgs: Sequence[ChannelConfig],
    *,
    num_tokens: int = 32,
    seed: int = 0,
    oracle_ce: bool = True,
    path_gain_overrides: Sequence[torch.Tensor | None] | None = None,
    physical_artifact_path: str | Path | None = None,
    require_cp_sufficient: bool = False,
) -> list[ScenarioResult]:
    """Run closed-loop evaluation across channel configs.

    ``path_gain_overrides`` is intended for controlled acceptance scenarios,
    such as an exact identity path with gain 1+0j. Production-like scenarios
    should leave it as None so the physical channel model draws fading gains.

    When ``physical_artifact_path`` is supplied, the strict loader validates
    the exported physical codeword book and the harness transmits those loaded
    codewords directly. No random codebook is initialized in that mode.

    Returns list of ScenarioResult.
    """
    if (
        path_gain_overrides is not None
        and len(path_gain_overrides) != len(channel_cfgs)
    ):
        raise ValueError(
            "path_gain_overrides length must match channel_cfgs length."
        )
    results: list[ScenarioResult] = []

    masks = build_transmitter_pilot_masks(tx_cfg)
    data_mask = masks.data_mask

    # Build the codeword source. Artifact-driven evaluation must not silently
    # replace the exported TX state with a fresh random initialization.
    artifact_payload = None
    tx = None
    if physical_artifact_path is not None:
        artifact_payload = load_exported_physical_dd_artifact(
            physical_artifact_path,
        )
        cw = artifact_payload["codeword_book"]
    else:
        tx = LearnableTokenDDTransmitter(tx_cfg)
        book_gen = torch.Generator(device="cpu").manual_seed(seed + 1)
        initialize_token_codebook_(
            tx.codebook,
            mode="random_phase",
            generator=book_gen,
        )
        with torch.no_grad():
            cw = tx.codebook.forward(data_mask=data_mask)

    # Build receiver.
    rx = LearnableOTFSReceiver(rx_cfg)
    rx.eval()

    # Token prior.
    token_prior = TokenCodewordPrior(codeword_book=cw)

    # Run each channel config.
    for ch_idx, ch_cfg in enumerate(channel_cfgs):
        ch_seed = seed + 100 + ch_idx
        ch_cfg = copy.deepcopy(ch_cfg)
        ch_cfg.seed = ch_seed
        ch_cfg.randomize_each_forward = False
        contract = validate_system_physical_contract(
            tx_cfg,
            rx_cfg,
            pilot_cfg,
            modem,
            artifact_payload=artifact_payload,
            channel_cfg=ch_cfg,
            require_cp_sufficient=require_cp_sufficient,
        )

        # Generate token IDs.
        token_ids = torch.randint(
            0, tx_cfg.vocab_size, (num_tokens,),
            generator=torch.Generator(device="cpu").manual_seed(ch_seed + 50),
        )

        # TX forward.
        if artifact_payload is None:
            tx_out: TokenDDTransmitterOutput = tx.forward(
                token_ids, return_time=True, modem=modem,
            )
            time_signal = tx_out.time_signal  # [B, N*(M+cp_len)]
            x_dd = tx_out.x_dd
        else:
            x_dd, time_signal = _artifact_tx_frame(
                token_ids,
                cw,
                tx_cfg,
                masks,
                modem,
            )

        # Channel.
        ch = TimeVaryingMultipathChannel(ch_cfg)
        gain_override = (
            None if path_gain_overrides is None
            else path_gain_overrides[ch_idx]
        )
        with torch.no_grad():
            ch_out = ch(
                time_signal,
                path_gains=gain_override,
                return_info=True,
            )

        # RX demod.
        y_dd = modem.demodulate(ch_out.y)
        y_dd_clean = modem.demodulate(ch_out.clean)
        wf = _build_waveform_profile(
            tx_cfg,
            modem,
            ch_cfg,
            realized_delays=ch_out.delays,
        )

        # Nearest-bin DD surrogate. Compare against the clean waveform output
        # so AWGN is not mislabeled as model mismatch.
        x_dd_surrogate, path_shifts = _build_dd_surrogate(
            x_dd,
            ch_out.delays,
            ch_out.dopplers_hz,
            ch_out.path_gains,
            modem,
            ch_cfg.sample_rate,
        )
        nmse = float(
            (y_dd_clean - x_dd_surrogate).abs().pow(2).sum().item()
            / x_dd_surrogate.abs().pow(2).sum().clamp_min(1e-12).item()
        )

        # Diagnostic only: unequalized nearest physical codeword.
        pred_nn = unequalized_codeword_nn_argmin(
            y_dd, cw, data_mask,
        )

        # Receiver decisions. In oracle mode, h_dd is a nearest-bin DD
        # surrogate of the waveform channel. It is not x_dd and must not be
        # reported as an exact physical-waveform oracle off-grid.
        with torch.no_grad():
            rx_kwargs = dict(
                y_dd=y_dd,
                channel_estimator_mode="pilot",
                pilot_config=pilot_cfg,
                token_prior=token_prior,
                return_details=True,
                snr_db=ch_cfg.snr_db,
                data_mask=masks.data_mask,
                pilot_mask=masks.pilot_mask,
                guard_mask=masks.guard_mask,
            )
            if oracle_ce:
                rx_kwargs["channel_estimator_mode"] = "oracle"
                rx_kwargs["h_dd"] = _build_oracle_dd_surrogate(
                    x_dd,
                    path_shifts,
                    ch_out.path_gains,
                )
            rx_out = rx.forward(**rx_kwargs)
        pred_detector = receiver_detector_posterior_argmax(rx_out)
        pred_cls = receiver_classifier_argmax(rx_out) \
            if rx_out.token_logits is not None else None

        # Decision 3: fused (only if fusion enabled).
        pred_fused = None
        if rx_out.logit_fusion_output is not None:
            pred_fused = rx_out.logit_fusion_output.fused_logits.argmax(-1)

        ter_nn = _ter(pred_nn, token_ids)
        ter_detector = _ter(pred_detector, token_ids)
        ter_cls = _ter(pred_cls, token_ids)
        ter_fused = _ter(pred_fused, token_ids)

        has_frac_delay, has_frac_doppler = _fractional_path_flags(
            ch_out.delays,
            ch_out.dopplers_hz,
            modem,
            ch_cfg.sample_rate,
        )
        has_frac = has_frac_delay or has_frac_doppler
        active_path_count = _max_active_path_count(rx_out.sparse_estimate)

        results.append(ScenarioResult(
            name=f"ch{ch_idx}_{'frac' if has_frac else 'int'}_"
                 f"SNR{ch_cfg.snr_db}dB",
            token_ids=token_ids,
            predicted_unequalized_nn=pred_nn,
            predicted_detector=pred_detector,
            predicted_classifier=pred_cls,
            predicted_fused=pred_fused,
            ter_unequalized_nn=ter_nn,
            ter_detector=ter_detector,
            ter_classifier=ter_cls,
            ter_fused=ter_fused,
            waveform_vs_dd_surrogate_nmse=nmse,
            cp_sufficient=wf.cp_sufficient,
            num_paths=ch_cfg.num_paths,
            has_fractional=has_frac,
            has_fractional_delay=has_frac_delay,
            has_fractional_doppler=has_frac_doppler,
            snr_db=ch_cfg.snr_db,
            seed=ch_seed,
            oracle_ce=oracle_ce,
            waveform_evaluation=True,
            waveform_profile=wf,
            physical_contract=contract.to_dict(),
            rx_sparse_active_path_count=active_path_count,
            pilot_ce_path_count=None if oracle_ce else active_path_count,
            metadata={
                "oracle_dd_surrogate": bool(oracle_ce),
                "nmse_excludes_awgn": True,
                "codeword_source": (
                    "exported_physical_artifact"
                    if artifact_payload is not None
                    else "random_initialized_smoke"
                ),
            },
        ))

    return results


# ==============================================================================
# 5. CLI smoke
# ==============================================================================

def _default_tx() -> TransmitterConfig:
    return TransmitterConfig(
        M=8, N=6, vocab_size=8,
        pilot_delay=4, pilot_doppler=3,
        pilot_guard_delay=2, pilot_guard_doppler=2,
        pilot_obs_delay_radius=0, pilot_obs_doppler_radius=0,
        max_channel_delay=2, max_channel_doppler=2,
        data_power=1.0,
        pilot_value_real=2.0, pilot_value_imag=0.0,
        complex_dtype="complex64",
    )


def _default_rx(tx_cfg: TransmitterConfig):
    return ReceiverConfig(
        M=tx_cfg.M, N=tx_cfg.N, vocab_size=tx_cfg.vocab_size,
        token_embedding_dim=32, num_unfolded_layers=2, topk_paths=4,
        hidden_channels=16, noise_var=0.01, use_refinement_net=False,
        channel_estimator_mode="pilot",
        pilot_delay=tx_cfg.pilot_delay,
        pilot_doppler=tx_cfg.pilot_doppler,
        pilot_guard_delay=tx_cfg.pilot_guard_delay,
        pilot_guard_doppler=tx_cfg.pilot_guard_doppler,
        pilot_obs_delay_radius=tx_cfg.pilot_obs_delay_radius,
        pilot_obs_doppler_radius=tx_cfg.pilot_obs_doppler_radius,
        pilot_value_real=tx_cfg.pilot_value_real,
        pilot_value_imag=tx_cfg.pilot_value_imag,
    )


def _default_pilot(tx_cfg: TransmitterConfig):
    return EmbeddedPilotConfig(
        pilot_delay=tx_cfg.pilot_delay,
        pilot_doppler=tx_cfg.pilot_doppler,
        guard_delay=tx_cfg.pilot_guard_delay,
        guard_doppler=tx_cfg.pilot_guard_doppler,
        obs_delay_radius=tx_cfg.pilot_obs_delay_radius,
        obs_doppler_radius=tx_cfg.pilot_obs_doppler_radius,
        pilot_value=tx_cfg.pilot_value,
    )


def _default_channels() -> list[ChannelConfig]:
    return [
        ChannelConfig(num_paths=1, snr_db=None, add_awgn=False,
                      seed=100, fractional_delays=False,
                      max_delay_samples=0.0, max_doppler_hz=0.0,
                      path_delays=[0.0], path_dopplers_hz=[0.0],
                      path_powers_db=[0.0]),
        ChannelConfig(num_paths=1, snr_db=30.0, add_awgn=True,
                      seed=101, fractional_delays=False,
                      max_delay_samples=2.0, max_doppler_hz=100.0,
                      path_delays=[1.0], path_dopplers_hz=[50.0],
                      path_powers_db=[0.0]),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run minimal OTFS token closed-loop acceptance scenarios.",
    )
    parser.add_argument(
        "--physical-artifact",
        type=Path,
        default=None,
        help="Optional strict physical codeword book artifact to transmit.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path for the serialized acceptance report.",
    )
    parser.add_argument("--num-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--require-cp-sufficient",
        action="store_true",
        help="Reject scenarios whose configured delay exceeds the CP.",
    )
    args = parser.parse_args()
    if args.num_tokens <= 0:
        parser.error("--num-tokens must be positive.")

    tx_cfg = (
        load_exported_physical_dd_config(args.physical_artifact)
        if args.physical_artifact is not None
        else _default_tx()
    )
    modem = OTFSModem(dd_shape=(tx_cfg.M, tx_cfg.N), cp_len=tx_cfg.max_channel_delay + 1)
    rx_cfg = _default_rx(tx_cfg)
    pilot_cfg = _default_pilot(tx_cfg)
    channels = _default_channels()

    results = run_system_closeloop(
        tx_cfg, modem, rx_cfg, pilot_cfg, channels,
        num_tokens=args.num_tokens, seed=args.seed, oracle_ce=True,
        path_gain_overrides=[
            torch.tensor([1.0 + 0.0j], dtype=torch.complex64),
            None,
        ],
        physical_artifact_path=args.physical_artifact,
        require_cp_sufficient=args.require_cp_sufficient,
    )
    report = {
        "waveform_variant": "CP-OFDM-based OTFS-per-slot-CP",
        "physical_artifact": (
            str(args.physical_artifact)
            if args.physical_artifact is not None
            else None
        ),
        "require_cp_sufficient": args.require_cp_sufficient,
        "results": [result.to_dict() for result in results],
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )
    print(json.dumps({
        "waveform_variant": report["waveform_variant"],
        "physical_artifact": report["physical_artifact"],
        "require_cp_sufficient": report["require_cp_sufficient"],
    }))
    for r in results:
        print(json.dumps(r.to_dict(), indent=2))


if __name__ == "__main__":
    main()
