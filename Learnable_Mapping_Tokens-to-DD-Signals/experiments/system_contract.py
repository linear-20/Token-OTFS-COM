"""Cross-end physical contract validation for OTFS token experiments.

This module validates configuration and artifact alignment only. It does not
train models, alter transmitter codewords, or add receiver processing blocks.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

_TX_ROOT = Path(__file__).resolve().parents[1]
_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
_RX_ROOT = _WORKSPACE_ROOT / "Learnable_Receiver_DD-Signals-to-Tokens"
for _path in (_TX_ROOT, _RX_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from transmitter import TransmitterConfig, build_transmitter_pilot_masks
from receiver.config import ReceiverConfig
from receiver.pilot_ce import (
    EmbeddedPilotConfig,
    build_embedded_pilot_masks,
    validate_pilot_ce_layout,
)


@dataclass(frozen=True)
class SystemPhysicalContractReport:
    """Auditable TX/RX/modem alignment report for one waveform profile."""

    M: int
    N: int
    vocab_size: int
    complex_dtype: str
    modem_cp_len: int
    artifact_driven: bool
    artifact_sha256: str | None
    tx_rx_masks_equal: bool
    artifact_mask_matches_tx_data_region: bool | None
    pilot_observation_within_guard: bool
    tx_guard_covers_tx_channel_and_observation: bool
    tx_max_channel_delay_bins: int
    cp_covers_tx_max_channel_delay: bool
    channel_max_delay_samples: float | None
    cp_covers_channel_config_max_delay: bool | None
    waveform_variant: str = "CP-OFDM-based OTFS-per-slot-CP"

    def to_dict(self) -> dict[str, Any]:
        return {
            "M": self.M,
            "N": self.N,
            "vocab_size": self.vocab_size,
            "complex_dtype": self.complex_dtype,
            "modem_cp_len": self.modem_cp_len,
            "artifact_driven": self.artifact_driven,
            "artifact_sha256": self.artifact_sha256,
            "tx_rx_masks_equal": self.tx_rx_masks_equal,
            "artifact_mask_matches_tx_data_region": (
                self.artifact_mask_matches_tx_data_region
            ),
            "pilot_observation_within_guard": (
                self.pilot_observation_within_guard
            ),
            "tx_guard_covers_tx_channel_and_observation": (
                self.tx_guard_covers_tx_channel_and_observation
            ),
            "tx_max_channel_delay_bins": self.tx_max_channel_delay_bins,
            "cp_covers_tx_max_channel_delay": (
                self.cp_covers_tx_max_channel_delay
            ),
            "channel_max_delay_samples": self.channel_max_delay_samples,
            "cp_covers_channel_config_max_delay": (
                self.cp_covers_channel_config_max_delay
            ),
            "waveform_variant": self.waveform_variant,
        }


def validate_system_physical_contract(
    tx_cfg: TransmitterConfig,
    rx_cfg: ReceiverConfig,
    pilot_cfg: EmbeddedPilotConfig,
    modem: object,
    *,
    artifact_payload: dict | None = None,
    channel_cfg: object | None = None,
    require_cp_sufficient: bool = False,
) -> SystemPhysicalContractReport:
    """Fail fast on TX/RX/modem/artifact mismatch and report CP conditions.

    CP sufficiency and pilot observation containment are separate conditions:

    - CP sufficiency concerns waveform delay spread in samples.
    - Pilot observation containment concerns the DD pilot observation window
      and its guard rectangle.

    ``require_cp_sufficient=False`` records insufficient-CP stress scenarios
    without rejecting them. Set it to True for production-like evaluation.
    """

    if not isinstance(tx_cfg, TransmitterConfig):
        raise TypeError("tx_cfg must be a TransmitterConfig.")
    if not isinstance(rx_cfg, ReceiverConfig):
        raise TypeError("rx_cfg must be a ReceiverConfig.")
    if not isinstance(pilot_cfg, EmbeddedPilotConfig):
        raise TypeError("pilot_cfg must be an EmbeddedPilotConfig.")
    if not isinstance(require_cp_sufficient, bool):
        raise TypeError("require_cp_sufficient must be bool.")

    _expect_equal("rx_cfg.M", rx_cfg.M, tx_cfg.M)
    _expect_equal("rx_cfg.N", rx_cfg.N, tx_cfg.N)
    _expect_equal("rx_cfg.vocab_size", rx_cfg.vocab_size, tx_cfg.vocab_size)

    _validate_modem(tx_cfg, modem)
    _validate_rx_pilot_fields(tx_cfg, rx_cfg)
    _validate_embedded_pilot_fields(tx_cfg, pilot_cfg)
    validate_pilot_ce_layout(tx_cfg.M, tx_cfg.N, pilot_cfg)

    tx_masks = build_transmitter_pilot_masks(tx_cfg)
    rx_masks = build_embedded_pilot_masks(tx_cfg.M, tx_cfg.N, pilot_cfg)
    masks_equal = (
        torch.equal(tx_masks.pilot_mask[0], rx_masks.pilot_mask)
        and torch.equal(tx_masks.guard_mask[0], rx_masks.guard_mask)
        and torch.equal(tx_masks.data_mask[0], rx_masks.data_mask)
    )
    if not masks_equal:
        raise ValueError("TX and RX pilot/guard/data masks must match exactly.")

    artifact_sha256 = None
    artifact_mask_matches = None
    if artifact_payload is not None:
        artifact_sha256 = _validate_artifact_against_runtime(
            artifact_payload,
            tx_cfg,
            tx_masks.data_mask,
        )
        artifact_mask_matches = True

    observation_within_guard = (
        pilot_cfg.obs_delay_radius <= pilot_cfg.guard_delay
        and pilot_cfg.obs_doppler_radius <= pilot_cfg.guard_doppler
    )
    tx_guard_covers = (
        tx_cfg.pilot_guard_delay
        >= tx_cfg.max_channel_delay + tx_cfg.pilot_obs_delay_radius
        and tx_cfg.pilot_guard_doppler
        >= tx_cfg.max_channel_doppler + tx_cfg.pilot_obs_doppler_radius
    )
    cp_covers_tx = modem.cp_len >= tx_cfg.max_channel_delay
    channel_max_delay = _optional_channel_max_delay(channel_cfg)
    cp_covers_channel = (
        None
        if channel_max_delay is None
        else modem.cp_len >= math.ceil(channel_max_delay)
    )
    if require_cp_sufficient and not cp_covers_tx:
        raise ValueError(
            "modem.cp_len must cover tx_cfg.max_channel_delay when "
            "require_cp_sufficient=True."
        )
    if require_cp_sufficient and cp_covers_channel is False:
        raise ValueError(
            "modem.cp_len must cover channel_cfg.max_delay_samples when "
            "require_cp_sufficient=True."
        )

    return SystemPhysicalContractReport(
        M=tx_cfg.M,
        N=tx_cfg.N,
        vocab_size=tx_cfg.vocab_size,
        complex_dtype=tx_cfg.complex_dtype,
        modem_cp_len=modem.cp_len,
        artifact_driven=artifact_payload is not None,
        artifact_sha256=artifact_sha256,
        tx_rx_masks_equal=masks_equal,
        artifact_mask_matches_tx_data_region=artifact_mask_matches,
        pilot_observation_within_guard=observation_within_guard,
        tx_guard_covers_tx_channel_and_observation=tx_guard_covers,
        tx_max_channel_delay_bins=tx_cfg.max_channel_delay,
        cp_covers_tx_max_channel_delay=cp_covers_tx,
        channel_max_delay_samples=channel_max_delay,
        cp_covers_channel_config_max_delay=cp_covers_channel,
    )


def _validate_modem(tx_cfg: TransmitterConfig, modem: object) -> None:
    if modem is None:
        raise TypeError("modem must not be None.")
    if not hasattr(modem, "dd_shape"):
        raise TypeError("modem must have a dd_shape attribute.")
    if tuple(modem.dd_shape) != (tx_cfg.M, tx_cfg.N):
        raise ValueError(
            f"modem.dd_shape {tuple(modem.dd_shape)} must match "
            f"TX DD shape {(tx_cfg.M, tx_cfg.N)}."
        )
    if not hasattr(modem, "cp_len") or not isinstance(modem.cp_len, int):
        raise TypeError("modem.cp_len must be an int.")
    if modem.cp_len < 0:
        raise ValueError("modem.cp_len must be non-negative.")
    if not hasattr(modem, "dtype"):
        raise TypeError("modem must have a dtype attribute.")
    if modem.dtype != tx_cfg.torch_complex_dtype:
        raise ValueError(
            f"modem.dtype {modem.dtype} must match TX dtype "
            f"{tx_cfg.torch_complex_dtype}."
        )


def _validate_rx_pilot_fields(
    tx_cfg: TransmitterConfig,
    rx_cfg: ReceiverConfig,
) -> None:
    fields = {
        "pilot_delay": tx_cfg.pilot_delay,
        "pilot_doppler": tx_cfg.pilot_doppler,
        "pilot_guard_delay": tx_cfg.pilot_guard_delay,
        "pilot_guard_doppler": tx_cfg.pilot_guard_doppler,
        "pilot_obs_delay_radius": tx_cfg.pilot_obs_delay_radius,
        "pilot_obs_doppler_radius": tx_cfg.pilot_obs_doppler_radius,
        "pilot_value_real": tx_cfg.pilot_value_real,
        "pilot_value_imag": tx_cfg.pilot_value_imag,
    }
    for name, expected in fields.items():
        _expect_equal(f"rx_cfg.{name}", getattr(rx_cfg, name), expected)


def _validate_embedded_pilot_fields(
    tx_cfg: TransmitterConfig,
    pilot_cfg: EmbeddedPilotConfig,
) -> None:
    fields = {
        "pilot_delay": tx_cfg.pilot_delay,
        "pilot_doppler": tx_cfg.pilot_doppler,
        "guard_delay": tx_cfg.pilot_guard_delay,
        "guard_doppler": tx_cfg.pilot_guard_doppler,
        "obs_delay_radius": tx_cfg.pilot_obs_delay_radius,
        "obs_doppler_radius": tx_cfg.pilot_obs_doppler_radius,
        "pilot_value": tx_cfg.pilot_value,
        "wrap_around": False,
    }
    for name, expected in fields.items():
        _expect_equal(f"pilot_cfg.{name}", getattr(pilot_cfg, name), expected)


def _validate_artifact_against_runtime(
    payload: dict,
    tx_cfg: TransmitterConfig,
    expected_data_mask: torch.Tensor,
) -> str:
    if not isinstance(payload, dict):
        raise TypeError("artifact_payload must be a dict.")
    if "codeword_book" not in payload or "data_mask" not in payload:
        raise ValueError("artifact_payload must contain codeword_book and data_mask.")
    if "metadata" not in payload or not isinstance(payload["metadata"], dict):
        raise ValueError("artifact_payload must contain metadata dict.")

    metadata = payload["metadata"]
    fields = {
        "M": tx_cfg.M,
        "N": tx_cfg.N,
        "vocab_size": tx_cfg.vocab_size,
        "complex_dtype": tx_cfg.complex_dtype,
        "data_power": tx_cfg.data_power,
        "pilot_delay": tx_cfg.pilot_delay,
        "pilot_doppler": tx_cfg.pilot_doppler,
        "pilot_guard_delay": tx_cfg.pilot_guard_delay,
        "pilot_guard_doppler": tx_cfg.pilot_guard_doppler,
        "pilot_obs_delay_radius": tx_cfg.pilot_obs_delay_radius,
        "pilot_obs_doppler_radius": tx_cfg.pilot_obs_doppler_radius,
        "pilot_value_real": tx_cfg.pilot_value_real,
        "pilot_value_imag": tx_cfg.pilot_value_imag,
        "max_channel_delay": tx_cfg.max_channel_delay,
        "max_channel_doppler": tx_cfg.max_channel_doppler,
    }
    for name, expected in fields.items():
        _expect_equal(f"artifact metadata {name}", metadata.get(name), expected)

    data_mask = payload["data_mask"]
    if not torch.is_tensor(data_mask):
        raise TypeError("artifact data_mask must be a torch.Tensor.")
    if tuple(data_mask.shape) != tuple(expected_data_mask.shape):
        raise ValueError(
            "Artifact-driven runtime requires one shared data_mask "
            f"with shape {list(expected_data_mask.shape)}. "
            "Per-token masks and subset masks are not accepted."
        )
    if not torch.equal(data_mask.cpu(), expected_data_mask.cpu()):
        raise ValueError(
            "Artifact data_mask must match the TX/RX data region exactly."
        )

    sha256 = metadata.get("artifact_sha256")
    if not isinstance(sha256, str):
        raise ValueError("artifact metadata artifact_sha256 must be a string.")
    return sha256


def _optional_channel_max_delay(channel_cfg: object | None) -> float | None:
    if channel_cfg is None:
        return None
    if not hasattr(channel_cfg, "max_delay_samples"):
        raise TypeError("channel_cfg must have max_delay_samples.")
    value = channel_cfg.max_delay_samples
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("channel_cfg.max_delay_samples must be a number.")
    if not math.isfinite(float(value)) or float(value) < 0:
        raise ValueError("channel_cfg.max_delay_samples must be finite and >= 0.")
    return float(value)


def _expect_equal(name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(f"{name} must be {expected!r}, got {actual!r}.")


__all__ = [
    "SystemPhysicalContractReport",
    "validate_system_physical_contract",
]
