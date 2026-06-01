"""Tier 1 controlled waveform validation for frozen TX artifacts.

This entry compares baseline and shaped physical codeword books from one
completed TX-only run under paired CP-OFDM-based OTFS waveform channels.
It does not train or modify the transmitter or receiver.

Tier 1 intentionally stays inside a controlled regime:
- sufficient per-slot cyclic prefix;
- causal integer delay taps;
- integer Doppler-bin taps;
- distinct effective delay-Doppler taps per scenario;
- identical channel draws, token IDs, and AWGN for baseline and shaped books.

The waveform detector is conditional-CSI Euclidean ML over noiseless waveform
templates. It is an auditable controlled upper bound, not an engineering
receiver result. The sparse-DD template detector is reported separately to
measure surrogate mismatch.

Run from the workspace root:

    python -m experiments.tx_waveform_tier1 \
        --run-dir artifacts/tx_studies/.../gamma2_mean_seed2027 \
        --config-json configs/tx_waveform_tier1.json \
        --output-dir artifacts/waveform_tier1/gamma2_mean_seed2027
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

import torch

_TX_ROOT = Path(__file__).resolve().parents[1]
_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
for _path in (_TX_ROOT, _WORKSPACE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from channel_model import ChannelConfig, TimeVaryingMultipathChannel
from otfs_modem import OTFSModem, normalized_mse
from transmitter import (
    SparseMultipathDDChannel,
    apply_sparse_multipath_dd_operator,
    build_transmitter_pilot_masks,
    insert_transmitter_pilot,
    load_exported_physical_dd_artifact,
    load_exported_physical_dd_config,
    sample_normalized_sparse_multipath_scenario_bank,
    sample_uniform_cross_token_pairs,
)

from .tx_cloud_train import build_causal_integer_shift_set


TIER1_SCHEMA_VERSION = 1
_ARTIFACT_VARIANTS = ("baseline", "shaped")


@dataclass(frozen=True)
class Tier1WaveformConfig:
    """Controls for paired on-grid waveform validation."""

    seed: int = 41001
    cp_len: int = 4
    sample_rate: float = 15.36e6
    num_scenarios: int = 16
    num_tokens_per_scenario: int = 256
    num_separation_pairs: int = 1024
    snr_db_values: tuple[float, ...] = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0)
    num_paths: int = 3
    max_delay: int = 3
    max_doppler: int = 4
    device: str = "cpu"

    def __post_init__(self) -> None:
        _validate_non_negative_int("seed", self.seed)
        _validate_non_negative_int("cp_len", self.cp_len)
        _validate_finite_positive("sample_rate", self.sample_rate)
        for name in (
            "num_scenarios",
            "num_tokens_per_scenario",
            "num_separation_pairs",
            "num_paths",
        ):
            _validate_positive_int(name, getattr(self, name))
        _validate_non_negative_int("max_delay", self.max_delay)
        _validate_non_negative_int("max_doppler", self.max_doppler)
        if not isinstance(self.snr_db_values, tuple) or not self.snr_db_values:
            raise TypeError("snr_db_values must be a non-empty tuple.")
        for value in self.snr_db_values:
            if (isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)):
                raise ValueError(
                    f"snr_db_values must contain finite numbers, got {value}."
                )
        if not isinstance(self.device, str):
            raise TypeError("device must be a string.")
        try:
            torch.device(self.device)
        except Exception as exc:
            raise ValueError(
                f"device must be a valid torch device string, got {self.device!r}."
            ) from exc


_CONFIG_KEYS = {item.name for item in fields(Tier1WaveformConfig)}


def load_tier1_waveform_config(path: str | Path) -> Tier1WaveformConfig:
    """Load and validate one Tier 1 waveform JSON config."""

    source = Path(path)
    with open(source, encoding="ascii") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Tier 1 config JSON must contain an object.")
    if payload.get("schema_version") != TIER1_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {TIER1_SCHEMA_VERSION}."
        )
    values = {key: value for key, value in payload.items()
              if key != "schema_version"}
    unknown = set(values) - _CONFIG_KEYS
    if unknown:
        raise ValueError(
            f"Unknown Tier 1 config fields: {sorted(unknown)}."
        )
    if "snr_db_values" in values:
        raw_snr = values["snr_db_values"]
        if not isinstance(raw_snr, list):
            raise ValueError("snr_db_values must be a JSON array.")
        values["snr_db_values"] = tuple(raw_snr)
    return Tier1WaveformConfig(**values)


def oracle_ml_template_argmin(
    y_dd: torch.Tensor,
    template_book_dd: torch.Tensor,
) -> torch.Tensor:
    """Return conditional-CSI Euclidean ML token IDs for DD observations.

    The full DD grid is used. The embedded pilot is common to every token
    template and therefore cancels in pairwise template differences.
    """

    if not torch.is_tensor(y_dd) or not torch.is_tensor(template_book_dd):
        raise TypeError("y_dd and template_book_dd must be tensors.")
    if not torch.is_complex(y_dd) or not torch.is_complex(template_book_dd):
        raise TypeError("y_dd and template_book_dd must be complex tensors.")
    if y_dd.ndim != 3 or template_book_dd.ndim != 3:
        raise ValueError(
            "y_dd and template_book_dd must have shapes [B, M, N] and [V, M, N]."
        )
    if tuple(y_dd.shape[1:]) != tuple(template_book_dd.shape[1:]):
        raise ValueError("Observation and template DD shapes must match.")
    if y_dd.device != template_book_dd.device:
        raise ValueError("Observation and template devices must match.")
    if y_dd.dtype != template_book_dd.dtype:
        raise TypeError("Observation and template dtypes must match.")
    if not torch.isfinite(y_dd).all() or not torch.isfinite(template_book_dd).all():
        raise ValueError("Observations and templates must be finite.")

    observations = y_dd.reshape(y_dd.shape[0], -1)
    templates = template_book_dd.reshape(template_book_dd.shape[0], -1)
    observation_power = observations.abs().pow(2).sum(dim=1, keepdim=True)
    template_power = templates.abs().pow(2).sum(dim=1).unsqueeze(0)
    inner = observations.conj() @ templates.transpose(0, 1)
    distance = observation_power + template_power - 2.0 * inner.real
    return distance.argmin(dim=1)


def run_tx_waveform_tier1(
    run_dir: str | Path,
    output_dir: str | Path,
    *,
    config: Tier1WaveformConfig,
) -> dict:
    """Run paired Tier 1 waveform validation for one completed TX-only run."""

    if not isinstance(config, Tier1WaveformConfig):
        raise TypeError("config must be a Tier1WaveformConfig.")
    device = _validate_runtime_device(config.device)
    source_dir = Path(run_dir)
    output = Path(output_dir)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        raise ValueError(
            f"Tier 1 output already exists: {manifest_path}. "
            "Use a new output directory."
        )
    training_manifest = _load_training_manifest(source_dir)
    artifact_paths = _artifact_paths(source_dir, training_manifest)
    payloads, tx_config = _load_artifacts(artifact_paths)
    _validate_tier1_against_tx_contract(config, tx_config)

    output.mkdir(parents=True, exist_ok=True)
    masks = build_transmitter_pilot_masks(tx_config)
    modem = OTFSModem(
        dd_shape=(tx_config.M, tx_config.N),
        cp_len=config.cp_len,
        device=str(device),
        dtype=tx_config.torch_complex_dtype,
    )
    doppler_bin_hz = (
        config.sample_rate
        / (tx_config.N * (tx_config.M + config.cp_len))
    )
    target_margin = float(
        training_manifest.get("run_config", {}).get("target_margin", 0.0)
    )

    generators, derived_seeds = _build_generators(config.seed)
    support = build_causal_integer_shift_set(
        max_delay=config.max_delay,
        max_doppler=config.max_doppler,
        name="tier1_causal_integer_dd_support",
    )
    scenario_bank = sample_normalized_sparse_multipath_scenario_bank(
        support,
        num_scenarios=config.num_scenarios,
        num_paths=config.num_paths,
        generator=generators["scenarios"],
        complex_dtype=tx_config.torch_complex_dtype,
        name="tier1_paired_waveform_scenarios",
        unique_shifts_per_scenario=True,
    )
    duplicate_scenarios = _count_duplicate_dd_tap_scenarios(
        scenario_bank.channel.path_shifts,
    )
    if duplicate_scenarios != 0:
        raise ValueError("Tier 1 scenarios must contain distinct DD taps.")

    codeword_books = {
        name: payloads[name]["codeword_book"].to(device=device)
        for name in _ARTIFACT_VARIANTS
    }
    x_dd_books = {
        name: insert_transmitter_pilot(
            codeword_books[name], tx_config, masks=masks,
        )
        for name in _ARTIFACT_VARIANTS
    }
    x_time_books = {
        name: modem.modulate(x_dd_books[name])
        for name in _ARTIFACT_VARIANTS
    }
    transmit_powers = {
        name: float(x_time_books[name].abs().pow(2).mean().item())
        for name in _ARTIFACT_VARIANTS
    }
    if not math.isclose(
        transmit_powers["baseline"], transmit_powers["shaped"],
        rel_tol=1e-5, abs_tol=1e-7,
    ):
        raise ValueError(
            "Baseline and shaped waveform books must have equal average "
            "transmit power for paired Tier 1 validation."
        )
    noise_reference_power = 0.5 * (
        transmit_powers["baseline"] + transmit_powers["shaped"]
    )

    token_ids = torch.randint(
        0, tx_config.vocab_size,
        (config.num_scenarios, config.num_tokens_per_scenario),
        generator=generators["tokens"],
        dtype=torch.long,
    )
    token_pairs = sample_uniform_cross_token_pairs(
        tx_config.vocab_size,
        config.num_separation_pairs,
        generator=generators["pairs"],
        device=device,
    )
    scenario_payload = _scenario_bank_payload(
        scenario_bank,
        doppler_bin_hz=doppler_bin_hz,
    )
    torch.save(scenario_payload, str(output / "scenario_bank.pt"))

    records: list[dict[str, Any]] = []
    metrics_path = output / "metrics.jsonl"
    start_time = time.perf_counter()
    with torch.no_grad(), open(metrics_path, "w", encoding="ascii") as handle:
        for scenario_index in range(config.num_scenarios):
            shifts_cpu = scenario_bank.channel.path_shifts[scenario_index]
            gains_cpu = scenario_bank.channel.path_gains[scenario_index]
            delays = shifts_cpu[:, 0].to(dtype=tx_config.torch_real_dtype)
            dopplers_hz = (
                shifts_cpu[:, 1].to(dtype=tx_config.torch_real_dtype)
                * doppler_bin_hz
            )
            clean = {
                name: _scenario_templates(
                    x_dd_books[name],
                    x_time_books[name],
                    shifts_cpu,
                    gains_cpu,
                    delays,
                    dopplers_hz,
                    modem=modem,
                    sample_rate=config.sample_rate,
                )
                for name in _ARTIFACT_VARIANTS
            }
            common = {
                name: _scenario_static_metrics(
                    clean[name],
                    token_pairs,
                    masks.data_mask,
                    target_margin=target_margin,
                )
                for name in _ARTIFACT_VARIANTS
            }
            ids = token_ids[scenario_index].to(device=device)
            for snr_db in config.snr_db_values:
                noise = _paired_awgn(
                    batch_size=config.num_tokens_per_scenario,
                    num_samples=x_time_books["baseline"].shape[1],
                    reference_power=noise_reference_power,
                    snr_db=float(snr_db),
                    dtype=tx_config.torch_complex_dtype,
                    generator=generators["noise"],
                    device=device,
                )
                artifact_metrics = {}
                for name in _ARTIFACT_VARIANTS:
                    received_time = clean[name]["waveform_time"][ids] + noise
                    received_dd = modem.demodulate(received_time)
                    waveform_prediction = oracle_ml_template_argmin(
                        received_dd,
                        clean[name]["waveform_dd"],
                    )
                    surrogate_prediction = oracle_ml_template_argmin(
                        received_dd,
                        clean[name]["surrogate_dd"],
                    )
                    artifact_metrics[name] = {
                        **common[name],
                        "waveform_oracle_ml_ter": _ter(
                            waveform_prediction, ids,
                        ),
                        "surrogate_oracle_ml_ter": _ter(
                            surrogate_prediction, ids,
                        ),
                    }
                record = {
                    "event": "tier1_waveform_evaluation",
                    "scenario_index": scenario_index,
                    "snr_db": float(snr_db),
                    "path_shifts": shifts_cpu.tolist(),
                    "path_gains_real": gains_cpu.real.tolist(),
                    "path_gains_imag": gains_cpu.imag.tolist(),
                    "artifacts": artifact_metrics,
                    "paired_delta_shaped_minus_baseline": {
                        key: (
                            artifact_metrics["shaped"][key]
                            - artifact_metrics["baseline"][key]
                        )
                        for key in (
                            "waveform_oracle_ml_ter",
                            "surrogate_oracle_ml_ter",
                            "waveform_vs_sparse_dd_nmse",
                            "waveform_separation_mean",
                            "waveform_separation_p01",
                        )
                    },
                }
                records.append(record)
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            elapsed = time.perf_counter() - start_time
            print(json.dumps({
                "event": "tier1_progress",
                "completed_scenarios": scenario_index + 1,
                "target_scenarios": config.num_scenarios,
                "elapsed_seconds": elapsed,
                "estimated_remaining_seconds": (
                    elapsed / (scenario_index + 1)
                    * (config.num_scenarios - scenario_index - 1)
                ),
            }, sort_keys=True))

    aggregate = _aggregate_records(records, config.snr_db_values)
    manifest = {
        "schema_version": TIER1_SCHEMA_VERSION,
        "source_tx_run_dir": str(source_dir),
        "output_dir": str(output),
        "config": _config_to_dict(config),
        "tx_config": asdict(tx_config),
        "waveform_profile": {
            "variant": "CP-OFDM-based OTFS-per-slot-CP",
            "M": tx_config.M,
            "N": tx_config.N,
            "cp_len": config.cp_len,
            "sample_rate": config.sample_rate,
            "doppler_bin_hz": doppler_bin_hz,
            "max_delay_samples": config.max_delay,
            "cp_sufficient": config.cp_len >= config.max_delay,
        },
        "training_target_margin": target_margin,
        "derived_seeds": derived_seeds,
        "runtime_environment": _runtime_environment(device),
        "artifacts": {
            name: {
                "path": str(artifact_paths[name]),
                "artifact_sha256": payloads[name]["metadata"]["artifact_sha256"],
                "average_transmit_waveform_power": transmit_powers[name],
            }
            for name in _ARTIFACT_VARIANTS
        },
        "noise_convention": {
            "description": (
                "Paired complex AWGN uses one fixed transmit-waveform-power "
                "reference shared by baseline and shaped artifacts."
            ),
            "reference_power": noise_reference_power,
            "identical_noise_for_baseline_and_shaped": True,
        },
        "scenario_audit": {
            "num_scenarios": config.num_scenarios,
            "num_paths": config.num_paths,
            "scenarios_with_duplicate_dd_taps": duplicate_scenarios,
            "unit_active_path_power_normalization": True,
            "scenario_bank_file": "scenario_bank.pt",
        },
        "aggregate_metrics": aggregate,
        "artifact_files": {
            "scenario_bank": "scenario_bank.pt",
            "metrics_jsonl": "metrics.jsonl",
            "manifest": "manifest.json",
        },
        "flags": {
            "waveform_evaluation_performed": True,
            "controlled_tier1_only": True,
            "paired_baseline_shaped_comparison": True,
            "causal_integer_delay_support": True,
            "integer_doppler_bin_support": True,
            "unique_dd_taps_per_scenario": True,
            "cp_sufficient": config.cp_len >= config.max_delay,
            "fractional_delay_doppler_evaluated": False,
            "receiver_network_used": False,
            "receiver_training_performed": False,
            "transmitter_training_performed": False,
            "transmitter_architecture_changed": False,
            "waveform_oracle_ml_requires_known_channel": True,
            "engineering_receiver_ter_evaluated": False,
        },
        "claim_boundary": (
            "Tier 1 controlled waveform validation only. "
            "waveform_oracle_ml_ter uses exact per-scenario waveform templates "
            "and known channel draws. It is not trained-RX TER, BER, an "
            "off-grid robustness result, or an engineering link guarantee."
        ),
    }
    _write_json(manifest_path, manifest)
    print(json.dumps({
        "event": "tier1_complete",
        "output_dir": str(output),
        "num_records": len(records),
        "aggregate_metrics": aggregate,
    }, sort_keys=True))
    return manifest


def _scenario_templates(
    x_dd_book: torch.Tensor,
    x_time_book: torch.Tensor,
    shifts_cpu: torch.Tensor,
    gains_cpu: torch.Tensor,
    delays: torch.Tensor,
    dopplers_hz: torch.Tensor,
    *,
    modem: OTFSModem,
    sample_rate: float,
) -> dict[str, torch.Tensor]:
    """Build exact waveform and sparse-DD template books for one scenario."""

    device = x_dd_book.device
    vocab_size = x_dd_book.shape[0]
    dtype = x_dd_book.dtype
    channel = TimeVaryingMultipathChannel(ChannelConfig(
        num_paths=shifts_cpu.shape[0],
        sample_rate=sample_rate,
        snr_db=None,
        max_delay_samples=float(delays.max().item()),
        max_doppler_hz=float(dopplers_hz.abs().max().item()),
        fading="fixed",
        add_awgn=False,
        randomize_each_forward=False,
        fractional_delays=False,
        complex_dtype=(
            "complex128" if dtype == torch.complex128 else "complex64"
        ),
        seed=0,
    ))
    waveform_time = channel(
        x_time_book,
        path_gains=gains_cpu,
        path_delays=delays,
        path_dopplers_hz=dopplers_hz,
    )
    waveform_dd = modem.demodulate(waveform_time)
    sparse_channel = SparseMultipathDDChannel(
        path_shifts=shifts_cpu.unsqueeze(0).expand(
            vocab_size, -1, -1,
        ).to(device=device),
        path_gains=gains_cpu.unsqueeze(0).expand(
            vocab_size, -1,
        ).to(device=device, dtype=dtype),
    )
    surrogate_dd = apply_sparse_multipath_dd_operator(
        x_dd_book, sparse_channel,
    )
    return {
        "waveform_time": waveform_time,
        "waveform_dd": waveform_dd,
        "surrogate_dd": surrogate_dd,
    }


def _scenario_static_metrics(
    templates: dict[str, torch.Tensor],
    token_pairs: torch.Tensor,
    evidence_mask: torch.Tensor,
    *,
    target_margin: float,
) -> dict[str, float]:
    waveform_dd = templates["waveform_dd"]
    surrogate_dd = templates["surrogate_dd"]
    mask = evidence_mask[0].to(
        device=waveform_dd.device, dtype=waveform_dd.real.dtype,
    )
    active_count = float(mask.sum().item())
    waveform_scores = _template_separation_scores(
        waveform_dd, token_pairs, mask, active_count,
    )
    surrogate_scores = _template_separation_scores(
        surrogate_dd, token_pairs, mask, active_count,
    )
    return {
        "waveform_vs_sparse_dd_nmse": float(
            normalized_mse(surrogate_dd, waveform_dd)
        ),
        **{
            f"waveform_{key}": value
            for key, value in _summarize_scores(
                waveform_scores, target_margin=target_margin,
            ).items()
        },
        **{
            f"surrogate_{key}": value
            for key, value in _summarize_scores(
                surrogate_scores, target_margin=target_margin,
            ).items()
        },
    }


def _template_separation_scores(
    templates: torch.Tensor,
    token_pairs: torch.Tensor,
    mask: torch.Tensor,
    active_count: float,
) -> torch.Tensor:
    delta = templates[token_pairs[:, 0]] - templates[token_pairs[:, 1]]
    return (delta.abs().pow(2) * mask).sum(dim=(-2, -1)) / active_count


def _summarize_scores(
    scores: torch.Tensor,
    *,
    target_margin: float,
) -> dict[str, float]:
    return {
        "separation_mean": float(scores.mean().item()),
        "separation_min": float(scores.min().item()),
        "separation_p001": float(torch.quantile(scores, 0.001).item()),
        "separation_p005": float(torch.quantile(scores, 0.005).item()),
        "separation_p01": float(torch.quantile(scores, 0.01).item()),
        "separation_p05": float(torch.quantile(scores, 0.05).item()),
        "outage_probability": float((scores < target_margin).float().mean().item()),
    }


def _paired_awgn(
    *,
    batch_size: int,
    num_samples: int,
    reference_power: float,
    snr_db: float,
    dtype: torch.dtype,
    generator: torch.Generator,
    device: torch.device,
) -> torch.Tensor:
    real_dtype = torch.float64 if dtype == torch.complex128 else torch.float32
    noise_power = reference_power / (10.0 ** (snr_db / 10.0))
    scale = math.sqrt(noise_power / 2.0)
    real = torch.randn(
        batch_size, num_samples, generator=generator, dtype=real_dtype,
    )
    imag = torch.randn(
        batch_size, num_samples, generator=generator, dtype=real_dtype,
    )
    return torch.complex(real * scale, imag * scale).to(
        device=device, dtype=dtype,
    )


def _aggregate_records(
    records: list[dict[str, Any]],
    snr_values: tuple[float, ...],
) -> list[dict[str, Any]]:
    aggregate = []
    metric_names = (
        "waveform_oracle_ml_ter",
        "surrogate_oracle_ml_ter",
        "waveform_vs_sparse_dd_nmse",
        "waveform_separation_mean",
        "waveform_separation_p01",
        "waveform_outage_probability",
        "surrogate_separation_mean",
        "surrogate_separation_p01",
        "surrogate_outage_probability",
    )
    for snr_db in snr_values:
        selected = [
            record for record in records
            if record["snr_db"] == float(snr_db)
        ]
        row: dict[str, Any] = {
            "snr_db": float(snr_db),
            "num_scenarios": len(selected),
            "artifacts": {},
        }
        for name in _ARTIFACT_VARIANTS:
            row["artifacts"][name] = {
                metric: float(sum(
                    record["artifacts"][name][metric]
                    for record in selected
                ) / len(selected))
                for metric in metric_names
            }
        row["paired_delta_shaped_minus_baseline"] = {
            metric: (
                row["artifacts"]["shaped"][metric]
                - row["artifacts"]["baseline"][metric]
            )
            for metric in metric_names
        }
        aggregate.append(row)
    return aggregate


def _load_training_manifest(run_dir: Path) -> dict:
    path = run_dir / "manifest.json"
    with open(path, encoding="ascii") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("TX training manifest must contain an object.")
    if payload.get("complete") is not True:
        raise ValueError("Tier 1 validation requires a completed TX-only run.")
    return payload


def _artifact_paths(run_dir: Path, manifest: dict) -> dict[str, Path]:
    artifact_files = manifest.get("artifact_files")
    if not isinstance(artifact_files, dict):
        raise ValueError("TX training manifest missing artifact_files.")
    return {
        "baseline": run_dir / _required_artifact_file(
            artifact_files, "baseline_physical_codeword_book",
        ),
        "shaped": run_dir / _required_artifact_file(
            artifact_files, "shaped_physical_codeword_book",
        ),
    }


def _required_artifact_file(files: dict, key: str) -> str:
    value = files.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"artifact_files missing {key!r}.")
    return value


def _load_artifacts(
    artifact_paths: dict[str, Path],
) -> tuple[dict[str, dict], Any]:
    payloads = {
        name: load_exported_physical_dd_artifact(path)
        for name, path in artifact_paths.items()
    }
    configs = {
        name: load_exported_physical_dd_config(path)
        for name, path in artifact_paths.items()
    }
    if configs["baseline"] != configs["shaped"]:
        raise ValueError("Baseline and shaped TX artifact configs must match.")
    baseline_mask = payloads["baseline"]["data_mask"]
    shaped_mask = payloads["shaped"]["data_mask"]
    if not torch.equal(baseline_mask, shaped_mask):
        raise ValueError("Baseline and shaped artifact data masks must match.")
    expected_mask = build_transmitter_pilot_masks(configs["baseline"]).data_mask
    if not torch.equal(baseline_mask, expected_mask):
        raise ValueError(
            "Tier 1 requires the standard full TX data mask in both artifacts."
        )
    return payloads, configs["baseline"]


def _validate_tier1_against_tx_contract(
    config: Tier1WaveformConfig,
    tx_config: Any,
) -> None:
    if config.cp_len > tx_config.M:
        raise ValueError("cp_len must be <= TX M.")
    if config.cp_len < config.max_delay:
        raise ValueError(
            "Tier 1 requires cp_len >= max_delay for sufficient CP."
        )
    if config.max_delay > tx_config.max_channel_delay:
        raise ValueError("Tier 1 max_delay exceeds TX contract.")
    if config.max_doppler > tx_config.max_channel_doppler:
        raise ValueError("Tier 1 max_doppler exceeds TX contract.")
    support_size = (config.max_delay + 1) * (2 * config.max_doppler + 1)
    if config.num_paths > support_size:
        raise ValueError(
            "num_paths exceeds distinct effective DD support size."
        )


def _scenario_bank_payload(
    scenario_bank: Any,
    *,
    doppler_bin_hz: float,
) -> dict:
    channel = scenario_bank.channel
    return {
        "schema_version": TIER1_SCHEMA_VERSION,
        "path_shifts": channel.path_shifts.detach().cpu(),
        "path_gains": channel.path_gains.detach().cpu(),
        "path_active_mask": (
            channel.path_active_mask.detach().cpu()
            if channel.path_active_mask is not None else None
        ),
        "source_shift_indices": (
            scenario_bank.source_shift_indices.detach().cpu()
            if scenario_bank.source_shift_indices is not None else None
        ),
        "scenario_weights": scenario_bank.scenario_weights,
        "name": scenario_bank.name,
        "doppler_bin_hz": doppler_bin_hz,
    }


def _count_duplicate_dd_tap_scenarios(path_shifts: torch.Tensor) -> int:
    count = 0
    for shifts in path_shifts:
        if torch.unique(shifts, dim=0).shape[0] != shifts.shape[0]:
            count += 1
    return count


def _ter(prediction: torch.Tensor, token_ids: torch.Tensor) -> float:
    return float((prediction != token_ids).float().mean().item())


def _build_generators(seed: int) -> tuple[dict[str, torch.Generator], dict[str, int]]:
    names = ("scenarios", "tokens", "pairs", "noise")
    derived = {
        name: int((seed + 104729 * (index + 1)) % (2**31 - 1))
        for index, name in enumerate(names)
    }
    return {
        name: torch.Generator(device="cpu").manual_seed(value)
        for name, value in derived.items()
    }, derived


def _config_to_dict(config: Tier1WaveformConfig) -> dict:
    payload = asdict(config)
    payload["snr_db_values"] = list(config.snr_db_values)
    return payload


def _runtime_environment(device: torch.device) -> dict:
    gpu_name = None
    gpu_total_memory_bytes = None
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        gpu_name = torch.cuda.get_device_name(device)
        gpu_total_memory_bytes = int(properties.total_memory)
    return {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "requested_device": str(device),
        "gpu_name": gpu_name,
        "gpu_total_memory_bytes": gpu_total_memory_bytes,
    }


def _validate_runtime_device(device_text: str) -> torch.device:
    device = torch.device(device_text)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"Requested CUDA device {device}, but CUDA is unavailable.")
    return device


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="ascii") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _validate_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}.")


def _validate_non_negative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value}.")


def _validate_finite_positive(name: str, value: float) -> None:
    if (isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0):
        raise ValueError(f"{name} must be finite and positive, got {value}.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run paired Tier 1 waveform validation for frozen TX artifacts.",
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device override, such as cuda:0 or cpu.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    config = load_tier1_waveform_config(args.config_json)
    if args.device is not None:
        config = replace(config, device=args.device)
    run_tx_waveform_tier1(
        args.run_dir,
        args.output_dir,
        config=config,
    )


if __name__ == "__main__":
    main()
