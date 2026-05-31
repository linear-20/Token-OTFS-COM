"""Cloud-ready TX-only L_core training orchestration.

This experiment harness keeps the Step 17C2 transmitter objective unchanged:

    L_core = mean_p sum_r w[r] * relu(gamma - d[p, r]) ** 2

It adds experiment controls needed for longer runs:
- causal integer-delay and signed integer-Doppler support;
- progressive DD-support curriculum;
- refreshed training scenario banks;
- fixed held-out validation and test banks from independent RNG streams;
- resumable checkpoints with optimizer and RNG state.

This remains an on-grid sparse-DD surrogate experiment. It does not train the
receiver, apply an SNR schedule, model fractional shifts, or claim waveform TER.

Run from the workspace root:

    $env:PYTHONPATH = ".\\Learnable_Mapping_Tokens-to-DD-Signals"
    python -m experiments.tx_cloud_train --output-dir ".\\artifacts\\tx_cloud_smoke"
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from transmitter.codebook import TokenDDCodebook, initialize_token_codebook_
from transmitter.config import TransmitterConfig
from transmitter.export import (
    export_physical_codeword_book,
    save_raw_tx_state,
)
from transmitter.metrics import data_codeword_power
from transmitter.multipath_margin import (
    sparse_multipath_operator_margin_loss,
    sparse_multipath_operator_separation_scores,
)
from transmitter.multipath_scenarios import (
    SparseMultipathScenarioBank,
    sample_normalized_sparse_multipath_scenario_bank,
)
from transmitter.pilot_frame import build_transmitter_pilot_masks
from transmitter.sampling import sample_uniform_cross_token_pairs
from transmitter.shaping import SparseShiftSet
from transmitter.sparse_multipath import SparseMultipathDDChannel

from .step19_lcore_harness import summarize_separation_scores


CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TXCurriculumStage:
    """Inclusive training-step boundary and causal integer DD support."""

    until_step: int
    max_delay: int
    max_doppler: int

    def __post_init__(self) -> None:
        _validate_positive_int("until_step", self.until_step)
        _validate_non_negative_int("max_delay", self.max_delay)
        _validate_non_negative_int("max_doppler", self.max_doppler)


@dataclass(frozen=True)
class TXCloudTrainConfig:
    """Controls for cloud-ready TX-only surrogate training."""

    seed: int = 2026
    train_steps: int = 1000
    train_pairs_per_step: int = 128
    validation_pairs: int = 512
    test_pairs: int = 512
    eval_every: int = 100
    checkpoint_every: int = 100
    progress_every: int = 25
    train_bank_refresh_every: int = 25
    num_train_scenarios: int = 64
    num_validation_scenarios: int = 128
    num_test_scenarios: int = 128
    num_paths: int = 3
    target_margin: float = 1.0
    learning_rate: float = 0.01
    device: str = "cpu"
    curriculum: tuple[TXCurriculumStage, ...] = ()

    def __post_init__(self) -> None:
        _validate_non_negative_int("seed", self.seed)
        for name in (
            "train_steps",
            "train_pairs_per_step",
            "validation_pairs",
            "test_pairs",
            "eval_every",
            "checkpoint_every",
            "progress_every",
            "train_bank_refresh_every",
            "num_train_scenarios",
            "num_validation_scenarios",
            "num_test_scenarios",
            "num_paths",
        ):
            _validate_positive_int(name, getattr(self, name))
        _validate_finite_non_negative("target_margin", self.target_margin)
        _validate_finite_positive("learning_rate", self.learning_rate)
        if not isinstance(self.device, str):
            raise TypeError(
                f"device must be str, got {type(self.device).__name__}."
            )
        try:
            torch.device(self.device)
        except Exception as exc:
            raise ValueError(
                f"device must be a valid torch device string, "
                f"got {self.device!r}."
            ) from exc
        if not isinstance(self.curriculum, tuple):
            raise TypeError(
                f"curriculum must be a tuple, "
                f"got {type(self.curriculum).__name__}."
            )
        if not all(isinstance(stage, TXCurriculumStage)
                   for stage in self.curriculum):
            raise TypeError(
                "curriculum entries must be TXCurriculumStage instances."
            )


def build_causal_integer_shift_set(
    *,
    max_delay: int,
    max_doppler: int,
    name: str = "causal_integer_dd_support",
) -> SparseShiftSet:
    """Build physical on-grid path support: delay >= 0 and signed Doppler."""
    _validate_non_negative_int("max_delay", max_delay)
    _validate_non_negative_int("max_doppler", max_doppler)
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be a non-empty string.")

    delays = torch.arange(0, max_delay + 1, dtype=torch.long)
    dopplers = torch.arange(-max_doppler, max_doppler + 1, dtype=torch.long)
    delay_grid, doppler_grid = torch.meshgrid(
        delays, dopplers, indexing="ij",
    )
    shifts = torch.stack(
        [delay_grid.reshape(-1), doppler_grid.reshape(-1)], dim=-1,
    )
    return SparseShiftSet(shifts=shifts, weights=None, name=name)


def run_tx_cloud_training(
    tx_config: TransmitterConfig,
    output_dir: str | Path,
    *,
    run_config: TXCloudTrainConfig | None = None,
    resume_checkpoint: str | Path | None = None,
    stop_after_step: int | None = None,
    emit_progress: bool = False,
) -> dict:
    """Run or resume cloud-ready TX-only on-grid surrogate training.

    Args:
        tx_config: Physical transmitter configuration.
        output_dir: Persistent output directory. Resume must use the same path.
        run_config: Long-run controls.
        resume_checkpoint: Optional trusted checkpoint produced by this harness.
        stop_after_step: Optional controlled early stop for preemption testing.
        emit_progress: If True, print lightweight JSON progress records.

    Returns:
        JSON-serializable manifest dict.
    """
    if not isinstance(tx_config, TransmitterConfig):
        raise TypeError(
            f"tx_config must be a TransmitterConfig, "
            f"got {type(tx_config).__name__}."
        )
    if run_config is None:
        run_config = TXCloudTrainConfig()
    if not isinstance(run_config, TXCloudTrainConfig):
        raise TypeError(
            f"run_config must be a TXCloudTrainConfig, "
            f"got {type(run_config).__name__}."
        )
    if stop_after_step is not None:
        _validate_positive_int("stop_after_step", stop_after_step)
        if stop_after_step > run_config.train_steps:
            raise ValueError(
                f"stop_after_step must be <= train_steps "
                f"({run_config.train_steps}), got {stop_after_step}."
            )
    if not isinstance(emit_progress, bool):
        raise TypeError(
            f"emit_progress must be bool, "
            f"got {type(emit_progress).__name__}."
        )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    resolved_out = str(out.resolve())
    metrics_path = out / "metrics.jsonl"
    resumed = resume_checkpoint is not None
    if not resumed and metrics_path.exists():
        raise ValueError(
            f"output_dir already contains {metrics_path.name}. "
            "Use a new directory or pass resume_checkpoint."
        )
    stages = _resolve_curriculum(tx_config, run_config)
    derived_seeds = _derive_seeds(run_config.seed)
    resume_payload = None
    if resumed:
        resume_payload = _load_checkpoint(Path(resume_checkpoint))
        _validate_resume_checkpoint(
            resume_payload, resolved_out, tx_config, run_config, stages,
        )
    device = torch.device(run_config.device)
    masks = build_transmitter_pilot_masks(tx_config)
    data_mask = masks.data_mask
    evidence_mask = data_mask

    generators = {
        key: torch.Generator(device="cpu").manual_seed(seed)
        for key, seed in derived_seeds.items()
    }

    validation_bank = _sample_bank(
        stages[-1], run_config.num_validation_scenarios,
        run_config.num_paths, generators["validation_scenarios"],
        tx_config.torch_complex_dtype, "held_out_validation_scenarios",
    )
    test_bank = _sample_bank(
        stages[-1], run_config.num_test_scenarios,
        run_config.num_paths, generators["test_scenarios"],
        tx_config.torch_complex_dtype, "held_out_test_scenarios",
    )
    validation_pairs = sample_uniform_cross_token_pairs(
        tx_config.vocab_size, run_config.validation_pairs,
        generator=generators["validation_pairs"],
    )
    test_pairs = sample_uniform_cross_token_pairs(
        tx_config.vocab_size, run_config.test_pairs,
        generator=generators["test_pairs"],
    )
    _atomic_torch_save(
        _bank_to_payload(validation_bank, stages[-1]),
        out / "validation_scenario_bank.pt",
    )
    _atomic_torch_save(
        _bank_to_payload(test_bank, stages[-1]),
        out / "test_scenario_bank.pt",
    )

    with torch.random.fork_rng(devices=[]):
        codebook = TokenDDCodebook(tx_config)
        if resume_checkpoint is None:
            initialize_token_codebook_(
                codebook, mode="random_phase",
                generator=generators["codebook"],
            )
    codebook.to(device=device)
    optimizer = torch.optim.Adam(
        [codebook.raw_real, codebook.raw_imag],
        lr=run_config.learning_rate,
    )

    checkpoint_path = out / "checkpoint_latest.pt"
    completed_step = 0
    current_stage_index: int | None = None
    current_train_bank: SparseMultipathScenarioBank | None = None
    train_bank_draw_count = 0
    latest_train_loss: float | None = None

    if resumed:
        payload = resume_payload
        codebook.load_state_dict(payload["codebook_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        _optimizer_to_device(optimizer, device)
        completed_step = payload["completed_step"]
        current_stage_index = payload["current_stage_index"]
        train_bank_draw_count = payload["train_bank_draw_count"]
        latest_train_loss = payload["latest_train_loss"]
        current_train_bank_payload = payload["current_train_scenario_bank"]
        if current_train_bank_payload is not None:
            current_train_bank = _bank_from_payload(
                current_train_bank_payload,
            )
        generators["train_scenarios"].set_state(
            payload["generator_states"]["train_scenarios"],
        )
        generators["train_pairs"].set_state(
            payload["generator_states"]["train_pairs"],
        )
        if stop_after_step is not None and stop_after_step < completed_step:
            raise ValueError(
                f"stop_after_step {stop_after_step} is below resumed "
                f"completed_step {completed_step}."
            )
        if completed_step >= run_config.train_steps:
            return _read_json(out / "manifest.json")
    else:
        save_raw_tx_state(codebook, out / "baseline_raw_state.pt")
        export_physical_codeword_book(
            codebook, out / "baseline_physical_codeword_book.pt",
            data_mask=data_mask,
        )

    t_start = time.perf_counter()
    if emit_progress:
        _emit_progress({
            "event": "training_start",
            "completed_step": completed_step,
            "target_step": (
                run_config.train_steps if stop_after_step is None
                else stop_after_step
            ),
            "resumed": resumed,
        })

    def evaluate(
        *,
        phase: str,
        split: str,
        step: int,
        bank: SparseMultipathScenarioBank,
        pairs: torch.Tensor,
    ) -> dict:
        with torch.no_grad():
            cw = codebook.forward(data_mask=data_mask)
            eval_l_core = sparse_multipath_operator_margin_loss(
                cw, pairs, bank, evidence_mask,
                target_margin=run_config.target_margin,
            )
            scores = sparse_multipath_operator_separation_scores(
                cw, pairs, bank, evidence_mask,
            )
            diagnostics = summarize_separation_scores(
                scores, target_margin=run_config.target_margin,
            )
            powers = data_codeword_power(cw, data_mask)
        record = {
            "phase": phase,
            "split": split,
            "step": step,
            "eval_l_core": float(eval_l_core.item()),
            **diagnostics,
            "data_power_min": float(powers.min().item()),
            "data_power_max": float(powers.max().item()),
            "latest_train_l_core": latest_train_loss,
            "run_elapsed_seconds": time.perf_counter() - t_start,
        }
        _append_jsonl(metrics_path, record)
        if emit_progress:
            _emit_progress({"event": "evaluation", **record})
        return record

    if not resumed:
        if emit_progress:
            _emit_progress({
                "event": "baseline_validation_start",
                "step": 0,
                "validation_pairs": run_config.validation_pairs,
                "validation_scenarios":
                    run_config.num_validation_scenarios,
            })
        evaluate(
            phase="baseline", split="validation", step=0,
            bank=validation_bank, pairs=validation_pairs,
        )
        _save_checkpoint(
            checkpoint_path, out, tx_config, run_config, stages,
            codebook, optimizer, generators, completed_step,
            current_stage_index, current_train_bank,
            train_bank_draw_count, latest_train_loss,
        )

    target_step = (
        run_config.train_steps if stop_after_step is None
        else stop_after_step
    )
    for step in range(completed_step + 1, target_step + 1):
        stage_index = _stage_index_for_step(stages, step)
        refresh_bank = (
            current_train_bank is None
            or current_stage_index != stage_index
            or (step - 1) % run_config.train_bank_refresh_every == 0
        )
        if refresh_bank:
            current_train_bank = _sample_bank(
                stages[stage_index], run_config.num_train_scenarios,
                run_config.num_paths, generators["train_scenarios"],
                tx_config.torch_complex_dtype,
                f"train_scenarios_draw_{train_bank_draw_count + 1}",
            )
            current_stage_index = stage_index
            train_bank_draw_count += 1

        train_pairs = sample_uniform_cross_token_pairs(
            tx_config.vocab_size, run_config.train_pairs_per_step,
            generator=generators["train_pairs"],
        )
        cw = codebook.forward(data_mask=data_mask)
        loss = sparse_multipath_operator_margin_loss(
            cw, train_pairs, current_train_bank, evidence_mask,
            target_margin=run_config.target_margin,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        latest_train_loss = float(loss.item())
        completed_step = step
        if emit_progress and step % run_config.progress_every == 0:
            elapsed = time.perf_counter() - t_start
            _emit_progress({
                "event": "train_progress",
                "step": step,
                "target_step": target_step,
                "latest_train_l_core": latest_train_loss,
                "elapsed_seconds": elapsed,
                "estimated_remaining_seconds": (
                    elapsed / step * (target_step - step)
                ),
            })

        if (step % run_config.eval_every == 0
                and step < target_step):
            evaluate(
                phase="intermediate", split="validation", step=step,
                bank=validation_bank, pairs=validation_pairs,
            )
        if step % run_config.checkpoint_every == 0:
            _save_checkpoint(
                checkpoint_path, out, tx_config, run_config, stages,
                codebook, optimizer, generators, completed_step,
                current_stage_index, current_train_bank,
                train_bank_draw_count, latest_train_loss,
            )

    complete = completed_step >= run_config.train_steps
    validation_record = evaluate(
        phase="final" if complete else "paused",
        split="validation", step=completed_step,
        bank=validation_bank, pairs=validation_pairs,
    )
    test_record = None
    if complete:
        test_record = evaluate(
            phase="final", split="test", step=completed_step,
            bank=test_bank, pairs=test_pairs,
        )

    save_raw_tx_state(codebook, out / "shaped_raw_state.pt")
    export_physical_codeword_book(
        codebook, out / "shaped_physical_codeword_book.pt",
        data_mask=data_mask,
    )
    _save_checkpoint(
        checkpoint_path, out, tx_config, run_config, stages,
        codebook, optimizer, generators, completed_step,
        current_stage_index, current_train_bank,
        train_bank_draw_count, latest_train_loss,
    )
    manifest = _build_manifest(
        tx_config, run_config, stages, derived_seeds, completed_step,
        complete, resumed, train_bank_draw_count, validation_record,
        test_record,
    )
    _write_json(out / "manifest.json", manifest)
    return manifest


def _sample_bank(
    stage: TXCurriculumStage,
    num_scenarios: int,
    num_paths: int,
    generator: torch.Generator,
    complex_dtype: torch.dtype,
    name: str,
) -> SparseMultipathScenarioBank:
    support = build_causal_integer_shift_set(
        max_delay=stage.max_delay,
        max_doppler=stage.max_doppler,
        name=f"{name}_support",
    )
    return sample_normalized_sparse_multipath_scenario_bank(
        support, num_scenarios=num_scenarios, num_paths=num_paths,
        generator=generator, complex_dtype=complex_dtype, name=name,
    )


def _resolve_curriculum(
    tx_config: TransmitterConfig,
    run_config: TXCloudTrainConfig,
) -> tuple[TXCurriculumStage, ...]:
    stages = run_config.curriculum or (
        TXCurriculumStage(
            until_step=run_config.train_steps,
            max_delay=tx_config.max_channel_delay,
            max_doppler=tx_config.max_channel_doppler,
        ),
    )
    previous_until = 0
    previous_delay = -1
    previous_doppler = -1
    for stage in stages:
        if stage.until_step <= previous_until:
            raise ValueError("curriculum until_step values must increase.")
        if stage.max_delay < previous_delay or stage.max_doppler < previous_doppler:
            raise ValueError(
                "curriculum DD support must expand monotonically."
            )
        if stage.max_delay > tx_config.max_channel_delay:
            raise ValueError(
                f"curriculum max_delay {stage.max_delay} exceeds "
                f"tx_config.max_channel_delay {tx_config.max_channel_delay}."
            )
        if stage.max_doppler > tx_config.max_channel_doppler:
            raise ValueError(
                f"curriculum max_doppler {stage.max_doppler} exceeds "
                f"tx_config.max_channel_doppler "
                f"{tx_config.max_channel_doppler}."
            )
        previous_until = stage.until_step
        previous_delay = stage.max_delay
        previous_doppler = stage.max_doppler
    if stages[-1].until_step < run_config.train_steps:
        raise ValueError(
            "curriculum final until_step must cover train_steps."
        )
    return stages


def _stage_index_for_step(
    stages: tuple[TXCurriculumStage, ...],
    step: int,
) -> int:
    for index, stage in enumerate(stages):
        if step <= stage.until_step:
            return index
    raise ValueError(f"No curriculum stage covers step {step}.")


def _derive_seeds(seed: int) -> dict[str, int]:
    gen = torch.Generator(device="cpu").manual_seed(seed)
    names = (
        "codebook",
        "train_scenarios",
        "validation_scenarios",
        "test_scenarios",
        "train_pairs",
        "validation_pairs",
        "test_pairs",
    )
    return {
        name: int(torch.randint(0, 2 ** 31, (1,), generator=gen).item())
        for name in names
    }


def _bank_to_payload(
    bank: SparseMultipathScenarioBank,
    support: TXCurriculumStage | None = None,
) -> dict:
    ch = bank.channel
    payload = {
        "path_shifts": ch.path_shifts.detach().cpu(),
        "path_gains": ch.path_gains.detach().cpu(),
        "path_active_mask": (
            ch.path_active_mask.detach().cpu()
            if ch.path_active_mask is not None else None
        ),
        "scenario_weights": (
            bank.scenario_weights.detach().cpu()
            if bank.scenario_weights is not None else None
        ),
        "source_shift_indices": (
            bank.source_shift_indices.detach().cpu()
            if bank.source_shift_indices is not None else None
        ),
        "name": bank.name,
    }
    if support is not None:
        payload["support"] = asdict(support)
    return payload


def _bank_from_payload(payload: dict) -> SparseMultipathScenarioBank:
    channel = SparseMultipathDDChannel(
        path_shifts=payload["path_shifts"],
        path_gains=payload["path_gains"],
        path_active_mask=payload["path_active_mask"],
    )
    return SparseMultipathScenarioBank(
        channel=channel,
        scenario_weights=payload["scenario_weights"],
        source_shift_indices=payload["source_shift_indices"],
        name=payload["name"],
    )


def _save_checkpoint(
    latest_path: Path,
    out: Path,
    tx_config: TransmitterConfig,
    run_config: TXCloudTrainConfig,
    stages: tuple[TXCurriculumStage, ...],
    codebook: TokenDDCodebook,
    optimizer: torch.optim.Optimizer,
    generators: dict[str, torch.Generator],
    completed_step: int,
    current_stage_index: int | None,
    current_train_bank: SparseMultipathScenarioBank | None,
    train_bank_draw_count: int,
    latest_train_loss: float | None,
) -> None:
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "output_dir": str(out.resolve()),
        "tx_config": asdict(tx_config),
        "run_config": _run_config_to_dict(run_config),
        "effective_curriculum": [asdict(stage) for stage in stages],
        "completed_step": completed_step,
        "current_stage_index": current_stage_index,
        "train_bank_draw_count": train_bank_draw_count,
        "latest_train_loss": latest_train_loss,
        "codebook_state": _tree_to_cpu(codebook.state_dict()),
        "optimizer_state": _tree_to_cpu(optimizer.state_dict()),
        "generator_states": {
            "train_scenarios": generators["train_scenarios"].get_state(),
            "train_pairs": generators["train_pairs"].get_state(),
        },
        "current_train_scenario_bank": (
            _bank_to_payload(current_train_bank)
            if current_train_bank is not None else None
        ),
    }
    snapshots = out / "checkpoints"
    snapshots.mkdir(parents=True, exist_ok=True)
    snapshot = snapshots / f"step_{completed_step:08d}.pt"
    _atomic_torch_save(payload, snapshot)
    _atomic_torch_save(payload, latest_path)


def _load_checkpoint(path: Path) -> dict:
    payload = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must contain a dict payload.")
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"checkpoint schema_version must be "
            f"{CHECKPOINT_SCHEMA_VERSION}."
        )
    return payload


def _validate_resume_checkpoint(
    payload: dict,
    resolved_out: str,
    tx_config: TransmitterConfig,
    run_config: TXCloudTrainConfig,
    stages: tuple[TXCurriculumStage, ...],
) -> None:
    if payload.get("output_dir") != resolved_out:
        raise ValueError("resume_checkpoint output_dir mismatch.")
    if payload.get("tx_config") != asdict(tx_config):
        raise ValueError("resume_checkpoint tx_config mismatch.")
    if payload.get("run_config") != _run_config_to_dict(run_config):
        raise ValueError("resume_checkpoint run_config mismatch.")
    if payload.get("effective_curriculum") != [
        asdict(stage) for stage in stages
    ]:
        raise ValueError("resume_checkpoint curriculum mismatch.")
    completed_step = payload.get("completed_step")
    if isinstance(completed_step, bool) or not isinstance(completed_step, int):
        raise ValueError("resume_checkpoint completed_step must be int.")
    if completed_step < 0 or completed_step > run_config.train_steps:
        raise ValueError("resume_checkpoint completed_step out of range.")


def _optimizer_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device=device)


def _tree_to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _tree_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_tree_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_tree_to_cpu(item) for item in value)
    return value


def _run_config_to_dict(run_config: TXCloudTrainConfig) -> dict:
    payload = asdict(run_config)
    payload["curriculum"] = [
        asdict(stage) for stage in run_config.curriculum
    ]
    return payload


def _build_manifest(
    tx_config: TransmitterConfig,
    run_config: TXCloudTrainConfig,
    stages: tuple[TXCurriculumStage, ...],
    derived_seeds: dict[str, int],
    completed_step: int,
    complete: bool,
    resumed: bool,
    train_bank_draw_count: int,
    validation_record: dict,
    test_record: dict | None,
) -> dict:
    return {
        "schema_version": 1,
        "tx_config": asdict(tx_config),
        "run_config": _run_config_to_dict(run_config),
        "effective_curriculum": [asdict(stage) for stage in stages],
        "derived_seeds": derived_seeds,
        "runtime_environment": _runtime_environment(
            torch.device(run_config.device),
        ),
        "completed_step": completed_step,
        "complete": complete,
        "resumed": resumed,
        "train_bank_draw_count": train_bank_draw_count,
        "validation_metrics": validation_record,
        "test_metrics": test_record,
        "artifact_files": {
            "baseline_raw_state": "baseline_raw_state.pt",
            "baseline_physical_codeword_book":
                "baseline_physical_codeword_book.pt",
            "shaped_raw_state": "shaped_raw_state.pt",
            "shaped_physical_codeword_book":
                "shaped_physical_codeword_book.pt",
            "validation_scenario_bank": "validation_scenario_bank.pt",
            "test_scenario_bank": "test_scenario_bank.pt",
            "checkpoint_latest": "checkpoint_latest.pt",
            "checkpoint_snapshots": "checkpoints/",
            "metrics_jsonl": "metrics.jsonl",
            "manifest": "manifest.json",
        },
        "flags": {
            "surrogate_only": True,
            "gradient_objective":
                "sparse_multipath_operator_margin_loss_only",
            "scenario_weights_downstream_uniform": True,
            "training_scenario_banks_refreshed": True,
            "validation_test_have_no_gradient": True,
            "causal_integer_delay_support": True,
            "signed_integer_doppler_support": True,
            "snr_training_applied": False,
            "fractional_delay_doppler_training_applied": False,
            "waveform_evaluation_performed": False,
            "ter_evaluated": False,
            "receiver_changed": False,
            "transmitter_architecture_changed": False,
        },
        "claim_boundary": (
            "TX-only on-grid sparse-DD surrogate shaping. "
            "Waveform SNR sweeps, off-grid mismatch, receiver training, "
            "and end-to-end TER evaluation remain separate stages."
        ),
    }


def _runtime_environment(device: torch.device) -> dict:
    """Record the software and requested accelerator used by a run."""
    gpu_name = None
    gpu_total_memory_bytes = None
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(
                f"Requested CUDA device {device}, but CUDA is unavailable."
            )
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


def _atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, str(temporary))
    temporary.replace(path)


def _append_jsonl(path: Path, record: dict) -> None:
    with open(path, "a", encoding="ascii") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def _emit_progress(record: dict) -> None:
    print(json.dumps(record, ensure_ascii=True), flush=True)


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="ascii") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
    temporary.replace(path)


def _read_json(path: Path) -> dict:
    with open(path, encoding="ascii") as handle:
        return json.load(handle)


def _validate_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}.")


def _validate_non_negative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"{name} must be a non-negative integer, got {value}."
        )


def _validate_finite_non_negative(name: str, value: float) -> None:
    if (isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0):
        raise ValueError(f"{name} must be finite and >= 0, got {value}.")


def _validate_finite_positive(name: str, value: float) -> None:
    if (isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0):
        raise ValueError(f"{name} must be finite and > 0, got {value}.")


def _default_tx_config() -> TransmitterConfig:
    return TransmitterConfig(
        M=12, N=12, vocab_size=16,
        pilot_delay=6, pilot_doppler=6,
        pilot_guard_delay=2, pilot_guard_doppler=2,
        pilot_obs_delay_radius=0, pilot_obs_doppler_radius=0,
        max_channel_delay=2, max_channel_doppler=2,
        data_power=1.0,
        pilot_value_real=2.0, pilot_value_imag=0.0,
        complex_dtype="complex64",
    )


def _load_tx_config_json(path: Path) -> TransmitterConfig:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise ValueError("tx-config JSON must contain an object.")
    return TransmitterConfig(**payload)


def _load_curriculum_json(path: Path) -> tuple[TXCurriculumStage, ...]:
    payload = _read_json(path)
    if not isinstance(payload, list) or not payload:
        raise ValueError(
            "curriculum JSON must contain a non-empty array."
        )
    if not all(isinstance(item, dict) for item in payload):
        raise ValueError("curriculum JSON entries must be objects.")
    return tuple(TXCurriculumStage(**item) for item in payload)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cloud-ready TX-only L_core surrogate training.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("./artifacts/tx_cloud_smoke"),
    )
    parser.add_argument("--tx-config-json", type=Path, default=None)
    parser.add_argument("--curriculum-json", type=Path, default=None)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train-steps", type=int, default=1000)
    parser.add_argument("--train-pairs-per-step", type=int, default=128)
    parser.add_argument("--validation-pairs", type=int, default=512)
    parser.add_argument("--test-pairs", type=int, default=512)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--train-bank-refresh-every", type=int, default=25)
    parser.add_argument("--num-train-scenarios", type=int, default=64)
    parser.add_argument("--num-validation-scenarios", type=int, default=128)
    parser.add_argument("--num-test-scenarios", type=int, default=128)
    parser.add_argument("--num-paths", type=int, default=3)
    parser.add_argument("--target-margin", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--warmup-steps", type=int, default=250)
    parser.add_argument("--stop-after-step", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    tx_config = (
        _default_tx_config()
        if args.tx_config_json is None
        else _load_tx_config_json(args.tx_config_json)
    )
    curriculum: tuple[TXCurriculumStage, ...]
    if args.curriculum_json is not None:
        curriculum = _load_curriculum_json(args.curriculum_json)
    elif 0 < args.warmup_steps < args.train_steps:
        curriculum = (
            TXCurriculumStage(
                until_step=args.warmup_steps,
                max_delay=min(1, tx_config.max_channel_delay),
                max_doppler=min(1, tx_config.max_channel_doppler),
            ),
            TXCurriculumStage(
                until_step=args.train_steps,
                max_delay=tx_config.max_channel_delay,
                max_doppler=tx_config.max_channel_doppler,
            ),
        )
    else:
        curriculum = ()
    run_config = TXCloudTrainConfig(
        seed=args.seed,
        train_steps=args.train_steps,
        train_pairs_per_step=args.train_pairs_per_step,
        validation_pairs=args.validation_pairs,
        test_pairs=args.test_pairs,
        eval_every=args.eval_every,
        checkpoint_every=args.checkpoint_every,
        progress_every=args.progress_every,
        train_bank_refresh_every=args.train_bank_refresh_every,
        num_train_scenarios=args.num_train_scenarios,
        num_validation_scenarios=args.num_validation_scenarios,
        num_test_scenarios=args.num_test_scenarios,
        num_paths=args.num_paths,
        target_margin=args.target_margin,
        learning_rate=args.learning_rate,
        device=args.device,
        curriculum=curriculum,
    )
    manifest = run_tx_cloud_training(
        tx_config, args.output_dir, run_config=run_config,
        resume_checkpoint=args.resume_checkpoint,
        stop_after_step=args.stop_after_step,
        emit_progress=True,
    )
    print(json.dumps({
        "completed_step": manifest["completed_step"],
        "complete": manifest["complete"],
        "validation_l_core":
            manifest["validation_metrics"]["eval_l_core"],
        "test_l_core": (
            manifest["test_metrics"]["eval_l_core"]
            if manifest["test_metrics"] is not None else None
        ),
        "output_dir": str(args.output_dir),
    }, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
