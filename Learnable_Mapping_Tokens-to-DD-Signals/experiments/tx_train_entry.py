"""Unified cloud entry for TX-only training and explicit ablation suites.

This module is an orchestration layer over experiments.tx_cloud_train.
It does not duplicate the optimizer, add transmitter modules, or change the
physical objective. Each configured run receives an independent output
directory and an auditable resolved configuration snapshot.

Run from the workspace root:

    python -m experiments.tx_train_entry \
        --experiment-json configs/tx_train_main.json

List or inspect a suite without training:

    python -m experiments.tx_train_entry \
        --experiment-json configs/tx_ablation_suite.json --list-runs
    python -m experiments.tx_train_entry \
        --experiment-json configs/tx_ablation_suite.json --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

from transmitter.config import TransmitterConfig

from .tx_cloud_train import (
    TXCloudTrainConfig,
    TXCurriculumStage,
    _load_curriculum_json,
    _load_tx_config_json,
    _run_config_to_dict,
    run_tx_cloud_training,
)


ENTRY_SCHEMA_VERSION = 1
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_RUN_CONFIG_FIELDS = {
    item.name for item in fields(TXCloudTrainConfig)
    if item.name != "curriculum"
}
_ENTRY_KEYS = {
    "schema_version",
    "study_name",
    "output_root",
    "tx_profile_json",
    "curriculum_json",
    "base_run_config",
    "runs",
}
_RUN_KEYS = {"name", "description", "tags", "overrides", "curriculum_json"}


@dataclass(frozen=True)
class ResolvedTXRun:
    """One explicit TX-only experiment run ready for execution."""

    name: str
    description: str
    tags: tuple[str, ...]
    output_dir: Path
    tx_config: TransmitterConfig
    run_config: TXCloudTrainConfig
    curriculum_source: str | None

    def snapshot(self) -> dict:
        """Return a JSON-serializable resolved run snapshot."""
        return {
            "schema_version": ENTRY_SCHEMA_VERSION,
            "run_name": self.name,
            "description": self.description,
            "tags": list(self.tags),
            "output_dir": str(self.output_dir),
            "tx_config": asdict(self.tx_config),
            "run_config": _run_config_to_dict(self.run_config),
            "curriculum_source": self.curriculum_source,
        }


@dataclass(frozen=True)
class ResolvedTXStudy:
    """Validated explicit TX-only training study."""

    name: str
    output_dir: Path
    source_json: Path
    runs: tuple[ResolvedTXRun, ...]

    def plan(self) -> dict:
        """Return the auditable execution plan."""
        return {
            "schema_version": ENTRY_SCHEMA_VERSION,
            "study_name": self.name,
            "source_json": str(self.source_json),
            "output_dir": str(self.output_dir),
            "run_count": len(self.runs),
            "runs": [run.snapshot() for run in self.runs],
        }


def load_tx_study(path: str | Path) -> ResolvedTXStudy:
    """Load and validate one explicit TX-only study JSON."""
    source = Path(path).resolve()
    payload = _read_json(source)
    _reject_unknown_keys(payload, _ENTRY_KEYS, "study")
    if payload.get("schema_version") != ENTRY_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {ENTRY_SCHEMA_VERSION}."
        )
    study_name = _validate_safe_name("study_name", payload.get("study_name"))
    output_root = _required_path_string(payload, "output_root")
    tx_profile_path = _required_path_string(payload, "tx_profile_json")
    curriculum_source = _optional_path_string(payload, "curriculum_json")
    base = payload.get("base_run_config")
    if not isinstance(base, dict):
        raise ValueError("base_run_config must be an object.")
    _validate_run_overrides(base, "base_run_config")
    tx_config = _load_tx_config_json(tx_profile_path)
    output_dir = output_root / study_name
    raw_runs = payload.get("runs")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise ValueError("runs must be a non-empty array.")
    resolved_runs = []
    names = set()
    for index, raw_run in enumerate(raw_runs):
        if not isinstance(raw_run, dict):
            raise ValueError(f"runs[{index}] must be an object.")
        _reject_unknown_keys(raw_run, _RUN_KEYS, f"runs[{index}]")
        name = _validate_safe_name(f"runs[{index}].name", raw_run.get("name"))
        if name in names:
            raise ValueError(f"Duplicate run name {name!r}.")
        names.add(name)
        description = raw_run.get("description", "")
        if not isinstance(description, str):
            raise ValueError(f"runs[{index}].description must be a string.")
        tags = raw_run.get("tags", [])
        if (not isinstance(tags, list)
                or not all(isinstance(tag, str) and tag for tag in tags)):
            raise ValueError(
                f"runs[{index}].tags must be an array of non-empty strings."
            )
        overrides = raw_run.get("overrides", {})
        if not isinstance(overrides, dict):
            raise ValueError(f"runs[{index}].overrides must be an object.")
        _validate_run_overrides(overrides, f"runs[{index}].overrides")
        merged = {**base, **overrides}
        run_curriculum_source = curriculum_source
        if "curriculum_json" in raw_run:
            run_curriculum_source = _optional_path_string(
                raw_run, "curriculum_json",
            )
        curriculum = _load_curriculum(run_curriculum_source)
        run_config = TXCloudTrainConfig(
            **merged, curriculum=curriculum,
        )
        resolved_runs.append(ResolvedTXRun(
            name=name,
            description=description,
            tags=tuple(tags),
            output_dir=output_dir / name,
            tx_config=tx_config,
            run_config=run_config,
            curriculum_source=(
                str(run_curriculum_source)
                if run_curriculum_source is not None else None
            ),
        ))
    return ResolvedTXStudy(
        name=study_name,
        output_dir=output_dir,
        source_json=source,
        runs=tuple(resolved_runs),
    )


def run_tx_study(
    study: ResolvedTXStudy,
    *,
    only_runs: tuple[str, ...] = (),
    dry_run: bool = False,
    resume: bool = False,
    emit_progress: bool = True,
) -> dict:
    """Execute selected explicit TX-only runs sequentially."""
    if not isinstance(study, ResolvedTXStudy):
        raise TypeError(
            f"study must be a ResolvedTXStudy, got {type(study).__name__}."
        )
    selected = _select_runs(study, only_runs)
    plan = {
        **study.plan(),
        "selected_run_count": len(selected),
        "selected_runs": [run.name for run in selected],
    }
    if dry_run:
        return plan
    study.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(study.output_dir / "study_plan.json", plan)
    state_path = study.output_dir / "study_manifest.json"
    state = _load_or_initialize_study_manifest(state_path, study)
    _write_json(state_path, state)
    for run in selected:
        snapshot = run.snapshot()
        run.output_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = run.output_dir / "entry_resolved_config.json"
        _write_or_validate_snapshot(snapshot_path, snapshot)
        checkpoint = run.output_dir / "checkpoint_latest.pt"
        resume_checkpoint = checkpoint if resume and checkpoint.exists() else None
        _set_run_state(state, run.name, status="running")
        _write_json(state_path, state)
        try:
            manifest = run_tx_cloud_training(
                run.tx_config,
                run.output_dir,
                run_config=run.run_config,
                resume_checkpoint=resume_checkpoint,
                emit_progress=emit_progress,
            )
        except Exception as exc:
            _set_run_state(
                state, run.name, status="failed", error=str(exc),
            )
            _write_json(state_path, state)
            raise
        _set_run_state(
            state,
            run.name,
            status="completed" if manifest["complete"] else "paused",
            manifest_file=str(run.output_dir / "manifest.json"),
            completed_step=manifest["completed_step"],
            validation_metrics=manifest["validation_metrics"],
            test_metrics=manifest["test_metrics"],
        )
        _write_json(state_path, state)
    return state


def _load_curriculum(path: Path | None) -> tuple[TXCurriculumStage, ...]:
    if path is None:
        return ()
    return _load_curriculum_json(path)


def _select_runs(
    study: ResolvedTXStudy,
    only_runs: tuple[str, ...],
) -> tuple[ResolvedTXRun, ...]:
    if not isinstance(only_runs, tuple):
        raise TypeError("only_runs must be a tuple.")
    available = {run.name: run for run in study.runs}
    if not only_runs:
        return study.runs
    unknown = sorted(set(only_runs) - set(available))
    if unknown:
        raise ValueError(f"Unknown --only-run value(s): {unknown}.")
    selected_names = set(only_runs)
    return tuple(run for run in study.runs if run.name in selected_names)


def _initial_study_manifest(
    study: ResolvedTXStudy,
) -> dict:
    return {
        "schema_version": ENTRY_SCHEMA_VERSION,
        "study_name": study.name,
        "source_json": str(study.source_json),
        "output_dir": str(study.output_dir),
        "runs": [
            {
                "name": run.name,
                "description": run.description,
                "tags": list(run.tags),
                "output_dir": str(run.output_dir),
                "status": "pending",
            }
            for run in study.runs
        ],
    }


def _load_or_initialize_study_manifest(
    path: Path,
    study: ResolvedTXStudy,
) -> dict:
    if not path.exists():
        return _initial_study_manifest(study)
    state = _read_json(path)
    expected = {
        "schema_version": ENTRY_SCHEMA_VERSION,
        "study_name": study.name,
        "source_json": str(study.source_json),
        "output_dir": str(study.output_dir),
    }
    for key, value in expected.items():
        if state.get(key) != value:
            raise ValueError(
                f"Existing study manifest {key} mismatch at {path}."
            )
    run_states = state.get("runs")
    if not isinstance(run_states, list):
        raise ValueError(f"Existing study manifest runs must be an array.")
    existing_names = {
        run_state.get("name") for run_state in run_states
        if isinstance(run_state, dict)
    }
    for run in study.runs:
        if run.name not in existing_names:
            run_states.append(_pending_run_state(run))
    return state


def _pending_run_state(run: ResolvedTXRun) -> dict:
    return {
        "name": run.name,
        "description": run.description,
        "tags": list(run.tags),
        "output_dir": str(run.output_dir),
        "status": "pending",
    }


def _set_run_state(state: dict, name: str, **updates: Any) -> None:
    for run_state in state["runs"]:
        if run_state["name"] == name:
            run_state.update(updates)
            return
    raise ValueError(f"Missing run state {name!r}.")


def _required_path_string(payload: dict, key: str) -> Path:
    if key not in payload:
        raise ValueError(f"Missing required key {key!r}.")
    value = payload[key]
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty path string.")
    return Path(value)


def _optional_path_string(payload: dict, key: str) -> Path | None:
    if key not in payload or payload[key] is None:
        return None
    value = payload[key]
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be null or a non-empty path string.")
    return Path(value)


def _validate_safe_name(label: str, value: Any) -> str:
    if not isinstance(value, str) or not _SAFE_NAME.fullmatch(value):
        raise ValueError(
            f"{label} must match {_SAFE_NAME.pattern!r}, got {value!r}."
        )
    return value


def _validate_run_overrides(overrides: dict, label: str) -> None:
    unknown = sorted(set(overrides) - _RUN_CONFIG_FIELDS)
    if unknown:
        raise ValueError(f"{label} contains unknown field(s): {unknown}.")


def _reject_unknown_keys(payload: Any, allowed: set[str], label: str) -> None:
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be an object.")
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"{label} contains unknown field(s): {unknown}.")


def _write_or_validate_snapshot(path: Path, payload: dict) -> None:
    if path.exists():
        if _read_json(path) != payload:
            raise ValueError(
                f"Resolved run snapshot mismatch at {path}. "
                "Use a new run name for changed parameters."
            )
        return
    _write_json(path, payload)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="ascii") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
    temporary.replace(path)


def _read_json(path: Path) -> dict:
    with open(path, encoding="ascii") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified TX-only cloud training and ablation entry.",
    )
    parser.add_argument("--experiment-json", type=Path, required=True)
    parser.add_argument(
        "--only-run", action="append", default=[],
        help="Execute only the named run. Repeat to select multiple runs.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Resolve and print the selected execution plan without training.",
    )
    parser.add_argument(
        "--list-runs", action="store_true",
        help="List configured run names without training.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume selected runs from checkpoint_latest.pt when present.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    study = load_tx_study(args.experiment_json)
    if args.list_runs:
        print(json.dumps({
            "study_name": study.name,
            "runs": [
                {
                    "name": run.name,
                    "description": run.description,
                    "tags": list(run.tags),
                }
                for run in study.runs
            ],
        }, indent=2, ensure_ascii=True))
        return
    result = run_tx_study(
        study,
        only_runs=tuple(args.only_run),
        dry_run=args.dry_run,
        resume=args.resume,
        emit_progress=not args.dry_run,
    )
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
