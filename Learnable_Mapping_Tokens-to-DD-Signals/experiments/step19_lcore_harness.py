"""Standalone L_core surrogate shaping harness.

Optimizes a TokenDDCodebook using the existing Step 17C2
sparse_multipath_operator_margin_loss and emits auditable artifacts.
This is NOT waveform validation and NOT a new transmitter module.

Run from the workspace root:

    $env:PYTHONPATH = ".\\Learnable_Mapping_Tokens-to-DD-Signals"
    python -m experiments.step19_lcore_harness --output-dir ".\\artifacts\\step19_smoke"
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

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
from transmitter.shaping import SparseShiftSet, default_integer_shift_set


@dataclass(frozen=True)
class Step19LCoreRunConfig:
    """Experiment-run controls (not transmitter architecture changes).

    Attributes:
        seed: Base random seed for all derived generators.
        train_steps: Number of optimizer steps.
        train_pairs_per_step: Number of ordered non-self pairs sampled
            uniformly per training step (with replacement).
        eval_pairs: Number of ordered non-self pairs in the fixed
            evaluation set.
        eval_every: Evaluate every N training steps; also evaluate at
            baseline (step 0) and final step.
        num_scenarios: R, number of fixed multipath scenarios.
        num_paths: K, number of path slots per scenario.
        target_margin: gamma >= 0, global physical hyperparameter.
        learning_rate: Positive finite float for Adam.
        device: Torch device string (e.g. "cpu" or "cuda:0").
    """

    seed: int = 42
    train_steps: int = 100
    train_pairs_per_step: int = 32
    eval_pairs: int = 128
    eval_every: int = 20
    num_scenarios: int = 16
    num_paths: int = 3
    target_margin: float = 1.0
    learning_rate: float = 0.01
    device: str = "cpu"

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError(
                f"seed must be int, got {type(self.seed).__name__}."
            )
        if self.seed < 0:
            raise ValueError(f"seed must be >= 0, got {self.seed}.")

        for name in ("train_steps", "train_pairs_per_step",
                     "eval_pairs", "eval_every", "num_scenarios",
                     "num_paths"):
            val = getattr(self, name)
            if isinstance(val, bool) or not isinstance(val, int):
                raise TypeError(
                    f"{name} must be int, got {type(val).__name__}."
                )
            if val <= 0:
                raise ValueError(
                    f"{name} must be > 0, got {val}."
                )

        # target_margin: float, not bool, finite, >= 0
        if isinstance(self.target_margin, bool) or not isinstance(
            self.target_margin, (int, float),
        ):
            raise TypeError(
                f"target_margin must be a number, "
                f"got {type(self.target_margin).__name__}."
            )
        if not math.isfinite(self.target_margin) or self.target_margin < 0.0:
            raise ValueError(
                f"target_margin must be finite and >= 0, "
                f"got {self.target_margin}."
            )

        # learning_rate: float, not bool, finite, > 0
        if isinstance(self.learning_rate, bool) or not isinstance(
            self.learning_rate, (int, float),
        ):
            raise TypeError(
                f"learning_rate must be a number, "
                f"got {type(self.learning_rate).__name__}."
            )
        if (not math.isfinite(self.learning_rate)
                or self.learning_rate <= 0.0):
            raise ValueError(
                f"learning_rate must be finite positive, "
                f"got {self.learning_rate}."
            )

        # device: validate by constructing torch.device
        if not isinstance(self.device, str):
            raise TypeError(
                f"device must be str, got {type(self.device).__name__}."
            )
        try:
            torch.device(self.device)
        except Exception:
            raise ValueError(
                f"device must be a valid torch device string, "
                f"got {self.device!r}."
            )

# -- diagnostics ---------------------------------------------------------------


def summarize_separation_scores(
    scores: torch.Tensor,
    *,
    target_margin: float,
) -> dict:
    """Compute auditable separation diagnostics from scores [P, R].

    Args:
        scores: Real finite tensor [P, R].
        target_margin: gamma >= 0.

    Returns:
        Dict with Python float values: separation_mean, separation_min,
        separation_p01, outage_probability.
    """
    if not torch.is_tensor(scores):
        raise TypeError(
            f"scores must be a torch.Tensor, got {type(scores).__name__}."
        )
    if torch.is_complex(scores) or not torch.is_floating_point(scores):
        raise TypeError(
            f"scores must be a real floating tensor, got dtype {scores.dtype}."
        )
    if not torch.isfinite(scores).all():
        raise ValueError("scores must be finite.")
    if scores.ndim != 2:
        raise ValueError(
            f"scores must have 2 dimensions [P, R], "
            f"got ndim={scores.ndim}."
        )
    if scores.shape[0] <= 0 or scores.shape[1] <= 0:
        raise ValueError(
            f"scores dimensions P and R must be > 0, got {list(scores.shape)}."
        )
    _validate_target_margin(target_margin)

    flat = scores.reshape(-1)
    return {
        "separation_mean": flat.mean().item(),
        "separation_min": flat.min().item(),
        "separation_p01": torch.quantile(
            flat, 0.01, interpolation="linear",
        ).item(),
        "outage_probability": (flat < target_margin).float().mean().item(),
    }


# -- training harness ----------------------------------------------------------


def run_step19_lcore_harness(
    tx_config: TransmitterConfig,
    source_shift_set: SparseShiftSet,
    output_dir: str | Path,
    *,
    run_config: Step19LCoreRunConfig | None = None,
) -> dict:
    """Run standalone L_core surrogate shaping and emit artifacts.

    Args:
        tx_config: Validated TransmitterConfig.
        source_shift_set: SparseShiftSet with .shifts long [S, 2].
        output_dir: Directory for all output artifacts.
        run_config: Experiment-run controls.

    Returns:
        Manifest dict with keys: tx_config, run_config, derived_seeds,
        artifact_files, baseline_metrics, final_metrics, flags.
    """
    if run_config is None:
        run_config = Step19LCoreRunConfig()
    if not isinstance(tx_config, TransmitterConfig):
        raise TypeError(
            f"tx_config must be a TransmitterConfig, "
            f"got {type(tx_config).__name__}."
        )
    if not isinstance(source_shift_set, SparseShiftSet):
        raise TypeError(
            f"source_shift_set must be a SparseShiftSet, "
            f"got {type(source_shift_set).__name__}."
        )
    if not isinstance(run_config, Step19LCoreRunConfig):
        raise TypeError(
            f"run_config must be a Step19LCoreRunConfig, "
            f"got {type(run_config).__name__}."
        )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(run_config.device)
    dtype = tx_config.torch_complex_dtype

    # -- derived deterministic seeds ------------------------------------------
    base_gen = torch.Generator(device="cpu").manual_seed(run_config.seed)
    seed_book = int(torch.randint(0, 2 ** 31, (1,), generator=base_gen).item())
    seed_scenario = int(torch.randint(0, 2 ** 31, (1,), generator=base_gen).item())
    seed_train_pairs = int(torch.randint(0, 2 ** 31, (1,), generator=base_gen).item())
    seed_eval_pairs = int(torch.randint(0, 2 ** 31, (1,), generator=base_gen).item())
    derived_seeds = {
        "seed_book": seed_book,
        "seed_scenario": seed_scenario,
        "seed_train_pairs": seed_train_pairs,
        "seed_eval_pairs": seed_eval_pairs,
    }

    # -- masks -----------------------------------------------------------------
    masks = build_transmitter_pilot_masks(tx_config)
    data_mask = masks.data_mask
    evidence_mask = data_mask  # data-mask evidence projection

    # -- codebook --------------------------------------------------------------
    book_gen = torch.Generator(device="cpu").manual_seed(seed_book)
    # TokenDDCodebook.__init__ initializes raw parameters before the explicit
    # initializer overwrites them. Preserve the caller's global CPU RNG state.
    with torch.random.fork_rng(devices=[]):
        codebook = TokenDDCodebook(tx_config)
        initialize_token_codebook_(codebook, mode="random_phase",
                                   generator=book_gen)
    codebook.to(device=device)

    # -- scenario bank (sampled once, frozen) ----------------------------------
    scenario_gen = torch.Generator(device="cpu").manual_seed(seed_scenario)
    scenario_bank = sample_normalized_sparse_multipath_scenario_bank(
        source_shift_set,
        num_scenarios=run_config.num_scenarios,
        num_paths=run_config.num_paths,
        generator=scenario_gen,
        complex_dtype=dtype,
    )

    # -- fixed evaluation pairs ------------------------------------------------
    eval_pair_gen = torch.Generator(device="cpu").manual_seed(seed_eval_pairs)
    eval_pairs = sample_uniform_cross_token_pairs(
        tx_config.vocab_size, run_config.eval_pairs, generator=eval_pair_gen,
    )

    # -- train pair generator --------------------------------------------------
    train_pair_gen = torch.Generator(device="cpu").manual_seed(seed_train_pairs)

    # -- optimizer -------------------------------------------------------------
    optimizer = torch.optim.Adam(
        [codebook.raw_real, codebook.raw_imag], lr=run_config.learning_rate,
    )

    # -- baseline exports ------------------------------------------------------
    baseline_raw_path = out / "baseline_raw_state.pt"
    baseline_phys_path = out / "baseline_physical_codeword_book.pt"
    save_raw_tx_state(codebook, baseline_raw_path)
    export_physical_codeword_book(codebook, baseline_phys_path,
                                  data_mask=data_mask)

    # -- training log ----------------------------------------------------------
    metrics_log: list[dict] = []
    t_start = time.perf_counter()

    def _evaluate(phase: str, step: int) -> None:
        with torch.no_grad():
            cw = codebook.forward(data_mask=data_mask)
            eval_l_core = sparse_multipath_operator_margin_loss(
                cw, eval_pairs, scenario_bank, evidence_mask,
                target_margin=run_config.target_margin,
            )
            scores = sparse_multipath_operator_separation_scores(
                cw, eval_pairs, scenario_bank, evidence_mask,
            )
            diag = summarize_separation_scores(
                scores, target_margin=run_config.target_margin,
            )
            dcp = data_codeword_power(cw, data_mask)
        metrics_log.append({
            "phase": phase,
            "step": step,
            "eval_l_core": float(eval_l_core.item()),
            **diag,
            "data_power_min": float(dcp.min().item()),
            "data_power_max": float(dcp.max().item()),
            "elapsed_seconds": time.perf_counter() - t_start,
        })

    # baseline evaluation
    _evaluate("baseline", 0)

    # -- training loop ---------------------------------------------------------
    train_pair_sampling_count = 0
    for step in range(1, run_config.train_steps + 1):
        train_pairs = sample_uniform_cross_token_pairs(
            tx_config.vocab_size, run_config.train_pairs_per_step,
            generator=train_pair_gen,
        )
        train_pair_sampling_count += 1

        cw = codebook.forward(data_mask=data_mask)
        loss = sparse_multipath_operator_margin_loss(
            cw, train_pairs, scenario_bank, evidence_mask,
            target_margin=run_config.target_margin,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step % run_config.eval_every == 0
                and step != run_config.train_steps):
            _evaluate("intermediate", step)

    # final evaluation
    _evaluate("final", run_config.train_steps)

    # -- shaped exports --------------------------------------------------------
    shaped_raw_path = out / "shaped_raw_state.pt"
    shaped_phys_path = out / "shaped_physical_codeword_book.pt"
    save_raw_tx_state(codebook, shaped_raw_path)
    export_physical_codeword_book(codebook, shaped_phys_path,
                                  data_mask=data_mask)

    # -- scenario bank export --------------------------------------------------
    scenario_path = out / "scenario_bank.pt"
    torch.save({
        "path_shifts": scenario_bank.channel.path_shifts.cpu(),
        "path_gains": scenario_bank.channel.path_gains.cpu(),
        "path_active_mask": (scenario_bank.channel.path_active_mask.cpu()
                             if scenario_bank.channel.path_active_mask
                             is not None else None),
        "source_shift_indices": (scenario_bank.source_shift_indices.cpu()
                                 if scenario_bank.source_shift_indices
                                 is not None else None),
        "name": scenario_bank.name,
        "scenario_weights": None,
        "scenario_weights_is_none": scenario_bank.scenario_weights is None,
        "note": "sampled-bank downstream aggregation uses uniform implied scenario weights",
    }, str(scenario_path))

    # -- metrics.jsonl ---------------------------------------------------------
    metrics_path = out / "metrics.jsonl"
    with open(metrics_path, "w", encoding="ascii") as f:
        for rec in metrics_log:
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")

    # -- manifest --------------------------------------------------------------
    bl = metrics_log[0]
    fl = metrics_log[-1]  # "final" record
    manifest = {
        "tx_config": {
            "M": tx_config.M,
            "N": tx_config.N,
            "vocab_size": tx_config.vocab_size,
            "pilot_delay": tx_config.pilot_delay,
            "pilot_doppler": tx_config.pilot_doppler,
            "pilot_guard_delay": tx_config.pilot_guard_delay,
            "pilot_guard_doppler": tx_config.pilot_guard_doppler,
            "pilot_obs_delay_radius": tx_config.pilot_obs_delay_radius,
            "pilot_obs_doppler_radius": tx_config.pilot_obs_doppler_radius,
            "max_channel_delay": tx_config.max_channel_delay,
            "max_channel_doppler": tx_config.max_channel_doppler,
            "data_power": tx_config.data_power,
            "complex_dtype": tx_config.complex_dtype,
        },
        "run_config": {
            "seed": run_config.seed,
            "train_steps": run_config.train_steps,
            "train_pairs_per_step": run_config.train_pairs_per_step,
            "eval_pairs": run_config.eval_pairs,
            "eval_every": run_config.eval_every,
            "num_scenarios": run_config.num_scenarios,
            "num_paths": run_config.num_paths,
            "target_margin": run_config.target_margin,
            "learning_rate": run_config.learning_rate,
            "device": run_config.device,
        },
        "derived_seeds": derived_seeds,
        "artifact_files": {
            "baseline_raw_state": str(baseline_raw_path.name),
            "baseline_physical_codeword_book": str(baseline_phys_path.name),
            "shaped_raw_state": str(shaped_raw_path.name),
            "shaped_physical_codeword_book": str(shaped_phys_path.name),
            "scenario_bank": str(scenario_path.name),
            "metrics_jsonl": str(metrics_path.name),
            "manifest": "manifest.json",
        },
        "baseline_metrics": {
            "eval_l_core": bl["eval_l_core"],
            "separation_mean": bl["separation_mean"],
            "separation_min": bl["separation_min"],
            "separation_p01": bl["separation_p01"],
            "outage_probability": bl["outage_probability"],
        },
        "final_metrics": {
            "eval_l_core": fl["eval_l_core"],
            "separation_mean": fl["separation_mean"],
            "separation_min": fl["separation_min"],
            "separation_p01": fl["separation_p01"],
            "outage_probability": fl["outage_probability"],
        },
        "flags": {
            "surrogate_only": True,
            "waveform_evaluation_performed": False,
            "ter_evaluated": False,
            "transmitter_architecture_changed": False,
        },
    }
    manifest_path = out / "manifest.json"
    with open(manifest_path, "w", encoding="ascii") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=True)

    return manifest


def _validate_target_margin(value: float) -> None:
    """Validate a finite non-negative diagnostic margin."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"target_margin must be a number, got {type(value).__name__}."
        )
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(
            f"target_margin must be finite and >= 0, got {value}."
        )


# -- CLI -----------------------------------------------------------------------


def _default_tx_config() -> TransmitterConfig:
    """Minimal valid config for a quick CPU smoke run."""
    return TransmitterConfig(
        M=8, N=6, vocab_size=8,
        pilot_delay=4, pilot_doppler=3,
        pilot_guard_delay=2, pilot_guard_doppler=2,
        pilot_obs_delay_radius=0, pilot_obs_doppler_radius=0,
        max_channel_delay=1, max_channel_doppler=1,
        data_power=1.0,
        pilot_value_real=2.0, pilot_value_imag=0.0,
        complex_dtype="complex64",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone L_core surrogate shaping harness (Step 19).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("./artifacts/step19_smoke"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-steps", type=int, default=100)
    parser.add_argument("--train-pairs-per-step", type=int, default=32)
    parser.add_argument("--eval-pairs", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--num-scenarios", type=int, default=16)
    parser.add_argument("--num-paths", type=int, default=3)
    parser.add_argument("--target-margin", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    tx_cfg = _default_tx_config()
    shift_set = default_integer_shift_set(
        max_delay=tx_cfg.max_channel_delay,
        max_doppler=tx_cfg.max_channel_doppler,
        include_zero=True,
    )
    run_cfg = Step19LCoreRunConfig(
        seed=args.seed,
        train_steps=args.train_steps,
        train_pairs_per_step=args.train_pairs_per_step,
        eval_pairs=args.eval_pairs,
        eval_every=args.eval_every,
        num_scenarios=args.num_scenarios,
        num_paths=args.num_paths,
        target_margin=args.target_margin,
        learning_rate=args.learning_rate,
        device=args.device,
    )
    manifest = run_step19_lcore_harness(
        tx_config=tx_cfg,
        source_shift_set=shift_set,
        output_dir=args.output_dir,
        run_config=run_cfg,
    )
    summary = {
        "baseline_l_core": manifest["baseline_metrics"]["eval_l_core"],
        "final_l_core": manifest["final_metrics"]["eval_l_core"],
        "output_dir": str(args.output_dir),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
