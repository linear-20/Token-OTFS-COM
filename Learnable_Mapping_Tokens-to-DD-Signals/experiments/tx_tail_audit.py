"""Audit difficult pair-scenario tails from a completed TX-only run.

This is a read-only surrogate diagnostic. It does not train or modify the
transmitter and does not claim waveform TER or BER performance.

Run from the workspace root:

    python -m experiments.tx_tail_audit \
        --run-dir artifacts/tx_v256_m32_seed2026 \
        --split test --artifact both --device cuda:0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from transmitter.export import load_exported_physical_dd_artifact
from transmitter.multipath_margin import (
    sparse_multipath_operator_separation_scores,
)
from transmitter.sampling import sample_uniform_cross_token_pairs

from .step19_lcore_harness import summarize_separation_scores
from .tail_risk import summarize_tail_separation_scores
from .tx_cloud_train import _bank_from_payload


def audit_tx_tail(
    run_dir: str | Path,
    *,
    split: str = "test",
    artifact: str = "both",
    device: str = "cpu",
    top_k: int = 5,
    tail_cvar_fraction: float = 0.05,
) -> dict:
    """Audit held-out sparse-DD tail behavior from saved run artifacts."""
    if split not in {"validation", "test"}:
        raise ValueError(
            f"split must be 'validation' or 'test', got {split!r}."
        )
    if artifact not in {"baseline", "shaped", "both"}:
        raise ValueError(
            "artifact must be 'baseline', 'shaped', or 'both', "
            f"got {artifact!r}."
        )
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError(f"top_k must be a positive integer, got {top_k}.")
    resolved_device = torch.device(device)
    root = Path(run_dir)
    manifest = _read_json(root / "manifest.json")
    target_margin = manifest["run_config"]["target_margin"]
    pair_count = manifest["run_config"][f"{split}_pairs"]
    pair_seed = manifest["derived_seeds"][f"{split}_pairs"]
    vocab_size = manifest["tx_config"]["vocab_size"]
    pairs = sample_uniform_cross_token_pairs(
        vocab_size, pair_count,
        generator=torch.Generator(device="cpu").manual_seed(pair_seed),
    )
    bank_payload = torch.load(
        str(root / f"{split}_scenario_bank.pt"),
        map_location="cpu", weights_only=True,
    )
    bank = _bank_from_payload(bank_payload)
    duplicate_scenario_count = _count_scenarios_with_duplicate_dd_taps(
        bank_payload["path_shifts"],
    )
    selected_artifacts = (
        ("baseline", "baseline_physical_codeword_book.pt"),
        ("shaped", "shaped_physical_codeword_book.pt"),
    )
    if artifact != "both":
        selected_artifacts = tuple(
            item for item in selected_artifacts if item[0] == artifact
        )
    reports = {}
    for label, filename in selected_artifacts:
        payload = load_exported_physical_dd_artifact(
            root / filename, map_location=resolved_device,
        )
        scores = sparse_multipath_operator_separation_scores(
            payload["codeword_book"],
            pairs,
            bank,
            payload["data_mask"],
        )
        reports[label] = _summarize_breakdown(
            scores, pairs, bank_payload,
            target_margin=target_margin,
            top_k=top_k,
            tail_cvar_fraction=tail_cvar_fraction,
        )
    return {
        "schema_version": 1,
        "run_dir": str(root),
        "split": split,
        "pair_count": pair_count,
        "scenario_count": int(bank.channel.path_shifts.shape[0]),
        "scenarios_with_duplicate_dd_taps": duplicate_scenario_count,
        "target_margin": target_margin,
        "tail_cvar_fraction": tail_cvar_fraction,
        "flags": {
            "surrogate_only": True,
            "waveform_evaluation_performed": False,
            "ter_evaluated": False,
            "read_only_audit": True,
        },
        "artifacts": reports,
    }


def _summarize_breakdown(
    scores: torch.Tensor,
    pairs: torch.Tensor,
    bank_payload: dict,
    *,
    target_margin: float,
    top_k: int,
    tail_cvar_fraction: float,
) -> dict:
    outage = scores < target_margin
    pair_outage = outage.float().mean(dim=1)
    scenario_outage = outage.float().mean(dim=0)
    worst_pair_indices = torch.topk(
        pair_outage, k=min(top_k, pair_outage.numel()),
        largest=True, sorted=True,
    ).indices.cpu()
    worst_scenario_indices = torch.topk(
        scenario_outage, k=min(top_k, scenario_outage.numel()),
        largest=True, sorted=True,
    ).indices.cpu()
    report = {
        **summarize_separation_scores(
            scores, target_margin=target_margin,
        ),
        **summarize_tail_separation_scores(
            scores,
            target_margin=target_margin,
            tail_fraction=tail_cvar_fraction,
        ),
        "pair_outage_max": pair_outage.max().item(),
        "scenario_outage_max": scenario_outage.max().item(),
        "worst_pairs": [],
        "worst_scenarios": [],
    }
    for index in worst_pair_indices.tolist():
        report["worst_pairs"].append({
            "pair_index": index,
            "token_pair": pairs[index].tolist(),
            "outage_probability": pair_outage[index].item(),
            "separation_min": scores[index].min().item(),
            "separation_mean": scores[index].mean().item(),
        })
    for index in worst_scenario_indices.tolist():
        gains = bank_payload["path_gains"][index]
        report["worst_scenarios"].append({
            "scenario_index": index,
            "outage_probability": scenario_outage[index].item(),
            "separation_min": scores[:, index].min().item(),
            "separation_mean": scores[:, index].mean().item(),
            "has_duplicate_dd_taps": (
                torch.unique(
                    bank_payload["path_shifts"][index], dim=0,
                ).shape[0]
                != bank_payload["path_shifts"][index].shape[0]
            ),
            "path_shifts": bank_payload["path_shifts"][index].tolist(),
            "path_gains_real": gains.real.tolist(),
            "path_gains_imag": gains.imag.tolist(),
        })
    return report


def _count_scenarios_with_duplicate_dd_taps(path_shifts: torch.Tensor) -> int:
    return sum(
        torch.unique(row, dim=0).shape[0] != row.shape[0]
        for row in path_shifts
    )


def _read_json(path: Path) -> dict:
    with open(path, encoding="ascii") as handle:
        return json.load(handle)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit held-out TX-only sparse-DD tail behavior.",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("validation", "test"), default="test",
    )
    parser.add_argument(
        "--artifact", choices=("baseline", "shaped", "both"),
        default="both",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--tail-cvar-fraction", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = audit_tx_tail(
        args.run_dir,
        split=args.split,
        artifact=args.artifact,
        device=args.device,
        top_k=args.top_k,
        tail_cvar_fraction=args.tail_cvar_fraction,
    )
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
