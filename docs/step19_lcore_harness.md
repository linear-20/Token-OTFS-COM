# Step 19: Standalone L_core Surrogate Shaping Harness

**This is a standalone surrogate-shaping harness, NOT waveform validation
and NOT a new transmitter module.**

## Purpose

Optimizes a TokenDDCodebook using the single Step 17C2 objective:

```
L_core = mean_{(u,v)} sum_r w[r] * relu(gamma - d_uv,r)^2
```

Exports auditable artifacts: baseline and shaped codebooks, the fixed
scenario bank, per-evaluation metrics, and a manifest with explicit
surrogate-only flags.

## Single Train Objective

Exactly one differentiable loss is used for training:

```python
loss = sparse_multipath_operator_margin_loss(
    cw, train_pairs, scenario_bank, evidence_mask,
    target_margin=gamma,
)
```

No other loss term, regularizer, or weighting is added. Diagnostics
(evaluation scores, separation statistics, data power) run under
`torch.no_grad()` and do not affect gradients.

## Emitted Artifacts

| File | Description |
|------|-------------|
| `baseline_raw_state.pt` | Raw learnable params before training |
| `baseline_physical_codeword_book.pt` | Physical codeword book before training |
| `shaped_raw_state.pt` | Raw learnable params after training |
| `shaped_physical_codeword_book.pt` | Physical codeword book after training |
| `scenario_bank.pt` | Fixed scenario bank (shifts, gains, mask, provenance) |
| `metrics.jsonl` | One JSON object per evaluation record |
| `manifest.json` | Configuration, seeds, baseline/final metrics, surrogate-only flags |

## CLI

From the workspace root:

```
$env:PYTHONPATH = ".\Learnable_Mapping_Tokens-to-DD-Signals"
python -m experiments.step19_lcore_harness --output-dir ".\artifacts\step19_smoke"
```

All arguments have defaults that produce a valid quick CPU smoke run.

## Reproducibility

- The base seed derives four deterministic CPU generators for:
  codebook initialization, scenario bank generation, training pair
  sampling, and fixed evaluation pair sampling.
- The scenario bank is sampled once and never modified.
- The scenario bank has `scenario_weights=None`; q is used during
  shift sampling and is not multiplied again downstream.
- The evaluation pair set is sampled once and fixed.

## Pending External Validation

- Tier 1 (controlled consistency) and Tier 2 (mismatch robustness)
  external validation remain pending per
  [otfs_token_external_validation_protocol.md](otfs_token_external_validation_protocol.md).
- This harness evaluates surrogate metrics only; it does NOT perform
  waveform simulation, TER evaluation, or physical channel evaluation.

## Related Documents

- [Operator Margin Theory](otfs_token_operator_margin_theory.md)
- [External Validation Protocol](otfs_token_external_validation_protocol.md)
- [Transmitter Scope Freeze](otfs_token_tx_scope_freeze.md)
