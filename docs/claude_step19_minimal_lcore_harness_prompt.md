# Claude Prompt: Step 19 Minimal L_core Optimization Harness

Copy the prompt below into Claude Code from the workspace root:

```text
You are working in:
F:\OTFS_For_LLM-Com

Task name:
Step 19 - minimal standalone L_core optimization and audit-log harness.

Read the actual repository before editing. Do not rely only on this prompt.
First inspect at least:

- docs/otfs_token_tx_scope_freeze.md
- docs/otfs_token_operator_margin_theory.md
- docs/otfs_token_external_validation_protocol.md
- Learnable_Mapping_Tokens-to-DD-Signals/transmitter/codebook.py
- Learnable_Mapping_Tokens-to-DD-Signals/transmitter/config.py
- Learnable_Mapping_Tokens-to-DD-Signals/transmitter/pilot_frame.py
- Learnable_Mapping_Tokens-to-DD-Signals/transmitter/sampling.py
- Learnable_Mapping_Tokens-to-DD-Signals/transmitter/shaping.py
- Learnable_Mapping_Tokens-to-DD-Signals/transmitter/multipath_scenarios.py
- Learnable_Mapping_Tokens-to-DD-Signals/transmitter/multipath_margin.py
- Learnable_Mapping_Tokens-to-DD-Signals/transmitter/export.py
- Learnable_Mapping_Tokens-to-DD-Signals/transmitter/metrics.py
- Learnable_Mapping_Tokens-to-DD-Signals/transmitter/model.py
- tests/test_transmitter_step17c2_multipath_margin_loss.py

Goal:
Add one small standalone experiment harness that optimizes TokenDDCodebook
using the existing Step 17C2 L_core objective and emits auditable artifacts.
This step wires the already-frozen algorithm into a reproducible shaping run.
It is NOT external waveform validation and NOT a transmitter expansion.

Hard restrictions:

1. Do NOT modify any existing Python source file.
2. Do NOT modify transmitter/model.py, transmitter/__init__.py, receiver code,
   otfs_modem.py, or channel_model.py.
3. Do NOT add a transmitter module or expose new symbols from the transmitter
   package root.
4. Do NOT add any loss, regularizer, weighted objective term, CVaR variant,
   semantic weighting, UEP, resource allocation, PAPR term, fractional model,
   waveform channel model, receiver model, or TER calculation.
5. The optimization loop must use exactly one train objective:

       sparse_multipath_operator_margin_loss(...)

   Do not combine it with diagnostics. Diagnostics must run under
   torch.no_grad() and must not affect gradients.
6. Do NOT claim TER improvement, physical-channel robustness, or full OTFS
   validation. This step logs surrogate behavior only.
7. Keep implementation compact. Prefer existing transmitter APIs.
8. Use ASCII only in new source files.

Allowed file changes:

1. Create:
   Learnable_Mapping_Tokens-to-DD-Signals/experiments/__init__.py
2. Create:
   Learnable_Mapping_Tokens-to-DD-Signals/experiments/step19_lcore_harness.py
3. Create:
   tests/test_transmitter_step19_lcore_harness.py
4. Create:
   docs/step19_lcore_harness.md

Do not change any other file.

Required public surface inside experiments/step19_lcore_harness.py:

1. A frozen dataclass:

   Step19LCoreRunConfig

   It must contain only experiment-run controls, not transmitter
   architecture changes. Include at least:

   - seed
   - train_steps
   - train_pairs_per_step
   - eval_pairs
   - eval_every
   - num_scenarios
   - num_paths
   - target_margin
   - learning_rate
   - device

   Validate booleans vs integers carefully, require positive counts,
   finite non-negative target_margin, finite positive learning_rate,
   and a valid torch device string. CPU must work.

2. A diagnostic function:

   summarize_separation_scores(scores, *, target_margin) -> dict

   Input scores are real finite [P, R]. Return Python numeric values:

   - separation_mean
   - separation_min
   - separation_p01
   - outage_probability

   Use torch.quantile for P01. Outage means fraction of d[p,r] < gamma.
   This is diagnostics only.

3. A run function:

   run_step19_lcore_harness(
       tx_config,
       source_shift_set,
       output_dir,
       *,
       run_config,
   ) -> dict

   The function must:

   a. Require TransmitterConfig and SparseShiftSet inputs.
   b. Build masks with build_transmitter_pilot_masks(tx_config).
   c. Construct one TokenDDCodebook and initialize it reproducibly using the
      existing initialize_token_codebook_ utility with mode="random_phase".
      Use an explicit CPU torch.Generator. Move the initialized codebook to the
      requested device only after reproducible CPU initialization.
   d. Sample exactly one fixed scenario bank for the entire run using
      sample_normalized_sparse_multipath_scenario_bank. Use a dedicated CPU
      generator derived from the run seed.
   e. Sample one fixed evaluation pair set using a separate CPU generator.
      Evaluation pairs must not be reused as a fixed training set.
   f. Before optimization, export:
      - baseline_raw_state.pt
      - baseline_physical_codeword_book.pt
   g. Optimize only codebook.raw_real and codebook.raw_imag using
      torch.optim.Adam. For every training step:
      - sample fresh uniform ordered non-self token pairs with replacement via
        sample_uniform_cross_token_pairs using a dedicated training-pair CPU
        generator
      - compute physical codeword_book = codebook.forward(data_mask)
      - compute exactly one differentiable train loss using
        sparse_multipath_operator_margin_loss
      - zero gradients, backward, optimizer.step
      - never detach the train loss before backward
   h. Evaluate at baseline, every eval_every steps, and final. Evaluation uses
      the fixed eval pairs, the same fixed scenario bank, and torch.no_grad().
      Log:
      - phase and step
      - eval_l_core
      - separation_mean
      - separation_min
      - separation_p01
      - outage_probability
      - data_power_min
      - data_power_max
      - elapsed_seconds
   i. Compute eval_l_core by calling the existing public
      sparse_multipath_operator_margin_loss under torch.no_grad(). Do not
      reimplement a second margin formula.
   j. Compute separation diagnostics from the existing public
      sparse_multipath_operator_separation_scores.
   k. Compute data power through the existing data_codeword_power diagnostic.
   l. After optimization, export:
      - shaped_raw_state.pt
      - shaped_physical_codeword_book.pt
   m. Save:
      - scenario_bank.pt with path shifts, gains, active mask,
        source_shift_indices, name, and an explicit statement that sampled-bank
        downstream aggregation uses uniform implied scenario weights
      - metrics.jsonl with one JSON object per evaluation record
      - manifest.json with tx config, run config, derived seeds, artifact file
        names, baseline/final metrics, and explicit flags:
          surrogate_only: true
          waveform_evaluation_performed: false
          ter_evaluated: false
          transmitter_architecture_changed: false
   n. Return the manifest dictionary.

Seed discipline:

- Derive and record separate deterministic seeds for:
  codebook initialization, scenario bank generation, training pair sampling,
  and fixed evaluation pair sampling.
- Avoid hidden dependence on the global torch RNG after setup.
- The scenario bank is sampled once only and remains frozen.
- The sampled scenario bank has scenario_weights=None. q is used during shift
  sampling and must not be multiplied again during downstream aggregation.

CLI:

- Add a small argparse CLI in step19_lcore_harness.py.
- Defaults must form a valid, quick CPU smoke run.
- Construct the default TransmitterConfig and default_integer_shift_set in the
  CLI only.
- Print a concise JSON summary after completion.
- The module must be runnable from the workspace root using:

  $env:PYTHONPATH = ".\Learnable_Mapping_Tokens-to-DD-Signals"
  python -m experiments.step19_lcore_harness --output-dir ".\artifacts\step19_smoke"

Documentation requirements for docs/step19_lcore_harness.md:

- State clearly that Step 19 is a standalone surrogate-shaping harness, not
  waveform validation and not a new transmitter module.
- Document the single train objective and emitted artifacts.
- Document the CLI command.
- State that Tier 1 and Tier 2 external validation remain pending.
- Link to:
  - otfs_token_operator_margin_theory.md
  - otfs_token_external_validation_protocol.md
  - otfs_token_tx_scope_freeze.md

Test requirements for tests/test_transmitter_step19_lcore_harness.py:

Use unittest and temporary directories. Keep tests small and CPU-friendly.
At minimum verify:

1. summarize_separation_scores matches a manual small tensor calculation.
2. Invalid scores and invalid Step19LCoreRunConfig values fail fast.
3. A tiny CPU run creates all seven required artifacts:
   baseline_raw_state.pt
   baseline_physical_codeword_book.pt
   shaped_raw_state.pt
   shaped_physical_codeword_book.pt
   scenario_bank.pt
   metrics.jsonl
   manifest.json
4. The baseline and shaped exported physical books both preserve:
   - hard pilot/guard zeros
   - equal per-token data-region power
5. metrics.jsonl contains baseline and final records and all required fields.
6. manifest.json contains the explicit surrogate-only flags.
7. scenario_bank.pt records scenario_weights as None and fixed provenance.
8. Two tiny runs with the same seed produce identical baseline and shaped
   physical codeword books and deterministic metrics, excluding
   elapsed_seconds.
9. The scenario bank sampler is called exactly once per run.
10. Fresh training pair sampling is called once per training step, plus one
    separate fixed evaluation-pair sampling call.
11. The training path updates codebook parameters through autograd when the
    chosen tiny fixture has an active margin penalty.
12. transmitter package root exports remain unchanged; the harness lives under
    experiments only.
13. New Python files are ASCII-only and have module/function docstrings.

Do not require that every individual training step monotonically decreases
the loss. Do not claim TER improvement from this harness.

Verification:

1. Run the targeted test:

   python -m unittest tests.test_transmitter_step19_lcore_harness -v

2. Run the full root transmitter suite:

   python -m unittest discover -s tests -p "test_*.py"

3. Compile:

   python -m compileall Learnable_Mapping_Tokens-to-DD-Signals tests

4. Run the CLI smoke test into a temporary or clearly named output directory.
   Report baseline and final surrogate metrics without interpreting them as
   TER.

5. Confirm only the four allowed files changed. If __pycache__ files are
   generated, report them as generated artifacts and do not treat them as
   source changes.

Final report format:

1. Modified files
2. Standalone harness API
3. Single-objective training loop description
4. Reproducibility and fixed-scenario-bank discipline
5. Emitted artifacts
6. Targeted test result
7. Full test-suite result
8. compileall result
9. CLI smoke command and baseline/final surrogate metrics
10. Explicit statement that no existing Python file, transmitter module,
    receiver file, modem file, or channel model changed
11. Remaining limitations: no waveform evaluation, no TER evaluation,
    Tier 1/Tier 2 external validation still pending
```

## Review Gate

Do not start external waveform validation until the Step 19 result report and
actual file changes have been reviewed.
