# OTFS Token Transmission: External Validation Protocol

**This is an external evaluation protocol, not a new transmitter module.**
No transmitter source files or tests are modified by this step.

## Evaluation Tiers

### Tier 1 -- Controlled Consistency

All channel realizations are on-grid integer delays and Dopplers with
sufficient CP (cp_len >= L_delay, the maximum discrete time-domain channel
memory in samples). When path delays use a zero-delay reference path, use the
conservative check cp_len >= ceil(max(path_delay_samples)). The noise
convention is explicitly recorded.

This CP check is separate from the DD pilot-layout guard condition:

```
pilot_guard_delay >= max_channel_delay + pilot_obs_delay_radius
```

**Goal**: Verify that the shaped codebook produces consistently higher
pairwise separation energy than the unshaped baseline, and that token
error rate moves in the same direction.

**Required comparisons**:
1. Baseline: equal-power unshaped codebook (random-phase or fixed
   initializer, no L_core optimization)
2. Shaped: codebook optimized with L_core (Step 17C2)
3. (Optional) Diagnostic-only ablations, clearly labeled as non-core

### Tier 2 -- Mismatch Robustness

Channel realizations introduce fractional delays, fractional Doppler /
IDI, pulse or channel mismatch when supported by the available channel
model. Sufficient CP vs intentionally insufficient CP are separate
experiments.

**Goal**: Quantify how much the on-grid surrogate advantage degrades
under model mismatch. The shaped codebook is not claimed to be optimal
under mismatch; the experiment measures degradation empirically.

**Required stress tests**:
1. Fractional delay (channel_model.py fractional_delays=True)
2. Fractional Doppler / IDI (when supported by simulation tooling)
3. Sufficient CP (baseline) vs insufficient CP (stress test)
4. Disjoint random channel draws not overlapping with scenario bank
5. SNR sweep covering at least 0--25 dB

## Experiment Matrix

| Row | Codebook       | Evidence       | Channel Type  | CP            | SNR Points | Seeds |
|-----|----------------|----------------|---------------|---------------|------------|-------|
| 1   | unshaped       | data-mask      | on-grid       | sufficient    | 5+ points  | 3+    |
| 2   | L_core-shaped  | data-mask      | on-grid       | sufficient    | 5+ points  | 3+    |
| 3   | unshaped       | full-grid      | on-grid       | sufficient    | 5+ points  | 3+    |
| 4   | L_core-shaped  | full-grid      | on-grid       | sufficient    | 5+ points  | 3+    |
| 5   | unshaped       | data-mask      | fractional    | sufficient    | 5+ points  | 3+    |
| 6   | L_core-shaped  | data-mask      | fractional    | sufficient    | 5+ points  | 3+    |
| 7   | unshaped       | data-mask      | on-grid       | insufficient  | 5+ points  | 3+    |
| 8   | L_core-shaped  | data-mask      | on-grid       | insufficient  | 5+ points  | 3+    |

Rows 1--2 are mandatory consistency checks.
Rows 3--8 are paired diagnostic / ablation rows. Keep unshaped and shaped
comparisons paired so degradation of the shaping advantage is measurable.
Additional rows may be added for specific analysis; label them clearly.

**Evidence mask note**: Rows 3--4 using full-grid evidence (M_e = all-ones on
[M,N]) is a diagnostic comparison. The data-mask evidence (excluding pilot
and guard) is the default simplified shared token-evidence projection. Do not
claim that it is universally superior; report the diagnostic comparison.

## Required Metrics

Per evaluation point (codebook + channel type + CP + SNR + seed):

- **Token error rate (TER)** -- fraction of incorrectly decoded tokens
- **Pairwise separation statistics**:
  - Mean d_uv,r over sampled pairs and scenarios
  - Minimum d_uv,r (worst-pair separation)
  - P01 separation (1st percentile over sampled pairs)
  - Outage probability: fraction of sampled pairs with d_uv,r < gamma
- **Convergence**: L_core value vs optimization steps
- **Resource**: wall-clock time per iteration, peak GPU memory

**Reporting discipline**:
- Always report separation statistics alongside TER.
- Improved surrogate loss (lower L_core) alone is NOT sufficient
  evidence of TER improvement; both must be reported.
- When TER improves, report which separation statistics improved
  and by how much.

## Reporting Requirements

### Configuration Record

Every experiment must record:
- M, N, vocab_size V
- CP length (in samples)
- Maximum channel delay (in DD bins and samples)
- Waveform CP check: cp_len >= L_delay in samples
- DD pilot-layout check:
  pilot_guard_delay >= max_channel_delay + pilot_obs_delay_radius
- Maximum channel Doppler (in DD bins and Hz)
- SNR convention: define sigma_c^2 explicitly (e.g., per-DD-bin noise
  variance before or after evidence projection)
- Evidence mask: data_mask or full_grid, with shape
- Path count K per scenario
- Scenario count R in training bank
- Target margin gamma
- Random seeds for: codebook initialization, pair sampling, scenario
  bank generation, channel draws
- Optimizer, learning rate, iteration count

### Train / Test Separation

- Train scenario bank: drawn once, fixed, used for L_core optimization
- Test channels: drawn from DISJOINT random generator states, never
  used during shaping
- Train token pairs: sampled uniformly per iteration (no fixed train
  pair set)
- Report results on test channels only; training-set results are for
  convergence monitoring

### Limitations Statement

Every result presentation must include:
- "The on-grid surrogate H_r is a receiver-aligned approximation,
  NOT a complete OTFS twisted-convolution channel."
- "d_uv,r is a separation-energy proxy with monotonic PEP motivation
  under assumptions A1--A8. It is NOT a posterior probability and NOT
  a TER guarantee."
- "Improved L_core does not imply improved TER under model mismatch;
  empirical verification is required."
- Record and state the CP configuration explicitly.

## Acceptance Gates

| Gate | Condition | Status |
|------|-----------|--------|
| G1 | Tier 1: shaped L_core <= unshaped L_core, and mean/P01 separation do not regress | Required |
| G2 | Measurable TER improvement OR clearly diagnosed failure in Tier 1 | Required |
| G3 | No hidden UEP (verify gamma is same for all pairs) | Required |
| G4 | CP configuration recorded and A3 verified | Required |
| G5 | Paired unshaped-vs-shaped off-grid degradation quantified (Tier 2) and stated alongside on-grid results | Required |
| G6 | All claims mapped to evidence table with row references | Required |

Failure of G2 does not invalidate the approach; it requires a diagnosis
section explaining why separation improvement did not translate to TER
and what conditions would be needed.
