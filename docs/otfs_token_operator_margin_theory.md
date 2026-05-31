# OTFS Token Transmission: Operator Margin Theory

## Paper Focus

The paper studies direct token transmission over OTFS:

```
token index --> equal-power physical DD codeword --> hard pilot/guard
            --> CP-OTFS waveform --> sparse doubly selective channel
            --> model-driven token receiver
```

Each token index v is mapped directly to one physical complex DD codeword C_v
with shape [M, N]. The complete codeword book has shape [V, M, N]. All
codewords satisfy the same data-region equal-power constraint
(config.data_power). Pilot and guard insertion is a hard physical constraint,
not a soft loss term.

## Receiver-Aligned On-Grid Sparse Surrogate

The existing receiver (dd_circular_convolve_sparse) uses on-grid integer DD
shifts via torch.roll. The transmitter-side operator H_r is aligned to this
convention:

```
H_r(X) = sum_{k=1}^{K} a[r,k] * h[r,k] * Pi_{delta[r,k]}(X)
```

where:
- Pi_{delta} is the receiver-aligned integer DD circular shift using
  torch.roll along dims (-2, -1) matching the receiver's dims (0, 1) per
  DD grid element
- a[r,k] in {0, 1} is a strict boolean active gate (True = active,
  False = padded/inactive), NOT a soft confidence or learnable reliability
- h[r,k] are fixed complex path gains (requires_grad=False)
- delta[r,k] = (delay_shift[r,k], doppler_shift[r,k]) are integer shifts

**Explicit qualification**: This is an on-grid CP-OTFS surrogate aligned
with the implemented receiver. It is NOT complete pulse-dependent OTFS
twisted convolution and NOT an off-grid fractional Doppler model.

The receiver-side operator is `dd_circular_convolve_sparse` in
`receiver/dd_ops.py`. Verification through our receiver contract tests
(Step 17A) confirms numerical equivalence between the transmitter-side
`apply_sparse_multipath_dd_operator` and the receiver's
`dd_circular_convolve_sparse` for on-grid paths.

## Pairwise Separation Energy

For token pair (u, v) and multipath scenario r:

```
Delta_uv = C_u - C_v

d_uv,r = || M_e * H_r(Delta_uv) ||_F^2 / |Omega_e|
```

where:
- M_e is a shared hard binary evidence mask applied AFTER H_r
- Omega_e = {(m,n) | M_e[m,n] = 1} is the active evidence support
- |Omega_e| is the number of active DD grid bins in the evidence region
- M_e is NOT a learned confidence map and NOT a posterior probability

The linearity of H_r is exploited: H_r(C_u - C_v) = H_r(C_u) - H_r(C_v),
avoiding redundant operator applications.

## Explicit Assumptions for Pairwise Error Probability Interpretation

Under the following assumptions, d_uv,r admits a monotonic relationship to
the conditional pairwise ML error probability.

- **A1 -- Equal-power tokens**: All physical token codewords satisfy
  mean(|C_v|^2)_{data region} = config.data_power. Verified by the hard
  binary data-mask projection and per-token equal-power normalization in
  TokenDDCodebook.forward.

- **A2 -- CP-OTFS waveform convention**: Waveform experiments use the
  multicarrier CP-OTFS convention implemented by the deterministic modem in
  otfs_modem.py. ISFFT uses ifft(dim=2, ortho), fft(dim=1, ortho). OFDM
  modulation uses ifft(dim=1, ortho), transpose, and per-symbol CP insertion.

- **A3 -- Sufficient cyclic prefix**: The CP length satisfies
  cp_len >= L_delay, where L_delay is the maximum discrete time-domain channel
  memory measured in samples. When path delays are measured relative to a
  zero-delay reference path, a conservative check is
  cp_len >= ceil(max(path_delay_samples)). This is an experiment-harness
  obligation; it is NOT validated by TransmitterConfig.__post_init__.
  Sufficient CP prevents delay-induced inter-symbol interference under the
  waveform convention, but it does NOT make the torch.roll surrogate a
  complete physical OTFS input-output law.

  Do not conflate this waveform condition with the separately validated pilot
  layout condition:

  ```
  pilot_guard_delay >= max_channel_delay + pilot_obs_delay_radius
  ```

  The latter protects the pilot observation window in DD bins; it is not a CP
  length requirement.

- **A4 -- On-grid integer delay-Doppler shifts**: The surrogate H_r uses
  only integer (delay, Doppler) shifts. Fractional delay and fractional
  Doppler / inter-Doppler interference (IDI) lie outside the surrogate
  model.

- **A5 -- Fixed projected channel and shared evidence projection**: Each
  scenario r provides a fixed channel operator H_r (gains are frozen,
  requires_grad=False). The evidence mask M_e is a shared hard binary
  projection applied identically to all pairs. It is typically the
  transmitter data_mask, excluding pilot and guard regions.

- **A6 -- Equal priors and pairwise ML comparison**: Token priors are
  uniform. The exact PEP interpretation considers an ideal pairwise ML
  Euclidean comparison in the projected DD domain with a known physical
  codeword book. This motivates the model-driven receiver, but it does NOT
  claim that every implemented receiver path is exactly ML.

- **A7 -- Projected complex AWGN**: For a single DD grid element after
  projection by M_e, the noise is complex AWGN:
  n_i ~ CN(0, sigma_c^2), with E[|n_i|^2] = sigma_c^2.
  The noise is white and independent across active DD bins in the ideal
  projected PEP model. With pure time-domain white AWGN, ideal CP removal, and
  unitary modem transforms, whiteness is preserved. Receiver-side nonidealities
  or mismatch processing that introduce colored effective noise lie outside
  the exact PEP claim.

- **A8 -- Unit active-path-power normalization**: For every scenario r,
  sum_k a[r,k] * |h[r,k]|^2 = 1. This isolates channel direction and
  coherent-cancellation geometry from overall amplitude/SNR scaling.
  Verified by SparseMultipathScenarioBank.__post_init__.

## Pairwise Error Probability Derivation (Short)

Consider transmitting C_u and receiving z = M_e H_r(C_u) + n, where
n ~ CN(0, sigma_c^2 I) on the |Omega_e| active DD bins.

The ML receiver compares projected distances. For the pair (u, v), the
conditional pairwise error probability under Gaussian noise is:

```
P(u -> v | H_r)
  = Q( || M_e H_r(C_u) - M_e H_r(C_v) ||_F / sqrt(2 sigma_c^2) )
  = Q( || M_e H_r(C_u - C_v) ||_F / sqrt(2 sigma_c^2) )
  = Q( sqrt( |Omega_e| * d_uv,r / (2 sigma_c^2) ) )
```

where Q(.) is the Gaussian Q-function, monotonically decreasing in its
argument.

**Critical observation**: P(u -> v | H_r) is a monotonically DECREASING
function of d_uv,r. Therefore, increasing pairwise separation energy d_uv,r
reduces the conditional PEP. The exact PEP constant depends on the specific
complex-noise convention, but the monotonic relationship is the key design
justification for the margin objective.

**Limitation**: This is a conditional pairwise PEP under a single fixed
H_r, NOT a union bound, NOT an average over random channels, and NOT a
full TER guarantee. It applies to the ideal projected white-AWGN model.
Estimated channels, mismatch processing, or colored effective noise require
external evaluation. The derivation serves as monotonic motivation, not as a
provable TER bound for the implemented end-to-end receiver.

## Core Shaping Objective

The Step 17C2 objective directly operationalizes the monotonic relationship:

```
L_core = mean_{(u,v)} sum_{r=1}^{R} w[r] * relu(gamma - d_uv,r)^2
```

where:
- gamma is one global physical hyperparameter (target_margin >= 0)
- w[r] = scenario_bank.normalized_weights() are fixed physical scenario
  weights (real, non-negative, sum to 1, requires_grad=False)
- d_uv,r is from sparse_multipath_operator_separation_scores
- relu(x) = max(0, x)

Key properties:
- No per-token margin: gamma is the same for all token pairs
- No semantic weighting: w[r] depends only on physical scenario statistics
- No unequal error protection
- No dynamic resource allocation
- Scenario weights enter aggregation ONCE ONLY
- For sampled banks (scenario_weights=None), the downstream normalized
  weights are uniform, because the sampling distribution q has already
  been used during multinomial scenario generation. This avoids double
  weighting.

## Exact Claims vs. Surrogate-Only Claims

### Exact under A1--A8

- d_uv,r is the normalized projected received separation energy for
  token pair (u, v) under scenario r
- Conditional pairwise ML PEP is monotone decreasing in d_uv,r
- L_core penalizes pair-scenario cases where separation falls below
  the global target gamma
- Scenario gains are frozen and non-learnable; the only learnable
  parameters are the raw token codewords (raw_real, raw_imag)

### Surrogate-only (outside A1--A8)

- Estimated (rather than known) channels
- Receiver confidence / uncertainty weighting
- Fractional delay
- Fractional Doppler and inter-Doppler interference (IDI)
- OTFS pulse mismatch
- Insufficient CP (resulting ISI)
- Generalization from sampled scenarios to unseen channels
- Any TER/BER guarantee

## Why This Is Not Module Stacking

The novelty claim is the compact formulation of equal-power direct
token-to-DD codebook learning as sparse-operator-induced robust packing:

1. **Single shaping principle**: receiver-aligned separation-energy
   hinge margin under fixed sparse multipath scenarios.

2. **One learnable component**: the raw token codeword parameters
   (raw_real, raw_imag). Everything else is fixed, auditable,
   non-learnable: pilot/guard layout, OTFS modulation, scenario
   shifts, scenario gains, evidence mask, CP length.

3. **Single loss term**: L_core, with one hyperparameter (gamma).
   Orbit correlation, self-shift sidelobe, and visibility are
   diagnostic / ablation tools, not parallel main contributions.

4. **No module stacking**: Diagnostic tools (orbit correlation,
   visibility, hard-negative mining) observe the codebook from
   different angles; they are not summed into a multi-term objective.
   The paper should not market diagnostics or ablations as
   independent contributions unless ablation evidence clearly
   supports their addition.

## Proof Obligations Before Submission

- [ ] Verify CP constraint (A3) in all waveform experiments
- [ ] Report the exact noise convention (sigma_c^2 definition)
  used in every PEP plot and table
- [ ] Justify the evidence projection M_e: compare data-mask evidence
  vs full-grid evidence as a diagnostic; do not silently claim one is
  always better
- [ ] Isolate scenario-direction shaping from external amplitude/SNR
  evaluation: keep scenario gain normalization fixed and report SNRs
  using the external channel model, not the surrogate
- [ ] Evaluate on disjoint train / validation / test channel draws;
  report the split criteria
- [ ] Test off-grid and model-mismatch robustness externally (fractional
  delay, fractional Doppler) and report degradation on the same metric
  axes
- [ ] Inspect tail metrics before changing the mean aggregation:
  P01 separation (1st percentile), worst-pair separation, outage
  probability below gamma

## References

- P. Raviteja, K. T. Phan, and Y. Hong, "Embedded Pilot-Aided Channel
  Estimation for OTFS in Delay-Doppler Channels," arXiv:1808.08360.
  -- embedded pilot and guard arrangements
- P. Raviteja et al., "Interference Cancellation and Iterative Detection
  for Orthogonal Time Frequency Space Modulation," arXiv:1802.05242.
  -- integer/fractional Doppler and IDI
- G. D. Surabhi, R. M. Augustine, and A. Chockalingam, "On the Diversity
  of Uncoded OTFS Modulation in Doubly-Dispersive Channels,"
  arXiv:1808.07747. -- diversity analysis
- S. K. Mohammed et al., "OTFS -- Predictability in the Delay-Doppler
  Domain and its Value to Communication and Radar Sensing,"
  arXiv:2302.08705. -- Zak-OTFS predictability context only; this does not
  identify the repository's multicarrier CP-OTFS modem with Zak-OTFS

These references are used narrowly. Claims not directly verifiable from
repository code or the cited references are explicitly marked as
assumptions or limitations.
