# Claude Prompt: Step 18 Theory Closure And Validation Protocol

Copy the prompt below into Claude Code from the workspace root:

```text
You are working in:
F:\OTFS_For_LLM-Com

Task name:
Step 18 - paper-level theory closure and external validation protocol for
direct OTFS token transmission.

Read the actual repository before editing. Do not rely only on this prompt.
First inspect at least:

- docs/otfs_token_tx_scope_freeze.md
- Learnable_Mapping_Tokens-to-DD-Signals/transmitter/codebook.py
- Learnable_Mapping_Tokens-to-DD-Signals/transmitter/config.py
- Learnable_Mapping_Tokens-to-DD-Signals/transmitter/pilot_frame.py
- Learnable_Mapping_Tokens-to-DD-Signals/transmitter/sparse_multipath.py
- Learnable_Mapping_Tokens-to-DD-Signals/transmitter/multipath_scenarios.py
- Learnable_Mapping_Tokens-to-DD-Signals/transmitter/multipath_margin.py
- Learnable_Mapping_Tokens-to-DD-Signals/otfs_modem.py
- channel_model.py
- Learnable_Receiver_DD-Signals-to-Tokens/receiver/dd_ops.py
- Learnable_Receiver_DD-Signals-to-Tokens/receiver/token_prior.py

Goal:
Close the mathematical claim boundary around the existing compact transmitter.
Do not expand the transmitter architecture. Produce paper-usable documentation
and an auditable experimental protocol. This is a theory and validation step,
not a new modeling step.

Hard restrictions:

1. Do NOT modify Python source files or tests in this step.
2. Do NOT add transmitter modules, losses, regularizers, combined objectives,
   model.py wiring, training code, fractional Doppler transmitter extensions,
   fractional scenario banks, semantic weighting, unequal error protection,
   dynamic resource allocation, or PAPR regularization.
3. Do NOT claim that torch.roll is the full physical OTFS channel.
4. Do NOT claim TER/BER optimality, posterior equivalence, or end-to-end
   robustness guarantees.
5. Use ASCII in new files unless a mathematical symbol is genuinely clearer.
6. Keep the argument compact. The paper needs one core shaping idea, not a
   collection of unrelated modules.

Create exactly these two files:

1. docs/otfs_token_operator_margin_theory.md
2. docs/otfs_token_external_validation_protocol.md

Update exactly this existing file:

3. docs/otfs_token_tx_scope_freeze.md

Required content for docs/otfs_token_operator_margin_theory.md:

A. State the paper focus:
direct token index -> equal-power physical DD codeword -> hard pilot/guard ->
CP-OTFS waveform -> sparse doubly selective channel -> model-driven token
receiver.

B. Define the existing receiver-aligned on-grid sparse surrogate:

    H_r(X) = sum_k a[r,k] h[r,k] Pi_{delta[r,k]}(X)

where Pi is the receiver-aligned integer DD circular shift. State explicitly:
this is an on-grid CP-OTFS surrogate aligned with the implemented receiver,
not complete pulse-dependent OTFS twisted convolution and not an off-grid
fractional Doppler model.

C. Define:

    Delta_uv = C_u - C_v
    d_uv,r = || M_e H_r(Delta_uv) ||_F^2 / |Omega_e|

Explain that M_e is a shared hard binary projection applied after H_r and
Omega_e is its active support. It represents the simplified shared receiver
token-evidence region. It is not a learned confidence map and not a posterior.

D. State explicit assumptions for the exact PEP interpretation:

- A1: equal-power token codewords on the data support
- A2: CP-OTFS waveform convention
- A3: cyclic prefix length is at least the maximum discrete delay spread when
  using the circular-shift surrogate
- A4: on-grid sparse integer delay-Doppler shifts for the surrogate derivation
- A5: a fixed projected channel H_r and shared hard evidence projection M_e
- A6: equal token priors and pairwise ML Euclidean comparison
- A7: projected complex AWGN n ~ CN(0, sigma_c^2 I), where
  E[|n_i|^2] = sigma_c^2
- A8: scenario gains are normalized so sum_k a[r,k]|h[r,k]|^2 = 1; this
  isolates channel direction and coherent cancellation geometry from
  amplitude/SNR scaling

E. Give a short derivation, not merely a statement:

    z = M_e H_r(C_u) + n
    P(u -> v | H_r)
      = Q( || M_e H_r(C_u - C_v) ||_F / sqrt(2 sigma_c^2) )
      = Q( sqrt( |Omega_e| d_uv,r / (2 sigma_c^2) ) )

Explain that the exact constant depends on the stated complex-noise
convention. The monotonic relationship is the key design justification.

F. Define the existing Step 17C2 objective exactly:

    L_core = mean_(u,v) sum_r w_r relu(gamma - d_uv,r)^2

Explain:
- gamma is one global physical hyperparameter
- no per-token margin
- no semantic weighting
- no UEP
- no resource allocation
- scenario weights enter aggregation once only
- for sampled banks, q is already used during sampling and the downstream
  implied weights are uniform, avoiding double weighting

G. Separate exact claims from surrogate claims:

Exact under A1-A8:
- d_uv,r is normalized projected received separation energy
- conditional pairwise ML PEP is monotone decreasing in d_uv,r
- L_core penalizes pair-scenario cases below the global separation target

Surrogate-only outside A1-A8:
- estimated channels
- receiver confidence/uncertainty weighting
- fractional delay
- fractional Doppler and IDI
- pulse mismatch
- insufficient CP and resulting ISI
- generalization from sampled scenarios to unseen channels

H. Include a brief "why this is not module stacking" section. The novelty
claim is the compact formulation of equal-power direct token-to-DD codebook
learning as sparse-operator-induced robust packing. Do not market diagnostics
or ablations as parallel contributions.

I. Include a "proof obligations before submission" checklist:
- verify CP constraint in all waveform experiments
- report the exact noise convention
- justify the evidence projection and compare data-mask vs full-grid evidence
- isolate scenario-direction shaping from external amplitude/SNR evaluation
- evaluate disjoint train/validation/test channel draws
- test off-grid and model-mismatch robustness externally
- inspect tail metrics before changing the mean aggregation:
  P01 separation, worst-pair separation, outage probability

Required content for docs/otfs_token_external_validation_protocol.md:

1. State that this is an external evaluation protocol, not a new TX module.
2. Define two evaluation tiers:

Tier 1 - controlled consistency:
- on-grid integer delays and Dopplers
- sufficient CP
- known noise convention
- compare baseline equal-power codebook against Step 17C2-shaped codebook
- verify separation statistics and token error behavior move consistently

Tier 2 - mismatch robustness:
- fractional delay
- fractional Doppler / IDI
- pulse or channel mismatch when supported
- sufficient CP vs intentionally insufficient CP as separate experiments
- disjoint random channel draws and SNR sweep

3. Require an experiment matrix with:
- baseline equal-power unshaped codebook
- Step 17C2 L_core-shaped codebook
- optional diagnostic-only ablations clearly labeled as non-core
- data-mask evidence vs full-grid evidence diagnostic comparison
- on-grid vs off-grid evaluation
- sufficient CP vs insufficient CP stress test
- multiple SNR points
- multiple random seeds

4. Require metrics:
- token error rate
- pairwise separation distribution
- minimum separation
- P01 separation
- outage probability below gamma
- average separation
- convergence curve
- runtime and memory

5. Require reporting discipline:
- distinguish train scenario bank from external test channels
- record M, N, CP length, maximum delay, maximum Doppler, SNR convention,
  evidence mask, path count, scenario count, target margin gamma, seeds
- never interpret improved surrogate loss alone as proof of TER improvement
- state limitations plainly

6. Add a concise table of acceptance gates. At minimum:
- no regression in Tier 1 consistency
- measurable TER improvement or clearly diagnosed failure
- no hidden UEP
- CP configuration recorded and checked
- off-grid degradation quantified
- all claims mapped to evidence

Required update to docs/otfs_token_tx_scope_freeze.md:

- Preserve the architecture freeze.
- Replace the old Next Step section with links to the two new Step 18
  documents.
- State that the next implementation step after Step 18 is minimal
  end-to-end wiring of L_core with experiment logging, still without adding
  transmitter modules.

Primary references:

- Embedded pilot OTFS:
  https://arxiv.org/abs/1808.08360
- OTFS integer/fractional Doppler and IDI:
  https://arxiv.org/abs/1802.05242
- OTFS PEP/diversity:
  https://arxiv.org/abs/1808.07747
- Zak-OTFS twisted convolution analysis:
  https://arxiv.org/abs/2302.08705

Use these references narrowly and accurately. Do not invent claims or cite
unverified papers. If you cannot verify a statement from repository code or
the references, mark it as an assumption or limitation.

Verification:

1. Confirm only the three allowed Markdown files changed.
2. Search the new text for forbidden overclaims such as:
   "full OTFS channel", "TER guarantee", "BER guarantee", "optimal", and
   "posterior equivalent". If a phrase appears, it must be explicitly negated
   or qualified.
3. Confirm the formula contains |Omega_e| in the PEP expression.
4. Confirm the CP >= maximum discrete delay spread condition is explicit.
5. Do not run or modify the test suite unless necessary to prove that no code
   changed.

Final report format:

1. Modified files
2. Mathematical assumptions A1-A8
3. Exact claims vs surrogate-only claims
4. PEP formula copied from the new document
5. External validation tiers and acceptance gates
6. Verification commands and results
7. Explicit statement that no Python source, tests, or transmitter
   architecture changed
8. Remaining uncertainties requiring later implementation or experiment
```

## Review Gate

Do not start Step 19 implementation until the Step 18 report has been reviewed
against the restrictions and formulas above.
