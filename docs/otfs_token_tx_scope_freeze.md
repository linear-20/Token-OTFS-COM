# OTFS Token Transmission: Transmitter Scope Freeze

## Paper Focus

The paper studies direct token transmission over OTFS:

    token index
      -> equal-power physical DD codeword
      -> hard embedded pilot / guard insertion
      -> OTFS modulation
      -> sparse DD channel
      -> model-driven receiver
      -> token logits

The transmitter is intentionally compact. It is not a semantic resource
allocator, a bit/QAM redesign, or a collection of independent shaping modules.

## Core Transmitter Principle

Each token `v` is mapped directly to one physical complex DD codeword `C_v`.
All token codewords satisfy the same data-region power constraint. Pilot and
guard constraints are hard physical constraints.

The single core shaping principle is receiver-aligned sparse-operator-aware
separation:

    Delta_p = C_u - C_v

    d[p, r] = || M_e * H_r(Delta_p) ||_F^2 / |Omega_e|

    L_core = mean_p sum_r w[r] * relu(target_margin - d[p, r])^2

`H_r` is a fixed auditable sparse on-grid DD scenario operator. This objective
directly penalizes token pairs that become difficult to separate after sparse
multipath superposition, including coherent cancellation.

## Main Paper Contributions

1. Direct physical token-to-DD modulation for OTFS.
2. Equal-power token codewords with hard pilot and guard constraints.
3. Sparse-operator-aware DD codebook shaping using the single multipath
   separation-margin objective above.
4. A receiver-aligned token-logit link using physical codewords rather than
   raw transmitter parameters.

## Implementation Classification

### Core Path

- `config.py`, `pilot_frame.py`
- `codebook.py`, `model.py`
- `otfs_modem.py`, `modem_adapter.py`
- `sparse_multipath.py`, `multipath_scenarios.py`, `multipath_margin.py`

### Engineering Support

- `export.py`: lifecycle separation between raw TX state and physical book
- `metrics.py`: DD and time-domain diagnostics only
- `sampling.py`: scalable pair and scenario sampling utility
- structured initializers: initialization utility only

### Diagnostics Or Ablations Only

- orbit correlation
- self-shift sidelobe
- shift-orbit visibility
- hard-negative mining
- PAPR metric

These are not parallel main contributions and should not be combined into a
large default transmitter objective without ablation evidence.

The package boundary now enforces this distinction:

- `transmitter`: compact physical path and the Step 17C2 operator margin
- `transmitter.diagnostics`: physical metrics, including PAPR measurement
- `transmitter.ablations`: pre-17C2 orbit experiments and hard-negative tools

The unused pre-17C2 `SparseAwareShapingConfig` and
`SparseShapingDiagnostics` placeholders were removed.

## Frozen Items

Do not add transmitter modules for:

- fractional TX leakage extension
- fractional scenario bank
- PAPR regularizer
- a multi-term combined SATO objective
- semantic token weighting
- unequal error protection
- dynamic resource allocation

Receiver off-grid code may be reviewed for model understanding, but it does
not justify adding a transmitter-side fractional model by default.

## Claim Discipline

- `torch.roll` sparse DD paths are a receiver-aligned on-grid approximation,
  not a complete OTFS twisted-convolution channel.
- `d[p, r]` is a received-DD separation-energy proxy, not a posterior
  probability and not a TER guarantee.
- Waveform CP sufficiency and DD pilot-layout protection are separate checks:
  `cp_len >= L_delay` in time-domain samples for waveform experiments, while
  `pilot_guard_delay >= max_channel_delay + pilot_obs_delay_radius` protects
  the pilot observation window in DD bins.
- The paper contribution is the compact OTFS token-transmission formulation
  and operator-aware codebook shaping, not the number of implemented modules.

## Next Step

Architecture frozen. Step 18 produced two paper-usable documents:

- [Operator Margin Theory](otfs_token_operator_margin_theory.md) --
  formalizes A1--A8, PEP derivation, exact vs surrogate claims, and
  proof obligations before submission.
- [External Validation Protocol](otfs_token_external_validation_protocol.md) --
  defines Tier 1 (controlled consistency) and Tier 2 (mismatch
  robustness) evaluation with an experiment matrix, required metrics,
  and acceptance gates.

The next implementation step after Step 18 is minimal end-to-end wiring
of L_core with experiment logging, still without adding transmitter
modules, combined objectives, or multi-term losses.
