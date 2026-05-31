# OTFS Token Communication: Learnable Receiver -- Technical Report

## Abstract

We present a learnable receiver for token-based communication over Orthogonal
Time Frequency Space (OTFS) modulation.  Unlike conventional OTFS systems that
map bits to QAM symbols and perform separate channel estimation, equalization,
and decoding, the proposed receiver directly maps Delay-Doppler (DD) domain
received signals to token logits through a fixed chain of model-based and
data-driven stages.  The receiver interfaces with the channel exclusively
through a parametric sparse DD operator H_theta that captures the dominant
K-path time-varying multipath response as a sum of weighted circular shifts,
with optional approximate off-grid leakage kernels.  The channel-coupled
detection modules -- unfolded detector and data-consistency correction -- use
only the `apply` and `matched_filter` interfaces of this operator, avoiding the
construction of dense MN-by-MN channel matrices.  The denoiser, classifier,
and logit fusion consume the resulting DD estimates, reliability maps, and
token-level evidence.  Token posterior complexity is controlled via data-adaptive sketch
candidate pruning (O(V*S + Kc*M*N) vs. O(V*M*N) for full-vocab, where Kc
is the number of selected candidates).  The fixed 10-stage chain is fully
ablatable, each module controlled by configuration flags.  For the
receiver-only regime, the trainable parameter count is dominated by the
embedding codebook [V, D].

---

## 1. Introduction & System Model

### 1.1 Token Communication over OTFS

Consider a token vocabulary of size V.  The transmitter maps a discrete token
index t in {0,...,V-1} to a complex DD-domain signal C_t in C^{M x N}, where M
denotes the number of delay bins and N the number of Doppler bins.  The DD
codeword book C in C^{V x M x N} is shared between transmitter and receiver.

The transmitted DD signal S_t includes both the token codeword and an embedded
pilot P (a known complex scalar at a known DD position, surrounded by a guard
region of zeros):

```
S_t = P + M_d * C_t
```

where M_d in {0,1}^{M x N} is a binary data mask (1 on data-bearing positions,
0 on pilot and guard positions), and * denotes element-wise multiplication.
For the intended trainable transmitter, token codewords should be normalized or
regularized to satisfy an equal-power constraint:

```
(1/|Omega_d|) * sum_{(m,n) in Omega_d} |C_t[m,n]|^2 = E_s
```

where Omega_d = {(m,n) : M_d[m,n] = 1} is the data region.

The signal traverses a time-varying multipath channel.  **Channel model
assumption:** We adopt a K-sparse 2D circular convolution model in the DD
domain:

```
H_theta(X)[m,n] = sum_{k=1}^{K} g_k * X[(m - l_k)_M, (n - nu_k)_N]
```

where (a)_b denotes modulo-b circular indexing.  Each path k is parameterized
by integer delay l_k, integer Doppler shift nu_k, and complex gain g_k.  This
model assumes that (i) the channel is sparse in DD, (ii) fractional delay and
Doppler are either negligible or absorbed into an approximate leakage kernel
(see Section 3.4), and (iii) the standard OTFS phase terms e^{j*phi_k(m,n)}
are absorbed into the complex gain g_k.  **This is a simplified model**;
a fully rigorous OTFS twisted convolution includes path-dependent phase
rotation terms.  We adopt the simplified model as our baseline and note that
the modular architecture can accommodate a more precise operator without
changing the detection/classification chain.

The received DD signal Y in C^{M x N} is:

```
Y = H_theta(S_t) + W = H_theta(P + M_d * C_t) + W
```

where W is additive white circularly symmetric complex Gaussian noise with
variance sigma^2 per dimension.

The receiver's task is to recover the token index t from Y.  It has access to
the known pilot P and mask M_d, but the channel H_theta (including K, the
individual l_k, nu_k, g_k) is unknown and must be estimated from the pilot
observation window.  The receiver does NOT separately demodulate bits or QAM
symbols -- it produces token logits directly.

**Pilot handling in RX-v1:** The frozen receiver implementation consumes the
DD observation tensor `y_dd` exactly as it is passed to
`LearnableOTFSReceiver.forward`.  It does not internally synthesize and
subtract `H_theta(P)`.  Instead, the embedded pilot is used for sparse channel
estimation, while `data_mask`, `pilot_mask`, and `guard_mask` restrict
token-posterior distances, classifier evidence, and data-consistency weighting
to data-bearing DD positions.  If a future experiment wants to operate on a
pilot-subtracted observation, that subtraction should be performed by the TX/RX
experiment harness before calling the frozen receiver, or introduced as a
separate versioned receiver change.

### 1.2 Notation

| Symbol | Shape | Meaning |
|---|---|---|
| M, N | scalar | Delay bins, Doppler bins |
| B | scalar | Batch size |
| K | scalar | Number of sparse DD paths |
| V | scalar | Token vocabulary size |
| D | scalar | Token embedding dimension |
| L | scalar | Number of unfolded equalizer layers |
| H | scalar | Number of classifier evidence heads |
| Y | [B, M, N] cplx | Received DD grid |
| X | [B, M, N] cplx | Estimated DD signal |
| h_k, l_k, nu_k, g_k | -- | k-th path parameters |
| H_theta | operator | Sparse DD operator |
| C | [V, M, N] cplx | Token DD codeword book |
| E | [V, D] real | Token embedding codebook |

---

## 2. Receiver Architecture Overview

The receiver chain is a fixed 10-stage pipeline.  Each stage consumes tensors
from the previous stage and produces tensors consumed by the next.  The chain
is:

```
Y_DD [B, M, N]
  -> 1.  Embedded Pilot Sparse Channel Estimation
  -> 2.  Off-grid Sparse Path Refinement
  -> 3.  Physics-Guided Gain Re-estimation
  -> 4.  Parametric Sparse DD Operator H_theta / H_theta^H
  -> 5.  Variance-Aware Token Posterior Unfolded Detector
  -> 6.  Adaptive Candidate-Pruned Token Posterior Prox
  -> 7.  Confidence/Symbol-Reliability-Gated Residual Denoiser
  -> 8.  Data-Consistency Correction
  -> 9.  Multi-Head Evidence Token Classifier
  -> 10. Detector-Classifier Logit Fusion
  -> token_logits [B, V]
```

Stages 1-4 constitute the **Channel Estimation (CE) segment**, which estimates
the sparse DD channel and constructs the parametric operator H_theta.
Stages 5-8 constitute the **Detection segment**, which iteratively refines an
estimate of the transmitted DD signal X.
Stages 9-10 constitute the **Classification segment**, which maps the refined
DD estimate to token logits.

Every stage that interacts with the channel does so through the shared
`SparseDDOperator` interface (apply for forward convolution, matched_filter
for adjoint correlation).  No stage constructs a dense MN-by-MN matrix.

---

## 3. Channel Estimation Segment (Stages 1-4)

### 3.1 Embedded Pilot Sparse CE (Stage 1)

**Principle:** An embedded pilot -- a known complex scalar p at a known DD
position (l_p, nu_p) -- is inserted into the transmit DD grid.  The pilot is
surrounded by a guard region of zero symbols to prevent interference from
token data.  The receiver extracts a local observation window of size
P-by-Q around the pilot position:

```
Y_obs = Y[l_p - R_d : l_p + R_d, nu_p - R_nu : nu_p + R_nu]
```

Dividing by the known pilot value yields a noisy observation of the
channel response at positions offset from the pilot:

```
H_obs[Delta_l, Delta_nu] = Y[(l_p + Delta_l)_M, (nu_p + Delta_nu)_N] / p
```

for all `|Delta_l| <= obs_delay_radius`, `|Delta_nu| <= obs_doppler_radius`.
The detected path coordinates (l_k, nu_k) = (Delta_l, Delta_nu) are
expressed as **offsets relative to the pilot position**.  These relative
coordinates are the correct inputs to the sparse DD operator, which models
path-induced circular shifts.

Sparse on-grid paths are detected by thresholding |H_obs| with a
CFAR-style criterion:

```
threshold = cfar_scale * sqrt(noise_var)
path_set = {(l_k, nu_k) : |H_obs[l_k, nu_k]| >= threshold}
```

if a threshold is supplied, else by top-K magnitude selection.  Each detected
path is assigned an initial gain estimate g_k = H_obs[l_k, nu_k] and a
normalized confidence:

```
c_k = (|g_k| / max_j |g_j|)^{1 / temp}
```

**Key design choice:** The pilot observation window must lie entirely within
the guard region, ensuring that only pilot energy contributes to the channel
estimate.  This constraint is enforced by the `require_obs_within_guard` flag.

### 3.2 Off-Grid Path Refinement (Stage 2)

**Principle:** Real-world propagation paths have fractional (non-integer)
delay and Doppler shifts.  After on-grid detection, a small CNN
(`OffGridRefinementNet`) processes local DD patches around each detected
path to predict sub-bin fractional offsets:

```
delta_l_k, delta_nu_k = CNN(h_dd_patch_k)
```

The offsets are constrained to [-max_offset, max_offset] with
max_offset <= 0.5 by construction.  Low-confidence paths are gated out
to avoid refining noise.

### 3.3 Physics-Guided Gain Re-estimation (Stage 3)

**Principle:** The initial gain estimates from threshold-based CE are noisy.
A model-based ridge LS/MMSE re-estimation is performed using the sparse DD
operator dictionary.  For each path, the expected channel response to the
pilot impulse is computed via the sparse DD operator, forming a dictionary
D in C^{M*N x K}.  The LS problem is:

```
g_refined = argmin_g || W_fit * (y_vec - D @ g) ||^2 + ridge * ||g - g_init||^2
```

where W_fit in {0,1}^{MN x MN} is a diagonal binary mask selecting only the
CE observation (pilot window) region.  Writing the objective with W_fit^{1/2}:

```
||W_fit^{1/2} (y_vec - D @ g)||^2 + ridge * ||g - g_init||^2
```

The closed-form solution is:

```
g_refined = (D^H @ W_fit @ D + ridge * I)^{-1} @ (D^H @ W_fit @ y_vec + ridge * g_init)
```

This is ridge-regularized weighted least squares.  If a probabilistic
interpretation is desired, it corresponds to a Gaussian prior
g ~ CN(g_init, ridge^{-1} * I) with a Gaussian likelihood having
precision W_fit / sigma^2.  We refer to it as **ridge LS** to avoid
implying a strict MMSE interpretation.

**Critical safety design:** The CE fitting mask W_fit must be restricted to
the pilot observation window.  It is **never** the full token data region.
This is enforced by default (`physics_refinement_require_fit_mask=True`),
and the mask source is traced (pilot_observation, explicit_ce_data_mask,
or none).  The full-grid residual is computed only for diagnostics; the
confidence update uses only the fit-masked residual power:

```
fit_residual_power = sum(W_fit * |y - D @ g|^2) / sum(W_fit)
confidence_k = |g_k|^2 / (|g_k|^2 + fit_residual_power + covariance_diag_k)
```

### 3.4 Parametric Sparse DD Operator (Stage 4)

**Principle:** The estimated path parameters (l_k, nu_k, g_k, delta_l_k,
delta_nu_k) parameterize a sparse linear operator H_theta : C^{M x N} ->
C^{M x N}.  The operator is constructed as a parametric module
(`SparseDDOperator`) supporting two operations:

**Forward apply (H_theta):**
```
H_theta(X)[m, n] = sum_{k=1}^{K} g_k * K_{l_k, nu_k}^{delta}(X)[m, n]
```

where K is either an integer circular shift (ongrid mode) or a local
leakage-kernel convolution (offgrid mode).  For on-grid integer paths:

```
H_theta(X) = sum_{k=1}^{K} g_k * roll(X, shifts=(l_k, nu_k))
```

**Matched filter (H_theta^H):** The adjoint operation for on-grid paths:

```
H_theta^H(Y) = sum_{k=1}^{K} conj(g_k) * roll(Y, shifts=(-l_k, -nu_k))
```

For off-grid paths, a differentiable leakage kernel (linear or truncated sinc)
distributes the path response over a local P x Q neighborhood:

```
kernel(l, nu) = w_delay(l - l_k - delta_l_k) * w_doppler(nu - nu_k - delta_nu_k)
```

The off-grid forward operator H_off applies this kernel at each path position.
The matched-filter (adjoint) H_off^H is constructed by applying the same
kernel weights with conjugated gains and reverse shifts, analogous to the
on-grid case.  Formally, the adjoint must satisfy:

```
<H_off(X), Y> = <X, H_off^H(Y)>
```

for all X, Y in C^{M x N}, where <A, B> = sum_{m,n} conj(A_{m,n}) * B_{m,n}.
This property is verified by the adjoint consistency check in the physics
validation harness (Section 8.1), which requires relative error < 1e-5.
In the implementation, H_off^H is NOT constructed heuristically -- it is
built as the exact transpose of the discrete leakage stencil used in the
forward operator H_off.  For the special case of zero fractional offsets,
the off-grid operator reduces to the on-grid operator and the adjoint is
exact.

**Complexity:** Each apply or matched_filter operation costs O(K * M * N),
where K is typically 3-10.  No dense MN x MN matrix is ever materialized.

**Theoretical justification:** The DD channel of a time-varying multipath
environment is sparse in the delay-Doppler representation -- only K << M*N
combinations of (delay, Doppler) carry significant energy.  The sparse
operator approximates the true channel with K dominant paths, where K is a
hyperparameter.  The approximation error decreases as K increases, but the
computational cost scales linearly with K.  In practice, K = topk_paths is
set based on the expected channel sparsity level.

---

## 4. Detection Segment (Stages 5-8)

### 4.1 Variance-Aware Unfolded Detector (Stage 5)

**Principle:** The detector recovers an estimate X_hat of the transmitted DD
signal X from the receiver input observation Y using the estimated channel
operator H_theta.
The detector is an L-layer unfolded iterative soft interference cancellation
(unfolded model-based denoising) framework, with L1 soft-thresholding
as the sparsity-inducing proximal step and a Bayesian token posterior
denoiser as the structured prior.  The design is inspired by UAMP/BPIC
principles.

**Initialization:**
```
X_0 = H_theta^H(Y) / (sum_k |g_k|^2 + noise_var)
```

where Y denotes the DD observation tensor supplied to `forward`.  RX-v1 does
not internally compute a pilot-subtracted observation.

**Denominator:**  The denominator `sum_k |g_k|^2 + noise_var` approximates
the diagonal of H_theta^H @ H_theta.  When the K paths have approximately
orthogonal shifts (non-overlapping DD supports), this is exact.  For
general non-orthogonal shifts or off-grid leakage kernels, `sum_k |g_k|^2`
may underestimate the spectral norm of H_theta^H @ H_theta.  The current
implementation uses the per-path-power denominator

```
denom = sum_k |g_k|^2 + noise_var
```

A more conservative spectral bound such as `(sum_k |g_k|)^2 + noise_var`, or
a power-iteration estimate of the norm of H_theta^H H_theta, is a possible
future robustness enhancement but is not part of RX-v1.

**Per-layer update (layer index l):**

```
r_l     = Y - H_theta(X_l)                           # residual
grad_l  = H_theta^H(r_l) / denom                     # matched-filter gradient
z_l     = X_l + alpha_l * grad_l                     # damped update
X_{l+1} = D(z_l, threshold_l, prior)                 # denoising step
```

where alpha_l in (0, 1) is a learnable damping factor, and D(.) is one of:

- **L1 prox:** Complex soft-thresholding.
  ```
  prox_l1(z) = z * relu(|z| - tau) / (|z| + eps)
  ```

- **Token posterior denoiser/projection (D_token):** Projects z onto the token DD
  codeword space via a Bayesian posterior mean.  Given a token DD codeword
  book C in C^{V x M x N} and a probabilistic model:

  ```
  z = C_t + epsilon,  epsilon ~ CN(0, sigma_z^2 * W^{-1})
  ```

  where W = diag(w_{m,n}) with w_{m,n} = data_mask * confidence /
  (1 + uncertainty), the posterior probability of token v is:

  ```
  p(t=v | z) propto exp(-(1/sigma_z^2) * sum_{m,n} w_{m,n} * |z_{m,n} - C_{v,m,n}|^2)
  ```

  With temperature T playing the role of sigma_z^2, the posterior weights are:

  ```
  p_v = softmax(-d_v / T),  d_v = sum_{m,n} w_{m,n} * |z_{m,n} - C_{v,m,n}|^2
  ```

  The denoised signal is the posterior mean:

  ```
  D_token(z) = E[C_t | z] = sum_{v} p_v * C_v
  ```

  **Note:** This is a Bayesian posterior mean denoiser, NOT a strict
  proximal operator in the sense of convex optimization.  We use the
  notation D_token rather than prox_token to avoid implying a proximal
  mapping onto a convex set.

- **Hybrid step:** A convex combination of L1 soft-thresholding and token
  posterior denoising:
  ```
  X_{l+1} = (1 - beta_l) * X_{l1} + beta_l * D_token(z_l)
  ```
  where beta_l is a learnable interpolation weight.

**Variance Tracking:** Each layer estimates the residual variance and uses it
to adaptively control:

1. **Posterior temperature:** tau_l = base_temp * residual_variance_l.

   When residual variance is high (uncertain estimate), the posterior
   temperature increases, making the token posterior distribution more
   uniform -- the detector relies more on L1 sparsity than on token prior.

2. **Detector damping:** Damping_l scales beta_l, preventing over-commitment
   to the token posterior when the channel estimate is uncertain.

   Damping is modulated by the active path confidence:

   ```
   residual_variance_l = residual_power_l / mean(active_confidence)
   ```

   This closes the loop between channel estimation quality (CE confidence)
   and detection robustness.

**Theoretical connection:** The unfolded detector is an instance of
proximal gradient descent on the composite objective:

```
min_X  ||Y - H_theta(X)||^2 / (2*sigma^2) + lambda * R(X)
```

where R(X) is either the L1 norm (sparsity prior) or the negative
log-posterior of the token codeword model.  The per-layer variance tracking
plays an AMP-inspired adaptive damping and temperature-control role
(residual variance modulates posterior temperature; path confidence
modulates detector damping).  However, it does NOT explicitly implement
the Onsager correction term of AMP, which involves the divergence of the
denoiser.  The design is more accurately described as **variance-aware
learned damping**.

### 4.2 Adaptive Candidate-Pruned Token Posterior Prox (Stage 6)

**Principle:** The full-vocab token posterior projection requires computing
distances to all V codewords at all M*N DD positions -- O(V*M*N) per layer.
For practical vocab sizes (V = 1024) and grid sizes (M = 32, N = 32), this is
approximately 1 million complex multiply-adds.  Candidate pruning reduces this
to O(V*S + Kc*M*N) where S is a small sketch size (default 64) and Kc is the
number of selected candidates (default 3-10).

**Sketch-based candidate selection (select_token_candidates_by_sketch):**

1. **Position selection:** Choose S DD grid positions via one of three modes:

   - **strided:** Fixed deterministic stride positions [S] (baseline).
   - **energy_topk:** Top-S positions by |z|^2 * data_mask * confidence/(1+uncertainty).
   - **hybrid:** Half strided + half energy_topk, deduplicated, padded to S.

2. **Scoring:** At the S selected positions, compute per-codeword scores using
   one of three score modes:

   - **distance:** score_v = -sum_s w_s * |z_s - C_{v,s}|^2
   - **corr:** score_v = Re(sum_s conj(C_{v,s}) * z_s) / (|z| * |C_v|)
   - **energy_weighted_distance:** w_s = data_mask_s * confidence_s / (1 + uncertainty_s),
     with a reliability floor on nonzero weights and uniform fallback if all
     sketch weights are zero,
     score_v = -sum_s w_s * |z_s - C_{v,s}|^2 / sum_s w_s

3. **Selection:** Take top-K candidates by score.  Optionally, compute
   candidate entropy and margin diagnostics, and adaptively expand the
   candidate set if the selector is uncertain (entropy > threshold or
   margin < threshold).

4. **Posterior denoising over candidates:** Only the K selected codewords
   participate in the full DD-domain distance computation.  Unselected
   codewords are touched only on the S sketch positions during candidate
   scoring; they are not evaluated over the full M*N grid.

**Complexity analysis:**

```
Full vocab:  O(V * M * N) per denoising call
Sketch:      O(V * S) for scoring + O(Kc * M * N) for denoising
             S = min(M*N, 64), Kc = candidate_count (3-10)
```

For V=1024, M=32, N=32: full vocab = 1,048,576 ops; sketch = 65,536 + ~10,240
= 75,776 ops -- a ~14x reduction.

**Adaptive expansion:** If the top-K candidate scores have high normalized
entropy (> 0.85) or low top1-top2 margin (< 0.05), the candidate set is
expanded (K_new = ceil(K_old * expand_factor), capped by max_candidates).
This provides a safety net when the sketch selector is uncertain, without
requiring full-vocab computation in the common case.

**Theoretical justification:** When the sketch positions Omega_S are drawn
uniformly at random (without replacement) from the M*N DD positions, the
sketch estimate:

```
d_hat_v = (MN/S) * sum_{s in Omega_S} w_s * |z_s - C_{v,s}|^2
```

is an unbiased estimator of the full weighted distance d_v.  By Hoeffding's
inequality for bounded random variables, when the margin between the K-th
and (K+1)-th true distances is nonzero, the probability of a top-K ordering
error decreases exponentially with the sketch size S.

For the **strided** sketch, the positions are deterministic, trading
probabilistic guarantees for implementation simplicity.

For **energy_topk** and **hybrid** sketches, the position selection is
data-adaptive (based on |z|^2 * weight).  These are heuristics designed
to handle the case where codeword energy is concentrated in few DD
positions that strided sampling might miss.  They do not satisfy the
independent-sampling assumption required by JL-type bounds.  Their
effectiveness is validated empirically through the **candidate recall
metric** (Section 9.3): the fraction of samples for which the true token
appears in the top-K candidates, measured on held-out data.

### 4.3 Confidence-Gated Residual DD Denoiser (Stage 7)

**Principle:** The model-based equalizer output X_eq contains residual errors
from channel estimation inaccuracies and the approximate nature of the off-grid
leakage kernel.  A lightweight CNN acts as a data-driven denoiser:

```
X_refined = X_eq + delta
delta = gate * CNN_head(CNN_encoder(input_channels))
```

where input_channels = 12 (real/imag of X_eq, Y, H; abs of X_eq and H; support
mask; SNR map; confidence map; uncertainty map).

**Gating mechanism:**
```
support_gate = 0.5 + sigmoid(conv1(support))
confidence_gate = (0.5 + sigmoid(conv1(conf))) * (1.25 - 0.75*conf)
uncertainty_gate = (0.5 + sigmoid(conv1(unc))) * (0.75 + unc)
features = features * support_gate * confidence_gate * uncertainty_gate
correction_gate = sigmoid(gate_head(features)) * (0.25 + unc)
```

The gating follows a "trust the model where it is confident" principle:
- **Confidence gate:** 1.25 - 0.75*conf decreases with confidence.
  High-confidence estimates need LESS feature modulation (the equalizer
  already did a good job).
- **Uncertainty gate:** 0.75 + unc increases with uncertainty.
  High-uncertainty estimates get MORE feature processing (the denoiser
  needs to work harder).
- **Correction gate:** 0.25 + unc increases with uncertainty.
  High-uncertainty estimates allow LARGER delta corrections.

In summary: the denoiser is more aggressive where the model-based detector
is uncertain, and more conservative where the detector is confident.  The
block is identity-initialized (delta_head and gate_head weights set to
zero at initialization).

**Architecture:** Two 3x3 Conv2d layers with SiLU activation, followed by a
complex delta head (2-channel real+imag output) and a scalar gate head.  Total
parameters: a few thousand.

**Theoretical justification:** The residual error of model-based detectors
often has structure that can be learned from data.  The CNN denoiser operates
on a 12-channel feature map that encodes both the signal estimate and the
channel/uncertainty context, allowing it to learn spatially-local corrections
conditioned on channel reliability.

### 4.4 Data-Consistency Correction (Stage 8)

**Principle:** The CNN denoiser may produce an estimate X_refined that no
longer satisfies the channel equation.  A single model-based gradient step
restores data consistency.  RX-v1 uses the receiver input observation Y and
weights the objective by the data-region/reliability mask:

```
f(X) = (1/2) * ||W^{1/2} (H_theta(X) - Y)||^2
```

where W = data_mask * reliability / (1 + uncertainty).  The gradient is:

```
nabla f(X) = H_theta^H(W * (H_theta(X) - Y))
```

The correction step is therefore:

```
Y_hat    = H_theta(X_refined)
r        = Y_hat - Y
obs_w    = data_mask * reliability / (1 + uncertainty)
r_w      = obs_w * r
g        = H_theta^H(r_w)                          # weighted LS gradient
gate     = scalar_gate                              # learnable x-domain gate
X_cons   = X_refined - step * gate * g
```

Since obs_w contains the data mask, pilot and guard positions do not contribute
to the weighted LS gradient.  RX-v1 does not internally subtract H_theta(P);
pilot-response subtraction, if desired, should be handled outside the frozen
receiver or added in a versioned receiver update.

Two learnable parameters: step (via softplus) and gate (via sigmoid), both
initialized near zero so the block defaults to identity.

**Semantic separation:** The observation weight obs_w is used ONLY within
H_theta^H(obs_w * r) -- the weighted LS gradient.  The update gate is a
separate scalar that controls the correction magnitude in the x-domain.
This separation prevents the observation weight from incorrectly suppressing
the correction in high-confidence regions.

**Connection to Plug-and-Play (PnP):** This is a single iteration of a PnP
scheme: data-driven denoiser (Stage 7) as the "prior" step, followed by
model-based data-consistency (this stage) as the "forward model" step.

---

## 5. Classification Segment (Stages 9-10)

### 5.1 Multi-Head Evidence Token Classifier (Stage 9)

**Principle:** The task is to classify X_cons in C^{M x N} into one of V token
classes.  This is done via multi-head evidence pooling followed by codebook
cosine similarity.

**Input channels:** Stack of 12 real channels:
- Real(X), Imag(X), |X| (3 channels)
- DD position encoding (6 sinusoidal channels: delay, Doppler, sin/cos)
- Support mask, confidence map, uncertainty map (3 channels)

**Feature encoder:** 2-layer CNN (3x3 Conv + SiLU), producing hidden features
F in R^{hidden x M x N}.

**Multi-head evidence:**
```
local_emb    = Conv2d(hidden -> D, kernel=1)(F)      # [B, D, M, N]
evidence_raw = Conv2d(hidden -> H, kernel=1)(F)      # [B, H, M, N]
evidence_score = evidence_raw/T + log(support) + log(conf)
                 - uncertainty                        # [B, H, M, N]
a_h         = softmax_{m,n}(evidence_score_h)         # [B, H, M, N]
```

**Per-head embedding:**
```
e_h = sum_{m,n} a_{h,m,n} * local_emb_{:,m,n}        # [B, H, D]
```

**Per-head codebook similarity:**
```
head_logits_h = normalize(e_h) @ normalize(E)^T / temperature   # [B, H, V]
```

**Head fusion:**
```
token_logits = sum_h pi_h * head_logits_h             # [B, V]
```

where the fusion weights pi_h depend on the fusion mode:
- **mean:** pi_h = 1/H (uniform)
- **learned_static:** pi = softmax(head_fusion_logits) (learnable scalar per head)
- **confidence:** pi_h = (1/entropy_h) / sum_j (1/entropy_j) (low-entropy heads weighted higher)

**Why multi-head?** A single evidence head pools the entire DD grid into one
D-dimensional vector.  If the DD codeword has multiple spatially separated
clusters of energy, a single softmax can only focus on one cluster.  Multiple
heads allow different heads to attend to different clusters, and the fusion
combines their decisions.

**Theoretical connection:** This is a form of "mixture of experts" where each
head is an expert in a different spatial region of the DD grid.  Unlike
self-attention, the heads are independent -- there is no cross-head
interaction, which keeps the complexity at O(H*M*N*D + H*V*D).

### 5.2 Detector-Classifier Logit Fusion (Stage 10)

**Principle:** The unfolded detector produces token posterior logits (from
the last layer's token posterior projection).  The classifier produces
independent token logits.  These two sources of token-level evidence are fused
at the logit level:

```
fused_logits = classifier_logits + gate * detector_logits_centered
```

**Gate modes:**

1. **static:** gate = sigmoid(raw_gate) * scale (learnable scalar)
2. **reliability:** gate = sigmoid(softplus(a) * detector_score + b) * scale

   where detector_score combines:
   ```
   detector_score = posterior_confidence - posterior_entropy
                  + reliability_scalar
                  - classifier_confidence + classifier_entropy
   ```

The gate is initialized near zero (raw_b = -3), so the system starts as
classifier-only and learns to incorporate detector information over training.

**Detector logit handling:** If detector logits are top-K [B, K], they are
expanded to full vocab [B, V] via:

```
l_v^{det} = { -d_v / T,        if v in C
            { b_miss,            if v not in C
```

In the current implementation, we use `b_miss = 0` after per-sample
centering, which avoids incorrectly suppressing un-candidate tokens
without requiring a learned baseline.  Setting `b_miss = -inf` would
incorrectly bias the fused posterior toward top-K candidates.
A more principled approach is a learnable `b_miss` trained as
`min_{v in C} l_v^{det} - delta`; this is left for future work.
The per-sample mean of the detector logits is subtracted before fusion
(centering), ensuring that only the *relative* confidence among candidates
influences the gate.

**Theoretical justification:** The detector and classifier provide
complementary evidence: the detector is model-based (constrained by H_theta)
while the classifier is data-driven (learned codebook).  The fusion gate,
especially in reliability mode, enables the system to weigh these sources
based on per-sample uncertainty estimates.

---

## 6. Uncertainty Propagation Architecture

A distinguishing feature of this receiver is the systematic propagation of
uncertainty estimates through all stages.  The uncertainty chain is:

```
CE confidence [B, K]
  -> path_confidence_map [B, M, N]
    -> detector_reliability_map
      -> variance tracking -> posterior temperature -> damping
      -> denoiser confidence/uncertainty gating
      -> DC correction obs_weight and update_gate
      -> classifier evidence prior (log confidence - uncertainty)
      -> candidate pruning sketch weight
    -> symbol_reliability_map
      -> classifier confidence_map
      -> denoiser conditioning
  -> fusion gate (reliability mode)
  -> reliability_diagnostics (calibration)
```

This architecture ensures that every downstream decision is informed by the
quality of upstream estimates, enabling graceful degradation under poor channel
conditions rather than catastrophic failure.

---

## 7. Token DD Codeword Book and Pilot Design

### 7.1 Token DD Codeword Book C [V, M, N]

The token DD codeword book C in C^{V x M x N} is the central shared
representation between transmitter and receiver.  Each row C_v in C^{M x N}
is a complex-valued DD-domain signal that encodes token index v.

**Critical distinction:** The receiver uses TWO codebooks, serving different
purposes:

| Codebook | Shape | Domain | Used by |
|---|---|---|---|
| DD Codeword Book C | [V, M, N] complex | Delay-Doppler | Detector posterior denoiser, DD codeword loss, sketch scoring |
| Embedding Codebook E | [V, D] real | Abstract embedding | Classifier cosine similarity |

The DD codeword book is stored in `TokenCodewordPrior.codeword_book` and
shared with the transmitter.  It represents the actual DD signal that the
transmitter would emit for each token.  The embedding codebook is stored in
`TokenEmbeddingClassifier.token_codebook` and is purely a receiver-side
learned embedding for classification.  **They must never be confused** --
the DD codeword book [V, M, N] cannot be used as the embedding codebook
[V, D], and vice versa.

When the TX side is trainable, the DD codeword book becomes the learnable
mapping from tokens to DD signals: X = C[token_id].

### 7.2 TokenCodewordPrior

`TokenCodewordPrior` is a lightweight container that wraps the DD codeword
book and provides auxiliary parameters used by the unfolded detector's token
posterior denoiser:

```
TokenCodewordPrior(
    codeword_book: complex [V, M, N]   -- DD token codewords
    prior_strength: float              -- projection strength in [0, inf)
    temperature: float                 -- posterior temperature
    similarity: "real" | "abs"         -- correlation mode for prior
)
```

It does not contain learnable parameters itself; it is a configuration +
data object that the detector references.

### 7.3 Embedded Pilot Layout

The transmitter embeds a known complex pilot p at DD position (l_p, nu_p)
surrounded by a guard region of zeros.  The layout is defined by
`EmbeddedPilotConfig`:

```
pilot_delay, pilot_doppler    -- pilot position
guard_delay, guard_doppler     -- zero guard radii
obs_delay_radius, obs_doppler_radius -- observation window radii
pilot_value = p_real + j * p_imag
```

The DD grid is partitioned into three disjoint regions:

```
pilot_mask[m, n] = 1  iff (m, n) == (l_p, nu_p)         -- exactly one cell
guard_mask[m, n] = 1  iff |m-l_p| <= guard_delay
                        and |n-nu_p| <= guard_doppler
                        and (m, n) != (l_p, nu_p)         -- zero-protected area
data_mask[m, n]  = 1  iff pilot_mask=0 and guard_mask=0   -- token data region
```

The observation window is the subset of the guard region used for CE:

```
obs_window = { (m, n) : |m-l_p| <= obs_delay_radius
                     and |n-nu_p| <= obs_doppler_radius }
```

The constraint `obs_delay_radius <= guard_delay` (with
`require_obs_within_guard=True`) ensures that the observation window lies
entirely within the zero-guard region, preventing token data from leaking
into the channel estimate.

### 7.4 data_mask / pilot_mask / guard_mask Semantics

These three masks are passed through the entire receiver and consumed by:

| Consumer | Masks Used | Purpose |
|---|---|---|
| Sparse CE | pilot_mask, guard_mask | Derives data_mask; restricts CE observation |
| Physics refinement | ce_data_mask (obs window) | Restricts LS fitting to known pilot region |
| Detector | data_mask, confidence | Weights token posterior distance |
| Denoiser | support_mask (from CE) | Gates residual correction |
| DC correction | data_mask | Weights observation-domain gradient |
| Classifier | data_mask, confidence, uncertainty | Evidence prior computation |
| Candidate pruning | data_mask, confidence, uncertainty | Sketch position selection and scoring |

Masks can be provided as [B, M, N] or [1, M, N] (broadcast) and are always
real-valued in [0, 1].

---

## 8. Validation and Harness Infrastructure

The receiver includes three auxiliary subsystems that support paper-level
evaluation without modifying the main detection/classification algorithms.

### 8.1 Physics Validation (physics_validation.py)

Validates that the sparse DD operator implementation is mathematically
consistent.  Key metrics:

**Integer operator NMSE:** Compares `H_theta.apply(x)` against an independent
`torch.roll` reference for on-grid paths.  Must be numerically zero.

**Adjoint consistency:** Verifies `<Hx, y> = <x, H^H y>` (complex inner
product).  The relative error must be below 1e-5 for on-grid operators.

**Off-grid zero-offset:** When fractional offsets are exactly zero, the
off-grid operator output must match the on-grid operator output within 1e-6
NMSE.

**Leakage diagnostics:** For an impulse input at a known DD position, measures
output/input energy ratio, centroid shift, and peak position.  Detects kernel
blow-up or systematic bias.

**Pilot alignment:** Constructs a probe signal from the CE's path estimates
and compares `H(probe)` against the received Y_DD within the pilot observation
mask only.

**Gain refinement residual ratio:** `physics_fit_residual_power /
pilot_residual_power` -- quantifies whether LS re-estimation actually reduces
residual.

### 8.2 Paper Harness (paper_harness.py)

Provides paper-level reporting and experiment management:

- `summarize_receiver_config(config)` -- extracts paper-relevant config keys
  into a flat dictionary for experiment logging.
- `count_receiver_parameters(model)` -- per-module parameter count
  (classifier, denoiser, equalizer, offgrid, fusion, other).
- `estimate_receiver_ops(config)` -- analytic rough operation count per
  sample for sparse operator, detector, token posterior projection (full vs sketch),
  classifier, and data consistency.
- `build_receiver_paper_trace(output, config)` -- extracts tensor shapes,
  enabled module flags, scalar metrics, and algorithm flags from a
  `ReceiverOutput` for experiment logging.
- `named_receiver_paper_ablation(name, config)` -- 14 ablation configurations
  controlled purely by config flag changes.
- `receiver_paper_summary(model, config, output)` -- combined report
  aggregating config, complexity, ops, and trace.

### 8.3 Freeze Manifest (freeze_manifest.py)

Declares the frozen receiver interface for the TX development phase:

- `RECEIVER_FREEZE_VERSION = "rx_v1"`
- `FROZEN_RECEIVER_CHAIN`: canonical ordering of all 12 stages
- `FROZEN_PUBLIC_APIS`: 36 public API items
- `FROZEN_TX_RX_CONTRACT`: specifies that TX must produce X_DD [B,M,N]
  complex from token indices, sharing the DD codeword book C [V,M,N].
  Documents data_mask/pilot_mask/guard_mask semantics and forbidden
  architecture components (bit/QAM, Transformer, GNN, VAE, dense MNxMN).
- `FROZEN_BASELINE_ABLATIONS`: 14 ablation names

### 8.4 ReceiverOutput Structure

The full forward pass with `return_details=True` produces a `ReceiverOutput`
dataclass carrying all intermediate tensors for analysis and deep supervision:

| Field | Shape | Description |
|---|---|---|
| token_logits | [B, V] | Fused final logits |
| rx_embedding | [B, D] | Fused classifier embedding |
| sparse_estimate | SparseChannelEstimate | h_dd, support, paths, gains, confidence, offgrid offsets, physics residuals |
| operator_state | SparseDDOperatorState | Paths and mode for H_theta |
| x_equalized | [B, M, N] cplx | Equalizer output |
| x_refined | [B, M, N] cplx | Denoiser output |
| x_consistent | [B, M, N] cplx | After DC correction |
| equalizer_output | EqualizerOutput | Per-layer estimates, residuals, variances, posterior logits/weights/indices, candidate diagnostics |
| aux_logits | list of [B, V] | Per-layer classifier logits |
| denoiser_output | DenoiserOutput | delta [B,M,N], confidence/uncertainty maps |
| classifier_output | TokenClassifierOutput | Head-level embeddings/logits/evidence/fusion |
| logit_fusion_output | TokenLogitFusionOutput | Fused logits, classifier/detector logits, gate |
| classifier_token_logits | [B, V] | Pre-fusion classifier logits |
| detector_token_logits | [B, V] | Expanded detector logits |
| reliability_diagnostics | ReliabilityDiagnostics | Classifier confidence/entropy, posterior entropy, error proxy |
| receiver_trace | dict | All configuration and scalar metric flags |

---

## 9. Loss Function Design

The training objective is a weighted sum of multiple loss components, each
controlled by a scalar weight (defaulting to 0 except token CE = 1.0):

### 9.1 Primary objective

```
L_token_ce = CrossEntropy(token_logits, token_ids)
```

### 9.2 Model-based consistency losses

```
L_dd_codeword = MSE(X_refined, M_d * codeword_book[token_ids])
L_data_consistency = weighted MSE(H_theta(X_refined), Y; mask=M_d)
L_physics_ce = (pilot_residual + fit_residual + cov_diag) / 3
```

### 9.3 Detection auxiliary losses

```
L_aux_ce = mean(CrossEntropy(aux_logits[l], token_ids))
L_posterior_nll = top-K cross-entropy with missing penalty
```

### 9.4 Regularization losses

```
L_denoiser_delta = weighted L2 norm of delta (high-reliability regions penalized)
L_classifier_head_diversity = mean pairwise overlap between head evidence distributions
L_posterior_entropy = MSE(normalized entropy, target)
L_candidate_margin = relu(target - (top1_prob - top2_prob)).mean()
L_logit_fusion_consistency = symmetric KL(classifier_softmax || detector_softmax)
```

### 9.5 Calibration losses

```
L_calibration = binned |confidence - accuracy| proxy
L_reliability_alignment = MSE(reliability_scalar, p_true)
```

All components are optional; the default loss is pure token CE, ensuring
backward compatibility with simpler training setups.

---

## 10. Complexity Analysis

### 10.1 Parameter count

| Module | Parameters |
|---|---|
| Off-grid refinement net | ~500 |
| Equalizer (L layers) | ~4L |
| Denoiser CNN | ~3000 |
| DC correction | 2 |
| Classifier CNN + heads | ~D*H + hidden*D + hidden*H + 2000 |
| Token codebook | V * D |
| Token fusion | 3 |
| CE / DD operator | 0 (non-parametric) |
| **Total (RX trainable)** | **~V*D + 6000** |

For V=1024, D=128: ~137K RX-only parameters, dominated by the classifier
embedding codebook [V, D].

**Note on end-to-end training:** When the transmitter-side DD codeword book
C in C^{V x M x N} (stored in `TokenCodewordPrior.codeword_book`) is jointly
trained with the receiver, the total trainable parameter count becomes:

```
2 * V * M * N  +  V * D  +  ~6000
```

For V=1024, M=32, N=32, D=128: 2*1024*1024 + 131,072 + 6000 ~= 2.23M
parameters, dominated by the complex DD codeword book (each complex entry
requires two real parameters).  This distinction between receiver-only and
end-to-end parameter regimes is critical for fair complexity comparisons.

### 10.2 Per-sample computational complexity

| Operation | Complexity |
|---|---|
| DD operator (apply / matched_filter) | O(K * M * N) |
| Detector (L layers) | O(L * K * M * N) |
| Token posterior projection (full vocab) | O(V * M * N) |
| Token posterior projection (sketch) | O(V * S + Kc * M * N) |
| Denoiser (CNN) | O(hidden * M * N) |
| DC correction | O(K * M * N) |
| Classifier | O(H * M * N * D + H * V * D) |
| Logit fusion | O(V) |

All operations scale linearly in the grid size M*N (no O(M^2 * N^2) terms),
linearly in the number of paths K, and for the token posterior projection either linearly in
V with small constant S or directly in V*M*N for the full-vocab path.

---

## 11. Ablation Design

The receiver supports 14 named ablation configurations, each disabling one
or more components:

| Ablation | Disables | Purpose |
|---|---|---|
| full | (none) | Baseline |
| no_offgrid | Off-grid refinement, forces ongrid operator | Evaluate off-grid contribution |
| no_physics_gain_refinement | Physics LS gain re-est | Evaluate LS refinement benefit |
| ongrid_operator | Off-grid operator mode | Baseline for fractional Doppler |
| no_token_posterior_prox | Token posterior projection disabled (L1 only) | Evaluate token prior in detection |
| no_candidate_pruning | Sketch candidate selection | Full-vocab vs sketch complexity |
| no_adaptive_candidate_pruning | Adaptive candidate expansion | Static vs adaptive candidate pool |
| no_energy_weighted_candidate_score | Falls back to distance scoring | Evaluate reliability weighting |
| no_denoiser | CNN denoiser | Evaluate model-based vs data-driven |
| no_data_consistency | DC correction step | Evaluate PnP-style consistency |
| single_head_classifier | H=1 evidence head | Evaluate multi-head contribution |
| no_logit_fusion | Disabled fusion | Classifier-only baseline |
| classifier_only | L1-only detector, no token posterior projection, no DD denoiser, no DC, no fusion | Minimalist baseline |
| no_reliability_calibration | (loss-level, inference unchanged) | |

This comprehensive ablation suite enables systematic evaluation of every
architectural component.

---

## 12. Key Innovations Summary

1. **DD-domain token communication:** The entire receiver operates in the
   Delay-Doppler domain, leveraging the inherent sparsity of the DD channel
   representation.  No bits, no QAM, no frequency-domain processing.

2. **Sparse DD operator H_theta as the channel-coupled detection interface:**
   The model-based detector and data-consistency correction interact with the
   channel exclusively through a parametric sparse operator supporting forward
   and adjoint operations.  Complexity O(KMN), no dense matrices.

3. **Model-based/data-driven alternation with full uncertainty propagation:**
   The PIC/Prox equalizer (model-based) -> CNN denoiser (data-driven) ->
   DC correction (model-based) chain, where every stage consumes and produces
   uncertainty estimates.

4. **Sketch-based adaptive candidate pruning:** Genuine complexity reduction
   from O(VMN) to O(VS+KcMN) by scoring all vocabulary entries only on a
   small sketch and evaluating the full DD-grid distance only for selected
   candidates.

5. **Multi-head evidence pooling:** Multiple independent evidence heads
   address the spatial multi-modality of DD token codewords.

6. **Reliability-gated detector-classifier fusion:** Learnable per-sample
   fusion of two independent token-level evidence sources.

7. **Physics-guided CE with strict observation semantics:** Pilot-only
   fitting prevents data contamination of channel estimates.

---

## 13. References

- R. Hadani et al., "Orthogonal Time Frequency Space Modulation," IEEE WCNC, 2017.
- Y. Yue et al., "Model-Driven Deep Learning Assisted Detector for OTFS With Channel Estimation Error," IEEE CL, 2024.
- S. V. Venkatakrishnan et al., "Plug-and-Play Priors for Model Based Reconstruction," IEEE GlobalSIP, 2013.
- K. Gregor and Y. LeCun, "Learning Fast Approximations of Sparse Coding," ICML, 2010.
- J. Ma et al., "Deep Unfolding Network for Delay-Doppler Sparse Channel Estimation in OTFS Systems," IEEE WCL, 2025.
- X. Bi et al., "Deep Learning-based OTFS Channel Estimation and Symbol Detection with Plug-and-Play Framework," IEEE TCOM, 2025.
- M. Borgerding et al., "AMP-Inspired Deep Networks for Sparse Linear Inverse Problems," IEEE TSP, 2017.
