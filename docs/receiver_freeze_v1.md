# Receiver Freeze Manifest -- rx_v1

## Frozen date

2026-05-28

## Frozen receiver chain

```
Y_DD [B,M,N] complex
  -> Embedded Pilot Sparse CE
  -> Off-grid Sparse Path Refinement
  -> Physics-Guided Gain Re-estimation
  -> Parametric Sparse DD Operator H_theta / H_theta^H
  -> Variance-Aware Token Posterior Unfolded Detector
  -> Adaptive Candidate-Pruned Token Posterior Prox
  -> Confidence / Symbol-Reliability-Gated Residual Denoiser
  -> Data-Consistency Correction
  -> Multi-Head Evidence Token Classifier
  -> Detector-Classifier Logit Fusion
  -> token_logits [B,V]
```

## Frozen public API

### Model
- `LearnableOTFSReceiver(config: ReceiverConfig)` -- the ONLY entry point
- `model(y_dd)` returns `Tensor [B, V]` (token logits)
- `model(y_dd, return_details=True)` returns `ReceiverOutput`

### Config
- `ReceiverConfig` -- all fields frozen; new fields only by addition with safe defaults

### Core dataclasses
- `ReceiverOutput`
- `SparseChannelEstimate`
- `SparseDDOperatorState`
- `EqualizerOutput`
- `DenoiserOutput`
- `DataConsistencyOutput`
- `TokenClassifierOutput`
- `TokenLogitFusionOutput`
- `ReliabilityDiagnostics`
- `PhysicsValidationReport`
- `ReceiverComplexityReport`
- `ReceiverPaperTrace`
- `ReceiverLossWeights`
- `ReceiverLossOutput`

### TX dependencies (TX will need these)

| Item | Shape / Type | Notes |
|---|---|---|
| `ReceiverConfig` | dataclass | M, N, vocab_size, token_embedding_dim |
| `TokenCodewordPrior.codeword_book` | complex `[V, M, N]` | DD token codewords shared with TX |
| `ReceiverOutput.token_logits` | float `[B, V]` | final receiver output |
| `data_mask` | real `[B, M, N]` or `[1, M, N]` | marks data-bearing DD positions |
| `pilot_mask` | real `[B, M, N]` or `[1, M, N]` | marks pilot position |
| `guard_mask` | real `[B, M, N]` or `[1, M, N]` | marks guard region around pilot |

### Hardware requirements

- No bit head, QAM head, Linear(vocab_size) classifier head
- No Transformer, self-attention, GNN, VAE
- No dense MN x MN channel matrix
- All operations bounded by O(KMN) or O(VS+KMN) where K = topk_paths, S = sketch_size

## Frozen test baseline

```
E:\pytorch\python.exe -m unittest discover -s tests -v
```

All tests must pass.  376+ tests expected.

## Modification policy

After freeze:
- Bug fixes: allowed
- Trace / harness / diagnostics fixes: allowed
- Test additions (non-breaking): allowed
- Config field additions (with safe default): allowed
- Main chain modification: **FORBIDDEN**
- Architecture changes (CNN/Transformer/GNN/VAE): **FORBIDDEN**
