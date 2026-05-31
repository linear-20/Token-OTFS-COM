# TX-AUDIT-01: 发送端全面只读审查

**日期**: 2026-05-31
**审查范围**: 发送端全部模块、接收端关键接口、实验 harness、理论文档
**结论**: 允许进入封装实现阶段，4 项 HIGH findings 需在封装前修复

---

## 1. 架构分类

### Core Path（核心链）

| Module | Role |
|--------|------|
| `config.py` | TX 超参数唯一 source of truth |
| `pilot_frame.py` | Hard pilot/guard layout + no-wrap boundary safety |
| `codebook.py` | Equal-power physical codeword book [V, M, N] |
| `model.py` | Codebook → mask → pilot insertion → optional OTFS modulation |
| `otfs_modem.py` | Deterministic ISFFT/SFFT + OFDM modulation/demodulation |
| `modem_adapter.py` | OTFS modem contract validation (modulate, dd_shape, cp_len) |
| `sparse_multipath.py` | On-grid sparse multipath DD operator H_r |
| `multipath_scenarios.py` | Fixed normalized scenario bank with unit active-path-power |
| `multipath_margin.py` | **唯一默认 shaping 目标**: L_core = mean_p sum_r w[r] * relu(gamma - d_uv,r)^2 |
| `export.py` | Raw state / physical book lifecycle separation |

### Engineering Support（工程支撑）

| `dd_shifts.py` | Receiver-aligned integer DD shift: torch.roll(dims=(-2,-1)) |
| `hard_negative.py` | Selection-only mining utility (no_grad, detached) |
| `sampling.py` | O(P) pair sampling + weighted shift sampling |
| `experiments/step19_lcore_harness.py` | Standalone surrogate shaping harness |

### Diagnostics / Ablations Only（仅诊断/消融，不进入默认目标）

| Module | Role |
|--------|------|
| `orbit_correlation.py` | masked_normalized_dd_correlation, masked_shift_orbit_correlation |
| `orbit_visibility.py` | Shift-orbit energy visibility: eta = ||M_e * Pi_s(C_v)||^2 / ||C_v||^2 |
| `regularizers.py` | Cross-token confusion, self-shift sidelobe, visibility hinge loss |
| `metrics.py` | DD-domain + time-domain power metrics, PAPR |

**默认不删除** ablations。合并多项损失需要独立 ablation 证据。

---

## 2. Findings 表格

| # | Severity | File & Line | Problem | Minimal Fix |
|---|----------|-------------|---------|-------------|
| **F1** | **HIGH** | `export.py:152` | `load_exported_codeword_book` 使用裸 `torch.load()` 无 `weights_only=True`，存在 pickle 安全风险 | 默认使用 `weights_only=True`，或要求 `--trusted-source` flag |
| **F2** | **HIGH** | `export.py:313-316` | `_validate_physical_export_constraints` 不检查 `isfinite(codeword_book)`。NaN/Inf codeword 可成功 export 且标记 `receiver_prior_ready: True` | 增加 `torch.isfinite(codeword_book).all()` |
| **F3** | **HIGH** | `export.py:395-422` | metadata 不含 `schema_version`。未来 artifact 格式变更时 load 侧无法区分版本 | 新增 `SCHEMA_VERSION = 1`，写入 metadata，loading 时校验 |
| **F4** | **HIGH** | `token_prior.py:35-36` | `TokenCodewordPrior.__post_init__` 不检查 V>0, M>0, N>0, isfinite。空/NaN codeword 被静默接受 | 拆分校验：is_tensor → is_complex → ndim==3 → V,M,N>0 → isfinite |
| **F5** | **MEDIUM** | `export.py:354-367` | `_validate_loaded_codeword_book` 不检查 isfinite（与 F2 关联） | 与 F2 同步修复 |
| **F6** | **MEDIUM** | `export.py:413-422` | metadata 不含 codeword_book / data_mask hash | 可选：增加 SHA-256 字段（非阻断） |
| **F7** | **MEDIUM** | `model.py:165-183` | `_validate_compatible_codebook` 仅检查 M, N, vocab_size, data_power, complex_dtype — **这是合理的最小集合**，codebook 应保持 layout-independent | 保持现状。在 PhysicalDDProfile 中注明 |
| **F8** | **LOW** | `pilot_frame.py:56-61` | Guard 边界 clip (no-wrap rectangular) 已验证通过 `_validate_insertion_masks` | 无需修改 |
| **F9** | **LOW** | `sparse_multipath.py:145-148` | All-inactive 防御: `y_b = x_dd[b]*0.0 + gains.sum()*0.0` 保持 zero-gradient anchor | 无需修改 |
| **F10** | **LOW** | `codebook.py:94-100` | `power <= 0` 已在 L94 拒绝，`scale = sqrt(data_power)/power.sqrt()` 始终 finite | 无需修改 |
| **F11** | **INFO** | `export.py:221-230` | 显式 mask 绕过路径已在 Step 8a 修补就位 | 无需修改 |
| **F12** | **INFO** | `multipath_margin.py` | Evidence mask 限制为 shared [1,M,N]，与 A5 一致 | 文档已说明 |

---

## 3. TX/RX 轴顺序和 Shift 契约验证

| 组件 | Delay 维度 | Doppler 维度 | Shift 语义 |
|------|-----------|-------------|-----------|
| TX `dd_circular_shift` | dim=-2 (M) | dim=-1 (N) | `torch.roll(x, shifts=(d,v), dims=(-2,-1))` |
| RX `dd_circular_convolve_sparse` | dim=0 (per [M,N]) | dim=1 (per [M,N]) | `torch.roll(x[b], shifts=(d,v), dims=(0,1))` |
| TX `apply_sparse_multipath_dd_operator` | same as TX shift | same as TX shift | 按 batch/path 调用 `dd_circular_shift(x[b:b+1], d, v)` |
| RX path_indices 顺序 | [delay_idx, doppler_idx] | 同左 | Step 17A `ReceiverContractTests` 验证数值等价 |

**结论**: TX dims(-2,-1) ≈ RX dims(0,1) per [M,N] slice。Step 17A 7 个 receiver contract 测试全部 `torch.allclose`。

---

## 4. Pilot/Guard/Mask 契约验证

| 检查项 | TX | RX | 状态 |
|--------|-----|-----|------|
| Pilot position | `TransmitterConfig.pilot_delay/doppler` | `ReceiverConfig.pilot_delay/doppler` | 独立配置，需人工对齐 |
| Guard boundary | no-wrap clip (min/max) | N/A (pilot CE 独立) | TX 已验证 |
| Data mask | `build_transmitter_pilot_masks(config).data_mask` [1,M,N] bool | `data_mask` [B,M,N] or [1,M,N] | export 传递 |
| Evidence mask | `sparse_multipath_operator_margin_loss` 用 shared [1,M,N] data_mask | N/A | 理论一致 |
| Zero pilot/guard in codeword | `_validate_physical_export_constraints` 强制检查 | N/A (receiver 不验证 codeword 本身) | TX 保证 |
| Per-token equal power | `_validate_physical_export_constraints` 逐 token 检查 | N/A | TX 保证 |

---

## 5. Export/Load 边界评估

### 当前能力

- `save_raw_tx_state`: 保存 `state_dict` (raw_real, raw_imag) + config metadata
- `export_physical_codeword_book`: 保存 physical normalized codeword_book + data_mask + metadata，包含 pilot/guard zero 和 per-token equal power 校验
- `load_exported_codeword_book`: 拒绝 raw state payload，校验 metadata flags (receiver_prior_ready, artifact_type)，校验 codeword_book shape

### 缺陷

- **无 schema version** (F3): 格式演进无法追踪
- **无 isfinite check** (F2/F5): NaN codeword 可静默通过
- **无 hash** (F6): 无法检测 artifact 损坏或篡改
- **裸 torch.load** (F1): pickle 安全风险

### 封装建议

```python
SCHEMA_VERSION = 1

def export_physical_codeword_book(...):
    # ... existing checks ...
    # ADD: torch.isfinite(codeword_book).all()
    # ADD: metadata["schema_version"] = SCHEMA_VERSION
    # ADD: metadata["codeword_is_finite"] = True

def load_exported_codeword_book(path, *, weights_only=True):
    payload = torch.load(path, weights_only=weights_only, map_location=...)
    # ADD: schema_version = metadata.get("schema_version")
    # ADD: if schema_version not in {1}: raise ...
    # ... existing checks ...
```

---

## 6. TokenCodewordPrior 边界校验评估

**当前** (token_prior.py:35-36):
```python
if not torch.is_tensor(self.codeword_book) or self.codeword_book.ndim != 3:
    raise ValueError("codeword_book must have shape [V, M, N].")
if not torch.is_complex(self.codeword_book):
    raise TypeError("codeword_book must be complex with shape [V, M, N].")
```

**缺失**:
- `V <= 0`, `M <= 0`, `N <= 0` 未单独检查
- `torch.isfinite` 未检查
- codeword_book 与 RX 的 M, N 一致性在 `_validate_projection_inputs` 中检查 (`prior.codeword_book.shape[-2:] != z_dd.shape[-2:]`)，但那是运行时校验，不是构造时校验

**建议** (F4): 拆分校验为独立断言。

---

## 7. TX/RX 共享参数契约（草案）

### PhysicalDDProfile（source of truth: exported codeword_book.pt + manifest）

| 参数 | TX Source | RX 所需 | 校验 |
|------|-----------|---------|------|
| M, N | `TransmitterConfig` | codeword_book.shape[1:] | export 校验 |
| V | `TransmitterConfig.vocab_size` | codeword_book.shape[0] | export 校验 |
| dtype | `TransmitterConfig.complex_dtype` | codeword_book.dtype | export 校验 |
| data_power | `TransmitterConfig.data_power` | 不直接使用 | export 逐 token 校验 |
| pilot_delay/doppler | `TransmitterConfig` | `ReceiverConfig` | 独立配置对齐 |
| pilot_guard_delay/doppler | `TransmitterConfig` | `ReceiverConfig` | 独立配置对齐 |
| axis convention | `torch.roll(dims=(-2,-1))` | `torch.roll(dims=(0,1))` | Step 17A tests |
| data_mask (bool [1,M,N]) | `build_transmitter_pilot_masks` | 从 artifact 加载 | export 时 pilot/guard zero |

### WaveformProfile（仅 waveform 实验）

| 参数 | Source | 条件 |
|------|--------|------|
| dd_shape (M,N) | OTFSModem | == codeword (M,N) |
| cp_len | OTFSModem | >= L_delay |
| CP sufficient | experiment harness | 非 TransmitterConfig 校验 |

### TrainingProfile（仅训练，不进入 RX runtime）

| 参数 | Source | 条件 |
|------|--------|------|
| gamma | Step19LCoreRunConfig.target_margin | >= 0, 全局共享 |
| P (pairs/step) | train_pairs_per_step | > 0 |
| R, K | num_scenarios, num_paths | > 0 |
| seeds | derived from base seed | 记录在 manifest |
| learning_rate | Step19LCoreRunConfig | finite, > 0 |

---

## 8. Artifact Schema 草案

```json
{
  "schema_version": 1,
  "codeword_book": "<complex float [V, M, N], CPU>",
  "data_mask": "<bool [1, M, N], CPU>",
  "metadata": {
    "schema_version": 1,
    "artifact_type": "physical_codeword_book",
    "M": "<int>",
    "N": "<int>",
    "vocab_size": "<int>",
    "data_power": "<float>",
    "complex_dtype": "complex64|complex128",
    "pilot_delay": "<int>",
    "pilot_doppler": "<int>",
    "pilot_guard_delay": "<int>",
    "pilot_guard_doppler": "<int>",
    "pilot_obs_delay_radius": "<int>",
    "pilot_obs_doppler_radius": "<int>",
    "max_channel_delay": "<int>",
    "max_channel_doppler": "<int>",
    "physical_codeword_book_exported": true,
    "hard_data_mask_projection": true,
    "receiver_prior_ready": true,
    "raw_tx_state_dict": false,
    "codeword_is_finite": true,
    "pilot_guard_zeros_verified": true,
    "per_token_equal_power_verified": true
  }
}
```

---

## 9. 云端 TX-only 训练前置条件

### 显存估算

| Tensor | Shape | Size (complex64) |
|--------|-------|------------------|
| codeword_book | [V=256, M=64, N=64] | 8 MB |
| delta_flat | [P*R=4096, M=64, N=64] | 128 MB |
| y_flat | [P*R=4096, M=64, N=64] | 128 MB |
| ch_flat | [P*R=4096, K=4, *] | < 1 MB |
| Gradients | ~ same as forward | ~264 MB |
| **Total peak** | | **~300-400 MB** |

**结论**: 默认配置下单 GPU (< 1 GB) 可运行。不需要 exact chunking。

### 前置条件

- [ ] F1 (torch.load 安全) 已修复
- [ ] F2/F5 (isfinite export/load) 已修复
- [ ] F3 (schema_version) 已增加
- [ ] GPU 可用性确认
- [ ] 在 manifest 中记录 GPU 型号和显存

---

## 10. 声明

- 本次审查未修改任何 Python 源文件、测试文件、或 transmitter 架构
- 未生成文档文件（除本报告外）
- 未启动训练
- 未运行测试套件
- 所有 findings 基于 2026-05-31 代码库只读审查
- 修复优先级: F1 > F2 > F3 > F4 > F5 > F6
