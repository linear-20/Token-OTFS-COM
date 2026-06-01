# Token-OTFS-COM

面向 token 直接传输的 OTFS 物理层实验代码。当前仓库已经完成：

- TX 端 token-to-DD codebook 建模与 `L_core` 分离裕量训练；
- TX/RX 参数契约校验；
- 严格 physical artifact 导出、哈希校验与断点恢复；
- controlled closed-loop smoke 验收；
- RX 端检测、信道估计、refinement 和分类模块。

当前可直接执行的是 **TX-only surrogate shaping**。它用于训练发送端物理 codebook，不等同于完整波形层端到端训练。

## 1. 当前训练目标

发送端训练脚本：

```text
Learnable_Mapping_Tokens-to-DD-Signals/experiments/tx_cloud_train.py
```

唯一梯度目标为：

\[
L_{\text{core}}
=
\operatorname{mean}_p
\sum_r w_r
\left[
\max(0,\gamma-d_{p,r})
\right]^2
\]

其中：

- \(p=(u,v)\) 是两个不同 token 的有序 pair；
- \(r\) 是随机稀疏 DD 多径场景；
- \(d_{p,r}\) 是经过场景算子后的接收端投影分离能量；
- \(\gamma\) 是全局 margin，由 `--target-margin` 设置；
- 所有 token 使用相同功率、相同 margin，不使用 UEP 或语义加权。

## 2. 云服务器准备

克隆仓库：

```bash
git clone https://github.com/linear-20/Token-OTFS-COM.git
cd Token-OTFS-COM
```

准备 PyTorch 环境。推荐使用支持 CUDA 的 PyTorch。确认 GPU：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

设置 Python 模块路径：

```bash
export PYTHONPATH="$PWD/Learnable_Mapping_Tokens-to-DD-Signals"
```

Windows PowerShell 对应命令：

```powershell
$env:PYTHONPATH = ".\Learnable_Mapping_Tokens-to-DD-Signals"
```

先运行云端训练入口的定向测试：

```bash
python -m unittest tests.test_tx_cloud_train tests.test_tx_train_entry
```

### 推荐：使用统一训练入口

仓库提供两个可直接修改的实验配置：

```text
configs/tx_train_main.json
configs/tx_ablation_suite.json
```

主训练前先检查解析后的参数，不会启动训练：

```bash
python -m experiments.tx_train_entry \
  --experiment-json configs/tx_train_main.json \
  --dry-run
```

确认无误后启动 corrected `mean L_core` 主训练：

```bash
python -m experiments.tx_train_entry \
  --experiment-json configs/tx_train_main.json
```

每个 run 使用独立目录：

```text
artifacts/tx_studies/<study_name>/<run_name>/
```

目录内除了 TX artifact，还会保存：

```text
entry_resolved_config.json
```

study 级目录会保存：

```text
study_plan.json
study_manifest.json
```

断点恢复：

```bash
python -m experiments.tx_train_entry \
  --experiment-json configs/tx_train_main.json \
  --resume
```

### 运行消融实验

先列出消融项：

```bash
python -m experiments.tx_train_entry \
  --experiment-json configs/tx_ablation_suite.json \
  --list-runs
```

按名称运行单个消融，避免误启动整套实验：

```bash
python -m experiments.tx_train_entry \
  --experiment-json configs/tx_ablation_suite.json \
  --only-run margin_1p5_mean
```

可重复指定 `--only-run`，顺序执行多个实验：

```bash
python -m experiments.tx_train_entry \
  --experiment-json configs/tx_ablation_suite.json \
  --only-run margin_1p0_mean \
  --only-run margin_1p5_mean
```

修改参数时，优先编辑 JSON。`base_run_config` 放共享参数；每个 run 的 `overrides` 只放该消融项变化的参数。修改已有 run 参数后应同时修改 run 名称，避免覆盖旧 artifact。

## 3. 已提供的 TX 配置

仓库已经包含：

```text
configs/tx_profile.json
configs/tx_curriculum.json
configs/tx_train_main.json
configs/tx_ablation_suite.json
```

默认 pilot profile：

| 参数 | 当前值 | 含义 |
|---|---:|---|
| `M` | 32 | DD 网格 delay bins |
| `N` | 32 | DD 网格 Doppler bins |
| `vocab_size` | 256 | 通信 token 词表大小 \(V\) |
| `pilot_delay` | 16 | pilot 的 delay-bin 位置 |
| `pilot_doppler` | 16 | pilot 的 Doppler-bin 位置 |
| `pilot_guard_delay` | 4 | delay 方向 guard 半径 |
| `pilot_guard_doppler` | 5 | Doppler 方向 guard 半径 |
| `pilot_obs_delay_radius` | 1 | pilot CE 的 delay 观测半径 |
| `pilot_obs_doppler_radius` | 1 | pilot CE 的 Doppler 观测半径 |
| `data_power` | 1.0 | 每个 token codeword 的数据区平均功率 |
| `pilot_value_real` | 2.0 | pilot 实部 |
| `pilot_value_imag` | 0.0 | pilot 虚部 |
| `max_channel_delay` | 3 | TX surrogate 最大因果 delay bins |
| `max_channel_doppler` | 4 | TX surrogate 最大绝对 Doppler bins |
| `complex_dtype` | `complex64` | 复数计算精度 |

硬约束：

```text
pilot_guard_delay
  >= max_channel_delay + pilot_obs_delay_radius

pilot_guard_doppler
  >= max_channel_doppler + pilot_obs_doppler_radius
```

pilot 还必须处于无 wrap 的安全区域。配置不满足约束时，脚本会 fail fast。

## 4. 词表如何设置

`vocab_size` 表示通信 codebook 中 token 的数量。每个通信 token ID 对应一个可学习 DD codeword：

\[
C_v \in \mathbb{C}^{M \times N},
\qquad
v \in \{0,\ldots,V-1\}
\]

首轮训练建议使用：

```json
"vocab_size": 256
```

扩展顺序：

| 阶段 | 建议 `vocab_size` | 用途 |
|---|---:|---|
| Smoke | 16 | 验证代码链路 |
| Pilot run | 256 | 调整 margin、学习率和信道支持 |
| 论文主实验 | 1024 | 验证规模扩展能力 |
| 工程扩展 | 4096 或更高 | 接近真实 tokenizer 子词表 |

当前 TX-only 脚本只需要固定 token-ID 空间。接入真实 LLM tokenizer 时，还需要额外保存 tokenizer 名称、版本、哈希，以及原始 token ID 到通信 token ID 的映射。

## 5. Doppler curriculum 如何设置

`configs/tx_curriculum.json` 控制整数 DD 支持逐步扩展：

```json
[
  {
    "until_step": 20000,
    "max_delay": 1,
    "max_doppler": 1
  },
  {
    "until_step": 60000,
    "max_delay": 2,
    "max_doppler": 2
  },
  {
    "until_step": 120000,
    "max_delay": 3,
    "max_doppler": 4
  }
]
```

语义：

- `until_step`：该阶段包含的最后一个训练 step；
- `max_delay`：因果 delay 支持范围为 `0..max_delay`；
- `max_doppler`：Doppler 支持范围为 `-max_doppler..+max_doppler`；
- 后续阶段只能扩大支持，不能缩小；
- 最终支持不能超过 `tx_profile.json` 中的 `max_channel_delay` 和 `max_channel_doppler`。

例如：

```json
{"until_step": 60000, "max_delay": 2, "max_doppler": 2}
```

表示在该阶段随机采样：

```text
delay bins:   0, 1, 2
Doppler bins: -2, -1, 0, 1, 2
```

### Doppler bin 与 Hz 的关系

TX-only surrogate 使用整数 Doppler bin，不直接使用 Hz。进入波形验证时，需要锁定采样率 \(f_s\) 与 CP 长度 \(L_{\text{CP}}\)：

\[
\Delta \nu
=
\frac{f_s}{N(M+L_{\text{CP}})}
\]

\[
k_{\max}
=
\left\lceil
\frac{f_{D,\max}}{\Delta \nu}
\right\rceil
\]

其中 \(k_{\max}\) 对应 `max_doppler`。

## 6. 先运行 Pilot Run

不要直接启动 120000 steps。先运行 5000 steps，确认显存、耗时和验证曲线：

```bash
python -m experiments.tx_cloud_train \
  --output-dir artifacts/tx_v256_m32_pilot_seed2026 \
  --tx-config-json configs/tx_profile.json \
  --curriculum-json configs/tx_curriculum.json \
  --device cuda:0 \
  --seed 2026 \
  --train-steps 5000 \
  --train-pairs-per-step 128 \
  --validation-pairs 1024 \
  --test-pairs 2048 \
  --eval-every 250 \
  --checkpoint-every 250 \
  --progress-every 25 \
  --train-bank-refresh-every 25 \
  --num-train-scenarios 64 \
  --num-validation-scenarios 256 \
  --num-test-scenarios 512 \
  --num-paths 3 \
  --target-margin 1.0 \
  --learning-rate 0.003
```

Pilot run 可以测试少量组合：

```text
target_margin: 0.5, 1.0, 2.0
learning_rate: 0.001, 0.003, 0.01
```

每组实验使用独立 `--output-dir`，不要覆盖已有输出目录。

## 7. 正式 TX-only 训练

确认 pilot run 正常后，启动 120000 steps：

```bash
python -m experiments.tx_cloud_train \
  --output-dir artifacts/tx_v256_m32_seed2026 \
  --tx-config-json configs/tx_profile.json \
  --curriculum-json configs/tx_curriculum.json \
  --device cuda:0 \
  --seed 2026 \
  --train-steps 120000 \
  --train-pairs-per-step 128 \
  --validation-pairs 1024 \
  --test-pairs 2048 \
  --eval-every 500 \
  --checkpoint-every 500 \
  --progress-every 25 \
  --train-bank-refresh-every 25 \
  --num-train-scenarios 64 \
  --num-validation-scenarios 256 \
  --num-test-scenarios 512 \
  --num-paths 3 \
  --target-margin 1.0 \
  --learning-rate 0.003
```

建议至少运行多个随机种子：

```text
2026
2027
2028
```

## 8. 断点恢复

脚本会周期性写入：

```text
artifacts/<run-name>/checkpoint_latest.pt
artifacts/<run-name>/checkpoints/
```

恢复训练时必须：

- 使用同一个 `--output-dir`；
- 保持原有 profile、curriculum 和所有训练参数不变；
- 增加 `--resume-checkpoint`。

示例：

```bash
python -m experiments.tx_cloud_train \
  --output-dir artifacts/tx_v256_m32_seed2026 \
  --tx-config-json configs/tx_profile.json \
  --curriculum-json configs/tx_curriculum.json \
  --device cuda:0 \
  --seed 2026 \
  --train-steps 120000 \
  --train-pairs-per-step 128 \
  --validation-pairs 1024 \
  --test-pairs 2048 \
  --eval-every 500 \
  --checkpoint-every 500 \
  --progress-every 25 \
  --train-bank-refresh-every 25 \
  --num-train-scenarios 64 \
  --num-validation-scenarios 256 \
  --num-test-scenarios 512 \
  --num-paths 3 \
  --target-margin 1.0 \
  --learning-rate 0.003 \
  --resume-checkpoint artifacts/tx_v256_m32_seed2026/checkpoint_latest.pt
```

checkpoint 包含 optimizer 状态、当前 codebook、当前训练场景和 RNG 状态。测试已经验证：断点恢复结果与不中断训练逐 bit 一致。

## 9. 训练参数说明

| CLI 参数 | 含义 | 调参建议 |
|---|---|---|
| `--output-dir` | 当前实验输出目录 | 每组实验单独设置 |
| `--tx-config-json` | TX 物理 profile | 正式训练必须显式指定 |
| `--curriculum-json` | delay/Doppler curriculum | 正式训练必须显式指定 |
| `--device` | PyTorch 设备 | 云端使用 `cuda:0` |
| `--seed` | 基础随机种子 | 多 seed 独立运行 |
| `--train-steps` | 优化步数 | pilot 用 5000，正式用 120000 |
| `--train-pairs-per-step` | 每步采样 token pairs 数量 \(P\) | 显存不足时优先减小 |
| `--validation-pairs` | 固定验证 pair 数量 | 建议 1024 |
| `--test-pairs` | 固定测试 pair 数量 | 建议 2048 或更高 |
| `--eval-every` | 验证间隔 | 建议 250 或 500 |
| `--checkpoint-every` | checkpoint 间隔 | 建议与验证间隔一致 |
| `--progress-every` | 终端轻量进度输出间隔 | 建议 25，不额外触发验证 |
| `--train-bank-refresh-every` | 刷新训练信道场景库的间隔 | 建议 25 |
| `--num-train-scenarios` | 每次训练场景库大小 \(R\) | 建议 64 |
| `--num-validation-scenarios` | 固定 held-out 验证场景数 | 建议 256 |
| `--num-test-scenarios` | 固定 held-out 测试场景数 | 建议 512 |
| `--num-paths` | 每个稀疏场景的路径数 \(K\) | 首轮建议 3 |
| `--target-margin` | 全局裕量 \(\gamma\) | pilot 中比较 0.5、1.0、2.0 |
| `--training-risk-aggregation` | TX-only penalty 聚合方式 | 主基线使用 `mean`；`tail_cvar` 仅用于尾部消融 |
| `--tail-cvar-fraction` | `tail_cvar` 保留的最差 pair-scenario 比例 | 尾部消融建议先用 `0.05` |
| `--allow-duplicate-dd-taps` | 允许同一场景重复抽取 DD bin | 仅用于复现旧逻辑，不建议用于主实验 |
| `--learning-rate` | Adam 学习率 | pilot 中比较 0.001、0.003、0.01 |

计算复杂度近似为：

\[
O(P R K M N)
\]

显存不足时，优先降低：

```text
train_pairs_per_step
num_train_scenarios
M, N
```

不要为了省显存修改验证集和测试集的独立性。

### 有效 DD tap 与尾部审计

云训练默认将每个随机场景建模为 \(K\) 个不同的 on-grid 有效 DD taps。同一场景不重复抽取同一个 DD bin，避免重复 bin 的复增益相消破坏单位有效 tap 功率语义。底层稀疏算子仍然支持 duplicate shifts，用于受控实验和一般线性组合。

对已完成 run 进行只读尾部审计：

```bash
python -m experiments.tx_tail_audit \
  --run-dir artifacts/tx_v256_m32_seed2026 \
  --split test \
  --artifact both \
  --device cuda:0
```

审计会输出困难 pair、困难 scenario、尾部分位数以及 `scenarios_with_duplicate_dd_taps`。旧训练逻辑的复现实验才应显式增加 `--allow-duplicate-dd-taps`。

如果 corrected `mean` 基线在多个 seed 下仍有稳定尾部差距，再运行单独的 empirical CVaR 消融：

```bash
--training-risk-aggregation tail_cvar \
--tail-cvar-fraction 0.05
```

这不会增加 TX 模块，也不会修改全局物理 margin。它只将同一个 squared margin deficit 聚合为最差 \(5\%\) pair-scenario penalty 的均值。

## 10. SNR 在哪里设置

当前 `tx_cloud_train.py` **没有** `--snr-db` 参数，这是有意设计的。

TX-only 的 `L_core` 在固定发送功率下优化 DD 分离能量。AWGN SNR 只改变 PEP 的标度，不改变 codeword 分离距离排序，因此不应把 SNR 强行加入发送端 shaping 损失。

底层波形信道已经支持：

```text
ChannelConfig.snr_db
ChannelConfig.max_doppler_hz
TimeVaryingMultipathChannel.forward(..., snr_db=...)
```

实现位置：

```text
channel_model.py
```

真正的 SNR curriculum 应用于后续 RX-only 和端到端波形训练。建议按 batch 随机采样，而不是固定单点：

```json
[
  {
    "until_step": 30000,
    "snr_db_min": 20.0,
    "snr_db_max": 30.0,
    "max_doppler_hz": 1000.0
  },
  {
    "until_step": 80000,
    "snr_db_min": 10.0,
    "snr_db_max": 30.0,
    "max_doppler_hz": 2500.0
  },
  {
    "until_step": 150000,
    "snr_db_min": 0.0,
    "snr_db_max": 30.0,
    "max_doppler_hz": 5000.0
  }
]
```

注意：上面的 RX/E2E SNR curriculum 是后续训练规划，当前仓库尚未提供对应的长训 CLI。

## 11. 输出文件

每次 TX-only 训练输出：

```text
artifacts/<run-name>/
  baseline_raw_state.pt
  baseline_physical_codeword_book.pt
  shaped_raw_state.pt
  shaped_physical_codeword_book.pt
  validation_scenario_bank.pt
  test_scenario_bank.pt
  checkpoint_latest.pt
  checkpoints/
  metrics.jsonl
  manifest.json
```

重点文件：

| 文件 | 用途 |
|---|---|
| `shaped_physical_codeword_book.pt` | RX runtime 使用的物理 codebook |
| `checkpoint_latest.pt` | 断点恢复 |
| `metrics.jsonl` | 训练曲线和 held-out 指标 |
| `manifest.json` | 参数、随机种子、软件版本、GPU 和 claim boundary |

`artifacts/` 已在 `.gitignore` 中排除。正式实验结果应保存到云磁盘或对象存储，不要提交到 Git。

CLI 还会按 `--progress-every` 在终端输出轻量 JSON 进度，包括：

```text
step
target_step
latest_train_l_core
elapsed_seconds
estimated_remaining_seconds
```

该进度输出不会额外执行 validation。

## 12. TX 训练后验收

将训练后的 physical artifact 接回 controlled closed-loop：

```bash
python -m experiments.system_closeloop_harness \
  --physical-artifact artifacts/tx_v256_m32_seed2026/shaped_physical_codeword_book.pt \
  --require-cp-sufficient \
  --output-json artifacts/tx_v256_m32_seed2026/closeloop_gate.json
```

该步骤用于验证：

- artifact hash 与 schema；
- TX/RX 的 `M,N,V`、dtype、pilot、guard 和 mask；
- CP 条件；
- codebook 能否进入接收链路；
- controlled token 恢复是否工作。

它不是完整论文性能实验。论文结论还需要：

1. 波形层 SNR sweep；
2. fractional delay/Doppler mismatch；
3. 非 oracle channel estimation；
4. RX-only 训练；
5. 端到端微调；
6. 多随机种子对照实验。

## 13. 常见错误

### 输出目录已存在

错误：

```text
output_dir already contains metrics.jsonl
```

处理方式：

- 新实验使用新的 `--output-dir`；
- 中断恢复使用 `--resume-checkpoint`。

### 恢复训练参数不一致

错误：

```text
resume_checkpoint run_config mismatch
```

处理方式：恢复训练时保持全部原始 CLI 参数不变。

### CUDA 不可用

错误：

```text
Requested CUDA device cuda:0, but CUDA is unavailable
```

处理方式：

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

确认安装 CUDA 版 PyTorch，并确认云服务器已挂载 GPU。

### curriculum 超出 TX profile

错误：

```text
curriculum max_doppler ... exceeds tx_config.max_channel_doppler ...
```

处理方式：同步调整 `configs/tx_profile.json` 的最大支持、pilot guard 和无 wrap 边界。

## 14. 当前边界

当前 TX-only 训练是 on-grid sparse-DD surrogate shaping：

- 支持随机稀疏多径；
- 支持因果整数 delay；
- 支持正负整数 Doppler-bin curriculum；
- 支持 held-out validation/test；
- 支持可重复和可恢复训练。

当前阶段不声称：

- 已完成完整波形层训练；
- 已完成 fractional delay/Doppler 训练；
- 已获得 BER/TER 保证；
- 已完成 RX 泛化训练；
- 已完成端到端论文实验。
