# 服务器资源与职责

更新时间：2026-09-04（Asia/Shanghai）

本页是服务器分工的唯一当前记录。A100 对本项目只开放 GPU 0–3；GPU 4–7 即使空闲也不得使用。SSH alias 来自本机 SSH 配置；不在仓库记录密码、key 或
其他凭据。

## 服务器总表

| alias | 主机标识 | 硬件 | 固定职责 | 当前观察 |
|---|---|---|---|---|
| `a100` | `39.170.34.180:48001`，user `dengyan` | 8 × NVIDIA A100-PCIE-40GB（40GB） | RL 主实验 | 当前 Qwen3 RL live gate 使用一张 trainer 卡，另一张 vLLM 卡 |
| `table_rl` | `10.214.243.15:222`，user `dengyan` | 2 × RTX 3090（24GB） | 评测、行为诊断；A100 不可用时可做降级 RL | 当前有 `qwen3_v26_new_screen_20260903` 固定池评测进程，需先做空闲检查 |
| `NewGNN` | `10.130.129.25:16014`，user `dengyan` | 8 × RTX 3090（24GB） | SFT、评测、行为诊断；A100 不可用时可做降级 RL | 当前有其他任务占用部分 GPU，需先做空闲检查 |

当前 RL 使用两张不同的 A100：一张 replicated trainer，一张 online vLLM。两张卡都必须来自 GPU 0–3，且每次 run manifest 记录实际 id。

## A100 主 RL 约定

- 当前正式方案：Atomic version26 + `saam-asymmetric-error` + four-level result reward +
  reason/tool 各 0.5 的加权 full response；每 update 30 题，目标 cohort 约500题、每题 K=8 且正确数2–6。
- 历史 run 曾使用 GPU4–6；这些 id 仅保留在历史 manifest，不得用于新 run。新 launcher 只在 0–3 中动态选择并写入 manifest。
- replicated trainer world size 固定为 1，不初始化 NCCL；BF16 冻结基座、FP32 AdamW；训练和 vLLM 端口必须写进 manifest。
- 当前输出根目录：
  `/home/dengyan/tabular_rl_outputs/qwen3_8b_atomic_v26_saam_fourlevel_batch14_fsdp3_a100_20260903_r1`
  （这是 live gate，不能当作正式 700-step 完成结果）。
- release lock 根目录：
  `/home/dengyan/tabular_rl_outputs/rl_release_qwen3_8b_v26_fsdp_pcie_safe_20260903_r1`。
- 初始 adapter 固定为 `/data4/dengyan/checkpoints/qwen3_atomic_v26_cumulative_augmented_fresh4ep_4gpu_newgnn_20260827/checkpoint-6380`。

当前 canonical 拓扑就是一张 A100 做单进程 `replicated` trainer，另一张 A100 做 online
vLLM；默认 BF16 冻结基座、FP32 AdamW 和独立 output root。

A100 不可用时的 fallback 是同一台 3090 服务器上的两张独立 3090：一张 trainer、一张
vLLM。默认从 1 row / 8,192 transition tokens 起步，必要时显式使用 4-bit base 或 8-bit
AdamW；所有降级参数写入 manifest，不能混入 A100 结果。

历史资源快照中的 GPU4–7 不属于本项目可用集合；新 launcher 必须在 GPU0–3 内完成空闲检查，不能抢占其他进程。

不要停止 A100 上与本项目无关的进程；启动前必须由 launcher 检查目标 GPU 和端口，失败就
停止，不要抢占或复用他人任务。

## table_rl / NewGNN 约定

这两台 3090 服务器不承担当前 RL 主优化器。允许用途：

- version26 matched evaluator 和 candidate-vs-baseline 对照；
- SFT 训练或数据准备（尤其 NewGNN）；
- 行为、协议、回放和结果审计；
- 明确标注为 diagnostic 的小规模训练。

评测输出必须记录 model/checkpoint、runtime、prompt、cohort、GPU、端口和 evaluator identity，
不能以“在 3090 上跑过”替代 matched contract。

2026-09-03 快照（仅作状态记录，不是永久资源分配）：`table_rl` 的两张卡分别约使用
21,204 MiB / 22,078 MiB，正在运行 `qwen3_v26_new_screen_20260903` 固定池评测；
`NewGNN` 的 GPU2–5 有其他任务占用，GPU0/1/6/7 当时接近空闲，未发现本项目 RL 主训练进程。
这些数字不能用于抢占 GPU，也不代表下一次评测的可用性。

## 资源记录规则

1. 每次 run 记录服务器 alias、hostname、GPU 型号/数量、实际 GPU id、端口和路径。
2. `nvidia-smi` 快照只说明采样时刻，不能据此推断永久空闲或资源归属。
3. A100、table_rl、NewGNN 的结果目录分开；禁止混用 trajectory、adapter 或 manifest。
4. 服务器职责只有在用户明确确认后才能改变。
