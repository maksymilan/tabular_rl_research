# 服务器资源与职责

更新时间：2026-09-03（Asia/Shanghai）

本页是服务器分工的唯一当前记录。SSH alias 来自本机 SSH 配置；不在仓库记录密码、key 或
其他凭据。

## 服务器总表

| alias | 主机标识 | 硬件 | 固定职责 | 当前观察 |
|---|---|---|---|---|
| `a100` | `39.170.34.180:48001`，user `dengyan` | 8 × NVIDIA A100-PCIE-40GB（40GB） | RL 主实验 | 当前 Qwen3 RL live gate 使用两张 trainer 卡，另有一张 vLLM 卡 |
| `table_rl` | `10.214.243.15:222`，user `dengyan` | 2 × RTX 3090（24GB） | 评测和其他行为诊断 | 当前有 `qwen3_v26_new_screen_20260903` 固定池评测进程；不是主 RL 机 |
| `NewGNN` | `10.130.129.25:16014`，user `dengyan` | 8 × RTX 3090（24GB） | SFT、评测、数据/行为诊断 | 当前有其他任务占用部分 GPU；不是主 RL 机 |

“双卡训练”指 A100 的 FSDP actor 使用两个 trainer ranks（world size=2）。online rollout
vLLM 需要第三张 A100；因此正式 RL 资源是 2 张 trainer + 1 张 serving，而不是把三张都
算作优化器 rank。GPU 编号不是职责定义，必须由每次 run manifest 记录。

## A100 主 RL 约定

- 当前正式方案：Atomic version26 + `saam-asymmetric-error` + four-level result reward +
  reason/tool 各 0.5 的加权 full response。
- 2026-09-03 live gate 的实际拓扑是 trainer `CUDA_VISIBLE_DEVICES=4,6`、vLLM GPU5；
  后续 launcher 按空闲情况动态选择，不固定这些 id，并把最终选择写入 manifest。
- FSDP world size 固定为 2；BF16 full-shard base；训练和 vLLM 端口必须写进 manifest。
- 当前输出根目录：
  `/home/dengyan/tabular_rl_outputs/qwen3_8b_atomic_v26_saam_fourlevel_batch14_fsdp3_a100_20260903_r1`
  （这是 live gate，不能当作正式 700-step 完成结果）。
- release lock 根目录：
  `/home/dengyan/tabular_rl_outputs/rl_release_qwen3_8b_v26_fsdp_pcie_safe_20260903_r1`。
- 初始 adapter 固定为 `/data4/dengyan/checkpoints/qwen3_atomic_v26_cumulative_augmented_fresh4ep_4gpu_newgnn_20260827/checkpoint-6380`。

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
