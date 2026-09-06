# A100 单卡 replicated RL 拓扑

更新时间：2026-09-04（Asia/Shanghai）

这是当前 A100 的 canonical RL 拓扑：一张 A100 运行单进程 replicated trainer，另一张 A100 运行 online vLLM。它不改变 reward、credit 或 protocol。

## 保持不变的研究契约

- Atomic version26、Qwen3-8B、`checkpoint-6380`、目标约500题 screened cohort、30 prompts/update、K=8、
  200 optimizer updates；
- four-level result reward、`saam-asymmetric-error`、`trajectory_token_mean`、
  `span_balance_alpha=0.5`、`kl_beta=0`；
- trainer 与 vLLM 使用不同 GPU；launcher 只在 GPU 0–3 中动态选择空闲卡并记录实际 GPU、端口和路径；
- 输出仍必须包含 immutable manifest、implementation lock、precision audit 和 replay/gate 证据。

## 单卡实现

- trainer：`world_size=1`、`trainer_sharding=replicated`，不初始化 NCCL；
- base：默认 BF16 冻结基座，保持与 FSDP 路径一致的 base compute；
- actor：继续训练同一份 LoRA adapter，不切换到旧的全参数或旧 protocol 路径；
- optimizer：默认 FP32 `adamw_torch`，显存不足时才显式设置 `OPTIMIZER_NAME=paged_adamw_8bit`；
- micro-batch：默认 4 rows / 16,384 transition tokens，先以此做单卡 gate；确认显存余量后，
  才允许通过环境变量提高到 8 / 32,768。

入口和配置：

- `src/rl/scenarios/rl_main/run_qwen3_8b_atomic_v26_saam_fourlevel_spanbalanced_700_single_gpu_a100.sh`
- `src/rl/configs/experiments/qwen3_8b_atomic_v26_saam_screened500_single_gpu.yaml`

在 `a100` 上先做静态 preflight，再做一次一更新 live gate；gate 通过后才启动正式 200
updates。未指定 GPU 时 launcher 会在 0–3 中重新检查并选择两张严格空闲卡；也可以在确认资源归属后
显式指定 `TRAIN_GPU` 和 `VLLM_GPU`：

```bash
bash src/rl/scenarios/rl_main/run_qwen3_8b_atomic_v26_saam_fourlevel_spanbalanced_700_single_gpu_a100.sh preflight
TRAIN_GPU=0 VLLM_GPU=1 bash src/rl/scenarios/rl_main/run_qwen3_8b_atomic_v26_saam_fourlevel_spanbalanced_700_single_gpu_a100.sh gate
bash src/rl/scenarios/rl_main/run_qwen3_8b_atomic_v26_saam_fourlevel_spanbalanced_700_single_gpu_a100.sh run
```

`gate` 默认使用 4 rows / 16,384 tokens；确认显存余量后可另设
`TRANSITION_MICRO_BATCH_SIZE=8 TRANSITION_MICRO_BATCH_TOKENS=32768`，但必须保留独立
output root 和 manifest，不能覆盖历史双卡实验。

单卡路径的速度、峰值显存和最终 accuracy 必须由独立 run manifest 记录；在 gate 和 matched
evaluation 完成前，不能把它与历史双卡 FSDP 结果合并。若 BF16 replicated 路径
在 40GB 卡上不足，再使用 `REPLICATED_BASE_STORAGE=4bit`，该结果必须标注为低显存降级路径。
