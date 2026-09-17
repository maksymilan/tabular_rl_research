# Qwen3-4B trainer profiler（2026-09-13）

这是 table_rl GPU0 上的隔离诊断，不调用 `optimizer.step()`，不改变模型、adapter 或训练
输出。模型为 Qwen3-4B SFT `checkpoint-6380`，4-bit NF4 base、FP32 LoRA、SDPA、gradient
checkpointing；PyTorch 2.9.0+cu128。远端完整 Chrome trace 和 top-op 表保存在：

`table_rl:/home/dengyan/tabular_rl_outputs/perf_profiler_4b_20260913/`

第一份诊断未启用正式 trainer 的 BF16 autocast，属于失配配置，数据仅保留为反例，
不得用于判断正式训练速度或显存边界。正式路径的结果在远端 `..._r2_autocast`。

正式 BF16-autocast 的 4-bit 基准结果：

| 总序列长度 | 无 profiler forward+backward 中位时间 | peak allocated | peak reserved |
|---:|---:|---:|---:|
| 2048 | 1.481 s | 4.92 GiB | 5.52 GiB |
| 4096 | 3.016 s | 6.12 GiB | 6.69 GiB |
| 8192 | 6.505 s | 8.90 GiB | 9.96 GiB |

正式路径的 8192 trace 使用 FlashAttention forward/backward；CUDA self time 主要为
`aten::mm`（39.91%）、FlashAttention backward（11.53%）、forward（8.81%）、copy（12.29%）
和 elementwise mul（10.48%）。`Command Buffer Full` 约占 profiler CPU self time 的45%，
表示 GPU 命令队列存在回压，但 profiler 本身也会改变 CPU 时间，不能单独解释成 kernel launch
瓶颈。4096 与8192均未 OOM；这份 synthetic last-token loss 仍不是完整 GRPO update。

Profiler 会增加记录开销，表中时间只用于阶段和算子排序；无 profiler 中位数用于 paired
比较。BF16 base 的 paired 结果（同一 adapter、同一 LoRA 参数）为：2048/4096/8192
分别 1.257/2.591/5.751s，较 4-bit 快 15.2%/14.1%/11.6%，peak allocated 为
8.60/9.55/11.47GiB。长序列 12288/14700 的 BF16 为 9.538/12.238s、13.39/14.52GiB；
4-bit 为10.686/13.516s、11.66/13.30GiB。BF16更快但改变量化执行和显存合同，暂不静默
切换正式训练；下一步仍需完整 trainer update 的数值/梯度和 OOM 门禁。

真实 Atomic v26 单组 stage 也完成了 BF16 验证（8 episodes、54 transitions、actor
old-policy、microbatch=1/4096）：总 211.24s，rollout 84.91s、old-policy 28.54s、
policy 95.36s。manifest 的 `base_storage=bf16` 已核验，资源已释放；transition 数与历史
4-bit smoke 不同，该结果只证明 BF16 路径可运行，不能单独作为端到端排名。
