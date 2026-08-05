# NewGNN SFT2 Rank 并行运行时审计（2026-07-31）

## 目的与实验分配

`table_rl` 保留 Exp6 NoBackSlice+NoNormalize 与 Exp7 NoBackSlice+StrongLocalPenalty。
共享服务器 `NewGNN` 并行承接：

- Exp8：NoBackSlice + Action-only Rank；
- Exp9：NoBackSlice + Conservative Legal-Action Rank。

两项训练都从 SFT2 `checkpoint-1682` 初始化。`Qwen2.5-Coder-7B-Instruct` 仅作为加载该
QLoRA adapter 的底座；禁止裸 base 训练。

## 共享 GPU 边界

首次审计时，GPU0-3 被其他用户的四卡 vLLM 占用，GPU4 被另一服务占用，GPU5 随后被
另一用户训练进程占用。未停止、修改或抢占任何进程。最终队列只允许 GPU6/7，并使用：

- 显存空闲阈值：每卡不超过 512 MiB；
- watchdog 连续空闲门禁：至少 300 秒；
- 队列每一训练/评测阶段再次执行连续空闲检查；
- `flock` 防止本用户重复队列；
- `failed` 状态停止并等待审计，不覆盖不完整产物。

## 同步内容

从 `table_rl` 同步至 NewGNN `/home/dengyan` 的隔离路径：

- 精确 Qwen2.5-Coder-7B-Instruct 快照，约 15 GB；
- SFT2 checkpoint 目录；
- `trl-table` 隔离 Conda 环境，约 11 GB，不修改 NewGNN 已有 `sft`/`vllm` 环境；
- process counterfactual suite v2，约 11 GB；
- 23 条 RL 记录实际引用的 18 个 BIRD train 数据库，约 5.3 GB；
- equal-300 正式评测引用的 BIRD dev 数据库，约 1.4 GB；
- 冻结训练、version36 评测代码与输入索引。

## Fail-closed readiness

最终 readiness marker：

`/home/dengyan/tabular_rl_outputs/logs/newgnn_rank_runtime_ready_20260731`

已核验：

- SFT2 adapter SHA-256：
  `d880e2d7cc3203fdb0d11a7c188d8f607fd297b174eff23f741b6fe73cc3ce6e`；
- counterfactual passed manifest SHA-256：
  `5624c75c6ebbca556f1e0946069fdb95e923dc95d2d7692c8d929bedebd9568b`；
- 23/23 RL 记录数据库存在；
- 固定 equal-300 的 300/300 数据库存在；
- torch 2.9.0、transformers 4.57.6、TRL 0.29.0、vLLM 0.12.0 等版本与 table_rl
  冻结运行时一致；
- Exp8/Exp9 关键代码与配置哈希同本地一致。

## GPU smoke

在启动正式队列前完成并释放 GPU：

1. GPU6：CUDA kernel 成功；4-bit Qwen2.5-Coder-7B-Instruct 加载 SFT2
   `checkpoint-1682`，可训练 PEFT 模型完成真实 forward；
2. GPU7：vLLM 0.12 加载同一底座和 SFT2 LoRA，`/v1/chat/completions` 返回成功；
3. smoke 仅终止自身 PID/进程组；结束后 GPU6/7 均回落至 1 MiB。

## 去重与回退

`table_rl` 的 Exp8/Exp9 后续队列保留为回退，直到 NewGNN 队列实际通过所有门禁并开始
占用 GPU6/7。确认 NewGNN 实际启动后才创建：

`/home/dengyan/tabular_rl_outputs/logs/rank_followup_handed_off_to_newgnn_20260731`

该 marker 会令 table_rl 后续 watchdog 停止，从而避免两个服务器重复执行 Exp8/Exp9。
