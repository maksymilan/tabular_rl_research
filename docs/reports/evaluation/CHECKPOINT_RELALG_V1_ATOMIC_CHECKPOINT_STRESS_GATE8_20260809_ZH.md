# Checkpoint-RelAlg V1 Atomic Checkpoint Stress Gate8（2026-08-09）

## 结论

本次诊断支持两个不同强度的结论：

1. **Harness 的 checkpoint/restore 状态语义有效。** 定向回归覆盖 commit、错误分支、restore、artifact 去活、阶段切换、单调 ID、状态哈希精确恢复和 fresh replay；checkpoint-relalg 全套测试为 `208 passed, 34 subtests passed`。
2. **Atomic 模型可以在语义里程碑主动使用 checkpoint。** 使用 teacher-only `checkpoint-stress-v1` 后，7/8 episode 使用 checkpoint，共 8 次 commit；默认 Atomic Prefix100 的同题基线为 0 次。

本次**没有证明模型侧 restore 策略已有效**：8 个 episode 共 0 次 restore。唯一发生在 checkpoint 之后的工具错误是 `unordered_limit_input`，模型用 `sort -> limit` 在当前分支局部修正，不需要回滚到阶段起点。因而 0 restore 在该轨迹上是合理行为，但不能作为 restore 能力的正证据。

## 冻结设置

- 模式：`atomic`
- carrier：`text-json`（A 臂）
- provider/model：官方 DeepSeek / `deepseek-v4-flash`
- checkpoint guidance：`checkpoint-stress-v1`（teacher-only）
- cohort：teacher1500 的 positions 64–71；与既有 Atomic Prefix100 同题配对
- 并行：两个独立 4 题 shard；单 episode 内保持因果串行
- 上限：每 shard 1,250,000 provider tokens，总硬上限 2,500,000
- 每 episode：最多 2 commit、1 restore、20 model turns、30 primitive calls
- 轨迹继续保持 diagnostic-only；不进入 SFT/RL

结果目录：

- `data/results/checkpoint_relalg_v1_flash_text_json_atomic_checkpoint_stress_gate8_20260809/shard_064_067`
- `data/results/checkpoint_relalg_v1_flash_text_json_atomic_checkpoint_stress_gate8_20260809/shard_068_071`

两个 shard 的 structure、manifest binding、cohort identity、provider budget、batch control 与 fresh replay 均通过，合计 8/8 records 无审计问题。

## 同题配对结果

| 指标 | 默认 Atomic 同题基线 | Checkpoint stress | 变化 |
|---|---:|---:|---:|
| bird-set 正确 | 4/8 | 6/8 | +2 |
| strict artifact 正确 | 3/8 | 4/8 | +1 |
| schema match | 4/8 | 5/8 | +1 |
| legal termination | 6/8 | 8/8 | +2 |
| model turns | 79 | 76 | -3 |
| tool errors | 6 | 2 | -4 |
| provider tokens | 1,803,216 | 1,438,552 | -364,664（-20.2%） |
| checkpoint calls | 0 | 8 | +8 |
| restore calls | 0 | 0 | 0 |

配对 outcome 为 2 gains、0 regressions：positions 67 和 70 从错误/非法终止变为正确且合法；其余 6 题的 bird-set outcome 不变。样本只有 8 题，且 teacher-only guidance 是行为干预，因此不能把准确率提升因果归于 checkpoint 状态压缩本身。

## Checkpoint 行为

- 7/8 episode 至少 commit 一次。
- 6 个 episode commit 一次；1 个 episode commit 两次；1 个短 episode不 commit。
- 每次 commit 都发生在已有关系构造或检查之后，并由后续 answer-relevant 原子操作继续推进，没有出现 commit 后立即机械 answer 的统一模式。
- phase 从单一 `phase_000` 变为最多 `phase_000 -> phase_001 -> phase_002`；所有 active-path/fresh-replay 审计通过。
- 唯一 checkpoint 后错误没有破坏既有阶段事实，采用局部修复比 restore 更合适。

## 判定与下一步

- **Checkpoint 状态机制：PASS。** Harness 语义与 replay 已被确定性验证。
- **Atomic commit 可用性：初步 PASS。** 模型能形成真实阶段边界，且本 Gate8 没有观察到成本或正确率退化。
- **Atomic restore 决策：INCONCLUSIVE。** 没有出现应当放弃整个 checkpoint 后分支的自然样本。

若要验证模型侧 restore，应使用一个单独的小型、预注册的 restore-trigger gate：选择会在 checkpoint 后暴露关系粒度/表示矛盾的困难任务，或使用 Harness 控制的非金标故障注入；评价指标应是“需要回滚的事件中 restore 的精确率/召回率”，而不是强迫每题 restore。该后续实验必须独立授权，不能把本 Gate8 的 0 restore 误报为失败或成功。
