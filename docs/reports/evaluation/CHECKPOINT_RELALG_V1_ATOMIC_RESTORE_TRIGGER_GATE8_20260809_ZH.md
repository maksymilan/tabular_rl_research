# Checkpoint-RelAlg V1 Atomic Restore-Trigger Gate8（2026-08-09）

## 结论

本 Gate8 首次在真实 DeepSeek Flash Atomic 轨迹中观察到 `restore_checkpoint`，证明模型能够调用该工具、Harness 能执行恢复转换、provider phase history 能重置，并且完整轨迹可通过 structure、identity、budget 与 fresh replay 审计。

但观察到的 restore **不是有效的语义分支回滚**：模型在 position 93 尚未 commit 时恢复 `root`，调用发生在触发错误之后四个回合，并且中间已经有三个成功观察。恢复没有 abandoned checkpoint，也没有纠正最终 outcome；该题仍以 `max_model_turns` 失败。因此 restore transport/state mechanism 为 PASS，restore policy/timing 为 FAIL。

## 冻结设置

- 模式：`atomic`
- carrier：A / `text-json`
- provider/model：官方 DeepSeek / `deepseek-v4-flash`
- teacher guidance：`restore-trigger-v1`
- cohort：teacher1500 positions 87–94，与此前 Gate8/Gate24 不重叠
- 两个并行 4 题 shard；单 episode 内因果串行
- 每 episode 最多 2 checkpoint nodes、1 restore、20 model turns、30 primitive calls
- 每 shard token 硬上限 1,250,000；授权总硬上限 2,500,000
- 实际 provider tokens：1,743,602
- admission：diagnostic-only，不进入 SFT/RL

结果目录：

- `data/results/checkpoint_relalg_v1_flash_text_json_atomic_restore_trigger_gate8_20260809/shard_087_090`
- `data/results/checkpoint_relalg_v1_flash_text_json_atomic_restore_trigger_gate8_20260809/shard_091_094`

8/8 records 通过 structure、cohort identity、manifest binding、provider budget、batch control 与 fresh replay。

## 同题配对

| 指标 | 默认 Atomic 基线 | Restore trigger | 变化 |
|---|---:|---:|---:|
| bird-set 正确 | 5/8 | 7/8 | +2 |
| strict artifact 正确 | 2/8 | 4/8 | +2 |
| schema match | 4/8 | 5/8 | +1 |
| legal termination | 7/8 | 7/8 | 0 |
| model turns | 70 | 85 | +15（+21.4%） |
| tool errors | 9 | 7 | -2 |
| provider tokens | 1,405,153 | 1,743,602 | +338,449（+24.1%） |
| commit calls | 0 | 4 | +4 |
| restore calls | 0 | 1 | +1 |

paired outcome 为 positions 87、91 两个 gains、0 regressions。两条 gain 轨迹均未使用 restore，因此不能把正确率提升归因于恢复机制。

## 唯一 restore 事件

Position 93 的公开过程序列：

1. turn 2 `filter_rows`：`canonical_type_mismatch`；
2. turn 3 `project`：同一错误；
3. turns 4–6：三个成功 perception actions；
4. turn 7：`restore_checkpoint(checkpoint_id="root")`；
5. Harness 创建 `checkpoint_001 / phase_001`，phase history 正确清空；
6. 该 restore 没有 abandoned checkpoint；
7. turn 16 才执行第一次 `commit_checkpoint`，进入 `checkpoint_002 / phase_002`；
8. turn 20 未形成终端答案，以 `max_model_turns` 失败。

这说明模型理解了“重复类型错误可触发 restore”的表面规则，却没有遵守“只在 commit 后恢复 phase-start checkpoint”的语义边界。恢复 root 发生太晚，并丢弃了触发后得到的有效观察；它不是期望的 `commit -> bad branch -> restore committed snapshot -> changed branch`。

## 判定

- Restore tool provider carrier / validation / execution：**PASS**。
- Snapshot hash、membership、phase transition、history reset、fresh replay：**PASS**。
- 模型能否主动发出 restore：**PASS（1 个真实事件）**。
- Restore target 选择：**FAIL**。
- Restore trigger latency：**FAIL**。
- Restore 后语义收益：**FAIL / 未观察到恢复**。
- 将 `restore-trigger-v1` 设为默认或准入训练：**REJECT**。

## 下一步

若继续，只应做最小策略修订，不再扩大样本：

1. restore 只能指向当前 active path 上最近一个由 `commit_checkpoint` 创建的节点；第一次 commit 前禁止 restore root；
2. 连续第二次合格错误后立即 restore，不能在出现成功 data/perception action 后延迟补做；
3. 若错误调用没有改变逻辑状态，不能仅为了清空历史而 restore root；
4. 后续验证应要求出现 `commit -> successful branch artifact -> contradiction/error -> restore committed snapshot -> different successful branch` 的完整序列。

该修订仍应保持 teacher-only、diagnostic-only，并使用新的 prompt/profile identity；不得改写本 Gate8 artifact。
