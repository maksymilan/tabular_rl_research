# Checkpoint-RelAlg V1 Atomic Checkpoint Stress Gate24（2026-08-09）

## 结论

扩展 Gate24 进一步确认：checkpoint 状态机与 commit 行为可用，但当前 teacher-only guidance 尚未形成可观察的 restore 行为。

- 24/24 episode 完成，全部记录通过 structure、cohort identity、manifest binding、provider budget、batch control 与 fresh replay。
- 15/24 episode 使用 checkpoint，共 20 次 commit；0 次 restore。
- commit 之后没有任何 episode 立即机械 answer。
- 唯一明显接近 restore 触发条件的样本是 position 20：`checkpoint_001` 之后连续三次 `canonical_type_mismatch`，涉及 `filter_rows`、`project`、`aggregate`。模型没有 restore，而是在当前 phase 继续修复，随后虽有成功操作但在 20 turns 上限前未终止。
- position 22 和 35 的 checkpoint 后错误均为 carrier/shape 错误；按冻结 guidance 不应 restore。

因此：**Harness restore 语义仍为确定性 PASS；自然 Atomic 轨迹中的模型 restore 决策仍未通过行为验证。** 继续单纯扩大自然任务数量的边际价值较低；下一步应采用独立、受控的 restore-trigger gate，而不是要求所有困难题机械 restore。

## 冻结设置

- 模式：`atomic`
- carrier：A / `text-json`
- provider/model：官方 DeepSeek / `deepseek-v4-flash`
- teacher guidance：`checkpoint-stress-v1`
- cohort：teacher1500 positions 12–35，与 Gate8 positions 64–71 不重叠
- 4 个并行 shard，每个 6 题；单 episode 内因果串行
- 每 episode 最多 2 commit、1 restore、20 model turns、30 primitive calls
- 每 shard token 硬上限 1,750,000；全批授权硬上限 7,000,000
- 实际 provider tokens：5,029,484
- admission：diagnostic-only，不进入 SFT/RL

结果目录：

- `data/results/checkpoint_relalg_v1_flash_text_json_atomic_checkpoint_stress_gate24_20260809/shard_012_017`
- `data/results/checkpoint_relalg_v1_flash_text_json_atomic_checkpoint_stress_gate24_20260809/shard_018_023`
- `data/results/checkpoint_relalg_v1_flash_text_json_atomic_checkpoint_stress_gate24_20260809/shard_024_029`
- `data/results/checkpoint_relalg_v1_flash_text_json_atomic_checkpoint_stress_gate24_20260809/shard_030_035`

## Gate24 同题配对

| 指标 | 默认 Atomic 基线 | Checkpoint stress | 变化 |
|---|---:|---:|---:|
| bird-set 正确 | 15/24 | 13/24 | -2 |
| strict artifact 正确 | 10/24 | 8/24 | -2 |
| schema match | 12/24 | 11/24 | -1 |
| legal termination | 22/24 | 22/24 | 0 |
| model turns | 224 | 238 | +14 |
| tool errors | 20 | 18 | -2 |
| provider tokens | 5,370,280 | 5,029,484 | -340,796（-6.3%） |
| checkpoint calls | 0 | 20 | +20 |
| restore calls | 0 | 0 | 0 |

配对 bird-set outcome 为 0 gains、2 regressions（positions 26、32），其余 22 题不变。小样本下不能将差异解释为显著准确率退化，但本 Gate24 不支持 checkpoint stress 提升正确率。

## Restore 机会审计

Checkpoint 后共有三个 error-bearing episode：

1. position 20：三次连续 `canonical_type_mismatch`，均发生在 `checkpoint_001 / phase_001`。这是唯一语义/表示层 restore 候选；restore 未发生。
2. position 22：两次 `invalid_shape`。这是 carrier/argument envelope 问题，状态分支未变化；按 guidance 不 restore。
3. position 35：两次 `invalid_shape`。同样不属于语义分支回滚条件。

对 position 20 也需保留边界：这些失败均保持状态不变，模型最终在同一 phase 得到一次成功 aggregate，说明局部修复并非完全不可行。因此它是明确的 stress-trigger miss，但还不是“restore 必然提高最终正确率”的反事实证明。

## Gate8 + Gate24 合并描述

两个不重叠 gate 合计 32 题：

| 指标 | 默认 Atomic 同题基线 | Checkpoint stress | 变化 |
|---|---:|---:|---:|
| bird-set 正确 | 19/32 | 19/32 | 0（2 gains / 2 regressions） |
| strict artifact 正确 | 13/32 | 12/32 | -1 |
| schema match | 16/32 | 16/32 | 0 |
| legal termination | 28/32 | 30/32 | +2 |
| model turns | 303 | 314 | +11（+3.6%） |
| tool errors | 26 | 20 | -6 |
| provider tokens | 7,173,496 | 6,468,036 | -705,460（-9.8%） |
| checkpoint calls | 0 | 28 | +28 |
| restore calls | 0 | 0 | 0 |

22/32 stress episode 至少 commit 一次。总体上，checkpoint guidance 降低了 tokens 和错误数、提高了合法终止，但没有提高正确率，并略增加模型回合数。

## 当前判定

- Checkpoint snapshot/restore 状态语义：**PASS**。
- Atomic 模型主动建立语义 phase：**PASS**。
- Commit 的总体行为收益：**MIXED / diagnostic-only**。
- Atomic 模型主动 restore：**NOT DEMONSTRATED**。
- 将 stress guidance 设为默认或准入 SFT/RL：**REJECT**。

下一步若继续，应只做小型受控 restore-trigger gate：由 Harness 在 commit 后制造一个明确、状态保持且不泄露 gold 的分支矛盾信号，验证模型是否 restore、恢复后的 active membership/hash 是否正确、以及是否避免重建同一错误分支。该实验应与自然正确率评测分开报告。
