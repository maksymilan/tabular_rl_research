# checkpoint-relalg Initial-Target 有序目标诊断（2026-08-11）

## 结论

本轮验证了 checkpoint 的控制状态机可以工作，但尚未证明它提高答案正确率。最终的 `initial-target-v4` 能在只读感知和无状态变化的失败调用之后、首个成功关系产物之前建立 bootstrap checkpoint，并按有序 target 逐阶段推进。它在代表性长轨迹上明显减少 turns 和 tokens，但最终答案仍错。因此该线继续保持 `diagnostic-only`，不得进入 SFT/RL。

## 冻结实现

- `98aab59c199d055668f073deba3d7d587f263e4b`：`initial-target-v2`，有序 active/remaining targets；达到 producer quota 后强制迁移。
- `101c2270c46a378a8409e53c5eccf8000ea1df11`：`initial-target-v3`，允许只读感知后、生产前 bootstrap。
- `b27f13df4f80a4d4001e065a9a637e0280cac3b6`：`initial-target-v4`，无状态变化的失败调用不关闭 bootstrap 窗口。

三版均保留 Atomic `semantic-v2`、最多 8 次 checkpoint、目标 distinct 约束与可选 restore；未改变关系算子语义、工具 schema 或 verifier。

## Gate3：initial-target-v1 vs ordered-target-v2

数据位置为 31（简单 control）、47/54（长轨迹 hard targets）。全部使用官方 `deepseek-v4-flash`、Atomic `semantic-v2`、Text-JSON carrier。6/6 records 的结构、cohort、provider identity 与 fresh replay 通过。

| arm | correct | legal | turns | tokens | checkpoints | tool errors |
|---|---:|---:|---:|---:|---:|---:|
| initial-target-v1 | 1/3 | 3/3 | 34 | 573,107 | 4 | 2 |
| ordered-target-v2 | 1/3 | 3/3 | 37 | 691,985 | 2 | 7 |

简单位置 31 在两臂均正确、5 turns、0 checkpoint，说明简单题零 bootstrap 路径保持。两个 v2 hard episode 都先做感知，之后才尝试 commit；由于 v2 只接受绝对第一步 bootstrap，提交被当成普通 evidence checkpoint 并以 `checkpoint_phase_progress_insufficient` 拒绝。有序 target 生命周期因而没有可靠进入。

## Gate2：ordered-target-v2 vs perception-bootstrap-v3

fresh hard positions 47/54。所有 record 的结构与 fresh replay 通过；v2 位置 54 达到 448,758 tokens 后触发 425k batch cap，因此该 pair 不是完整语义配对。

| arm | correct | legal | turns | tokens | checkpoints | tool errors |
|---|---:|---:|---:|---:|---:|---:|
| ordered-target-v2 | 0/2 | 1/2 | 35 | 769,935 | 5 | 4 |
| perception-bootstrap-v3 | 0/2 | 2/2 | 29 | 501,686 | 3 | 7 |

v3 位置 54 成功执行 `describe_table → bootstrap(3 targets) → 两个 producer → evidence checkpoint(剩余2 targets)`，以 12 turns / 191,088 tokens / 0 errors 合法终止，但答案仍错。位置 47 在 bootstrap 前出现 Text-JSON 格式拒绝；v3 错误地把该无状态变化 step 视为窗口关闭，仍未建立 bootstrap。

## v4 单点机制复测

位置 47 的 v4 轨迹为：

`describe_table → inspect_column → failed filter_rows(no artifact) → bootstrap(3) → evidence checkpoint(2) → rejected target transition → corrected evidence checkpoint(1) → answer`

该轨迹 15 turns、227,052 tokens、2 errors、3 accepted checkpoints，合法但回答错误。全部审计通过。与同位置 fresh v2 的 18 turns、321,177 tokens、2 errors、3 checkpoints、同样错误相比，v4 减少 3 turns 和 94,125 tokens（约 29.3%）。这证明：

1. perception/bootstrap 入口、active/remaining target 渲染、quota gate 与 target-transition validator 可共同工作；
2. 模型可从 `checkpoint_remaining_target_lost` 恢复并完成剩余 target 迁移；
3. checkpoint 能改善部分长轨迹的组织和成本，但不能自动纠正错误 population、join grain、aggregation 或 exact-output 决策。

## 决策

- 保留 checkpoint 与最多 8 次上限；restore 继续可选，不作为成功条件。
- 冻结 v1-v4 artifacts，禁止混合或原地重放为同一协议身份。
- 不继续仅靠强化 checkpoint 文案扩题。下一步若继续，应把重点放在每个 active target 的可验证完成条件：checkpoint 只能在对应 answer-relevant artifact 被 Harness 证实时推进，而不是只按 producer 数计数。
- 在新的 paired accuracy gate 通过前，不得把这些轨迹用于 SFT/RL。
