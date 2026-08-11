# Checkpoint-RelAlg 模型验证阶段 v3 Holdout Gate8（2026-08-11）

## 结论

`model-choice-commit-v3` 在 disjoint holdout Gate8 中再次出现零 checkpoint：8/8 题、90 turns
没有一次 commit 尝试。它得 2/8，fresh checkpoint-disabled 对照为 3/8；tokens 为 2,132,990
对 1,624,193（+31.3%），tool/process errors 为 11 对 6。v3 失败于 exposure、reliability 和
efficiency gate，应冻结，不能扩大或进入 SFT/RL。

## 设计与结果

两臂均为 Atomic `semantic-v2`、A/Text-JSON、官方 `deepseek-v4-flash`，positions
`6,18,20,26,32,33,35,37`。v3 继续使用 Harness eligibility=`none`、max checkpoints=8、
restore=0；唯一变化是要求模型在 commit 前对最后形成的候选 artifact 做成功的
`read_rows`/`inspect_column` 验证，错误后验证修正结果，并在只剩最终整理/回答时跳过 commit。

| 指标 | model-choice v3 | disabled |
|---|---:|---:|
| correct | 2/8 | 3/8 |
| strict artifact | 1/8 | 2/8 |
| schema match | 4/8 | 5/8 |
| legal | 6/8 | 7/8 |
| turns | 90 | 76 |
| provider attempts | 100 | 85 |
| total tokens | 2,132,990 | 1,624,193 |
| errors | 11 | 6 |
| checkpoint episodes / commits | 0 / 0 | 0 / 0 |
| token-cap stops | 2 | 0 |
| provider failures | 0 | 1 |

配对结果为双方都对 2、仅 v3 对 0、仅 disabled 对 1、双方都错 5。16/16 record structure、
manifest/cohort binding 与 fresh replay 通过；14/16 overall audit 通过，两个失败均为 v3 的
per-episode token cap stop。全批使用 3,757,183 tokens，低于 6.8M 名义上限。

## 解释

v2 已证明模型自主规则能够制造 checkpoint exposure（5/8 题、7 次成功 commit），但没有准确率
收益。v3 把“verified stage”操作化为每次 commit 前的强制 perception，结果不是更高质量的
checkpoint，而是更多 turns/tokens 与完全不提交。这表明 checkpoint 策略不应再叠加机械前置链。

下一步返回 v2 的模型自主语义选择，只保留两个窄的否定边界：当前 phase 出错后，修正结果尚未
验证时不能 commit；如果预计只剩一个最终变换加 answer，则直接完成而不 commit。不再要求所有
候选阶段额外 read/inspect，Harness 仍不决定时机或计算 producer quota。

结果目录：
`data/results/checkpoint_relalg_v1_flash_text_json_atomic_model_choice_v3_vs_disabled_holdout_gate8_20260811/`
