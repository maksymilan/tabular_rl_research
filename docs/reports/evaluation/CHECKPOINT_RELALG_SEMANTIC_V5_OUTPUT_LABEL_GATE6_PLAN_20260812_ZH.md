# Semantic-v4-interface vs Semantic-v5-output-label Gate6 plan

日期：2026-08-12
状态：已完成；结果见
`CHECKPOINT_RELALG_SEMANTIC_V5_OUTPUT_LABEL_GATE6_RESULT_20260812_ZH.md`

## 目标

V4 Target Gate22 证明省略 `as` 时保留 exact logical column 能消除错误拒绝，但暴露终端标签
选择问题：join role prefix 是机械 provenance，通常应在最终输出显式去掉；真实含空格 source
column 则应原样保留。V5 只新增这条 teacher/student compact output-name rule，执行器、工具、
schema、checkpoint、recent-4 与 v4 完全相同。

## 冻结 cohort 与预算

- positions：`[233,245,252,263,278,293]`
- 选择依据：V4 Gate22 中 strict/schema/legal 任一 paired 分歧；不读 question/gold/reasoning
- 6 pairs / 12 paid episodes；v4/v5 fresh，交替 pair order，最多四路并发
- official endpoint/model：`https://api.deepseek.com/chat/completions` / `deepseek-v4-flash`
- Atomic / A Text-JSON / `model-choice-commit-v6`
- 每 episode：20 turns、30 primitive calls、8 checkpoints、3 restores、50 attempts、350k tokens
- 总硬上限：4,200,000 tokens

## Gate

V5 必须：correct 不低于 v4；strict/schema paired v4-only=0；legal 不低于 v4；
`invalid_identifier=0`；errors 不高于 v4；tokens 不超过 1.10x。所有 records 必须通过 current
identity、structure、cohort/budget 与 fresh replay。未通过则冻结，不扩大；无 SFT/RL admission。
