# Harness 执行契约

更新时间：2026-09-03

当前训练、评测和 RL 使用 Atomic version26 的 Harness。代码实现以
`src/rl/tool_environment_v26.py`、`src/sft/protocol.py` 和冻结 runtime 为准。

## 执行边界

1. 模型每轮只能发一个符合 version26 schema 的 action；parser 先校验 carrier/schema，再
   交给 Harness。
2. Harness 只执行当前 state 中可验证的 relation/table/column/artifact 引用，维护 resident
   state 和 relation derivation/provenance。
3. SQLite 执行结果、结构化 observation/error 和 exact result artifact 由 Harness 产生；模型
   reasoning 不得覆盖这些事实。
4. terminal 只能引用 Harness 产生的 artifact/列/shape。正确性使用 `bird-set`，不接受模型
   自报答案值或 gold path。
5. 每个 error 结构化记录并计入 shared action budget；transport failure 与 semantic
   argument error 分开审计。

## 因果和隐私

模型/teacher 只能看到合法 episode prefix、当前 resident state 和最新 feedback。Gold SQL、
gold result、后续轨迹和 privileged schema 只留在 Harness 端，不能进入 provider request、
SFT target 或 reward。

## 身份

运行必须绑定 protocol `version26`、runtime commit
`4cd47c957fc6ae791e76a10594c8cd22f4d3b6de`、protocol hash `4da19387399bd3a5`、prompt
SHA、carrier、history、model、cohort 和 scorer。不同身份必须隔离 output root。

## 历史兼容

旧 checkpoint-relalg、native-tool-bundle、Atomic v39/v51/v54 及 SQL scheme 只允许原始
runner 在隔离目录 replay；它们不是当前 Harness 的新实验入口，也不具备 SFT/RL admission。
