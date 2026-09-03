# Atomic version26 工具协议

更新时间：2026-09-03

这是训练、评测和 RL 唯一采用的 model-visible protocol。完整 Harness 实现以代码和冻结
runtime 为准；本页只保留当前契约，不再列举历史版本号。

## 身份

- protocol：`version26`
- runtime export commit：`4cd47c957fc6ae791e76a10594c8cd22f4d3b6de`
- protocol hash：`4da19387399bd3a5`
- carrier：`think-json-v1`
- student prompt SHA-256：`848598074e653648d52b58dfa1a54dd777beabae16cb91d83c4ba1a54b5d7316`
- history：rolling legal history 最近 4 轮，附当前完整 Harness resident state
- limits：最多 30 semantic steps；单轮最多 2,048 new tokens

## 模型输出

每个 assistant turn 必须包含一个非空 think block 和一个 action JSON，且只能有一个 action：

```text
<think>short private reasoning</think>
{"tool":"<official_tool>","arguments":{...}}
```

严格 parser 拒绝额外 tagged action、多个 JSON action、空 think、未知 tool 或不符合 schema 的
arguments。tool call 之后模型只能基于 Harness 返回的 observation/error 继续；不能引用未观察
的 schema、结果或后续轨迹。

## 工具边界

- perception：读取当前可见 schema/rows/subtable；不隐式创建 filtered handle。
- relational：join、project、group/aggregate、scalar compute 等 typed 操作；输入必须是
  当前 resident state 中已存在且可验证的 relation handle/column。
- terminal：只提交一个由 Harness 产生的 exact result artifact 和必要的列/shape 引用；模型
  不携带自己计算的答案值。
- Harness 在 SQLite 执行调用，验证 table/column ownership、predicate operands、join edge、
  expression 和 terminal evidence；错误以结构化 feedback 返回，计入 shared action budget。
- relation derivation/provenance 是 Harness-owned 事实，不能由 reasoning 覆盖。

实现位置：

- action schema/parser：`src/sft/protocol.py`、`src/sft/prompt_contract.py`
- environment：`src/rl/tool_environment_v26.py`
- relation facts：`src/harness/relation_derivation/`
- frozen runtime：`/home/dengyan/tabular_rl_outputs/runtime/version26-4cd47c957fc6ae791e76a10594c8cd22f4d3b6de`

## 身份和审计

每个 trajectory、SFT record、RL run 和 evaluator result 必须绑定 protocol hash、prompt hash、
runtime tree hash、model revision、carrier、history、cohort 和 scorer。Gold SQL 只允许留在
Harness 内部用于 denotation/兼容检查，禁止进入 provider request、student context、SFT target
或 reward。

## 当前不支持的入口

checkpoint-relalg、Atomic v24/v39/v51/v54、Direct/Hybrid/iterative-SQL、native-tool-bundle、
projection/rewrite/delete 及任何版本迁移脚本均为历史诊断或 replay 兼容，不得作为新训练/评测
默认协议。需要复现历史结果时，必须使用其原始 identity 和隔离 output root。
