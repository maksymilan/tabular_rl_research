# BIRD relational-program v5 类型化 scalar 6 题门控

日期：2026-07-27

模型：DeepSeek v4 Flash，thinking enabled，reasoning effort high，temperature 0

指标：`bird-set`

数据准入：`diagnostic_only_pending_protocol_scale_gate`

## 预注册门

v5 相对已失败的 v4 只改变 scalar-value 表面：

- scalar operand 直接使用 `{"node":"..."}` 或 `{"resident_step":"..."}`，可带一个
  `column`；
- filter/aggregate predicate 通过 `value_from` 使用同一 typed source；
- compiler 确定性降为未改变的 harness-grounded producing-step 语义；
- 模型可见错误把私有 atomic 名和 `$node` 引用反向映射为公开 operation 和 typed ref。

冻结集合：

- scalar 目标：`00593`、`05440`；
- v4 正确控制：`05952`、`03275`、`02437`、`04619`。

扩展门要求：至少恢复一个 scalar 目标，同时 4/4 控制保留。未满足时立即停止，不跑
20/50/200。

## 结果

| 指标 | v5 gate6 |
|---|---:|
| 正确 | 3/6 |
| 合法结束 | 5/6 |
| 目标恢复 | 0/2 |
| 控制保留 | 3/4 |
| 模型轮次 | 38 |
| 原子动作 | 52 |
| process error | 7 |
| blocked | 2 |
| 总 token | 184,182 |
| 耗时 | 152.7s |

门控失败，不扩展。

## 公开轨迹审计

- 目标题 `00593` 已实际采用直接 typed scalar operand，说明新语法可被模型调用；
- 初始 scalar source 有两行而非一行，grounding 正确拒绝；
- 模型随后把两行分别过滤，但使用 bare `START` / `STOP` 指向带 namespace 的本轮节点
  输出，仍未形成合法 scalar；
- 目标题 `05440` 没有选择 scalar 路径，而是选择 rank→join。该路径暴露一个确定性
  compiler bug：base 为本轮 node 且指定 `base_role` 时，`on.left` 仍被内部 resolver
  降为动态 handle namespace，未采用模型声明的 role namespace；
- 控制题 `05952` 出现一个运行时列错误并翻转；另外三个控制保留。

模型可见 `turns[].observation` 与后续 `model_input` 已审计：没有私有 atomic 工具名、
`action_block` 或 `$node` 泄漏。审计级 `error_events` 保留私有执行消息，用于开发者
复现，但不会发给模型。

## 结论

统一 typed scalar 语法本身没有恢复目标能力，不能推广。下一步只允许修复
`base_role + current node` 的确定性 namespace lowering；这是公开参数执行正确性问题，
不是模型语义补丁。修复后只重测 `05440` 加少量控制。`00593` 的多行 scalar 和列
namespace 问题不在同一版本继续修改。

## 产物

- 轨迹：
  `data/results/relational_program_v5_typed_scalar_gate6_deepseek_v4_flash_20260727/all.jsonl`
- manifest：
  `data/results/relational_program_v5_typed_scalar_gate6_deepseek_v4_flash_20260727/all.manifest.json`

所有产物均为 diagnostic-only，未进入训练数据。
