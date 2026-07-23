# BIRD 条件聚合 clean-8 与命名标量引用 pilot

日期：2026-07-23

## 目的

在不改变 resident plan 实现、可见性或 optional 默认策略的前提下，验证两件事：

1. version18 的条件聚合与宽输出是否能在一组明确的条件计数/比例题上稳定保留旧正确题并
   恢复旧失败题；
2. 一行多指标聚合能否直接作为后续标量算术的输入，避免把同一聚合拆成多个 1x1 分支。

Gold SQL 只用于本地 verifier 和离线错误审计，没有进入 actor/teacher prompt。在线调用均
使用 DeepSeek v4 Flash、temperature=0、max tokens=2048、max steps=30、
rolling legal history=4、optional plan、strict no-repair parser 和 tool-call carrier。

## version18 clean-8

冻结样本：

`00165, 00770, 01308, 03145, 00600, 01050, 01692, 05873`

其中前四题是 version11 正确对照，后四题是 version11 失败题。结果：

| 指标 | version11 | version18 |
| --- | ---: | ---: |
| 正确 | 4/8 | **7/8** |
| 合法终局 | 8/8 | **8/8** |
| 保留旧正确题 | — | **4/4** |
| 恢复旧失败题 | — | **3/4** |

version18 结果：

`data/trajectories/tool_usability_20260723/conditional_clean8_version18.*`

恢复的三题是 `00600`、`01692`、`05873`。唯一错误 `01050` 把利润理解为
`(Unit Price - Unit Cost) * Order Quantity`，而题目 external knowledge 与 gold 都明确要求
`Unit Price - Unit Cost`；这是任务语义选择错误，不是工具无法表达。

## version19：命名标量引用

version18 已能一次产生：

```json
{
  "table": "group_003",
  "columns": ["total_nominees", "usa_nominees"],
  "row_count": 1
}
```

但原 `scalar_compute` 只接受 1x1 producing step。模型在 `01692` 上先错误引用该 1x2
结果，再把同一聚合拆成两个 1x1 调用。version19 增加：

```json
{
  "operation": "percent",
  "operands": [
    {"value_ref": "step_6", "column": "usa_nominees"},
    {"value_ref": "step_6", "column": "total_nominees"}
  ],
  "result_name": "percentage"
}
```

这不是模型提供数值。Harness 验证 producing step 指向一个驻留表、表恰好一行、列名唯一
匹配、cell 非 NULL，再读取真实 cell。每个 operand 的 provenance edge 同时记录 step、
operand index 和列名；省略 `column` 时仍保持原来的严格 1x1 规则。

### 目标题 `01692`

结果：

`data/trajectories/tool_usability_20260723/named_scalar_01692_version19.*`

| 指标 | version18 | version19 |
| --- | ---: | ---: |
| 正确 | 1/1 | **1/1** |
| 总动作 | 13 | **8** |
| 错误动作 | 2 | **0** |
| 重复 1x1 聚合 | 2 | **0** |
| 总 tokens | 63,861 | **39,340** |

version18 的两个错误分别是一次 provider visible-content carrier error 和一次把同一个
多列 `read_subtable` step 同时作为两个 scalar operand 的 execution error。version19 的
模型在第一次尝试中自然使用 `step_6 + column`，得到
`66.10169491525424`，没有 read-back、重复聚合或恢复动作。

### 正确题对照

另外复测两个原本正确的比例题：

- `00165`：1/1，8 actions，0 semantic errors；
- `03145`：1/1，10 actions，0 semantic errors；记录的两次 errors 是 API transport retry，
  不是模型动作或工具错误。

两题都自然把 total 与 conditional count 合并到一个一行多列 `group_aggregate`，随后用
同一步的两个命名列完成 `percent`。三个 version19 小样本合计 **3/3**，没有观察到旧
1x1 语义回归。

## 当前结论与下一 gate

1. `group_aggregate(where=...)` 与宽输出合并方案在 clean-8 上从 4/8 提升到 7/8，并保留
   4/4 旧正确题。
2. `scalar_compute(value_ref+column)` 是一个真实接口修复：它没有合并 aggregation 与
   arithmetic 的语义边界，却把同一驻留结果的引用粒度细化到列，显著缩短正确轨迹。
3. 模型在 3/3 小样本中无需错误反馈就自然采用新形状，说明参数符合调用直觉。
4. 这些小样本仍不足以宣称达到 75%。进入固定 200 前，应先在与 version11 完全相同的
   JSON Output carrier、rolling history 和首 30 题上达到至少 24/30；否则不扩展，也不构造
   新 SFT 数据。

## 回归

- Harness：64/64
- SFT：78/78
- RL：35/35
- Eval：25/25

覆盖范围包括 strict operand schema、metadata-only 一行表的 harness-owned cell 读取、
NULL/行数/列名拒绝、replay plan 执行和逐列 value provenance。Plan 行为没有改变。
