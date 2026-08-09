# 当前原子工具集详细说明书（version54）

本文描述截至 2026-08-08 仓库中当前前向实验工具集：
`version54 / native-tool-bundle`。它用于理解现有设计、审查模型行为以及设计后续版本，
不是对历史轨迹兼容接口的汇总。

## 1. 协议身份与源码口径

| 项目 | 当前值 |
|---|---|
| Tool scheme | `native-tool-bundle` |
| Protocol | `version54` |
| 原子执行语义基线 | `version39` |
| Provider carrier | `deepseek-native-tool-bundle-v1` |
| 模型可见工具数 | 11 |
| 硬性 bundle 上限 | 每个 assistant turn 1–8 个调用 |
| Prompt 软约束 | 默认 1 个，通常不超过 3 个 |
| 历史窗口 | 最近 4 个真实 provider assistant turn 及其全部工具结果 |
| 终止证据 | 一张精确的 grounded table |
| 当前准入状态 | diagnostic-only，尚未提升为 SFT/RL 正式协议 |

`version54` 并没有重新实现一套关系算子。它继承 `version39` 的工具参数、状态校验、
SQLite 执行和 resident state，只从公开工具面删除了 `plan`，并通过 DeepSeek 原生
`tools/tool_calls` 传输一次一个或多个独立原子调用。

当前工具 schema hash：

```text
fc3ab5950dd09bbbe0860bc3eced03bd746921e59deff2c92c2d3823042416ba
```

可执行源码优先级高于本文：

- v54 工具集合：`src/tool_modules/native_tool_bundle/no_plan_protocol.py`
- Provider JSON Schema：`src/tool_modules/native_tool_bundle/provider_tools.py`
- 公共工具语义：`src/sft/public_tool_contract.py`
- 静态参数校验：`src/sft/protocol.py`
- 状态感知校验与工具分发：`src/eval/rollout.py`
- SQLite 执行：`src/harness/executor.py`
- Resident state：`src/harness/environment_state.py`
- 派生关系事实：`src/harness/relation_derivation/`
- Native bundle 执行循环：`src/sft/generate_teacher_rollouts.py`

## 2. 整体工作模式

一个 episode 的核心状态转换为：

```text
QUESTION + EXTERNAL KNOWLEDGE + catalog
                  │
                  ▼
       CURRENT ENVIRONMENT STATE
                  │
                  ▼
 provider assistant turn: reasoning_content + 1..8 tool_calls
                  │
                  ▼
 所有调用先对同一个 bundle-pre-state 做静态和状态校验
                  │
                  ▼
 按 provider 顺序逐个执行，逐个返回 role=tool 结果
                  │
                  ▼
 更新 resident state，进入下一模型轮次
                  │
                  ▼
 answer_from_context 单独终止并引用一张精确结果表
```

### 2.1 Bundle 语义

1. 一个模型轮次必须包含 1–8 个 native `tool_calls`。
2. 同一 bundle 内每个调用都只能引用模型发起该轮时已经存在的表、列、step 和值。
3. 同一 bundle 后面的调用不能消费前面调用刚产生的 handle 或 scalar，因为模型尚未观察
   前一个结果。
4. 所有调用先基于共同 pre-state 校验，再按 provider 顺序执行。
5. 同一 bundle 内完全相同的 `tool + arguments` 不能重复。
6. 一个调用校验失败时，该调用返回错误且不改变状态；同 bundle 中其他合法、独立调用仍可
   执行。
7. 如果某次执行在失败前意外改变状态，会被标为 `nonrecoverable_execution_error`，其后的
   bundle 调用被阻止。
8. `answer_from_context` 必须是该 assistant turn 的唯一调用。

Prompt 的软策略是：默认只调用一个工具，通常 bundle 只用于相互独立的 schema 检查、列值
检查、行读取或不同 resident handle 上的独立筛选。Join、aggregate、project、scalar、ranking
和 set operation 通常应单独执行，让模型先观察结果再决定下一步。

### 2.2 v54 与原始 atomic version39 的区别

| 维度 | version39 `atomic` | version54 `native-tool-bundle` |
|---|---|---|
| 调用载体 | `<think>` + 一个 raw JSON action | Provider `reasoning_content` + `tool_calls` |
| 每个模型轮次 | 恰好一个调用 | 1–8 个调用 |
| `plan` | 可见 | 已删除 |
| 原子工具执行 | version39 | 继承 version39 |
| Resident state | version39 | 继承 version39 |
| 表和列语义 | version39 | 继承 version39 |
| 错误反馈 | 单调用错误 | 每个 call id 独立返回结果或错误 |

不要把一个 v54 多调用 assistant turn 展平成多个虚构的 assistant turn。每个 primitive step
保留共同的 `model_turn_index` 和各自的 `native_tool_call_id`。

## 3. 当前工具总览

| 类型 | 工具 | 主要作用 | 是否产生新 handle |
|---|---|---|---|
| Schema perception | `describe_table` | 查看表结构、PK、FK、行数 | 否 |
| Value perception | `inspect_column` | 查看一列的 distinct 数、常见值、NULL | 否 |
| Row perception | `read_subtable` | 条件读取最多 20 行 | 否 |
| Relational | `condition_filter` | 按 typed predicate 筛选行 | 是 |
| Relational | `project` | 选择、排序、重命名或计算输出列 | 是 |
| Scalar | `scalar_compute` | 对 grounded scalar 做计算，生成 1×1 表 | 是 |
| Relational | `join_tables` | 连接一个有序、连通的多表组件 | 是 |
| Relational | `group_aggregate` | 分组、全局或条件聚合，可做宽表输出 | 是 |
| Relational | `extreme_value_select` | 排序并选取 top-k 行 | 是 |
| Relational | `set_op` | 集合并、交、差 | 是 |
| Terminal | `answer_from_context` | 引用精确结果表并终止 | 终止 |

v54 不公开以下工具：

- `plan`：已在 v54 删除；
- `search_values`：只属于 version44/version45 诊断分支；
- `inspect_rows`：version40–45 使用的重命名接口，v54 仍叫 `read_subtable`；
- `aggregate`、`pivot`：仅历史 replay；
- 旧的 binary join 参数：仅历史 replay。

## 4. 公共数据对象

### 4.1 Table 与 handle

`table`、`base`、`left`、`right` 等字段可引用：

- Catalog 中的 source table 名；
- Harness 已经产生并放入 resident state 的 derived handle，例如 `filter_001`、
  `join_002`、`group_003`。

模型不能自己预测或发明 derived handle。后续调用必须使用环境实际返回的 handle。

### 4.2 列名

- Source table 使用 schema 中看到的列名。
- 普通 derived table 使用其返回的 `columns`。
- Join 输出采用扁平逻辑名 `relation.column`，例如 `orders.customer_id`。
- Harness 在无歧义时支持部分 bare-name 解析，但模型应优先复制环境显示的精确逻辑列名。
- 列顺序是终止答案的一部分；reasoning 不能调整或解释错误的列顺序。

### 4.3 Step 与 `value_ref`

Harness 为每次 primitive 调用分配 `step_n`。模型只能引用已经观察到的 producing step：

```json
{"value_ref":"step_7"}
```

如果 `step_7` 产生一行多列的 metric table，必须同时指定列：

```json
{"value_ref":"step_7","column":"total_count"}
```

不指定 `column` 时，来源必须能够解析为一个唯一的 1×1 scalar。Schema 查看、列检查、行读取、
reasoning 和已删除的 plan 都不能作为 `value_ref`。

### 4.4 Typed predicate grammar

`condition_filter.conditions`、`read_subtable.conditions` 和
`group_aggregate.aggregations[].where` 共用同一套 predicate tree。

逻辑组合：

```json
{"and":[PREDICATE_1,PREDICATE_2]}
{"or":[PREDICATE_1,PREDICATE_2]}
{"not":PREDICATE}
```

顶层 list 会被当成隐式 AND。一个逻辑节点只能包含 `and`、`or`、`not` 中的一个。

叶子谓词：

| 类别 | `op` | 所需字段 |
|---|---|---|
| 字面量比较 | `=`, `!=`, `>`, `>=`, `<`, `<=` | `column` + `value` |
| 同行列比较 | 同上 | `column` + `column_value` |
| Grounded scalar 比较 | 同上 | `column` + `value_ref` |
| 字符串 | `contains`, `not_contains`, `like`, `not_like` | `column` + `value` |
| 日期 | `on_date` | `column` + `value` |
| 集合字面量 | `in`, `not_in` | `column` + `values`，也接受单个 `value` |
| 集合表 | `in`, `not_in` | `column` + `in_table` |
| 区间 | `between` | `column` + `low` + `high` |
| NULL | `is_null`, `is_not_null` | `column` |

下划线会在执行时规范化为空格，例如 `on_date` → `on date`。比较谓词必须在 `value`、
`column_value`、`value_ref` 中选择恰好一个。`in_table` 最好引用单列 derived handle；若为多列，
Harness 会尝试按目标列名找到唯一成员列，否则报错。

当前实现对省略 `op` 的叶子默认使用 `=`，但这没有在 Provider schema 中表达为显式默认值。
新数据和新设计应始终显式写出 `op`。

## 5. 通用输出与状态更新

### 5.1 成功 envelope

每个非终止工具的模型可见结果是：

```json
{
  "step_id": "step_4",
  "status": "success",
  "output": {}
}
```

对于 table-producing 工具，`output` 通常是：

```json
{
  "table": "filter_001",
  "kind": "filter",
  "columns": ["id", "name"],
  "row_count": 8,
  "derivation": {
    "schema": "relation-derivation-v1",
    "operator": "condition_filter",
    "inputs": [],
    "semantics": {}
  }
}
```

含义：

- `table`：Harness 分配的新 handle；
- `kind`：`filter/project/scalar/join/group/top/setop` 等执行类型；
- `columns`：完整、有序输出列；
- `row_count`：整个 derived relation 的行数；
- `derivation`：Harness 根据真实执行构造的 fact-only 关系语义，不是下一步建议。

默认不会把普通 derived table 的行直接放进输出。只有 1×1 结果会自动带一个 `rows` cell；
其他行必须调用 `read_subtable` 观察。评测参数可以请求 inline preview，但它不是 v54 的公共
语义保证。

### 5.2 Resident state

成功调用后，环境会持久保存已经暴露的事实：

- Source/derived table 的 schema、列、行数；
- `describe_table` 看到的 PK/FK；
- `inspect_column` 看到的列值域；
- `read_subtable` 看到的有限行；
- Derived handle 的 producing step；
- 每张派生表的 `relation-derivation-v1`；
- 可供 `value_ref` 使用的 scalar/metric 来源。

历史只保留最近四个真实 provider turn，但 `CURRENT ENVIRONMENT STATE` 是当前事实权威。
历史 reasoning 只是决策轨迹，不是数据库证据。

### 5.3 错误 envelope

模型可见错误为：

```json
{
  "step_id": "step_5",
  "status": "error",
  "error": {
    "type": "argument_validation_error",
    "message": "...",
    "code": "unknown_column",
    "details": {
      "argument_path": "condition_filter.conditions.column",
      "available_columns": ["id", "name"]
    }
  },
  "attempted_action": {
    "tool": "condition_filter",
    "arguments": {}
  }
}
```

正常的 protocol、参数和执行错误都应保持状态不变。Native bundle 会保留错误 call 的
assistant turn 和匹配的 `role=tool` 错误消息，使后续模型能够学习按环境反馈修复。

## 6. 工具详细说明

### 6.1 `describe_table`

作用：查看一个或多个可用表的结构信息，不读取行值，不创建新表。

输入：

| 参数 | 必填 | 类型 | 约束 |
|---|---|---|---|
| `tables` | 是 | `string[]` | 非空列表；每项必须是当前可用 table/handle |

调用示例：

```json
{"tables":["orders","customers"]}
```

输出：

```json
{
  "tables": [
    {
      "table_name": "orders",
      "row_count": 1200,
      "columns": [
        {"name":"order_id","type":"integer","pk":true},
        {"name":"customer_id","type":"integer","pk":false}
      ],
      "foreign_keys": [
        {"column":"customer_id","references":"customers.customer_id"}
      ]
    }
  ]
}
```

行为细节：

- Source table 返回列名、SQLite 类型、PK 标记、FK 和行数。
- 当前状态校验也允许 derived handle；derived handle 只返回列名和行数，没有 PK/FK。
- 不返回 sample rows 或 distinct values。
- 结果写入对应表的 resident schema。

### 6.2 `inspect_column`

作用：查看一列的值域概况，用于确认精确存储拼写、枚举值和 NULL 情况。

输入：

| 参数 | 必填 | 类型 | 默认 | 约束 |
|---|---|---|---|---|
| `table` | 是 | string | — | 当前可用 table/handle |
| `column` | 是 | string | — | 该表的现有列 |
| `top_k` | 否 | integer | 10 | 至少 1；当前 schema 没有显式最大值 |

调用示例：

```json
{"table":"customers","column":"country","top_k":10}
```

输出：

```json
{
  "column": "country",
  "distinct_count": 32,
  "has_null": false,
  "frequent_values": ["USA","Canada","France"],
  "truncated": false
}
```

行为细节：

- `distinct_count` 使用 `COUNT(DISTINCT column)`。
- `has_null` 表示是否存在 NULL。
- 当 distinct 数不超过 50 时，当前 executor 尝试返回完整值域，不受较小 `top_k` 限制。
- 大值域按频次排序只返回 `top_k`，并设置 `truncated=true`。
- 不返回数值 min/max/mean，也不生成筛选 handle。

### 6.3 `read_subtable`

作用：只读观察表中的有限行，可以选择列、按条件读取、排序和确定性翻页；不会产生新 handle。

输入：

| 参数 | 必填 | 类型 | 默认 | 约束 |
|---|---|---|---|---|
| `table` | 是 | string | — | 当前可用 table/handle |
| `limit` | 否 | integer | 20 | 1–20 |
| `columns` | 否 | `string[]` | 全部列 | 非空且都是现有列 |
| `conditions` | 否 | predicate | 无 | 非空 typed predicate |
| `order_by` | 否 | `string[]` | 无 | 精确列名，可加 `ASC`/`DESC` |
| `offset` | 否 | integer | 0 | 非负；正数 offset 必须同时提供 `order_by` |

调用示例：

```json
{
  "table":"orders",
  "columns":["order_id","amount"],
  "conditions":{"column":"created_at","op":"on_date","value":"2024-01-31"},
  "order_by":["order_id ASC"],
  "limit":20,
  "offset":0
}
```

输出：

```json
{
  "rows": [[101,29.5],[104,18.0]],
  "row_count": 2,
  "limit": 20,
  "offset": 0,
  "order_by": ["order_id ASC"],
  "conditions": {"column":"created_at","op":"on_date","value":"2024-01-31"}
}
```

关键语义：

- 这里的 `row_count` 是本次实际返回的行数，不是完整表行数或全部匹配行数。
- 不提供 `order_by` 时，行顺序不应被视为稳定事实。
- `offset` 只是读取分页，不改变 underlying relation。
- 读取结果进入 resident state，但不会产生可以被后续关系算子引用的新表。
- 如果需要“筛选后的表”，必须使用 `condition_filter`。

### 6.4 `condition_filter`

作用：根据 typed predicate 从现有表派生匹配行，可同时限制输出列。

输入：

| 参数 | 必填 | 类型 | 默认 | 约束 |
|---|---|---|---|---|
| `table` | 是 | string | — | 当前可用 table/handle |
| `conditions` | 是 | predicate | — | 第 4.4 节定义的非空谓词 |
| `return_columns` | 否 | `string[]` | 保留全部列 | 非空且都是输入表现有列 |

调用示例：

```json
{
  "table":"customers",
  "conditions":{
    "and":[
      {"column":"country","op":"=","value":"France"},
      {"column":"active","op":"=","value":true}
    ]
  },
  "return_columns":["customer_id","name"]
}
```

输出：新的 `filter_nnn` handle，包含 `kind=filter`、有序 columns、完整结果 row_count 和
filter derivation。

关键语义：

- 产生真实 derived relation；零行也会产生合法 handle。
- `return_columns` 使该工具同时执行筛选和列投影；省略时保留全部输入列。
- `value_ref` 会在执行前从 producing step 解析为真实 scalar。
- `in_table` 以一个 resident table 作为 membership 子查询。
- 默认不展示匹配行；使用 `read_subtable` 检查结果。

### 6.5 `project`

作用：派生精确的输出列、列顺序、别名和有限的行级表达式；可去重。

输入：

| 参数 | 必填 | 类型 | 默认 | 约束 |
|---|---|---|---|---|
| `table` | 是 | string | — | 当前可用 table/handle |
| `expressions` | 是 | array | — | 至少一个字符串或 typed date expression |
| `distinct` | 否 | boolean | false | 是否对完整投影行去重 |

字符串表达式示例：

```json
{
  "table":"filter_001",
  "expressions":["first_name","last_name","salary * 12 AS annual_salary"],
  "distinct":false
}
```

Typed date expression：

```json
{
  "op":"date_diff_days",
  "operands":[{"column":"start_date"},{"column":"end_date"}],
  "as":"duration_days"
}
```

```json
{
  "op":"extract_year",
  "operands":[{"column":"created_at"}],
  "as":"created_year"
}
```

Typed expression 规则：

- `date_diff_days` 恰好两个 operands，结果是第二个日期减第一个日期的天数；
- `extract_year` 恰好一个 operand；
- 每个 operand 必须恰好是 `{"column":...}` 或 `{"value":...}`；
- `as` 必须是合法标识符。

输出：新的 `project_nnn` handle；`columns` 顺序严格等于 expressions 的输出顺序。

关键语义：

- `project` 不改变 category rows 的朝向，不能替代宽表聚合。
- 字符串表达式目前仍是受 Harness prepare 校验的 SQL 风格 scalar expression，而不是完全
  封闭的 typed AST。
- 缺失列和无法 prepare 的表达式在创建任何 handle 前返回结构化错误。
- 终止表有 helper columns 或列顺序错误时，应使用本工具生成精确答案形状。

### 6.6 `scalar_compute`

作用：对常量或真实 producing step 中的 scalar 做计算，产生一个有名称的 1×1 table。

输入：

| 参数 | 必填 | 类型 | 默认 | 约束 |
|---|---|---|---|---|
| `operation` | 是 | enum | — | 见下表 |
| `operands` | 是 | array | — | 至少两个；每项键集合必须严格合法 |
| `result_name` | 否 | string | `value` | 合法标识符 |

Operand 只允许三种形状：

```json
{"value":25}
{"value_ref":"step_5"}
{"value_ref":"step_7","column":"female_count"}
```

操作定义：

| Operation | Operand 数 | 精确公式 |
|---|---:|---|
| `add` | ≥2 | 所有 operands 相加 |
| `subtract` | ≥2 | `a - b - c ...` |
| `multiply` | ≥2 | 所有 operands 相乘 |
| `divide` | 2 | `a / b` |
| `percent` | 2 | `a * 100 / b` |
| `percent_change` | 2 | `(a - b) / b * 100` |
| `date_diff_days` | 2 | `second - first` 的天数 |

调用示例：

```json
{
  "operation":"percent",
  "operands":[
    {"value_ref":"step_7","column":"female_count"},
    {"value_ref":"step_7","column":"total_count"}
  ],
  "result_name":"female_percentage"
}
```

输出：

```json
{
  "table":"scalar_001",
  "kind":"scalar",
  "columns":["female_percentage"],
  "row_count":1,
  "rows":[[42.5]],
  "derivation": {"schema":"relation-derivation-v1","operator":"scalar_compute"}
}
```

关键语义：

- 数值计算拒绝 bool、NULL、非有限数和除零。
- `date_diff_days` 接受 ISO date/time string。
- 无 `column` 的 `value_ref` 必须指向唯一 scalar；带 `column` 时来源必须恰好一行且列唯一。
- `value` 是模型直接提供的常量，Harness 不会自动证明它来自数据库；这和 `value_ref` 的
  grounded 性不同。

### 6.7 `join_tables`

作用：从一个 base 开始，按顺序把一个或多个新关系接入同一个连通组件，一次生成一个 join
handle。

输入：

| 参数 | 必填 | 类型 | 默认 | 约束 |
|---|---|---|---|---|
| `base` | 是 | string | — | 当前可用 table/handle |
| `joins` | 是 | object[] | — | 非空、有序 |
| `base_role` | 否 | identifier | base 名 | 主要用于 repeated relation |

每个 `joins[]`：

| 字段 | 必填 | 类型 | 默认 | 约束 |
|---|---|---|---|---|
| `table` | 是 | string | — | 本次新接入的 table/handle |
| `on` | 是 | edge[] | — | 非 cross 时至少一条；cross 时必须 `[]` |
| `type` | 否 | enum | `inner` | `inner`, `left`, `cross` |
| `role` | 否 | identifier | table 名 | repeated relation 必须提供唯一 role |

每条 edge 必须恰好是：

```json
{"left":"already_introduced_relation.column","right":"bare_column_of_new_table"}
```

普通三表连接：

```json
{
  "base":"orders",
  "joins":[
    {
      "table":"customers",
      "on":[{"left":"orders.customer_id","right":"customer_id"}]
    },
    {
      "table":"regions",
      "on":[{"left":"customers.region_id","right":"region_id"}]
    }
  ]
}
```

Self join：

```json
{
  "base":"employees",
  "base_role":"employee",
  "joins":[{
    "table":"employees",
    "role":"manager",
    "on":[{"left":"employee.manager_id","right":"employee_id"}]
  }]
}
```

输出：新的 `join_nnn` handle。输出列是扁平 namespace：

```json
["orders.order_id","orders.customer_id","customers.name","customers.region_id"]
```

关键语义：

- `left` 必须来自 base 或前面 joins 已经引入的逻辑列。
- `right` 只能是当前新 table 的 bare column，不能写 `customers.customer_id`。
- 普通 relation 使用 table/handle 名作为 namespace；重复 relation 用 role 区分。
- Join 不自动投影最终字段；需要另调 `project`。
- Catalog FK 只是候选路径，不保证唯一性；join 可能改变 row grain 和 row count。
- `left` join 保留当前 accumulated left 的未匹配行，`inner` 只保留匹配行。

### 6.8 `group_aggregate`

作用：在一张固定输入表上执行分组、全局或条件聚合，也可以把一个 category 轴直接输出为一
行多列。

输入：

| 参数 | 必填 | 类型 | 默认 | 约束 |
|---|---|---|---|---|
| `table` | 是 | string | — | 当前可用 table/handle |
| `group_by` | 是 | `string[]` | — | 可为空；空表示一个 global group |
| `aggregations` | 是 | object[] | — | Native schema 要求至少一个 |
| `passthrough` | 否 | `string[]` | `[]` | 只应使用函数依赖于 group key 的列 |
| `output_layout` | 否 | enum | `rows` | `rows` 或 `columns` |
| `category_values` | 条件必填 | scalar[] | — | 仅 `columns` layout，非空且唯一、有序 |
| `output_columns` | 否 | `string[]` | category 值字符串 | 长度必须等于 category_values |

每个 aggregation：

```json
{
  "op":"count_distinct",
  "column":"patient_id",
  "as":"patient_count",
  "where":{"column":"gender","op":"=","value":"F"}
}
```

| 字段 | 必填 | 允许值/含义 |
|---|---|---|
| `op` | 是 | `sum`, `count`, `count_distinct`, `mean`, `min`, `max` |
| `column` | 是 | 现有列；`count` 可用 `*` |
| `as` | 是 | 非空输出列名 |
| `where` | 否 | 与 filter 相同的 typed predicate |

全局多指标示例：

```json
{
  "table":"patients",
  "group_by":[],
  "aggregations":[
    {
      "op":"count",
      "column":"*",
      "as":"female_count",
      "where":{"column":"gender","op":"=","value":"F"}
    },
    {
      "op":"count",
      "column":"*",
      "as":"total_count"
    }
  ]
}
```

`output_layout="rows"`：

- 输出列依次为 `group_by + passthrough + aggregation aliases`；
- `group_by=[]` 通常产生一行；
- 每个 aggregation 的 `where` 在同一输入 population 上做条件聚合。

`output_layout="columns"` 额外要求：

- 恰好一个 `group_by`；
- 恰好一个 aggregation；
- 不允许 passthrough；
- category_values 决定输出列和值的顺序；
- 可用等长 output_columns 重命名。

宽输出示例：

```json
{
  "table":"patients",
  "group_by":["gender"],
  "aggregations":[
    {"op":"count_distinct","column":"patient_id","as":"patient_count"}
  ],
  "output_layout":"columns",
  "category_values":["M","F"],
  "output_columns":["male_count","female_count"]
}
```

关键语义：

- `count(*)` 是行数；`count(column)` 是非 NULL 数；`count_distinct(column)` 是不同非 NULL
  值数；三者不能互换。
- 条件 metrics 共享同一输入表和 row grain，适合计算同分母的 numerator/denominator。
- `passthrough` 依赖 SQLite bare-column 扩展；如果它不由 group key 唯一决定，值可能没有
  稳定语义。
- 宽布局是 aggregate 内部的 category reshape；独立 `pivot` 不再公开。
- 输出是新的 `group_nnn` handle。

### 6.9 `extreme_value_select`

作用：按一个或多个键排序，选择 top-k，并可限制输出列。它产生新 relation，不是只读观察。

输入：

| 参数 | 必填 | 类型 | 默认 | 约束 |
|---|---|---|---|---|
| `table` | 是 | string | — | 当前可用 table/handle |
| `order_by` | 是 | `string[]` | — | 非空；列名可加 `ASC` 或 `DESC` |
| `top_k` | 否 | integer | 无 | 至少 1；省略时保留全部排序后行 |
| `return_columns` | 否 | `string[]` | 全部列 | 非空且为现有列 |

调用示例：

```json
{
  "table":"employees",
  "order_by":["salary DESC","employee_id ASC"],
  "top_k":3,
  "return_columns":["employee_id","name","salary"]
}
```

输出：新的 `top_nnn` handle。

关键语义：

- 多列排序按列表顺序决定 tie-breaking。
- `top_k` 省略时只是产生完整排序 relation。
- `return_columns` 让 ranking 和 projection 在同一工具中完成。
- 排名应在完整 eligibility relation 构造完成后执行，否则可能对错误 population 排名。

### 6.10 `set_op`

作用：对两张列兼容 relation 执行集合操作。

输入：

| 参数 | 必填 | 类型 | 约束 |
|---|---|---|---|
| `left` | 是 | string | 当前可用 table/handle |
| `right` | 是 | string | 当前可用 table/handle |
| `op` | 是 | enum | `union`, `union_all`, `intersect`, `except` |

调用示例：

```json
{"left":"project_001","right":"project_002","op":"intersect"}
```

操作语义：

| `op` | SQL 语义 |
|---|---|
| `union` | 合并并去重 |
| `union_all` | 合并并保留重复 |
| `intersect` | 左右共有行并去重 |
| `except` | 左侧有、右侧没有的行并去重 |

关键语义：

- 集合比较按列位置进行。
- 最稳妥的做法是先用 `project` 把两侧变成相同列数、含义和列顺序。
- 当前 executor 在某些列数不同时会尝试按左侧列名从右侧投影；不能依赖这种兼容行为替代
  明确对齐。
- 输出为新的 `setop_nnn` handle。

### 6.11 `answer_from_context`

作用：引用一张已经存在的 grounded result table，提交最终答案并终止 episode。

输入：

| 参数 | 必填 | 类型 | 约束 |
|---|---|---|---|
| `evidence` | 是 | object | 必须恰好是 `{"table":"derived_handle"}` |
| `reason` | 否 | string | 仅用于审计，不参与答案数据和评分 |

调用示例：

```json
{
  "evidence":{"table":"project_004"},
  "reason":"The cited table contains exactly the requested rows and columns."
}
```

终止规则：

1. 必须是 assistant turn 中唯一的 function call。
2. 模型不能在调用参数里填写 answer rows 或 scalar 值。
3. 被引用表的全部行、全部列和列顺序就是提交答案。
4. Scalar 答案也必须引用 `group_aggregate` 或 `scalar_compute` 产生的 1×1 table。
5. 如果引用表有 helper column、错误 representation 或多余行，必须先用关系工具修正。
6. 最稳妥的是引用 derived handle；即使答案等于完整 source table，也先 `project` 成明确结果。

内部 evaluator 会计算 `correct/pred_sample/gold_sample` 写入私有审计记录，但 provider-facing
tool result只返回类似：

```json
{"step_id":"step_12","terminal_ready":true}
```

Episode 随即结束。Gold rows 和评分结果不会进入后续模型轮次。

## 7. 错误反馈机制

### 7.1 Provider/carrier 错误

在任何工具执行前检查：

- 没有 `tool_calls`；
- 调用数不在 1–8；
- function name 不存在；
- arguments 不是合法 JSON object；
- 缺少或重复 call id；
- assistant 普通 content 违反当前 carrier 规则。

这类错误保持环境状态不变，作为 `protocol_error` 返回。

### 7.2 静态参数错误

检查：

- 缺少必填字段；
- 出现未声明顶层字段；
- enum、类型、列表长度或 nested object 不合法；
- join edge、aggregate、scalar operand、typed date expression 等形状不合法。

错误通常为 `argument_validation_error`，并附带该工具 expected required/optional fields。

### 7.3 状态感知错误

执行前根据当前 resident state 检查：

- unknown table/handle；
- unknown/ambiguous column；
- 条件操作符或 operand 不合法；
- join left 尚未引入、right 不是新表列；
- project expression 无法 prepare；
- set、aggregate、ranking 引用了错误 relation；
- terminal evidence handle 不存在。

典型 `code` 包括：

```text
unknown_tool
invalid_table_reference
unknown_table
unknown_column
invalid_condition
invalid_condition_operator
ambiguous_condition_operand
invalid_project_expression
no_op_aggregation
```

错误 `details` 尽量包含精确 `argument_path`、requested table/column、当前可用 handles 或
columns。

### 7.4 执行错误

通过前置校验后仍可能发生：

- SQLite expression 或类型错误；
- scalar source 不是唯一 cell；
- NULL、非数值、除零或非法日期；
- set 两侧不能对齐；
- join namespace 冲突；
- 其他 Harness invariant 失败。

普通执行错误也应保持状态不变。若 state hash 发生变化，则升级为
`nonrecoverable_execution_error`。

## 8. 当前设计的关键不变量

1. **Harness 是事实权威**：模型 reasoning、reason、参数说明都不是数据库事实。
2. **Gold 对模型不可见**：Gold SQL/rows 只供终止 verifier 使用。
3. **显式感知**：Handle 默认只暴露 schema/row_count；行必须 `read_subtable`。
4. **派生表可审计**：每个 table-producing 工具都生成 `relation-derivation-v1`。
5. **Scalar grounding**：`value_ref` 只能读取既有 producing step 的真实 cell。
6. **精确 terminal table**：最终表本身必须满足输出 rows、columns、order 和 representation。
7. **错误状态保持**：被拒调用不产生证据或新 handle。
8. **Bundle pre-state**：同一模型轮次中的调用不能形成隐藏的顺序依赖。
9. **Scheme 隔离**：v54 不能混用 direct SQL、iterative SQL、action-block 等顶层协议。

## 9. 基于当前模式优化时需要重点审查的接口缝隙

以下是现状，不代表应一次性全部修改。

### 9.1 Provider schema 比运行时校验更宽松

- Predicate schema 在 API 层只约束为 object/array，操作符和字段组合由 Harness 二次校验。
- `scalar_compute.operands[]` 的 Provider nested schema 允许额外字段，但运行时只接受三种精确
  key set。
- Typed project operand 的 Provider schema 同时展示 `column` 和 `value`，运行时要求二选一。

结果是模型可能生成 Provider 接受、Harness 拒绝的调用。优化时可选择增强 JSON Schema，或保留
宽 schema 但将精确错误作为显式训练信号；两种策略必须做配对测试。

### 9.2 `project` 仍有非完全 typed 的 SQL 风格字符串表达式

Typed date expression 是封闭 AST，但普通算术、CAST、字符串计算仍通过字符串表达式传递。
这降低了工具数量，却增加了 parser、列引用、可审计性和奖励归因复杂度。可以考虑扩充有限 typed
expression，而不是直接把 `project` 变成任意 SQL。

### 9.3 观察结果的“证据充分性”表达不足

- `read_subtable.row_count` 是本页返回行数，不是总匹配数；
- 未排序读取不能证明“第一行”；
- Handle 的 row_count 不等于行已被模型观察；
- `inspect_column.truncated=true` 时目标值未出现不能证明不存在。

这些边界适合通过事实字段改进，例如 `returned_count`、`has_more`、`total_match_count_known`，而不
应加入策略性建议。

### 9.4 部分工具不是严格单一原子操作

- `condition_filter.return_columns` = filter + projection；
- `extreme_value_select.return_columns` = ranking + projection；
- `group_aggregate(output_layout="columns")` = aggregation + reshape；
- 多边 `join_tables` = 多个 join edge 的一个连通组件。

这些组合减少步骤和 token，但使 process credit 粒度更粗。优化时应先明确目标是模型易用性、
轨迹长度还是细粒度奖励，不要只按“一个 SQL operator 一个 tool”机械拆分。

### 9.5 Native bundle 能力与学生训练表示必须一致

一次多个调用可能提高探索覆盖率，但它不是 pass@k：所有调用共享一个 pre-state，而且同轮结果不能
互相依赖。SFT 导出若把 bundle 展平成多个 assistant turn，会制造模型从未观察到的中间状态并破坏
因果性。训练格式必须保留 `model_turn_index`、call 顺序和一对一 tool result。

### 9.6 模型直接常量与 grounded scalar 的信任边界不同

`scalar_compute` 接受 `{"value":...}`，所以模型可以直接提供数字常量。它适合题目明确给出的常量，
但不能自动证明数字来自数据库。若后续做 dense process reward，应区分 question literal、database
cell `value_ref` 和模型无来源常量。

### 9.7 当前 terminal 是不可恢复的

`answer_from_context` 一旦调用就结束 episode。非法 evidence 在前置校验阶段可以返回错误，但合法
handle 指向错误 denotation 时不会把 verifier 的 wrong-answer 反馈给模型继续修复。若要研究“最终
提交后修复”，需要另设非泄漏的结构错误门或可恢复 submit 状态，不能暴露 gold rows。

### 9.8 `passthrough` 和 set positional semantics 有隐含风险

- Aggregate passthrough 使用 SQLite bare-column 语义，只有在列由 group key 函数决定时才可靠。
- Set operation 本质按列位置比较；只看列名相似并不足以证明语义对齐。

可以考虑更强的 pre-execution invariant，而不是只依赖 Prompt 提醒。

## 10. 新版本的建议修改边界

如果基于当前模式实现下一版，建议建立独立版本覆盖层，而不是直接改写冻结的 version39 或
version51–54：

1. 在 `src/tool_modules/` 下建立独立模块或 version protocol；
2. 在 `public_tool_contract.py` 定义公开语义和参数；
3. 在 `provider_tools.py` 提供完全对应的 native JSON Schema；
4. 在 `protocol.py` 增加静态 nested validation；
5. 在状态感知校验层检查 handles、columns 和 references；
6. 在 `Harness` 实现确定性执行；
7. 为 table-producing 工具增加 derivation builder；
8. 在 `EnvironmentState` 定义模型下一轮能够看到的事实；
9. 增加成功、错误、bundle pre-state、replay、no-leak 和 terminal tests；
10. 记录 prompt hash、tool schema hash 和明确的 diagnostic/training admission。

任何优化实验都应冻结任务集合并同时统计：denotation accuracy、legal termination、错误类型、
primitive calls、真实 model turns、token、bundle 宽度、同轮独立性违规和 fresh replay 结果。
