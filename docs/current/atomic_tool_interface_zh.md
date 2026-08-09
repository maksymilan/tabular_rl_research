# Atomic version39 工具接口文档

本文档描述仓库当前 `atomic` 工具方案的模型可见接口，并说明最近 fixed-200 实验使用的
F profile。可执行源码优先级高于本文档：

- 公共工具结构：`src/sft/public_tool_contract.py`
- 协议、严格解析与参数校验：`src/sft/protocol.py`
- 执行与状态校验：`src/eval/rollout.py`
- Harness 执行器：`src/harness/executor.py`
- F profile：`src/eval/atomic_database_context.py`

## 1. 协议身份

| 项目 | 当前值 |
|---|---|
| Tool scheme | `atomic` |
| Protocol | `version39` |
| Scheme registry | `tool-scheme-registry-v11` |
| Canonical action carrier | `think-json-v1` |
| 默认历史策略 | 最近 4 个合法 assistant/observation pair |
| BIRD denotation metric | `bird-set` |
| 当前 admission | diagnostic-only，尚未获得准确率推广 |

Atomic、action-block 和 relational-program 是互斥的三套模型接口。本文只描述 atomic；
模型不能在一个 prompt 中混用另外两套顶层动作。

## 2. 一轮只能提交一个动作

Canonical 形式严格为：

```text
<think>本轮动作的非空理由</think>
{"tool":"tool_name","arguments":{...}}
```

要求：

1. 恰好一个非空 `<think>`。
2. 后面直接跟一个完整 JSON object。
3. JSON 顶层键必须恰好是 `tool` 和 `arguments`。
4. `arguments` 必须是 object。
5. 不允许 `<tool_call>` 标签、Markdown、额外文字、第二个动作或参数自动修复。

DeepSeek API 可以把理由放在 provider-native `reasoning_content`，visible content 只返回
严格 JSON。Adapter 只负责把两个原始字段封装为 canonical carrier，不修改工具名、
参数、列名或 JSON。

## 3. 上下文与环境状态

每个 episode 的模型输入由以下部分组成：

```text
system: 工具协议
user: DATASET OVERVIEW + QUESTION + 可选 EXTERNAL KNOWLEDGE
[最多 4 对合法 assistant action + harness observation]
user: 最新 observation + CURRENT ENVIRONMENT STATE + 可选 LAST TOOL ERROR
```

Harness 维护：

- source table 与 derived table；
- derived handle、列、行数和创建 step；
- 观察过的 schema、column domain 和 rows；
- scalar-producing step；
- plan 控制状态；
- 每张派生表的 fact-only `relation-derivation-v1`；
- 错误事件、provenance 与 terminal evidence。

模型只负责 `think + tool + arguments`。模型不创建 step id、table handle、执行 SQL、
provenance、环境状态或结果行。

被拒绝的 assistant 文本不会进入合法历史。下一轮只看到结构化 `LAST TOOL ERROR` 和
未改变的 resident state。

## 4. 数据库上下文 profiles

### 4.1 默认 atomic：`catalog-v1`

开局只提供：

- table name；
- row count；
- foreign-key relation catalog；
- question 与 external knowledge。

模型可使用全部 12 个工具，包括：

- `describe_table`
- `inspect_column`

### 4.2 F：`full-bird-schema-samples-v1`

这是最近 fixed-200 使用的诊断 profile。开局额外提供：

- 所有 source tables；
- 所有列、类型和 row count；
- primary key / foreign key；
- BIRD column name、短语义说明和 data format；
- 每列最多两个 distinct、非空真实示例值；
- 示例值最长 40 个字符，并明确不是完整值域。

F 从 prompt 和运行时合法工具集同时移除：

- `describe_table`
- `inspect_column`

因此 F 只有 10 个模型可调用工具。调用被移除的工具会返回 `unknown_tool`，环境不会
自动改写。

### 4.3 Full context with inspect

`full-bird-schema-samples-with-inspect-v1` 只移除 `describe_table`，保留
`inspect_column`。它是独立诊断 profile，不是 F。

## 5. 工具总览

| 工具 | 必填参数 | 可选参数 | 是否产生 table handle | F 可用 |
|---|---|---|---|---|
| `plan` | `ops` | — | 否 | 是 |
| `describe_table` | `tables` | — | 否 | **否** |
| `inspect_column` | `table`, `column` | `top_k` | 否 | **否** |
| `read_subtable` | `table` | `limit`, `columns`, `conditions`, `order_by`, `offset` | 否 | 是 |
| `condition_filter` | `table`, `conditions` | `return_columns` | 是 | 是 |
| `project` | `table`, `expressions` | `distinct` | 是 | 是 |
| `scalar_compute` | `operation`, `operands` | `result_name` | 是，1×1 | 是 |
| `join_tables` | `base`, `joins` | `base_role` | 是 | 是 |
| `group_aggregate` | `table`, `group_by`, `aggregations` | `passthrough`, `output_layout`, `category_values`, `output_columns` | 是 | 是 |
| `extreme_value_select` | `table`, `order_by` | `top_k`, `return_columns` | 是 | 是 |
| `set_op` | `left`, `right`, `op` | — | 是 | 是 |
| `answer_from_context` | `evidence` | `reason` | 终止 | 是 |

`aggregate`、`pivot`、旧 binary join 参数和 parser shorthand 只用于历史 replay；新动作
调用会被拒绝。

## 6. 通用返回语义

关系工具通常返回：

```json
{
  "table": "filter_001",
  "kind": "filter",
  "columns": ["id", "name"],
  "row_count": 8,
  "derivation": {
    "schema": "relation-derivation-v1",
    "operator": "condition_filter"
  }
}
```

其中：

- `table` 是 harness 分配的真实 resident handle；
- 后续调用必须复制该 handle，不能自行预测；
- `columns` 和 `row_count` 是事实；
- 普通 table-producing 工具不会自动展示全部 rows；
- `read_subtable` 用于查看 rows，但不会生成新 handle；
- `derivation` 只描述已执行的关系语义，不提供下一步建议。

## 7. Predicate grammar

`condition_filter`、`read_subtable.conditions` 和 aggregation-level `where` 共用 typed
predicate tree。

### 比较字面量

```json
{"column":"age","op":">=","value":18}
```

`op`：

```text
=  !=  >  >=  <  <=
```

### 列与列比较

```json
{"column":"actual_quantity","op":"<","column_value":"ordered_quantity"}
```

### 与 scalar step 比较

```json
{"column":"price","op":">","value_ref":"step_5"}
```

`step_5` 必须产生一个可 grounding 的 scalar。

### 集合

```json
{"column":"city","op":"in","values":["Paris","London"]}
```

```json
{"column":"customer_id","op":"in","in_table":"project_003"}
```

### 区间、字符串、日期与 NULL

```json
{"column":"year","op":"between","low":2010,"high":2020}
```

```json
{"column":"name","op":"contains","value":"Smith"}
```

```json
{"column":"created_at","op":"on_date","value":"2024-01-31"}
```

```json
{"column":"deleted_at","op":"is_null"}
```

### 组合

```json
{
  "and": [
    {"column":"city","op":"=","value":"Paris"},
    {
      "or": [
        {"column":"age","op":">=","value":18},
        {"column":"is_admin","op":"=","value":true}
      ]
    },
    {"not":{"column":"status","op":"=","value":"blocked"}}
  ]
}
```

## 8. 工具详情

### 8.1 `plan`

更新 harness-owned 控制状态，不产生事实或答案。

```json
{
  "tool": "plan",
  "arguments": {
    "ops": [
      {
        "op": "create",
        "id": "find_population",
        "goal": "确定目标 population",
        "status": "in_progress"
      },
      {
        "op": "update",
        "id": "find_population",
        "status": "done",
        "evidence": "step_4"
      }
    ]
  }
}
```

规则：

- `op`: `create | add | update | delete`
- status: `pending | in_progress | done | blocked`
- `evidence` 只能引用既有 step id；
- evidence 的事实内容由 harness 从真实 step output 复制；
- plan 不能作为 `value_ref` 或 terminal evidence；
- plan goal 不允许夹带模型声称的答案值。

### 8.2 `describe_table`

默认 atomic 中查看一个或多个 source table 的列、类型、PK/FK。

```json
{
  "tool": "describe_table",
  "arguments": {
    "tables": ["orders", "customers"]
  }
}
```

F profile 不可用。

### 8.3 `inspect_column`

查看列的 distinct count、frequent values 和 NULL 状态。

```json
{
  "tool": "inspect_column",
  "arguments": {
    "table": "customers",
    "column": "country",
    "top_k": 10
  }
}
```

它只观察值域，不创建筛选 handle。F profile 不可用。

### 8.4 `read_subtable`

只读观察最多 20 行。

```json
{
  "tool": "read_subtable",
  "arguments": {
    "table": "orders",
    "columns": ["order_id", "amount"],
    "conditions": {
      "column": "created_at",
      "op": "on_date",
      "value": "2024-01-31"
    },
    "order_by": ["order_id ASC"],
    "offset": 0,
    "limit": 20
  }
}
```

严格约束：

- `limit` 必须是整数 `1..20`，不会自动 clamp；
- `columns` 必须是非空列名 list；
- `offset` 必须是非负整数；
- `offset > 0` 时必须同时提供 `order_by`；
- `conditions`、`order_by`、`columns` 只控制本次观察；
- 不会产生 filtered/sorted/projected table handle；
- 完全相同的相邻调用会读到相同 rows，并触发 `no_progress_error`。

### 8.5 `condition_filter`

产生满足 predicate 的 derived table。

```json
{
  "tool": "condition_filter",
  "arguments": {
    "table": "customers",
    "conditions": {
      "and": [
        {"column":"country","op":"=","value":"France"},
        {"column":"active","op":"=","value":true}
      ]
    },
    "return_columns": ["customer_id", "name"]
  }
}
```

`return_columns` 可在同一步保留指定列；省略时保留输入 relation 的所有列。

### 8.6 `project`

选择、排序或计算最终列。

```json
{
  "tool": "project",
  "arguments": {
    "table": "filter_001",
    "expressions": ["first_name", "last_name"],
    "distinct": true
  }
}
```

字符串 expression 可以是：

```json
"orders.price * orders.quantity AS revenue"
```

Version39 还支持两个 typed row-date expressions：

```json
{
  "op": "date_diff_days",
  "operands": [
    {"column":"start_date"},
    {"column":"end_date"}
  ],
  "as": "duration_days"
}
```

```json
{
  "op": "extract_year",
  "operands": [{"column":"created_at"}],
  "as": "year"
}
```

约束：

- expression list 非空；
- typed object 必须恰好有 `op, operands, as`；
- `as` 必须是 identifier；
- `date_diff_days` operand 顺序是 start, end；
- `project` 保持行方向，不能把 category rows 转成 category columns。

### 8.7 `scalar_compute`

对 harness-grounded scalar 做算术，返回 1×1 table。

```json
{
  "tool": "scalar_compute",
  "arguments": {
    "operation": "percent",
    "operands": [
      {"value_ref":"step_7","column":"usa_count"},
      {"value_ref":"step_7","column":"total_count"}
    ],
    "result_name": "percentage"
  }
}
```

operations：

```text
add
subtract
multiply
divide
percent
percent_change
date_diff_days
```

operand 必须恰好是以下一种：

```json
{"value": 100}
```

```json
{"value_ref": "step_4"}
```

```json
{"value_ref": "step_7", "column": "metric"}
```

规则：

- `value_ref` 引用 producing step，不是 `read_subtable` 或 plan step；
- 无 `column` 时 producer 必须是 1×1；
- 有 `column` 时 producer 必须是一行，且 column 精确唯一；
- `divide`、`percent`、`percent_change`、`date_diff_days` 恰好两个 operands；
- `percent(part, whole) = part / whole * 100`；
- `percent_change(new, old) = (new-old)/old*100`，operands 顺序是 new, old；
- 非交换运算的 operand 顺序具有语义。

### 8.8 `join_tables`

一次调用构造一个 connected join component，可含多条顺序 join edge。

```json
{
  "tool": "join_tables",
  "arguments": {
    "base": "orders",
    "joins": [
      {
        "table": "customers",
        "on": [
          {"left":"orders.customer_id","right":"id"}
        ]
      },
      {
        "table": "regions",
        "on": [
          {"left":"customers.region_id","right":"id"}
        ],
        "type": "left"
      }
    ]
  }
}
```

核心规则：

- `base` 必须是一个 source table 或 resident handle 字符串；
- `joins` 必须是非空 list；
- 每项必填 `table, on`，可选 `type, role`；
- `type`: `inner | left | cross`，默认 `inner`；
- 非 cross 的 `on` 至少一个 equality edge；
- cross 必须 `type="cross"` 且 `on=[]`；
- `on.left` 是已经引入的精确 `relation.column`；
- `on.right` 是新表的 bare column，不能带点；
- 输出列保持 flat namespace，如 `orders.id`、`customers.name`；
- 后续 join 不会生成 `join_003.orders.id`；
- 普通 join 不使用 role；
- self-join 时才使用 `base_role` 和 item-level `role`。

Self-join：

```json
{
  "tool": "join_tables",
  "arguments": {
    "base": "employees",
    "base_role": "employee",
    "joins": [
      {
        "table": "employees",
        "role": "manager",
        "on": [
          {"left":"employee.manager_id","right":"id"}
        ]
      }
    ]
  }
}
```

宽 join 在模型状态中可能显示为：

```json
{
  "table": "join_003",
  "column_namespaces": {
    "orders": ["id", "customer_id"],
    "customers": ["id", "name"]
  }
}
```

实际精确列名仍是 `orders.id`、`customers.name`。`base` 使用 `join_003`，但
`on.left` 使用逻辑 namespace，不使用 `join_003.column`。

### 8.9 `group_aggregate`

默认 rows layout：

```json
{
  "tool": "group_aggregate",
  "arguments": {
    "table": "sales",
    "group_by": ["brand"],
    "aggregations": [
      {"op":"sum","column":"amount","as":"total_amount"}
    ]
  }
}
```

aggregation op：

```text
sum
count
count_distinct
mean
min
max
```

规则：

- `group_by=[]` 表示一个 global group；
- 每个 aggregation 必须恰好包含 `op, column, as`，可选 `where`；
- `where` 只作用于所在 aggregation；
- `column="*"` 仅用于 count 类操作；
- conditional `count_distinct` 必须给具体 column；
- `passthrough` 是额外保留列的 list；
- 顶层没有 `conditions`、`where` 或 `result_name`。

共享 population 的多个条件指标：

```json
{
  "tool": "group_aggregate",
  "arguments": {
    "table": "patients",
    "group_by": [],
    "aggregations": [
      {
        "op":"count",
        "column":"*",
        "as":"female_count",
        "where":{"column":"gender","op":"=","value":"F"}
      },
      {
        "op":"count",
        "column":"*",
        "as":"male_count",
        "where":{"column":"gender","op":"=","value":"M"}
      }
    ]
  }
}
```

Columns layout：

```json
{
  "tool": "group_aggregate",
  "arguments": {
    "table": "patients",
    "group_by": ["gender"],
    "aggregations": [
      {"op":"count_distinct","column":"patient_id","as":"patient_count"}
    ],
    "output_layout": "columns",
    "category_values": ["M", "F"],
    "output_columns": ["male", "female"]
  }
}
```

Columns layout 要求：

- 恰好一个 group column；
- 恰好一个 aggregation；
- 不允许 passthrough；
- `category_values` 非空、有序、唯一；
- `output_columns` 如存在，长度必须与 category values 相同且名称唯一。

### 8.10 `extreme_value_select`

排序并可选 top-k。

```json
{
  "tool": "extreme_value_select",
  "arguments": {
    "table": "sales_by_brand",
    "order_by": ["total_amount DESC", "brand ASC"],
    "top_k": 3,
    "return_columns": ["brand", "total_amount"]
  }
}
```

约束：

- `order_by` 是非空字符串 list；
- `top_k` 如存在必须是正整数；
- `return_columns` 如存在必须是非空列 list；
- 没有题目依据时，不应自行用 `top_k=1` 消除并列。

### 8.11 `set_op`

```json
{
  "tool": "set_op",
  "arguments": {
    "left": "project_001",
    "right": "project_002",
    "op": "intersect"
  }
}
```

`op`：

```text
union
union_all
intersect
except
```

左右 relation 必须列兼容。通常先用 `project` 对齐列数、顺序和类型。

### 8.12 `answer_from_context`

唯一 terminal tool。

```json
{
  "tool": "answer_from_context",
  "arguments": {
    "evidence": {
      "table": "project_005"
    },
    "reason": "该表已包含准确的答案行与列。"
  }
}
```

严格规则：

- `evidence` 必须恰好是 `{"table":"result_handle"}`；
- table 必须已经存在于 resident state；
- table rows、columns、column order、duplicates 和 representation 就是最终答案；
- 不会隐藏投影、隐藏排序、隐藏 aggregation 或读取 reason 中的答案；
- scalar 答案也必须引用 1×1 table；
- `read_subtable` 的 observation step 不能作为 terminal table；
- 如仍有 helper column，必须先 `project`。

## 9. 错误与恢复

成功 observation：

```json
{
  "step_id": "step_4",
  "status": "success",
  "output_summary": {
    "table": "filter_002",
    "columns": ["id", "name"],
    "row_count": 3
  },
  "full_output": "resident_in_current_environment_state"
}
```

错误 observation：

```json
{
  "step_id": "step_5",
  "status": "error",
  "error": {
    "type": "argument_validation_error",
    "code": "unknown_column",
    "message": "...",
    "details": {}
  },
  "attempted_action": {
    "tool": "condition_filter",
    "arguments": {}
  }
}
```

主要 failure types：

| 类型 | 含义 | 状态 |
|---|---|---|
| `protocol_error` | carrier、JSON、工具名或顶层结构非法 | 不执行，状态不变 |
| `argument_validation_error` | 参数 shape、table/column/reference 非法 | 不执行，状态不变 |
| `execution_error` | 合法参数在执行/grounding 时失败 | 必须保持状态不变才可恢复 |
| `no_progress_error` | 与相邻动作的 tool+arguments 完全一致 | 不执行，状态不变 |
| `nonrecoverable_execution_error` | 失败执行意外改变状态 | 立即终止 |
| `provider_carrier_error` | provider 未形成可接受 carrier | 不属于语义动作 |
| `context_overflow` / `api_error` | API 基础设施失败 | 不属于关系执行 |
| `wrong_answer` | 合法 terminal，但 `bird-set` 不匹配 | episode 结束 |
| `max_steps` | 达到动作预算仍未 terminal | episode 结束 |

恢复规则：

- error action 消耗共享 action budget；
- 被拒绝文本不进入 legal history；
- `LAST TOOL ERROR` 保留结构化根因；
- 下一动作必须根据错误修改参数或工具；
- 除 `no_progress_error` 外，同类型 recoverable error 达到当前配置上限时终止；
- `no_progress_error` 可继续恢复到 `max_steps`；
- API transport/completion retry 是 client event，不是 semantic action。

## 10. 最小完整示例

目标：找出 Paris 的成年用户姓名。

### 默认 atomic 第一步：查看 schema

```json
{"tool":"describe_table","arguments":{"tables":["people"]}}
```

F 中 schema 已开局可见，省略该步。

### 派生目标 rows

```json
{
  "tool": "condition_filter",
  "arguments": {
    "table": "people",
    "conditions": {
      "and": [
        {"column":"city","op":"=","value":"Paris"},
        {"column":"age","op":">=","value":18}
      ]
    }
  }
}
```

假设 harness 返回 `filter_001`。

### 构造精确输出

```json
{
  "tool": "project",
  "arguments": {
    "table": "filter_001",
    "expressions": ["name"]
  }
}
```

假设 harness 返回 `project_002`。

### 独立终止

```json
{
  "tool": "answer_from_context",
  "arguments": {
    "evidence": {"table":"project_002"}
  }
}
```

## 11. 当前已知接口摩擦

最近 F fixed-200 显示：

- `join_tables` 的 flat namespace 与 bare-right 是最常见参数错误源；
- 模型容易把 `read_subtable` 误当作会产生 filtered handle；
- 完整 schema 降低了探索动作，但没有消除 population/grain 与输出槽位错误；
- F 为诊断 profile，不是已推广的默认工具接口；
- 不应由环境猜测列名、修正 join、把 read 自动转为 filter，或从 reason 推断答案。

接口优化应通过独立版本和 paired gate 验证，不能静默改变当前 atomic 或 F 的执行语义。
