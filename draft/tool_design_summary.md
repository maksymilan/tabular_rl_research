# 表格推理的工具设计与数据构造 — v2 总结

日期 2026-06-08。本文说明三件事：当前工具集与 SQL 原子操作的对应、基于原子操作的奖励设计、从 gold SQL 编译训练数据的方案。

## 一、概述

- 目标：训练 LLM 通过调用一组抽象工具，在结构化表格上完成多步推理。
- 工具是关系 / 感知 / 记忆三类操作。harness 把每个工具调用翻译成 SQLite 上的组合 SQL 执行。模型不接触 SQL。
- 两个设计目标：(1) 把 Spider/BIRD 的 gold SQL 编译成工具调用轨迹，作为训练数据；(2) 工具的原子化 + 行级引用，支持稠密的过程奖励。

## 二、工具集与 SQL 原子操作对应

**关系变换（可由 SQL 表达，由编译器直接产生）**

| 工具 | SQL 原子操作 | 关系代数 |
|---|---|---|
| `condition_filter` | WHERE / HAVING | σ 选择 |
| `project` | SELECT 列投影 | π 投影 |
| `join_tables` | JOIN ... ON（列限定在工具内部完成） | ⋈ 连接 |
| `group_aggregate` | GROUP BY + 聚合 / DISTINCT | γ 分组聚合 |
| `aggregate` | 无 GROUP BY 的标量聚合（COUNT/SUM/...） | 标量 γ |
| `extreme_value_select` | ORDER BY ... LIMIT | 排序 + 截断 |
| `set_op` | UNION / INTERSECT / EXCEPT | ∪ ∩ − |
| `derive_column` | SELECT 中的标量表达式（算术 / CASE / 函数） | 扩展投影 |
| `window` | 窗口函数 OVER() | 窗口 |

**感知（无 SQL 对应）**

| 工具 | 作用 |
|---|---|
| `inspect_column` | 读某列的取值分布，用于给过滤条件的取值做 grounding |
| `read_subtable` | 读有界的行×列片段，供决策前查看 |
| `semantic_match` | 嵌入相似度匹配，模糊检索，替代精确 `=` |

**记忆与终结（无 SQL 对应）**

| 工具 | 作用 |
|---|---|
| `add_to_memory` | 记录一个标量结论，引用来源步 |
| `refine_memory` | 修订已有结论 |
| `answer_from_context` | 终结步，引用证据行给出答案 |

## 三、状态与执行模型

- `data_view`：harness 侧表注册表，含源表与所有中间表（filter/group/join/derive/...）。模型按表名引用，不直接拿到全表。
- `static_task_memory`：模型写入的标量结论，每条必须引用来源（表或步）。
- `dataset_overview`：初始 schema + 主外键关系，作为初始信息提供。
- 统一行级寻址：每行可由 `(table_name, row_id)` 引用。
- 感知内联：建表步的输出内联表内容（单元格数 ≤ 100 全量返回，超出则截断 + 标记），模型逐步看到中间数据。

## 四、奖励设计（基于原子操作）

**结果奖励**：终答案 == gold SQL 执行结果。所有训练轨迹均 execution-verified。

**过程奖励（逐步，每个分量都来自原子操作的结构）**

1. 合法性：参数良构、引用的表/行存在、类型匹配。
2. 步级执行一致：轨迹编译自 gold SQL，因此每一步都有对应的 gold 中间表；该步输出与 gold 流水线对应阶段逐步比对。
3. 贡献度（provenance）：`answer_from_context` 引用一组证据行；沿 `(table, row_id)` 反向切片，得到真正流入答案的步骤集合；切片内的步给正分，切片外不给。
4. 引用完整性：`answer_from_context` 与 `add_to_memory` 引用的 `(table, row_id)` 必须存在于已产生的表中。

**稠密度的形式化**

- 整条 SQL 只产生一个终结果，奖励信号只在终点，稀疏。
- 分解为 N 个带类型 I/O 和行级引用的原子步后，每步都有上述 4 个可计算信号，稠密。
- 度量：定义 fidelity = per-step reward 与该步反事实贡献（消融该步后答案是否改变）的相关性。provenance 反向切片是反事实贡献的可计算代理，且方向健全：**步在切片外 ⟹ 其输出不流入答案 ⟹ 消融后答案不变 ⟹ 贡献为 0**。切片内为贡献的候选上界。

## 五、数据构造：从 gold SQL 编译

```
gold SQL
  → compiler（sqlglot 解析，按逻辑流水线 FROM/JOIN → WHERE → GROUP → HAVING → ORDER/LIMIT → SELECT → DISTINCT 走，每阶段产一个工具步）
  → 工具调用 Plan
  → harness 执行（每个工具翻译成 SQLite 上的组合 SQL）
  → 与 gold SQL 直接执行的结果比对
  → 仅保留 execution-verified 且结构合法 的轨迹
```

- 轨迹格式：`dataset_overview` 初始状态 + 每步 `think / tool_call / tool_output`（输出内联表内容）+ 终结 `answer_from_context`。
- `think` 当前为模板。计划用以可见状态为条件的 LLM 后处理填写，仅作 SFT 种子，不参与奖励。
- 现状（Spider 全量）：train 6256 / 7000（89.4%），dev 914 / 1034（88.4%）；编译覆盖 91.2%，真库执行验证 91.6%；85 个单测通过。轨迹长度 2–14 步。

## 六、工具覆盖与补全

纯 SQL 编译只覆盖关系变换子集；实测 15 个工具中 8 个被使用。其余按机制补全，每种都保持 execution-verified：

| 工具 | 来源机制 | 需要外部模型 |
|---|---|---|
| `derive_column` | 扩编译器：把计算列从 project 拆成独立步 | 否 |
| `window` | 扩编译器：加 OVER() 分支（executor 已支持） | 否 |
| `add_to_memory` | 标量子查询编译：算标量 → 记入 memory → 当阈值复用 | 否 |
| `inspect_column` | 注入 grounding 步：在按字面量过滤前先读该列取值 | 否 |
| `semantic_match` | 字面量扰动增强：把问题中的精确值改为模糊指代，拒绝采样验证解析回同一行 | 嵌入模型 + LLM |
| `refine_memory` | 多轮数据（CoSQL/SParC）中后一轮对前一轮结论的修订；或 RL | 新数据集 |
| `read_subtable` | 已被感知内联取代，候选裁撤 | — |

关系工具来自 gold SQL；感知 / 记忆 / 模糊工具由上述增强机制单独构造。这一区分也是对"工具只是 SQL 子集"质疑的回应：发布数据显式包含 SQL 无法表达的感知、记忆、模糊操作。

## 七、现状与下一步

已完成：v1 SQL 核心工具链（编译器 + harness + 验证器 + 轨迹发射器），Spider 全量轨迹生成，工具合并/内化重构（join 列限定内化、order_limit 并入 extreme_value_select），感知内联。

下一步（按收益排序）：

1. 子查询编译：一次性解锁 `add_to_memory`、多跳深度、覆盖率（过 96%）。
2. 字面量扰动管线：产生 `semantic_match` 数据。
3. 感知注入：产生 `inspect_column` 数据。
4. 扩编译器：`derive_column`（un-fold）、`window`。
5. 接入 BIRD：更难的嵌套、算术、脏数据。
6. `refine_memory`：接入多轮数据集或留给 RL。
