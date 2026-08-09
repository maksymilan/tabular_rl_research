# BIRD iterative-SQL v6 关系型输出形状规则

日期：2026-08-05  
状态：已实现；后续独立 Gate20 总分持平但未覆盖目标题，diagnostic-only

## 结论

`iterative-sql-v6` 是 v5 之后的最小 prompt-only 版本。公开工具仍只有
`execute_sql(sql)` 与 `submit_sql(sql)`；SQL 安全、执行、preview、错误反馈、上下文管理、
prior-execution grounding、隐藏判分和预算均不变。

v6 只把最终答案的关系型表示写成显式规则：

1. 每个答案都是 SQL 结果表；标量是 1×1 表。
2. 答案实体按题目要求的粒度分别占行；一个映射字段对应一个输出列。
3. 多个映射字段按 external knowledge 声明的顺序分别输出为多列。
4. 问题中的单数名称（例如 `full name`）不授权把多个映射字段拼成一个字符串。
5. 只有 QUESTION 或 EXTERNAL KNOWLEDGE 明确要求一个 formatted、combined 或 string value
   时，才允许拼接。
6. 多个命名标量默认是一行中的多个列；只有任务明确要求 label/value rows 时才转成长表。

## 修改动机与边界

冻结 Prefix50 中，v5 相对 v4 的两条回归都属于同一种表示错误：external knowledge 已把
`full name` 映射到两个或三个字段，v5 却把它们拼成一个字符串；v4 保留了分列输出。旧 v5
规则已经说“多个映射字段保持多列”，但模型仍把单数自然语言名称误当成拼接许可，因此 v6
增加了标量/表、映射字段/列和拼接许可的明确对应关系。

这条规则是普适的输出表示约束，不包含题目 id、表名、数据库列名、gold SQL、gold rows 或
隐藏验证器信息。它适用于任何“一个语义槽位由多个数据库字段表示”的任务，而非为两道题
加入特例。

## 版本与可复现性

- active protocol：`iterative-sql-v6`
- active interface：`execute-sql-submit-sql-v6`
- registry：`tool-scheme-registry-v9`
- DeepSeek Flash + lazy catalog protocol hash：`176ce977b411636e`
- 冻结 v5 interface：`execute-sql-submit-sql-v5`
- 冻结 v5 hash：`b6465c6e222c12c2`
- v4 hash：`00b4e630adf2faaf`
- v3 hash：`64d865970246b3c9`

实现保留 v5 的独立 prompt 模板、interface 与审计映射，历史 v5 artifact 可按原协议重放。

## 评测约束

后续冻结第 51–70 题的独立 Gate20 已完成：v6 与 v5 均为 12/20、20/20 legal，配对一增一退，
v6 tokens 增加 2.8%。该 slice 没有 multi-field name 目标题或 explicit concatenation 控制题，
所以目标规则仍未被直接验证。Prefix50 已用于发现该失败模式，也不能用其中两道回归题的恢复
宣称泛化改进。因此随后冻结一个与前 70 题不重叠的新定向 slice，设计上包含：

- 多字段映射但不要求拼接的目标题；
- 明确要求格式化/组合字符串的反向控制题；
- 1×1 标量、单字段多行和多命名标量控制题；
- 与输出表示无关的普通 SQL 控制题。

该定向 Target Gate20 后续已完成：v6 15/20 versus v5 10/20，5 gains / 0 regressions；
multi-field targets 为 5/6 versus 1/6，所有 single-field/ordinary controls retained，且 v6
errors/actions/tokens 全部改善。所有数值门通过。原标为 explicit-combined 的 `field1+field2`
类别被 reference 证明仍要求分列，因此反向能力没有得到验证。最终结论只授权代表性 paired
gate，不开放 SFT/RL。详见
`BIRD_ITERATIVE_SQL_V6_TARGET_GATE20_RESULT_20260805_ZH.md`。

尽管目标门的数值、replay、结构、可靠性和成本阈值均通过，v6 在新的代表性 paired gate
完成前仍不进入 SFT/RL，也不能描述为 v5 或 v4 的普遍准确率提升版本。
