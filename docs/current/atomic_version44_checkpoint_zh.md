# Atomic version44 工具 checkpoint

日期：2026-07-30

## 定位

`version44` 是从 atomic version39 单独派生的工具接口 checkpoint。它保留 version39 的
单动作 carrier、recent-4 legal history、关系执行语义、resident state、grounding、
relation derivation 和 `answer_from_context(evidence={"table":...})`，只合入以下四项：

1. 删除模型可见 `plan`；
2. 将模型可见 `read_subtable` 重命名为 `inspect_rows`；
3. 新增只读 `search_values`；
4. BIRD `semantic_name`/`column_description` 只随 `inspect_column` 返回，
   `describe_table` 保持 raw schema。

version44 不继承 version40 的精简 prompt、全 reasoning history，也不继承 version42/43
的显式 terminal columns。checkpoint-560 的 version26 评测链保持冻结。

## 公开工具

version44 有 12 个公开工具：

- `describe_table`
- `inspect_column`
- `search_values`
- `inspect_rows`
- `condition_filter`
- `project`
- `scalar_compute`
- `join_tables`
- `group_aggregate`
- `extreme_value_select`
- `set_op`
- `answer_from_context`

私有 executor 继续保留 `read_subtable` 作为历史 replay/implementation alias，但 version44
strict parser 拒绝模型调用它，也拒绝 `plan`。

## search_values

公开接口：

```json
{
  "tool": "search_values",
  "arguments": {
    "table": "SeasonStatus",
    "query": "Avangrd Omsk",
    "column": "TEAM",
    "limit": 20,
    "offset": 0
  }
}
```

边界：

- `table` 和 `query` 必填；
- `column` 选填，省略表示该 table 的全部逻辑列；
- `limit` 默认 20，合法范围 `1..20`；
- `offset` 默认 0，必须是非负整数；
- 只搜索一个明确 table，不执行全库搜索；
- exact、normalized exact、prefix、token、substring、fuzzy 按稳定顺序排序；
- fuzzy 是确定性 lexical matching，不使用 embedding、模型、question、gold 或 verifier；
- 返回 exact stored value、table、column、frequency、match type 和确定性 score；
- `has_more/next_offset` 支持稳定分页；
- perception-only，不过滤行、不创建 handle、不能作为 terminal evidence。

搜索 observation 进入 resident state，最多保留每个 table 最近两次搜索。后续
`condition_filter`/conditional aggregate 使用返回的 exact value 时，Harness 可生成
`domain_observation` grounding edge。

## inspect-only 列语义

只有成功的 `inspect_column(table,column)` 可附加：

- BIRD CSV `column_name` → `semantic_name`；
- BIRD CSV `column_description` → `column_description`。

`describe_table` 不返回上述字段。raw table/column name 是唯一可执行标识。overlay 不返回
`data_format`、`value_description` 或 metadata sample values；每次 enrichment 记录来源文件
SHA-256、canonical output hash 和 model-visible output hash。

## 本地验证

- version40–44、protocol、rolling history、atomic context、environment state 和
  process credit 组合测试：168 passed，5 subtests passed；
- version44/inspect-only/process-credit 定向复测：60 passed；
- harness executor 原有 48 项：48 passed，0 failed；
- `git diff --check`：通过。

覆盖 exact/fuzzy 排序、table/column 约束、20 项分页、retired 名称拒绝、
`inspect_rows` alias、search resident state、search→filter grounding、inspect-only
metadata overlay、DeepSeek JSON Output prompt 和 version44 hash 注册。

## 外部教师接口 smoke test

自然 BIRD 任务 `bird_train_04820` 使用 DeepSeek v4 Flash，5 个动作、0 process errors、
合法终止并且 `bird-set` correct。它验证了 version44 prompt/parser/executor 的自然调用，
但没有主动使用新增 perception。

第二条 diagnostic-only 任务在同一 BIRD 数据库上将 team 写成 typo
`Avangrd Omsk`，明确要求覆盖新接口：

- 8 个动作；
- 0 process errors；
- legal termination；
- `bird-set` correct；
- 实际工具链：
  `describe_table → inspect_column → search_values → condition_filter →
  join_tables → inspect_rows → project → answer_from_context`。

关键返回：

- `inspect_column(SeasonStatus.TEAM)` 返回
  `column_description="which team the player belong to"`；
- `search_values` 返回 `Avangard Omsk`（frequency 15，fuzzy score 0.96）和
  `Avangard Omsk-2`（frequency 6，fuzzy score 0.888889）；
- 教师选择第一项 exact stored value；
- `inspect_rows` 成功读取最终 join 结果；
- 最终 distinct projection 与 gold denotation 一致。

两个运行都使用 strict no-repair parser、真实 SQLite、hidden gold verifier、DeepSeek
thinking + JSON Output，没有 transport、completion、context 或 carrier retry。

## 当前状态

version44 的接口、执行、prompt、外部教师闭环和 provenance 已打通，但冻结 paired
Gate50 没有通过准确率 promotion：

- version39 baseline：39/50（78%）；
- version44：38/50（76%）；
- 4 gains、5 regressions，50/50 legal termination；
- 使用 `search_values` 的 19 题，两组均为 16/19；
- version44 墙钟时间增加 62.8%，百万 distinct value 列的全量 fuzzy 枚举是主要问题。

因此 version44 当前仍是 diagnostic-only，不应直接作为新的 SFT/RL 协议，也不运行
剩余 150 题。先优化有界候选检索和候选值使用策略，再重跑同一 Gate50。完整报告见
`docs/reports/evaluation/BIRD_ATOMIC_VERSION44_SEARCH_VALUES_PAIRED_GATE50_20260730_ZH.md`。
