# BIRD schema-context hard60 诊断集冻结说明

日期：2026-07-29

## 结论

新的 schema-context 诊断集已冻结为 60 题：

`data/eval_inputs/bird_train_schema_context_hard60_20260729.jsonl`

它使用原冻结 200 题中尚未用于当前 context 消融的完整 hard stratum（源索引
140--199），与此前使用的前 50 题完全不重叠。历史 atomic version24 lazy-catalog
`bird-set` reference 为：

| 指标 | hard60 | 旧 first50 |
|---|---:|---:|
| 正确率 | **42/60 = 70.0%** | 42/50 = 84.0% |
| 错误目标 | 18 | 8 |
| 正确 control | 42 | 42 |
| legal termination | 58/60 | 50/50 |
| process errors | 12 | 3 |

新集合同时增加了可恢复目标数和过程难度，更适合区分 schema/context 条件；但它是
selection-defined hard diagnostic cohort，不能当作无偏的 BIRD-wide accuracy。

## 选择规则

选择器：

`src/eval/select_schema_context_hard_cohort.py`

选择发生在读取任何模型结果之前，仅使用：

1. `source_index >= 50`，排除此前 context 消融 first50；
2. `metadata.difficulty_proxy == "hard"`；
3. `metadata.difficulty_proxy_features.score >= 6`。

没有使用 gold SQL、正确率、legal、failure type、模型输出或工具轨迹。满足规则的题恰好
为源索引 140--199 的 60 题，因此没有抽样、随机种子或依据历史结果挑题。

选择完成、ID 冻结后，才关联历史 version24 轨迹生成 reference target/control 列表。
reference 不参与选择。

## 构成

- 60 题；
- 34 个数据库；
- complexity score 分布：
  - 6：21
  - 7：15
  - 8：9
  - 9：7
  - 10：3
  - 11：1
  - 12：1
  - 13：2
  - 14：1
- source SHA-256：
  `6b469215ac96e09c1046fee539ff4c4885b58e9b7a0b6fdb2187082267dcde36`
- output SHA-256：
  `e93b913f9628984a05924dc043ad52fcc03d9bc7e870e403635a163f157b6f94`

历史 reference 的 18 个失败包括 16 个 `wrong_answer`、一个 `max_steps` 和一个
`execution_error`。其余 42 题作为回归 control。

## 后续评测规则

所有新条件必须使用同一 hard60 文件、原始顺序和 `bird-set`，并同时报告：

- 总正确数；
- 相对 fresh lazy control 的 paired gains / regressions / net；
- 18 个历史 failure target 的 recovery 数；
- 42 个历史 correct control 的 retention 数；
- legal termination 和 process errors；
- mean steps、API requests 和 total tokens；
- API/transport failure 单独报告。

历史 42/60 只用于预估难度。正式 A/B 最好在同一时间窗口 fresh rerun
`lazy-catalog-v1`，避免把 provider 漂移误当作 context 效果。

建议第一阶段先跑 hard60 的：

1. fresh atomic lazy baseline；
2. lazy + `describe_table` 返回短 `semantic_name`；
3. lazy + `describe_table` 返回长 `column_description`。

不先跑 full-schema/value 条件，除非按需条件出现正的 paired signal。

## Artifacts

- 测试集：
  `data/eval_inputs/bird_train_schema_context_hard60_20260729.jsonl`
- 选择审计、ID、数据库分布和 reference target/control：
  `data/eval_inputs/bird_train_schema_context_hard60_20260729.jsonl.selection.json`
- 历史 reference：
  `data/trajectories/tool_usability_20260724/version24_fixed200_remaining150_bird_set.all.jsonl`

