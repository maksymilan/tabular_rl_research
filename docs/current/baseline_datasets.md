# Baseline 数据集管理与变更记录

本文档是仓库中 baseline cohort 的唯一当前台账。以后任何 baseline 数据新增、替换、扩容、
重排、排除规则或选择算法变更，都必须在同一个变更中更新本文档、版本化 manifest 和
`AGENTS.md`；不得只替换 JSONL。

## 当前状态

| 状态 | cohort | 用途 | 记录数 | 数据库数 | SHA-256 |
|---|---|---|---:|---:|---|
| **active** | `data/eval_inputs/bird_train_baseline300_v1.jsonl` | 所有新的 BIRD-train baseline 实验 | 300 | 69 | `87b37b307ea2710acdff7d03bdc474a6ba2cdb8ed51b005cff0fe8fa54c67c03` |
| **deprecated** | `data/eval_inputs/bird_train_tool_interface_validation200_version4.jsonl` | 只复现历史 fixed-200 报告 | 200 | 57 | `6b469215ac96e09c1046fee539ff4c4885b58e9b7a0b6fdb2187082267dcde36` |

旧 200 题的 JSONL 内容没有删除、重排或改写，旧实验和报告仍可按原 task ids 复现；其
manifest 已明确写入 `deprecated_for_new_experiments`。从 2026-08-05 起，任何新实验如果仍
使用旧 200，必须明确声明为“历史复现”，不能把结果称为当前 baseline，也不能与新 300
直接做逐题配对比较。

## active baseline300 v1 的构造口径

- 分布参照：`data/eval_inputs/bird_train_filtered.jsonl`，即完整 normalized official BIRD
  train，6,601 条、69 个数据库。
- 可评测候选：`data/eval_inputs/bird_train_tool_compatible.jsonl`，5,915 条、69 个数据库；
  这些任务已在冻结源数据中标记 `metadata.tool_round_trip == verified`。
- 排除集合：旧 200 题，最终 overlap 为 0。
- 固定数量：300；固定 seed：`bird-train-baseline300-v1-20260805`。
- 选择字段只有 `example_id`、`db_id`、`question` 和 `external_knowledge`。
- 先按完整训练集的数据库比例分配整数 quota，并保证 69 个数据库各至少一题；再在每个
  数据库内部按 external knowledge 是否存在 × 问题字符长度五分位分配 quota。最后只在同一
  数据库内交换任务以校准全局联合分布，并用 SHA-256 做确定性 tie-break。
- 输出顺序也被冻结；使用 32/50/100/200 等前缀做诊断时，必须声明它是该 active cohort 的
  固定前缀，不能重新抽样后沿用同一 cohort 名称。

官方 BIRD train 不提供 difficulty 标签。历史 200 的 easy/medium/hard 来自 gold SQL
结构代理，而且使用了人为 80/60/60 quota；新 baseline 不再用它。`gold_sql`、`query`、
`gold_exec_results` 和 `gold_sql_path` 均不参与选择或排序，只保留在 task record 中供 harness
隐藏评分，绝不能进入模型或 teacher 上下文。单元测试会检查改变这些 gold 字段不改变所选
task ids。

## 训练任务的非空结果门槛（与评测 baseline 分离）

评测 baseline300 v1 保持冻结，不因训练数据门槛而改名或重排。用于 student/teacher 数据生成
的任务池从 2026-08-06 起必须先通过 `gold-denotation-nonempty-task-filter-v1`：harness 私下在
只读 SQLite 中执行 gold SQL，只判断是否至少返回一行，不读取 `gold_exec_results` 占位字段，
也不向模型、教师或外部 API 暴露 SQL、结果行、结果值或空/非空标签。

完整 6,601 条 train 的实测结果为 6,599 条非空、0 条真实空结果、2 条执行错误；后两条按
fail-closed 规则不再是训练任务。5,915 条 tool-compatible 候选全部成功且非空。当前 1,500
教师候选为 `bird_train_atomic_teacher1500_v2_nonempty.jsonl`；由于旧 1,500 本身全部非空，
v2 保留了完全相同的 task ids、顺序和任务文件 SHA-256，仅新增与过滤 manifest 的哈希绑定。
旧 v1 只用于历史复现。

完整的 task ids、数据库 quota、每个分层的计数、前缀审计和所有输入/输出 hash 见
`data/eval_inputs/bird_train_baseline300_v1.manifest.json`。可复现命令为：

```bash
.venv/bin/python src/eval/select_bird_train_baseline.py \
  --reference data/eval_inputs/bird_train_filtered.jsonl \
  --eligible data/eval_inputs/bird_train_tool_compatible.jsonl \
  --exclude data/eval_inputs/bird_train_tool_interface_validation200_version4.jsonl \
  --output data/eval_inputs/bird_train_baseline300_v1.jsonl \
  --manifest data/eval_inputs/bird_train_baseline300_v1.manifest.json \
  --count 300 \
  --seed bird-train-baseline300-v1-20260805 \
  --change-date 2026-08-05
```

## 分布审计

总变差距离（TV）为 0 表示两边分类分布完全相同，越小越接近。以下所有统计仅使用公开
选择字段：

| 指标（相对完整 6,601 条 train） | 旧 200 | 新 300 |
|---|---:|---:|
| 数据库覆盖 | 57/69 | **69/69** |
| 数据库分布 TV | 0.2184 | **0.0316** |
| 问题长度五分位 TV | 0.0429 | **0.0020** |
| external knowledge 分布 TV | 0.0165 | **0.0018** |
| knowledge × 长度联合分布 TV | 0.0553 | **0.0049** |
| 平均问题长度（字符） | 76.475 | **76.693**（全量 77.158） |
| 与旧 200 重叠 | — | **0** |

新 300 的 manifest acceptance gates 全部通过：记录数和 task id 唯一性正确、旧 cohort
overlap 为 0、覆盖全部数据库、数据库 TV ≤ 0.035、长度五分位 TV ≤ 0.03、knowledge TV
≤ 0.02、联合 TV ≤ 0.04，平均问题长度相对差 ≤ 5%。这些门槛描述的是公开输入分布相似性，
不声称 SQL 结构难度完全相同。

## 使用与比较规则

1. 新实验默认且只能把 `bird_train_baseline300_v1.jsonl` 称为当前 BIRD-train baseline。
2. 每个结果 manifest 必须记录 cohort path、SHA-256、task 数、输出顺序、模型、协议、解码、
   database snapshot 和 `bird-set` scorer；只写“baseline300”不够。
3. 逐题配对比较要求两臂使用完全相同的 300 个 task ids 和顺序。旧 200 与新 300 只能做
   分别报告，不能做 McNemar 等逐题配对统计。
4. 该 300 题是评测 cohort，不是 SFT/RL 训练数据来源。任何 trajectory 进入训练仍须遵守
   causal generation、fresh replay、no-leak 和 admission gates。
5. 新 300 尚未继承旧 200 上 version24/version50/version51 等历史分数；模型必须在新 cohort
   上重新运行后才有新的可比 baseline。

## 后续变更流程

每次 baseline 变更必须同时完成：

1. 新建递增版本文件（例如 `bird_train_baseline300_v2.jsonl`），禁止覆盖、删除或重排旧版本。
2. 冻结 reference、eligible、exclusion、seed、算法版本、所有输入 hash 和输出 hash。
3. 选择逻辑不得读取 gold；新增公开分层轴时，补充 gold-invariance 和确定性测试。
4. 验证记录数、唯一性、数据库可用性、与废弃/训练污染集合的 overlap，以及分布 gates。
5. 给被替代 manifest 写入 `deprecated_for_new_experiments`、replacement 和日期，但保留其
   JSONL 与历史报告。
6. 在本文档的“当前状态”“分布审计”和下面的“变更记录”追加一行，同时更新
   `docs/current/README.md`、`docs/current/evaluation.md` 和 `AGENTS.md`。
7. 不得把旧结果静默重标成新 cohort 结果；跨 cohort 只能清楚地分别报告。

## 变更记录

| 日期 | 变更 | 原因与兼容性 |
|---|---|---|
| 2026-08-05 | 激活 `bird_train_baseline300_v1.jsonl`；废弃旧 fixed-200 | 新 300 与旧 200 完全不重叠，覆盖 69/69 DB，并显著改善公开输入分布相似性。旧 JSONL 和历史报告保留，只允许复现。 |
