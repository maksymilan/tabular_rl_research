# BIRD + Spider + SynSQL：9,000 题训练任务筛选

日期：2026-08-11  
状态：**9,000 道题筛选完成；仅为 rollout candidates，不是已准入 SFT 轨迹**

## 1. 目标口径

本次“9k”明确指 **9,000 道独立问题任务**，不指 action 数、turn 数或导出后的监督记录数。
每题后续可能产生多个工具 action，但 action 数不参与本次规模计算。

先前生成的 5,000 题 action-target v1 是对目标单位的错误理解，已在其 manifest 中标记为
`superseded-wrong-target-unit-do-not-use`。当前唯一有效版本为
`multisource-sft-question-selector-v2`。

## 2. 配额

总难度严格为 20% / 60% / 20%，来源为 35% BIRD、30% Spider、35% SynSQL；人类标注
来源 BIRD + Spider 合计 65%，避免合成数据占据绝对多数。

| 来源 | 简单 | 中等 | 困难 | 合计 |
|---|---:|---:|---:|---:|
| BIRD train | 1,100 | 1,850 | 200 | 3,150 |
| Spider train + others | 400 | 1,700 | 600 | 2,700 |
| SynSQL-2.5M | 300 | 1,850 | 1,000 | 3,150 |
| **合计** | **1,800** | **5,400** | **1,800** | **9,000** |

队列顺序采用固定的 `easy, medium, medium, medium, hard` 五题周期。任意以 5 为边界的
rollout 分片都精确保持 20/60/20；100、500、1,000 题前缀的各来源计数相对 35/30/35
目标最多偏差 1 题，完整 9,000 题严格为 35/30/35。全部 9,000 条均属于主问题集合，不使用
action-yield reserve 分界。

## 3. 私有 SQL 画像

`src/harness/sql_task_coverage_profile.py` 使用 SQLGlot 30.9.0 生成 sampling-only AST
画像。它不产生可执行 action，不定义 privileged path，也不推断感知、checkpoint、restore 或
模型实际工具序列。gold SQL 只在本地用于：

1. AST 结构画像与三个来源共享的难度分类；
2. 只读 SQLite 非空结果门。

画像覆盖 SELECT scope、子查询位置、相关子查询、CTE、窗口、WHERE/HAVING、谓词叶子与
布尔深度、join 类型与 edge、聚合函数、projection/算术/CASE/DISTINCT、ORDER/LIMIT/OFFSET、
四类集合运算、scope operator skeleton，以及 primitive 的 pair/triple 组合。

难度标签由三个数据源共享的 AST 规则产生，不直接采用 SynSQL 的源标签，也不把
parse-failed/unsupported 当作困难题。SynSQL 的 `sql_complexity` 只用于第一阶段短名单分层，
最终难度由统一分类器重算。

## 4. 可扩展 SynSQL 筛选

SynSQL `data.json` 是 8.7 GiB 顶层 JSON 数组，共 2,544,390 条。筛选器使用流式解析和按源
complexity 标签的稳定 min-hash reservoir：

- 全量扫描不把数据载入内存；
- 每个源标签保留 15,000 条，共 60,000 条短名单；
- 缓存不保存 `cot`；
- 只对短名单执行 SQLGlot AST 画像；
- 缓存绑定 Hugging Face commit/blob identity、seed、源字节数和 shortlist SHA-256。

每个 source × difficulty cell 内使用带 diminishing return 的覆盖贪心选择。稀有 primitive、
参数子类、pair、triple 和 skeleton 具有更高边际收益，同时加入 database diversity；每个数据库
内同一 literal-masked template 最多 3 条，cell 内最多 25 条。跨来源的相同规范化问题 +
canonical SQL 不重复。

## 5. 最终审计

| 指标 | 结果 |
|---|---:|
| 问题任务 | 9,000 |
| 数据库 | 3,164 |
| 私有结构特征 | 599 |
| operator skeleton | 185 |
| pair 类型 | 66 |
| triple 类型 | 219 |
| unique canonical SQL | 7,942 |
| unique literal-masked template | 7,853 |
| 最大 template 重数 | 4 |
| 与 active baseline300 / deprecated fixed200 / output-shape Gate20 重叠 | 0 |

主要 primitive presence：

| 结构 | 题目数 |
|---|---:|
| filter | 6,697 |
| join | 6,618 |
| aggregate | 5,624 |
| rank/order/limit | 3,391 |
| subquery | 1,703 |
| scalar arithmetic | 1,449 |
| DISTINCT | 1,050 |
| CASE | 854 |
| CTE | 807 |
| set operation | 653 |
| window | 459 |

集合运算包含 except 175、intersect 191、union 117、union_all 172。普通聚合包含 count、sum、
avg、max、min、group_concat 和 json_array_agg；窗口函数与普通聚合分开计数。

BIRD 来源使用已认证的 `tool_compatible_nonempty_v1` 池。Spider 和 SynSQL 的最终入选题逐条
通过隐藏 gold SQL 的 read-only/nonempty gate；空结果和 SQLite 执行错误被拒绝。manifest
只保存原因计数，不保存 SQL、结果行或值。

teacher-visible 投影仅包含 `dataset/split/example_index/example_id/db_id/question/
external_knowledge`，没有 `gold_sql/query/sql/db_path/cot/profile/difficulty`。

## 6. 冻结文件

- `data/sft_task_selection/bird_spider_synsql_sft9k_questions_v2.tasks.jsonl`
  - SHA-256 `a18b8987591f0f5ab71db6cfd8e93c3422efabc70e67dbad9fac55542df7c911`
- `data/sft_task_selection/bird_spider_synsql_sft9k_questions_v2.private_profiles.jsonl`
  - SHA-256 `bea9dd416a06e6fc656e342c16fdabe6eb6b0a0b4cec3a962a6c293724fc2011`
- `data/sft_task_selection/bird_spider_synsql_sft9k_questions_v2.teacher_visible.jsonl`
  - SHA-256 `fa7f1c75c3aae098d22d0e64cf29b84dddde944fdb6c35069485e44a9420512d`
- `data/sft_task_selection/bird_spider_synsql_sft9k_questions_v2.manifest.json`
- `data/sft_task_selection/cache/synsql_sft9k_questions_v2_shortlist.jsonl` 及 manifest

`data/` 由仓库根 `.gitignore` 排除，任务和私有画像不会进入 Git。

复现命令：

```bash
.venv/bin/python -u src/sft/select_multisource_sft_tasks.py
```

## 7. 后续边界

这 9,000 条是问题任务集合，不是把 gold SQL 编译得到的训练轨迹。后续仍须对每题运行真实
model↔Harness 因果 episode，并经过 strict terminal correctness、fresh replay、no-leak 和质量门。
静态 SQL 难度与组合分布还必须在实际轨迹上二次审计工具序列、依赖深度、checkpoint/restore、
错误与成功淘汰率。

checkpoint-relalg 当前仍是 diagnostic-only；在专用 scheme-aware exporter 和明确准入门完成前，
本题集及未来 episode 都不得重标为 atomic SFT 或加入现有训练 mixture。
