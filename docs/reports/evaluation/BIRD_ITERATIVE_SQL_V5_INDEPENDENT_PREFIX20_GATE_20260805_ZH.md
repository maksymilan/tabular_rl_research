# BIRD iterative-SQL v5 独立 Prefix20 Gate

日期：2026-08-05  
状态：已完成；独立小样本呈正向结果，但未达到统计显著性，仍为 diagnostic-only

> 后续状态：Prefix50 扩展已完成，36/50 versus v4 34/50，但 legal、process errors 和 tokens
> 均回退，停止扩量。本报告只保留 Prefix20 的冻结中间结果；最终决策以
> `BIRD_ITERATIVE_SQL_V5_PREFIX50_EXPANSION_20260805_ZH.md` 为准。

## 结论

在 active `bird_train_baseline300_v1` 的冻结前 20 题上，DeepSeek v4 Flash 使用
`iterative-sql-v5` 得到 **14/20**，冻结 v4 对照为 **12/20**；两臂均为 **20/20 legal**。
逐题结果是 2 个 v5-only gains、0 regressions、12 个 both correct、6 个 both wrong。exact
two-sided paired `p=0.5`，所以这是正向小门信号，不是显著准确率提升。

v5 同时把总动作从 114 降到 103（-9.6%），总 tokens 从 383,497 降到 349,253（-8.9%）。
process errors 从 3 增到 4；没有 provider carrier 或 transport failure。两臂全部 20 条均通过
fresh replay、结构和 no-hidden-input-key 审计。

## 独立性与冻结设置

- cohort：`data/eval_inputs/bird_train_baseline300_v1.jsonl`；
- SHA-256：`87b37b307ea2710acdff7d03bdc474a6ba2cdb8ed51b005cff0fe8fa54c67c03`；
- slice：冻结顺序 `start=0, n=20`；
- 该 baseline300 与废弃 fixed-200 overlap 为 0，因此没有参与 v4 的 Gate15 prompt 优化；
- model/provider：`deepseek-v4-flash`，官方 `https://api.deepseek.com`；
- K=1；lazy catalog；recent-4；30 steps；每类最多 3 errors；
- 2,048 max tokens；20 preview rows；20 秒 SQLite deadline；`bird-set`；
- 两臂并发运行，任务、数据库、预算、context 和 scorer 相同；
- 运行前冻结 v4/v5 prompt，看到结果后没有修改 v5。

外部 API 只接收用户授权的 question、external knowledge、catalog/schema 与逐步只读工具反馈；
gold SQL、gold rows 和隐藏验证器数据只在本地 harness 中使用。

## 总体指标

| 指标 | v4 | v5 | 变化 |
|---|---:|---:|---:|
| correct | 12/20 | **14/20** | +2 / +10 pp |
| legal terminal | 20/20 | 20/20 | 0 |
| actions | 114 | **103** | -9.6% |
| mean / median actions | 5.70 / 5 | **5.15 / 5** | -0.55 / 0 |
| successful `execute_sql` | 91 | **79** | -12 |
| `submit_sql` attempts | 21 | **20** | -1 |
| process errors | **3** | 4 | +1 |
| prompt tokens | 355,687 | **322,349** | -9.4% |
| completion tokens | 27,810 | **26,904** | -3.3% |
| reasoning tokens | 22,596 | **21,966** | -2.8% |
| total tokens | 383,497 | **349,253** | -8.9% |
| API request attempts | 114 | **104** | -10 |
| carrier / transport retries | 0 / 0 | 0 / 0 | 0 |
| completion retries | 0 | 1 | +1 client retry |
| fresh replay correct | 12 | **14** | +2 |
| structural pass | 20/20 | 20/20 | 0 |

`execute_sql` 的 82 个 v5 parsed calls 中有 3 个 SQLite execution errors，所以产生 79 个成功
query audits。79 个 audit 均为 `literal_only_select=false`；本门没有观察到把 preview 值抄成
常量最终查询的行为。

## 两个 paired gains

### `bird_train_00218`

问题只要求 14 分钟电影的 genre。v4 返回 title、runtime、genre 三列；v5 在四步内只返回
`genre_name`。这是 v5 的 exact output-slot 和“不带 helper fields”约束直接覆盖的错误类型。

### `bird_train_02757`

问题要求最常见 contact role。v4 返回 top 5，并包含内部 `ContactTypeID`、名称和 count；v5
返回 top 1 的 role name。该恢复同时符合 top-N 默认和 exact output representation 约束。

本门没有 v4-correct / v5-wrong 的 paired regression。

## 六个共同失败

| 任务 | 主要现象 | 审核归类 |
|---|---|---|
| `bird_train_03272` | external formula 明写 percentage × population，v5 仍擅自除以 100，并多返回 country name | 明确 formula/output-slot 服从失败 |
| `bird_train_04559` | 在 `historical-terms` 与 `current-terms` 中选择了语义上更像历史记录的前者，reference 使用后者 | source-table 歧义未消除 |
| `bird_train_00110` | external knowledge 明确指定 `ser_time`；v5 验证其最大值仅 `00:28:59` 后忠实得到 0，reference 使用 `ser_start` | external-knowledge/reference 冲突 |
| `bird_train_01766` | question/external knowledge 要求最高 enrollment school；v5 按 enrollment 选校，reference 实际按 bankruptcy count 排名 | question/reference measurement 冲突 |
| `bird_train_03052` | external knowledge 要求拼接生日并取 `MAX(GA)`；v5 拼接且把文本 GA 转为数值，reference 返回三个字段并按原 TEXT 排序 | external-knowledge/reference representation 与 stored-type 冲突 |
| `bird_train_03493` | question 要求 crime types；v5 返回出现最多的 `primary_description`，reference 返回常量式 `domestic='TRUE'` | question/reference output-semantic 冲突 |

因此 raw 14/20 中，至少四个 verifier failure 不能通过“更加服从 question/external knowledge”
同时修复；把它们直接用作 prompt/SFT 负例会教模型忽略显式任务契约。剩余两个可操作问题是
formula/output-slot 服从和 source-table 消歧，但本次 prefix20 已成为测试集，不应再据此修改
v5 并在同一集合复测。

## Process errors

- v4：1 次未先执行 exact final SQL；2 次把 `execute_sql` 拼成 `execute_ssql`。
- v5：1 次 `execute_ssql`；1 次带连字符表名的未加引号 PRAGMA；1 次 ambiguous column；
  1 次错误列名。

四个 v5 errors 都发生在最终错误任务上，且 episode 全部恢复为合法提交。它们说明 v5 的
语义 prompt 没有造成合法性回退，但 schema/identifier 书写仍是独立过程错误源。

## 决策

1. 保留 v5 作为当前 iterative-SQL 候选；本 Gate 不回退到 v4。
2. 不宣称显著准确率提升，也不开放 SFT/RL admission。
3. 不用这 20 题继续调 v5 prompt；它们现在是冻结测试结果。
4. 后续已在不改 prompt 的前提下新增 30 条并完成 Prefix50；扩展结果未通过可靠性与成本门，
   不再继续。若设计 v6，必须使用与已消费 Prefix50 不重叠的新 slice。

## 产物

- v4：`data/trajectories/iterative_sql_20260805/baseline300_prefix20_v4_flash_r1/`；
- v5：`data/trajectories/iterative_sql_20260805/baseline300_prefix20_v5_flash_r1/`；
- paired machine-readable summary：
  `data/trajectories/iterative_sql_20260805/paired_baseline300_prefix20_v4_v5_summary.json`；
- 两臂目录中的 `replay_structural_audit.json` 均为 20/20 structural pass。
