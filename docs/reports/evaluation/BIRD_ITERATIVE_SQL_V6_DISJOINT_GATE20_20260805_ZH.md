# BIRD iterative-SQL v6 独立 Gate20

日期：2026-08-05  
状态：已完成；总分持平，目标规则未被该切片直接覆盖，不推广

## 结论

在 active `bird_train_baseline300_v1` 冻结第 51–70 题（`start=50, n=20`）上，官方
DeepSeek v4 Flash 的 `iterative-sql-v5` 与 `iterative-sql-v6` 都得到 **12/20**，且均为
**20/20 legal**。配对结果为 1 gain、1 regression、11 both correct、7 both wrong，exact
two-sided McNemar `p=1.0`。

v6 使用 97 actions，略少于 v5 的 100；process errors 都为 2。v6 总 tokens 为 340,873，
高于 v5 的 331,641（+2.8%）。因此没有准确率、可靠性或成本上的推广证据。

更重要的是，这 20 题中没有 external knowledge 把一个姓名槽位映射到多个字段，也没有明确
要求把多个姓名字段拼成一个 formatted/combined string 的反向控制题。唯一姓名映射题是
单字段 `supplier name -> s_name`，两臂都正确。因此本门只能检查一般行为保持，不能证明 v6
新增的 multi-field/full-name 规则已经修复 Prefix50 的目标失败。

## 冻结设置与授权边界

- cohort：`data/eval_inputs/bird_train_baseline300_v1.jsonl`；
- SHA-256：`87b37b307ea2710acdff7d03bdc474a6ba2cdb8ed51b005cff0fe8fa54c67c03`；
- slice：冻结顺序 `start=50, n=20`，与已消费 Prefix50 不重叠；
- model/provider：`deepseek-v4-flash`，官方 `https://api.deepseek.com`；
- K=1；lazy catalog；recent-4；30 steps；每类最多 3 errors；
- 2,048 max tokens；20 preview rows；20 秒 SQLite deadline；`bird-set`；
- v5/v6 并发运行，除协议版本和对应 prompt 外，其余任务、数据库、预算、上下文与 scorer
  相同；
- 用户明确授权发送这 20 题的 question、external knowledge、catalog/schema 和逐步只读
  工具反馈；gold SQL、gold rows 与隐藏验证器数据未进入外部模型输入。

## 总体统计

| 指标 | v5 | v6 | 变化 |
|---|---:|---:|---:|
| correct | 12/20 | 12/20 | 0 |
| legal terminal | 20/20 | 20/20 | 0 |
| actions | 100 | **97** | -3.0% |
| mean / median / max actions | 5.00 / 5.0 / **9** | **4.85 / 4.5** / 12 | -0.15 / -0.5 / +3 |
| `execute_sql` attempts | 80 | **77** | -3 |
| successful `execute_sql` | 78 | **75** | -3 |
| `submit_sql` attempts | 20 | 20 | 0 |
| process errors | 2 | 2 | 0 |
| prompt tokens | **296,392** | 306,143 | +3.3% |
| completion tokens | 35,249 | **34,730** | -1.5% |
| reasoning tokens | 31,123 | **30,420** | -2.3% |
| total tokens | **331,641** | 340,873 | +2.8% |
| API request attempts | 106 | **100** | -6 |
| transport retries | 4 | **1** | -3 |
| completion retries | 2 | 2 | 0 |
| carrier retries | 0 | 0 | 0 |
| fresh replay correct | 12 | 12 | 0 |
| structural pass | 20/20 | 20/20 | 0 |

## 配对差异

### v6-only gain：`bird_train_00054`

问题只要求列出高于平均年销量的 title，并按 publisher name 排序。v5 返回 publisher、title
和 `ytd_sales` 三列，把排序键和度量带进最终答案；v6 只返回 title，同时仍通过 join 保持所需
排序。这是一般 exact-output-slot 改进，但不涉及多字段姓名映射。

### v5-only regression：`bird_train_00324`

问题询问阿根廷哪个 city 发推最多。v5 只返回 city；v6 返回 city 加 `tweet_count` helper 列。
这直接违反 v6 自己的“without helper fields”规则，说明更明确的关系型输出说明没有稳定消除
所有 helper-field 行为。它同样不涉及多字段姓名映射。

两项差异互相抵消；不存在净准确率变化。

## 过程错误

- 两臂都在 `bird_train_03039` 首次引用不存在的列，收到 SQLite error 后恢复并最终正确；
- v5 在 `bird_train_00193` 首次使用不存在的 `name` 列，查看真实 schema 后恢复并正确；
- v6 在 `bird_train_01299` 一次提交多个 schema 查询，被只读单语句校验拒绝后恢复并正确。

没有 carrier error、危险 SQL、非法终止或隐藏 verifier feedback。

## 审计

两臂各 20 条都通过 strict reparse、SQL preview fresh replay、terminal denotation replay、结构、
prior-execution grounding 和 no-hidden-input-key 检查：

- v5：20/20 structural pass，12/12 recorded success fresh replay；
- v6：20/20 structural pass，12/12 recorded success fresh replay。

## 决策

1. 不把 v6 描述为准确率提升版本，也不进入 SFT/RL。
2. 本 Gate20 不支持回滚安全性以外的推广结论：总分持平，tokens 增加，且目标规则未被覆盖。
3. 不在已消费的第 51–70 题上继续调 prompt 后复测。
4. 如要验证 multi-field/full-name 规则，必须预先冻结一个新的、与前 70 题不重叠的定向 slice，
   同时包含多字段分列目标题、明确拼接反向控制题、单字段姓名控制题和普通输出控制题。
5. 在该定向门之前，保留 v5 可复现路径；v6 继续为 unpromoted diagnostic candidate。

## 产物

- v5：`data/trajectories/iterative_sql_20260805/baseline300_50_70_v5_flash_r1/`；
- v6：`data/trajectories/iterative_sql_20260805/baseline300_50_70_v6_flash_r1/`；
- paired summary：
  `data/trajectories/iterative_sql_20260805/paired_baseline300_50_70_v5_v6_summary.json`；
- 两臂目录内均有 `replay_structural_audit.json`。
