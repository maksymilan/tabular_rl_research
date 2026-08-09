# BIRD iterative-SQL v6 Flash baseline300 全量结果

日期：2026-08-05  
状态：full300 单臂诊断完成；绝对能力与可靠性可用，但没有代表性 v5 对照，不构成版本优越性证明

## 结论

在冻结的 `baseline300` 全部 300 题上，官方 DeepSeek v4 Flash 的
`iterative-sql-v6` 得到 **215/300（71.67%）**，95% Wilson 区间为
**66.32%–76.47%**；合法终止 **298/300（99.33%）**。全部 300 条记录通过 fresh replay
与结构审计：`recorded_correct=215`、`replay_correct=215`、`structural_pass=300/300`。

前 70 题曾参与 v5/v6 的设计或诊断，不能作为严格泛化集。主要推广读数应使用从未用于
v6 设计的冻结第 71–300 题：**168/230（73.04%）**，95% Wilson 区间
**66.96%–78.36%**；合法终止 **228/230（99.13%）**。两个新的 115 题切片都恰好得到
84/115，说明该主要结果不是由其中一个分片偶然拉高。

这次按照用户授权只运行 v6。没有在干净 230 题上运行 v5，因此不能用 73.04% 声称 v6
总体优于 v5，也不能与不同 cohort 上的 atomic/native-tool-bundle 数字做直接比较。前 70
题的历史同题 v5 结果反而是 48/70，v6 为 47/70（3 gains、4 regressions，双侧 exact
McNemar `p=1.0`）；它是已消费集合上的次要诊断，既不推翻目标 Gate20 的能力恢复，也不构成
总体版本提升证据。

## 冻结设置与授权边界

- cohort：`data/eval_inputs/bird_train_baseline300_v1.jsonl`；
- SHA-256：`87b37b307ea2710acdff7d03bdc474a6ba2cdb8ed51b005cff0fe8fa54c67c03`；
- 300 tasks、69 databases，冻结顺序不变；
- model/provider：`deepseek-v4-flash`，官方 `https://api.deepseek.com`；
- protocol/interface：`iterative-sql-v6` / `execute-sql-submit-sql-v6`；
- protocol hash：`176ce977b411636e`；public tool schema hash：
  `7948634d6c96fe3b19ea206fe2b5f9ede673e1b219879be4c4a83709336172d4`；
- K=1；lazy catalog；recent-4；30 steps；每类最多 3 errors；
- 2,048 max tokens；20 preview rows；20 秒 SQLite deadline；`bird-set`；
- 用户授权发送 question、external knowledge、catalog/schema 与逐步只读工具反馈；gold SQL、
  gold rows、gold sample、gold row count 和隐藏 verifier 数据未进入外部模型输入；
- 第 51–70 题复用已经完成并审计的 v6 轨迹；其余 280 个 episode 为本次新请求；
- 只产生 diagnostic artifacts，`sft_export_eligible=false`、
  `training_admission=diagnostic_only`。

## 总体与干净推广子集

| 指标 | full300 | 第 1–70 题（已消费） | 第 71–300 题（干净） |
|---|---:|---:|---:|
| correct | **215/300** | 47/70 | **168/230** |
| accuracy | **71.67%** | 67.14% | **73.04%** |
| accuracy 95% Wilson CI | 66.32%–76.47% | 55.50%–77.00% | 66.96%–78.36% |
| legal terminal | **298/300** | 70/70 | **228/230** |
| actions | 1,698 | 351 | 1,347 |
| mean / median actions | 5.66 / 5 | 5.01 / 4 | 5.86 / 5 |
| P90 / P95 / max actions | 9 / 12 / 30 | 约 8 / 10 / 12 | 约 9 / 13 / 30 |
| process errors | 72 | 10 | 62 |
| total tokens | 7,000,561 | 1,241,703 | 5,758,858 |
| mean / median tokens | 23,335 / 13,986 | 17,739 / 12,697 | 25,039 / 14,063 |
| ≥50k-token episodes | 23 | 3 | 20 |

### 运行切片

| baseline300 位置 | 来源 | correct | legal | actions | errors | tokens |
|---|---|---:|---:|---:|---:|---:|
| 1–50 | 本次新请求 | 35/50 | 50/50 | 254 | 8 | 900,830 |
| 51–70 | 授权复用 | 12/20 | 20/20 | 97 | 2 | 340,873 |
| 71–185 | 本次新请求 | 84/115 | 114/115 | 670 | 34 | 2,868,701 |
| 186–300 | 本次新请求 | 84/115 | 114/115 | 677 | 28 | 2,890,157 |

第 71–185 与第 186–300 两段的准确率、legal 和 token 量高度接近。数据库覆盖为 69 个；每库
1–17 题。按数据库不加权的 macro accuracy 为 72.82%，但许多数据库样本很少，所以主结论
仍采用逐题 micro accuracy。

## 工具调用、错误与恢复

全量 1,698 个模型 action 中有 1,694 个严格解析成功：

- `execute_sql`：1,394 次；
- `submit_sql`：300 次；
- 其余 4 次没有形成合法 parsed action。

`submit_sql` 次数不能等同于合法终止数：两个 episode 的初次非法提交收到反馈后恢复，另两个
episode 在 30 步上限前没有合法提交。因此最终为 298 个合法终止和两个 `max_steps`。

72 个过程错误分布如下：

| error type | 事件数 | 涉及任务数 | 其中最终正确 |
|---|---:|---:|---:|
| `no_progress_error` | **33** | 12 | 3 |
| `execution_error` | 28 | 21 | 16 |
| `sql_validation_error` | 5 | 5 | 4 |
| `argument_validation_error` | 3 | 3 | 1 |
| `protocol_error` | 3 | 3 | 2 |

共有 39/300 个 episode 至少发生一次过程错误，其中 25 个最终正确（64.10%）；无过程错误的
261 个 episode 中 190 个正确（72.80%）。最明显的剩余策略问题不是一般 SQLite 错误，而是
重复成功 SQL 或等价无进展探索：`no_progress_error` 集中在 12 题上产生 33 次，且只有 3 题
最终恢复。这说明 v6 的 repeat guard 能显式暴露循环，却还不能稳定把模型引导到新的证据或
尽早提交。

83 个失败是合法但语义错误的 `wrong_answer`；另 2 个为 `max_steps`。根据隐藏反馈边界，前者
终止后没有把 judge 差异返回给模型。

## Token 与请求可靠性

| 指标 | full300 | 干净 230 |
|---|---:|---:|
| prompt tokens | 6,085,538 | 4,961,571 |
| completion tokens | 915,023 | 797,287 |
| reasoning tokens | 836,128 | 734,781 |
| total tokens | **7,000,561** | **5,758,858** |
| P90 / P95 episode tokens | 约 39,030 / 61,789 | 约 40,063 / 74,679 |
| max episode tokens | 319,470 | 319,470 |
| API request attempts | 1,792 | 1,433 |
| completion retries | 93 | 86 |
| transport retries | 1 | 0 |
| carrier / context retries | 0 / 0 | 0 / 0 |

93 次 completion retry 是长度截断后的有界同轮客户端重试，不是额外语义 action。没有
carrier 或 context-shape 故障，官方 provider 侧只有一次 transport retry，说明本次 legal
缺口来自模型在 episode 内的长循环，而不是 provider carrier 不稳定。

成本呈明显长尾：

| baseline300 位置 | task | correct/legal | steps | errors | tokens |
|---:|---|---|---:|---:|---:|
| 269 | `bird_train_02549` | 否 / 否 | 30 | 10 | 319,470 |
| 80 | `bird_train_01431` | 否 / 否 | 30 | 5 | 291,170 |
| 166 | `bird_train_05038` | 是 / 是 | 29 | 4 | 166,518 |
| 82 | `bird_train_05012` | 否 / 是 | 26 | 5 | 202,214 |
| 215 | `bird_train_05347` | 否 / 是 | 23 | 4 | 154,544 |
| 87 | `bird_train_04127` | 否 / 是 | 20 | 1 | 155,201 |

两个 `max_steps` episode 消耗 610,640 tokens，占全量 token 的 **8.72%**，并产生 15/72
过程错误。六个 ≥20-step episode 合计消耗 1,289,117 tokens，占 **18.41%**。下一轮优化应
优先处理无进展循环和截断重试，而不是继续加长通用 prompt。

## v6 输出形状规则的自然覆盖

干净 230 题不是为该规则定向挑选，因此覆盖量很小，但观察到的自然目标全部正确：

- 3 个 `full name refers to field1, field2[, field3]` 任务均返回独立有序列：
  `bird_train_02519`、`bird_train_05131`、`bird_train_05613`，3/3；
- 2 个问题直接要求多个姓名字段的任务也保持分列：
  `bird_train_05532`、`bird_train_01144`，2/2；
- `bird_train_06007` 的 `full name refers to LongName` 是单字段映射，保持一列，1/1。

已消费前 70 题中的两个 multi-field full-name 映射
`bird_train_02546`、`bird_train_04251` 也都正确分列，但不计入泛化证据。完整 baseline300
没有“明确要求拼成一个 formatted/combined string”的真实反向控制，因此 Target Gate20 报告
中的 anti-overseparation 未验证边界仍然存在。

## 与历史 v5 的有限对照

将冻结 v5 的 Prefix20、21–50、51–70 三段按相同任务顺序拼成历史 first70，可得：

| first70 paired outcome | 数量 |
|---|---:|
| v5 correct | 48/70 |
| v6 correct | 47/70 |
| v6 gains / regressions | 3 / 4 |
| both correct / both wrong | 44 / 19 |
| exact two-sided McNemar | `p=1.0` |

这些任务已经用于 v5/v6 设计与诊断，不能作为推广检验。它只说明 v6 的输出形状规则不是
一个能在所有题型上自动提高准确率的通用增强。Target Gate20 的 5/20 定向净增益回答的是
“规则能否修复目标表示问题”，本 full300 的干净单臂 73.04% 回答的是“冻结 v6 在代表性任务
上的绝对表现”；由于缺少 clean230 v5 arm，两者都没有回答“v6 是否总体优于 v5”。

## 决策

1. 冻结本次 `iterative-sql-v6` full300 结果，作为当前两 SQL 工具协议的绝对性能诊断：
   215/300 correct、298/300 legal。
2. 主要泛化读数固定为未参与设计的第 71–300 题 168/230；不得用 full300 的 71.67% 掩盖
   前 70 题已消费这一事实。
3. v6 继续保持 diagnostic-only，不从这些评测轨迹导出 SFT/RL 数据。
4. 不声称 v6 优于 v5、atomic 或 native-tool-bundle；证明总体版本增益需要在同一干净集合上
   运行冻结 v5 paired arm，而用户本次明确要求只运行 v6。
5. 下一次工程优化优先针对 12 个 no-progress-loop 任务、completion truncation，以及两个
   max-step 长尾；不要根据 83 个 hidden wrong-answer 的 gold 差异继续调 prompt。

## 产物与审计

- 严格顺序合并工件：
  `data/trajectories/iterative_sql_20260805/baseline300_full300_v6_flash_reuse_r1/`；
- 合并来源在工件 `manifest.json` 的 `composition_sources` 中逐段记录；
- 合并策略：`strict-disjoint-task-order-v1`；
- 全量审计：同目录 `replay_structural_audit.json`；
- 合并工具：`src/eval/merge_evaluation_slices.py`；
- 第 51–70 题原始配对报告：
  `docs/reports/evaluation/BIRD_ITERATIVE_SQL_V6_DISJOINT_GATE20_20260805_ZH.md`；
- v6 目标能力报告：
  `docs/reports/evaluation/BIRD_ITERATIVE_SQL_V6_TARGET_GATE20_RESULT_20260805_ZH.md`。
