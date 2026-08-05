# Atomic version45 有界值搜索

日期：2026-07-30

## 定位

version45 保留 version44 的所有模型可见调用签名、inspect-only BIRD 列语义、
`inspect_rows`、no-plan、recent-4 history、关系执行、grounding 和 exact-table terminal
语义，只替换 `search_values` 的内部候选召回和对应 prompt 说明。

version44 的 exhaustive search 继续保留用于历史 replay。version45 通过 episode context
显式选择 `bounded-v1`，不会静默改变 version44 artifact。

## 候选召回

每个 `search_values` 调用按以下固定顺序执行：

1. 在 SQLite 中对请求 table 的目标列执行 case-insensitive exact 查询；
2. 如果存在 exact/normalized-exact 候选，直接返回这些候选，不再混入更远的 fuzzy
   alternatives；
3. 没有 exact 候选时，从 query 生成稳定的 token/trigram anchors；
4. SQLite 为每列召回最多 4096 个候选，按长度差、frequency、规范化文本、类型和原始
   文本稳定排序；
5. 仅对该有界候选池运行原有 deterministic lexical matcher；
6. 最终仍按 exact、normalized exact、prefix、token、substring、fuzzy 排序。

公开参数保持：

```text
search_values(table, query, column?, limit?, offset?)
```

新增 observation audit 字段：

- `total_matches_scope="bounded_candidate_pool"`；
- `candidate_count`；
- `candidate_limit_per_column=4096`；
- `candidate_truncated`；
- `truncated_columns`。

`candidate_truncated=true` 表示候选召回达到上限。Prompt 要求缩小 table、column 或 query，
不能把当前候选池当作全列穷举结果。每页仍最多 20 条，offset 在同一固定候选池排序中
稳定分页。

## 本地结果

在 version44 Gate50 中实际发生的全部 22 次搜索上离线重放：

- 22/22 首候选与 version44 exhaustive search 完全一致；
- 总执行时间 1.033 秒；
- 最慢 0.406 秒；
- 0 次 candidate truncation。

两个原先最慢的 `authors.Paper.Title` 查询：

| 查询 | version44 episode | version45 executor |
|---|---:|---:|
| Dissimilarity...完整标题 | 382.94s | 0.374s |
| Testing timed automata | 380.36s | 0.132s |

`Paper.Title` 有约 1,979,379 个 distinct values。version45 分别只召回 1 和 2 个
exact/normalized-exact 候选。

测试：

- SFT 全量、version44/version45/context/process-credit：234 passed，7 subtests passed；
- harness self-test：passed；
- `git diff --check`：passed。

## 外部教师 Gate50

第一次 version45 Gate50 在前 19 条语义 episode 正常后，外部 DeepSeek provider 开始
持续返回 native reasoning 但不返回 visible JSON；version39 探针也复现并进一步收到
连接断开。该批次已停止且没有混入最终结果。

provider 恢复后，先用当前最终 hash 完成 1/1 correct 单题 probe，再从头重跑同一冻结
50 题：

| 配置 | bird-set | Legal | Process errors |
|---|---:|---:|---:|
| version39 baseline | 39/50 | 50/50 | 2 |
| version44 exhaustive | 38/50 | 50/50 | 3 |
| version45 bounded | 37/50 | 50/50 | 3 |

version45 相对 version39 为 2 gains、4 regressions，双侧 exact McNemar `p=0.6875`；
相对 version44 为 3 gains、4 regressions，`p=1.0`。50 题全部合法终止，没有
transport、context 或 carrier failure，只有一次 completion retry。

四个相对 version39 的 regression 都没有调用 `search_values`，分别是 country code
未 join 成名称、最终列顺序错误、作者关联路径漏一行、算出两组计数后未选 top-1。
因此本次负差不是 bounded candidate recall 漏候选。工程上，整次墙钟从 version44 的
405.83s 降至 317.21s；同样执行大标题搜索的 `bird_train_06151` 从 382.94s 降至
20.38s。搜索延迟问题已解决，但能力没有获得 promotion。

version45 保持 diagnostic-only，不运行剩余 150 题，也不用于 SFT/RL。完整结果见
`docs/reports/evaluation/BIRD_ATOMIC_VERSION45_BOUNDED_SEARCH_PAIRED_GATE50_20260730_ZH.md`。
