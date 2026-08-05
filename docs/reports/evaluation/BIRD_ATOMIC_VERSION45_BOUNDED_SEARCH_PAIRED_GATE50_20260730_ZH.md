# BIRD Atomic version45 有界搜索 Paired Gate50

日期：2026-07-30

## 结论

外部 provider 恢复后，使用当前最终 prompt/hash 从头完成了 `version45` 的同一冻结
Gate50。运行没有 carrier、transport 或 context failure，但准确率没有超过
`version39` baseline：

| 配置 | bird-set | Legal | Process errors |
|---|---:|---:|---:|
| version39 baseline | 39/50（78%） | 50/50 | 2 |
| version44 exhaustive search | 38/50（76%） | 50/50 | 3 |
| version45 bounded search | 37/50（74%） | 50/50 | 3 |

version45 相对 version39 为 35 题都对、9 题都错、2 个 gain、4 个 regression，净
-2 题（-4pp），双侧 exact McNemar `p=0.6875`。相对 version44 是 3 个 gain、4 个
regression，双侧 exact McNemar `p=1.0`。两组差异都不显著，但 version45 没有达到准确率
promotion 条件，不运行剩余 150 题，也不作为 SFT/RL 数据协议。

工程结论与能力结论必须分开：

- **工程问题已解决**：有界 SQL 候选召回消除了百万 distinct value 列的全量 Python
  fuzzy 扫描；离线重放保持 22/22 首候选。
- **能力没有提升**：新增值检索、inspect-only 列语义、no-plan 和 `inspect_rows` 这组
  接口变化，在单次 K=1 Gate50 上仍未稳定优于 version39。

## 严格设置

- 冻结题集：
  `data/eval_inputs/bird_train_tool_interface_validation200_version4.jsonl` 前 50 题；
- 题集 SHA-256：
  `6b469215ac96e09c1046fee539ff4c4885b58e9b7a0b6fdb2187082267dcde36`；
- 模型：`deepseek-v4-flash`；
- 单次采样，`K=1`；
- `bird-set`；
- strict no-repair parser；
- DeepSeek thinking + JSON Output；
- max steps 30，max tokens 2048；
- recent-4 rolling legal history；
- 12 workers；
- protocol：`version45`；
- protocol hash：`6826c126bd946021`；
- value search policy：`bounded-sql-candidate-v1`；
- provider prompt SHA-256：
  `0be16f4500dd138e7dc0e86defc3134fab415ea8ae547c15b9ccd3b0ee22f55c`；
- model-visible tool schema SHA-256：
  `5db2fa63fc7a138e71df1638598e3397ad9973f9f9b07cbb6011b354ce61a04b`。

正式运行前先用当前 hash 做了单题 provider probe，结果 1/1 correct。随后从空目录启动
50 题，未复用 provider 故障期间的任何 partial record。

## 配对变化

### 两个 gain

1. `bird_train_04189`：free sports Apps。version39 使用 left join，保留没有 review
   match 的行；version45 使用 inner join，denotation 正确。该题没有调用
   `search_values`，属于关系决策采样差异。
2. `bird_train_00593`：Berry Keebler 的用药时长。两组都找到了两段 11/18 天记录；
   version39 终表错误保留 `START/STOP`，version45 最后只投影 `duration_days`。version45
   调用五次值搜索以确认姓名、药名和病因，但直接修复得分的是最终输出投影。

### 四个 regression

1. `bird_train_03390`：返回了 country code，而不是 join 后的 nation name；
2. `bird_train_06336`：终表列顺序为 `id, word`，题目和 gold 要求 `word, id`；
3. `bird_train_06171`：选择了会丢失一名作者的 Author 关联路径，少返回一行；
4. `bird_train_06454`：已经算出 RAIL 和 MAIL 的计数，但没有再选 top-1，终表仍是两行
   `mode,count`。

这四题均为 0 process error，且均未调用 `search_values`。因此不能把 4 个 regression
解释成 bounded recall 漏候选；它们是输出形状、关联路径和终止决策错误。13 个 version45
wrong answer 也全部没有 process error，当前主要剩余瓶颈仍是合法关系程序的语义和精确
输出，而不是协议可执行性。

## search_values 行为

version45 在 19/50 题中共调用 33 次搜索：

- 29 次限定 column，4 次搜索 table 全列；
- 首候选类型：19 exact、3 normalized exact、5 prefix、2 substring、4 empty；
- 0 次使用非零 offset；
- 2 次 `candidate_truncated=true`，均来自同一错误任务对 `Journal.Id = "0"` 的重复宽
  查询；
- 最大候选池 4096，平均候选数 285.1。

在“本次 version45 实际选择调用搜索”的 19 题上，version45 为 15/19，同题 version39
为 14/19，只有 `bird_train_00593` 一个 gain、无 regression。这个集合是模型行为产生的
post-treatment subset，不能把 15/19 对 14/19 当作独立的工具增益估计；它只能说明本次
负净差不是发生在实际搜索题上。

inspect-only 列语义实际出现在 8 题、17 次调用中；version45 为 5/8，同题 version39
为 6/8。唯一 paired regression 是已经算出正确计数但未选择 top-1 的 ship-mode 题，
没有证据表明列描述给了错误事实，也没有形成稳定增益。

## 效率

| 指标 | version39 | version44 | version45 |
|---|---:|---:|---:|
| 总动作数 | 314 | 321 | 333 |
| 平均动作/题 | 6.28 | 6.42 | 6.66 |
| 总 tokens | 2,097,832 | 2,129,359 | 2,268,122 |
| API request attempts | 316 | 321 | 334 |
| 逐题 elapsed 求和 | 1,623.14s | 2,432.90s | 1,874.14s |
| 整次墙钟时间 | 249.26s | 405.83s | 317.21s |

相对 version44，version45 墙钟下降 21.8%，逐题 elapsed 求和下降 23.0%。由于本次
version45 多了 12 个动作和 6.5% tokens，且 provider 延迟会波动，Gate50 总墙钟不能
单独当作 executor microbenchmark。更直接的证据是：

- version44 的 22 个真实 search 调用离线重放到 version45：总计 1.033s、最慢
  0.406s、22/22 首候选保持；
- `bird_train_06151` 同样执行标题搜索，整题从 version44 的 382.94s 降至 version45
  的 20.38s；
- version44 两个约 380s 的大标题任务在 version45 新运行中均不再是慢任务。

version45 有 0 transport retry、0 context retry、0 carrier retry、1 completion retry。
因此本次 37/50 是完整的语义评测结果，不再受此前 provider outage 污染。

## Artifact

- version45 manifest：
  `data/trajectories/version45_paired_gate50_retry_20260730/version45_bounded_fresh/verified.manifest.json`
- version45 all records：
  `data/trajectories/version45_paired_gate50_retry_20260730/version45_bounded_fresh/verified.all.jsonl`
- current-hash provider probe：
  `data/trajectories/version45_paired_gate50_retry_20260730/provider_probe/verified.manifest.json`
- frozen version39 baseline：
  `data/trajectories/version44_paired_gate50_20260730/version39_baseline/verified.all.jsonl`
- version44 exhaustive control：
  `data/trajectories/version44_paired_gate50_20260730/version44_new_tools/verified.all.jsonl`
