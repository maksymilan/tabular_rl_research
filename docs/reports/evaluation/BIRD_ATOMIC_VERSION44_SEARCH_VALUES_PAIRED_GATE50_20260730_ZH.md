# BIRD Atomic version44 新工具 Paired Gate50

日期：2026-07-30

## 结论

`version44` 的新接口能够被 DeepSeek v4 Flash 稳定调用，但在这次冻结 50 题配对评测中
没有超过 `version39` baseline：

| 配置 | bird-set | Legal | Process errors |
|---|---:|---:|---:|
| version39 baseline | 39/50（78%） | 50/50 | 2 |
| version44 新工具 | 38/50（76%） | 50/50 | 3 |
| 净变化 | -1 题（-2pp） | 0 | +1 |

配对结果为 34 题都对、7 题都错、4 个 gain、5 个 regression。双侧 exact McNemar
`p=1.0`，没有显著差异。不要把 version44 当作准确率 promotion，也不要运行剩余 150
题或用于 SFT/RL；应先解决搜索延迟和候选值使用策略，再重跑同一 Gate50。

## 严格配对设置

- 冻结题集：
  `data/eval_inputs/bird_train_tool_interface_validation200_version4.jsonl` 前 50 题；
- 题集 SHA-256：
  `6b469215ac96e09c1046fee539ff4c4885b58e9b7a0b6fdb2187082267dcde36`；
- 33 个数据库；
- 模型：`deepseek-v4-flash`；
- 单次采样，`K=1`；
- `bird-set`；
- strict no-repair parser；
- DeepSeek thinking + JSON Output；
- max steps 30，max tokens 2048；
- recent-4 rolling legal history；
- 两组各 12 workers，同时运行，总并发 24。

唯一成组变化是 version44 已冻结的四项接口变更：

1. 删除模型可见 `plan`；
2. `read_subtable` 重命名为 `inspect_rows`；
3. 新增 `search_values`；
4. 只有 `inspect_column` 返回 BIRD 列语义。

baseline 在本次 50 题中没有实际调用 `plan`，两组 row-inspection 调用量也接近，因此
实际轨迹中的主要新增行为是 `search_values` 和 inspect-only 语义；但由于四项变化共享
一个 prompt/schema，本实验仍不能把所有差异严格归因于单一工具。

## 总体效率

| 指标 | version39 | version44 | 变化 |
|---|---:|---:|---:|
| 总动作数 | 314 | 321 | +2.2% |
| 平均动作/题 | 6.28 | 6.42 | +0.14 |
| 总 tokens | 2,097,832 | 2,129,359 | +1.5% |
| 平均 tokens/题 | 41,956.6 | 42,587.2 | +630.6 |
| 整次墙钟时间 | 249.26s | 405.83s | +62.8% |

version44 没有 transport、context、completion 或 carrier retry。baseline 有一次 transport
retry 和一次 completion retry，因此 version44 的时间退化不能用更多 API retry 解释。

## `search_values` 覆盖和检索质量

version44 在 19/50 题中调用 `search_values`，共 22 次：

- 18 次限定 column，4 次搜索 table 的全部列；
- 15 次首项为 exact，2 次 normalized exact，2 次 prefix，1 次 fuzzy；
- 2 次没有候选；
- 4 次返回 `has_more=true`，但模型没有使用非零 offset；
- 17/22 次首项是 exact 或 normalized exact。

这说明教师理解并能调用新接口，候选值检索本身大多有效。但对使用搜索的 19 题：

| 配置 | 正确 |
|---|---:|
| version39（同一批题） | 16/19 |
| version44 | 16/19 |

其中 2 个 gain、2 个 regression，净增益为 0。未调用搜索的其余 31 题从 23/31 变为
22/31。因此当前证据是：搜索扩大了可见候选值，但尚未把候选发现稳定转化为更好的
关系程序。

### 有帮助的案例

- `bird_train_02901`：搜索揭示多个
  `Production Technician - WCxx` 实际值，教师据此使用共同前缀，同时补齐 male 条件并
  输出三个独立姓名字段，baseline 错而 version44 对。这是最清楚的 search-supported
  gain，但 gain 同时包含条件和输出形状修正，不能全部归因于检索。
- `bird_train_00593`：全列搜索先找不到完整姓名，随后用姓氏找到实际列和值。version44
  最终只投影 duration，baseline 额外保留 START/STOP。这里直接修复正确率的是输出形状，
  搜索主要提供了探索帮助。
- 单独接口 smoke task 中，typo `Avangrd Omsk` 能找到 `Avangard Omsk`，证明 fuzzy
  分支可以工作。

### 失败案例

- `bird_train_01393`：搜索同时返回 Berkeley affiliation 的多种拼写。模型把候选集合
  误解释成应该扩大语义范围，使用
  `LIKE '%University of California%Berkeley%'`，得到 574；gold 的精确文字条件为 238。
  工具返回的是候选，不代表所有近似值都应并入谓词。
- `bird_train_06171`：搜索正确返回标题的 exact/normalized variants，但模型从过滤后的
  Paper relation 中手工取一个 PaperId，再过滤 PaperAuthor，丢失其他匹配记录的作者。
  正确策略是保留过滤 relation 并整体 join，而不是把多行集合坍缩为一个观察值。

## inspect-only 列语义

version44 有 7 题实际收到 `semantic_name` 或 `column_description`：

| 配置 | 正确 |
|---|---:|
| version39（同一批题） | 7/7 |
| version44 | 6/7 |

没有 gain，出现一个 regression：`bird_train_04413`。列描述本身没有给出错误事实，但
模型额外排除了 `person IS NULL` 的获奖记录，使四行答案只剩两行。当前样本不能证明列
语义有害，但也没有发现正向证据。

## 大表延迟问题

当前 `search_values` 对目标列执行 `GROUP BY`，随后在 Python 中对每个 distinct value
运行 lexical/fuzzy 匹配。`authors.Paper` 有 2,254,920 行，`Title` 有 1,979,379 个
distinct values。两道搜索标题的 version44 题分别耗时 382.94s 和 380.36s，而对应
baseline 分别为 18.27s 和 32.66s。

结合完全相同的数据库、相近的两次约 380 秒耗时和当前 executor 实现，可以判断全列
distinct 枚举及 `SequenceMatcher` 是主要延迟来源。19 道搜索题的累计 episode 时间从
625.87s 增至 1,509.79s（+141%）。

后续实现应先用 SQLite 进行 exact、normalized exact、prefix/substring 的有界候选
生成，仅对有界候选做 fuzzy 排序；不能再对百万级 distinct 文本逐项运行
`SequenceMatcher`。优化必须保持确定性排序、20 条分页和 replay 一致性。

## 配对变化

Gains：

- `bird_train_06489`
- `bird_train_04189`
- `bird_train_00593`
- `bird_train_02901`

Regressions：

- `bird_train_06336`
- `bird_train_06171`
- `bird_train_04163`
- `bird_train_01393`
- `bird_train_04413`

非搜索变化主要来自输出列/表示选择、tie/population 处理和 NULL 语义，并非
`search_values` 的直接效果。

## Artifact

- baseline manifest：
  `data/trajectories/version44_paired_gate50_20260730/version39_baseline/verified.manifest.json`
- version44 manifest：
  `data/trajectories/version44_paired_gate50_20260730/version44_new_tools/verified.manifest.json`
- baseline all SHA-256：
  `2fbc3534cea1ca471acc3b1f8c4b118dba7dc9cb894801f2020b89f7d6d7c57a`
- version44 all SHA-256：
  `9f04049a02a3f27fa217acffb4d35efeb4e454bc3d6768ab0ee846171b10cd86`
