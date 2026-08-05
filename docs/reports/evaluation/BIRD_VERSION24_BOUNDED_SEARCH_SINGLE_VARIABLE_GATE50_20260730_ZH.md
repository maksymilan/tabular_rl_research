# BIRD version24 + bounded search_values 单变量 Gate50

日期：2026-07-30

## 结论

先从历史提交 `22196e2` 重建原始 atomic version24，再在完全相同的代码基线上只增加
`search_values(table, query, column?, limit?, offset?)` 及其必要的 parser、只读 executor、
resident observation 和 grounding plumbing。

结果没有发现准确率收益：

| 配置 | bird-set | Legal | Process errors |
|---|---:|---:|---:|
| 历史 version24 artifact | 42/50（84%） | 50/50 | 3 |
| 新鲜 version24 复现 | 41/50（82%） | 50/50 | 1 |
| version24 + bounded search | 39/50（78%） | 50/50 | 4 |

新鲜 version24 相对历史 artifact 有 40 题都对、7 题都错、1 gain、2 regressions，
47/50 结果一致，双侧 exact McNemar `p=1.0`。虽然没有逐字复现 42/50，但 41/50
属于同一统计行为，用户确认可作为复现成功并继续单变量实验。

搜索组相对同轮新鲜 baseline 有 37 题都对、7 题都错、2 gains、4 regressions，净
-2 题（-4pp），双侧 exact McNemar `p=0.6875`。相对原历史 42/50 则是 0 gain、
3 regressions，`p=0.25`。差异不显著，但没有任何证据支持 promotion；不扩展剩余
150 题，也不把该接口直接用于新的 SFT/RL 协议。

## 严格控制

### Baseline 重建

- 历史代码提交：`22196e2`；
- 隔离分支：`codex/version24-search-ablation`；
- 搜索单变量实现提交：`deaa0e4`；
- 原始 protocol hash：`25ac4c10ef96365c`；
- 原始 provider system prompt 长度：16,400 字符；
- 原工具保持 `plan`、`read_subtable` 和 exact-table terminal；
- 没有 BIRD column description/semantic name；
- 没有 version37 之后的 typed row read/date expression；
- 没有 version40 之后的 no-plan、`inspect_rows` 或 terminal-columns 改动。

新鲜 baseline 的 prompt hash 和长度与 2026-07-24 artifact 完全一致。

### 唯一公开接口变化

搜索分支公开工具从 12 个变为 13 个，只新增：

```text
search_values(table, query, column?, limit?, offset?)
```

所有原工具名称、参数和执行语义不变。搜索：

- `table`、`query` 必填；
- `column` 选填，省略时搜索该 table 全列；
- 每页最多 20 条，`offset` 稳定分页；
- exact/case-insensitive exact 优先，并抑制更远 fuzzy alternatives；
- 没有 exact 时，以稳定 token/trigram anchors 在 SQLite 召回每列最多 4096 个候选，
  再运行确定性 lexical matcher；
- 只读，不创建 relation，不改变数据库；
- observation 进入 resident state，最多保留每表最近两次；
- 后续精确 literal 可形成 harness-owned `domain_observation` grounding edge。

新增 prompt 内容只有工具签名/边界、一个 canonical call 和一条使用规则。搜索组：

- protocol version：`version24-search-values-v1`；
- protocol hash：`dd8723215fb988a4`；
- provider prompt 长度：17,477 字符，比 baseline 多 1,077 字符。

这次没有删除 `plan`、没有把 `read_subtable` 改名、没有返回列语义，因此避免了
version44/45 中四项接口同时变化的混杂。

### 共同评测设置

- 题集：
  `data/eval_inputs/bird_train_tool_interface_validation200_version4.jsonl` 前 50 题；
- 题集 SHA-256：
  `6b469215ac96e09c1046fee539ff4c4885b58e9b7a0b6fdb2187082267dcde36`；
- 模型：`deepseek-v4-flash`；
- 单次采样，K=1；
- `bird-set`；
- strict no-repair parser；
- DeepSeek thinking + JSON Output；
- rolling legal history，recent-4；
- canonical full prompt；
- optional plan；
- max steps 30，max tokens 2048；
- attempts per example 1；
- 12 workers。

两次新鲜运行均为 50/50 合法终止，0 transport/context/carrier/completion retry。

## 搜索覆盖与质量

模型在 13/50 题中调用搜索，共 16 次：

- 10 次限定 column，6 次搜索 table 全列；
- 首候选：12 exact、1 prefix、3 empty；
- 0 次使用非零 offset；
- 0 次 `candidate_truncated=true`；
- 最大候选池 178，平均候选数 24.125。

实际使用搜索的 13 题：

| 配置 | 正确 |
|---|---:|
| 新鲜 version24 baseline | 11/13 |
| version24 + search | 10/13 |

该集合是模型运行后选择出来的 post-treatment subset，不能当作独立因果估计；但其中
0 gain、1 regression，至少说明本轮没有观察到搜索支持的准确率恢复。

唯一使用搜索且发生 paired regression 的任务是 `bird_train_05316`：

- 问题要求最高评分的 chicken restaurant；
- `search_values(generalinfo, "chicken", column="food_type")` 正确返回 exact stored
  value `chicken`，frequency 50；
- filter 和 top-1 都正确；
- 模型最终保留 `label, review` 两列，而 gold 只要求 restaurant label。

因此失分不是候选召回错误，而是精确终表输出错误。

## 配对变化

### 两个 gain

- `bird_train_02408`：补做最终投影，只返回 author name，不再附带 author id；
- `bird_train_06454`：在两个 ship-mode count 后补做 top-1，只返回 RAIL。

两题都没有调用 `search_values`，属于 K=1 策略波动，不能记为搜索收益。

### 四个 regression

- `bird_train_03390`：返回 country code，没有 join country name；
- `bird_train_04189`：使用 left join，保留额外无匹配 review 行；
- `bird_train_02901`：把三个姓名字段拼成一个字符串；
- `bird_train_05316`：搜索正确，但终表保留额外 ranking helper `review`。

前三题没有调用搜索。四个回退都不是候选召回漏值；总体负差主要体现 prompt/action-space
改变后的策略采样，而非 bounded search executor 的检索失败。

## 成本

| 指标 | 新鲜 version24 | version24 + search | 变化 |
|---|---:|---:|---:|
| 总动作数 | 286 | 313 | +9.4% |
| 平均动作/题 | 5.72 | 6.26 | +0.54 |
| 总 tokens | 1,512,070 | 1,727,622 | +14.3% |
| API request attempts | 286 | 313 | +9.4% |
| 逐题 elapsed 求和 | 2,553.36s | 2,822.51s | +10.5% |
| 整次墙钟时间 | 242.37s | 283.38s | +16.9% |
| Process errors | 1 | 4 | +3 |

搜索组的 16 次新工具调用只解释了部分动作增长；模型还增加了 describe/filter/aggregate
等探索动作。`inspect_column` 从 13 次降到 9 次，说明搜索部分替代了旧列域观察，但没有
减少总交互成本。

## 判断

1. **原最佳版本可以统计复现。** 41/50 对历史 42/50，47/50 逐题一致，没有基础设施
   故障或协议漂移。
2. **单独增加 search_values 没有涨点。** 同轮对照为 39/50 对 41/50，2 gains、
   4 regressions，且实际搜索题没有 gain。
3. **检索功能本身工作正常。** exact/prefix 候选正确、无候选池饱和、无搜索执行错误；
   当前瓶颈仍是发现值之后的关系程序和精确输出。
4. **不应扩量。** 当前工具增加动作、tokens 和延迟，却没有形成可测准确率收益。
5. 如果继续研究 search，应改用预先冻结的 value-uncertain hard cohort，并将“找到 gold
   literal 后是否构造出正确 predicate/terminal table”作为过程指标，而不是继续在高分
   随机 50 题上做全量 K=1。

## Artifact

- 历史 baseline：
  `data/trajectories/tool_usability_20260724/version24_fixed200_first50_bird_set.all.jsonl`
- 新鲜 version24：
  `data/trajectories/version24_search_single_variable_20260730/version24_repro_network_retry/verified.all.jsonl`
- 搜索单变量组：
  `data/trajectories/version24_search_single_variable_20260730/version24_plus_bounded_search/verified.all.jsonl`
- 搜索组 manifest：
  `data/trajectories/version24_search_single_variable_20260730/version24_plus_bounded_search/verified.manifest.json`
