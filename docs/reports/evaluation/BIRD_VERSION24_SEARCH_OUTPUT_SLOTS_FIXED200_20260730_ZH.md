# BIRD version24 + bounded search + 精确输出字段：Fixed200

日期：2026-07-30

## 结论

按用户要求，将已经完成的前 50 题续跑到同一冻结 200 题 cohort。前 50 不重跑，
`--resume` 按 `trajectory_id` 跳过已完成记录，只执行剩余 150 题。

最终结果：

| 配置 | bird-set | Legal | Wrong answer | 非语义终止 |
|---|---:|---:|---:|---:|
| 历史原始 version24 | **145/200（72.5%）** | 197/200 | 52 | 3 |
| version24 + search + 输出字段约束 | **141/200（70.5%）** | 195/200 | 54 | 5 |

新方案前 50 为 41/50，剩余 150 为 100/150；前 50 的 82% 明显高估了完整 cohort
表现。

相对历史 version24，逐题配对为：

- 129 题都正确；
- 43 题都错误；
- 12 gains；
- 16 regressions；
- 净 -4 题（-2pp）；
- 双侧 exact McNemar `p=0.571588`。

差异不显著，不能证明新方案必然降低能力；但完整 200 题没有任何总体增益证据，且点估计、
合法率和成本均不优于历史 version24。因此不 promotion，不用于新的 SFT/RL 协议。

## 实验边界

### 新方案

- 分支：`codex/version24-search-output-slots`
- 代码提交：`cb159cc`
- protocol version：`version24-search-values-output-slots-v1`
- 实际 provider protocol hash：`761946c155142c5d`
- provider system prompt：18,358 字符
- 模型：`deepseek-v4-flash`
- 单次采样，K=1
- `bird-set`
- strict no-repair parser
- thinking + JSON Output
- rolling legal history，recent-4
- canonical full prompt
- optional plan
- max steps 30，max tokens 2048
- 12 workers

它相对历史 version24 同时包含两个模型可见变化：

1. 新增 bounded `search_values`；
2. 新增答案字段 / helper 字段判别规则和一对 top-1 输出示例。

完整 200 题没有 search-only 对照，因为 search-only 按原 Gate50 失败结论停止在前 50。
所以 200 题只能比较“search + 输出 prompt”整体与历史 version24，不能在后 150 题中
单独估计输出 prompt 的因果效果。

### 冻结 cohort

`data/eval_inputs/bird_train_tool_interface_validation200_version4.jsonl`

SHA-256：

`6b469215ac96e09c1046fee539ff4c4885b58e9b7a0b6fdb2187082267dcde36`

历史 version24 的前 50 与后 150 artifact 使用同一 cohort、模型、K=1、`bird-set`、
recent-4、optional plan、max steps 30、max tokens 2048 和 12 workers。

## Prompt 目标是否奏效

有五个相对历史 version24 的 gain 直接符合新增输出约束：

| 任务 | 历史输出 | 新输出 |
|---|---|---|
| `bird_train_05316` | restaurant label + review | 只保留 label |
| `bird_train_03969` | gender + age | 只保留 gender |
| `bird_train_03042` | 拼接的 full name | first name、last name 分列 |
| `bird_train_02512` | 拼接的 full name | first name、last name 分列 |
| `bird_train_03724` | film title + rental count | 只保留 title |

这说明“问题明确要求的答案字段”和“只用于排名/计算的 helper 字段”对照能被模型理解，
不是完全无效的提示。

但也出现至少四个输出槽/表示相关 regression：

- `bird_train_06489`：benchmark 要求 `OBJ_SAMPLE_ID=18`，模型改答自然语义标签
  `"paper"`；
- `bird_train_03191`：要求 ranking system name，却额外保留 ranking-system id；
- `bird_train_03131`：要求 language 与 official flag，却额外保留 country code；
- `bird_train_00600`：要求一行两列的 male/female counts，模型返回两行
  `gender,count`。

因此 prompt 强化了“主动选择答案槽”，却不能可靠判定自然语言槽究竟对应 ID、label、
多列字段还是 reshape 后的列。直接输出形状修复和新输出形状回退大致抵消；其余
gain/regression 主要来自关系路径、人口口径、聚合或随机策略分支变化。

## Search 使用情况

在完整 200 题中：

- 50/200 题调用过 `search_values`；
- 共 78 次 search 调用；
- 这 50 题新方案为 39/50；
- 历史 version24 在同一批题上为 35/50；
- 配对为 4 gains、0 regressions。

但“是否调用 search”是新策略运行后才产生的 post-treatment 分组，不能据此宣称
search 因果提升 4 题。对应的未调用 search 的 150 题，新方案为 102/150，历史
version24 为 110/150；总体净损失来自该组的更多策略回退。合理解释是 search 在部分
value-uncertain 题中确有用，但新增 action/prompt 同时改变了其他题的采样策略；当前
全局接口没有把局部收益转化为总体收益。

## 失败与合法性

新方案 59 个失败：

- 54 `wrong_answer`；
- 3 `argument_validation_error`；
- 1 `execution_error`；
- 1 `max_steps`。

195/200 合法终止。历史 version24 为 197/200。

新方案五个非语义终止中，四个没有调用 search；最后的
`bird_train_06299` 调用了 search，并在多个 Menu/MenuPage ID 搜索之间循环到 30 步。
这些失败不是 API transport failure。

全 200 题共记录 29 次工具/process error，与历史 version24 的 29 次相同；区别在于
新方案有更多题未能从错误中恢复到合法终答。

## 成本

| 指标 | 历史 version24 | 新方案 | 变化 |
|---|---:|---:|---:|
| 总动作数 | 1,490 | 1,505 | +1.0% |
| 平均动作/题 | 7.45 | 7.525 | +0.075 |
| 总 tokens | 8,420,861 | 9,204,430 | +9.3% |
| API request attempts | 1,509 | 1,516 | +0.5% |
| 两段运行墙钟合计 | 1,087.95s | 1,546.70s | +42.2% |
| Process errors | 29 | 29 | 0 |

墙钟时间容易受 provider 当时负载影响，不能全部归因于 prompt；tokens 增长更直接地
反映新增工具说明、输出规则以及由此变化的轨迹长度/上下文。

## 判断

1. 前 50 的 41/50 不能代表完整 200；扩展后是 141/200。
2. 输出字段规则确实修复了一组 helper-column 和字段拼接错误，但也引入答案槽解释回退。
3. `search_values` 在模型主动使用的题上表现有希望，但该分组是 post-treatment，
   不能作为独立因果证据。
4. 新方案相对历史 version24 是 12 gains、16 regressions，`p=0.5716`，没有显著差异。
5. 新方案准确率低 2pp、合法率低 1pp、tokens 高 9.3%，所以不 promotion。
6. 后续若继续验证 search，应冻结训练前定义的 value-uncertain cohort，并同时运行
   no-search / search 两个完整配对组；不要再用“实际调用 search 的题”作主要因果指标。

## Artifacts

- 新方案完整 200：
  `data/trajectories/version24_search_single_variable_20260730/version24_search_output_prompt_gate50/verified.all.jsonl`
- 完整 200 artifact SHA-256：
  `b01a0a5c262bcd67d3e2b652562d4baa70eb9f5bbd715649c59c363b7ef33fb1`
- 新方案 manifest：
  `data/trajectories/version24_search_single_variable_20260730/version24_search_output_prompt_gate50/verified.manifest.json`
- 历史 version24 前 50：
  `data/trajectories/tool_usability_20260724/version24_fixed200_first50_bird_set.all.jsonl`
- 历史 version24 后 150：
  `data/trajectories/tool_usability_20260724/version24_fixed200_remaining150_bird_set.all.jsonl`
