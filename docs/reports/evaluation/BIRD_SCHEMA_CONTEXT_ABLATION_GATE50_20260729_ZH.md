# BIRD 工具代理初始数据库信息消融：DeepSeek v4 Flash Gate20/Gate50

日期：2026-07-29  
指标：`bird-set`  
模型：DeepSeek v4 Flash，temperature=0，thinking enabled，reasoning effort=high  
协议基座：atomic version24，rolling legal history=4，JSON Output，max_steps=30

## 结论

在冻结 200 题队列的原前 20 / 前 50 题上，增加数据库信息没有超过历史 lazy
catalog 基线：

| 条件 | 前 20 | 相对基线 | 前 50 | 相对基线 |
|---|---:|---:|---:|---:|
| 1. 历史 lazy catalog（不重跑） | **16/20** | — | **42/50** | — |
| 2. 全量 raw schema | 15/20 | -1 | 40/50 | -2 |
| 3. 全量 schema + BIRD 语义名 | **16/20** | 0 | **42/50** | 0 |
| 4. 全量 schema + 语义名 + 每列最多两值 | 13/20 | -3 | 38/50 | -4 |
| 附加：lazy catalog + describe 返回语义名 | 14/20 | -2 | 40/50 | -2 |

所有新条件均为 20/20、50/50 legal termination，没有最终 API/transport failure。
条件 3 只是追平而不是提升：前 20 与基线逐题完全相同；前 50 有一个 gain 和一个
regression，净变化为 0。附加条件没有显示“保持 lazy acquisition、仅在 schema 返回时增加
语义名”能够提升能力。

因此，这组实验不支持“当前和较强相关工作的主要差距只是初始 schema 信息太少”。完整
schema 能减少 schema acquisition 动作，但没有提升正确率；BIRD 语义名抵消了 raw-schema
条件的部分退化，却没有产生净 gain；两值条件反而造成最明显下降。更可能的剩余瓶颈是
问题语义、关系/聚合决策和最终输出形状，而不是列是否在首轮可见。

## 冻结题组与条件 1

题组保持原顺序：

`data/eval_inputs/bird_train_tool_interface_validation200_version4.jsonl`

- Gate20：该文件前 20 条；
- Gate50：该文件前 50 条；
- 逐题配对使用 `trajectory_id`，不使用并发写入顺序。

条件 1 直接复用：

`data/trajectories/tool_usability_20260724/version24_fixed200_first50_bird_set.all.jsonl`

没有重新调用模型。代码审计重新构造出的 lazy 系统提示与历史轨迹首条 system message
逐字相同：长度均为 16,400 字符，protocol hash 均为 `25ac4c10ef96365c`。

## 五个互斥 context profile

1. `lazy-catalog-v1`
   - 初始 overview：表名、row count、FK；
   - 无列、语义名和值；
   - 使用历史提示原文。
2. `full-schema-v1`
   - 初始 overview：所有 raw 表/列、SQLite type、PK、FK；
   - 无语义名和值。
3. `full-schema-bird-semantic-v1`
   - 条件 2 加 BIRD `train_tables.json` 中的 `table_names` / `column_names`；
   - 字段明确命名为 `semantic_table_name` / `semantic_name`；
   - raw `table_name` / `name` 才是工具可执行标识。
4. `full-schema-bird-semantic-values2-v1`
   - 条件 3 加每列最多两个 live `example_values`；
   - 从每列前 200 个非 NULL 单元中确定性取前两个 distinct 值；
   - 长字符串截到 80 字符，提示明确这些值不是完整值域。
5. `lazy-catalog-semantic-describe-v1`
   - 初始 overview 与条件 1 完全相同；
   - `describe_table` 返回 raw 名和 BIRD 语义名；
   - 语义字段同时写入 resident schema state。

条件 2–4 没有通过“在旧 prompt 末尾追加例外”来覆盖 lazy 指令。实现会先严格替换
version24 基础提示中三处冲突：

- “opening overview 只有 catalog、没有列”；
- `describe_table` 工具说明中的 “overview 只有表和关系”；
- “先 describe needed tables” 的全局规则。

若冻结文本不能各匹配一次，运行会 fail closed。之后才加入每个 profile 独立的 context
条款。最终 protocol hash 分别为：

| 条件 | protocol hash |
|---|---|
| 1. lazy baseline | `25ac4c10ef96365c` |
| 2. full schema | `b2c9b7703db27b94` |
| 3. full schema + semantic | `b2128ca5517b8fb6` |
| 4. full schema + semantic + values2 | `8b2ca134ae685d78` |
| 附加 lazy + semantic describe | `d2c5d5e93709f4a2` |

每个有效目录的 manifest 保存完整 system prompt、system-prompt SHA-256、context contract
SHA-256、profile ID 和 schema metadata 路径。

## 配对结果

### Gate20

| 条件 | gains | regressions | net | exact two-sided p |
|---|---:|---:|---:|---:|
| 2. full schema | 1 | 2 | -1 | 1.0000 |
| 3. + semantic | 0 | 0 | 0 | 1.0000 |
| 4. + values2 | 0 | 3 | -3 | 0.2500 |
| 附加 lazy + semantic describe | 0 | 2 | -2 | 0.5000 |

### Gate50

| 条件 | gains | regressions | net | exact two-sided p |
|---|---:|---:|---:|---:|
| 2. full schema | 1 | 3 | -2 | 0.6250 |
| 3. + semantic | 1 | 1 | 0 | 1.0000 |
| 4. + values2 | 1 | 5 | -4 | 0.2188 |
| 附加 lazy + semantic describe | 1 | 3 | -2 | 0.6250 |

Gate50 相对条件 1 的具体变化：

- full schema
  - gain：`bird_train_00593`
  - regressions：`bird_train_03390`, `bird_train_04189`, `bird_train_02901`
- full schema + semantic
  - gain：`bird_train_05316`
  - regression：`bird_train_02901`
- full schema + semantic + values2
  - gain：`bird_train_05316`
  - regressions：`bird_train_06489`, `bird_train_03390`, `bird_train_04189`,
    `bird_train_02901`, `bird_train_01393`
- lazy + semantic describe
  - gain：`bird_train_05316`
  - regressions：`bird_train_06489`, `bird_train_04189`, `bird_train_02901`

样本量不足以支持统计显著的提升或下降；这里的主要证据是没有任何条件产生正的净 gain。

## 效率、上下文与过程行为

以下均按 Gate50 的全部 50 条 episode 统计，包括错误答案：

| 条件 | mean steps | describe calls | inspect calls | API requests | total tokens | 首轮 user chars |
|---|---:|---:|---:|---:|---:|---:|
| 1. lazy baseline | 5.88 | 55 | 13 | 295 | 1,557,640 | 1,920.5 |
| 2. full schema | **5.02** | **0** | 9 | 252 | 1,669,944 | 6,168.7 |
| 3. + semantic | 5.66 | 1 | 18 | 283 | 2,201,233 | 9,181.0 |
| 4. + values2 | **4.82** | 9 | 10 | **242** | 2,233,311 | 12,813.9 |
| 附加 lazy + semantic describe | 5.92 | 53 | 21 | 296 | 1,641,786 | 1,920.5 |

全量 schema 确实消除了大部分 `describe_table`，条件 2 的 requests 比基线少 14.6%，但
token 仍多 7.2% 且正确率低 4pp。条件 3 的 token 多 41.3%，但正确率完全不变。条件 4
requests 少 18.0%，却因首轮输入最宽使 token 多 43.4%，正确率低 8pp。这里展示的是
“减少交互步数”和“提升能力”两个不同目标，不能把前者当作后者。

## 审计异常与排除项

第一次启动条件 2–4 后，manifest 审计发现 version24 基础提示的 lazy-only 语句仍与末尾
full-schema 条款并存。三组轨迹被整体移动到：

`data/trajectories/schema_context_ablation_20260729/rejected_ambiguous_prompt/`

它们没有进入任何上述统计。修复加入严格 prompt 替换与单元测试后，条件 2–4 从空的正式
目录重新完整运行。附加 lazy 条件不存在该冲突，因此其原运行有效。

## 有效 artifacts

- 配对汇总：
  `data/trajectories/schema_context_ablation_20260729/paired_analysis.json`
- 条件 2：
  `data/trajectories/schema_context_ablation_20260729/full_schema/gate50.*`
- 条件 3：
  `data/trajectories/schema_context_ablation_20260729/full_schema_semantic/gate50.*`
- 条件 4：
  `data/trajectories/schema_context_ablation_20260729/full_schema_semantic_values2/gate50.*`
- 附加条件：
  `data/trajectories/schema_context_ablation_20260729/lazy_semantic_describe/gate50.*`

这些轨迹是 evaluation-only diagnostic，不是 SFT 数据源。
