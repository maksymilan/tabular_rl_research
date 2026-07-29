# Atomic 完整 BIRD 上下文 Gate32 审计

## 结论

把完整表结构、所有列、BIRD 列语义、每列两个真实示例值以及主外键一次性提供给
DeepSeek v4 Flash，并从 atomic 工具面移除 `describe_table` 和 `inspect_column` 后，本次
冻结 32 题得到 **19/32 = 59.4% `bird-set`**。

相对历史配对 atomic version38 的 **18/32**，只有 3 个增益、2 个回归，净增 1 题，
discordant pair 的 exact two-sided p 值为 **1.0**。因此这不是“正确率已经提高”的证据。
它更明确的收益是效率：模型动作从 290 降到 196，保留 episode tokens 从 3,105,935
降到 2,094,317；但合法终止从 32/32 降到 30/32，过程错误从 7 增至 11。

目前证据支持：完整上下文能消除 schema 获取成本，并能修复一部分列语义歧义；它不能
替代关系语义、答案形态和工具参数能力，也可能让模型过早选择一个看似合理的表或列。

## 实验设计

- 冻结集合：
  `data/eval_inputs/bird_train_version38_semantic_discipline_gate32_20260729.jsonl`
- 外部教师：DeepSeek v4 Flash，temperature 0，thinking enabled，reasoning effort high，
  JSON Output。
- 工具方案：atomic；动作载体 `think-json-v1`。
- 当前执行协议：version39；recent-4 rolling legal history。
- 最大动作数 30；每题一次语义尝试；可在同一 episode 内恢复协议/参数/执行错误。
- 指标：`bird-set`。
- 上下文配置：`full-bird-schema-samples-v1`。
- 每题模型可见输入包含：
  - 所有源表、行数、所有列和列类型；
  - 主键、外键与关系；
  - BIRD `database_description/*.csv` 中的 `column_name`、
    `column_description` 和 `data_format`；
  - 每列最多两个从当前 SQLite 实例读取的 distinct 非空示例值，字符串截断到 40 字符；
  - 明确说明示例值不是完整值域。
- 模型可见工具 schema 和运行时校验都移除了 `describe_table`、`inspect_column`。
  32 条轨迹中没有模型尝试调用这两个工具。
- gold SQL 不进入模型上下文，只由 harness 用于终止 denotation 校验。
- 输出强制为 diagnostic-only，不能进入 SFT。

预检时 32/32 上下文均可构建。`app_store` 的描述文件名
`googleplaystore*.csv` 与真实表名 `playstore`/`user_reviews` 不一致；renderer 通过
原始列集合做唯一确定性匹配，使该库 15/15 列均获得 BIRD 描述。

## 总体结果

| 指标 | atomic version38 历史配对控制 | 完整上下文 atomic | 变化 |
|---|---:|---:|---:|
| `bird-set` 正确 | 18/32 | 19/32 | +1 |
| 合法终止 | 32/32 | 30/32 | -2 |
| 总动作 | 290 | 196 | -94（-32.4%） |
| 平均动作/题 | 9.06 | 6.13 | -2.93 |
| 过程错误 | 7 | 11 | +4 |
| 保留 episode tokens | 3,105,935 | 2,094,317 | -1,011,618（-32.6%） |
| API request attempts | 476 | 196 | -280 |
| API/carrier 终止失败 | 0（恢复合并口径） | 0 | 0 |

当前运行没有 transport retry、completion retry 或 carrier retry。version38 的 476 次请求
来自基础设施恢复后的合并轨迹，因而 request-attempt 降幅不能完全解释为上下文设计收益；
动作数和保留 episode tokens 更可比。

基线的 schema 感知动作包括 35 次 `describe_table` 和 25 次 `inspect_column`。完整上下文
不仅移除了这 60 次动作，总动作还额外减少 34 次，说明模型也缩短了部分后续探索。

## 逐题配对

增益 3 题：

- example 6489，`image_and_language`：BIRD 描述直接说明目标对象标识列，模型按图像 ID
  和两个坐标过滤后只投影目标 ID；从 6 步缩短到 4 步。
- example 1692，`simpson_episodes`：模型在完成必要连接后，用同一 joined population
  计算总数和条件计数，避免了基线中分支计数口径漂移；从 10 步缩短到 6 步。
- example 3042，`hockey`：关系计算与基线相近，但最终保留题目要求的两个姓名字段，
  没有把它们拼成一个自定义格式列。

回归 2 题：

- example 2512，`food_inspection_2`：已找到同一目标员工，但把 first/last 两列拼成一个
  `full_name` 列；这是答案形态回归，不是 schema 缺失。
- example 2682，`works_cycles`：看到 `Product.StandardCost` 后直接对单条当前产品记录求
  平均，没有继续使用历史成本关系。完整 schema 虽然暴露了相关历史表，却没有阻止模型
  选择最短、看似合理的路径。这是“信息齐全但语义选表错误”的典型反例。

其余 16 题两边都正确，11 题两边都错误。3 gain / 2 regression 的 exact two-sided
paired p = 1.0，不支持显著能力提升。

## 过程错误

完整上下文运行共有 11 个过程错误：

- 9 个 `argument_validation_error`；
- 1 个可恢复 `protocol_error`；
- 1 个可恢复 `execution_error`。

两个 episode 未能合法终止：

- example 2438，`books`：模型先把 derived join handle 当作逻辑列 namespace，恢复后又
  产生非法 `read_subtable.limit` 和 dotted `right` join column。完整源 schema无法解决
  派生关系 namespace 的工具协议问题。
- example 2088，`address`：模型把 `read_subtable` 当成支持空 conditions 或
  `distinct` 的关系变换工具，连续三次被拒绝。这说明移除 `inspect_column` 后，
  `read_subtable` 承担了更多值域探索，但 observation-only 边界仍会被误用。

另有一个成功 episode 在 join namespace 上两次出错后恢复。没有错误是对
`describe_table` 或 `inspect_column` 的调用，因此“工具实际上没有从接口移除”不是问题。

## 轨迹验证

- 19 个正确 episode 的独立结构审计：19/19 pass。
- profile-aware deterministic replay：19/19 pass。
- 以 `max_steps=30`、`max_think_words=300` 运行当前质量门：
  12/19 pass，7 条仅因 `think_limit` 被拒绝。

冻结运行最初把模型可见的完整数据库上下文写入了
`initial_state.dataset_overview`，而通用 replay 将该字段解释为 canonical lazy execution
catalog。审计时按 profile 重新构建普通 execution catalog 后，未修改任何模型动作，
19/19 均可重放。runner 已修正：后续轨迹在 canonical `initial_state` 中保存 execution
catalog，并单独记录模型可见完整上下文的 SHA-256。当前冻结产物仍仅作诊断，不能作为
训练样本。

## 因果解释与限制

当前运行使用工作区里的 version39，而历史配对控制是 version38。version39 只增加
model-visible resident-state compaction，公开工具与有效调用执行语义不变；但这仍不是
同时段、同协议的严格单变量 A/B。因此净 +1 更不能被解释为完整上下文的确定因果收益。
若要精确估计因果效应，需要再跑一个同一 version39、同一 provider 时段的
`catalog-v1` 控制。

即使暂不补跑控制，本次结果也足以否定“完整上下文会带来明显正确率提升”的强假设：
收益主要落在减少 schema 探索、降低动作和 token，而 13 个失败中大部分仍是合法关系
语义或输出形态错误。

## 建议

1. 不把该配置替换为默认 atomic，也不据此构造 SFT。
2. 如果继续消融，优先测试“完整 schema + BIRD 描述 + 保留 `inspect_column`、移除
   `describe_table`”。完整 schema可以替代表结构查询，但两个示例值不能替代完整值域
   检查；这可能减少 `read_subtable` 被当作 distinct/value-domain 工具的误用。
3. 对派生 namespace、observation-only 读工具和最终输出槽位继续使用统一协议约束或
   工具设计改进；这些问题与源 schema 是否完整无关。
4. 若目标是训练，先解决 7 条 `think_limit` 质量门失败，并使用修正后的 canonical
   initial-state 记录格式重新生成，不能迁移当前诊断轨迹进入 SFT。

## 产物

- 运行 manifest：
  `data/trajectories/bird_train_atomic_full_bird_context_gate32_20260729/verified_success.manifest.json`
- 全部 32 条审计记录：
  `data/trajectories/bird_train_atomic_full_bird_context_gate32_20260729/verified_success.all.jsonl`
- 19 条 verifier-correct 诊断轨迹：
  `data/trajectories/bird_train_atomic_full_bird_context_gate32_20260729/verified_success.jsonl`
- 结构审计：
  `data/trajectories/bird_train_atomic_full_bird_context_gate32_20260729/structural_audit.json`
- profile-aware replay/质量摘要：
  `data/trajectories/bird_train_atomic_full_bird_context_gate32_20260729/profile_aware_replay_audit.json`
