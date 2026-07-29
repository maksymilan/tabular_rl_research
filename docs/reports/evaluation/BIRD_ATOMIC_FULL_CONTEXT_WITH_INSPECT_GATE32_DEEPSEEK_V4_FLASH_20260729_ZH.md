# Atomic 完整 BIRD 上下文并保留 inspect_column Gate32 审计

## 结论

按照“完整 schema + BIRD 列描述 + 每列两个真实示例值 + 保留
`inspect_column`，只移除 `describe_table`”运行同一冻结 32 题后，DeepSeek v4 Flash
得到 **17/32 = 53.1% `bird-set`**。

相对同一 version39 的“完整上下文且同时移除 `describe_table`/`inspect_column`”
配置，正确率从 **19/32 降到 17/32**：2 个配对增益、4 个配对回归，净 -2；
discordant pair 的 exact two-sided p 值为 **0.6875**。过程接口则明显更稳定：
合法终止从 30/32 提升到 31/32，过程错误从 11 个降到 5 个；代价是动作从
196 增到 216，tokens 从 2,094,317 增到 2,251,910。

因此，保留 `inspect_column` 的确减少了“缺乏值域观察手段”造成的接口摩擦，但没有把
这种收益转化为更高的题目正确率。当前证据不支持把它提升为完整上下文 atomic 的默认
方案，也不能据此构造 SFT 数据。

## 实验设计

- 冻结集合：
  `data/eval_inputs/bird_train_version38_semantic_discipline_gate32_20260729.jsonl`
- 集合 SHA-256：
  `0993665a2df7bc6447ee1b558119fa41a674251262b90c4130f5f22bc9a79cb8`
- 外部教师：DeepSeek v4 Flash，temperature 0，thinking enabled，
  reasoning effort high，JSON Output。
- 工具方案：atomic version39；载体 `think-json-v1`；recent-4 rolling legal history。
- 最大动作数 30；每题一次语义尝试；指标 `bird-set`。
- 上下文配置：`full-bird-schema-samples-with-inspect-v1`。
- 完整上下文与前一配置逐字节相同：全部源表、行数、列名、类型、主外键、BIRD
  短语义，以及每列最多两个 distinct 非空真实示例值。
- 模型可见工具与运行时只移除 `describe_table`；`inspect_column` 的签名、提示和执行
  均保留。公开工具 schema 哈希为
  `b104763dd2f5faf82d43ba00625e9899604617741b16ab0ff14610d0b4f95b3b`。
- gold SQL 不进入模型上下文，仅由 harness 做终止 denotation 校验。
- 强制 `diagnostic_only_pending_protocol_scale_gate`，不可进入 SFT。

## 三方总体比较

| 指标 | 历史 atomic v38 | 完整上下文、无 inspect | 完整上下文、保留 inspect |
|---|---:|---:|---:|
| `bird-set` 正确 | 18/32 | **19/32** | 17/32 |
| 合法终止 | **32/32** | 30/32 | 31/32 |
| 总动作 | 290 | **196** | 216 |
| 平均动作/题 | 9.06 | **6.13** | 6.75 |
| 过程错误 | 7 | 11 | **5** |
| 总 tokens | 3,105,935 | **2,094,317** | 2,251,910 |
| API request attempts | 476 | **196** | 218 |

“保留 inspect”相对“无 inspect”的直接变化：

- 正确率 -2 题；
- 合法终止 +1 题；
- 过程错误 -6 个；
- 动作 +20（+10.2%）；
- tokens +157,593（+7.5%）；
- API 请求 +22。

历史 v38 来自 version38 和基础设施恢复后合并轨迹，不能视为严格单变量控制。两种完整
上下文配置均为 version39、同一冻结集合和相同参数，是本报告更重要的配对比较；但两次
API 运行仍为先后执行，无法完全排除 provider 的残余非确定性。

## inspect_column 的实际使用

32 题共调用 `inspect_column` 11 次，分布在 9 题，其中 5 题正确。调用对象包括源表值域，
也包括派生表中的列；没有对已移除的 `describe_table` 的调用。

这个使用率说明模型没有普遍重复检查完整上下文已提供的 schema，而是大致按提示把
inspect 用于值域或存储格式不确定处。不过，“使用 inspect 的题目 5/9 正确”不是
inspect 的因果成功率；题目难度和模型是否决定检查存在选择偏差。

## 与“无 inspect”配置的逐题配对

### 增益 2 题

- example 2088，`address`：这是最明确的 inspect 直接收益。无 inspect 轨迹先后把
  `read_subtable` 当成支持空 conditions 和 `distinct` 的关系工具，连续触发参数错误并
  非法终止；新轨迹分别检查 area code 与地址类型值域，然后正常过滤、连接和条件计数，
  8 步零错误完成。它验证了 `inspect_column` 能承担 observation-only 的值域查询，
  避免把 `read_subtable` 错当 distinct 接口。
- example 1050，`regional_sales`：新轨迹检查了派生订单中的价格、成本存储形式，并按
  题目给定公式直接计算单价减成本；旧轨迹自行乘入订单数量，偏离外部知识。这里 inspect
  提供了数值文本格式证据，但真正的修复是新轨迹更严格地遵循题目公式，不能全部归功于
  inspect。新轨迹有一次分页参数错误，随后恢复成功。

### 回归 4 题

- example 1692，`simpson_episodes`：这是唯一实际使用 inspect 的回归。模型正确确认
  `Award.result` 的值并筛选 nominee，但发现 inner join 有未匹配记录后改用 left join，
  把未能连接到人物信息的记录也放入百分比分母，改变了评分所要求的 joined population。
  这是观察反馈后的语义决策错误，不是 inspect 接口错误。
- example 2408，`books`：没有调用 inspect。模型返回 `author_id + author_name`，而前一
  轨迹只返回题目要求的姓名列；这是精确输出形态回归。
- example 3042，`hockey`：没有调用 inspect。模型直接在单赛季记录上排名，漏掉按 goalie
  汇总整个历史的步骤；同时把 first/last 两列拼成自定义 `full_name`。选中的人物碰巧
  没变，但表形态与推理路径都不稳健。
- example 4189，`app_store`：没有调用 inspect。模型用 left join 保留没有 review 的
  free sports app，生成额外 NULL review 行；前一轨迹使用 inner join，只保留实际存在
  translated review 的配对。

4 个回归中只有 1 个直接经过 inspect，另外 3 个是在相同输入信息下出现的策略分岔。
这意味着新增工具面不仅增加了一条可调用能力，也改变了模型在后续关系语义和终止表形态
上的采样路径；不能把净 -2 简化为“inspect 给了错误数据”。

## 过程错误与合法性

本次共有 5 个过程错误，全部是 `argument_validation_error`：

- example 2438，`books`：3 个错误并最终非法终止，分别为
  `read_subtable.limit=21`、派生 join 表使用裸列名排序、以及把新 join 右侧列写成
  dotted identifier。它与上一配置一样暴露派生关系 namespace 和工具参数边界问题；
  `inspect_column` 不能修复这类协议理解。
- example 1050，`regional_sales`：正 offset 未提供 deterministic `order_by`，随后恢复。
- example 5440，`authors`：`read_subtable.limit=26` 超出 20，上下文保持后继续并合法终止，
  但答案仍错。

前一“无 inspect”配置的 11 个错误包括 9 个参数校验、1 个协议和 1 个执行错误。本次
完全没有 provider carrier、transport 或执行错误，且只有一题非法终止，说明保留 inspect
确实降低了接口摩擦。

## 轨迹验证

- 17 条 verifier-correct 轨迹的独立结构审计：17/17 pass。
- profile-aware deterministic replay：17/17 pass；没有修改模型动作。
- 16 条 clean success，1 条 recovered success。
- 以 `max_steps=30`、`max_think_words=300` 运行质量门：14/17 pass；
  3 条仅因 `think_limit` 被拒绝。
- 总用量：prompt 2,178,672 tokens，completion 73,238 tokens，其中 reasoning
  63,957；218 次 API 请求，2 次 completion length retry，0 次 transport/carrier retry。

## 判断与下一步

这次消融回答了两个不同问题：

1. **是否降低接口摩擦？是。** 错误 11→5，合法终止 30→31，尤其修复了
   `address` 中把 `read_subtable` 当值域工具的失败。
2. **是否提高题目能力？没有。** 正确率 19→17，且新增错误主要是合法但语义或输出形态
   错误，无法通过继续规范 inspect 参数来修复。

因此建议保持两种配置都为诊断方案，不提升、不生成 SFT。如果继续研究，最有信息量的
方向不是再扩写通用 prompt，而是做一个更小的冻结配对集，专门覆盖“示例值不足、必须
确认完整值域”的题目，检验 inspect 的局部因果收益；在通用题集上，当前“完整上下文且
无 inspect”仍然是三者中正确率和效率最好的配置。

## 产物

- 运行 manifest：
  `data/trajectories/bird_train_atomic_full_bird_context_with_inspect_gate32_20260729/verified_success.manifest.json`
- 全部 32 条审计记录：
  `data/trajectories/bird_train_atomic_full_bird_context_with_inspect_gate32_20260729/verified_success.all.jsonl`
- 17 条 verifier-correct 诊断轨迹：
  `data/trajectories/bird_train_atomic_full_bird_context_with_inspect_gate32_20260729/verified_success.jsonl`
- 结构审计：
  `data/trajectories/bird_train_atomic_full_bird_context_with_inspect_gate32_20260729/structural_audit.json`
- replay/质量摘要：
  `data/trajectories/bird_train_atomic_full_bird_context_with_inspect_gate32_20260729/replay_quality_audit.json`
