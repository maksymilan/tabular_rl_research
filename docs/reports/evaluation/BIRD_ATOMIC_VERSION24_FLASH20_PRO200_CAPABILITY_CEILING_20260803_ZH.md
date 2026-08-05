# BIRD atomic version24 Flash20 复现与条件式 Pro200 能力上限实验

日期：2026-08-03
状态：**已完成**；Flash20 复现通过，Pro200 已完成并通过全部审计
用途：diagnostic-only；所有轨迹均不得进入 SFT/RL 训练集

## 问题与版本选择

本实验回答：在保持工具、prompt、上下文、反馈、动作预算、任务和 verifier 不变时，
把外部教师从 DeepSeek v4 Flash 换成 DeepSeek v4 Pro，是否提高 atomic 工具调用的
可达正确率。

按“已完成、同一冻结 200 题、`bird-set`、K=1”的可比证据选择 atomic `version24`：

| 工具方案 | 模型 | 冻结 200 题 | Legal | 过程错误 | 状态 |
|---|---|---:|---:|---:|---|
| atomic version24 | DeepSeek v4 Flash | **145/200** | 197/200 | 29 | 已完成，未过 150/200 晋级线 |
| action-block v32 | DeepSeek v4 Flash | 144/200 | 197/200 | 112 | 已完成，未晋级 |

action-block v18 在一个 20 题 pilot 上为 17/20，高于同 cohort 的 version24 16/20，
但没有完成同协议 200 题；其后继 v32 在固定 200 题为 144/200。因此本实验把
version24 作为当前“有完整 200 题证据的最佳工具版本”，不把小样本最高分误当作
总体最佳。

## 冻结实现与共同配置

- 历史代码：`22196e247e96ea2d7a9cbf84e94328b6462725c1`；
- 隔离 worktree：`/private/tmp/tabular-v24-ceiling-20260803`；
- protocol hash：`25ac4c10ef96365c`；
- provider system prompt：16,400 字符；
- 工具：version24 的 12 个 atomic 工具；
- carrier：DeepSeek native reasoning + JSON Output；
- decoding：temperature 0、thinking enabled、reasoning effort high；
- 上下文：`rolling-legal-history`、recent-4、full rolling prompt；
- policy prompt：canonical；plan：optional；
- 每题一次语义轨迹，K=1；
- max actions / max tokens：30 / 2048；
- terminal metric：`bird-set`；
- strict parser，不做 action/argument repair；
- gold SQL、gold rows 和隐藏 verifier 状态只留在本地 harness；
- 发给 provider 的内容只能是自然语言问题、external knowledge、数据库 catalog/schema、
  当前 resident state、最近合法历史和因果产生的只读工具 observation/error。

Flash 和 Pro 的模型可见 prompt hash 与长度已经离线验证完全相同。Pro200 的唯一预期
语义变量是 `model: deepseek-v4-flash -> deepseek-v4-pro`。传输重试只属于同一请求的
基础设施恢复，不形成新语义 action；实际使用次数必须单独报告。

## 阶段 A：Flash20 复现门

20 题使用既有 action-block v18 pilot 的固定 task-id 集合，并保留其既定顺序。该集合
在历史 atomic version24 结果中的锚点是 **16/20 correct、20/20 legal、1 个过程错误**。

- 输入：`data/eval_inputs/bird_train_version24_capability_ceiling_pilot20_20260803.jsonl`；
- 来源 200 题 SHA-256：
  `6b469215ac96e09c1046fee539ff4c4885b58e9b7a0b6fdb2187082267dcde36`；
- task-id 顺序 SHA-256：
  `cbbe5606b70689dac3c7d721936351ec8e8e2e5116e605438e2bea5c30290273`；
- 20 题文件 SHA-256：
  `d180ef3380ca49b90497dbe7d58d59c755e6fa8e0ff84dffdce0905c8776d0b8`。

在看到新结果前预注册“复现成功”为同时满足：

1. protocol hash 仍为 `25ac4c10ef96365c`；
2. `bird-set` correct 至少 **15/20**（不低于历史 16/20 超过 1 题）；
3. legal 至少 **19/20**；
4. 过程错误不超过 **4**；
5. 没有未恢复的 API/transport/context/carrier 基础设施失败；
6. 所有正确轨迹通过独立结构、fresh replay 和 gold-no-leak 审计。

任何一条未满足都判为 Flash20 复现失败，并停止，不启动 Pro200。若运行中出现五个连续
语义失败、五个连续 provider/carrier 缺陷，或 protocol/cohort hash 漂移，也立即停止。

## 阶段 B：仅在 Flash20 通过后运行 Pro200

Pro 使用完整冻结文件：
`data/eval_inputs/bird_train_tool_interface_validation200_version4.jsonl`。

先按原始任务顺序单 worker 跑前 5 题，执行连续失败/基础设施停止检查；通过后再用完全
相同输出目录的 resume 机制完成剩余 195 题。前 5 题不是额外尝试，每题仍只有一个
语义轨迹。

主要比较是 Pro200 对历史同 cohort、同 protocol 的 Flash200（145/200）。逐 task-id
报告：both correct、both wrong、Pro-only gain、Flash-only regression，以及 exact paired
binomial/McNemar p 值。由于 Flash 基线来自历史运行，还必须同时报告时间/provider 漂移
这一限制，不能把非显著差异解释成模型因果效应。

支持“模型能力提高工具调用上限”需要同时满足：

1. Pro 至少达到原 tool-usability 线 **150/200**；
2. Pro-only gains 多于 Flash-only regressions；
3. exact paired 双侧 `p < 0.05`；
4. legal 不低于 **197/200**，且没有 unresolved provider failure；
5. 正确轨迹全部通过独立审计。

若只满足部分条件，结论写为“有方向性信号但不足以证明”。若 Pro 未超过 145/200 或
paired gains 不多于 regressions，结论写为“当前证据不支持更强模型提高该工具协议的
能力上限”。

## 必报指标

- correct、legal、clean success、recovered success；
- process error 总数、类型和涉及任务数；
- provider/transport/context/carrier/completion retry 与最终失败；
- 总 action、平均 action、各工具调用分布；
- prompt/completion/total tokens、API request、墙钟时间；
- Flash/Pro 逐题配对与 exact p 值；
- 独立 audit 和 fresh replay 通过数；
- `sft_export_eligible=false`、`training_admission=diagnostic_only`。

不得读取成功轨迹内容来选择、编辑或“修好”某条动作；只允许用聚合统计、失败轨迹和
harness 反馈做诊断。本报告的门槛在任何外部请求之前冻结。

## 执行记录

### Flash20：复现通过

冻结 version24 + DeepSeek v4 Flash 得到：

| 指标 | 历史锚点 | 本次复现 |
|---|---:|---:|
| correct | 16/20 | **16/20** |
| legal | 20/20 | **20/20** |
| 过程错误 | 1 | **1** |
| clean / recovered success | 未单列 | **15 / 1** |
| unresolved provider failure | 0 | **0** |
| total tokens | 690,165 | **752,907** |
| semantic actions | 127 | **131** |

本次有 3 次 completion-length retry，均在同一请求/action 边界内恢复；transport、context、
carrier 最终失败均为 0。protocol hash 保持 `25ac4c10ef96365c`。16 条正确轨迹全部通过：

- structural audit：16/16；
- fresh `bird-set` replay：16/16；
- provider payload boundary audit：131 个请求 turn 中 gold-key 和 gold-SQL-text 命中均为 0。

因此 Flash20 满足全部预注册扩展条件，合法启动 Pro200。

### Pro200：完成结果

余额恢复后，实验没有为中断任务另起第二条语义轨迹。恢复器先重放每条已接受的因果前缀，
并要求重建的下一次 provider input 与原失败请求逐字节一致；58/58 均通过后才继续请求。
其中 14 条保留已有合法前缀，44 条在第一个语义 action 前中断，另外 67 条此前尚未开始。
最终 200 个 task-id 全覆盖、无重复，每题 `attempt_index=1`、`attempts_per_example=1`。

| 指标 | DeepSeek v4 Pro |
|---|---:|
| correct | **145/200（72.5%）** |
| 95% Wilson CI | 65.9%–78.2% |
| legal | **199/200（99.5%）** |
| clean / recovered success | **137 / 8** |
| wrong answer | 54 |
| 非 legal 最终失败 | 1 `argument_validation_error` |
| 过程错误 | **22**，涉及 16 题 |
| semantic actions | **1,454**，均值 7.27，中位数 7，P90 11，范围 3–27 |
| unresolved provider failure | **0** |

### 与冻结 Flash200 的逐题配对

主要比较使用历史冻结 Flash200；它与 Pro200 具有完全相同的 200 个 task-id、代码 commit、
protocol hash、prompt、工具、recent-4 上下文、动作预算和 `bird-set` verifier。

| 配对结果 | 题数 |
|---|---:|
| both correct | **130** |
| both wrong | **40** |
| Pro-only gain | **15** |
| Flash-only regression | **15** |
| discordant pairs | 30 |
| 净增益 | **0** |
| exact paired binomial/McNemar 双侧 p | **1.0** |

两个模型虽然总分同为 145/200，但并非逐题完全相同：30 题发生翻转，且正反各 15 题。
Legal 的逐题配对为 both-legal 197、Pro-only legal 2、both-illegal 1、Flash-only legal 0；
Pro 的合法终止率更高，但 2 对 0 的 exact 双侧检验也只有 `p=0.5`。

按难度分层：

| 难度 | 题数 | Flash correct | Pro correct | both correct | Pro-only | Flash-only | both wrong |
|---|---:|---:|---:|---:|---:|---:|---:|
| easy | 80 | **66（82.5%）** | 64（80.0%） | 62 | 2 | 4 | 12 |
| medium | 60 | 37（61.7%） | **40（66.7%）** | 32 | 8 | 5 | 15 |
| hard | 60 | **42（70.0%）** | 41（68.3%） | 36 | 5 | 6 | 13 |

Pro 的小幅优势只出现在 medium（+3），被 easy（-2）和 hard（-1）抵消；没有形成随难度
上升而稳定扩大的能力优势。

### 过程可靠性与错误

| 指标 | Flash200 | Pro200 | Pro 相对变化 |
|---|---:|---:|---:|
| legal | 197 | **199** | +2 |
| clean success | 130 | **137** | +7 |
| recovered success | **15** | 8 | -7 |
| 过程错误 | 29 | **22** | -7（-24.1%） |
| 涉及过程错误的任务 | 23 | **16** | -7（-30.4%） |
| 正确任务中的过程错误 | 15 | **9** | -6 |
| argument validation | 9 | **6** | -3 |
| execution error | 17 | **16** | -1 |
| protocol error | 3 | **0** | -3 |

Flash 的非正确最终失败为 52 wrong answer、1 argument validation、1 execution、1 max-steps；
Pro 为 54 wrong answer、1 argument validation。Pro 明显减少了协议和过程层面的失败，却没有
把这种可靠性提升转换成更多 denotation-correct：多出的 legal 终止仍然是 wrong answer。

### 工具调用分布

下表统计所有 200 条轨迹中模型实际提交的 semantic actions，包括被 harness 拒绝后获得
结构化反馈的 action，而不只统计正确轨迹。

| 工具 | Flash200 | Pro200 | 差值 |
|---|---:|---:|---:|
| `plan` | 12 | 9 | -3 |
| `describe_table` | 223 | 222 | -1 |
| `inspect_column` | 78 | 121 | +43 |
| `read_subtable` | 179 | 130 | -49 |
| `condition_filter` | 379 | 290 | -89 |
| `project` | 91 | 110 | +19 |
| `scalar_compute` | 26 | 30 | +4 |
| `join_tables` | 119 | 168 | +49 |
| `group_aggregate` | 133 | 125 | -8 |
| `extreme_value_select` | 38 | 39 | +1 |
| `set_op` | 3 | 5 | +2 |
| `answer_from_context` | 197 | 199 | +2 |
| **合计** | **1,490** | **1,454** | **-36（-2.4%）** |

Pro 更偏向 `inspect_column`、`join_tables` 和 `project`，少用 `read_subtable` 与
`condition_filter`。这说明模型能力会改变工具策略，但在当前 version24 接口上，策略变化
没有提高最终正确率。

### Token、API 与时间

| 指标 | Flash200 | Pro200 | Pro 相对变化 |
|---|---:|---:|---:|
| prompt tokens | 8,003,435 | **7,783,467** | -2.75% |
| completion tokens | 417,426 | **375,655** | -10.01% |
| total tokens | 8,420,861 | **8,159,122** | -3.11% |
| tokens / task | 42,104.3 | **40,795.6** | -3.11% |
| tokens / correct | 58,074.9 | **56,269.8** | -3.11% |
| cached prompt tokens | 5,943,040 | 4,909,568 | -17.39% |
| API request attempts | 1,509 | **1,479** | -1.99% |
| transport retries | **11** | 19 | +8 |
| context retries | 0 | 0 | 0 |
| completion-length retries | 8 | **6** | -2 |
| unresolved final API failure | 0 | 0 | 0 |

Flash 两个冻结 shard 报告的推理墙钟合计为 1,087.953 秒（18.13 分钟）。Pro 因余额中断，
端到端日历时间包含充值等待，不能直接比较；200 条记录的累计 episode latency 为
9,581.803 秒，均值 47.91 秒，Flash 为 7,516.005 秒、均值 37.58 秒，即 Pro 单题延迟
高约 27.5%。Pro 的完整第一阶段 manifest 被后续 resume 覆盖：可审计的 first-5、前缀恢复、
未开始 67 题分别为 146.744、371.852、639.887 秒；由文件时间界定的中间 70 题约 322 秒，
故排除充值停机和审计后，有效 API 推理墙钟约 1,480 秒（24.7 分钟），该数是估计值。

收集中曾产生 58 个 `insufficient_user_quota` 基础设施失败记录；它们没有被计作模型错误。
14 条部分轨迹与 44 条零-action 轨迹均按完全相同的下一请求恢复，最终未解决 provider failure
为 0。常规 token/API 统计包含完成后的唯一语义轨迹；quota 停机不消耗模型 token。

### Flash20 当前复现明细

当前 provider 上的 Flash 复现并不是用 20 题去直接比较 Pro200，而是启动 Pro200 前的稳定性门：

| 指标 | Flash20 当前复现 |
|---|---:|
| correct / legal | **16/20 / 20/20** |
| clean / recovered | 15 / 1 |
| 过程错误 | 1，涉及 1 题 |
| semantic actions | 131；均值 6.55，中位数 6 |
| prompt / completion / total tokens | 698,270 / 54,637 / 752,907 |
| API attempts | 134 |
| transport / context / completion retries | 0 / 0 / 3 |
| 墙钟 | 218.214 秒 |
| structural / fresh replay | 16/16 / 16/16 |

其全部 action 分布为：`condition_filter` 32、`describe_table` 23、
`answer_from_context` 20、`read_subtable` 19、`project` 14、`join_tables` 11、
`inspect_column` 4、`group_aggregate` 4、`extreme_value_select` 2、`scalar_compute` 1。
这 20 题全部是 easy；结果与历史锚点的 16/20、20/20、1 个过程错误完全复现。

### 审计与预注册判定

Pro 的 145 条正确轨迹全部通过：

- structural audit：145/145；
- fresh independent `bird-set` replay：145/145；
- provider payload boundary：200 records、1,454 provider turns，gold-key 0、gold-SQL-text 0；
- protocol hash：`25ac4c10ef96365c`，无漂移；
- `sft_export_eligible=false`、`training_admission=diagnostic_only`。

预注册的五个能力上限条件中：

1. Pro ≥150/200：**失败**（145）；
2. Pro-only > Flash-only：**失败**（15 = 15）；
3. exact paired `p < 0.05`：**失败**（1.0）；
4. legal ≥197 且无 unresolved provider failure：**通过**（199，0）；
5. 所有正确轨迹通过独立审计：**通过**（145/145）。

## 结论

**当前证据不支持 DeepSeek v4 Pro 提高 atomic version24 的工具调用能力上限。** Pro 与 Flash
在同一冻结 200 题上都为 **145/200**，逐题净增益为 0，exact `p=1.0`。Pro 确实带来更好的
工程可靠性和轻微效率提升：legal +2、过程错误 -24.1%、actions -2.4%、tokens -3.1%，
但这些提升没有转化为 denotation accuracy。

更准确的解释是：在 version24 的 prompt、recent-4 上下文、结构化错误反馈和 atomic 工具边界
下，当前瓶颈不是只靠把 Flash 换成 Pro 就能突破的“裸模型能力”；工具语义、输出形状约束与
任务级语义决策仍是主要限制。由于 Flash200 是 2026-07-24 的历史运行，而 Pro200 是
2026-08-03 的当前运行，即便 task/protocol/code 完全冻结，仍不能排除 provider 后端与时间漂移。
当前 Flash20 的精确复现降低了这项风险，但不能把非显著同分解释为严格的模型因果等价。
