# Atomic F 方案完整上下文 fixed-200 评估

日期：2026-07-29

## 结论

在 atomic 历史最佳 **145/200 = 72.5% `bird-set`** 的完全相同冻结 200 题上，
F 方案得到 **136/200 = 68.0%**，净退化 9 题：

- 配对结果为 F 独赢 9 题、baseline 独赢 18 题；
- 27 个 discordant pairs 的 exact two-sided p 值为 `0.1221`，未达到统计显著；
- 合法终止从 197/200 降至 192/200；
- 语义动作从 1,490 降至 1,216（-18.4%），但总 token 从 8,420,861 增至
  14,832,505（+76.1%）；
- 过程错误从 29 增至 43。

因此，F 能减少逐步 schema 探索和模型动作，但不能提高这批任务的端到端正确率，
且把完整宽 schema 重复放进每轮上下文造成了很高的 prompt-token 成本。它不应替换
当前 atomic 默认方案，也不能作为 SFT 数据源。

## F 方案

F 是 `full-bird-schema-samples-v1`：

- 开局一次性提供全部 source tables、全部列和类型、行数、主外键；
- 提供 BIRD `database_description` 的短列语义与格式说明；
- 每列提供最多两个从当前 SQLite 实例读取的 distinct 非空示例值，字符串最长
  40 个字符，并明确示例不是完整值域；
- 从模型可见 prompt 和运行时合法工具面同时移除 `describe_table` 与
  `inspect_column`；
- `gold_sql` 不进入模型上下文，只由 harness 用于 `bird-set` 终止校验。

200 题预检覆盖 57 个数据库，57/57 上下文构建成功；共有 483 个 source tables、
3,258 个 source columns，最大单库模型可见数据库上下文约 106,295 字符。

## 冻结比较设置

两边使用同一题目文件：

`data/eval_inputs/bird_train_tool_interface_validation200_version4.jsonl`

SHA-256：

`6b469215ac96e09c1046fee539ff4c4885b58e9b7a0b6fdb2187082267dcde36`

共同设置：

- model：`deepseek-v4-flash`
- temperature：0
- thinking：enabled
- reasoning effort：high
- provider carrier：JSON Output
- context：recent-4 rolling legal history
- rolling prompt：full
- policy prompt：canonical
- plan policy：optional
- maximum semantic actions：30
- maximum completion tokens：2048
- attempts per task：1
- metric：`bird-set`
- workers：8

baseline 是 2026-07-24 的 atomic version24；F 使用当前 atomic version39。因此这是
同 cohort、同模型和主要解码参数下的方案比较，不是严格单变量 A/B。Version39 还包含
后续教师语义约束和 resident-state compaction；结果不能把所有变化都归因于完整上下文。

正式启动时 runner 未显式指定 `--limit`，默认只选择 20 题；在写盘 6 条后立即停止。
随后以 `--start 0 --limit 200 --resume` 继续其余 194 条。最终 200 个
`trajectory_id` 全部唯一、无重复 attempt，且 200 条记录的 protocol hash、profile、
carrier、metric 与 attempt index 完全一致。

## 总体结果

| 指标 | atomic version24 baseline | F | 变化 |
|---|---:|---:|---:|
| `bird-set` 正确 | **145/200** | 136/200 | -9（-4.5 pp） |
| 合法终止 | **197/200** | 192/200 | -5 |
| 语义动作 | 1,490 | **1,216** | -274（-18.4%） |
| 平均动作/题 | 7.45 | **6.08** | -1.37 |
| 过程错误 | **29** | 43 | +14（+48.3%） |
| 有过程错误的题 | **23** | 28 | +5 |
| API request attempts | 1,509 | **1,404** | -105（-7.0%） |
| prompt tokens | **8,003,435** | 14,401,673 | +79.9% |
| completion tokens | **417,426** | 430,832 | +3.2% |
| total tokens | **8,420,861** | 14,832,505 | +76.1% |

F 的 64 个失败包括：

- 56 个合法 `wrong_answer`；
- 4 个 `provider_carrier_error`；
- 2 个 `max_steps`；
- 2 个 `argument_validation_error`。

即使反事实地把 F 的全部 8 个非合法终止都算对，成绩也只有 144/200，仍低于 baseline
的 145/200。这说明退化不能仅由 provider 或接口终止故障解释。

## 配对结果

| 配对结果 | 数量 |
|---|---:|
| 两边都正确 | 127 |
| 两边都错误 | 46 |
| F 独赢 | 9 |
| baseline 独赢 | 18 |

F 独赢：

`00796, 02088, 03042, 03262, 03636, 03724, 03969, 03977, 06324`

baseline 独赢：

`00421, 00454, 00833, 01050, 01148, 01393, 01787, 02507, 02682, 02709,
02901, 03131, 03312, 03505, 04244, 04906, 05053, 06336`

18 个 baseline 独赢中：

- 3 个在 F 下以 `provider_carrier_error` 终止：`00421, 00454, 05053`；
- 1 个以 `argument_validation_error` 终止：`03505`；
- 其余 14 个均为合法完成但答案错误。

排除上述 4 个非语义终止翻转后，F 仍是 9 个语义增益对 14 个合法语义回归，净 -5。
完整 schema 确实能修复部分选表、列义与输出形态错误，但也会让模型更早锁定一个
“看起来合理”的关系路径；信息增加没有稳定转化为更好的关系人口、grain、连接语义和
最终输出槽位判断。

## 接口与 provider 失败

F 的 43 个过程错误按工具分布：

- `join_tables`：21 个 argument validation + 1 个 execution error；
- `read_subtable`：8 个 argument validation + 1 个 no-progress；
- `group_aggregate`：3 个 argument validation；
- `project`：2 个 argument validation；
- `condition_filter`：1 个 argument validation；
- `answer_from_context`：1 个 argument validation；
- carrier/parser 层：5 个 protocol error。

这说明移除 schema perception 工具没有降低总体接口摩擦：模型动作虽更少，却更频繁地
在 join namespace、派生列和 observation-only `read_subtable` 参数边界上出错。

API 侧有 13 次 transport retry、15 次 completion-length retry 和 156 次有界
same-turn carrier retry，最终仍有 4 个 carrier failure。此前 32 题 F 诊断没有 carrier
retry/failure，因此这部分更像 provider 时段或当前 adapter 交互稳定性，而不是数据库
上下文本身的确定性影响；但它仍属于端到端工具方案的实际可靠性成本。

## 验证

- 200 条 all records，200 个唯一任务，0 个重复 attempt；
- 136 条 verifier-correct 轨迹结构审计：136/136 pass；
- profile-aware deterministic replay：136/136 pass；
- 当前质量门（`max_steps=30`, `max_think_words=300`）：103/136 pass，
  33 条仅因 `think_limit` 被拒绝；
- 全部输出均为 `diagnostic_only_pending_protocol_scale_gate`，
  `sft_export_eligible=false`。

## 判断

F 的作用更接近“用更大的静态 prompt 换更少的探索动作”，而不是提升教师模型的关系
推理能力：

1. 它把动作数降低 18.4%，但总 token 增加 76.1%，效率并不更好；
2. 它有 9 个真实配对增益，证明完整 schema/BIRD 描述对一部分歧义题有用；
3. 但有 14 个合法语义回归，说明全量信息也会诱导过早承诺，且不能替代交互验证；
4. 过程错误和非法终止均高于 baseline；
5. 136/200 明确低于 145/200，因此不推广、不用于 SFT。

若继续实验，更合理的方向不是继续扩大全量静态上下文，而是保留 atomic 的按需探索，
只把紧凑的 BIRD 列语义作为 catalog augmentation，或让 harness 根据题目相关性提供
确定性裁剪后的 schema 子集；任何新方案都应先做同时段、小规模 paired control。

## 产物

- baseline 报告：
  `docs/reports/evaluation/BIRD_VERSION24_RELATION_DERIVATION_FIXED200_20260724.md`
- F manifest：
  `data/trajectories/bird_train_atomic_full_bird_context_fixed200_20260729/verified_success.manifest.json`
- F 全部 200 条审计记录：
  `data/trajectories/bird_train_atomic_full_bird_context_fixed200_20260729/verified_success.all.jsonl`
- F 136 条 verifier-correct 轨迹：
  `data/trajectories/bird_train_atomic_full_bird_context_fixed200_20260729/verified_success.jsonl`
- 结构审计：
  `data/trajectories/bird_train_atomic_full_bird_context_fixed200_20260729/structural_audit.json`
- replay/质量摘要：
  `data/trajectories/bird_train_atomic_full_bird_context_fixed200_20260729/profile_aware_replay_audit.json`
- 配对摘要：
  `data/trajectories/bird_train_atomic_full_bird_context_fixed200_20260729/paired_vs_version24.json`
