# Checkpoint-RelAlg Model-Choice V6 与历史 Atomic Fixed-200 对比（2026-08-11）

## 结论

在冻结 fixed-200 cohort 上，当前 `checkpoint-relalg` Atomic `semantic-v2`、
`model-choice-commit-v6` 得到 **142/200（71.0%）** `bird-set`；历史 Atomic version24 为
**145/200（72.5%）**。逐题配对为 129 题双方都对、13 题仅 v6 对、16 题仅 version24 对、
42 题双方都错；McNemar 双侧精确检验 `p=0.7111`。因此当前结果是统计上未区分的 -3 题，
不能声称 v6 或 checkpoint 提高了正确率。

checkpoint 已经获得充分 exposure：107/200 题至少提交一次，共 130 次 accepted commit，
分布为第一次 107、第二次 19、第三次 4；没有连续 commit，restore 按本实验配置为 0。
但是，在模型自行选择 checkpoint 的这 107 题上，v6 为 70/107，历史 version24 同题为
71/107（9 gain、10 regression）。这同样没有 checkpoint 净增益证据。该子集由 v6 自身行为
内生选择，只能作诊断，不能当作随机 checkpoint treatment 的因果估计。

当前最明确的负面结果是可靠性与成本：v6 legal 为 188/200，低于历史 197/200；工具错误
96 对 29；provider tokens 为 28,614,486 对 8,420,861（3.40x）。因此 v6 仍为
diagnostic-only，不进入 SFT/RL。

## 冻结身份与运行方式

- candidate：official DeepSeek `deepseek-v4-flash`；`mode=atomic`；
  `atomic_operator_profile=semantic-v2`；Text-JSON carrier A；
  `checkpoint_guidance_profile=model-choice-commit-v6`；max checkpoint 8；restore 0。
- dataset：`bird_train_tool_interface_validation200_version4.jsonl`，SHA256
  `6b469215ac96e09c1046fee539ff4c4885b58e9b7a0b6fdb2187082267dcde36`。
- historical control：version24 fixed200 first50 + remaining150，按
  `trajectory_id == example_id` 配对，不按 artifact 物理行序配对。
- candidate positions 0–199 精确覆盖，200 unique IDs，无缺失、无重复。
- 20 个已完成 prefix records 被原样复用；只新请求 positions 20–199。剩余任务用四路并行，
  单题内部保持因果串行。
- 155–199 分片在完成 positions 155–198 后触发局部 7.5M token guard；position 199 从未被该
  分片请求，随后由唯一的 199-only top-up 分片完成。没有重复付费 episode。
- 聚合只读取公开 outcome、工具/预算/身份与审计字段；未读取或输出 gold SQL、gold rows、
  question、external knowledge 或模型 reasoning。

## 主结果

| 指标 | v6 semantic-v2 | 历史 Atomic version24 | 差异 |
|---|---:|---:|---:|
| `bird-set` correct | 142/200 (71.0%) | 145/200 (72.5%) | -3 / -1.5pp |
| legal termination | 188/200 (94.0%) | 197/200 (98.5%) | -9 / -4.5pp |
| correct / legal（描述性） | 142/188 (75.5%) | 145/197 (73.6%) | +1.9pp |
| strict artifact | 78/200 (39.0%) | unavailable | — |
| schema match | 95/200 (47.5%) | unavailable | — |
| model turns / actions | 1,635 | 1,490 | +145 (+9.7%) |
| provider attempts | 1,694 | 1,509 | +185 (+12.3%) |
| provider tokens | 28,614,486 | 8,420,861 | 3.40x |
| tool/process errors | 96 | 29 | 3.31x |

`correct/legal` 仅说明：在能够合法提交 exact relation 的记录中，v6 的关系语义并未明显崩溃；
它不是预注册总体正确率，不能排除非随机 failure conditioning，不能替代 142/200 主结果。

## 逐题配对

| 配对类别 | 数量 |
|---|---:|
| both correct | 129 |
| v6 only | 13 |
| version24 only | 16 |
| both wrong | 42 |

不一致题共 29，v6 gains 少于 regressions 3；精确 `p=0.7111`。本对照复用相同任务身份，
但不是同时期、同 carrier、同 schema/prompt 的单变量 fresh pair。因此它回答的是“当前完整方案
相对历史 Atomic baseline 的表现”，不能把差异单独归因于 checkpoint。

## Checkpoint 行为

| 指标 | 结果 |
|---|---:|
| checkpoint-used episodes | 107/200 |
| accepted commits | 130 |
| commit ordinal 1 / 2 / 3 | 107 / 19 / 4 |
| max accepted commits in one episode | 3 |
| consecutive-commit episodes | 0 |
| restore | 0 |
| checkpoint subset v6 | 70/107 (65.4%) |
| same IDs historical version24 | 71/107 (66.4%) |
| checkpoint subset paired v6-only / v24-only | 9 / 10 |
| no-checkpoint subset v6 | 72/93 (77.4%) |
| same IDs historical version24 | 74/93 (79.6%) |
| mean v6 turns, checkpoint / no-checkpoint | 9.89 / 6.20 |
| mean v6 tokens, checkpoint / no-checkpoint | 171,783 / 110,039 |

v6 的触发策略达到了两个设计目标：模型自主选择边界；第一阶段后允许有限的多次 commit，并且
不再连续空 commit。它没有达到核心效果目标：checkpoint-used 子集没有正确率净增益，而且
phase reset 节省的上下文不足以抵消长轨迹、完整 schema 和额外决策成本。

## 失败与可靠性

v6 failure types：142 correct/`none`、46 `wrong_answer`、10 `provider_error`、1
`max_model_turns`、1 `batch_limit_reached`。188 个 legal records 中 46 个关系/输出答案错误；
12 个非 legal failures 使总体正确率额外下降。

96 个工具错误的主要 code：

- `invalid_text_json_content` 25；
- `unknown_column` 15；
- `unexpected_field` 13；
- `invalid_arguments` 10；
- `canonical_type_mismatch` 9；
- `type_mismatch` 7；
- `missing_required_field` 6；
- `result_too_large` 4；
- 其余 7。

这说明当前瓶颈并不只是 checkpoint 触发。Text-JSON envelope、复杂 typed schema、列身份和
严格 output contract 共同造成了高错误率；checkpoint 只能压缩历史，不能修正错误关系语义，
也不能消除 provider carrier failure。

## 审计

- 200/200 record structure 通过；200/200 fresh replay 通过；provider response model 共 1,625
  个 model turns，全部为 `deepseek-v4-flash`。
- 8 个正常完成分片 overall audit 通过。
- `full200_lane_155_199` 的 44/44 records structure/fresh replay 均通过，但该分片 overall
  audit 按设计失败：最后一个在途 response 使 tokens 达到 7,510,365，超过局部 7,500,000
  guard 10,365 tokens，并标记 incomplete/stopped。缺失 position 199 由独立、审计通过的 top-up
  完成。不能把该 stopped shard 描述为 overall-audit pass。
- 全 candidate 实际 provider tokens 28,614,486，低于 40,000,000 全局硬上限。

## 研究判断

1. **原子工具简化决策空间的主张在本次完整方案对比中没有获得新支持。** 正确率与历史
   Atomic 统计持平，但 actions、errors 和 tokens 都更高。
2. **checkpoint 机制已经“会被用”，但尚未“产生净收益”。** 107 题 exposure 足够排除小样本
   的“模型根本不 commit”解释；同题 -1 的结果排除当前 v6 已有明显正增益。
3. **不要据此删除 checkpoint。** 本实验同时改变了工具粒度、carrier、prompt、schema 和
   provider 运行时期。下一步若继续优化，应冻结当前 semantic-v2 工具与 carrier，做 fresh
   v6 versus checkpoint-disabled paired control，才能估计 checkpoint 本身的因果效果。
4. 在进入下一次付费 gate 前，应优先降低 carrier/schema 错误和 exact-output 失败；否则
   checkpoint 的小幅上下文收益会继续被非阶段性错误淹没。

本批保持 diagnostic-only；不得直接进入 SFT/RL 或与历史 Atomic 轨迹混合。
