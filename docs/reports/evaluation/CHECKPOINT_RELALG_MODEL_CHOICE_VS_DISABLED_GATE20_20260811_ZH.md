# checkpoint-relalg 模型自主 Checkpoint vs 禁用 Checkpoint Gate20（2026-08-11）

## 结论

扩展到 20 个配对任务后，`model-choice-commit-v1` 仍未形成 checkpoint treatment：自主臂
20/20 个 episode、223 个 model turns、222 个 primitive calls 中，`commit_checkpoint` 的尝试数
和成功数均为 0。较长的 13、14、16、18 turn 轨迹也没有尝试 commit。对照臂按配置禁用
checkpoint，同样为 0。

因此 9/20 对 11/20 的 `bird-set` 结果不能解释为 checkpoint 降低了正确率。两臂实际都执行了
零-checkpoint 策略，观察到的两题差异只能归因于不同 teacher guidance 和模型采样。该结果以比
Gate4 更大的样本再次确认：完全可选的“may commit”指导不足以让 DeepSeek v4 Flash 主动把
checkpoint 当作长轨迹控制动作。

`restore_checkpoint` 在两臂都保持预算 0。本轮不测试 restore，也没有 restore 调用。

## 冻结设计

- 实现提交：`1ff03b60f60caf8b4c48b3d5ade6cbd1eacd2ff9`。
- 自主臂：`model-choice-commit-v1`，Harness commit eligibility=`none`，checkpoint 上限 8，
  restore 上限 0。
- 禁用臂：`checkpoint-disabled-v1`，checkpoint/restore 上限均为 0。
- 共同配置：Atomic `semantic-v2`、A/Text-JSON、官方
  `https://api.deepseek.com/chat/completions`、`deepseek-v4-flash`、20 model turns、30 primitive
  calls、每 episode 425,000 provider-token 闸值。
- 数据源：`bird_train_atomic_teacher1500_v2_nonempty.jsonl`，SHA-256
  `9c4995daf81f94053931fbdaa7bd11beef9157909bacde03be6cb61904952264`；manifest SHA-256
  `4c6b96e523d02b45c01eaa60df627abac4a6c3720a89f6573f3eec92740537fe`。
- 协议 hash：`bcbd2d563730f29f88823a922c229a381f340fa25e36a3079e7fa005de9bf17d`；
  student prompt SHA-256：`d44276fca1e9f4b1241467962147a64b9625dc2c2c9ef5cce1c08f3bbee4c67f`；
  schema SHA-256：`dc6baa7f9df1cc578da825bd66043dd25146464b003947cc543cb20bcea932d8`。
- 两个 teacher prompt SHA-256 分别为 `de4361249b693676120b59cfc0aedcfff93d1a2455d3130d31829ce8ffd581ca`
  与 `c8368c8269ec09c9f39324a02d8b2e17883cb072d40532d1e5ef7ba680fc5bfa`。

冻结位置为：1、6、14、18、19、20、22、26、32、33、35、37、42、44、46、47、54、63、
67、79。保留 Gate4 的 1/14/42/47；保留此前无内容 checkpoint 诊断的冻结位置；额外的
67/79/6/33 是旧 Atomic Prefix100 中未入选位置按 `model_turns` 降序选出的前四项，tie 仅按
position 升序。选择过程没有读取 question、gold、答案、reasoning 或 correctness。

新增 16 对共 32 个付费 episode 的预注册总上限为 13,600,000 provider tokens，实际使用
7,271,682 tokens、353 provider attempts、334 turns。四条动态 lane 并行，每题两臂相邻串行，
最大四个 episode 同时在途；总结果再与冻结 Gate4 的 8 个 episode 合并。

## Gate20 结果

| arm | bird-set | strict artifact | schema match | legal | turns | attempts | tokens | errors | checkpoint attempts/success |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| model-choice | 9/20 | 6/20 | 9/20 | 15/20 | 223 | 233 | 4,783,849 | 16 | 0 / 0 |
| checkpoint-disabled | 11/20 | 6/20 | 10/20 | 17/20 | 206 | 218 | 4,536,765 | 16 | 0 / 0 |

逐题 `bird-set` 配对为：both correct 9、model-choice only 0、disabled only 2、neither 9。
两条 disabled-only 是 position 14 和 42；精确双侧 McNemar `p=0.5`。strict artifact 的
逐题结果完全一致：both strict 6、两侧独占均为 0、neither 14。

自主臂比禁用臂多 17 turns（+8.3%）、15 attempts（+6.9%）和 247,084 tokens（+5.4%），
但差异不能归因于 checkpoint，因为没有任何 checkpoint action。两臂各有 16 个工具/协议/
执行错误事件。

失败分布如下：

- model-choice：9 correct、6 legal wrong、4 `max_provider_tokens` stop、1 provider error；
- disabled：11 correct、6 legal wrong、2 `max_provider_tokens` stop、1 provider error。

## 审计

- 40/40 requested/response identity 只出现 `deepseek-v4-flash`，endpoint 为官方 DeepSeek；
- 40/40 record structure、manifest binding、cohort identity 与 fresh replay 通过；
- 33/40 overall audit 通过；其余 7 条仅因冻结的 425,000 token 闸值及 batch stop 失败；
- 超限位置为 Gate4 的 position 42 两臂，以及扩展中的 position 35 两臂、position 54
  model-choice、position 79 两臂；
- 没有 protocol/hash/cohort/provider-model/fresh-replay 问题。

单个已经在途的 provider response 会在 Harness 观察到闸值前返回，因此 position 35 的最高记录
达到 547,686 tokens。这些记录保留为预算停止，不重跑、不做 verifier selection。

## 为什么模型仍不主动 checkpoint

1. **即时动作成本清楚，未来收益不清楚。** commit 会多占一轮；只有后续上下文真的变长时，
   phase reset 的收益才显现。模型在当前轮更容易选择继续生产 artifact。
2. **“may commit”不是行动决策。** prompt 描述了何时可以 commit，却没有要求模型先把任务判断为
   单阶段或多阶段，也没有要求多阶段任务在稳定阶段完成后在 commit/answer 之间作显式选择。
3. **restore 已禁用后，checkpoint 只剩压缩/阶段记忆价值。** 它不能纠正已经错误的关系选择，
   对模型的短期成功概率没有直接可见反馈。
4. **长轨迹本身不会自然触发。** 本轮最长的自主轨迹达到 18 turns，仍然 0 attempt，说明单纯扩大
   样本或等待轨迹变长不会解决 exposure 问题。

## 决策

`model-choice-commit-v1` 冻结为 **underexposed / manipulation failure**。它既不能证明 checkpoint
有效，也不能证明 checkpoint 无效或有害；不得把 9/20 对 11/20 写成 checkpoint 的行为比较。

下一轮若继续保留模型对阶段边界的所有权，应只改 teacher policy：模型在第一轮自行分类
single-stage/multi-stage；single-stage 直接求解，multi-stage 在每个自己认定的稳定、可复用阶段完成后
必须 commit，Harness 仍只验证 max-8、distinct goal 和快照一致性，不按 producer 数量决定时机。
首先必须通过 checkpoint exposure gate，再比较正确率和成本。

本实验继续为 `diagnostic-only`，所有轨迹均不得进入 SFT/RL。
