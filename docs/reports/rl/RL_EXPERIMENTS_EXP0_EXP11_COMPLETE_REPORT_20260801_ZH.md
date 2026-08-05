# 当前 RL 实验完整报告：Exp0–Exp11

日期：2026-08-01
状态：Exp0–Exp10 的 equal-300、K=4 正式评测完成；Exp11 训练完成，正式评测进行中
主评测指标：BIRD `bird-set`

## 1. 执行摘要

截至本报告生成时，已经完成 1 个 SFT2 控制、1 个 Result-only RL、1 个历史
Process-RL，以及 8 个通过强制门禁后、从同一 SFT2 初始化的 Process-RL 消融。
Exp11 的训练也已完成，但 300 题 K=4 正式评测尚未结束，因此不能进入正式排名。

当前最重要的结果是：

1. **没有任何一个已完成 RL 模型在 equal-300 上显著超过 SFT2。** 最高 pass@1
   是 Exp5 Rank 的 44.67%，比 SFT2 高 2.33 个百分点，但逐题 exact paired
   `p=0.4101`。最高 pass@4 是 Exp8 ActionOnlyRank 的 59.00%，比 SFT2 高 3.33
   个百分点，`p=0.0872`。这些都是候选信号，不是统计上已经成立的提升。
2. **旧 Process-RL 的主要问题不是能力完全消失，而是首轨迹和行为稳定性恶化。**
   Exp2 pass@1 为 37.67%，比 SFT2 低 4.67 个百分点；pass@4 仍为 55.33%，只低
   0.33 个百分点。它还增加了协议错误、无进展行为和错误轨迹长度。
3. **单独去掉 BackSlice 不会自动提升性能。** Exp3 NoBackSlice 的 pass@1/4
   是 38.67%/52.67%，仍低于 SFT2 和旧 Process-RL。它消除了一个有争议的事后
   credit 来源，但剩余 reward 本身不足以保证更好的策略。
4. **Rank 是目前对 pass@1 最有效的单项改动。** Exp5 相对其直接父实验 Exp3，
   pass@1 提升 6.00 个百分点，逐题 gain/regression 为 34/16，未经多重校正的
   exact `p=0.0153`。但它相对 SFT2 的提升仍不显著。
5. **只对工具 token 做 Rank 更偏向扩大可解覆盖。** Exp8 的 pass@4 达到当前最高
   59.00%，但 pass@1 为 42.67%，比 Exp5 低 2.00 个百分点。它更像提高“4 次至少
   有一次成功”的概率，而不是把正确轨迹稳定推到第一条随机样本。
6. **Exp9 的 conservative update 单独使用会退化；Exp10 修正了 score/update
   不一致。** Exp9 相对 Exp8 的 pass@1/2/4 分别下降 1.67/4.67/3.67 个百分点。
   Exp10 同时对 score 和 gradient 做 conservative masking 后，相对 Exp9 恢复
   3.33/6.33/1.00 个百分点，达到 44.33%/52.00%/56.33%。
7. **Exp10 的总正确轨迹最多，但覆盖不是最多。** 它在 1200 条轨迹中有 520 条
   正确（43.33%，当前最高），而 Exp8 只有 508 条正确却覆盖了 177 个问题。这表明
   Exp10 把成功概率集中在已会的问题，Exp8 把成功概率扩散到更多问题。
8. **强惩罚不是全局冠军。** Exp7 相对 NoBackSlice 的 pass@4 提升 4.67 个百分点，
   未校正 `p=0.0288`；但相对 SFT2 仅为 pass@1 −0.33、pass@4 +1.67 个百分点，
   都不显著。它是有价值的局部消融，不足以宣称“强惩罚稳定领先”。
9. **当前 300 题适合筛掉明显失败策略，不足以稳定排序 1–3 个百分点的差异。**
   各单一比例的 95% 区间宽约 11 个百分点；又因为只有一个训练 seed、评测采用
   随机采样，最终候选必须在完整 1534 题或更大分层集合上复验，并增加训练 seed。

如果按当前研究目标分别选择候选：

- 首轨迹候选：Exp5 Rank、Exp10 ScoreMasked；
- 最大覆盖候选：Exp8 ActionOnlyRank；
- 轨迹总正确率候选：Exp10 ScoreMasked；
- 效率候选：Exp6 NoNormalize，但其 pass@1 没有优势；
- Exp11 是否值得进入最终候选，必须等正式 300 题评测完成。

## 2. 实验边界与可比性

### 2.1 初始化与训练协议

Exp1–Exp11 的 RL checkpoint 都以同一个 SFT2 `checkpoint-1682` 为策略初始化，
**包括历史 Exp2**；没有任何一项从裸 `Qwen2.5-Coder-7B-Instruct` base 开始训练。
除 Exp2 的旧后端/admission 差异外，其余正式 RL 训练统一使用：

- 底座：`Qwen2.5-Coder-7B-Instruct`；
- 初始化：底座上加载冻结的 SFT2 `checkpoint-1682` adapter，**不是裸 base**；
- SFT2 adapter SHA-256：
  `d880e2d7cc3203fdb0d11a7c188d8f607fd297b174eff23f741b6fe73cc3ce6e`；
- 训练问题：同一 23 条 BIRD train 任务；
- 每题 rollout：4 条完整 agent trajectory，共 92 条；
- 每条 trajectory 最多 30 次工具动作；
- 优化：AdamW，learning rate `1e-6`，23 optimizer steps，cosine scheduler，
  warmup ratio `0.03`，PPO clip `0.2`，本轮 KL `0`；
- rollout：temperature `0.7`，top-p `0.95`，max new tokens `1024`，rolling legal
  history 4；
- 后端：Exp1、Exp3–Exp11 使用 TRL transition-level GRPO/PPO；
- trainer GPU 和 vLLM rollout GPU 分离，QLoRA 参数每组同步到推理卡，平均同步时间
  约 1.3–1.6 秒；
- 训练协议：atomic version26；正式评测统一使用 version36。

因此“23 个训练问题”不是 23 次模型输出。每题有 4 条多步 trajectory；例如 Exp3
实际产生 92 条 trajectory、828 个 assistant/tool transitions。不同实验的训练
transitions 在 810–947 之间。

### 2.2 Process-RL 强制门禁

Exp3–Exp11 都走 `counterfactual-completeness` admission，并绑定同一个通过的
`process-counterfactual-suite-v2`：

- deterministic completeness：通过；
- independent grounding edge precision：通过；
- quality audit SHA-256：
  `6ba130933ec4ce20e58117b506d57cd80181f9b1c7c044765572ca8b04ba8e49`；
- passed manifest SHA-256：
  `5624c75c6ebbca556f1e0946069fdb95e923dc95d2d7692c8d929bedebd9568b`。

该门禁要求模型自己的工具程序在反事实数据库上仍然保持正确，排除只在原数据库上
碰巧正确、把观察值硬编码进后续动作、或缺少实际依赖关系的 trajectory。Gold SQL
不会展示给 actor，只由 harness 用于隐藏的终止判分和反事实一致性检查。

### 2.3 正式评测协议

Exp0–Exp10 的下列字段一致，可以按同题逐题比较：

| 项目 | 固定值 |
|---|---|
| 数据 | `bird_dev_equal_difficulty300_seed20260729.indices.json` |
| 难度组成 | Simple / Moderate / Challenging 各 100 题 |
| 每题采样 | K=4，共 1200 条 trajectory |
| 解码 | temperature `0.7`，top-p `0.95` |
| 工具预算 | max steps `30`，max tokens/turn `1024` |
| 工具方案 | atomic version36 |
| 上下文 | rolling legal history，history turns `4` |
| 终止判分 | `bird-set`，exact cited evidence table |
| 概率记录 | sampled-token logprob + top-20 logprobs，1200/1200 覆盖 |

需要强调：这里的 pass@1 是 K=4 中**第一条 temperature=0.7 随机轨迹**，不是从
四条候选中经过 reranker 选出的 top-1，也不是 temperature=0 的 greedy accuracy。

### 2.4 仍然存在的混杂

- Exp2 是历史 Accelerate checkpoint，使用较弱的 `denotation-nonempty` admission，
  训练中还有 3 次梯度 OOM；它只能作为历史失败对照，不能与 Exp3–Exp11 做严格的
  单 reward 因果归因。
- Exp3–Exp11 虽然起点、训练任务和配置框架一致，但每次重新随机采样 rollout；本轮
  只有一个训练 seed，因此实验间差异同时包含策略变量和训练采样方差。
- 正式评测也独立随机采样。逐题 paired test 降低了任务难度差异，但没有消除采样噪声。
- 本轮检验的模型和指标很多；报告给出未校正 exact p 值，同时把多重比较作为限制，
  不根据单个 `p<0.05` 直接宣布最终结论。

## 3. Exp0–Exp11 的变量设计

| 实验 | 名称 | 相对直接父实验的变量 | 目的 |
|---|---|---|---|
| Exp0 | SFT2 | 无 RL | 固定初始化与评测基线 |
| Exp1 | Result-only | terminal `{0,1}`，无 process shaping | 验证 TRL 管线本身 |
| Exp2 | 历史 Process-RL | BackSlice + search reduction + 局部惩罚，旧后端 | 复现过去性能下降 |
| Exp3 | NoBackSlice | `w_back_slice: 1 -> 0` | 去掉事后依赖 credit |
| Exp4 | NoBackSlice + ToolOnly | process loss 只训练 raw JSON 工具 token | 排除 `<think>` token 更新 |
| Exp5 | NoBackSlice + Rank | 加 trajectory pairwise Rank，all tokens/full trajectory | 直接提高正确轨迹相对概率 |
| Exp6 | NoBackSlice + NoNormalize | 正向 credit 不再按轨迹归一化 | 减少 reward diffusion |
| Exp7 | NoBackSlice + StrongPenalty | 三类确定性局部惩罚 ×3，cap 0.95 | 更强抑制明确坏动作 |
| Exp8 | ActionOnlyRank | Rank score/gradient 只看工具 token | 将 preference 集中到动作载体 |
| Exp9 | ConservativeLegalActionRank | 正确轨迹只奖 clean legal 动作；错误轨迹只压明确坏动作 | 不惩罚合法探索 |
| Exp10 | ConservativeScoreMaskedRank | Rank score 也使用相同 conservative mask | 消除 Exp9 score/update 不一致 |
| Exp11 | ConservativeActionMeanRank | selected tool tokens 先动作内平均，再轨迹内平均 | 消除参数长度和动作数偏差 |

Exp3–Exp11 的普通 process-loss 分支均为 NoBackSlice；后续 Rank 实验不是从上一轮 RL
checkpoint 续训，而是各自重新从相同 SFT2 checkpoint 独立开始。

## 4. Reward 与 Rank 的精确定义

### 4.1 Result-only

Exp1 使用：

```text
R(trajectory) = 1  if terminal cited table has correct BIRD denotation
                0  otherwise
```

episode-level advantage 会施加到完整 assistant turn。成功轨迹里早先犯错、后来恢复的
动作也可能得到正方向更新，因此只把它当作 coarse control。

### 4.2 普通 NoBackSlice process reward

Exp3、Exp4、Exp5、Exp8–Exp11 使用同一基础配置：

| 项目 | 权重 |
|---|---:|
| terminal correct | 1.00 |
| executed BackSlice | **0.00** |
| fixed-root search reduction | 1.00 |
| tool / protocol / execution error | −0.08 |
| adjacent exact repeat | −0.06 |
| legal but no state change | −0.03 |
| trajectory penalty cap | 0.80 |
| normalize positive mass | true |

`new_evidence`、`feedback_response`、`target_potential`、`terminal_failure`、
`ignored_feedback`、`unsupported_guess`、`empty_result` 当前权重都是 0。
`ignored_feedback` 会把“无法证明已经恢复”与“确实忽略反馈”混在一起；
`unsupported_guess` 对合法的组合日期 pattern 有系统性误报，因此当前只审计、不入 loss。

对正确 trajectory，harness-grounded 的正向 feature 在动作之间归一化；局部错误仍在
造成错误的动作上扣分。对错误 trajectory，合法探索默认不被惩罚，只有确定性的局部
坏动作产生负 reward。

### 4.3 NoNormalize 和 StrongPenalty

Exp6 保持权重不变，但把正向 credit 从“轨迹内总质量归一到 1”改为“每步 raw 正向
credit 截断到 `[0,1]`”。因此 Exp6 的 trajectory reward 均值大于 1 是 reward 标度
变化，不能与其它实验的均值直接比较。

Exp7 保持 normalized NoBackSlice 正向 credit，只改三类确定性惩罚：

| 惩罚 | 普通 | Exp7 |
|---|---:|---:|
| tool/protocol/execution error | 0.08 | 0.24 |
| adjacent exact repeat | 0.06 | 0.18 |
| legal no-state-change | 0.03 | 0.09 |
| penalty cap | 0.80 | 0.95 |

cap 0.95 保证具有 normalized positive mass 1.0 的正确 trajectory 即使触顶，总 reward
仍至少为 `+0.05`，不会把“最终正确但中途犯错”的 trajectory 整体变成负样本。

### 4.4 Trajectory Rank

同一问题 K=4 的 trajectory 中，构造全部 correct × incorrect pair。基础形式为：

```text
L_rank = coefficient * softplus(-beta * (S_positive - S_negative))
coefficient = 0.5
beta = 0.1
```

其中 `S` 是选定 token/action 的 log-probability 聚合。该分支与普通 NoBackSlice
process loss 相加。

- Exp5：对完整 assistant response token 求和，所有 trajectory turns 接受 rank 梯度；
- Exp8：每个 turn 只对 raw JSON tool-action suffix 求和，`<think>` 不接收 rank 梯度，
  但正确/错误 trajectory 的所有 turns 都参与；
- Exp9：score 仍由 trajectory 的所有工具调用组成，但 rank 梯度只路由到正确轨迹的
  `legal_success && p_local=0` 动作，以及错误轨迹的 `p_local>0` 明确坏动作；错误轨迹
  的合法探索为中性；
- Exp10：score 和 gradient 都使用上述 conservative action mask；任意一侧没有可评分
  动作的 pair 被丢弃；
- Exp11：在 Exp10 基础上，先对每个动作的工具 token logprob 取均值，再对 trajectory
  的选中动作取均值，避免长参数和动作多的 trajectory 仅因 token 多而控制 score。

这套 Rank 不要求两条 trajectory 拥有相同 resident state，也不要求找到共同 prefix
或第一个分叉点。两条轨迹从第一步就不同也可以成对；它比较的是同题终局正确性下的
整条策略偏好，conservative mask 再决定哪些动作实际获得 rank 更新。

## 5. 正式主结果：Exp0–Exp10

### 5.1 Pass@k、合法率、步数和 entropy

| 实验 | pass@1 | pass@2 | pass@4 | 全轨迹 valid | 每题至少一条 valid | 平均步数 | entropy 下界 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Exp0 SFT2 | 127/300 = 42.33% | 146/300 = 48.67% | 167/300 = 55.67% | **78.08%** | **93.00%** | 8.495 | 0.10354 |
| Exp1 Result-only | 124 = 41.33% | 155 = 51.67% | 175 = 58.33% | 77.42% | 91.67% | 8.635 | 0.10482 |
| Exp2 Process | 113 = 37.67% | 140 = 46.67% | 166 = 55.33% | 76.25% | 91.67% | 8.679 | 0.10363 |
| Exp3 NoBackSlice | 116 = 38.67% | 141 = 47.00% | 158 = 52.67% | 76.50% | 92.67% | 8.693 | 0.10249 |
| Exp4 ToolOnly | 121 = 40.33% | 146 = 48.67% | 168 = 56.00% | 76.58% | 92.00% | 8.594 | 0.10503 |
| Exp5 Rank | **134 = 44.67%** | 153 = 51.00% | 170 = 56.67% | **78.25%** | **93.00%** | 8.587 | **0.10559** |
| Exp6 NoNormalize | 120 = 40.00% | 143 = 47.67% | 170 = 56.67% | 76.67% | 92.33% | **8.277** | 0.10209 |
| Exp7 StrongPenalty | 126 = 42.00% | 148 = 49.33% | 172 = 57.33% | 76.92% | 92.33% | 8.645 | 0.10513 |
| Exp8 ActionOnlyRank | 128 = 42.67% | 151 = 50.33% | **177 = 59.00%** | 77.08% | 92.33% | 8.567 | 0.10524 |
| Exp9 Conservative | 123 = 41.00% | 137 = 45.67% | 166 = 55.33% | 75.75% | 92.33% | 8.758 | 0.10376 |
| Exp10 ScoreMasked | 133 = 44.33% | **156 = 52.00%** | 169 = 56.33% | 77.33% | 92.00% | 8.678 | 0.10289 |

entropy 是把 top-20 之外剩余概率质量合并为一个 bucket 后的 token entropy 下界。
所有方法都在 0.102–0.106 的很窄范围内，没有全局概率分布坍缩的证据。

### 5.2 相对 SFT2 的绝对变化

| 实验 | Δpass@1 | Δpass@2 | Δpass@4 |
|---|---:|---:|---:|
| Exp1 Result-only | −1.00 pp | +3.00 pp | +2.67 pp |
| Exp2 Process | −4.67 pp | −2.00 pp | −0.33 pp |
| Exp3 NoBackSlice | −3.67 pp | −1.67 pp | −3.00 pp |
| Exp4 ToolOnly | −2.00 pp | ±0.00 pp | +0.33 pp |
| Exp5 Rank | **+2.33 pp** | +2.33 pp | +1.00 pp |
| Exp6 NoNormalize | −2.33 pp | −1.00 pp | +1.00 pp |
| Exp7 StrongPenalty | −0.33 pp | +0.67 pp | +1.67 pp |
| Exp8 ActionOnlyRank | +0.33 pp | +1.67 pp | **+3.33 pp** |
| Exp9 Conservative | −1.33 pp | −3.00 pp | −0.33 pp |
| Exp10 ScoreMasked | +2.00 pp | **+3.33 pp** | +0.67 pp |

### 5.3 95% Wilson 区间

| 实验 | pass@1 95% CI | pass@4 95% CI |
|---|---:|---:|
| Exp0 | [36.87%, 47.99%] | [50.01%, 61.18%] |
| Exp1 | [35.90%, 46.98%] | [52.68%, 63.77%] |
| Exp2 | [32.37%, 43.27%] | [49.68%, 60.86%] |
| Exp3 | [33.33%, 44.29%] | [47.02%, 58.25%] |
| Exp4 | [34.94%, 45.97%] | [50.34%, 61.51%] |
| Exp5 | [39.14%, 50.32%] | [51.01%, 62.15%] |
| Exp6 | [34.62%, 45.64%] | [51.01%, 62.15%] |
| Exp7 | [36.55%, 47.65%] | [51.68%, 62.80%] |
| Exp8 | [37.20%, 48.32%] | [53.35%, 64.42%] |
| Exp9 | [35.58%, 46.65%] | [49.68%, 60.86%] |
| Exp10 | [38.82%, 49.99%] | [50.68%, 61.83%] |

这些区间高度重叠，说明 300 题不足以仅凭 1–3 个百分点宣布最终胜者。

## 6. 难度分层结果

每个单元格为 `pass@1 / pass@2 / pass@4`，每层都是 100 题，因此数值也直接是百分比。

| 实验 | Simple | Moderate | Challenging |
|---|---:|---:|---:|
| Exp0 SFT2 | 60 / 64 / 69 | 37 / 42 / 50 | 30 / 40 / 48 |
| Exp1 Result-only | 59 / 67 / 72 | 34 / 46 / 54 | 31 / 42 / 49 |
| Exp2 Process | 59 / 65 / 72 | 30 / 37 / 49 | 24 / 38 / 45 |
| Exp3 NoBackSlice | 59 / 63 / 70 | 33 / 42 / 46 | 24 / 36 / 42 |
| Exp4 ToolOnly | 57 / 64 / 71 | 30 / 43 / 49 | **34 / 39 / 48** |
| Exp5 Rank | 60 / 67 / 74 | **38 / 43 / 50** | **36 / 43 / 46** |
| Exp6 NoNormalize | 54 / 62 / 71 | 34 / 42 / 51 | 32 / 39 / 48 |
| Exp7 StrongPenalty | 61 / 66 / 70 | 36 / 43 / **54** | 29 / 39 / 48 |
| Exp8 ActionOnlyRank | 62 / 68 / **75** | 36 / 41 / 51 | 30 / 42 / **51** |
| Exp9 Conservative | 55 / 57 / 69 | **40 / 44 / 47** | 28 / 36 / 50 |
| Exp10 ScoreMasked | **67 / 72 / 72** | 35 / 43 / 51 | 31 / 41 / 46 |

分层信号并不统一：

- Exp10 的总体 pass@1 优势主要来自 Simple：67%，比 SFT2 高 7 个百分点；
- Exp5 在 Challenging pass@1 达到 36%，比 SFT2 高 6 个百分点，是当前最好的难题
  首轨迹结果，但 pass@4 反而低 2 个百分点；
- Exp8 的 pass@4 覆盖提升同时出现在 Simple（+6）和 Challenging（+3）；
- StrongPenalty 的主要亮点在 Moderate pass@4（54%，+4）；
- Exp9 的 Moderate pass@1 较高，但 Simple 和总体 pass@2 明显退化。

每个难度层只有 100 题，这些分层结果的方差比总体更大，应视为定位信号而不是独立
显著结论。

## 7. 逐题配对显著性

`G/R/Net/p` 表示相对 SFT2 的 paired gains、regressions、净变化，以及只在 discordant
题目上计算的双侧 exact McNemar/sign-test p 值。

| 实验 | pass@1 G/R/Net/p | pass@2 G/R/Net/p | pass@4 G/R/Net/p |
|---|---:|---:|---:|
| Exp1 | 23/26/−3/0.7754 | 22/13/+9/0.1755 | 17/9/+8/0.1686 |
| Exp2 | 19/33/−14/0.0704 | 16/22/−6/0.4177 | 13/14/−1/1.0000 |
| Exp3 | 15/26/−11/0.1173 | 21/26/−5/0.5601 | 12/21/−9/0.1628 |
| Exp4 | 20/26/−6/0.4614 | 22/22/±0/1.0000 | 15/14/+1/1.0000 |
| Exp5 | 30/23/+7/0.4101 | 24/17/+7/0.3489 | 18/15/+3/0.7283 |
| Exp6 | 20/27/−7/0.3817 | 20/23/−3/0.7608 | 14/11/+3/0.6900 |
| Exp7 | 18/19/−1/1.0000 | 17/15/+2/0.8601 | 16/11/+5/0.4421 |
| Exp8 | 25/24/+1/1.0000 | 20/15/+5/0.4996 | 19/9/+10/0.0872 |
| Exp9 | 20/24/−4/0.6516 | 16/25/−9/0.2110 | 16/17/−1/1.0000 |
| Exp10 | 24/18/+6/0.4408 | 23/13/+10/0.1325 | 16/14/+2/0.8555 |

没有一个 RL 方法相对 SFT2 达到未校正 `p<0.05`。如果把这里 10 个方法 × 3 个指标
视作同一检验族，Bonferroni 阈值为 0.00167，更不可能宣称显著提升。

### 7.1 相对直接父实验的单变量结果

| 子实验 vs 父实验 | pass@1 父→子（G/R/p） | pass@4 父→子（G/R/p） |
|---|---:|---:|
| Exp1 vs SFT2 | 42.33→41.33（23/26/0.7754） | 55.67→58.33（17/9/0.1686） |
| Exp2 vs SFT2 | 42.33→37.67（19/33/0.0704） | 55.67→55.33（13/14/1.0000） |
| Exp3 vs Exp2 | 37.67→38.67（30/27/0.7914） | 55.33→52.67（13/21/0.2295） |
| Exp4 vs Exp3 | 38.67→40.33（27/22/0.5682） | 52.67→56.00（24/14/0.1433） |
| Exp5 vs Exp3 | 38.67→44.67（34/16/**0.0153**） | 52.67→56.67（26/14/0.0807） |
| Exp6 vs Exp3 | 38.67→40.00（26/22/0.6655） | 52.67→56.67（23/11/0.0576） |
| Exp7 vs Exp3 | 38.67→42.00（30/20/0.2026） | 52.67→57.33（25/11/**0.0288**） |
| Exp8 vs Exp5 | 44.67→42.67（22/28/0.4799） | 56.67→59.00（17/10/0.2478） |
| Exp9 vs Exp8 | 42.67→41.00（21/26/0.5601） | 59.00→55.33（11/22/0.0801） |
| Exp10 vs Exp9 | 41.00→44.33（26/16/0.1641） | 55.33→56.33（19/16/0.7359） |

Exp5 的 pass@1 和 Exp7 的 pass@4 是两个最强的父子消融信号，但它们都没有通过
整组实验的多重比较校正，而且训练 rollout 是重新采样的。正确表述是“值得复验的
候选效果”，不是已证实算法提升。

## 8. Trajectory 覆盖与稳定性

| 实验 | 正确轨迹 / 1200 | 0/4 题 | 1/4 | 2/4 | 3/4 | 4/4 | pass@4−pass@1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Exp0 | 504（42.00%） | 133 | 26 | 31 | 24 | 86 | 13.33 pp |
| Exp1 | 501（41.75%） | 125 | 38 | 24 | 37 | 76 | 17.00 pp |
| Exp2 | 494（41.17%） | 134 | 27 | 33 | 23 | 83 | 17.67 pp |
| Exp3 | 475（39.58%） | 142 | 28 | 23 | 27 | 80 | 14.00 pp |
| Exp4 | 503（41.92%） | 132 | 28 | 28 | 29 | 83 | 15.67 pp |
| Exp5 | 507（42.25%） | 130 | 30 | 27 | 29 | 84 | **12.00 pp** |
| Exp6 | 492（41.00%） | 130 | 33 | 30 | 29 | 78 | 16.67 pp |
| Exp7 | 506（42.17%） | 128 | 34 | 27 | 26 | 85 | 15.33 pp |
| Exp8 | 508（42.33%） | **123** | 39 | 30 | 23 | 85 | 16.33 pp |
| Exp9 | 491（40.92%） | 134 | 30 | 26 | 31 | 79 | 14.33 pp |
| Exp10 | **520（43.33%）** | 131 | 21 | 28 | **37** | 83 | **12.00 pp** |

这张表区分了两种不同改进：

- Exp8 把 0/4 题降到 123，说明它覆盖最多问题，但产生 39 个只成功 1/4 的不稳定题；
- Exp10 的正确 trajectory 总量最多，并有 37 个 3/4 成功题，但 0/4 仍有 131；
- Exp5 和 Exp10 的 pass@4−pass@1 gap 最小，不过 pass@1 仍是第一条随机样本，不能
  把 gap 直接解释成模型显式 reranking 能力。

## 9. 合法性、失败类型与动作行为

### 9.1 最终 trajectory 状态

| 实验 | Correct | Wrong answer | Protocol | Argument validation | Max steps | Context overflow | Execution | 轨迹内错误事件 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Exp0 | 504 | 433 | 135 | 68 | 25 | 24 | 11 | 2001 |
| Exp1 | 501 | 428 | 152 | 59 | 23 | 25 | 12 | 2117 |
| Exp2 | 494 | 421 | 162 | 61 | 30 | 22 | 10 | 2191 |
| Exp3 | 475 | 443 | 160 | 57 | 30 | 23 | 12 | 2247 |
| Exp4 | 503 | 416 | 160 | 63 | 28 | 19 | 11 | 2132 |
| Exp5 | 507 | 432 | **143** | 67 | 24 | 20 | **7** | 1999 |
| Exp6 | 492 | 428 | 171 | **50** | 22 | 23 | 14 | **1936** |
| Exp7 | 506 | 417 | 159 | 58 | 25 | 21 | 14 | 2192 |
| Exp8 | 508 | 417 | 158 | 68 | **17** | **19** | 13 | 2056 |
| Exp9 | 491 | 418 | 158 | 66 | 34 | **18** | 15 | 2270 |
| Exp10 | **520** | **408** | 146 | 60 | 32 | 21 | 13 | 2193 |

Exp5 同时具有较高 accuracy 和较少 protocol/execution 终局失败；Exp6 的错误事件总数
最低、平均步数也最低，但没有转化为 pass@1 优势。Exp9 的 max-step 失败和总错误事件
最高，与其性能退化一致。Exp10 提高正确 trajectory 数，但仍有较多轨迹内错误事件，
说明它的收益不是简单来自“更少犯格式/工具错误”。

### 9.2 主要动作分布

下表均为该实验全部已解析动作中的百分比。

| 实验 | filter | read | join | aggregate | project | answer | 总动作数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Exp0 | 27.82 | 10.27 | 9.24 | 8.56 | 5.94 | 10.89 | 8726 |
| Exp1 | 27.22 | 11.09 | 9.69 | 8.72 | 5.30 | 10.75 | 8747 |
| Exp2 | 27.05 | 11.37 | 9.83 | 8.47 | 5.91 | 10.57 | 8781 |
| Exp3 | 27.94 | 11.19 | 9.02 | 8.33 | 5.53 | 10.65 | 8669 |
| Exp4 | 27.04 | 10.60 | 9.36 | 8.60 | **6.24** | 10.70 | 8729 |
| Exp5 | 27.68 | 11.21 | 9.27 | 8.38 | 5.75 | 10.83 | 8796 |
| Exp6 | 26.92 | 10.37 | **8.95** | 8.68 | 5.99 | **10.98** | **8468** |
| Exp7 | 26.89 | 11.04 | 9.49 | 8.68 | 5.60 | 10.70 | 8717 |
| Exp8 | 27.46 | **11.35** | 9.24 | **8.11** | 5.78 | 10.63 | 8792 |
| Exp9 | 27.68 | 10.37 | 9.64 | **8.83** | 5.70 | 10.53 | 8769 |
| Exp10 | 27.17 | 11.01 | 9.34 | 8.37 | **5.57** | 10.74 | 8723 |

动作边际分布的差异总体不大，没有某种方法通过完全停止探索来获得 accuracy。Exp6
更短、更少 join，Exp8 read 较多且覆盖更广，Exp10 project 较少而 scalar_compute
占比升至 4.12%。这些只能描述相关变化；要判断某一固定 resident state 上动作 logits
是否改善，仍需 fixed-prefix checkpoint rescoring，不能从独立生成轨迹反推。

## 10. 训练过程统计

下表中的时间是 trainer state 内 23 个 `step_time` 的总和；不同服务器速度、轨迹长度
和 rank 计算都会影响它。`Rank pairs` 是实际进入 rank loss 的 correct×incorrect pair
累计数。Exp5 的旧日志没有 selected-transition 字段。

| 实验 | 训练正确 / 92 | 训练 legal / 92 | transitions | Σ step time | Rank pairs | Rank selected transitions | logged max grad_norm |
|---|---:|---:|---:|---:|---:|---:|---:|
| Exp1 Result-only | 59 | 82 | 851 | 2.70 h | 0 | 0 | 0.984 |
| Exp2 历史 Process | 62 | — | 808 | 7.47 h | 0 | — | 3 次梯度 OOM |
| Exp3 NoBackSlice | 56 | 81 | 828 | 2.47 h | 0 | 0 | 0.271 |
| Exp4 ToolOnly | 52 | 72 | **947** | 3.41 h | 0 | 0 | 0.316 |
| Exp5 Rank | 54 | 80 | 852 | 4.02 h | **43** | 未记录 | **76.500** |
| Exp6 NoNormalize | 59 | 80 | 915 | 3.31 h | 0 | 0 | 0.551 |
| Exp7 StrongPenalty | **61** | 77 | 905 | 2.81 h | 0 | 0 | 0.336 |
| Exp8 ActionOnlyRank | 58 | 78 | 845 | **5.02 h** | **43** | 527 | 14.438 |
| Exp9 Conservative | 51 | 76 | 821 | 3.96 h | 36 | 228 | 8.188 |
| Exp10 ScoreMasked | 56 | **81** | 823 | 3.55 h | 16 | 223 | 17.125 |
| Exp11 ActionMean | 56 | 79 | **810** | 3.68 h | 14 | 181 | **0.539** |

Exp11 把 logged max grad_norm 从 Exp10 的 17.125 降到 0.539，同时保持相近的训练
正确率、legal 数和 trajectory reward。这与“action mean 去掉 token 长度/动作数尺度
偏差”的设计目标一致，是一个明确的训练稳定性信号；但是否改善泛化必须等正式评测。

### 10.1 训练 rollout 的 reward 事件

| 实验 | mean trajectory reward | 正/负/零 reward steps | tool error / adjacent repeat / legal no-op |
|---|---:|---:|---:|
| Exp3 | 0.511 | 136 / 171 / 521 | 37 / 128 / 134 |
| Exp4 | 0.415 | 120 / 306 / 521 | 46 / 219 / 260 |
| Exp5 | 0.485 | 131 / 181 / 540 | 46 / 118 / 135 |
| Exp6 | 1.134 | 144 / 234 / 537 | 49 / 155 / 186 |
| Exp7 | 0.468 | 147 / 263 / 495 | 30 / 212 / 233 |
| Exp8 | 0.517 | 139 / 196 / 510 | 44 / 133 / 152 |
| Exp9 | 0.438 | 114 / 187 / 520 | 51 / 119 / 136 |
| Exp10 | 0.512 | 133 / 175 / 515 | 43 / 125 / 132 |
| Exp11 | 0.514 | 124 / 178 / 508 | 44 / 123 / 134 |

这些计数来自每个实验自己在线采样的训练轨迹，不是同一批固定轨迹，因此不能把
“某实验遇到更多 repeat”直接当作 reward 系数的因果结果。Exp6 的 reward scale 因
NoNormalize 与其余实验不同，均值不可横向排序。Exp3–Exp11 的 92 条轨迹均通过当前
process admission 路径进入可审计更新记录。

## 11. 每个实验回答了什么

### Exp1：RL 框架能否工作？

能。Result-only 没有训练崩溃，pass@1 与 SFT2 接近，pass@4 反而提高 2.67 个百分点。
它证明新的 TRL 管线、双卡 rollout/update 和参数同步可以工作，但没有证明 RL 改善
单轨迹能力。

### Exp2：旧 Process-RL 为什么看起来“训废”？

它并未完全丢失可行轨迹：pass@4 基本保留，但 pass@1、valid、错误行为明显变差。
旧后端 OOM 和旧 admission 又引入额外混杂。更准确的诊断是“首轨迹概率与工具行为
恶化”，而不是“所有能力归零”。

### Exp3：BackSlice 是唯一问题吗？

不是。去掉 BackSlice 后 pass@1 只恢复 1 个百分点，pass@4 还下降 2.67 个百分点。
BackSlice 确实是不可靠的事后 credit，但简单删除不足以产生更好的相对偏好信号。

### Exp4：只训练工具 token 是否足够？

有小幅恢复：相对 Exp3，pass@1 +1.67、pass@4 +3.33 个百分点，但 paired p 都不显著。
说明排除 reasoning token 更新可能有帮助，但不是主要增益来源。

### Exp5：Trajectory Rank 是否有效？

这是目前最明确的正向消融。相对 Exp3，pass@1 +6.00 个百分点，未经校正 `p=0.0153`；
同时 valid 恢复到 78.25%。但相对 SFT2 只有 +2.33 个百分点且不显著。Rank 值得继续，
但目前证据还不能支持“超过 SFT2”。

### Exp6：取消 positive normalization 是否有效？

它缩短轨迹并降低错误事件，相对 Exp3 的 pass@4 +4.00 个百分点，但 pass@1 只有
40.00%。更强的 step-local credit 改善了效率和覆盖，没有改善首轨迹成功率。

### Exp7：增强明确坏动作惩罚是否有效？

相对 Exp3，pass@1 +3.33、pass@4 +4.67；后者未校正 `p=0.0288`。但对 SFT2 没有
显著提升。它支持“只惩罚可确定坏动作”的方向，却不支持继续无界增大惩罚。

### Exp8：Action-only Rank 是否优于普通 Rank？

它把 pass@4 提到 59.00%，0/4 问题最少，但 pass@1 比 Exp5 低 2 个百分点。因此它
更适合作为 coverage-oriented 策略，不能说对首轨迹排序更好。

### Exp9：只做 conservative gradient routing 是否有效？

没有。相对 Exp8，三个 pass@k 都下降。原因与实现上的 score/update mismatch 一致：
合法失败探索仍影响完整 trajectory score，却不接收相应梯度；优化目标和实际更新的
动作集合不一致。

### Exp10：score 和 update 同时 conservative 是否有效？

相对 Exp9，pass@1/2/4 恢复 3.33/6.33/1.00 个百分点；pass@2 为当前最高，正确
trajectory 总数也是当前最高。它验证了“score 集合必须和 update 集合一致”的设计，
但还没有超过普通 Exp5 Rank 的 pass@1。

### Exp11：Action-mean 是否有效？

训练已完成。它显著降低了 logged grad_norm 尖峰，说明 rank score 尺度更稳定；正式
泛化结果尚不可用。截至 2026-08-01 09:08 CST，equal-300 K4 评测完成 16/300 题，
远不足以报告准确率或与全 300 结果比较。该实验不进入本报告的正式排名。

## 12. 统计结论与推荐决策

### 12.1 现在能下的结论

- 新 TRL RL 管线可以稳定训练，不存在所有 RL 都必然“训废”的系统性故障；
- BackSlice 不应恢复到后续实验，但去掉它本身不是性能提升机制；
- trajectory pairwise Rank 是当前最值得保留的组件；
- action-only scoring 更偏覆盖，full-response Rank 更偏首轨迹；
- conservative scoring 和 conservative update 必须使用同一 mask；
- 只惩罚明确错误的思路合理，但三倍惩罚的收益尚未超过 Rank；
- action-mean 显著改善训练梯度尺度，等待 Exp11 泛化结果确认。

### 12.2 现在不能下的结论

- 不能声称某个 RL 模型已经显著超过 SFT2；
- 不能用第一条随机 sample 的 pass@1 直接证明模型拥有更好的显式 reranking；
- 不能把 Exp5、Exp7 的单个未校正 `p<0.05` 当作多实验搜索后的最终显著性；
- 不能从边际动作分布或现有生成 logprob 推断固定状态下某个工具的 logits 改变量；
- 不能用 300 题内 1–3 个百分点差异预测完整 1534 题的稳定排序。

### 12.3 下一阶段的最小验证集

Exp11 完成后，建议只保留少量候选进入昂贵验证：

1. SFT2：固定控制；
2. Exp5 Rank：当前最高 pass@1；
3. Exp8 ActionOnlyRank：当前最高 pass@4；
4. Exp10 ScoreMasked：当前最高 pass@2 和总正确 trajectory；
5. Exp11：仅在 equal-300 不明显退化时进入。

对候选进行：

- 完整 BIRD dev 1534 题、同协议 K=4；
- temperature=0 greedy@1，区分采样首轨迹和确定性策略；
- 至少 3 个训练 seed；
- fixed-prefix action rescoring，直接比较正确/错误工具调用的 checkpoint 前后
  logprob margin；
- 预先指定单一主指标和配对检验，避免继续用大量消融后的最优点做无校正选择。

如果 Exp11 保持 Exp10 的 accuracy，同时继续维持较小 grad_norm，下一项最有价值的
组合实验是：**Exp11 ActionMean + Exp7 StrongLocalPenalty**。它组合了当前两个相互
正交的候选优势：稳定的长度无关 Rank score，以及只针对确定性坏动作的更强局部
负 credit。该组合必须继续从 SFT2 独立初始化，不能从 Exp7/Exp11 checkpoint 续训。

## 13. 产物与复现入口

实验配置：

- `src/rl/configs/experiments/phase1_result_only.yaml`
- `src/rl/configs/experiments/phase1_process_current.yaml`
- `src/rl/configs/experiments/phase1_process_no_backslice.yaml`
- `src/rl/configs/experiments/phase1_process_tool_only.yaml`
- `src/rl/configs/experiments/phase2_process_rank.yaml`
- `src/rl/configs/experiments/phase2_process_no_normalize.yaml`
- `src/rl/configs/experiments/phase3_process_strong_penalty.yaml`
- `src/rl/configs/experiments/phase4_process_rank_action_only.yaml`
- `src/rl/configs/experiments/phase5_process_rank_conservative.yaml`
- `src/rl/configs/experiments/phase6_process_rank_conservative_score_masked.yaml`
- `src/rl/configs/experiments/phase7_process_rank_conservative_action_mean.yaml`

Reward 配置：

- `src/rl/configs/simple_process_reward.json`
- `src/rl/configs/simple_process_reward_no_backslice.json`
- `src/rl/configs/simple_process_reward_no_backslice_no_normalize.json`
- `src/rl/configs/simple_process_reward_no_backslice_strong_penalty.json`

队列与运行入口：

- Exp3–Exp7：`src/rl/experiments/run_gated_process_ablation_queue_table_rl.sh`
- Exp8–Exp9：`src/rl/experiments/run_gated_rank_followup_queue_newgnn.sh`
- Exp10–Exp11：`src/rl/experiments/run_rank_score_followup_queue_table_rl.sh`

正式评测结果位于服务器：

```text
/home/dengyan/tabular_rl_outputs/eval_runtime_version36_20260728/data/results/
```

每个已完成目录包含：

- `accuracy.json`
- `valid_rate.json`
- `avg_steps.json`
- `per_question_result.json`
- `trajectory.json`
- `trajectory_entropy.json`
- `action_distribution.json`
- `all.jsonl`
- `manifest.json`

训练 checkpoint 目录包含 `rollouts.jsonl`、`run_manifest.json`、`checkpoint-23/` 和
`final/`。Exp8/Exp9 位于 NewGNN 的同名输出根目录，其余位于 table_rl。

## 14. 最终判断

现阶段最合理的论文叙事不是“Process reward 已经全面战胜 SFT2”，而是：

> 旧的 dense credit 会损害首轨迹稳定性；去除事后 BackSlice 后，单纯的 process
> shaping 仍不足以恢复性能。把同题正确/错误 trajectory 的偏好显式加入目标后，
> pass@1、覆盖率或总正确轨迹可以分别恢复到 SFT2 附近甚至出现小幅正向信号。
> 进一步将 Rank 限定到工具动作、只对可验证坏动作施加负更新，并使 score mask 与
> update mask 一致，可以控制错误 credit；但在 300 题、单 seed 下，尚无方法统计上
> 显著超过 SFT2。下一阶段的关键不是继续扩大超参数搜索，而是完成 Exp11、扩大评测
> 到完整 BIRD、增加训练 seed，并用 fixed-prefix action margin 验证真正的工具策略
> 变化。
