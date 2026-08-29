# Qwen3-8B Atomic v26 SAAM-GRPO：统一目标、训练实现与离线门实验

日期：2026-08-28

## 1. 结论

SAAM-GRPO 应当定义成 **GRPO 轨迹优势上的保守分步门控**，而不是新的逐工具奖励：

1. 终局奖励仍然只有 Harness 判定的 binary correctness；
2. 每题 K 条 on-policy 轨迹仍按 vanilla GRPO 计算轨迹优势；
3. 若同题、同确定性执行前缀、同完整工具动作同时出现在正确和错误轨迹中，
   则该步的轨迹优势置零；
4. 其余步骤保留原 GRPO 优势，掩码后不重新归一化。

这使方法回答一个很窄但可验证的问题：**当一个工具动作自身无法解释后续成功或失败时，
是否应停止把整条轨迹的结果反向归因给它？**

现有 K=8 冻结训练轨迹证明该门控有足够曝光量且能够实现，但尚未证明它比 vanilla
0/1 GRPO 提高最终准确率。下一步必须做匹配的在线 A/B 小门实验。

## 2. 统一数学目标

对问题 (x) 的 K 条可优化轨迹，终局奖励为 (R_i\in\{0,1\})，先计算普通 GRPO 优势：

\[
A_i=\frac{R_i-\operatorname{mean}(R)}
{\operatorname{std}(R)+\epsilon}.
\]

对第 (t) 步定义：

- (s_{i,t})：该题此前所有已解析完整动作与 Harness 结构化反馈构成的确定性执行前缀；
- (a_{i,t})：当前 `tool + complete arguments`；
- (C(x,s,a))：同一 K 组中执行该状态动作对的轨迹终局结果集合。

掩码为：

\[
m_{i,t}=
\begin{cases}
0,& C(x,s_{i,t},a_{i,t})=\{0,1\}\\
1,& \text{otherwise}.
\end{cases}
\]

最终分步优势只有：

\[
\widetilde A_{i,t}=m_{i,t}A_i.
\]

训练仍使用原来的 clipped GRPO surrogate。若省略 clip 等实现细节，其策略项可写为：

\[
\mathcal L_{SAAM}
=-\frac{1}{Z_0}\sum_{i,t}
w_{i,t}\,m_{i,t}A_i\log\pi_\theta(a_{i,t}\mid s_{i,t}),
\]

其中 (Z_0) 和 trajectory-token 权重都取自掩码前的 vanilla batch。不能用存活步数重新
计算 (Z)，否则会把“删除矛盾 credit”偷偷变成“放大其余 credit”。

这不是局部 (Q(s,a)) 估计，也没有定义某类工具天然更高的分数。因此原方案担心的
“某个工具平均奖励总比另一个工具高”不再是目标函数的组成部分。

## 3. 状态与动作匹配规则

主实验使用严格规则：

- 只在同一个 `example_index` 内匹配；
- 状态只读取此前的已解析动作与 Harness `tool_output/error_event`；
- 不读取 gold SQL、gold path、answer rows 或模型 `<think>` 作为奖励依据；
- 动作必须匹配工具名和完整参数；
- JSON object key 顺序忽略；
- 只有已证明无序的 `describe_table.arguments.tables` 做排序；
- 其他 list 顺序全部保留；
- 当前动作不可解析时，该步及其后继状态 fail closed：保留 vanilla GRPO 信号，但不参与
  SAAM 匹配；
- `describe` 超集匹配不是主规则，只能作为独立消融。

当前“状态”实际是 exact executed prefix。它比合并等价 Harness snapshot 更严格：可能漏掉
不同历史到达同一环境状态的情况，但不会因为激进状态合并制造错误匹配。冻结 v26 没有在
每个成功动作前记录统一 state digest，因此当前阶段不修改冻结 Harness 身份。

## 4. 训练实现边界

实现增加了默认关闭的 `--credit-assignment trajectory|saam-strict`：

- `trajectory` 保持 vanilla 路径；
- `saam-strict` 只允许 binary result-only、`train_turns=all`、KL=0、无 ranking loss；
- 掩码发生在零优势 transition 裁剪之前；
- policy reduction 始终使用掩码前 transition/trajectory 数；
- 运行时逐值断言所有未掩码 policy coefficient 与 vanilla 完全相等；
- manifest 和 implementation lock 记录 `credit_assignment`；
- 每个 batch 记录 mask 数、首步/非首步分布、token 数、移除的绝对系数质量和整轨迹
  是否失去信号。

核心实现：

- `src/rl/frameworks/trl/state_action_ambiguity.py`
- `src/rl/frameworks/trl/transition_grpo.py`
- `src/rl/frameworks/trl/run_transition_grpo.py`
- `src/rl/diagnostics/audit_saam_training_mask.py`

## 5. 冻结 K=8 池的生产同构回放

输入为 193 个 BIRD-train 问题、每题 K=8 的冻结策略轨迹：

- 输入：`selected193.mixed.jsonl.gz`
- SHA-256：`e71a8924ad5494b2c3a0a91329014d50b9f6c165286361139d42944f5415c698`
- 归约：`trajectory_token_mean`
- 审计回执：`saam_strict_training_mask_audit.json`
- 回执 SHA-256：`f9e3eddc7d773b2a0d79a051d8261f2bafda6c9fb3e52b9e3a10107474d10de9`

训练时真实 `process_update` 口径的结果：

| 指标 | 结果 |
|---|---:|
| 题数 / K | 193 / 8 |
| 可优化轨迹 | 1,544 |
| 正确 / 错误轨迹 | 1,036 / 508 |
| 可优化动作 | 11,959 |
| 可严格匹配动作 | 11,204 |
| fail-closed 不匹配动作 | 755 |
| ambiguous `(state, action)` 组 | 362 |
| 被置零动作 | 1,559（13.04%） |
| 首步 / 非首步置零 | 1,083 / 476 |
| 被置零 response token | 304,175 / 2,594,745（11.72%） |
| 移除绝对 policy coefficient 质量 | 15.65% |
| 完全失去信号的轨迹 | 0 |
| 仍有非零信号的错误轨迹 | 508 / 508 |
| 仍有非零信号的正确轨迹 | 1,036 / 1,036 |

深度分布为：depth0 1,083、depth1 341、depth2 112、depth3 19、depth4 4。因而 SAAM
主要消除共同开头，但不是简单的“永远忽略第一步”：有 476 个非首步动作被实际门控。

同一审计连续运行两次产生 byte-identical 回执，未掩码系数逐值一致性检查通过。

这与早期结构审计的 1,516 条轨迹 / 1,525 个 mask 口径不同。早期审计为保证统计结构完整，
整条排除了 28 条含不可解析或不完整 turn 的轨迹；实际 vanilla 训练器仍把这类可归因的模型
协议错误作为错误轨迹训练，只有非语义 runtime failure/timeout 才由 `process_update=false`
排除。生产同构口径因此保留全部 1,544 条轨迹，对不可解析步及其后继状态只禁止 SAAM
匹配，不删除其 vanilla 负向训练信号。这是 1,559 而不是 1,525 的原因。

### 5.1 同一步骤正负 credit 冲突

在上述 362 个 mixed state-action 组中，按实际 `trajectory_token_mean` 系数统计：

- vanilla 正向绝对系数质量：694.912；
- vanilla 负向绝对系数质量：802.108；
- ambiguous 总质量：1,497.019；
- 按每个 state-action 组分别计算的 two-sided conflict
  \(2\min(P_k,N_k)\)：1,021.980，占 ambiguous 质量的 68.27%；
- 多数方向残余 \(|P_k-N_k|\)：475.039；
- SAAM 对这些 mixed key 的直接终局 credit conflict：严格为 0。

因此 SAAM v1 确实消除了大量“语义上同一个工具动作同时被强化和抑制”的直接 credit。
同时它也删除了 475.039 的多数方向残余，这正是其保守性成本：当前先验证消除震荡是否有益，
以后才研究是否在有统计置信度时保留一部分多数方向信号。

## 6. 工具分布偏差审计

被置零动作按工具分布：

| 工具 | 被置零动作 | 该工具内部 mask 比例 |
|---|---:|---:|
| `describe_table` | 1,099 | 60.48% |
| `inspect_column` | 203 | 20.20% |
| `condition_filter` | 194 | 6.11% |
| `read_subtable` | 33 | 2.49% |
| `group_aggregate` | 13 | 1.56% |
| `join_tables` | 9 | 0.93% |
| `extreme_value_select` | 8 | 2.56% |

这种不均匀不等于给 `describe_table` 负奖励。它表示在相同开头，describe 同时通向正确和
错误续写，因此不应仅因终局不同而一边强化、一边抑制。置零后该动作保持 SFT anchor，
真正首次分叉的工具动作继续接受 GRPO 正负信号。

不过，共享参数仍可能间接改变工具频率。因此在线门实验必须把以下指标作为安全终点，而
不是再引入 per-tool reward normalization：

- 首步 `describe_table` 率与所描述表数；
- 合法终止率、工具错误率、平均动作数；
- 各工具调用分布及与 vanilla 的 JS divergence；
- 正确轨迹中的必要 perception 覆盖率；
- 非首步 SAAM 曝光率与完全失去信号的错误轨迹数。

## 7. 当前能成立和不能成立的结论

已成立：

1. SAAM 是一个单一、简化、确定性的目标，不需要 gold SQL 路径或人工逐工具打分；
2. 它仍属于 GRPO：轨迹间优势不变，只改变每个 segment 是否消费该优势；
3. K=8 现有轨迹中有 13.04% 动作和 15.65% 系数质量受到影响，信号不稀疏到无法实验；
4. 错误轨迹没有被整体丢弃，当前池中 508/508 仍训练其首次分叉后的错误动作；
5. 不需要 Monte Carlo tree search，也不额外 rollout 分支；复杂度只是 batch 内 hash/group。

尚未成立：

1. SAAM 比 vanilla 0/1 GRPO 提升 held-out denotation accuracy；
2. 置零共同动作一定优于让相反轨迹优势在期望上自行抵消；
3. 对整段 `<think> + tool call` 一起置零不会丢失有用的 reasoning credit；
4. 当前 exact-prefix 状态的保守 false negative 是否过多；
5. 单 seed 小实验不能构成统计结论。

因此论文叙事应是“contrastive prefix credit cancellation”，而不是“已经解决 agent 的 step
credit assignment”。如果在线 A/B 有收益，它才支持进一步研究更强的状态等价或局部 credit；
如果无收益，也能明确说明仅靠共同前缀消歧不够。

## 8. 下一步配对小门实验

先做机制 Gate，不直接启动大规模训练：

- 初始化：同一个 Qwen3-8B Atomic v26 SFT1 checkpoint-560；
- 训练数据：只用 BIRD-train 的 60 个冻结 boundary 身份，绝不使用 BIRD-dev 1534 训练；
- A：`credit_assignment=trajectory`；
- B：`credit_assignment=saam-strict`；
- 两臂其余完全一致：K=8、binary result-only、`trajectory_token_mean`、KL=0、无 rank、
  一次训练遍历、相同 seed/温度/采样和 optimizer；
- 规模：5 updates × 12 prompts × K8 = 每臂 480 条在线轨迹；
- 训练中机制门：mask 率非零、存在非首步 mask、未掩码系数一致、无完全失去信号的错误轨迹；
- 评测：使用与 train60 不重合的固定 holdout；BIRD-dev 只能作为 held-out evaluation，不能
  进入训练或 cohort 选择；
- 效果指标：paired `bird-set`、合法率、工具错误、平均步数、首步 describe 率和工具分布。

Gate60 只能决定是否值得扩展，不能单独宣称优于 GRPO。若机制审计失败立即停止；若机制
通过且无行为退化，再做多 seed 或更大训练身份的确认实验。

## 9. 当前执行状态

- 纯 SAAM 与训练接线已完成；
- 22 个纯 credit/transition 测试通过，其中 1 个 Torch 对齐测试因本机轻量环境无 Torch
  跳过；
- 3 个 experiment-config 约束测试在项目虚拟环境通过；
- 离线生产同构回放与确定性复跑通过；
- 2026-08-28 检查 NewGNN 时 8 张 RTX 3090 均处于 85%--100% 利用率，已有 4-GPU SFT
  任务运行中，因此没有抢占或启动 SAAM 在线训练。
