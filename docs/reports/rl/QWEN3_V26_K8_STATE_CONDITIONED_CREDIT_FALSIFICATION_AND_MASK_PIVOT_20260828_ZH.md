# Qwen3-8B v26 K8 状态条件 credit：离线证伪与歧义屏蔽转向

日期：2026-08-28

## 结论

当前证据支持工具轨迹具有可利用的分段结构，但不支持在 K=8 下估计
`Q(state, full_action)` 并据此发放逐步正负 credit。

更准确的结论不是“step-level credit 在理论上无效”，而是：

1. 相同中间状态确实大量存在；
2. 完整动作的重复样本和跨运行稳定性不足；
3. leave-one-out 检验没有证明动作条件 continuation value 优于状态均值；
4. 因而不应把当前样本内的动作成功率差直接当成局部因果 credit。

本轮得到一个更保守的可实验转向：保留 binary terminal GRPO，只屏蔽“同一确定性状态、同一完整
动作，却同时通向正确和错误终局”的轮次。该方法不判断哪个工具更好，也不新增 Gold SQL
过程奖励；它只删除当前数据无法辨认的整轨迹 credit，暂称
**State-Action Ambiguity-Masked GRPO（SAAM-GRPO）**。

SAAM-GRPO 仍是待验证假设，不是已证明优于 vanilla GRPO 的算法。下一步需要先与用户确认是否接受
这一研究转向，再实现 matched 训练 arm；当前没有启动任何新模型调用或 optimizer update。

## 数据边界

本报告只使用 BIRD-train，不使用 BIRD-dev 1534 训练。

### Policy-boundary K8 池

- 既有 S1 screening 实际完成 568/600 个 K8 group；没有补跑缺失 32 题。
- 其中 193 个 group 满足原始 8/8 `process_update=true` 且 terminal outcome mixed，共 1,544 条轨迹。
- 为保证逐轮前缀真实可重建，28 条含无法解析或不完整 carrier 的轨迹被 fail-closed 排除。
- 主统计剩余 1,516 条轨迹、193 题、11,645 个决策事件；排除后 192 题仍为 mixed。
- 正确/错误为 1,022/494；其中 mixed 题为 1,015/494，另有一题因排除后只剩 7 条正确轨迹。

冻结合并输入 SHA-256：

```text
e71a8924ad5494b2c3a0a91329014d50b9f6c165286361139d42944f5415c698
```

### 两次独立引擎重跑

额外发现两个既有 representative600 训练尝试的第一个 global step。两次均从同一个 SFT1
checkpoint 开始，题顺序相同，前 30 题各 K8，`policy_global_step=0`，因此在任何 optimizer
update 之前产生。它们不是不同随机 seed，而是同一 seed 的两个独立引擎运行，最多只能作为较弱的
复现性检查。

|  | Run A | Run B |
|---|---:|---:|
| 原始轨迹 | 240 | 240 |
| 严格解析轨迹 | 239 | 231 |
| 训练可用轨迹 | 226 | 222 |
| 正确 / 错误 | 189 / 37 | 194 / 28 |
| mixed 题 | 10 | 8 |

输入 SHA-256：

```text
Run A  1866940e5c0cde2f26780b5c7b2cd294c24f4f9bfe09054766df21f57109cd58
Run B  df418c4def1f331aeff7518766d2c9d94852ecc111ea806fc3901e6e9460dc98
```

仓库和两台机器上没有找到“同一初始策略、同一题集、不同 seed”的中间状态 rollout。训练过程中
策略已经变化的 rollouts 没有被伪装成跨 seed 验证。

## 状态与动作身份

动作身份始终是：

```text
parsed tool + complete arguments
```

- 忽略 JSON object key 顺序；
- 只规范化已证明无序的 `describe_table.tables` 顺序；
- 其它 list 顺序与所有动作参数均保留；
- 不按工具名合并不同参数；
- `return_columns` 去除和 tool-only 只用于敏感性分析，不用于主动作身份。

状态报告两种口径：

1. `strict_policy_prompt`：保存的实际 `prompt_ids`；旧 flat rollout 没有 token ids 时，用完全相同的
   已保存 `model_input` 作为更保守的相等键；
2. `exact_environment_prefix`：此前所有完整 authored action 与 Harness feedback 的确定性前缀，
   不把模型 reasoning 当作环境事实。

第二种更符合本项目“工具状态作为过程评估载体”的叙事，但它是对真实 policy prompt 的状态聚合；
忽略历史 reasoning 是否会合并不同策略上下文，是 SAAM-GRPO 仍需显式控制的风险。

## 结构门：通过

在 193 个 boundary mixed group 的严格 policy-state 口径下：

| 指标 | 结果 |
|---|---:|
| 非初始重复状态 | 812 个 / 191 题 / 2,515 事件 |
| 非初始状态且下一完整动作有分支 | 615 个 / 191 题 / 2,028 事件 |
| 再要求 terminal outcome mixed | 288 个 / 153 题 / 1,141 事件 |
| 再要求至少两个动作各 `n>=2` 且经验成功率不同 | 33 个 / 32 题 / 233 事件 |

因此不能再说“除了初始 `describe_table` 外没有可比较前缀”。153/192 个 mixed 题有至少一个
非初始、同状态、不同完整动作、结果有差异的候选。

但是最严格的重复动作支持只覆盖全部决策事件的 2.00%，且 33 个状态中 31 个在 depth 1、2 个在
depth 2。它不足以可靠估计深层局部动作 Q。

动作变体敏感性也不能忽略：288 个非初始 informative 状态中，53 个只在去掉
`return_columns` 后合并为同一动作；33 个有重复动作支持的状态中有 5 个属于这一类。报告没有将
这些动作强行视为语义等价，因为输出列会影响后续可用状态。

## 预测门：未通过

为了避免用当前样本自己的结果证明当前动作，针对每个 noninitial、branching、mixed state 做了
leave-one-out：

- 只有完整动作在该状态至少出现两次时才预测；
- 从状态均值与动作均值中同时移除被预测轨迹；
- 使用 Beta(1,1) 平滑；
- 评价动作条件预测相对只使用状态均值的 Brier 和 log-loss；
- 置信区间按题聚类 bootstrap 5,000 次。

共 155 个状态、557 个留出样本、117 个题簇：

| 动作值身份 | Δ Brier（action−state） | 95% CI | Δ log-loss | 95% CI |
|---|---:|---:|---:|---:|
| 完整动作 | +0.00408 | [−0.00634, +0.01362] | −0.00576 | [−0.02788, +0.01490] |
| 去掉 `return_columns` | +0.00059 | [−0.00864, +0.00939] | −0.00992 | [−0.03056, +0.00968] |
| tool-only | +0.00171 | [−0.00657, +0.00884] | −0.00676 | [−0.02545, +0.00942] |

负数表示动作条件值更好。三个口径的两个 proper loss 都没有形成一致方向，所有区间都跨 0。
`exact_environment_prefix` 在这个 boundary 池上得到完全相同的 155/557 样本与结果。

该检验本身仍偏乐观：193 题是用同一批 terminal outcomes 选出的 mixed group，所以这里只能作为
探索性否证，不能作为确认性泛化结果。即使在这个偏有利的数据上仍未观察到稳定收益，因此当前不应
进入 local-Q RL。

## 弱复现门：未通过

两个 initial-policy 重跑的 mixed 题集合分别为 10 和 8，交集 7、并集 11，Jaccard 为 0.636。

要求同一初始状态、同一对完整动作在两个运行中每个动作都至少出现两次时：

- 共 7 个可比动作对；
- 5 个至少一边经验成功率相同，不能确定排序；
- 剩余 2 个中，1 个方向一致、1 个方向翻转。

按确定性环境状态比较非初始动作时只剩 4 个动作对：3 个至少一边没有差异，只有 1 个在两边都非零
且方向一致。严格 policy-input 相等则没有任何非初始可复核动作对。

这不是对跨 seed 稳定性的充分检验，但它明确反对把当前 K8 内的偶然成功率排序当成可靠 step
credit。

## 为什么转向歧义屏蔽

在 vanilla result-only GRPO 中，同一轨迹 advantage 会广播到该轨迹的所有工具轮次。于是相同
`(state, full_action)` 若同时存在于正确和错误轨迹中，这一个动作轮次会收到相反方向的更新；差异
其实发生在它之后。

SAAM-GRPO 的最小定义为：

\[
M(s,a)=\mathbf{1}\{\text{该 on-policy K 组中 }(s,a)\text{ 的后继同时有 }R=0,1\},
\]

\[
\widetilde A_{i,t}=
\begin{cases}
0,&M(s_{i,t},a_{i,t})=1,\\
A_i^{\mathrm{GRPO}},&\text{otherwise}.
\end{cases}
\]

它不做以下事情：

- 不估计 `Q(s,a)`；
- 不给工具定义固定分数；
- 不比较不同工具的全局均值；
- 不读取 Gold SQL、Gold path 或模型 reasoning；
- 不把错误轨迹整条删除。

它只测试一个更窄的命题：删除无法由当前 state-action 区分的 credit，是否优于把终局 advantage
广播到所有轮次。

## 屏蔽覆盖不是零

在 193 个 boundary group 上：

| 指标 | 结果 |
|---|---:|
| mixed `(state, full_action)` 子组 | 357 |
| 涉及题 | 174/192 mixed 题 |
| 被标为歧义的决策事件 | 1,525 / 11,645 = 13.10% |
| 其中非初始事件 | 466 / 11,645 = 4.00% |
| 该事件持有的 response tokens | 295,440 / 2,498,058 = 11.83% |
| 按 `trajectory_token_mean` 计算的绝对 policy coefficient mass | 17.27% |

最后一项是 `|standardized advantage| × 该轮 response-token/整轨迹 token` 的质量占比，不是实际
gradient norm；batch 公共缩放因子在分子分母中抵消。

错误轨迹仍然训练：494 条 mixed-task 错误轨迹全部至少保留一个未屏蔽轮次；其中 367 条只屏蔽
部分歧义轮次，127 条完全不受影响。1,015 条正确轨迹也全部保留至少一个未屏蔽轮次。没有轨迹因
所有轮次都歧义而被整个删除。

在没有按 mixed outcome 选择的两个 representative 初始 batch 上，环境状态口径的歧义事件占比
分别为 8.08% 和 5.15%，非初始占比分别为 3.38% 和 1.74%。因此 13.10% 的 boundary 数字确有
选择放大，但歧义 credit 不是只存在于 outcome-selected 池。

## 当前决策与需要讨论的问题

建议冻结以下判断：

1. **停止当前 local-Q 方案。** 不使用 K8 样本内动作成功率产生正负 step advantage。
2. **保留工具分段研究。** 结构门已经通过，不能退回“工具状态没有作用”的结论。
3. **下一候选改为 SAAM-GRPO。** 它是 result-only GRPO 的 credit-removal ablation，而不是新的
   手工 process reward。
4. **不要立即跑正式 RL。** 需要先确认：用户是否接受“先验证删除错误 credit，而不是直接构造
   正确 credit”作为当前研究问题。

若接受，最小后续工作应为：

1. 在独立诊断模块实现只读的 transition mask 和完整审计，不改变默认 vanilla 路径；
2. 用冻结 rollout 验证相同 `(state, action)` 得到一致 mask、所有任务 advantage/identity 不被
   篡改、无 Gold 字段读取；
3. 设计同 checkpoint、同 train-only cohort、同 seed/预算的 vanilla 与 SAAM matched 小实验；
4. 小实验只决定是否值得扩展，不使用 BIRD-dev 1534 做训练或调参。

## 产物

- 主审计：`src/rl/diagnostics/audit_state_conditioned_prefix_structure.py`
- 重跑比较：`src/rl/diagnostics/compare_state_conditioned_action_value_reruns.py`
- 主测试：`src/rl/diagnostics/test_audit_state_conditioned_prefix_structure.py`
- 重跑测试：`src/rl/diagnostics/test_compare_state_conditioned_action_value_reruns.py`
- Boundary JSON：`data/results/qwen3_v26_boundary193_k8_state_prefix_20260828/audit.json`
- 重跑 JSON：`data/results/qwen3_v26_initial30_rerun_state_prefix_20260828/`

相关测试 7/7 通过。
