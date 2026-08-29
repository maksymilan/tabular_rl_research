# Qwen3-8B v26 first32/K8 状态前缀结构审计

日期：2026-08-28

## 结论

本次小型实验支持继续研究“同状态动作对比”，但还不支持直接启动这种 RL。

- 输入是 Qwen3-8B Atomic version26 SFT1 checkpoint-560 在 BIRD-train representative
  first32 上的既有 K=8、零更新 rollout，共 256 条；没有使用 BIRD-dev 1534。
- 原始 256/256 条轨迹的第一轮工具都是 `describe_table`。按既有训练门禁排除 7 条
  `generation_length` 后，可用的 249/249 条仍然全部如此。
- 但是“工具名相同”不等于“动作相同”。保守地规范化
  `describe_table.tables` 的无语义顺序后，只有 9/32 题的全部可用 rollout 第一轮
  `tool + full arguments` 完全一致；其余 23/32 题使用了 2--6 种不同参数。
- 同题第一轮 modal 完整动作平均覆盖 73.62% 的 rollout，说明即使不是 8 条全相同，仍可形成
  较大的相同动作子组。
- 使用最严格的实际 policy prompt token ids 定义决策状态时，31/32 题在第一轮之后仍有重复状态，
  31/32 题存在重复状态下的不同 next action。
- 11 个 mixed-outcome 题中，9 个在第一轮之后至少有一个“同状态、不同完整动作、后续结果有差异”的
  候选 anchor；共有 16 个非初始候选状态，覆盖 mixed 题 586 个决策事件中的 71 个，即 12.12%。
- 但如果进一步要求至少两个不同动作各自都出现至少两次，而且经验成功率不同，则只剩 2 个非初始
  状态、14 个事件，全部来自同一题；只占 mixed 题决策事件的 2.39%，占全体事件的 0.80%。

因此结构可行性门通过：中间重复状态并不罕见，不能把该方法简单判为“只有初始状态可比较”。
统计支持门尚未通过：K=8 下，多数局部动作仅出现一次，终态差异可能来自更晚的动作，不能直接解释为
当前动作的因果优劣。

## 动作与状态身份

本审计明确区分以下概念。

### 下一动作身份

主动作签名是：

```text
parsed tool + complete arguments
```

- 忽略 JSON object key 顺序；
- 只把已经证明无序的 `describe_table.tables` 排序；
- 其余 list 顺序全部保留；
- 不按工具名合并不同参数；
- 不读取模型 reasoning 来判断动作正确性。

例如：

```json
{"tool":"describe_table","arguments":{"tables":["A"]}}
```

和：

```json
{"tool":"describe_table","arguments":{"tables":["B"]}}
```

是两个不同动作；而 `tables=["A","B"]` 与 `tables=["B","A"]` 在主动作签名中视为
同一个 schema-observation 动作，同时在 authored-action 统计中保留两种原始顺序。

### 三档状态身份

| 档位 | 定义 | 角色 |
|---|---|---|
| `tool_prefix` | 之前的工具名序列 | 宽松上界，不可直接用于训练 |
| `exact_environment_prefix` | 之前完整 authored action 与 Harness feedback | 环境执行前缀 |
| `strict_policy_prompt` | 当前轮实际 `prompt_ids` 的 SHA-256 | 最严格、最接近真实策略条件 |

下一动作的分支数始终按规范化后的完整动作计算，不按工具名计算。

## 数据身份

| 项目 | 数值 |
|---|---:|
| 数据 split | BIRD-train |
| 问题 | 32 |
| 原始轨迹 | 256 |
| 训练可用轨迹 | 249 |
| 可用正确 / 错误 | 181 / 68 |
| mixed-outcome 问题 | 11 |
| 可用决策事件 | 1,759 |
| mixed 题决策事件 | 586 |
| 平均每轨迹决策轮数 | 7.064 |
| 模型调用 | 0 |
| optimizer update | 0 |
| BIRD-dev 使用数 | 0 |

7 条未进入结构主统计的轨迹均是原 probe 已记录的 `generation_length`：保存的
`policy_turns` 比完整 assistant message 多一个截断轮次。该排除与原 vanilla-GRPO probe 门禁一致。

输入轨迹 SHA-256：

```text
312961a1695f5b24100d81bd62f432bcf6b5994abaa85df66320c34a16e75a84
```

## 第一轮前缀

### 只看工具名

32/32 题的全部可用 rollout 第一轮都是 `describe_table`。如果只看工具名，会错误地认为第一轮动作
完全一致。

### 加入完整参数

| 每题第一轮规范化完整动作数 | 题数 |
|---:|---:|
| 1 | 9 |
| 2 | 8 |
| 3 | 9 |
| 4 | 3 |
| 5 | 2 |
| 6 | 1 |

若保留 `describe_table.tables` 的原始 authored 顺序，只有 3/32 题的八条第一动作全部一致；证明
无序字段规范化是必要的。但即使做了这一项保守规范化，仍有 23/32 题的第一轮参数实质不同。

要求同题全部 rollout 共享前缀时：

| 前缀口径 | 至少共同 1 轮 | 至少共同 2 轮 |
|---|---:|---:|
| 工具名 | 32/32 | 10/32 |
| 规范化完整动作 | 9/32 | 1/32 |
| authored action + exact feedback | 3/32 | 0/32 |

“全部 K 条共享”是过强条件。局部 advantage 应在同题内对共享同一状态的 rollout 子组计算，而不是
要求整组八条拥有完全相同前缀。

## 严格 policy-state anchor 覆盖

以实际 `prompt_ids` 定义状态，统计结果为：

| Anchor 类型 | 非初始状态数 | 涉及题数 | 覆盖事件 | 全体事件占比 |
|---|---:|---:|---:|---:|
| 重复状态 | 141 | 31 | 441 | 25.07% |
| 重复状态且 next action 不同 | 105 | 31 | 343 | 19.50% |
| 再要求 continuation 结果有正有负 | 16 | 9 | 71 | 4.04% |
| 再要求至少两个动作各有 `n>=2` 且成功率不同 | 2 | 1 | 14 | 0.80% |

第三行的 71 个事件全部位于 mixed 题，在 mixed 题 586 个事件中占 12.12%。9/11 个 mixed 题存在
至少一个这样的非初始候选 anchor，说明 boundary cohort 比代表性随机 cohort 更适合验证该方法。

但第四行的两个状态都来自 `bird_train_00805`：

1. depth 1 的两个 `condition_filter` 仅在 `return_columns` 是否额外保留 `Match_Id` 上不同，经验成功率
   为 5/6 与 2/2；
2. depth 2 比较 `condition_filter` 与 `join_tables`，经验成功率为 3/4 与 2/2。

这些只是最小重复支持，不是显著的动作因果证据。两个动作可能都合理，成功率差异也可能来自后续
决策。不能因它们满足 `n>=2` 就直接发放强局部 advantage。

## 当前研究判断

本实验改变了两个判断：

1. **支持的判断**：Qwen3 v26 的确定性工具轨迹确实存在可复用的中间 policy states；在 mixed 题中，
   9/11 有非初始分支候选，因此该方向有继续审计的合理性。
2. **仍被拒绝的判断**：不能把一次 success continuation 和一次 failure continuation 的差异直接解释为
   当前动作的 credit。K=8 first32 的重复 action-level 支持仍太弱。

下一步不应启动 RL，而应把同一审计器运行在已经生成的 BIRD-train policy-boundary K8 池上。预先只看
以下门：

- 非初始 strict-policy informative anchor 的题覆盖率；
- 每个动作至少两条 continuation 后的有效事件覆盖；
- 相同状态—动作成功率在不同 seed/batch 中的符号稳定性；
- 去掉只改变输出支持列、无序字段和其它确定性等价动作后的覆盖。

只有覆盖和跨 seed 稳定性同时成立，才值得实现 state-conditioned advantage；否则该方向应停止在结构
诊断，不进入优化器。

## 产物

- 审计实现：`src/rl/diagnostics/audit_state_conditioned_prefix_structure.py`
- 测试：`src/rl/diagnostics/test_audit_state_conditioned_prefix_structure.py`
- 完整 JSON：`data/results/qwen3_v26_first32_k8_prefix_structure_20260828/audit.json`
- 冻结输入：同目录 `trajectories.jsonl` 与 `source_manifest.json`

新增测试 2/2 通过。
