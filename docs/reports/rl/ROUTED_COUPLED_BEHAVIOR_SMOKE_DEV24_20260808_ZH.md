# 路由式推理—动作耦合奖励：5 步训练后的真实行为测试（2026-08-08）

## 测试目的

前一项冻结 token 验证已经表明，奖励路由后可以把联合 log-prob 的奖励方向一致率从 Exp16/Exp17 的约 50% 提高到 90% 以上。本测试继续验证下一段因果链：这些 log-prob 和 adapter 参数变化是否足以改变未见问题上的真实 greedy 工具闭环。

本测试不是准确率结论，也不是 60/120 题候选训练。

## 训练臂

两个 adapter 都从同一个正式 SFT2 `checkpoint-1682` 独立初始化，使用相同的 5 个 BIRD-train 问题和相同题序，分别训练 5 个 optimizer steps：

- routed-coupled `LR=1e-6`；
- routed-coupled `LR=4e-6`。

训练规则与冻结 token 验证完全相同：只奖励关键 evidence、backslice 和正确 terminal，只惩罚 `severe_local_error`，普通正确步骤和无法局部归因的失败步骤为中性；联合分数为 `0.5 * mean(think logp) + mean(action logp)`，rank 关闭。

保存版训练的 loss、梯度范数、裁剪状态、adapter 位移和 Δlogp 指标与第一次无保存运行完全一致。

## 固定评测集与协议

- 评测集：24 道未参与训练的 BIRD-dev 问题。
- 来源：预先冻结的 equal-difficulty 300 池。
- 选择：固定 seed `20260808`，在 simple/moderate/challenging 内分别随机取 8 道；选择不使用任何模型结果。
- 三臂：历史完整 SFT2 greedy、`LR=1e-6`、`LR=4e-6`。
- 协议：atomic version36、temperature `0`、top_p `1`、greedy@1、max_steps `30`、bird-set、logprobs20、动态并发。
- 两个候选均完成 24/24，索引、采样参数、协议和结果文件通过一致性校验。

## 汇总结果

| 模型 | 正确 | 合法终止 | 平均步骤 | simple | moderate | challenging | 相邻完全重复调用 |
|---|---:|---:|---:|---:|---:|---:|---:|
| SFT2 | 12/24 | 20/24 | 10.625 | 4/8 | 3/8 | 5/8 | 0 题 / 0 次 |
| routed `1e-6` | 9/24 | 14/24 | 9.833 | 5/8 | 2/8 | 2/8 | 2 题 / 2 次 |
| routed `4e-6` | 11/24 | 17/24 | 11.375 | 5/8 | 3/8 | 3/8 | 1 题 / 1 次 |

### 相对 SFT2 的逐题变化

| 候选 | gains | regressions | net | exact p | legal gains/regressions/net | legal exact p |
|---|---:|---:|---:|---:|---:|---:|
| `1e-6` | 1 | 4 | -3 | 0.3750 | 1 / 7 / -6 | 0.0703 |
| `4e-6` | 2 | 3 | -1 | 1.0000 | 2 / 5 / -3 | 0.4531 |

`4e-6` 相对 `1e-6` 为 3 gains、1 regression、net `+2`，exact `p=0.6250`；合法性为 4 gains、1 regression、net `+3`，exact `p=0.3750`。样本很小，不能解释为稳定提升。

## 策略是否真的变化

答案是肯定的：

- `1e-6` 相对 SFT2：完整 action 序列 24/24 不同，第一步 action 5/24 不同；
- `4e-6` 相对 SFT2：完整 action 序列 21/24 不同，第一步 action 3/24 不同；
- `4e-6` 相对 `1e-6`：完整 action 序列 20/24 不同，第一步 action 3/24 不同。

第一步变化较少、后续序列变化很多，表明多数分叉发生在模型看到环境反馈之后，而不是只改变开局语法。由于 greedy 推理的底层数值执行仍可能存在极小的批处理非确定性，完整序列变化率不应解释为 100% 的语义策略改变；paired 成败、合法性和第一动作变化是更稳健的证据。

因此，“Exp16/17 没有效果只是因为模型状态完全没变”这一假设可以排除。当前路由式目标在 5 步后已经足以改变真实闭环行为。

## 为什么真实指标没有同步改善

本次短训练中的负奖励位移仍明显大于正奖励位移：

- `1e-6`：明确局部错误的联合平均绝对 Δlogp 为 `0.01527`，正奖励转移为 `0.00264`，约 5.8 倍；
- `4e-6`：分别为 `0.07752` 与 `0.00942`，约 8.2 倍。

虽然负奖励只有 11 个训练转移，单个负转移造成的策略位移远大于单个正转移。这与 held-out 合法率下降、两个候选重新出现相邻重复调用相吻合。具体重复调用为：

- `1e-6`：题 116 重复 `scalar_compute(percent_change)`；题 775 重复同一个 `join_tables`；
- `4e-6`：题 446 重复同一个 `read_subtable`；
- SFT2：该 24 题中没有相邻完全重复调用。

这不是“奖励和惩罚继续 1:1 抵消”，而更像是路由后负奖励的单位强度过大：策略已经动了，但局部惩罚可能同时压低了邻近的合法恢复行为。

## 当前能下的结论

1. 奖励路由和推理—动作耦合可以在很少步骤内一致改变 log-prob，并传导到真实工具闭环。
2. 因而 Exp16/17 的近似不变不能用于否定规则式过程 RL。
3. 目前 5 步版本没有优于 SFT2：准确率和合法率均下降，且引入少量相邻重复调用。
4. `4e-6` 比 `1e-6` 在该切片上更接近 SFT2，但它不是显著结果，也不能说明继续增大学习率会更好。
5. 下一项最小控制变量应当是限制负奖励单位强度，而不是先扩大题量：保持相同路由、think/action 耦合和学习率，对 `severe_local_error` 做负 advantage cap 或把负项系数降至正项的约 `0.25–0.5`，再观察合法率与 paired regressions 是否恢复。

## 产物

- 训练指标：`artifacts/rl/routed_coupled_behavior_smoke_20260808/train5_lr1e6_lr4e6.json`
- 24 题 paired 汇总：`artifacts/rl/routed_coupled_behavior_smoke_20260808/dev24_paired_summary.json`
- 固定评测索引：`artifacts/rl/routed_coupled_behavior_smoke_20260808/dev24_indices.json`
- 训练脚本：`src/rl/diagnostics/run_routed_coupled_reward_smoke.py`
- paired 汇总脚本：`src/rl/diagnostics/summarize_routed_coupled_behavior_smoke.py`
- 远端根目录：`/home/dengyan/tabular_rl_outputs/diagnostics/routed_coupled_behavior_smoke_20260808`
- 远端候选评测：`/home/dengyan/tabular_rl_outputs/eval_runtime_version36_20260728/data/results/routed_coupled_train5_lr{1,4}e6_version36_dev24_greedy_20260808`

