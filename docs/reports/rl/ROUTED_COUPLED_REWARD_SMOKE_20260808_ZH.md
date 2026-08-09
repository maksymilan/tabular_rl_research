# 路由式推理—动作耦合奖励五步验证（2026-08-08）

## 目的

本验证只回答一个短问题：Exp16/Exp17 的策略几乎不变，究竟主要是“规则奖励本身无效”，还是奖励标签冲突、全响应广播和有效信号稀释使梯度不能一致推动策略。

它不是候选模型训练，不保存模型 checkpoint，不做 rollout，也不运行 BIRD-dev 全量评测。

## 固定数据与对照

- 起点：同一个正式 SFT2 `checkpoint-1682`；两个学习率分支都从完全相同的 adapter 参数独立重置。
- 数据：复用 Exp16 的冻结 mixed60 token，不重新生成轨迹。
- 训练题顺序：`3044, 4050, 1626, 5040, 5027`，每题一次 optimizer step，共 5 步。
- 独立保持性题：`4692`，只测其 8 个正确轨迹转移的 log-prob 变化，不参与训练。
- 有效训练转移：49，其中明确正奖励 38、明确局部错误负奖励 11。
- 其余 59 个训练转移置为中性：56 个 `incorrect_other_clean` 与 3 个 `correct_other_clean`。
- 学习率：`1e-6` 与 `4e-6`；weight decay `0.01`，梯度裁剪 `1.0`。
- rank 项关闭，以隔离规则过程奖励本身。

## 奖励路由和损失

只保留语义较明确的标签：

- 正奖励：`correct_key_evidence`、`correct_key_backslice`、`correct_terminal`；
- 负奖励：`severe_local_error`；
- 中性：普通正确步骤和“最终失败但没有明确局部错误”的合法步骤。

每个转移的联合分数为：

```text
joint_score = 0.5 * mean_logp(think tokens)
            + 1.0 * mean_logp(action tokens)
```

因此推理与工具调用共同训练，但分别按各自 span 取均值，避免较长 `<think>` 仅凭 token 数量淹没 action。每题损失只按当题 routed-active 转移数归一化：

```text
loss = mean_active[-advantage * joint_score]
```

这是奖励方向和强度的 first-order manipulation check，不是对 Exp16/Exp17 PPO/ratio-clipping 训练循环的逐位复现。

## 结果

下表的训练方向一致率由结果 JSON 中 57 个测量转移扣除 8 个未参与训练的保持性转移后得到；即只统计 49 个实际训练转移。

| 指标 | LR=1e-6 | LR=4e-6 |
|---|---:|---:|
| 训练联合分数奖励方向一致 | 44/49 = 89.80% | 48/49 = 97.96% |
| 训练 think 分数奖励方向一致 | 44/49 = 89.80% | 48/49 = 97.96% |
| 训练 action 分数奖励方向一致 | 38/49 = 77.55% | 40/49 = 81.63% |
| 全部测量转移联合平均绝对 Δlogp | 0.005076 | 0.022558 |
| 全部测量转移 think 平均绝对 Δlogp | 0.004209 | 0.015514 |
| 全部测量转移 action 平均绝对 Δlogp | 0.002997 | 0.014820 |
| raw adapter 更新范数 / 初始范数 | 0.04787% | 0.19136% |
| 保持性题 8 个正确转移联合平均 Δlogp | +0.002734 | +0.008639 |
| 保持性题联合分数上升 | 8/8 | 8/8 |
| 触发 gradient clip 的 step | 1/5 | 1/5 |

`4e-6` 相对 `1e-6`：

- raw adapter 相对变化恰好放大约 4.00 倍；
- 联合平均绝对 Δlogp 放大 4.44 倍；
- think 平均绝对 Δlogp 放大 3.69 倍；
- action 平均绝对 Δlogp 放大 4.95 倍。

明确局部错误的 11 个负奖励转移在 `4e-6` 下表现为：think 11/11 向下、action 10/11 向下、联合分数 11/11 向下。正确 terminal 的 10 个测量转移在两个学习率下，其 think、action 和联合分数均为 10/10 向上。

## 与 Exp16/Exp17 的关系

冻结前缀诊断中，Exp16/Exp17 的响应转移奖励方向一致率分别只有 49.60% 和 49.51%，与随机方向近似；完整 60 步后 response mean absolute Δlogp 也只有约 0.00123。此次路由后，5 步训练转移的联合方向一致率已经达到 89.80%/97.96%，且学习率能够近似按比例放大策略变化。

此外，路由后第一个训练题的梯度范数为 1.387，触发 `clip=1`；历史 Exp16/17 的 60 步中没有一步触发裁剪。这说明旧目标不仅学习率偏小，还被大量弱标签、互相矛盾的正负标签以及按全部转移归一化显著稀释。

## 初步判断与边界

短验证支持以下判断：

1. 不能据 Exp16/Exp17 的小变化断言规则式 RL 无效；旧实验首先没有把奖励稳定地转化为同方向策略更新。
2. 主要修复点是奖励路由：不再把“最终失败”广播成每个合法步骤的负标签，只惩罚可定位的局部错误。
3. 推理与 action 应联合训练。当前结果中 think 方向一致性并不低于 action，说明仅训练 action 会丢掉实质性的策略变化。
4. 在奖励路由后，`4e-6` 已足以在 5 步内产生明显而非数值噪声级别的变化；继续简单提高学习率之前，应先扩大到小规模行为评测，观察是否开始牺牲正确轨迹与合法率。
5. 保持性检查只有一个问题、8 个冻结正确转移，只能排除立即崩坏，不能替代 held-out accuracy、合法率和重复调用率评测。

因此下一阶段值得验证的对象不是原样 Exp16/17，而是“明确标签路由 + think/action 分 span 耦合 + active normalization”的小规模规则奖励版本，并设置 `1e-6`/`4e-6` 对照。只有看到真实 rollout 行为和 held-out 指标随联合 Δlogp 改变，才能决定是否扩大到 60/120 题。

## 产物

- 实验脚本：`src/rl/diagnostics/run_routed_coupled_reward_smoke.py`
- 完整结果：`artifacts/rl/routed_coupled_reward_smoke_20260808/five_step_lr1e6_lr4e6.json`
- 远端只读结果：`/home/dengyan/tabular_rl_outputs/diagnostics/routed_coupled_reward_smoke_20260808/five_step_lr1e6_lr4e6.json`

