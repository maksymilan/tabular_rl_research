# Routed coupled RL：负奖励尺度小样本扫掠（2026-08-08）

## 目的与边界

本实验只回答一个局部问题：在已确认能够显著改变策略的 5 题 coupled-RL smoke 中，
`severe_local_error` 的负奖励是否过强，以及缩小负奖励能否改善真实 rollout 行为。

这不是完整 BIRD-dev 结论。下述 24 题已经参与超参数选择，因此是 **tuning set**，不能再作为
独立泛化集。最终候选必须在不相交 holdout 上复核后，才能决定是否扩大训练规模。

## 固定配置

- 初始化：Qwen2.5-Coder-7B-Instruct + SFT2 checkpoint-1682。
- 训练题：`3044, 4050, 1626, 5040, 5027`，固定顺序，共 5 个 optimizer steps。
- 学习率：`4e-6`；weight decay：`0.01`；rank 项：关闭。
- 联合策略分数：`0.5 * mean(think token logp) + 1.0 * mean(action token logp)`。
- 正奖励：`correct_key_evidence`、`correct_key_backslice`、`correct_terminal`。
- 负奖励：`severe_local_error`；本次只改变其尺度。
- 中性：`correct_other_clean`、`incorrect_other_clean`，不进入损失。
- 每题按 routed-active transition 均值归一化。
- 训练集中实际进入目标的 transition 为 38 条正、11 条负；另用 1 道独立正确题的 8 条
  transition 测量 retention，不参与训练。

扫掠值为 `negative_scale ∈ {0, 0.25, 0.5}`，并复用原始 `negative_scale=1` 结果作为对照。

## 训练侧结果

| negative scale | 第一步梯度范数 | 第一步触发 clip | adapter 位移 / 初始 adapter | 正样本 joint mean abs Δlogp | 负样本 joint mean abs Δlogp | retention joint Δlogp |
|---:|---:|:---:|---:|---:|---:|---:|
| 0 | 0.3277 | 否 | 0.1881% | 0.01948 | 0.00416 | +0.02119 |
| 0.25 | 0.3934 | 否 | 0.1878% | 0.01567 | 0.05988 | +0.01598 |
| **0.5** | 0.7139 | 否 | 0.1912% | 0.01244 | 0.07194 | +0.01183 |
| 1 | 1.3873 | **是** | 0.1914% | 0.00942 | 0.07752 | +0.00864 |

表中的正样本冻结评分包含 38 条训练正 transition 和 8 条 retention 正 transition（共 46 条）；
负样本为 11 条训练 transition。retention 单独列出，未进入任何训练 loss。

主要事实：负奖励从 `1` 缩到 `0.25`，负样本上的策略位移仍为原来的约 77%；缩到 `0.5`
时仍为约 93%。因此在当前 Adam 更新下，统一乘一个小系数不会按比例缩小参数更新。尺度会改变
混合梯度方向和是否触发 clipping，但 Adam 的逐参数归一化大幅抵消了绝对缩放。只有
`negative_scale=0` 真正去除了负目标的直接支配。

## 真实 rollout tuning 结果

固定 24 道 BIRD-dev 题，难度为 simple/moderate/challenging 各 8 题。协议固定为 version36、
greedy@1、`temperature=0`、`top_p=1`、`max_steps=30`、`bird-set`。

| 模型 | 正确 | 合法终止 | 平均步数 | simple | moderate | challenging | 相邻精确重复调用 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 未训练 SFT2 | **12/24** | **20/24** | 10.625 | 4/8 | 3/8 | 5/8 | 0 |
| neg=0 | 10/24 | 18/24 | 10.542 | 5/8 | 2/8 | 3/8 | 0 |
| neg=0.25 | 8/24 | 14/24 | 10.333 | 4/8 | 2/8 | 2/8 | 2 |
| **neg=0.5** | **11/24** | **19/24** | **10.292** | 5/8 | 2/8 | 4/8 | 1 |
| neg=1 | 11/24 | 17/24 | 11.375 | 5/8 | 3/8 | 3/8 | 1 |

相对未训练 SFT2 的严格配对统计：

| negative scale | gains | regressions | net | exact p | legal gains | legal regressions | legal net | legal exact p |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 3 | -2 | 0.625 | 2 | 4 | -2 | 0.6875 |
| 0.25 | 0 | 4 | -4 | 0.125 | 1 | 7 | -6 | 0.0703 |
| **0.5** | 2 | 3 | -1 | 1.0 | 2 | 3 | -1 | 1.0 |
| 1 | 2 | 3 | -1 | 1.0 | 2 | 5 | -3 | 0.4531 |

`neg=0.5` 与 `neg=1` 的正确率配对净变化为 0（各有 2 gain、2 regression，`p=1.0`）；
合法率净增 2，`p=0.625`。它还把平均步数减少 1.083，重复调用没有增加。

## 小样本选择

按预先约定的选择顺序——先合法率，再准确率，再回归/重复调用——本轮选择：

> `learning_rate=4e-6, think_weight=0.5, action_weight=1.0, negative_scale=0.5`

这是四个 RL 候选中的局部最优，不是相对 SFT2 的胜出配置：它仍比 SFT2 少 1 道正确、少 1 道
合法终止，且样本很小、差异不显著。

## 方法诊断与下一步约束

1. 结果否定了“把负 reward 标量简单调小就能线性减弱惩罚”的假设；Adam 使这一关系高度非线性。
2. `neg=0.25` 比 `neg=0.5` 和 `neg=1` 都差，说明小样本行为不是 reward scale 的单调函数。
3. 下一次若继续优化，不宜在同一 24 题上密集搜小数点。应固定 `neg=0.5` 到不相交 holdout
   做一次确认；若仍不超过 SFT2，再改为显式的正/负组梯度范数约束、per-transition advantage
   clipping 或 SFT2 KL trust region，而不是继续统一缩放负 loss。

后续已完成 `think_weight=1.0` 与负梯度范数上限0.5的紧邻验证；两者均未超过本报告选择的
`think_weight=0.5, negative_scale=0.5`。详见
`ROUTED_COUPLED_THINK_WEIGHT_GRADIENT_CAP_DEV24_20260808_ZH.md`。

## 产物

- 训练与冻结策略评分：
  `artifacts/rl/routed_coupled_behavior_smoke_20260808/negative_scale_sweep_train5.json`
- 24 题行为与完整配对统计：
  `artifacts/rl/routed_coupled_behavior_smoke_20260808/negative_scale_sweep_dev24_summary.json`
- 训练脚本：`src/rl/diagnostics/run_routed_coupled_reward_smoke.py`
- 汇总脚本：`src/rl/diagnostics/summarize_routed_coupled_behavior_smoke.py`
- 远端 adapter 根目录：
  `/home/dengyan/tabular_rl_outputs/diagnostics/routed_coupled_behavior_smoke_20260808/negative_scale_sweep_adapters`
