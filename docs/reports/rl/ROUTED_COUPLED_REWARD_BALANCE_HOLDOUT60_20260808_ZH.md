# Routed coupled RL：正负更新重平衡与不相交 Holdout60（2026-08-08）

## 问题与边界

Holdout120 显示当前 `lr=4e-6, think_weight=0.5, negative_scale=0.5` 已广泛改变策略，但冻结
负 transition 的联合平均绝对位移为正 transition 的5.8倍。当前实验检验：这种有效惩罚过强
是否来自逐题 AdamW 更新与类别质量失衡，以及在保持相近总参数位移时，提高 evidence/backslice
正路径能否改善真实 rollout。

所有训练臂都从同一个正式 SFT2 checkpoint-1682 独立初始化，使用完全相同的5题、49条 active
transition 和8条只评分 retention transition。没有从历史 RL adapter 续训。

## 实现变量

新增两个默认关闭的诊断开关：

- `question_batch_size=5`：五道题的梯度累积后只执行一次 AdamW step；
- `loss_normalization=category_mean`：每个当前 batch 内先按 reward category 求均值，再按显式类别
  质量组合。

默认 `question_batch_size=1, loss_normalization=active_mean` 保留历史逐题行为。新增权重和默认行为有
4个纯函数单元测试，全部通过。

## 训练侧五个对照

固定 `think_weight=0.5`、action weight=1、negative scale=0.5、weight decay=0.01。表中 abs Δlogp
均在同一57条冻结 transition 上测量。

| 配置 | LR | optimizer steps | adapter位移 | 正 abs Δlogp | 负 abs Δlogp | evidence | backslice | terminal | retention |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 原逐题配置 | 4e-6 | 5 | 0.1912% | 0.01244 | 0.07194 | 0.00378 | 0.00703 | 0.03517 | 0.01183 |
| batch5 active mean | 4e-6 | 1 | 0.0678% | 0.00686 | 0.01573 | 0.00144 | 0.00307 | 0.02215 | 0.00754 |
| batch5 等类别 | 4e-6 | 1 | 0.0678% | 0.00741 | 0.01255 | 0.00240 | 0.00179 | 0.02704 | 0.00957 |
| batch5 等类别、幅度匹配 | 1.2e-5 | 1 | 0.2035% | 0.01780 | 0.04659 | 0.00662 | 0.00513 | 0.06193 | 0.02519 |
| **batch5 关键正路径加权** | **1.2e-5** | **1** | **0.2023%** | **0.01487** | **0.05525** | **0.01154** | **0.00935** | **0.03256** | **0.01305** |

关键正路径加权的类别质量为：

```text
evidence=2, backslice=2, terminal=0.25, severe_local_error=1
```

相对原逐题配置，它在几乎相同的 adapter 位移下：evidence 提高约3.1倍、backslice 提高约33%、
terminal 略降、负位移下降约23%、retention 略升。五题统一累积本身把更新位移降到约原来的35%，
因此用 `lr=1.2e-5` 做幅度匹配，而不是把更小位移误判为更稳定的方向。

`negative_grad_ratio_cap=1.0` 在 batch5 等类别臂没有触发：正梯度范数为0.2636，负梯度范数为
0.1240，负梯度缩放为1.0。说明逐题更新被合并后，标量 loss 上已经不是负梯度范数占优；历史
观察到的负策略位移支配不能简单归因于单批梯度范数。

## 新的冻结 Holdout60

- BIRD-dev 60题，simple/moderate/challenging 各20题；
- seed `20260810`；
- 排除此前 tuning24 与 Holdout120 的全部144题，实际交集为0；
- 选择只使用难度，不使用任何模型结果；
- 索引 SHA256：
  `0e86c84743f2384d074e8606f48acc6f26603db190ed1ee5428210da81ca6979`。

评测固定 version36、greedy@1、temperature=0、top_p=1、max_steps=30、bird-set、logprobs20。
原配置和关键正路径加权候选分别使用GPU0/GPU1并行评测，每卡并发16；SFT2直接切片既有完整dev
结果，没有重复推理。

## Holdout60 结果

| 模型 | 正确 | 合法终止 | 平均步数 | simple | moderate | challenging | 相邻精确重复 | action数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SFT2 | **24/60** | **41/60** | 9.883 | **11/20** | **9/20** | 4/20 | 3 | 426 |
| 原逐题配置 | 22/60 | 40/60 | **9.633** | 10/20 | 6/20 | 6/20 | **2** | 403 |
| **关键正路径加权** | **24/60** | 40/60 | 10.267 | **11/20** | 5/20 | **8/20** | 3 | **399** |

严格配对统计：

| 比较 | gains | regressions | net | exact p | legal net | legal exact p |
|---|---:|---:|---:|---:|---:|---:|
| 原配置 vs SFT2 | 4 | 6 | -2 | 0.7539 | -1 | 1.0 |
| 关键加权 vs SFT2 | 4 | 4 | 0 | 1.0 | -1 | 1.0 |
| **关键加权 vs 原配置** | **5** | **3** | **+2** | **0.7266** | **0** | **1.0** |

关键加权相对原配置有49/60题完整调用序列变化、14/60题第一调用变化，平均完整调用编辑距离4.6。
五个 gain 平均少3步；三个 regression 平均多11.33步。它恢复了总体正确数，并把 challenging
提高到8/20，但 moderate 降到5/20且平均步数增加0.633。因此“关键正奖励增强”产生了预期的
能力重分配，却仍未解决有害长轨迹。

## 当前解释与停止点

1. 有效惩罚偏重的判断获得训练侧支持，但根因不是正奖励 transition 数量少，也不是统一 batch
   中负梯度范数更大；逐题 Adam 状态、类别梯度方向和 greedy 决策边界共同放大了负策略位移。
2. 五题统一更新并定向提高 evidence/backslice，能在相同总位移下把 Holdout60 从22题恢复到24题，
   但差异不显著，且不同难度间存在明显转移。
3. 当前60题已经参与该权重的行为判断，不应继续用于搜索 `2/2/0.25/1` 附近的小数点。
4. 下一步如果继续，优先增加 balanced training questions 和加入针对长无进展轨迹的独立约束；
   不应继续提高全局学习率、增加同5题epoch，或进一步削弱所有负奖励。

## 产物

- 训练侧五臂：`artifacts/rl/routed_coupled_balance_smoke_20260808/*train5.json`
- Holdout60配对结果：
  `artifacts/rl/routed_coupled_balance_smoke_20260808/balance_holdout60_paired_summary.json`
- 候选相对原配置逐题策略位移：
  `artifacts/rl/routed_coupled_balance_smoke_20260808/keyweighted_vs_current_policy_shift.json`
- 候选相对SFT2逐题策略位移：
  `artifacts/rl/routed_coupled_balance_smoke_20260808/keyweighted_vs_sft2_policy_shift.json`
- 冻结索引：
  `artifacts/rl/routed_coupled_balance_smoke_20260808/routed_coupled_balance_holdout60_seed20260810.indices.json`
- 训练实现：`src/rl/diagnostics/run_routed_coupled_reward_smoke.py`

