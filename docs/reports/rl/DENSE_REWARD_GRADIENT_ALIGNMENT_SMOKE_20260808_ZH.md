# Dense RL 奖励—梯度一致性短诊断

日期：2026-08-08

## 目的

用短时、无参数更新的真实训练 batch 检查 Exp16–Exp18 策略几乎不移动的直接原因。诊断只读取 frozen mixed60 的精确 token，在 SFT2 checkpoint-1682 QLoRA 初始化上分别反传：

- 正奖励 full-response policy loss；
- 负奖励 full-response policy loss；
- 原配置 `lambda=0.5, beta=0.1` 的 dense-outcome Rank loss；
- 同一批奖励的 JSON action-token-only policy loss。

没有 rollout、optimizer step、模型保存或完整评测。

## 历史训练日志先验

| 实验 | steps | grad norm mean | grad norm max | `grad_norm > 1` |
|---|---:|---:|---:|---:|
| Exp16 | 60 | 0.1958 | 0.8281 | 0 |
| Exp17 | 60 | 0.2295 | 0.9492 | 0 |
| Exp18 | 120 | 0.1968 | 0.7383 | 0 |

因此 `max_grad_norm=1` 从未触发，梯度裁剪不是更新过小的原因。

## 两个真实 K4 batch

| 指标 | question 5027 | question 3044 |
|---|---:|---:|
| transitions | 16 | 20 |
| 正 / 负 transitions | 8 / 8 | 5 / 15 |
| response / action tokens | 2,282 / 568 | 3,183 / 808 |
| 正负 policy 梯度 cosine | **-0.855** | **0.028** |
| 合并 policy norm / 正负 norm 之和 | **44.6%** | **88.4%** |
| 净 policy 与 Rank cosine | **0.713** | **0.850** |
| policy+Rank norm / 两者 norm 之和 | **96.5%** | **98.4%** |
| full-response policy 与 action-only cosine | **0.501** | **0.191** |

question 5027 包含 2 条正确、2 条错误轨迹，正负数量对称；其中正负梯度强烈相反，合并后明显抵消。question 3044 包含 1 条正确、3 条错误轨迹，并含 3 个明确 severe local error；其正负梯度接近正交，没有同等程度的直接抵消。

两批中 Rank 都与净 policy 同向，而且合并后保留 96%–98% 的分量，所以 Rank 分支不是主要拉扯源。

最稳定的异常是 full-response 与 action-only 方向差异很大：两个余弦只有 0.50 和 0.19。也就是说，对整段 `<think> + JSON` 做平均后，当前奖励主要推动的参数方向并不等同于工具调用 token 的方向。由于 action token 只占这两批 response token 的约四分之一，reasoning token 的梯度几何实际主导或显著改写了策略更新方向。

## 初步定位

1. **不是梯度裁剪。** 180 个正式 optimizer step 无一次超过裁剪阈值。
2. **不是 Rank 与 policy 互相抵消。** 两个真实 batch 中二者明显同向。
3. **正负奖励冲突存在，但随题目分布变化。** 对称正负 batch 出现强冲突，错误轨迹占多数的 batch 没有；因此不能把全部结果概括成持续全局震荡。
4. **一致出现的问题是 full-response 信号与工具动作信号不一致。** 这是当前最优先的修复对象。
5. **LR/训练强度可能仍偏低，但不是首先应放大的量。** 在 action方向尚未对齐时直接提高 `1e-6` 学习率，可能只是更强地更新 reasoning/full-response 的混合方向。

## 下一步最小改动

先将 dense policy 与 Rank 两个分支都限制到 JSON action token，并在相同 frozen question 顺序上做 5–10 step smoke。只检查：

- 正奖励 action Δlogp 上升比例；
- 负奖励 action Δlogp 下降比例；
- backslice/evidence 分组的一致率；
- mean absolute action Δlogp；
- SFT2 保留样本的 action Δlogp。

只有 action 奖励方向一致率明显离开 50% 后，再做 `1e-6` 与 `4e-6` 的短学习率对照。这样可以把“目标方向错误”和“强度不足”分开，而不是同时改动。

## 产物

- 诊断实现：`src/rl/diagnostics/diagnose_reward_gradient_alignment.py`
- 本地 JSON：`artifacts/rl/reward_gradient_alignment_20260808/q5027_uniform.json`
- 本地 JSON：`artifacts/rl/reward_gradient_alignment_20260808/q3044_uniform.json`
- 远端目录：`/home/dengyan/tabular_rl_outputs/diagnostics/reward_gradient_alignment_20260808`

