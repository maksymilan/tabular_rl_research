# Routed coupled RL：think 权重与负梯度上限验证（2026-08-08）

## 问题与实验边界

负奖励尺度扫掠在同一 5 题训练、24题 tuning cohort 上选择了：

```text
lr=4e-6, think_weight=0.5, action_weight=1.0, negative_scale=0.5
```

本轮只验证两个紧邻假设：

1. 把 `think_weight` 从 0.5 提高到 1.0，是否能通过加强推理 token 训练改善闭环行为；
2. 在 `think_weight=1.0` 下，把每题负梯度范数限制为正梯度范数的 0.5 倍，是否能解决负更新支配。

24题 cohort 已用于此前调参，本轮结果仍然只是 tuning read，不是独立泛化结果。

## 固定项与三臂

- 全部从同一个正式 SFT2 checkpoint-1682 独立初始化。
- 训练题、顺序和 token 完全固定：`3044, 4050, 1626, 5040, 5027`，5个 optimizer steps。
- 训练信号：38条正 transition、11条 `severe_local_error` 负 transition；其余置中性。
- retention：问题4692的8条正确 transition，只评分、不训练。
- 固定 `lr=4e-6`、`negative_scale=0.5`、action weight=1、rank=0、weight decay=0.01。

三臂为：

| 名称 | think weight | negative scale | 负梯度/正梯度范数上限 |
|---|---:|---:|---:|
| 当前对照 | 0.5 | 0.5 | 无 |
| think增强 | 1.0 | 0.5 | 无 |
| 梯度平衡 | 1.0 | 0.5 | 0.5 |

梯度平衡臂不是再次缩放 reward。每题分别反传正、负目标，测量 `g_pos`、`g_neg`，然后执行：

```text
g_neg <- min(1, 0.5 * ||g_pos|| / ||g_neg||) * g_neg
g <- g_pos + g_neg
```

最后再执行共同的全局 `clip_grad_norm=1`。

## 训练侧结果

| 配置 | adapter位移/初始adapter | 正样本 joint abs Δlogp | 负样本 joint abs Δlogp | retention joint Δlogp | clip步数 |
|---|---:|---:|---:|---:|---:|
| think=0.5 | 0.1912% | 0.01244 | 0.07194 | +0.01183 | 0/5 |
| think=1.0 | 0.1894% | 0.01884 | 0.08645 | +0.01271 | 1/5 |
| think=1.0 + cap=0.5 | 0.1805% | **0.02637** | **0.03826** | **+0.01926** | 0/5 |

提高 think 权重同时放大了正负两侧，不能单独解决负位移支配。梯度上限则确实完成了预期的
first-order manipulation：正样本位移上升、负样本位移下降、retention 上升，并消除了裁剪。

上限在第1题最明显：`||g_pos||=0.284`、`||g_neg||=0.947`，负/正原为3.34倍；负梯度被乘
0.150后变为正梯度的0.5倍。第4题负梯度被乘0.697；第2、3题原本低于上限而未缩放；第5题
没有负 transition。

## 24题真实 rollout

评测固定 version36、greedy@1、temperature=0、top_p=1、max_steps=30、bird-set。

| 配置 | 正确 | 合法终止 | 平均步数 | simple | moderate | challenging | 相邻精确重复调用 | action数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 未训练 SFT2 | **12/24** | **20/24** | 10.625 | 4/8 | 3/8 | 5/8 | 0 | 177 |
| **think=0.5** | **11/24** | **19/24** | **10.292** | 5/8 | 2/8 | 4/8 | 1 | 178 |
| think=1.0 | 11/24 | 19/24 | 10.750 | 5/8 | 2/8 | 4/8 | 3 | 188 |
| think=1.0 + cap=0.5 | 9/24 | 16/24 | 11.042 | 5/8 | 2/8 | 2/8 | 1 | 174 |

严格配对结果：

- `think=1.0` 相对 `think=0.5`：1 gain、1 regression，净0，`p=1.0`；合法净0，`p=1.0`。
  但平均步数增加0.458、action增加10、相邻重复从1次增至3次。
- `think=1.0 + cap=0.5` 相对 `think=0.5`：0 gain、2 regressions，净-2，`p=0.5`；
  合法0 gain、3 regressions，净-3，`p=0.25`。
- 梯度平衡臂相对 SFT2：1 gain、4 regressions，净-3，`p=0.375`；合法净-4，`p=0.21875`。

## 当前选择

本轮不改变上一轮选择。三臂中仍保留：

```text
lr=4e-6
think_weight=0.5
action_weight=1.0
negative_scale=0.5
negative_grad_ratio_cap=None
```

客观上：

1. `think_weight=1.0` 确实增强推理 span 的更新，但没有增加正确率或合法率，反而增加动作和重复。
2. `cap=0.5` 在冻结训练 token 上产生了更一致的正负位移，却降低了闭环正确率与合法率。
3. 因而“训练前缀上的奖励方向一致率更高”不是足够的模型选择指标；真实状态分布下的 rollout
   仍是必要条件。
4. 这里只否定当前较强的 `cap=0.5` 配置，不能推广成所有梯度约束都无效。不过继续在同一24题
   密集搜索 `think=0.75` 或 `cap=0.75/1.0` 会明显增加调参过拟合风险，暂不继续。

## 产物

- `artifacts/rl/routed_coupled_hparam_sweep_20260808/think1_nocap_train5.json`
- `artifacts/rl/routed_coupled_hparam_sweep_20260808/think1_ngcap0p5_train5.json`
- `artifacts/rl/routed_coupled_hparam_sweep_20260808/hparam_sweep_dev24_summary.json`
- 训练实现：`src/rl/diagnostics/run_routed_coupled_reward_smoke.py`
- 行为汇总：`src/rl/diagnostics/summarize_routed_coupled_behavior_smoke.py`
- 远端根目录：
  `/home/dengyan/tabular_rl_outputs/diagnostics/routed_coupled_hparam_sweep_20260808`

后续已在与本调参集完全不相交的 120 题上复核 `think=0.5` 与 `think=1.0`；详见
`ROUTED_COUPLED_HOLDOUT120_20260808_ZH.md`。
