# Routed coupled RL：不相交 Holdout120 验证（2026-08-08）

## 目的与实验边界

此前 5 题训练、24 题 rollout 已用于超参数选择。本实验冻结两个候选，在一个更大的、与该
24 题调参集完全不相交的 BIRD-dev 子集上复核：

1. 当前选择的 `think_weight=0.5` 是否能在未参与调参的题目上改变策略并保留行为收益；
2. `think_weight=1.0` 在更大样本上是否优于当前选择；
3. 小样本上观察到的正确率、合法率与重复调用变化是否稳定。

这仍是单个冻结 holdout、单训练 seed 的诊断，不等同于完整 BIRD-dev 结果或多 seed 稳定性
结论。梯度上限配置在 24 题上已经明显劣于另外两臂，本次没有继续消耗计算量复测。

## 冻结 cohort

- 数据：BIRD-dev。
- 数量：120 题；simple、moderate、challenging 各 40 题。
- 选择 seed：`20260809`。
- 排除：调参 24 题的全部索引；实际交集为 0。
- 选择过程只使用题目难度，不使用任何模型成败结果。
- 索引 SHA256：
  `13f84241d898f2aa0c8d96a33a2d91ab0a5cbd3130eb1cb532085aa8ba26f175`。

## 候选与固定评测配置

三个比较臂均来自同一个 Qwen2.5-Coder-7B-Instruct + SFT2 checkpoint-1682：

| 名称 | 训练 | think weight | action weight | negative scale | 负梯度上限 |
|---|---|---:|---:|---:|---:|
| SFT2 | 无本轮 RL | — | — | — | — |
| think=0.5 | 5 题、5 optimizer steps | 0.5 | 1.0 | 0.5 | 无 |
| think=1.0 | 5 题、5 optimizer steps | 1.0 | 1.0 | 0.5 | 无 |

两个 RL 候选固定 `lr=4e-6`、weight decay=0.01、rank 项关闭，并使用同样的 38 条正
transition 与 11 条 `severe_local_error` 负 transition。除 think 权重外没有训练变量变化。

评测固定为 version36、greedy@1、`temperature=0`、`top_p=1`、`max_steps=30`、
`bird-set`、logprobs20。每个候选单卡并发 16；并发只影响吞吐，不改变采样协议。

## Holdout120 结果

| 模型 | 正确 | 合法终止 | 平均步数 | simple | moderate | challenging | 相邻精确重复调用 | 有重复的题 | action 数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SFT2 | 46/120 (38.33%) | 83/120 (69.17%) | **9.008** | 21/40 | 12/40 | **13/40** | 7 | 6 | **794** |
| **think=0.5** | **49/120 (40.83%)** | **89/120 (74.17%)** | 9.417 | **25/40** | 14/40 | 10/40 | **4** | **3** | 836 |
| think=1.0 | 48/120 (40.00%) | 85/120 (70.83%) | 10.392 | 24/40 | **15/40** | 9/40 | 6 | 5 | 807 |

相对 SFT2 的逐题严格配对统计：

| 候选 | gains | regressions | net | exact p | legal gains | legal regressions | legal net | legal exact p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| think=0.5 | 9 | 6 | +3 | 0.6072 | 17 | 11 | +6 | 0.3449 |
| think=1.0 | 8 | 6 | +2 | 0.7905 | 13 | 11 | +2 | 0.8388 |

`think=1.0` 相对 `think=0.5` 为 4 gains、5 regressions，净 -1，`p=1.0`；合法终止为
6 gains、10 regressions，净 -4，`p=0.4545`。两者之间没有显著正确率差异，但
`think=0.5` 的平均步数少 0.975、合法终止多 4 题、相邻重复少 2 次，因此继续保留
`think=0.5` 作为当前候选。

## 策略变化与观测

- `think=0.5` 相对 SFT2 有 98/120 题的完整 action 序列发生变化，24/120 题第一动作变化。
- `think=1.0` 相对 SFT2 有 96/120 题的完整 action 序列变化，25/120 题第一动作变化。
- 因而本轮不能再解释为“RL 没有改变策略”。变化很广，但正确率净收益只有 +3 或 +2，说明
  **策略位移已经发生，位移方向仍缺少足够稳定的任务级一致性**。
- `think=0.5` 的收益主要来自 simple（+4）和 moderate（+2），challenging 反而 -3；
  `think=1.0` 为 simple +3、moderate +3、challenging -4。两个候选都表现出相似的难度迁移，
  不能把总体小幅上升解释成困难组合能力的提升。
- `think=0.5` 相对 SFT2 减少了重复调用，但 action 总数增加 42、平均步数增加 0.408；它改善的
  是部分合法终止和局部循环，不是全局轨迹压缩。

### 变化幅度分解

| 层级 | 指标 | think=0.5 相对 SFT2 |
|---|---|---:|
| 参数 | adapter 更新范数 / 初始 adapter | 0.1912% |
| 冻结策略 | 正奖励 transition 联合平均绝对 Δlogp | 0.01244 |
| 冻结策略 | 负奖励 transition 联合平均绝对 Δlogp | 0.07194 |
| 闭环序列 | 完整调用序列改变 | 98/120 (81.7%) |
| 闭环序列 | 工具名序列改变 | 84/120 (70.0%) |
| 闭环序列 | 仅参数改变、工具名序列不变 | 14/120 (11.7%) |
| 闭环序列 | 第一调用改变 | 24/120 (20.0%) |
| 编辑距离 | 每题完整调用平均/中位编辑距离 | 4.09 / 4.00 |
| 编辑距离 | 相对最长序列的平均归一化距离 | 47.9% |
| 工具选择 | 工具名序列平均归一化距离 | 27.4% |
| 全局分布 | 边际工具频率 JS divergence | 0.00207 bit |
| 长度 | 平均/平均绝对 step 变化 | +0.408 / 3.625 |

参数位移很小，但 greedy 轨迹变化很广；同时边际工具频率几乎不变。这更符合“许多问题在相近
决策边界上被重新分流”，而不是模型整体换了一套工具偏好。负 transition 的冻结策略位移约为
正 transition 的 5.8 倍，说明当前 reward/optimizer 组合仍存在明显的不对称更新。

按最终结果分组，9 个 gain 的平均归一化调用编辑距离为 66.6%，平均少 1.56 步；6 个 regression
为 55.7%，但平均多 **9.17 步**。40 个双方都正确的问题中仍有 26 个轨迹改变；65 个双方都错误
的问题中有 57 个改变。说明大量更新消耗在结果不变的重路由上，而主要有害回归具有明显的长轨迹
特征。

## 结论边界

扩大到不相交 120 题后，当前配置不再低于 SFT2：正确率净 +3、合法终止净 +6，并减少相邻
精确重复。但所有配对差异都不显著，且 challenging 分层出现回归。因此本轮支持的最窄判断是：

1. coupled rule reward 确实能够广泛改变闭环策略；
2. `think_weight=0.5` 比 `think_weight=1.0` 有更好的行为效率与合法性趋势；
3. 现有 5 题奖励方向尚未形成稳定、尤其是面向 challenging 的能力提升。

若继续训练规模实验，应保持此 holdout 冻结，不再据此搜索小数点超参数；下一步应扩大训练题与
奖励覆盖，并在新的不相交集或完整 BIRD-dev 上做一次性确认。

后续已完成五题统一更新、类别重平衡和新的不相交 Holdout60；详见
`ROUTED_COUPLED_REWARD_BALANCE_HOLDOUT60_20260808_ZH.md`。

## 完整性与基础设施说明

两个候选均生成 120 条唯一结果，每题恰好一个 sample；索引集合、version36、temperature、
top-p、denotation、generation stats 和全部汇总文件通过离线完整性检查。评测完成后，旧 runtime
的后处理校验器曾只接受结构化索引对象，而本 cohort 文件是顶层数组，导致模型评测完成后抛出
`TypeError`。该错误没有影响 240 条已生成轨迹；校验器和汇总器现均兼容两种格式，重新执行的
只是不加载模型的后处理，两个状态均已标记 `complete rows=120`。

## 产物

- 配对统计：
  `artifacts/rl/routed_coupled_holdout120_20260808/holdout120_paired_summary.json`
- 完整策略编辑距离与逐题统计：
  `artifacts/rl/routed_coupled_holdout120_20260808/think0p5_vs_sft2_policy_shift.json`
- 冻结索引：
  `artifacts/rl/routed_coupled_holdout120_20260808/routed_coupled_holdout120_seed20260809.indices.json`
- cohort 清单：
  `artifacts/rl/routed_coupled_holdout120_20260808/routed_coupled_holdout120_seed20260809.indices.manifest.json`
- 远端结果根目录：
  `/home/dengyan/tabular_rl_outputs/eval_runtime_version36_20260728/data/results/`
- `think=0.5` 结果目录：
  `routed_coupled_think0p5_neg0p5_version36_holdout120_greedy_20260808`
- `think=1.0` 结果目录：
  `routed_coupled_think1_neg0p5_version36_holdout120_greedy_20260808`
