# Routed coupled reward：Scale20 与独立 Holdout90

日期：2026-08-09

## 目的

本实验把此前 5 题的 routed coupled reward 小样本训练扩大到 20 题，检验两个问题：

1. 增加训练问题后，冻结轨迹上的奖励方向能否继续一致地改变策略；
2. 更大的参数更新是否转化为独立任务上的准确率、合法终止率或轨迹行为收益。

它不是最终规模实验，也不用于从结果反向选择训练超参数。

## 训练数据与固定配置

20 题来自同一份 BIRD-train 冻结轨迹池，包含此前 Scale5 的全部 5 题，并补齐为：

| 难度 | 题数 |
|---|---:|
| simple | 6 |
| moderate | 7 |
| challenging | 7 |
| 合计 | 20 |

全部入选题的所有 transition 都通过完整 `<think>` 加 JSON action carrier 检查。训练集共有 220 个有效奖励 transition，另用 3 道独立正确题的 62 个 transition 测量 retention。优化顺序、题目索引和替换记录见 [train20_manifest.json](../../../artifacts/rl/routed_coupled_scale20_20260808/train20_manifest.json)。

固定训练配置：

| 配置项 | 数值 |
|---|---:|
| 初始模型 | 精确 SFT2 checkpoint-1682 |
| question batch size | 5 |
| optimizer steps | 4 |
| think token 权重 | 0.5 |
| action token 权重 | 1.0 |
| negative scale | 0.5 |
| loss normalization | category mean |
| correct key evidence | 2.0 |
| correct key backslice | 2.0 |
| correct terminal | 0.25 |
| severe local error | 1.0 |
| weight decay | 0.01 |

奖励路由只训练 `correct_key_evidence`、`correct_key_backslice`、`correct_terminal` 和 `severe_local_error`；其他 clean transition 为中性。训练 think 与 action 两部分，不是只训练 action token。

预先声明两个学习率分支 `1.2e-5` 和 `6e-6`。候选选择规则是在行为评测前选择 adapter update/reference ratio 位于 `[0.0020, 0.0035]` 的分支；若都满足或都不满足，则选更新幅度更小者。

## 显存故障与重启

首次运行的两个独立学习率分支都在第 4 个 batch 的同一条 5,778-token transition 上 OOM；当时只完成了 3 个 optimizer step，尚未保存可用训练产物。该尝试未续训。

修复保持难度层不变，将该 moderate 题 `2913` 替换为最长有效 transition 为 3,339 tokens 的 `5623`，并启用 expandable CUDA segments。两个学习率分支都从 SFT2 重新完整训练，4/4 optimizer steps 完成，无 gradient clipping。首次失败日志保留用于审计。

## 冻结 transition 上的训练结果

| LR | adapter update/reference | update norm | 正奖励 mean Δ | 负奖励 mean Δ | retention mean Δ | 是否进入行为评测 |
|---|---:|---:|---:|---:|---:|---|
| `1.2e-5` | 0.5007% | 0.18423 | +0.01082 | -0.06033 | +0.00190 | 否，超出预声明更新区间 |
| `6e-6` | 0.2511% | 0.09238 | +0.00654 | -0.02985 | +0.00154 | 是 |

选中的 `6e-6` 分支中：

| 信号类别 | 测量 transition | mean Δ | mean |Δ| | 奖励方向一致率 |
|---|---:|---:|---:|---:|
| 全部正奖励 | 246 | +0.00654 | 0.00691 | 88.62% |
| key evidence | 50 | +0.00590 | 0.00620 | 86.00% |
| key backslice | 151 | +0.00440 | 0.00485 | 88.08% |
| terminal | 45 | +0.01443 | 0.01459 | 93.33% |
| severe local error | 35 | -0.02985 | 0.03092 | 71.43% |
| retention correct | 62 | +0.00154 | 0.00232 | 74.14% |

这里的 transition 数包含训练后统一测量的训练与 retention 冻结 transition，因此正奖励测量数高于实际用于 optimizer 的正奖励 transition 数。完整结果见 [retry1_lr6e6_train20.json](../../../artifacts/rl/routed_coupled_scale20_20260808/retry1_lr6e6_train20.json)；高学习率对照见 [retry1_lr12e6_train20.json](../../../artifacts/rl/routed_coupled_scale20_20260808/retry1_lr12e6_train20.json)。

## Holdout90 评测设计

从 BIRD-dev 重新抽取 90 题，每个难度 30 题；排除此前 tuning24、Holdout120 和 Holdout60 的全部 204 个索引，重叠为零。索引 SHA256 为 `23dc6adb3e358c0a758a1c9ec5bcb045d6f3cb13485a93e373b6cfe24b8b7162`。

固定评测为 version36、`bird-set`、greedy@1、temperature 0、top-p 1、max steps 30、logprobs 20、并发 16 和动态 vLLM batching。对比三组：原始 SFT2、此前 Scale5 key-weighted、当前 Scale20 `lr=6e-6`。Scale5 与 Scale20 在两张 GPU 上并行完成，各 90/90，未发生结果级重试。

## Holdout90 结果

| 模型 | correct | accuracy | legal | legal rate | mean steps | simple | moderate | challenging | 相邻完全重复调用题数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SFT2 | 29/90 | 32.22% | 56/90 | 62.22% | 8.522 | 13/30 | 12/30 | 4/30 | 4 |
| Scale5 | 37/90 | 41.11% | 68/90 | 75.56% | 8.033 | 15/30 | 16/30 | 6/30 | 1 |
| Scale20 | 35/90 | 38.89% | 71/90 | 78.89% | 8.200 | 13/30 | 15/30 | 7/30 | 4 |

配对比较：

| 比较 | accuracy gains | regressions | net | exact p | legal gains | legal regressions | legal net | legal exact p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Scale5 vs SFT2 | 12 | 4 | +8 | 0.0768 | 16 | 4 | +12 | 0.0118 |
| Scale20 vs SFT2 | 11 | 5 | +6 | 0.2101 | 20 | 5 | +15 | 0.00408 |
| Scale20 vs Scale5 | 6 | 8 | -2 | 0.7905 | 10 | 7 | +3 | 0.6291 |

因此，Scale20 相对 SFT2 的合法终止改善在该 90 题样本上显著；准确率净增 6 题但不显著。扩大训练问题没有超过 Scale5 的准确率，二者差异也不显著。

## Scale20 相对 Scale5 的策略变化

扩大训练并非“模型没有变化”：

| 行为指标 | 数值 |
|---|---:|
| 完整 action 序列变化 | 73/90 |
| tool 序列变化 | 53/90 |
| 仅参数变化 | 20/90 |
| 首个 action 变化 | 21/90 |
| exact action edit distance 均值 | 4.189 |
| normalized exact edit distance 均值 | 0.507 |
| mean absolute step delta | 2.256 |
| 总 action 数变化 | -10（576 vs 586） |
| 工具边际分布 JS divergence | 0.000932 bits |

这表示单题轨迹发生了广泛重排，但全局工具使用频率几乎不变。Scale20 相对 Scale5 的 6 个 gain 和 8 个 regression 全部伴随 action 序列变化；regression 的平均 exact edit distance 为 6.75，高于 gain 的 4.67。

完整配对统计见 [holdout90_paired_summary.json](../../../artifacts/rl/routed_coupled_scale20_20260808/holdout90_paired_summary.json)，策略变化见 [scale20_vs_scale5_policy_shift.json](../../../artifacts/rl/routed_coupled_scale20_20260808/scale20_vs_scale5_policy_shift.json)，评测 cohort 见 [routed_coupled_scale20_holdout90_seed20260811.indices.json](../../../artifacts/rl/routed_coupled_scale20_20260808/routed_coupled_scale20_holdout90_seed20260811.indices.json)。

## 本轮回答的问题

- 扩大到 20 题后，奖励能够一致地改变冻结轨迹概率，且参数更新幅度达到 0.2511%；“RL 强度不足、策略几乎没变”不再符合本轮证据。
- 行为策略也明显变化：Scale20 与 Scale5 有 73/90 条 action 序列不同。
- 更大训练集没有在当前 Holdout90 上带来更高准确率：Scale20 为 35/90，Scale5 为 37/90，配对净差 -2，`p=0.7905`。
- Scale20 的清晰增益是合法终止：71/90，相对 SFT2 净增 15，`p=0.00408`。
- 本轮只说明“扩大数据能进一步改变策略，但当前奖励与规模组合主要改善协议完成度，尚未稳定转化为更高任务准确率”；不能据此单独判定奖励设计最终无效，也不能把 Scale5 的 37/90 当作已稳定复现的总体增益。

## 后续完整 BIRD-dev1534 验证

在 Holdout90 之后，只对 Scale5 与预注册选出的 Scale20 `lr=6e-6` 运行完整 BIRD-dev；SFT2 直接复用既有完整结果。两组均在第一次评测尝试完成 1534/1534，version36、greedy@1、temperature 0、top-p 1、max steps 30、bird-set、logprobs 20、并发 24、动态 vLLM batching。

| 模型 | correct | accuracy | valid | valid rate | avg steps | simple | moderate | challenging |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SFT2 | 762/1534 | 49.67% | 1194/1534 | 77.84% | 8.371 | 540/925 | 176/464 | 46/145 |
| Scale5 | 767/1534 | 50.00% | 1201/1534 | 78.29% | 8.468 | 540/925 | 181/464 | 46/145 |
| Scale20 | 770/1534 | 50.20% | 1218/1534 | 79.40% | 8.437 | 543/925 | 184/464 | 43/145 |

准确率配对统计：

| 比较 | gains | regressions | net | exact McNemar p |
|---|---:|---:|---:|---:|
| Scale5 vs SFT2 | 96 | 91 | +5 | 0.7700 |
| Scale20 vs SFT2 | 103 | 95 | +8 | 0.6190 |
| Scale20 vs Scale5 | 92 | 89 | +3 | 0.8819 |

合法终止配对统计：

| 比较 | legal gains | legal regressions | net | exact p |
|---|---:|---:|---:|---:|
| Scale5 vs SFT2 | 126 | 119 | +7 | 0.7016 |
| Scale20 vs SFT2 | 129 | 105 | +24 | 0.1325 |
| Scale20 vs Scale5 | 132 | 115 | +17 | 0.3086 |

全量策略变化显示更新并非没有改变模型行为：Scale5 相对 SFT2 有 1183/1534 条完整 action 序列变化，Scale20 相对 SFT2 有 1245/1534 条变化；二者分别有 906 与 932 条 tool 序列变化。Scale20 相对 Scale5 也有 1150/1534 条 action 序列变化。

完整 dev 将 Holdout90 的大幅点估计收缩为小幅净增：Scale20 是三者最高的 770/1534，但相对 SFT2 只有净增 8 题且不显著；Scale5 净增 5 题且不显著。Scale20 的 challenging 从 SFT2/Scale5 的 46 降到 43，同时 simple/moderate 分别增加 3/8，说明能力仍在难度层之间重新分配。现有证据因此支持“策略变化幅度问题已解决”，但不支持“奖励方向已经能稳定提高最终任务准确率”。

最终产物：

- [Scale5 全量汇总](../../../artifacts/rl/routed_coupled_scale20_20260808/scale5_full_dev_summary.json)
- [Scale20 全量汇总](../../../artifacts/rl/routed_coupled_scale20_20260808/scale20_full_dev_summary.json)
- [全量策略变化与合法终止配对](../../../artifacts/rl/routed_coupled_scale20_20260808/full_dev_policy_change_summary.json)
