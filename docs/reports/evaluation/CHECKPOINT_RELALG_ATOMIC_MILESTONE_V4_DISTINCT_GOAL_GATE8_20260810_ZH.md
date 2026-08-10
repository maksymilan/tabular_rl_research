# checkpoint-relalg Atomic milestone-v4 distinct-goal Gate8

日期：2026-08-10  
状态：diagnostic-only，不进入 SFT/RL

## 目的与设计

本 Gate8 验证两项变化：checkpoint 仍允许最多 8 次；每次 commit 的 `next_targets` 必须与当前
phase 和 active checkpoint path 上的旧 goal 集合不同。比较臂为同一当前 Harness 下的
`adaptive-v1` 与 `semantic-milestone-v4`。

- 官方模型：`deepseek-v4-flash`
- mode/profile：`atomic` / `semantic-v2`
- carrier：A / Text-JSON
- cohort：上一轮 v3 实际发生 commit 的 positions `1,14,19,22,42,44,47,54`
- 两臂均全新运行，共 16 episodes
- 单题 provider token 闸 400,000；总理论上限 6,400,000
- `bird-set`；gold SQL、gold rows 和 reasoning 不用于选题或分析

## 结果

| 指标 | adaptive-v1 | milestone-v4 |
|---|---:|---:|
| bird-set correct | 4/8 | 3/8 |
| legal termination | 6/8 | 7/8 |
| strict artifact correct | 2/8 | 2/8 |
| model turns | 91 | 107 |
| provider total tokens | 1,893,909 | 1,924,989 |
| tool errors | 9 | 15 |
| checkpoint episodes | 0/8 | 7/8 |
| checkpoint calls | 0 | 17 |
| restore calls | 0 | 0 |

正确率配对为 both=3、adaptive-only=1、v4-only=0、neither=4。v4 没有正确率 gain，并产生一个
regression。相比 adaptive，v4 增加 16 turns（+17.6%）和 31,080 tokens（+1.6%）。

adaptive 的 positions 42、54 与 v4 的 position 42 命中冻结的单题 token 闸。三条记录自身的
结构审计与 fresh replay 通过；对应目录按预算规则标记失败。其余 13 个目录总体审计通过。
206 次 provider attempts 的 `response_model` 全部精确为 `deepseek-v4-flash`。

## distinct-goal 审计

v4 的 17 个 commit 分布为：

- position 1：3
- position 14：0
- position 19：2
- position 22：1
- position 42：5
- position 44：2
- position 47：1
- position 54：3

对 NFKC、大小写、空白、末尾标点和目标数组顺序规范化后，每个 episode 内的 goal 集合全部
唯一：17/17 unique，重复 0，空 goal 0，`checkpoint_goal_not_distinct` 拒绝 0。说明 Harness
distinct-goal 约束和模型可见 v4 规则均按设计工作。

## 机械效率与失败原因

在实际 commit 的七题上：

- adaptive：81 turns、1,712,805 tokens、21,146 tokens/turn
- v4：99 turns、1,763,415 tokens、17,812 tokens/turn

checkpoint 仍使每 turn 上下文成本下降约 15.8%，但 18 个额外 turns 使七题总 token 反而增加
约 3.0%。工具错误也从 9 增至 15；v4 中 `unknown_column` 为 8 次，说明频繁 phase 切换没有
改善关系程序稳定性。

## 判断与下一步

1. 多 checkpoint、最多 8 次与 active-path goal 差异性实现有效。
2. 当前 v4 trigger 过早、过密；“goal 不同”不等于“checkpoint 值得创建”。
3. checkpoint 的上下文压缩效果再次复现，但未转化为正确率或总成本收益。
4. 下一版应保留 8 次上限和 distinct-goal，增加 Harness 可验证的最小 phase 进展门槛，例如
   commit 前要求足够数量的新成功 producer、新 active artifact 和最小 phase 长度；不能仅依赖
   模型把旧 goal 换成新措辞。
5. 在该门槛实现前，不扩大 v4 样本，不推断训练准入。

## Artifact

`data/results/checkpoint_relalg_v1_flash_text_json_atomic_semantic_v2_milestone_v4_effect_gate8_20260810/`

