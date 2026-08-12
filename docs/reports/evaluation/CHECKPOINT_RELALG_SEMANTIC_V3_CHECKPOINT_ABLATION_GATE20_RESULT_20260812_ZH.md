# Semantic-v3 Checkpoint-on vs Disabled fresh Gate20 结果

日期：2026-08-12  
状态：完成；诊断性 NO-GO，不扩大 checkpoint ablation

## 结论

这次 checkpoint 暴露充分，但没有带来正确性收益。`model-choice-commit-v6` 与
`checkpoint-disabled-v1` 在相同 20 题上的 `bird-set` 都是 **15/20**，且逐题正确/错误向量完全一致：
15 题两臂都对、5 题两臂都错，没有任一方向的 paired gain。Checkpoint 臂反而把 strict artifact
从 12/20 降到 10/20，把 schema match 从 12/20 降到 11/20，并增加 14 turns 与 170,624 provider
tokens。因此本 Gate 不支持“当前 checkpoint 机制优于始终只调用 Atomic 工具”。

这不是 underexposed 结果：checkpoint 臂在 8/20 题使用 checkpoint，共 11 次 accepted commit，正好达到
预注册的 exposure 门槛；没有 rejected commit。结论只约束当前 `semantic-v3-v24 +
model-choice-commit-v6 + restore disabled` 组合，不证明任何 checkpoint 思想一般无效。

## 冻结设计

- cohort：teacher1500 positions 300–319，20 个 fresh disjoint pairs；40 个付费 episode
- 共同配置：Atomic `semantic-v3-v24`、A/Text-JSON recent-4、official
  `deepseek-v4-flash`、同一 typed Harness
- treatment：`model-choice-commit-v6`、max checkpoints 8、restore 0、Harness eligibility `none`
- control：`checkpoint-disabled-v1`、max checkpoints 0、restore 0
- 每 episode：20 model turns、30 primitive calls、50 provider attempts、350k tokens、3600 秒
- 调度：四条动态 lanes；同题两臂相邻、先后次序按 position 奇偶交替
- 预注册总硬上限 14,000,000 tokens；实际 2,048,522

## 汇总

| 指标 | Checkpoint on | Disabled | 差值 |
|---|---:|---:|---:|
| `bird-set` correct | 15/20 | 15/20 | 0 |
| strict artifact | 10/20 | 12/20 | -2 |
| schema match | 11/20 | 12/20 | -1 |
| legal | 20/20 | 18/20 | +2 |
| model turns | 162 | 148 | +14 (+9.46%) |
| primitive calls | 162 | 147 | +15 |
| public tool errors | 13 | 13 | 0 |
| provider attempts | 167 | 149 | +18 |
| provider tokens | 1,109,573 | 938,949 | +170,624 (+18.17%) |
| prompt tokens | 988,036 | 864,272 | +123,764 |
| completion tokens | 121,537 | 74,677 | +46,860 |
| checkpoint task coverage | 8/20 | 0/20 | +8 |
| accepted commits | 11 | 0 | +11 |

正确性的 paired 表为：both-correct 15、checkpoint-only 0、disabled-only 0、both-wrong 5；exact
McNemar `p=1.0`。Strict 的 paired 表为：both 10、checkpoint-only 0、disabled-only 2、neither 8。
Schema 的 paired 表为：both 10、checkpoint-only 1、disabled-only 2、neither 7。

两条 disabled 非法终止分别是一次 provider error 与一次 max-model-turn stop；checkpoint 臂对应题
也都没有答对，所以 legal 的 +2 不是 correctness recovery。

## Checkpoint-used 对称子集

Checkpoint 出现在 positions 301、302、309、310、311、313、316、317：

- correct：7/8 vs 7/8；paired gain 0、regression 0
- strict：4/8 vs 6/8
- legal：8/8 vs 7/8
- turns：76 vs 53（+43.4%）
- tokens：508,153 vs 299,764（+69.5%）

其中 position 309 使用三次 commit，checkpoint 臂 19 turns 后合法但错误；disabled 臂在 4 turns
发生 provider failure，不能据此把 legal 改善归因于 checkpoint。Position 311 使用两次 commit，两个臂都正确，
但 checkpoint 臂多 2 turns、19,204 tokens。其余六题各 commit 一次。没有连续 accepted commit，也没有
restore 或 commit rejection。

在任一臂达到至少 10 turns 的四题对称子集中，两臂都是 1/4 correct；checkpoint 臂 60 turns、
573,628 tokens，disabled 臂 45 turns、397,747 tokens。当前 checkpoint 没有缓解长轨迹错误累积。

## 为什么没有发挥作用

1. Checkpoint 是上下文管理，不是语义验证。它能保留 active artifacts、目标和摘要，但不会判断此前的
   population、join、grain、duplicate 或 output-slot 决策是否正确；错误里程碑被 commit 后仍然是错误里程碑。
2. `semantic-v3-v24` 已使用 compact contract 与 recent-4 causal history，基础上下文负担较低。Commit 清空
   phase history的节省不足以抵消额外模型 turn、重新定向和重复确认。
3. Restore 被禁用后，checkpoint 不能回滚；它只做阶段边界。因而它无法直接修复“长轨迹后发现早期假设错误”这一
   原始目标，只能期待分阶段本身改善推理，本 Gate 没观察到这种改善。
4. 模型选中的 checkpoint 题确实更难，但在对称子集中 correctness 完全相同、strict 更差、成本显著更高，说明
   当前触发策略主要识别了复杂度，没有转化为更可靠的决策。

## 审计与处置

- 40/40 请求与所有 attempt 的 response model 均为 `deepseek-v4-flash`；endpoint 为官方 DeepSeek。
- 40/40 current structure、manifest/cohort/budget binding 与 fresh replay 通过。
- 没有 gold SQL、gold result 或未来状态进入 provider 输入；没有记录进入 SFT/RL。
- 保留 checkpoint 工具和 Harness snapshot 实现，不删除代码；冻结当前 v6 checkpoint-on 策略，不扩大本 ablation。
- 下一步若继续优化，应改变 checkpoint 的信息价值，而不是提高 commit 频率：例如在 commit 前产生 Harness 可验证、
  与答案契约直接相关的阶段证据或不确定性检查。任何新版本仍需与 checkpoint-disabled 做 fresh paired comparison。

