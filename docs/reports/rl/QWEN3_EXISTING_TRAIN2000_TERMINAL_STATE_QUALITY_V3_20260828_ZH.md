# 现有 BIRD-train 2000 轨迹的终态质量评分 v3

日期：2026-08-28

## 最终简化目标

根据既有 RL 实验中 backward slice / 调用依赖关系没有增益的结论，v3 完全删除依赖比例奖励、逐步骤 credit 和 `P_slice`。统一轨迹分数为：

\[
Q_{rank}(\tau)=
\begin{cases}
C(G,S_{terminal}), & \text{存在终态 evidence};\\
0.5\max_t C(G,S_t), & \text{没有终态 evidence}.
\end{cases}
\]

其中 `C` 仍是 Gold SQL 语义义务集合与确定性工具状态之间的分类宏平均 Jaccard。它只评价整个 artifact 的语义状态，不给单个工具或调用依赖边分数。

relation derivation 的输入 lineage 只用于确定性还原终态 artifact 已经包含的 source、join、predicate、grain、value、set、rank 和 output 语义。它是状态构造的一部分，不参与奖励加权，也不产生 backward-slice credit。

## 重算结果

使用与 v2 完全相同的 2000 条 BIRD-train 冻结轨迹重新计算：

| 指标 | v2 三项组合 | v3 无 dependency credit |
|---|---:|---:|
| 可评分轨迹 | 1701 | 1701 |
| 正确均值 | 0.7014 | 0.6780 |
| 错误均值 | 0.4843 | 0.4395 |
| 正确-错误均值差 | 0.2171 | **0.2386** |
| ROC-AUC | 0.7495 | **0.7550** |
| pooled Cohen's d | 0.9564 | **0.9825** |

v3 均值差 bootstrap 95% 区间为 `[0.2142, 0.2630]`，AUC 95% 区间为 `[0.7301, 0.7783]`。

纯 `C_terminal` 的 AUC 为 0.7560，比 v3 的 0.7550 略高。差异来自 9 条没有终态的错误轨迹使用了 `0.5*C_max` 回退；当前样本不足以验证该回退。后续同题多轨迹实验应把“纯终态 0 分”与“0.5 最大状态回退”作为一个单变量消融，不能把回退增益视为已成立。

12 条高/中/低分正确与错误轨迹的逐步人工审计进一步表明：该分数能把部分“小错误”排在“大错误”之前，
但会严重低估等价关系改写、工具查值后常量替代以及最终重新查询实体的路径；宏平均也可能稀释关键 output/predicate
错误。因此 v3 目前只能称为终态 Gold 结构接近度，不能直接称为统一轨迹质量或进入 RL。详见
`QWEN3_EXISTING_TRAIN2000_TERMINAL_STATE_QUALITY_V3_CASE_AUDIT_20260828_ZH.md`。

## 明确删除的内容

- 不计算 producer backward slice；
- 不统计依赖链覆盖率；
- 不按工具调用数量奖励或惩罚；
- 不把无用分支、错误恢复或调用依赖边作为 reward feature；
- 不做逐步骤 reward redistribution；
- 不按工具类型做均值校准。

添加一个与终态无关的成功分支现在不会改变终态质量分数。无终态时只比较成功 artifacts 的语义状态，不检查它们之间的调用依赖比例。

## RL 奖励接口

第一阶段仍是整轨迹 GRPO：

\[
R(\tau)=
\begin{cases}
+1, & \text{结果正确};\\
-1+\beta Q_{rank}(\tau), & \text{结果错误且可评分};\\
-1, & \text{不可评分}.
\end{cases}
\]

优势只在同一问题的 rollout group 内计算。当前仍不启动训练，因为 value/set 等价规范化和同题多轨迹排序门尚未通过。

## 产物

- 实现：`src/rl/tool_state_obligation_quality.py`
- 审计：`src/rl/diagnostics/audit_tool_state_obligation_quality.py`
- 测试：`src/rl/tests/test_tool_state_obligation_quality.py`
- 摘要：`data/results/existing_sft2k_terminal_state_quality_v3_20260828/summary.json`
- 逐轨迹分数：`data/results/existing_sft2k_terminal_state_quality_v3_20260828/trajectory_scores.jsonl`
- 统计分析：`data/results/existing_sft2k_terminal_state_quality_v3_20260828/validation_analysis.json`
- 身份清单：`data/results/existing_sft2k_terminal_state_quality_v3_20260828/manifest.json`

相关回归测试最终为 71/71 通过。模型调用、optimizer update 和 BIRD-dev 1534 使用数均为 0。

## 2026-08-28 扩展审计更新

后续已在全部 1701 条可评分轨迹上完成尾部统计，并把人工审计从 12 条扩展到 48 条，另对
value 降权的 8 个最大变化案例做了定向反例检查。结论是：只对错误轨迹保留 v3 排序；拒绝
value 乘法、output 位置弱化、relation-only 和工具均值校准；仅保留一个直接
`top-k -> base join -> terminal rows > Gold LIMIT` 的局部 rank-scope cap 作为待同题多 rollout
验证的候选。完整结果见
`QWEN3_EXISTING_TRAIN2000_TERMINAL_STATE_QUALITY_EXPANDED_V4_20260828_ZH.md`。
