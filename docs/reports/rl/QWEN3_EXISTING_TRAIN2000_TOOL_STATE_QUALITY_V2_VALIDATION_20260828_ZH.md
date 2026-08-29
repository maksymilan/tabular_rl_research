# 现有 BIRD-train 2000 轨迹的工具状态质量评分 v2 验证

日期：2026-08-28

> 状态：历史消融记录。根据既有 RL 结论和本报告中 `P_slice` 的弱增益结果，dependency/backward-slice credit 已从后续 v3 正式删除；当前定义见 `QWEN3_EXISTING_TRAIN2000_TERMINAL_STATE_QUALITY_V3_20260828_ZH.md`。

## 结论

本次验证没有启动训练、没有调用模型，也没有使用 1534 条 BIRD-dev 评测集。输入是已经存在的两批 BIRD-train 轨迹，共 2000 个不同任务、每个任务恰好一条轨迹。

当前 v2 分数有真实的正确/错误区分能力，但**还不能作为正式 RL 奖励投入训练**：

- 正确轨迹均值为 0.7014，错误轨迹均值为 0.4843，差值 0.2171；
- 正确性 ROC-AUC 为 0.7495，bootstrap 95% 区间为 `[0.7247, 0.7728]`；
- 但 1239 条可评分正确轨迹中有 269 条低于 0.5，占 21.7%；
- 错误轨迹中，质量分数与独立答案误差的 Spearman 相关仅为 -0.1876，而且四分位误差没有严格单调下降；
- 正确轨迹按工具出现与否分组时，均值从 `set_op` 的 0.4085 到 `join_tables` 的 0.7782，范围达到 0.3697。该统计受任务类型混杂，不能证明工具本身导致偏差，但足以说明当前分数尚未消除工具/任务分布风险；
- `P_slice` 单独判断正确性的 AUC 只有 0.5601；把它加入总分后，AUC 从仅使用 `C_terminal` 的 0.7560 降到 0.7495。因此当前数据不支持把 `P_slice` 放入主排序分数，也不支持现在就做逐步骤 credit assignment。

因此应保留“整轨迹语义排序”方向，但将第一阶段目标进一步简化为终态语义覆盖；`C_max` 只作为没有终态时的有上限回退，暂时移除 `P_slice`。

## 被验证的精确定义

评分器完全不读取答案行、答案相似度、正确/错误标签、模型 reasoning、错误次数或恢复行为。

Gold SQL 与每个成功产出的 relation artifact 都被规范化为八类集合：

1. `source`
2. `join`
3. `predicate`
4. `grain`
5. `value`
6. `set`
7. `rank`
8. `output`

Jaccard 只用于同一语义类别中的规范化义务集合：

\[
J_k(G,S)=\frac{|G_k\cap S_k|}{|G_k\cup S_k|}.
\]

随后对并集非空的类别做等权宏平均：

\[
C(G,S)=\frac{1}{|K^+|}\sum_{k\in K^+}J_k(G,S).
\]

本次被验证的整轨迹分数为：

\[
Q_{v2}=0.7C_{terminal}+0.2C_{max}+0.1P_{slice}.
\]

- `C_terminal`：终态 evidence relation 的语义覆盖；无终态时为 0。
- `C_max`：轨迹全部成功 relation-producing artifacts 中的最大覆盖。
- `P_slice`：被选 artifact 的递归派生祖先中，成功 relation-producing 步数占全部成功 producer 步数的比例。

只有 `condition_filter`、`project`、`join_tables`、`group_aggregate`、`scalar_compute`、`extreme_value_select`、`set_op` 被计为 producer。感知、plan、reasoning、错误和恢复均不计分。

无法保守规范化时整条记录记为 `NA`，不记为 0，也不对剩余类别重新分配预设权重。

## 数据与覆盖率

| 项目 | 数量 |
|---|---:|
| 冻结 BIRD-train 输入 | 2000 |
| 正确 / 错误 | 1436 / 564 |
| 非语义运行故障排除 | 1 |
| 成功规范化 | 1999 |
| 可评分 | 1701 |
| 可评分率 | 85.09% |
| 可评分正确 / 错误 | 1239 / 462 |
| 模型调用 | 0 |
| optimizer update | 0 |
| BIRD-dev 1534 使用数 | 0 |

主要不可评分来源包括 Gold 子查询 158 条、重复物理表/自连接 91 条、根节点 UNION 6 条，以及少量无法唯一解析的历史 trajectory column lineage。多种原因可能出现在同一条记录中。

## 正确与错误的分离

| 指标 | 正确 | 错误 |
|---|---:|---:|
| 数量 | 1239 | 462 |
| 均值 | 0.7014 | 0.4843 |
| 中位数 | 0.7000 | 0.4987 |
| 标准差 | 0.2341 | 0.2062 |

均值差为 0.2171，bootstrap 95% 区间为 `[0.1939, 0.2399]`，pooled Cohen's d 为 0.9564。说明分数携带明显信号，但排序仍有大量重叠：260 条正确轨迹不高于错误中位数，74 条错误轨迹不低于正确中位数；288 条正确轨迹和 4 条错误轨迹取得满分。

因此“正确平均值高于错误”成立，但“正确轨迹稳定地远高于错误轨迹”尚不成立。

## 错误轨迹的错误幅度

使用旧审计已经计算出的答案误差 `1-answer_score` 作为**评分完成后的独立验证量**；该量没有进入 v2 分数。

错误轨迹从低质量到高质量四分位的平均答案误差依次为：

| 质量四分位 | 平均质量 | 平均答案误差 |
|---|---:|---:|
| Q1 | 0.2174 | 0.8398 |
| Q2 | 0.4098 | 0.7755 |
| Q3 | 0.5634 | 0.7935 |
| Q4 | 0.7452 | 0.7590 |

总体 Spearman 相关为 -0.1876。方向符合“高质量错误通常较小”，但关联很弱，且 Q3 比 Q2 更差，不满足强单调排序要求。

## 分量消融

| 分数 | 正确均值 | 错误均值 | ROC-AUC |
|---|---:|---:|---:|
| `C_terminal` | 0.6780 | 0.4350 | **0.7560** |
| `C_max` | 0.6856 | 0.4744 | 0.7349 |
| `P_slice` | 0.8968 | 0.8490 | 0.5601 |
| `C_terminal+C_max`，保持 7:2 比例 | 0.6797 | 0.4438 | 0.7531 |
| 完整 `Q_v2` | 0.7014 | 0.4843 | 0.7495 |

`C_terminal` 是当前最有效且最简单的分量。`C_max` 会让已经产生过较好中间表、但最终 evidence 较差的轨迹得到额外分数；当前数据没有显示这种补偿能改善排序。`P_slice` 主要接近 1，正确/错误之间差异很小，加入后降低判别力。

## 各语义类别的匹配

| 类别 | 正确均值 | 错误均值 |
|---|---:|---:|
| source | 0.8751 | 0.7663 |
| join | 0.6460 | 0.4508 |
| predicate | 0.5598 | 0.3703 |
| grain | 0.8539 | 0.6156 |
| value | **0.2878** | 0.1389 |
| set | **0.4593** | 0.1667 |
| rank | 0.6097 | 0.3054 |
| output | 0.6625 | 0.3160 |

最大的规范化缺口是 `value`。常见等价改写包括：

- Gold 的 `SUM(CASE WHEN p THEN 1 ELSE 0 END)`，轨迹实现为 `filter(p) -> COUNT(*)`；
- Gold 的 `COUNT(non-null-key)`，轨迹实现为 `COUNT(*)`；
- join 后对不同非空键计数，但在数据库约束下结果等价；
- Gold 直接写复合算术，轨迹通过多个 aggregate artifact 和 `scalar_compute` 完成；
- set/UNION 与多个分支工具实现之间的等价。

当前评分器只做语法无关的表达式展开和有限代数规范化，没有证明 NULL 性、外键基数或函数依赖，因此不能安全地把上述所有写法合并。盲目把 `COUNT(column)` 统一成 `COUNT(*)` 会把真正不同的查询误判为等价。

## 工具分布偏差检查

以下是“轨迹是否包含该工具”的观察性分组，不是工具本身的因果效应：

| producer 工具 | 正确数 | 正确均值 | 错误数 | 错误均值 |
|---|---:|---:|---:|---:|
| condition_filter | 1085 | 0.6952 | 404 | 0.4826 |
| project | 520 | 0.7587 | 232 | 0.5053 |
| join_tables | 745 | 0.7782 | 256 | 0.5690 |
| group_aggregate | 619 | 0.6394 | 272 | 0.4758 |
| scalar_compute | 151 | 0.5740 | 98 | 0.4585 |
| extreme_value_select | 255 | 0.7755 | 113 | 0.5055 |
| set_op | 8 | 0.4085 | 19 | 0.4265 |

这说明“取消固定工具分数”仍不足以自动消除分布差异。差异同时来自任务难度、Gold 语义类别和规范化覆盖，不能通过对每种工具做 z-score 来修正；工具归一化会反过来奖励选择低均值工具，形成新的 reward hacking。

正确的控制位置是**同一问题的 GRPO group 内**：所有候选轨迹共享同一个 Gold obligation graph，优势只在同题轨迹之间计算。这样跨任务的工具均值不会直接进入优势。不过当前数据每题只有一条轨迹，因此这个关键性质尚未得到经验验证，只得到设计层面的保证和确定性等价测试支持。

## 等价轨迹与单点错误测试

确定性测试共 10 条，全部通过：

- join 前 filter 与 join 后 filter 得到相同分数；
- 合并 AND filter 与拆成两次 filter 得到相同分数；
- aggregate 后用 `scalar_compute` 的表达式被完整内联；
- 仅修改输出 alias 不改变分数；
- 添加不在终态派生祖先中的成功分支，只降低 `P_slice`；
- 删除 predicate 只降低对应语义重叠；
- 单独修改 join type、aggregate operation 或 top-k 均严格降低总分。

现有 2000 池每题只有一条轨迹，无法从真实数据中构造“同题、同答案、不同路径”的经验配对。低分正确轨迹只能说明存在两类混合现象：一类是上述尚未支持的等价改写；另一类是仅在当前数据库实例上碰巧得到正确答案、但并不与 Gold SQL 语义等价的 shortcut。仅凭 denotation 无法把两者完全分开。

## 建议采用的下一版统一目标

第一阶段不要继续使用当前三项加权和，也不要做逐步骤奖励。建议使用：

\[
Q_{rank}(\tau)=
\begin{cases}
C(G,S_{terminal}), & \text{存在合法终态 evidence};\\
\eta\max_t C(G,S_t), & \text{没有合法终态 evidence}.
\end{cases}
\]

其中回退系数先固定为 `η=0.5`，只用于极少数未终止轨迹。本池只有 9 条可评分错误轨迹没有终态；`η=0.5` 的 AUC 为 0.7550，和纯 `C_terminal` 接近，因此该回退仍需在同题多轨迹池中验证，而不是据此宣称有效。

轨迹级 RL 奖励保持：

\[
R(\tau)=
\begin{cases}
+1, & \text{终态结果正确};\\
-1+\beta Q_{rank}(\tau), & \text{终态错误且可评分};\\
-1, & \text{不可评分}.
\end{cases}
\]

然后只在同一问题的 GRPO group 内标准化优势。`β` 仍可暂用 0.4 作为待验证超参，但在完成同题 K 轨迹验证前不应启动训练。

## 下一道实验门

正式 RL 前需要一个同题多轨迹排序池，而不是更多单轨迹任务：

1. 从 BIRD-train 选取固定任务，每题至少 4 条已有或新生成轨迹；
2. 不向 actor 暴露 Gold SQL；Gold 只在离线 scorer 中使用；
3. 检查同题内正确轨迹是否总排在错误轨迹前；
4. 对同题、同答案、不同合法实现做人工或规则审计，确认等价路径不过度分叉；
5. 对 conditional count、NULL-sensitive count、函数依赖、set/UNION 等未证明等价的情况直接记为 `NA`，不要猜测等价；
6. 只有在同题排序和工具路径不变性通过后，才运行 result-only GRPO 与 `Q_rank` GRPO 的 matched 对照。

## 产物

- 实现：`src/rl/tool_state_obligation_quality.py`
- 全量审计：`src/rl/diagnostics/audit_tool_state_obligation_quality.py`
- 二次统计：`src/rl/diagnostics/analyze_tool_state_quality_audit.py`
- 确定性测试：`src/rl/tests/test_tool_state_obligation_quality.py`
- 全量摘要：`data/results/existing_sft2k_tool_state_quality_v2_20260828/summary.json`
- 逐轨迹分数：`data/results/existing_sft2k_tool_state_quality_v2_20260828/trajectory_scores.jsonl`
- 消融与区间：`data/results/existing_sft2k_tool_state_quality_v2_20260828/validation_analysis.json`
- 身份清单：`data/results/existing_sft2k_tool_state_quality_v2_20260828/manifest.json`

相关测试命令最终为 70/70 通过。
