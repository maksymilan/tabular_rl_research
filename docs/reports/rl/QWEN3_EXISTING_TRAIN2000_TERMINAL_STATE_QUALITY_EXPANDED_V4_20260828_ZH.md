# 现有 BIRD-train 2000 轨迹质量评分：扩展审计与 v4 调整

日期：2026-08-28

## 结论

本轮没有训练模型，也没有使用 BIRD-dev 1534。实验只读取冻结的 BIRD-train 2000
既有轨迹；其中 1701 条可由当前语义状态编译器评分，包含 1239 条结果正确轨迹和
462 条结果错误轨迹。

最初 48 条分层人工审计曾支持一个较窄的候选结论：终态语义重合度可能在错误轨迹内部提供整体
排序。随后又新增逐条审核 72 条错误轨迹，结果推翻了“可以直接把该分数用于全部错误轨迹 reward”
的判断：新增样本的 Spearman 只有 **0.113**，pairwise 排序正确率只有 **55.41%**。

新增 72 条中有 40 条存在题意—Gold—答案契约冲突、Gold SQL 语义问题或非唯一答案。排除这些
诊断案例后，剩余 32 条 Gold 相对可信轨迹的 Spearman 为 **0.447**，pairwise 排序正确率为
**73.02%**。因此分数的准确定义应是“Gold 可信条件下的错误接近度”，而不是可直接部署的轨迹
质量 reward。当前尚无可部署的 Gold 可靠性门，不能启动这种 reward 的 RL 训练。

局部 rank-scope 上限、value 表达式重合度乘法、output 位置弱化、relation-only 分数和按工具
均值校准均不能解决这一主要问题；前者只保留为 diagnostic invariant，其余方案继续拒绝。

## 72 条扩展前的候选 RL 目标（现未准入）

令 (C_{terminal}(\tau)\in[0,1]) 为 Gold SQL 与最终 evidence artifact 在
source、join、predicate、grain、value、set、rank、output 八类语义义务上的终态重合度。
它是整个终态的一个分数，不给任何单个工具调用发奖励。

定义一个局部、可执行验证的 rank-scope 事件 (V_{rank})：

1. Gold SQL 在完整查询末尾有字面量 `LIMIT k`；
2. 轨迹先执行 `extreme_value_select(top_k=k)`；
3. 该工具产生的确切 table handle 随后被直接作为 `join_tables.base`；
4. 沿后续显式单表变换正向跟踪到最终 evidence，终态仍有 (n>k) 行。

该条件只读取工具参数、工具输出行数和最终 evidence handle；不读取模型 reasoning，
不计算 backward slice，也不产生逐步骤 credit。候选错误轨迹质量为：

\[
Q_{wrong}(\tau)=
\begin{cases}
\min\left(C_{terminal}(\tau),\frac{k}{n}\right), & V_{rank};\\
C_{terminal}(\tau), & \text{otherwise}.
\end{cases}
\]

当时提出的统一轨迹奖励候选为：

\[
R(\tau)=
\begin{cases}
+1, & \text{结果正确};\\
-1+\beta Q_{wrong}(\tau), & \text{结果错误且语义可评分};\\
-1, & \text{结果错误且不可评分},
\end{cases}
\qquad 0<\beta<1.
\]

优势仍只在同一问题的 rollout group 内计算。这个定义有三个直接作用：

- 所有正确轨迹都高于所有错误轨迹，不会因与某一条 Gold SQL 路径不同而抑制正确的等价解；
- 当同组 rollout 全部错误、二值 GRPO 没有梯度时，(Q_{wrong}) 才提供“错误较小”的排序；
- 没有逐轮奖励，因此当前实验不声称解决 step-level credit assignment。

这里的 \(\beta\) 尚未选择。即使在新增 72 条审核之前，本轮数据也是一题一条冻结轨迹，无法验证
GRPO 同题组内标准化后的实际优化行为；新增审核进一步表明该公式缺少必要的 Gold 可靠性前置门。
因此这段公式现在仅作为被审计的研究假设保留，不是已准入 reward。

## 后续 72 条错误轨迹扩展审核：覆盖前述候选结论

在排除先前审核案例后，从 462 条可评分错误轨迹的六个分数区间各固定抽取 12 条，共 72 条；
每条均人工检查问题、Gold SQL、Gold/预测样例、完整工具参数、工具输出、错误事件和最终 evidence。

| 审核集合 | 数量 | Spearman | pairwise 排序正确率 |
|---|---:|---:|---:|
| 新增全部错误轨迹 | 72 | 0.113 | 55.41% |
| 新增且无 Gold 冲突 | 32 | 0.447 | 73.02% |
| 与先前兼容错误案例合并 | 96 | 0.211 | 59.78% |

新增 72 条的人工严重度为：0 级 35 条、1 级 10 条、2 级 17 条、3 级 10 条。40/72 的 Gold
冲突率仅描述这个“benchmark 判错轨迹”的分层诊断样本，不能外推成总体 Gold 错误率。它揭示的
是条件性：Gold 可靠时分数有排序信号，Gold 不可靠时该信号接近失效。新增样本没有触发一次
rank-scope cap；合并 96 条中也只有先前已知的一个案例触发，所以该 cap 不是总体修复。

完整的 72 条逐项中文判断见
`docs/reports/rl/QWEN3_EXISTING_TRAIN2000_INCORRECT_TRAJECTORY_MANUAL_AUDIT_72_20260828_ZH.md`。

## 一、1701 条全量统计

v3 原始终态分数的总体结果为：

| 指标 | 数值 |
|---|---:|
| 正确轨迹 | 1239 |
| 错误轨迹 | 462 |
| 正确均值 | 0.6780 |
| 错误均值 | 0.4395 |
| 均值差 | 0.2386 |
| 正确性 ROC-AUC（仅用于验证） | 0.7550 |

分数与正确率总体单调：

| 分数区间 | 样本数 | 正确率 |
|---|---:|---:|
| [0, 0.125) | 49 | 22.45% |
| [0.125, 0.25) | 74 | 32.43% |
| [0.25, 0.4) | 261 | 60.15% |
| [0.4, 0.6) | 423 | 63.36% |
| [0.6, 0.75) | 334 | 77.84% |
| [0.75, 0.9) | 222 | 86.94% |
| [0.9, 1.0] | 338 | 96.45% |

这说明信号不是随机噪声，但 AUC 只能说明跨题相关性。GRPO 需要的是同一问题多条 rollout
之间的排序；当前冻结池每题只有一条轨迹，因此尚不能据此声称比 result-only GRPO 更好。

### 两个尾部问题

- 192 条正确轨迹低于 0.40。它们最常见的零分类是 join 170 条、predicate 123 条、output
  119 条和 value 103 条。人工审计显示，其中相当一部分是查出 ID 后用常量替代、去掉冗余 join、
  filter+count 替代 conditional aggregate 等有效改写。
- 41 条错误轨迹高于或等于 0.75。其结构义务通常基本正确；多数错误来自边界、表示、输出约定，
  或问题与 Gold 本身存在冲突，而不是完全跑错表。

因此，(C_{terminal}) 的准确名称是“Gold 终态语义接近度”，不是无条件的“推理正确率”。

## 二、48 条分层人工审计

从正确/错误各自的高、中、低分段各取 8 条，共 48 条，并逐条检查问题、Gold SQL、工具参数、
工具输出、最终 evidence 和真实错误程度。严重度定义为：0=没有实质错误或问题/Gold 冲突，
1=轻微，2=中等，3=严重。该样本用于寻找失败模式，不代表总体发生率。

| 分层 | 数量 | 平均严重度 | 严重度分布 |
|---|---:|---:|---|
| 正确高分 | 8 | 0.000 | 8×0 |
| 正确中分 | 8 | 0.625 | 3×0，5×1 |
| 正确低分 | 8 | 1.250 | 2×0，3×1，2×2，1×3 |
| 错误高分 | 8 | 0.500 | 5×0，2×1，1×2 |
| 错误中分 | 8 | 1.250 | 2×0，4×1，2×3 |
| 错误低分 | 8 | 2.375 | 2×1，1×2，5×3 |

原始分数与人工质量的 Spearman 相关为：全体 0.631，错误轨迹 0.622。总体方向正确，
但仍存在关键反例。

### 正确高分：基本合理

8/8 都没有实质错误，说明 1.0 或接近 1.0 的正确轨迹通常确实完成了相同的数据源、约束、
运算和输出契约。

### 正确中低分：混合了好替代路径和真实脆弱性

- `bird_train_04195` 用 filter + `count_distinct` 替代 Gold 的 conditional sum，结果和语义正确；
- `bird_train_02167` 去掉了不影响结果的冗余 join；
- `bird_train_05254` 用 `in_table` 完成成员过滤，是工具机制本身支持的有效路径；
- `bird_train_00502` 与 `bird_train_01804` 先通过工具观察确定 ID，再继续查询，属于可行但终态
  lineage 不能完整表达的路径；
- `bird_train_04442` 的公式只在当前实例上碰巧相等；
- `bird_train_02826` 找不到目标实体后换了另一个人，因两者结果都为空而碰巧正确，属于严重的
  spurious success。

因此不能把正确轨迹按 Gold 重合度继续分出正负 advantage；否则会同时压制有效替代解和偶然正确解，
而当前分数没有能力稳定区分二者。

### 错误高分：多数是小差异或 Gold 冲突

- `bird_train_04949` 对“任意五个”返回了另一组合法五行；
- `bird_train_01227` 按自然语言的严格大于/小于执行，而 Gold 使用包含边界的 `BETWEEN`；
- `bird_train_04674` 按客户去重后求平均，Gold 按 shipment 行加权；自然语言更支持轨迹；
- `bird_train_06231` 正确过滤 Frozen 的 Elsa 演员，Gold 却返回 Frozen 的全部演员；
- `bird_train_05993` 输出国家名称，Gold 输出国家代码；
- `bird_train_02656` 仅有 `>100` 与 `>=100` 的边界差异。

这些高分错误的实际偏差普遍小于低分错误。说明“用语义接近度给错误轨迹排序”有研究价值，
也说明它不能替代二值结果正确性。

### 错误低分：多数确实严重，但仍有输出表示反例

错误低分中 5/8 为严重错误，主要是错表、错 source、错 join 或错 grain，例如
`bird_train_01904`、`bird_train_00145`、`bird_train_01156`、`bird_train_02129` 和
`bird_train_00991`。但 `bird_train_02570` 已找到正确名字，只是把两个目标列拼成一个字符串列，
分数仍会低估这种输出表示错误。

## 三、发现的两个“宏平均稀释关键错误”反例

### `bird_train_05299`：关键数值错误被平均到 0.667

轨迹计算平均年龄时直接用死亡日期参与表达式，没有正确处理仍在世记录的 NULL，最后得到
约 -3446 的荒谬平均年龄。source、predicate、grain 等类别仍然匹配，导致 value=0 的决定性错误
在分类宏平均中被稀释，最终分数仍为 0.667。

### `bird_train_00860`：分数 1.0，但 LIMIT 作用域错误

Gold 在 torrents 与 tags 完成 join 后排序并 `LIMIT 1`。轨迹先对 live album 做 `top_k=1`，
再与一对多 tags 表连接，把一行扩成四行并全部作为最终 evidence。source、join、predicate、rank、
output 的集合都相同，因此集合式终态分数为 1.0，却没有表达“LIMIT 在 join 前还是 join 后”。

`bird_train_01445` 也出现同类事件：先选出一个 business，再连接其四个 category，最终返回四行，
而 Gold 结果只有一行。这个实例的 Gold 单一 category 本身有一定任意性，但终态 cardinality 契约
确实不一致。

这两个案例促成了 (V_{rank}) 的严格定义。它不是“好状态被破坏”之类不可执行判断，而是四个
可直接检查的条件。

## 四、候选修正的全量消融

| 分数 | 正确均值 | 错误均值 | AUC | 决策 |
|---|---:|---:|---:|---|
| v3 原始 | 0.6780 | 0.4395 | 0.7550 | 保留为基础 |
| value 软乘法 | 0.6040 | 0.3506 | 0.7349 | 拒绝 |
| rank soft factor | 0.6780 | 0.4383 | 0.7562 | 仅诊断 |
| rank cap 候选 | 0.6780 | 0.4375 | 0.7570 | 保留候选 |
| value + rank | 0.6040 | 0.3494 | 0.7362 | 拒绝 |

rank-scope 条件在 1701 条中只触发 2 条，均为错误轨迹，正确轨迹触发 0 条。它把
`bird_train_00860` 从 1.0 限制为 0.25，把 `bird_train_01445` 从 0.389 限制为 0.25。
48 条人工集上，rank cap 后错误轨迹 Spearman 从 0.622 升到 0.723；但其中只有一个触发样本，
不能把这一变化解释为广泛增益。

### 为什么拒绝 value 乘法

全量 AUC 已明显下降。为避免只依赖一个反例，又定向抽取 value 乘法降幅最大的 8 条错误轨迹：

- 只有 2/8 是明确的真实错误：用 `rental_rate` 代替 `replacement_cost`，以及把 City Name
  错当 State；
- 5/8 是问题与 Gold 的计数粒度冲突。问题问 students、professors、patients、cars 或 competitors，
  轨迹使用 distinct 实体计数，Gold 则数 registration、teaching、condition、join 或 participation 行；
- 1/8 主体答案正确，只是多输出了支持排序的 count 列。

所以 value overlap 测到的是“是否使用 Gold 的表达式树”，不是稳定的数值推理质量。即使只在错误轨迹
内部使用，也会把自然语言上更合理的解排低，不能进入候选 reward。

### 为什么不修改 output 为位置不敏感

把 output 改成更重内容、较轻位置后，只有 45 条受影响，其中正确 7 条、错误 38 条；AUC 从
0.7550 降到 0.7286。输出列位置、列数和表示正是 benchmark 的重要终态契约，不能整体弱化。

### 为什么不改成 relation-only

relation-only AUC 为 0.6972，比原始 0.7550 更差。只看 source/join/predicate/grain 会放过
错误公式、错误输出和错误 rank，不能解决当前问题。

## 五、不同工具平均分偏差如何处理

全量观察中，携带不同工具的正确轨迹平均分差异很大，例如 join 轨迹约 0.759、scalar_compute
约 0.533、set_op 约 0.335。这些数值不能用于工具校准，因为工具是否出现由题目语义决定，且
set_op 只有 8 条正确样本。把每个工具减均值或除方差会把“选择某工具”本身变成奖励，并可能鼓励模型
绕开低均值工具或滥用高均值工具。

当前方案通过三点避免这种分布偏差：

1. 不定义任何 tool-type base reward；
2. 只比较工具执行后形成的统一语义状态，不比较工具名称；
3. GRPO advantage 只在同一问题的 rollout group 内计算，同题 Gold 义务相同，不做跨题全局工具均值归一化。

如果某工具长期偏低，应该修复该工具输出的语义规范化或增加确定性等价规则，而不是人为把它的分数抬高。
本轮 value 乘法反例已经证明，未经验证的“校准”会把 Gold 偏好放大。

## 六、当前研究判断与下一验证门

1701 条跨题统计只能证明分数不是随机噪声；新增 72 条逐项审核表明，在全部 benchmark-wrong
轨迹上，它还不能稳定表示真实错误严重度。尤其是 Gold 冲突会把合理替代答案排低，甚至鼓励模型
学习 Gold 的错误边界、错误粒度或任意 LIMIT 输出。

因此当前状态是：

- result-only binary reward 仍是唯一已准入基线；
- v3 (C_{terminal}) 只保留为 Gold 可信条件下的 diagnostic score；
- local rank cap 只保留为局部 diagnostic invariant，不作为总体修复；
- value 乘法、output 弱化、relation-only 和工具均值校准继续拒绝；
- 不启动训练，不增加逐步骤 reward，不恢复 backward slice。

后续必须先通过两个串行验证门。第一是 **Gold 可靠性门**：在不让 actor 看 Gold 的前提下，用多个
独立候选 SQL 的确定性执行结果检查 Gold 的 denotation、输出列数和类型是否得到足够一致支持；人工
`gold_issue` 只用于诊断，不能进入 reward。第二是在通过可靠性门的 BIRD-train 题上生成同题多条
冻结 rollout，验证 v3 对同题错误轨迹的 pairwise 排序。只有两门都通过，才值得与 result-only
GRPO 做匹配训练比较；step-level credit assignment 仍留作之后的独立研究。

## 产物

- 全量失败分析：`data/results/existing_sft2k_terminal_state_quality_v3_20260828/expanded_failure_analysis.json`
- 48 条分层 casebook：`data/results/existing_sft2k_terminal_state_quality_v3_20260828/casebook_expanded_48.json`
- 48 条人工标注与汇总：同目录下 `expanded_case_manual_audit_48.json`、
  `expanded_case_manual_audit_summary.json`
- v4 全量消融：`data/results/existing_sft2k_terminal_state_quality_v4_guards_20260828/guarded_summary.json`
- 两条 rank-scope casebook：同目录下 `rank_expansion_casebook_2.json`
- value 定向 8 条审计：同目录下 `value_guard_top8_casebook.json`、
  `value_guard_top8_manual_audit.json`、`value_guard_top8_manual_summary.json`
- 新增 72 条错误轨迹的固定选择、完整 casebook、逐条标注和汇总：
  `data/results/existing_sft2k_terminal_state_quality_v4_guards_20260828/incorrect_expansion72_*`
- 新增 72 条逐项报告：
  `docs/reports/rl/QWEN3_EXISTING_TRAIN2000_INCORRECT_TRAJECTORY_MANUAL_AUDIT_72_20260828_ZH.md`
- 分析与测试：`src/rl/diagnostics/analyze_terminal_state_quality_failures.py`、
  `analyze_expanded_quality_case_audit.py`、`audit_terminal_state_quality_v4_guards.py`、
  `select_incorrect_manual_audit_expansion.py`、`analyze_incorrect_manual_audit_expansion.py`

本轮模型调用 0，optimizer update 0，BIRD-dev 1534 使用 0。相关测试 20/20 通过。
