# Qwen3-8B Atomic v26 增强 4k 数据：epoch4 BIRD-dev1534 错误实例审计

日期：2026-08-29  
对象：会话 `019fe43c-f1ad-75e1-85dc-becdc6f2cd92` 产出的 `qwen3-v26-augmented-epoch4-cp6380`  评测轨迹  
范围：BIRD-dev1534，greedy，`temperature=0`，`max_tokens=2048`，`max_steps=30`，Atomic v26 / `think-json-v1`

## 1. 边界和输入

本报告是评测后只读诊断。dev 的 1534 条记录只用于本次错误分析；没有进入 SFT、RL、奖励设计、任务选择或任何优化更新。Gold SQL 只在本地评测器中用于核对终态义务和语义差异，未发送给模型。

完整逐题记录在：

- [`dev_incorrect_casebook.jsonl`](../../../data/results/qwen3_atomic_v26_augmented_epoch4_dev_error_audit_20260829/dev_incorrect_casebook.jsonl)
- [`dev_incorrect_casebook_index.md`](../../../data/results/qwen3_atomic_v26_augmented_epoch4_dev_error_audit_20260829/dev_incorrect_casebook_index.md)
- [`summary.json`](../../../data/results/qwen3_atomic_v26_augmented_epoch4_dev_error_audit_20260829/summary.json)

## 2. 结果概览

epoch4 为 **924/1534 = 60.23%**，合法终止 **1397/1534 = 91.07%**，剩余 610 条错误：

| 最终结果 | 条数 | 占全部错误 | 占全部 dev |
|---|---:|---:|---:|
| 合法但 `wrong_answer` | 473 | 77.54% | 30.83% |
| `protocol_error` | 121 | 19.84% | 7.89% |
| `execution_error` | 15 | 2.46% | 0.98% |
| `argument_validation_error` | 1 | 0.16% | 0.07% |
| 合计错误 | 610 | 100% | 39.77% |

和之前的 v26 SFT1 dev greedy（838/1534，54.63%）相比：

| 指标 | SFT1 | 增强 4k epoch4 | 变化 |
|---|---:|---:|---:|
| 正确 | 838 | 924 | +86 |
| 合法终止 | 1317 | 1397 | +80 |
| 最终 protocol error | 131 | 121 | -10 |
| 最终 execution error | 47 | 15 | -32 |
| 最终 argument validation error | 24 | 1 | -23 |
| max steps / context overflow | 14 / 1 | 0 / 0 | -15 |
| 合法 wrong answer | 479 | 473 | -6 |

因此 4k 数据的主要收益是减少非终态失败（尤其执行错误和参数错误），不是把合法轨迹的语义正确率显著推高。会话中 epoch3（918/1534）到 epoch4（924/1534）的配对比较只有 +6 条净正确，精确 McNemar `p=0.69634`，不能据此宣称再训练一个 epoch 带来可靠的语义增益。

## 3. 错误总体结构

### 3.1 合法错误的工具状态语义分层

对 473 条合法错误用当前工具状态义务评分器做只读分析。该分数是结构/义务差异，不是 denotation 正确性，也不能代替最终答案判定。

| 语义分组 | 条数 | 占合法错误 | 解释 |
|---|---:|---:|---|
| `wrong_source_or_join_route` + `partial_join_route` | 184 | 38.90% | 选择了错误基表、错误连接方向或不必要/缺失的关系路径 |
| `one_tail_near_miss` + `two_tail_near_miss` + `route_solved_multi_tail` | 185 | 39.11% | 数据源和大部分关系已对，尾部谓词、粒度、排序、值或输出列出错 |
| `semantic_unscorable` | 101 | 21.35% | 当前语义编译器覆盖不到，不能解释成“模型质量低” |
| `semantic_exact_but_wrong` | 3 | 0.63% | 现有 source/join/尾部义务全为 1，但实际 denotation 仍错，是评分器盲点 |

372/473 条合法错误有可计算的终态语义分数，均值 0.470；101 条（21.35%）没有可计算分数。因此当前分数不能直接作为所有合法错误的统一 RL reward。

### 3.2 评分为 0.8 以上不等于“几乎正确”

合法错误中有 39 条分数 `>=0.8`。其中有些确实是单一尾部错误，但也有会改变答案集合的严重问题；因此分数更适合作为排序/诊断信号，不能覆盖终态 denotation reward。

## 4. 逐题实例

### A. 当前协议错误的主因是思考截断，不是 JSON 小格式错误

`max_tokens=2048`。121 条最终 `protocol_error` 全部至少有一次未闭合的 `<think>`；其中 106/121 条三次连续尝试都是同一类未闭合输出。典型实例：

**例 41，`california_schools`**  
问题：按县找阅读平均分前 5 的 virtual schools。Gold SQL 使用 `RANK() OVER (PARTITION BY county ...)`。模型先读取表，随后连续三次输出很长的分析，没有闭合 `</think>`，没有产生工具 JSON，最终没有合法终态。

这里有两层问题：

1. 输出上限让“长思考”直接变成协议失败；
2. 当前 `extreme_value_select` 只有全局 top-k，没有“每个 group 的 top-k/rank”原子，模型在工具表达能力不足的任务上反复规划。

**例 31，`california_schools`**  
问题：最高 enrollment 的第 10、11 所学校的 eligible-free rate。模型已经得到 top-11，但在“取第 10、11 行”与输出之间不断长思考，三次协议错误后终止。这个任务不是数据库不可表达，而是已有中间状态没有被收束为可终止的操作。

**例 19，`california_schools`**  
问题：Math 平均分最高学校的电话。模型先在 `satscores` 上取全局最高，再和 `schools` 连接；最高 SAT 记录的 `cds` 在 `schools` 中没有匹配，于是反复检查格式，最后仍没有终态。正确路线应先 join 再排序：无匹配的最高 SAT 行不能先截断候选集。

**例 22，`california_schools`**  
问题：Contra Costa 测试人数最多的学校。Gold 直接在 `satscores.cname/sname` 上过滤和排序；模型却先过滤 `schools` 再按 CDS join，得到空表，随后长思考循环。这是“错误关系路径/过度连接”，不是单纯的协议熟练度问题。

因此，继续增加通用 SQL 思考文本不能直接解决这些错误；需要短思考约束、可表达性覆盖，和“先 join 再 top”这类因果工具轨迹监督。

### B. 合法但局部语义错：当前分数有用，但粒度/尾部必须单独看

**例 24，`california_schools`，分数 0.75，`one_tail_near_miss`**  
模型正确 join 了 `frpm` 和 `satscores`，但把“test takers whose test score is >=1500”解释成 `NumGE1500 >= 1500`。Gold 是 `NumGE1500 > 0`：`NumGE1500` 是达到 1500 分的人数，不是分数阈值。结果模型得到空集合，而 Gold 有 5 所学校。这个分数作为“源和连接已正确、谓词尾部错误”的诊断是合理的；若直接把 0.75 当成高 reward 则会掩盖空结果错误。

**例 15，`california_schools`，分数约 0.72，`two_tail_near_miss`**  
问题是“哪个 active district 的阅读平均分最高”。Gold 是先在学校行上取最大阅读分对应的 district；模型先按 district 求平均再取最大。关系路径正确，但聚合粒度完全不同。这是典型 `grain + rank` 错误，不应和单纯漏一个输出列混为一类。

**例 53，`california_schools`，分数 0.50，`route_solved_multi_tail`**  
Gold 返回每所 Fresno 邮寄城市学校的 `NumTstTakr` 列表：`[6, 5, 328, 335, 2216]`。模型在同一 join 结果上做全局 `SUM`，返回 `6070`。这里 source/join 完成，但把“逐行输出”错误地改成了“全局标量”。

**例 40，`california_schools`，分数 0.90，`one_tail_near_miss`**  
模型筛选和 join 都正确，却在“最低阅读分”上省略了 `ASC`，工具默认按降序取了最高者，电话从 Gold 的 `(559) 248-5100` 变成 `(559) 490-4290`。这是很局部的方向错误，0.90 的近失分基本合理。

**例 72，`california_schools`**  
问题可能返回两所 State Special School。Gold 是两行 `[40.0], [335.0]`；模型把两行相加返回 `[375.0]`。这再次说明“返回行粒度”是独立义务，不能仅以最终标量是否看起来合理来评分。

### C. 评分器当前的真正盲点：表达式、类型和 denotation

**例 13，`california_schools`，分数 1.0，`semantic_exact_but_wrong`**  
问题是 SAT excellence rate 前三学校电话。模型正确完成 join、投影、排序、top-3，但表达式写成 `NumGE1500 / NumTstTakr`，没有 CAST；SQLite 整数除法使大量比值变成 0，返回 Fresno 的三条电话，而 Gold 是另外三条。当前 source/join/尾部义务全为 1，却完全没有惩罚公式/类型语义。

**例 1276，`thrombosis_prediction`，分数 1.0，`semantic_exact_but_wrong`**  
Gold 使用文本列 `DNA >= 8` 的 SQLite 比较；模型先 `CAST(DNA AS REAL)` 再过滤。该数据列实际是 TEXT，CAST 改变了比较语义，模型返回 5 个诊断而 Gold 只有 `MCTD/SLE/SLE, SJS susp`。这是“工具动作合法但改变了数据库类型语义”。

**例 1277，`thrombosis_prediction`，分数 1.0，`semantic_exact_but_wrong`**  
Gold 对原始 `DNA < 8` 且 `Description IS NULL` 的患者计数为 4；模型把 DNA 转成 REAL 后计数为 0。NULL 与 TEXT/数值比较的交互没有进入当前语义义务评分。

**例 866，`formula_1`，分数约 0.93**  
模型正确筛选 race 161 的 `1:27%` lap time 并 join drivers，但随后错误地固定 `driverId=14`，只返回 David Coulthard；Gold 有 5 名驾驶员。评分器只看到少量 predicate 尾部差异，未反映“集合被硬截成单个 ID”的答案覆盖损失。

**例 926，`formula_1`，分数 0**  
问题要最快 lap time 的展示值。Gold 返回文本 `1:07.411`；模型取 `milliseconds` 的最小值，返回 `67411`。虽然这可以是同一物理量的另一表示，但不是要求的输出列/格式，说明 metric representation 也需要显式义务。

### D. 执行错误已大幅下降，但剩余的 15 条不是同一种问题

15 条最终 `execution_error` 的主要实例：

- **例 1、20、30、62、67、70**：带空格、括号、`%` 的列在 `inspect_column`、`condition_filter` 或 `extreme_value_select` 中触发 `near "Grade/Type/%"`、`no such function: Enrollment` 等错误；多次重复相同或等价动作。这里有明显的工具/SQL quoting 可用性问题，不能全部归咎于模型拟合不足。
- **例 431、506**：模型使用不存在的 `isFoilOnly` / `isNonFoilOnly` 列，属于 schema grounding 错误。
- **例 908、1027、1040**：聚合后的逻辑列被错误写成 `group_003.Player.player_api_id`、`top_003.races.raceId` 等，随后又触发重复 namespace/角色错误；属于 derived-column namespace 记忆和修复失败。
- **例 1163、1170**：使用 SQLite 不支持的 `YEAR/year` 函数，属于日期表达式能力/工具契约不一致。
- **例 1391**：对 `physics_count=0` 做 divide，且尝试使用不支持的 `read_subtable.offset`；属于零分母边界和无效分页动作。

这些最终执行错误中，很多 episode 在同一错误上重复三次，说明“收到错误后换动作”的恢复监督仍不够，而不只是首次动作错误。

## 5. 还存在的核心问题

1. **协议失败仍是非终态错误的主导项。** 4k 训练把最终 execution/argument 错误压低了，但非终态的 137 条中有 121 条最终是 protocol error；问题主要是长思考截断和在工具不可表达/空结果状态中无法收束。
2. **合法错误中，路径错误和粒度/尾部错误各占约 39%。** 仅奖励“调用了正确工具”会把 `SUM` 替代逐行输出、district 平均替代行级最大值等严重错误评为过好。
3. **当前 semantic score 有 101/473 条合法错误不可评分。** 编译器覆盖不足会产生缺失 reward；必须把 `unscorable` 和真实低质量分开。
4. **当前 source/join/尾部评分存在 denotation 盲点。** 公式、CAST、SQLite 类型比较、NULL、单位/表示、集合覆盖和列顺序尚未充分进入义务。例 13/1276/1277 都是分数 1 但答案错。
5. **训练集错误词汇覆盖不等于 dev 行为覆盖。** 610 条 dev 错误按当前较粗 signature 都能在训练错误库中找到相同或机制支持（604 条 exact signature），但训练错误中的 no-legal 只有 11/564，而 dev 仍有 137/610。说明现有签名过粗，不能据此认为模型已经学会了错误恢复。
6. **数据库之间仍有明显异质性。** epoch4 准确率最低的是 `thrombosis_prediction`（70/163，42.9%）和 `california_schools`（40/89，44.9%）；`formula_1` 为 96/174（55.2%），而 `superhero` 为 108/129（83.7%）。同一全局 reward 可能掩盖 schema、日期、文本类型和关系密度差异。

## 6. 对后续 RL 验证的直接含义

这批 dev 错误支持的结论是：先把目标拆成两个独立观察量，而不是把所有错误压成一个分数：

- **终态正确性**：仍以 Harness denotation 为唯一主奖励；
- **合法错误的结构排序**：只在同一题的候选轨迹之间，用 source/join、grain、predicate/rank/value/output、公式/类型和结果覆盖做近失排序。

协议错误要单独报告并单独处理；不能让一次长思考截断和一次局部谓词错误在同一 reward 标尺下竞争。当前最值得在训练集/新 on-policy rollout 上验证的方向是：

1. 短思考/强制尽快产出动作，减少 `</think>` 截断；
2. 对“先 join 再 top”“全局聚合 vs 逐行输出”“COUNT 阈值 vs 被计数值阈值”等高频结构做因果工具轨迹监督；
3. 在语义评分中补入可执行的表达式/类型/NULL/输出列与集合覆盖义务；
4. 对当前工具无法表达的 grouped top-k 等查询，增加明确的原子能力或将其作为工具覆盖缺口统计，而不是把模型的长思考失败当作普通语义错误。

这些改动必须在训练集或新的训练侧 on-policy rollout 上验证；本报告的 dev 记录保持评测只读。

