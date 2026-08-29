# 现有 BIRD-train 错误轨迹扩展人工审计：新增 72 条

日期：2026-08-28

## 结论先行

本轮从 462 条可评分错误轨迹中，排除先前已审案例后，按六个分数区间各抽取 12 条，新增逐条审核
72 条。每条均实际检查了问题、Gold SQL、Gold/预测样例、完整工具参数、工具输出行数/样例、错误事件
和最终 evidence；没有用答案相似度或 correctness 自动生成严重度。

扩展结果否定了“当前分数已经可以直接用于全部错误轨迹排序”的判断：

- 新增 72 条上，分数与人工质量 Spearman 只有 **0.113**；
- 可比较 pair 的排序正确率为 **55.41%**，只略高于随机；
- 其中 40/72 存在问题—Gold—答案契约冲突、Gold SQL 语义问题或非唯一答案，占 **55.56%**；
- 去掉这 40 条后，剩余 32 条 Gold 相对可信案例的 Spearman 为 **0.447**，pairwise 排序正确率为
  **73.02%**。

所以当前主要问题不是再添加一种 value、join 或 step credit，而是：只有当 Gold 监督本身可信时，
终态语义接近度才表现出可用的错误严重度排序。当前没有一个可部署的 Gold 可靠性门，因此
现阶段仍不能启动这种 reward 的 RL 训练。

## 抽样方法

从错误轨迹按 v3 分数划分六层：

| 分数层 | 新增审核数 | 排除旧案例后可选数 |
|---|---:|---:|
| [0, 0.125) | 12 | 35 |
| [0.125, 0.25) | 12 | 47 |
| [0.25, 0.40) | 12 | 99 |
| [0.40, 0.60) | 12 | 148 |
| [0.60, 0.75) | 12 | 65 |
| [0.75, 1.00] | 12 | 30 |

层内使用固定 SHA-256 key `incorrect-manual-expansion-20260828:<task_id>` 排序后取前 12 条，
不是按人工判断挑例子。新增 72 条与原先同标准审核的 24 条错误案例合并后，共有 96 条兼容人工标注，
占 462 条可评分错误轨迹的 20.78%。

严重度定义：

- 0：轨迹没有实质错误，或主要是问题/Gold/答案契约冲突；
- 1：轻微错误；
- 2：中等错误；
- 3：严重错误。

“Gold 冲突”包含 Gold 明显错误、自然语言与 Gold 粒度/边界冲突、任意 LIMIT 导致多个合法答案、
以及名称/ID 等未由问题规定的输出约定；它不是总体数据集 Gold 错误率估计，因为本样本专门从被判错轨迹中分层。

## 新增 72 条总体结果

| 指标 | 数值 |
|---|---:|
| 严重度 0 | 35 |
| 严重度 1 | 10 |
| 严重度 2 | 17 |
| 严重度 3 | 10 |
| Gold/问题/答案契约冲突 | 40 |
| 当前分数低估 | 41 |
| 当前分数大体合理 | 19 |
| 当前分数高估 | 12 |

按分数层观察：

| 分数层 | 平均严重度 | 严重度≥2 | Gold 冲突 |
|---|---:|---:|---:|
| [0, 0.125) | 1.167 | 5/12 | 6/12 |
| [0.125, 0.25) | 1.250 | 5/12 | 7/12 |
| [0.25, 0.40) | 1.333 | 6/12 | 5/12 |
| [0.40, 0.60) | 0.667 | 4/12 | 8/12 |
| [0.60, 0.75) | 0.917 | 3/12 | 8/12 |
| [0.75, 1.00] | 0.833 | 4/12 | 6/12 |

整体没有稳定的单调下降。低分层同时包含严重错表/错粒度和几乎正确但与 Gold 路径不同的轨迹；
高分层没有严重度 3，但仍存在多个严重度 2 的关键错误。

## 分数实际有效与失效的位置

### Gold 相对可信时，错误排序有明显信号

排除 40 条 Gold 冲突后，32 条案例的 pairwise 排序达到 73.02%。典型顺序基本合理：

- 低分严重错误：`bird_train_01110` 错表且错粒度、`bird_train_00608` 取最早 encounter 且返回天数、
  `bird_train_02949` 把 Location 的 Subassembly 错当 Product；
- 中分局部错误：`bird_train_05892` 漏 `DISTINCT Path`、`bird_train_05031` 百分比分母范围错误、
  `bird_train_04837` EmailPromotion 类别取错；
- 高分局部错误：`bird_train_03211` 和 `bird_train_05015` 是整数除法，`bird_train_01105`
  取错一个 offense 指标列。

这支持把终态语义重合看成“Gold 可信条件下的错误接近度”，但不能跨过 Gold 可靠性条件直接使用。

### Gold 冲突使全体排序接近失效

40 条冲突覆盖多种不可由当前集合重合分数解决的问题：

- 非唯一结果：`any three`、`any ten`、`at least 15`，Gold 用无序 LIMIT 固定任意子集；
- 自然语言边界：`at least 2`、`19 and above`，Gold 分别使用 `>2`、`>19`；
- 实体与记录粒度：问题问 airplanes/patients/professors，Gold 数 landing/condition/teaching 行；
- 输出表示：问题问 coach/car/capital city，轨迹返回人名、车名、城市名，Gold 返回内部 ID；
- Gold SQL 本身漏条件或运算不合理：漏 Vintage Cars 过滤、SUM paragraph number、
  整数比率截断、SUM/COUNT(*) 把 NULL 隐式当零；
- 工具路径比 Gold 更符合问题：用 transaction 表示 consumed、用 geo_sea 表示 bordering provinces、
  排除 unemployed 来表示 employed。

这些不是通过调高 value 权重、加入 backward slice 或按工具均值校准可以修复的。

## 与先前 24 条合并

合并后的 96 条错误轨迹结果：

| 指标 | v3 | v3 + local rank cap |
|---|---:|---:|
| Spearman | 0.211 | 0.234 |
| pairwise 排序正确率 | 59.78% | 60.97% |

rank cap 的变化仍主要来自 `bird_train_00860` 一个已知 top-k 后一对多膨胀案例；新增 72 条没有
触发该规则。因此 rank cap 可以保留为明确的局部 invariant，但它不能解决总体排序问题。

## 对 RL 目标的调整

原候选目标是：结果正确统一 +1；错误轨迹使用
`-1 + beta * Q_wrong` 打破全错 rollout group 的平局。新增审计表明，在没有 Gold 可靠性门时，
这个目标仍会系统性强化 Gold 的错误边界、任意输出和错误粒度。

现在应把研究拆成两个串行门：

1. **Gold 可靠性门**：先判断该题能否用 Gold SQL 作为语义排序权威；
2. **错误轨迹排序门**：只在通过第一门的题上验证 `Q_wrong` 的同题 pairwise 排序。

第一门目前还没有可部署实现，人工 `gold_issue` 只能用于诊断，不能进入 reward。一个可扩展、定义明确的
下一候选是隐藏 Gold 的多解执行一致性：独立产生 (K) 个 SQL，全部由 Harness 执行；只有当至少
(m) 个独立 SQL 与 Gold 在 denotation、输出列数和类型上同时一致时，Gold 才获得排序资格。
生成文本本身不作事实权威，只有确定性执行结果参与门控。该方案必须另行验证，当前未调用模型、未生成数据。

在可靠性门完成前，统一结论是：

- result-only binary reward 仍是唯一已准入基线；
- v3 语义分数和 local rank cap 都保持 diagnostic-only；
- 不增加逐步骤 reward，不恢复 backward slice；
- 不按工具类型做均值校准；
- 不启动 RL 训练。

## 72 条逐项审核索引

详细中文判断和原始工具轨迹分别保存在人工标注 JSON 与 casebook。下表列出每条的分数、严重度、
Gold 冲突标记、分数判断和错误类型。

| task | 分数 | 严重度 | Gold 冲突 | 分数判断 | 类型 |
|---|---:|---:|---|---|---|
| `bird_train_04434` | 0.083 | 0 | 是 | underestimate | `gold_percentage_semantics_conflict` |
| `bird_train_02015` | 0.100 | 1 | 否 | underestimate | `extra_output_column` |
| `bird_train_02922` | 0.083 | 0 | 是 | underestimate | `gold_question_category_conflict` |
| `bird_train_01660` | 0.000 | 1 | 是 | underestimate | `gold_question_and_output_conflict` |
| `bird_train_01921` | 0.000 | 3 | 否 | reasonable | `early_abort_after_plan_error` |
| `bird_train_01106` | 0.000 | 0 | 是 | underestimate | `gold_output_identifier_conflict` |
| `bird_train_02150` | 0.060 | 2 | 否 | underestimate | `incomplete_after_correct_prefix` |
| `bird_train_01110` | 0.100 | 3 | 否 | reasonable | `wrong_source_and_grain` |
| `bird_train_00010` | 0.071 | 2 | 否 | reasonable | `wrong_scope_aggregation` |
| `bird_train_04877` | 0.083 | 2 | 否 | reasonable | `unjustified_global_aggregation` |
| `bird_train_05049` | 0.083 | 0 | 是 | underestimate | `gold_missing_requested_output` |
| `bird_train_02938` | 0.000 | 0 | 是 | underestimate | `gold_question_source_conflict` |
| `bird_train_01520` | 0.228 | 3 | 否 | reasonable | `wrong_fan_predicate_and_output_semantics` |
| `bird_train_04789` | 0.214 | 0 | 是 | underestimate | `gold_missing_requested_order_ids` |
| `bird_train_05356` | 0.208 | 3 | 否 | reasonable | `wrong_rating_source_and_grain` |
| `bird_train_05187` | 0.240 | 3 | 否 | reasonable | `wrong_store_semantics_and_percentage` |
| `bird_train_00608` | 0.179 | 3 | 否 | reasonable | `wrong_extreme_and_wrong_unit` |
| `bird_train_02855` | 0.194 | 0 | 是 | underestimate | `gold_average_grain_ambiguity` |
| `bird_train_00095` | 0.167 | 0 | 是 | underestimate | `gold_count_vs_rating_conflict` |
| `bird_train_03348` | 0.250 | 0 | 是 | underestimate | `gold_proportion_output_conflict` |
| `bird_train_04613` | 0.190 | 0 | 是 | underestimate | `gold_full_name_representation` |
| `bird_train_02949` | 0.200 | 3 | 否 | reasonable | `wrong_entity_and_source_tables` |
| `bird_train_06157` | 0.139 | 0 | 是 | underestimate | `gold_year_output_ambiguity` |
| `bird_train_05067` | 0.139 | 0 | 是 | underestimate | `gold_geographic_relation_conflict` |
| `bird_train_04091` | 0.361 | 0 | 是 | underestimate | `gold_null_weighting_conflict` |
| `bird_train_04708` | 0.333 | 0 | 是 | underestimate | `gold_flight_event_vs_airplane_grain` |
| `bird_train_01134` | 0.250 | 2 | 否 | reasonable | `integer_division_in_rank_predicate` |
| `bird_train_06491` | 0.300 | 0 | 是 | underestimate | `gold_literal_plural_mismatch` |
| `bird_train_05892` | 0.250 | 2 | 否 | reasonable | `missing_distinct` |
| `bird_train_03585` | 0.267 | 0 | 是 | underestimate | `gold_limit_scope_error` |
| `bird_train_03330` | 0.292 | 1 | 否 | underestimate | `missing_city_join_with_most_rows_correct` |
| `bird_train_04490` | 0.250 | 2 | 否 | reasonable | `wrong_yes_or_maybe_population` |
| `bird_train_03827` | 0.267 | 3 | 否 | reasonable | `wrong_event_source_and_reversed_age_predicate` |
| `bird_train_04051` | 0.319 | 3 | 否 | overestimate | `wrong_description_table_and_no_answer` |
| `bird_train_06044` | 0.250 | 1 | 是 | underestimate | `gold_team_season_vs_franchise_ambiguity` |
| `bird_train_04149` | 0.333 | 2 | 否 | reasonable | `incomplete_asia_region_scope_and_output` |
| `bird_train_03826` | 0.400 | 0 | 是 | underestimate | `gold_integer_seconds_and_layout_conflict` |
| `bird_train_01853` | 0.583 | 0 | 是 | underestimate | `gold_missing_movie_title` |
| `bird_train_00012` | 0.583 | 0 | 是 | underestimate | `gold_count_rows_vs_comment_value` |
| `bird_train_05517` | 0.500 | 0 | 是 | underestimate | `gold_employed_unemployed_polarity_conflict` |
| `bird_train_04008` | 0.500 | 0 | 是 | underestimate | `gold_null_and_duplicate_weighting_conflict` |
| `bird_train_04330` | 0.533 | 2 | 否 | overestimate | `correct_author_wrong_null_titles` |
| `bird_train_05379` | 0.467 | 0 | 是 | underestimate | `gold_ignores_fewer_and_region_request` |
| `bird_train_05125` | 0.400 | 2 | 否 | reasonable | `wrong_category_literal` |
| `bird_train_04837` | 0.472 | 2 | 否 | reasonable | `wrong_email_promotion_category` |
| `bird_train_00266` | 0.500 | 0 | 是 | underestimate | `gold_aggregate_without_group_output` |
| `bird_train_06369` | 0.500 | 0 | 是 | underestimate | `gold_all_history_vs_latest_inspection` |
| `bird_train_05031` | 0.400 | 2 | 否 | reasonable | `wrong_percentage_denominator` |
| `bird_train_05595` | 0.667 | 2 | 是 | overestimate | `overly_restrictive_title_pattern` |
| `bird_train_01611` | 0.600 | 1 | 是 | reasonable | `unique_word_vs_token_share_ambiguity` |
| `bird_train_01402` | 0.667 | 1 | 否 | overestimate | `returns_all_instead_of_any_one` |
| `bird_train_02176` | 0.689 | 0 | 是 | underestimate | `gold_inventory_vs_consumption_events` |
| `bird_train_03445` | 0.600 | 1 | 否 | reasonable | `extra_identifier_columns` |
| `bird_train_00643` | 0.650 | 3 | 否 | overestimate | `wrong_credit_threshold_and_missing_job_filter` |
| `bird_train_03974` | 0.667 | 0 | 是 | underestimate | `gold_integer_ratio_truncation` |
| `bird_train_06334` | 0.625 | 1 | 是 | underestimate | `multiple_valid_three_with_extra_columns` |
| `bird_train_01182` | 0.667 | 0 | 是 | underestimate | `gold_sum_paragraph_numbers` |
| `bird_train_00451` | 0.600 | 0 | 是 | underestimate | `multiple_valid_at_least_fifteen` |
| `bird_train_03523` | 0.722 | 2 | 否 | overestimate | `wrong_business_grouping_key` |
| `bird_train_02477` | 0.600 | 0 | 是 | underestimate | `gold_boundary_operator_conflict` |
| `bird_train_01240` | 0.833 | 0 | 是 | underestimate | `gold_car_id_vs_name_output` |
| `bird_train_05994` | 0.800 | 0 | 是 | underestimate | `country_long_name_vs_short_name` |
| `bird_train_01500` | 0.773 | 2 | 否 | overestimate | `arbitrary_first_business_and_attribute` |
| `bird_train_05166` | 0.867 | 2 | 否 | overestimate | `formatted_salary_string_comparison` |
| `bird_train_05015` | 1.000 | 1 | 否 | overestimate | `integer_division_accumulation` |
| `bird_train_01105` | 0.875 | 2 | 否 | overestimate | `wrong_offense_metric_column` |
| `bird_train_03846` | 0.900 | 0 | 是 | underestimate | `multiple_valid_any_ten` |
| `bird_train_03119` | 0.767 | 0 | 是 | underestimate | `gold_capital_id_vs_city_name` |
| `bird_train_00653` | 0.769 | 0 | 是 | underestimate | `gold_missing_vintage_cars_predicate` |
| `bird_train_03211` | 1.000 | 1 | 否 | overestimate | `integer_division_truncation` |
| `bird_train_00934` | 0.833 | 2 | 否 | overestimate | `missing_top_k_returns_all_ranked_genres` |
| `bird_train_06021` | 0.867 | 0 | 是 | underestimate | `gold_at_least_boundary_error` |

## 产物

- 固定分层选择：`data/results/existing_sft2k_terminal_state_quality_v4_guards_20260828/incorrect_expansion72_selection.json`
- 72 条完整 casebook：同目录 `incorrect_expansion72_casebook.json`
- 72 条逐项人工判断：同目录 `incorrect_expansion72_manual_audit.json`
- 汇总与可复核行：同目录 `incorrect_expansion72_manual_summary.json`、
  `incorrect_expansion72_manual_rows.json`
- 选择和分析代码：`src/rl/diagnostics/select_incorrect_manual_audit_expansion.py`、
  `analyze_incorrect_manual_audit_expansion.py`

本轮模型调用 0、optimizer update 0、BIRD-dev 1534 使用 0。
