# BIRD iterative-SQL v4 小样本优化 Gate15

日期：2026-08-05  
状态：已完成；仅为开发集诊断，不具备 SFT/RL 准入或总体推广资格

## 结论

在同一批 15 条 BIRD-train 任务、同一 DeepSeek v4 Flash、同一 recent-4、30 步和单次语义
尝试设置下，`iterative-sql-v3` 为 **7/15**，基于其失败审计形成的 `iterative-sql-v4` 为
**12/15**。两者均为 15/15 合法终止。配对结果为 6 个 v4-only gain、1 个 v3-only
regression、6 个 both correct、2 个 both wrong；exact two-sided paired `p=0.125`。

这个结果说明优化方向在该小样本上有效，但 v4 正是在看过这 15 条 v3 失败后设计并在同一
集合复测，因此 12/15 是开发集结果，不是独立 holdout 估计。它只把 v4 提升为下一轮独立
小门候选，不能据此扩到 200、进入 SFT/RL 或宣称替代 atomic。

## 冻结设置

- task source：`data/eval_inputs/bird_train_version38_semantic_discipline_gate32_20260729.jsonl`；
- task slice：`start=17, n=15`；
- task SHA-256：`0993665a2df7bc6447ee1b558119fa41a674251262b90c4130f5f22bc9a79cb8`；
- model：`deepseek-v4-flash`，官方 `https://api.deepseek.com`；
- context：`lazy-catalog-v1`，successful recent-4；
- max steps / errors per type：30 / 3；
- max tokens / provider retries：2,048 / 10；
- preview rows：20；terminal comparison：`bird-set`；
- 每题一次语义轨迹；wrong answer 不返回 gold 或 judge 差异。

外部请求只包含已授权的自然语言问题、external knowledge、数据库 catalog/schema 和逐步
只读工具反馈。gold SQL、gold rows 和隐藏验证器数据没有进入模型输入。结构审计再次检查了
所有 turn 的 model input，不存在这些隐藏键。

## v3 基线失败

v3 的 8 个 silent wrong answer 可分为：

1. 输出表示错误三题：男女计数被输出为 label/count 两行；两道 `full name refers to
   first_name, last_name` 被拼成单列字符串；
2. 未授权规格改写三题：自行 `ROUND` 百分比、自行增加 `EndDate IS NULL`、把外部公式的
   `COUNT(recipe_id)` 改成 distinct recipe；
3. 数据来源错误一题：在实体表找到同名 `StandardCost` 后立即停止，没有检查已声明关联的
   `ProductCostHistory`；
4. 排名边界错误一题：在多个并列 rental count 上按内部 ID 分组，得到与请求输出粒度不同的
   top-5 集合。

v3 没有 protocol failure 或非法终止。问题主要发生在 SQL 可执行之后的规格落实阶段，而不是
SQL 语法、只读安全或错误恢复。

## v4 改动

v4 保持两个公开工具及参数完全不变：

- `execute_sql(sql)`；
- `submit_sql(sql)`。

只改变以下模型可见策略：

1. prompt 明确 question + external knowledge 是 binding contract，并细化输出槽位、表示、
   formula/population、同名列来源和排名粒度规则；
2. 每轮最新 user message 追加原样 `TASK CONTRACT REMINDER`，避免长 SQL workspace 把题目和
   external knowledge 推到注意力远端；
3. 每次成功 `execute_sql` 增加确定性的 `query_shape_audit`，只报告结果列数以及 SQL 是否包含
   DISTINCT、ROUND、WHERE、IS NULL、GROUP BY、ORDER BY、LIMIT、join 数和聚合函数。它不解释
   问题、不判断正确性，也不访问 gold。

v3 的 prompt、interface 和 `64d865970246b3c9` protocol hash 保留为显式历史入口。

## 指标

| 指标 | iterative-sql v3 | iterative-sql v4 | 变化 |
|---|---:|---:|---:|
| verifier-correct | 7/15 | **12/15** | +5 |
| legal terminal | 15/15 | 15/15 | 0 |
| model actions | 97 | 91 | -6.2% |
| mean actions | 6.47 | 6.07 | -0.40 |
| process errors | 4 | 1 | -3 |
| total tokens | 341,223 | 346,664 | +1.6% |
| fresh replay correct | 7 | 12 | +5 |
| structural pass | 15/15 | 15/15 | 0 |

v4 对 v3 的六个恢复为任务 600、1050、2507、2512、2682、4869。唯一回退为 5544：
`current-terms` 对同一代表存在多行，v4 遵循“不无依据 DISTINCT”但没有先检查 join
multiplicity，最终 `COUNT(*)=6`，正确实体数为 3。

## 与已有工具的同题配对

| 方案 | correct | legal | actions | errors | tokens |
|---|---:|---:|---:|---:|---:|
| direct-sql-search v2 | 6/15 | 15/15 | 88 | 1 | 327,401 |
| iterative-sql v3 | 7/15 | 15/15 | 97 | 4 | 341,223 |
| **iterative-sql v4** | **12/15** | **15/15** | 91 | 1 | 346,664 |
| atomic version39 | 10/15 | 15/15 | 115 | 1 | 854,227 |

v4 对 direct-sql-search v2 为 7 gains / 1 regression，exact paired `p=0.0703125`；对 atomic
version39 为 2 gains / 0 regressions，exact paired `p=0.5`。样本太小，且 v4 使用了同一
开发集的失败信息，不能把 12/15 与 atomic 10/15 解读为总体能力领先。值得保留的工程信号是：
这批题上 v4 使用 atomic 约 40.6% 的 tokens，并且没有合法性损失。

## 剩余失败

- `bird_train_05161`：external formula 明写 `COUNT(recipe_id)`，模型仍根据自然语言“recipes”
  改成两个 `COUNT(DISTINCT recipe_id)`，得到 77.78 而非 joined-row grain 的 80.0；这是公式
  粒度服从问题。
- `bird_train_05544`：一对多 term join 未按代表实体去重，是 v4 唯一回退；下一版本若继续，
  应要求任何跨表 COUNT 在提交前比较 row count 与 answer-entity key count，但必须同时保留
  显式 external formula 的优先级。
- `bird_train_03724`：external knowledge 明写 `most rented refers to MAX(inventory_id)`，v4
  忠实执行后得到按最大 inventory id 排序的结果；隐藏 gold 却按 rental count 排序。不能通过
  prompt 同时满足二者。该题应标记为 external-knowledge/gold contract conflict，而不应训练
  模型忽略 external knowledge。

## 审计与产物

- v3：`data/trajectories/iterative_sql_20260805/holdout15_v3_flash_r1/`；
- v4：`data/trajectories/iterative_sql_20260805/holdout15_v4_flash_r1/`；
- 配对摘要：`data/trajectories/iterative_sql_20260805/paired_gate15_summary.json`；
- 两个目录内的 `replay_structural_audit.json` 均为 15/15 structural pass，所有 recorded
  correct 均在 fresh database replay 下保持 correct。

下一步若继续评测，应冻结 v4，在未参与这次 prompt 设计的新任务上运行独立 Gate15/30，预先
声明不再根据该 holdout 改 prompt。只有独立门保留明显净增益和 100% legal termination，才有
理由扩大规模。
