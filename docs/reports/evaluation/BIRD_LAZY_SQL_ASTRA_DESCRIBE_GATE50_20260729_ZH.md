# BIRD atomic lazy catalog + 按需列文本描述消融

日期：2026-07-29  
指标：`bird-set`  
模型：DeepSeek v4 Flash，temperature=0，thinking enabled，reasoning effort=high  
协议：atomic version24，rolling legal history=4，JSON Output，max_steps=30

## 结论

恢复原 atomic 工具和 lazy catalog，只在 `describe_table` 返回 BIRD/SQL-ASTRA 风格列文本
描述，结果仍没有超过历史基线：

| 条件 | 前 20 | 前 50 | legal | process errors | total tokens |
|---|---:|---:|---:|---:|---:|
| 原 lazy catalog baseline | 16/20 | **42/50** | 50/50 | 3 | 1,557,640 |
| lazy + describe 返回 semantic name | 14/20 | 40/50 | 50/50 | 1 | 1,641,786 |
| lazy + describe 返回 column description | **16/20** | 40/50 | 50/50 | 2 | 1,696,534 |
| 全 schema + semantic + description | 16/20 | 41/50 | 50/50 | 7 | 2,403,643 |

新条件相对原 baseline 在前 20 题逐题完全相同；前 50 题为 **1 gain / 3
regressions / net -2**，exact two-sided sign-test `p=0.625`。该结果没有扩大到固定 200
或改变默认环境的正向信号。

## 严格控制变量

profile：`lazy-catalog-sql-astra-describe-v1`

- atomic version24 工具、参数和执行语义不变；
- opening overview 与历史 lazy catalog 一样，只有表名、row count 和 FK；
- 初始 overview 没有列、semantic name、description 或 example values；
- version24 基础 system prompt 不做 lazy/full-schema 替换；
- 只有成功执行 `describe_table` 时，列对象增加 `description`；
- description 来自 BIRD `database_description/*.csv` 的 `column_description`；
- 不返回 `semantic_name`、`value_description` 或 live values；
- raw `table_name` / column `name` 仍是唯一可执行标识。

异名 CSV 只通过“同名且完整列集合一致”或“唯一完整列签名一致”映射，不做模糊猜测；
UTF-8-sig 解码失败时固定回退 CP1252。

## 配对变化

相对原 lazy baseline：

- gain `bird_train_05316`
  - 问题只要求最高评分鸡肉餐厅的 `label`；
  - baseline 把排序用 `review` 也返回；
  - 新条件只返回 `label`，修复最终输出形状。
- regression `bird_train_02408`
  - 问题只要求名字为 George 的 `author_name`；
  - 新条件没有使用 `return_columns`，额外返回 `author_id`。
- regression `bird_train_02901`
  - 问题要求最年轻的已婚男性生产技术员；
  - 新条件遗漏 `Gender='M'`，混入女性，属于问题约束遗漏。
- regression `bird_train_06454`
  - 问题要求回答 rail 或 mail 哪个更多；
  - 新条件返回两个类别及计数，没有再选择最大者，属于终端表粒度错误。

四个 discordant 样本中，三个直接涉及最终输出列/粒度，一个是显式条件遗漏；没有出现
“原本不知道列含义，加入 description 后完成 schema linking”的语义 gain。

相对“lazy + semantic name”条件，新条件在前 50 为 2 gains / 2 regressions / net 0；
总分同为 40/50。description 比短 semantic name 在前 20 更好（16/20 vs 14/20），但该
优势没有延续到 50 题。

## 成本与行为

相对原 lazy baseline：

- 首轮 user context 均为 1,920.5 chars，证明 description 没有提前塞入 opening；
- `describe_table` 调用均为 55 次；
- mean steps：5.88 → 6.10；
- API requests：295 → 309；
- total tokens：1,557,640 → 1,696,534，增加 **8.9%**；
- `inspect_column`：13 → 12；
- process errors：3 → 2；
- 发生三次 API transport retry 和一次 completion retry，均恢复，无最终 API failure。

相对全量 description 条件，按需版本把 total tokens 从 2,403,643 降到 1,696,534，
减少 **29.4%**，并把首轮 user context 从 13,218.6 chars 恢复到 1,920.5；但正确率
从 41/50 降到 40/50。按需加载解决了信息成本问题，没有解决能力问题。

## 判断

目前证据更支持：

1. lazy schema acquisition 是合理的效率边界；
2. BIRD 列描述可以作为可选观察字段保留，但不能当作涨点机制；
3. 当前主要失败仍是问题条件覆盖、关系/聚合决策和 exact-output discipline；
4. 如果继续研究描述，应冻结“raw 名确实歧义且 description 能唯一消歧”的目标题组，
   而不是再跑全量随机队列。

该 profile 保持 evaluation-only diagnostic，不进入默认 version26、SFT 或 RL。

## Artifacts

- 轨迹：
  `data/trajectories/schema_context_ablation_20260729/lazy_sql_astra_describe/gate50.*`
- 相对原 baseline：
  `data/trajectories/schema_context_ablation_20260729/paired_condition1_vs_lazy_sql_astra_describe.json`
- 相对 lazy semantic：
  `data/trajectories/schema_context_ablation_20260729/paired_lazy_semantic_vs_lazy_descriptions.json`
- 相对全量 description：
  `data/trajectories/schema_context_ablation_20260729/paired_full_vs_lazy_sql_astra_descriptions.json`

