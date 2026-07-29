# BIRD 工具代理 SQL-ASTRA 列文本描述消融：Gate20/Gate50

日期：2026-07-29  
指标：`bird-set`  
模型：DeepSeek v4 Flash，temperature=0，thinking enabled，reasoning effort=high  
协议基座：atomic version24，rolling legal history=4，JSON Output，max_steps=30

## 结论

把 SQL-ASTRA 案例中使用的 BIRD 列文本描述加入当前工具代理的完整初始 schema，**没有
提升这批冻结样本的正确率**：

| 条件 | 前 20 | 前 50 | legal | process errors | total tokens |
|---|---:|---:|---:|---:|---:|
| 条件 1：历史 lazy catalog | 16/20 | **42/50** | 50/50 | 3 | 1,557,640 |
| 条件 3：全 schema + BIRD 语义名 | 16/20 | **42/50** | 50/50 | 7 | 2,201,233 |
| 新条件：条件 3 + BIRD 列文本描述 | 16/20 | 41/50 | 50/50 | 7 | 2,403,643 |

新条件相对条件 3 在前 20 题逐题完全相同；前 50 题为 **0 gains / 1 regression / net
-1**，exact two-sided sign-test `p=1.0`。条件 3 的八个错误全部保留，没有任何难题被文本
描述修复。唯一新增错误 `bird_train_05316` 是最终输出形状错误，不是 schema
链接错误：问题只要求最高评分鸡肉餐厅的 `label`，模型把用于排序的 `review` 也保留在
最终表中。

因此不应把这个全量描述条件扩到固定 200，也不应据此修改当前默认环境或构造 SFT
数据。样本量不足以证明真实退化，但已经明确缺乏扩大实验所需的正向信号。

## SQL-ASTRA 实际提供了什么

SQL-ASTRA PDF Appendix F 的完整案例把以下信息同时放入用户 prompt：

- 完整 `CREATE TABLE` DDL；
- primary/foreign keys；
- BIRD 列文本描述，例如 `NCESDist` 的自然语言含义；
- 每列示例值。

本消融只测试“文本描述”这一新增变量：

- 继承条件 3 的 raw schema、PK/FK 和 BIRD `semantic_name`；
- 增加 BIRD `database_description/*.csv` 的 `column_description`；
- **不加入** `value_description`，也不加入 live `example_values`；
- prompt 明确描述只是理解元数据，不是可执行标识、观测值或谓词证据。

SQL-ASTRA 案例中的这些描述来自 BIRD 随数据库提供的 description CSV，不是本实验人工
补写。此前条件 4 已经单独测试每列最多两个 live 示例值，结果为 38/50，因此这里不把值
和描述重新混在一起。

## 描述文件映射与覆盖率

BIRD 的 description CSV 文件名不总等于 SQLite 表名。例如：

- `googleplaystore.csv` 对应表 `playstore`；
- `googleplaystore_user_reviews.csv` 对应表 `user_reviews`。

正式实现采用保守、确定性的两阶段映射：

1. 文件 stem 与表名相同，且完整列集合相同；
2. 否则只接受在该数据库内具有唯一完整列集合匹配的文件。

没有唯一匹配时不做模糊或语义猜测。CSV 解码固定为 UTF-8-sig 优先、CP1252 回退。
冻结 50 题包含 33 个数据库、332 个 description CSV；按题加权的 4,192 个模型可见列中，
3,850 个获得非空文本描述，覆盖率 **91.84%**。所用 description corpus SHA-256 为：

`bc558841d62b470eefa1bb303a18d152802d26e3164248ef7dad450cf34aa0c8`

实现提交为 `dd9b50a`。九个单元测试覆盖：字段隔离、无示例值、CP1252、异名文件的唯一
列签名映射、prompt 去歧义和 contract hash 唯一性。

## 配对审计

### 相对条件 3

| 前缀 | gains | regressions | net | exact p |
|---|---:|---:|---:|---:|
| 20 | 0 | 0 | 0 | 1.0000 |
| 50 | 0 | 1 | -1 | 1.0000 |

唯一 regression：

- `bird_train_05316`
  - 问题：`Which chicken restaurant has the highest review?`
  - gold 输出：只有 `label`
  - 条件 3：`extreme_value_select(..., return_columns=["label"])`
  - 描述条件：`return_columns=["label", "review"]`
  - BIRD 描述把 `label` 写成 “the label of the restaurant”，把 `review` 写成
    “the review of the restaurant”。它可能增加了排序列的显著性，但单个样本不能建立
    因果；能确定的是描述没有改善 schema 选择，反而没有守住最终输出槽。

### 相对条件 1

前 50 同样为 0 gains / 1 regression / net -1；regression 为
`bird_train_02901`。这是条件 3 本来相对条件 1 的回归。条件 3 对条件 1 的 gain
`bird_train_05316` 在加入描述后消失，所以新条件恰好保留了两者错误集合的并集。

## 成本与行为变化

相对条件 3：

- 首轮 user context：9,181.0 → 13,218.6 chars，增加 **44.0%**；
- total tokens：2,201,233 → 2,403,643，增加 **9.2%**；
- mean steps：5.66 → 5.44；
- API requests：283 → 272；
- `describe_table`：1 → 5；
- `inspect_column`：18 → 13；
- legal termination 保持 50/50，process errors 均为 7；
- 无最终 API/transport failure。

它和此前完整 schema 消融得到同一方向的结论：首轮给更多信息可以少走少量交互步骤，
但更宽的上下文并没有转化成语义或关系推理增益。

## 解释与下一步

这批题上，BIRD `semantic_name` 已经提供了大部分低层 schema linking 信号；更长的
`column_description` 多数只是重复解释。当前剩余错误主要在关系选择、聚合粒度、题意
约束和最终输出形状，列描述无法教授这些操作决策。

如果继续研究描述信息，优先做较小的目标化实验，而不是全量铺开：

1. 仅在 `describe_table` 返回当前所需表的文本描述，避免把全库描述塞进首轮；
2. 冻结只包含“raw/semantic 名仍有歧义、而 description 明确消歧”的题组；
3. 预先声明成功标准为配对语义 gain，且不得增加输出形状 regression。

## Artifacts

- 正式轨迹：
  `data/trajectories/schema_context_ablation_20260729/full_schema_semantic_sql_astra_descriptions/gate50.*`
- 相对条件 3：
  `data/trajectories/schema_context_ablation_20260729/paired_condition3_vs_sql_astra_descriptions.json`
- 相对条件 1：
  `data/trajectories/schema_context_ablation_20260729/paired_condition1_vs_sql_astra_descriptions.json`
- 首轮错误文件名映射运行（不进入结论）：
  `data/trajectories/schema_context_ablation_20260729/rejected_incomplete_description_mapping/`

所有轨迹均为 evaluation-only diagnostic，不是 SFT 数据源。
