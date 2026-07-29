# 交互式 SQL 完整 Schema/示例值 Gate32 审计

## 结论

在相同冻结32题上，把初始数据库信息从懒加载 catalog 扩充为：

- 全部表名和行数；
- 每张表的完整列名、类型、NOT NULL 和主键标记；
- 每列最多2个真实、非空、distinct 示例值；
- 数据库声明的外键关系；

并没有提高 DeepSeek v4 Flash 的最终正确率：

| 初始环境信息 | 正确 | 合法终止 | 平均动作 | 总 tokens |
|---|---:|---:|---:|---:|
| `lazy-catalog-v1` | 10/32 | 32/32 | 7.47 | 538,304 |
| `full-schema-samples-v1` | 10/32 | 31/32 | 5.00 | 822,723 |

逐题配对发生2个恢复和2个回归，净变化为0，exact two-sided paired binomial
`p=1.0`。因此，在这组任务上，观察到的“环境信息缺乏造成的准确率影响”是
**0个百分点**。32题样本仍然较小，不能证明总体影响严格为零，但没有出现值得扩量的正向信号。

完整信息显著改变了效率而非能力：

- 模型动作从239降到160，减少33.1%；
- `execute_sql` 从192降到112，减少41.7%；
- schema `PRAGMA` 从75次降到0；
- 但由于完整 schema/value context 在每轮都重复进入 prompt，总 tokens 反而增加52.8%；
- 合法终止从32降到31；
- 中间错误从18变为19。

这说明完整 schema 和样例值能消除主动 schema 探索成本，但不能解决总体、粒度、公式和精确
输出字段等主要语义错误。把所有信息一次性放入上下文还可能让模型过早提交一个“可执行但输出
形态不精确”的查询。

该实验是诊断性接口 ablation，不准入 SFT。

## 实验控制

两轮保持不变的条件：

- 任务：
  `data/eval_inputs/bird_train_version38_semantic_discipline_gate32_20260729.jsonl`
- task SHA256：
  `0993665a2df7bc6447ee1b558119fa41a674251262b90c4130f5f22bc9a79cb8`
- 模型：`deepseek-v4-flash`
- SQL 动作：`execute_sql`、`submit_sql`
- 协议：`iterative-sql-v2`
- action carrier：`think-json-v1`
- 最大动作：30
- 同类错误上限：3
- SQL 执行超时：20秒
- 普通查询预览：20行
- 合法历史：recent 4
- API retries：10
- 判分：`bird-set`
- 每题一次语义尝试；错误答案不重试
- gold SQL 和 gold rows 对模型不可见

唯一实验变量是初始数据库上下文 profile。懒加载 system prompt 的 SHA256 重新计算为
`0289bc4929c1567a910d03608005ee3eb5309419dc97aaf0fe9f793baf679534`，
与原运行 manifest 完全一致。完整信息 profile 的 system prompt SHA256 为
`dffa7d2187082208a159535dd88670943938102d9ad7cebd32f95b37292c9a8c`。

完整信息使用 live SQLite schema 和值构造，不使用 BIRD column description、gold SQL、
gold rows 或其他隐藏标签，因此没有同时引入自然语言列描述这一额外变量。

## 完整上下文规模

每题的 schema/value context：

| 指标 | 数值 |
|---|---:|
| 字符数最小 | 1,763 |
| 字符数平均 | 12,536 |
| 字符数最大 | 65,563 |
| 平均列数 | 95.25 |
| 最大列数 | 457 |
| 平均示例值数 | 183.5 |
| 每列示例值上限 | 2 |

完整上下文通过紧凑 JSON 格式化。示例值只是 live distinct 样例，并明确标记为非穷举域。
模型仍可使用 `execute_sql` 查看分布、连接结果或候选答案。

## 正确集合变化

### 两轮都正确（8）

128、3042、3636、4163、4598、5554、5873、5952。

### 完整信息恢复（2）

#### 4189 · `app_store`

懒加载版本使用 left join，保留没有 translated review 的免费体育 App：

```sql
SELECT p.App, u.Translated_Review
FROM playstore p
LEFT JOIN user_reviews u ON p.App = u.App
WHERE p.Category = 'SPORTS' AND p.Type = 'Free'
```

完整信息版本改成 inner join，只返回有 translated review 的匹配：

```sql
SELECT p.App, u.Translated_Review
FROM playstore p
INNER JOIN user_reviews u ON p.App = u.App
WHERE p.Category = 'SPORTS' AND p.Type = 'Free'
ORDER BY p.App
```

完整关系结构可能帮助模型更快固定连接总体。

#### 5544 · `legislator`

懒加载版本用了28步，最终仍返回 representative 明细而不是女性人数。完整信息版本5步完成，
正确连接 `current` 与 `current-terms`，加入女性、Michigan、representative 三个条件，并返回
distinct representative count。

这是本轮最明确的信息可见性收益：完整列和样例值减少了长时间关系探索。

### 完整信息回归（2）

#### 2507 · `food_inspection_2`

懒加载版本经过6步探索后，最终只输出员工 first/last name。完整信息版本只用2步便直接提交，
但保留了辅助 `COUNT(*) AS cnt`，导致输出多一列。

#### 3969 · `talkingdata`

懒加载版本输出最大年龄对应的唯一 `gender` 列。完整信息版本只用2步便提交，但同时输出
`age, gender`，多出辅助年龄列。

两个回归都不是 schema 理解错误，而是完整信息使模型更快形成候选 SQL、减少了中间检查，
最终忽略 exact output slots。

### 两轮都错误（20）

大多数错误没有因完整 schema/value samples 改变，包括：

- 无依据选择单个疗程；
- 错误增加 order quantity、latest/current、distinct 等语义；
- row count 与 entity count 混淆；
- 使用错误历史关系；
- 最终携带 ID、count、排名键等辅助列；
- 提前 round 数值。

这些问题的所需表、列和值多数已经可见，继续增加环境数据不能直接告诉模型哪一种自然语言
解释符合任务的精确 denotation。

## 原目标与控制

冻结32题由16个 atomic version37 失败目标和16个 version37 正确控制组成。

| 分层 | 懒加载 SQL | 完整信息 SQL |
|---|---:|---:|
| 失败目标恢复 | 1/16 | 2/16 |
| 原正确控制保留 | 9/16 | 8/16 |
| 总计 | 10/32 | 10/32 |

完整信息将目标恢复增加1题，却同量损失1个控制，因此不能视为能力提升。

## 与 atomic version38

完整信息 SQL 为10/32，atomic version38 为18/32：

- SQL 相对 version38 新增成功：3042、5544；
- SQL 相对 version38 回归：600、2408、2507、2512、2513、2682、3011、3969、6165、6454；
- 净变化：-8；
- exact two-sided paired binomial `p=0.03857421875`；
- 两者并集：20/32。

完整信息使 SQL 与 atomic 的并集从懒加载时的19/32增加到20/32，说明存在少量互补题，但仍
不能证明完整 schema SQL 是更强的单一协议。

## 错误和接口行为

完整信息版本：

- 112次 `execute_sql`；
- 48次 `submit_sql`；
- 17个 `argument_validation_error`；
- 2个 SQLite `execution_error`；
- 14个 episode 使用了错误反馈；
- 5个正确答案经历过错误恢复；
- 13个错误答案全程没有过程错误。

唯一未合法终止的是1152。模型连续三次把尚未原样执行过的新 SQL 直接交给 `submit_sql`，
达到同类错误上限。这属于终止接口摩擦，不是 schema 缺失。即使把它视为潜在错误答案，
正确率也不会高于10/32。

完整 schema 仍未完全消除执行错误：模型在2438中引用了未定义 alias，在128中使用了上下文
里不存在的 join column。环境信息“已经可见”不等于模型会稳定读取和服从。

## Token 与注意力影响

动作减少但 token 增加的原因是：任务的完整 schema/value context 在每次模型调用中都会
重新出现。

| 指标 | 懒加载 | 完整信息 | 变化 |
|---|---:|---:|---:|
| prompt tokens | 494,767 | 786,552 | +59.0% |
| completion tokens | 43,537 | 36,171 | -16.9% |
| total tokens | 538,304 | 822,723 | +52.8% |
| 模型动作 | 239 | 160 | -33.1% |

完整信息正确题的平均 context 为9,671字符，错误题为13,839字符。这个关联受到数据库难度和
题目组成混淆，不能当作因果证据；但它没有支持“提供越多 schema/value 信息越好”的假设。

更可能的机制是：

1. schema 可见后，模型跳过探索并更早形成候选；
2. 对简单关系题有利；
3. 对 exact-output 题，减少中间反思会增加携带辅助列的风险；
4. 大数据库中几百列和几百个示例值会稀释问题相关信息；
5. 每轮重复完整上下文提高成本，却不提供新的任务语义反馈。

## 结论与下一步

本实验支持以下判断：

1. 当前懒加载环境确实造成了探索动作开销；
2. 它没有在这32题上造成可观测的净准确率损失；
3. 一次性提供所有表、列和示例值不是合适的最终方案；
4. 主要能力瓶颈仍是总体、粒度、关系语义和最终输出形态；
5. 更合理的环境设计是“相关 schema 优先 + harness 持久保存”，而不是每轮重发全库信息。

若继续实验，建议做相关性压缩 profile：

- 初始仍显示全部表名和行数；
- 给模型一次性提供问题相关候选表的完整 schema 和列样例；
- 其他表保留紧凑列名或按需展开；
- schema 结果由 harness 持久保存，不进入最近8条查询淘汰窗口；
- 终止前增加事实型输出形态摘要，如输出列名和列数，不给语义建议。

这种设计需要环境执行确定性的候选召回，不能让 router 查看 gold，也不能隐藏模型可能需要的
表。应先在同一32题上做小型 paired gate，而不是直接扩量。

## 验证与产物

权威结果：

`data/trajectories/bird_train_semantic_gate32_iterative_sql_full_schema_samples2_ds_v4_flash_network_retry1_20260729`

第一次受沙箱网络权限影响的无效部分运行：

`data/trajectories/bird_train_semantic_gate32_iterative_sql_full_schema_samples2_ds_v4_flash_20260729`

无效运行只有8个第一步 `api_error`，错误均为 `Operation not permitted`，没有形成语义结果，
未混入权威目录。

权威结果的31条合法 SQL 全部 fresh replay：

- replayed：31；
- correct：10；
- record/replay mismatch：0；
- execution failure：0；
- 非法终止跳过：1152。

独立结构检查覆盖32个 episode、160个模型 turn：

- strict carrier failures：0；
- model input forbidden gold markers：0；
- full-context header failures：0；
- terminal SQL 未预执行：0；
- structural gate：pass。

报告只依据问题、external knowledge、模型可见 schema/value context、模型生成 SQL、真实执行
反馈和隐藏 verifier 的布尔结果；未查看或暴露 gold SQL 或 gold answer rows。
