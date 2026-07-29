# BIRD action-block v34 单边 join 冻结 20 题诊断

日期：2026-07-27
教师：DeepSeek v4 Flash，thinking enabled，reasoning effort high，temperature 0
指标：`bird-set`
数据：`bird_train_tool_interface_validation200_version4.jsonl` 的冻结 20 题
身份：`action-block-v34`，protocol hash `670e5ef1bcd60fb3`

## 结论

v34 证明了接口复杂度确实是一个独立控制变量，但没有证明 action block 已经优于 atomic。

- v34：**14/20**，合法终止 **20/20**；
- 相比同题 v33：3 个 paired gain、0 个 regression，11/20 提升到 14/20；
- process errors：18 降到 7；
- blocked descendants：20 降到 2；
- 模型可见 `join` 调用错误：从 v33 的 15 次降到 0；
- 但仍低于 atomic version24 的 16/20，也低于历史 action-block v18 的 17/20。

因此保留“短连续原子操作”的研究方向，但不能把 v34 作为 SFT 来源或准确率晋升版本。
下一步不应继续给 `join` 堆提示文字；剩余差距主要是题意、关系粒度、异常数据判断和精确
终答形状。

## 接口

模型每个 work turn 只提交 1..8 个按列表顺序执行的连续操作：

- 不声明程序、DAG、dependency、result root、export、plan、status 或 handle；
- `$id` 只引用同一块中更早的结果；
- 如果必须看新 schema、值、行或错误才能决定下一步，就结束当前块等待反馈；
- 环境为每个已提交操作按原顺序返回完整 output、精确 error 或 blocked cause；
- 终止仍是下一回合独立顶层 `answer_from_context`。

v34 唯一新的公开工具语义是：

```text
join(left, right, left_on, right_on, how?)
```

它表示一条 join 边。`left_on` 是左输入的一个精确列，`right_on` 是右表的一个裸列，
`how` 默认为 `inner`，也可为 `left`；cross join 省略两个 key。多跳 join 由连续多个
单边 join 组成。harness 将其确定性 lowering 到冻结的 `join_tables` 执行器，不猜测
列、schema、predicate 或意图；私有执行器名称不会出现在 v34 的模型提示和反馈中。

## 分阶段冻结门控

先固定 9 个 v33 join-error 目标和 5 个零错误正确控制，共 14 题。预先门槛为：

- 14/14 合法终止；
- 至少保留 4/5 控制；
- 至少 10/14 正确；
- 目标上的 join 相关错误不超过 7 次。

结果是 11/14、14/14 合法、5/5 控制保留；总过程错误从 v33 的 17 次降到 4 次，
blocked 从 20 降到 1，join 相关错误从 15 降到 0。门控通过后，用同一输出、同一
prompt/hash 和 `--resume` 只补跑剩余 6 题，没有重算前 14 题。

## 冻结 20 题对比

| 方案 | 正确 | 合法 | 过程错误 | blocked | 模型轮次 | 尝试原子操作 | token |
|---|---:|---:|---:|---:|---:|---:|---:|
| atomic version24 | 16/20 | 20/20 | 1 | 不适用 | 127 | 127 | 690,165 |
| action-block v18 | 17/20 | 19/20 | 5 | — | 100 | 158 | 448,405 |
| action-block v32 | 15/20 | 20/20 | 7 | 0 | 98 | 160 | 435,396 |
| action-block v33 | 11/20 | 20/20 | 18 | 20 | 107 | 159 | 405,455 |
| **action-block v34** | **14/20** | **20/20** | **7** | **2** | **113** | **165** | **479,019** |
| relational-program v4 | 10/20 | 19/20 | 8 | 3 | 119 | 149 | 508,471 |

这只是 20 题诊断，不做显著性声明。V34 相比 v33 的收益是真实的配对接口信号，但其
最终正确率仍没有超过更简单的控制。

## v34 的七次过程错误

- 2 次 provider-visible 顶层 JSON/call 结构错误；
- 2 次 `scalar_compute` 把多列一行结果误当成可直接抽取的 scalar；
- 2 次 join 后继续使用歧义裸列；
- 1 次跨块继续使用 `$id`。

没有 `join` 参数形状、左右边、key list 或 private namespace 错误。所有 API transport
retry 均为 0；completion retry 为 2，carrier retry 为 3，它们是 provider 客户端事件，
没有计为模型语义失败。

## 六条错误终答

不查看 gold SQL，仅按问题、合法模型可见轨迹和 verifier 的正确/错误位审计：

1. `bird_train_06489`：精确筛中对象后同时输出 sample id 和类别，属于终答字段范围错误。
2. `bird_train_01152`：把“最年轻获奖者”简化为出生日期最大，没有计算获奖时年龄。
3. `bird_train_05440`：面对异常年份 `800190` 在“最大值”和人为回退到 2010 之间摇摆，
   最终仍按异常最大年份作答。
4. `bird_train_03390`：筛出英语占比 100 的 country code，但没有继续映射到国家名称。
5. `bird_train_06026`：同一地区存在重复 profit 行时直接 `union_all + sum`，没有确认题目
   要求的粒度，重复计数。
6. `bird_train_00593`：找到两个急性支气管炎用药区间后任意选择第一个；过程中还暴露
   了多列一行结果到 scalar operand 的接口摩擦。

其中 `06489`、`05440`、`03390` 是零过程错误的错误终答；`01152` 的列错误也已在轨迹
内恢复，最终错误仍是年龄语义。继续完善 join 反馈无法修复这些问题。最后一类提示下一项
可控接口消融可以是把一行结果的指定列直接作为 scalar operand，但必须作为正式公开
语法并确定性 lowering，不能作为静默修复。

## 产物与准入

- 结果：
  `data/results/action_block_v34_one_edge_join_gate14_deepseek_v4_flash_20260727/all.jsonl`
- manifest：
  `data/results/action_block_v34_one_edge_join_gate14_deepseek_v4_flash_20260727/all.manifest.json`
- prompt SHA256：
  `73468aac4e90d2d3459d1707993db8d5ba6bb2700924734217d39c3d76d9a5c6`
- `gold_sql_visible_to_model=false`
- `sft_export_eligible=false`

目录名保留最初 gate14 审计身份；manifest 的最终 `source_count=20`、requested ids 和
summary 记录了通过门控后的增量扩展。该产物是 diagnostic-only，不得进入 SFT 或 RL。
