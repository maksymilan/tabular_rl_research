# BIRD action-block v32 固定 200 题评估

日期：2026-07-26  
模型：DeepSeek v4 Flash，thinking enabled，reasoning effort high，temperature 0  
指标：`bird-set`  
数据：`bird_train_tool_interface_validation200_version4.jsonl`

## 结论

`action-block-v32` 的固定 200 题结果为 **144/200（72.0%）**，合法终止
**197/200**。它没有达到预先规定的 **150/200** SFT 构造门槛，也没有超过同 cohort
的 atomic version24（145/200）。因此该版本不得用于构造新的 SFT 数据。

v32 的工程收益明确，但不是能力提升：

- 所有 197 条合法轨迹都以独立顶层 `answer_from_context` 终止；
- 没有嵌套终止、混合终止或终止 carrier 错误；
- 每个成功的中间原子操作均返回完整 `output`，缺失数为 0；
- `join_tables.joins[].on` 被误写为单个对象的错误为 0；
- 相比 atomic version24，模型轮次从 1490 降到 1115（-25.2%），总 token 从
  8,420,861 降到 5,031,851（-40.2%）。

代价是原子工具工作量和恢复负担上升：v32 实际尝试 1756 个原子操作，另有 51 个
依赖节点被阻塞；atomic version24 只有 1490 个模型/原子步骤。v32 共有 112 个过程错误，
atomic version24 为 29 个。也就是说，action block 成功减少了模型往返和 token，但没有
减少模型为解决问题所需的原子操作，也没有提高最终正确率。

## 协议

v32 只修改 `action-block` 方案，不改变原始 atomic 环境：

- work turn：一个顶层 `action_block`，包含 1..5 个非终止原子调用；
- terminal turn：一个独立顶层 `answer_from_context`；
- `answer_from_context` 不得出现在 `action_block.calls`；
- calls 按列表顺序执行并按同一顺序返回完整结果；
- 失败调用记为 root error；依赖失败结果的后续调用记为 blocked，独立调用继续执行；
- `$id` 只引用同块中更早的调用；跨块使用 harness 返回的 resident handle 或 step id；
- harness 不进行拼写、schema、column、predicate、handle 或参数形态修复；
- terminal reason 不能代替行筛选、列投影、字段拆分或 ID/label 选择。

模型可见 carrier 仍为一个非空 `<think>` 后直接跟一个严格 raw JSON action；DeepSeek
provider adapter 使用 native reasoning 加可见 raw JSON。

## 40 题优化路径

| 版本 | 正确 | 合法 | 过程错误 | blocked | 模型轮次 | 原子尝试 | token |
|---|---:|---:|---:|---:|---:|---:|---:|
| v28 | 31/40 | 39/40 | 60 | 25 | 219 | 259 | 876,963 |
| v29 | 29/40 | 39/40 | 33 | 22 | 185 | 263 | 744,081 |
| v31 | 28/40 | 40/40 | 28 | 20 | 208 | 294 | 852,959 |
| v32 | 31/40 | 40/40 | 10 | 2 | 182 | 278 | 792,488 |

v29 用 prompt 区分 work/answer block 并补充 join namespace，显著减少终止摩擦，但产生
语义退化。v31 把 terminal 改成独立顶层 action，并补齐 DeepSeek terminal carrier
示例，完全消除了混合终止和空 visible response；但 join 参数形态和“可推断结果表”
仍导致退化。v32 增加两个通用不变量：

1. `on` 必须是 `[{left,right}]` 列表；
2. “可从表中推断答案”不等于“该表就是精确答案”，winner 行和答案字段必须实际派生。

v32 在固定 40 题恢复到 31/40，并显著降低过程摩擦，因而获得固定 200 扩测资格。

## 固定 200 结果

| 指标 | atomic version24 | action-block v32 |
|---|---:|---:|
| 正确 | 145/200 | 144/200 |
| 合法终止 | 197/200 | 197/200 |
| 过程错误 | 29 | 112 |
| 模型轮次 | 1490 | 1115 |
| 尝试的原子操作 | 1490 | 1756 |
| blocked 节点 | 不适用 | 51 |
| 总 token | 8,420,861 | 5,031,851 |

逐题配对：

- v32 gain：13 题；
- v32 regression：14 题；
- discordant：27 题；
- 双侧精确二项检验：`p = 1.0`。

因此 144 与 145 没有显著差异。v32 没有获得 atomic 能力的超集；两者仍存在真实的
解题分歧。

v32 的 144 条正确轨迹中：

- 98 条无过程错误；
- 46 条在收到错误反馈后恢复并正确终止。

## 三条未合法终止轨迹

1. `bird_train_04244`：3 次 argument validation error 后耗尽同类错误预算。模型跨块
   继续写 `$group_002` / `$filter_001`，并有一次 `join_tables` 缺少 `base`。
2. `bird_train_06299`：4 次 execution error。模型没有复制 `base_role` 引入的实际
   namespace，连续使用 `MenuPage.menu_id`、`filter_001.menu_id`、`filter_002.menu_id`，
   而反馈中的可用 namespace 分别是 `mp`、`f1`、`f2`。
3. `bird_train_03680`：先跨块误用 `$film_dream`，随后在多跳 join 中连续发明
   `inv_001`、`rent2_001`、`cust3_001` namespace，最终耗尽 execution error 预算。

这三题不是终止协议失败，而是长链 join 中没有服从最新 factual column namespace。
简单提高错误预算可能增加合法率，但不会自然提高语义正确率，且会改变固定评估预算，
不应在本次结果后直接补跑并混入固定 200。

## 剩余摩擦

112 个过程错误的主要重复模式包括：

- 7 次跨块 `$filter_001`；
- 6 次把 `DESC` 当成独立列；
- 5 次 `join_tables` 缺少 `base`；
- 4 次空 `calls`；
- 3 次非法 `read_subtable.limit`；
- 其余是多样化 column/value/scalar/join namespace 错误。

这些错误的长尾已经不适合继续堆叠 student prompt。项目已有正式 JSON grammar causal
pilot 的负面结果；继续增加参数语法文本可能重新造成 plan/格式循环。后续若继续优化，
应优先研究训练数据中的因果恢复样例或重新设计高熵工具参数，而不是为失败题添加
针对性 prompt。

## 决策

- 保留 v32 作为当前 action-block 工程候选：它满足独立终止、完整反馈、低模型往返和
  低 token 的设计目标。
- 不宣称能力提升；在固定 200 上它与 atomic 基本持平且低 1 题。
- 不构造 v32 SFT 数据；固定 200 未达到 150/200。
- 不继续在当前 prompt 上追加失败题补丁。

## 工件

- 前 40：
  `data/trajectories/batch_plan_20260726/action_block_v32_exact_terminal_join_shape_fixed40_r1.all.jsonl`
- 后 160：
  `data/trajectories/batch_plan_20260726/action_block_v32_exact_terminal_join_shape_remaining160_r1.all.jsonl`
- 两片的协议版本、协议 hash、system prompt hash、模型、carrier、解码、history、预算和
  `bird-set` 配置完全一致；合并后的 trajectory id 恰好覆盖冻结 200 cohort，互不重复。
