# BIRD action-block v22 语义决策边界试验

日期：2026-07-26  
模型：DeepSeek v4 Flash  
指标：`bird-set`  
协议：`action-block-v22`，每块最多 5 个原子调用，rolling history=4

## 结论

v22 把一个 action block 明确定义为一个语义决策边界：同块允许已经完全落地的执行
依赖，但如果未见的 schema、值、行或错误反馈可能改变后续工具、参数、字面量、关系路径、
粒度或答案列，观察必须结束当前 block。

这条约束在失败条件化的 29 题诊断集上有强恢复信号，但在固定前 40 题上没有带来一般
准确率提升：

- 条件化 29 题：v21 为 10/29，v22 为 20/29；
- 固定前 40 题：atomic version24 为 33/40，v21 为 32/40，v22 为 31/40；
- v22 的固定 40 题全部合法，但有 9 个过程错误，高于 v21 的 5 个；
- 因此 v22 是合理的策略边界改进，不是准确率 promotion，不应据此启动 fixed-200。

## 修改边界

修改只作用于 action-block：

- `UNIFIED_ACTION_BLOCK_PROTOCOL_VERSION` 从 `action-block-v21` 增加到
  `action-block-v22`；
- prompt 删除允许宽泛混合观察和执行的表述；
- 明确区分 execution dependency 与 semantic decision dependency；
- 本地引用只运输用途已经确定的结果，不能授权跨未见反馈猜测；
- 未修改 atomic prompt、原子工具参数、执行器、DAG 调度、错误传播或终止 grounding。

## 失败条件化 29 题

该集合由 v21 相对 atomic 的 19 个退化目标和 v21 独有成功的 10 个保持对照组成。
它按既有结果选择，只能用于机制诊断，不能当作无偏准确率。

| 指标 | v21 | v22 |
|---|---:|---:|
| 正确 | 10/29 | **20/29** |
| 合法终止 | 28/29 | **29/29** |
| 19 个退化目标正确 | 0/19 | **13/19** |
| 10 个优势对照保留 | 10/10 | 7/10 |
| 模型轮次 | 177 | **138** |
| action blocks | 175 | **136** |
| 原子动作 | 306 | **278** |
| 观察调用 | 148 | **125** |
| 关系执行调用 | 129 | 127 |
| 过程错误 | 14 | **10** |
| 总 token | 902,171 | **752,737** |

恢复的 13 个目标是：

`00454, 00833, 01148, 01290, 01297, 01425, 01599, 01926, 02734, 02901,
03373, 04290, 05053`。

丢失的 3 个旧优势是：

`03262, 03636, 05440`。

其中 `01297` 是直接机制证据。v21 在观察 `genre` 值以前猜测 `genre_id=7`，并在同块
继续过滤、连接和聚合；v22 第一块并行取得四张表的 schema 和 `genre_name` 值域，看到
`Puzzle` 后，第二块才构造完整关系链，最终得到正确的 Nintendo。

## 固定前 40 题

| 指标 | atomic v24 | v21 | v22 |
|---|---:|---:|---:|
| 正确 | **33/40** | 32/40 | 31/40 |
| 合法终止 | **40/40** | 39/40 | **40/40** |
| 过程错误 | 2 | 5 | 9 |
| 模型轮次 | — | 176 | **135** |
| action blocks | — | 174 | **134** |
| 原子动作 | — | 293 | **279** |
| 观察调用 | — | 138 | **119** |
| 关系执行调用 | — | 117 | 117 |
| 总 token | — | 705,989 | **569,602** |
| reasoning token | — | **54,070** | 64,205 |

v22 相对 v21：

- 恢复：`02901`；
- 退化：`00552, 05440`。

v22 相对 atomic 没有新增成功，并退化 `00552, 06489`。这说明 29 题的强恢复不能外推
为总体能力提升；它同时包含失败条件化选择和 provider 非确定性。

## 轨迹观察

语义边界没有简单地把 action block 退化为 atomic。固定 40 题中，v22 的模型轮次下降
23.3%，总 token 下降 19.3%，关系执行调用数不变，观察调用下降 13.8%。模型仍会把
schema 已知、字面量已知、关系路径和输出都已确定的过滤、连接、投影和终止链放进同块。

剩余错误主要不是跨反馈 speculative execution：

- `06489` 两次 v22 运行都忽略外部知识要求的 `OBJ_SAMPLE_ID`，连接
  `OBJ_CLASSES` 后回答 `"paper"`；这是输出实体映射错误。
- `05440` 两次 v22 运行都先在原始 `Paper` 上取异常最大年份 `800190`，再连接
  `Journal`；这是候选总体选择错误。
- `04906` 仍统计 distinct images，而外部知识要求 object-sample occurrences；
  这是计数粒度错误。
- `05760` 仍采用 `FBI_Code.title` 路径，而 gold 使用 IUCR primary description；
  这是语义关系路径错误。
- `02918` 得到空关系后直接引用空 ID 列，未构造标量 0；这是终止输出形态错误。

v22 的 9 个固定 40 题过程错误中，有两个来自把 observation-only 的
`read_subtable` 局部 id 当作终止 evidence table；其余包括列名猜错、一次 limit
越界、一次 carrier 结构错误和若干 namespace/表达式错误。语义边界规则本身不会消除
这些接口错误。

## 决策

保留 v22 作为 action-block 的实验策略边界，因为它：

- 与“环境负责调度、模型负责理解反馈”的设计一致；
- 在明确的跨反馈猜测案例上有效；
- 保持最多 5 调用和确定性流水线的效率；
- 不改变 atomic 工具方案。

但固定 40 题未给出扩展信号，因此：

- 不启动 v22 fixed-200；
- 不把 v22 轨迹用作 SFT 来源；
- 若继续优化，下一次应独立处理 observation-only 句柄误用和终止前输出形态检查，
  并先通过新的小规模 gate；不要增加题目补丁式 prompt。

## 产物

- 条件化 29 题有效运行：
  `data/trajectories/batch_plan_20260726/action_block_v22_semantic_boundary_gate29_r2.all.jsonl`
- 条件化 29 题首次本地沙箱外部失败：
  `data/trajectories/batch_plan_20260726/action_block_v22_semantic_boundary_gate29_r1.all.jsonl`
- 固定前 40 题：
  `data/trajectories/batch_plan_20260726/action_block_v22_semantic_boundary_fixed40_r1.all.jsonl`
- 本地 action-block 协议/执行测试：42/42；
- action-block SFT/工具方案测试：8/8；
- eval 测试：28/28。
