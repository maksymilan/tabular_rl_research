# BIRD relational-program v4 类型化引用 20 题诊断

日期：2026-07-27

模型：DeepSeek v4 Flash，thinking enabled，reasoning effort high，temperature 0

指标：`bird-set`

固定样本：`bird_train_tool_interface_validation200_version4.jsonl` 的前 20 题

数据准入：`diagnostic_only_pending_protocol_scale_gate`

运行前 fresh preflight 确认仓库相对 `origin/master` ahead 40、behind 0。工作区已有的其他
改动被原样保留。底层执行仍是共享 atomic harness；本实验只改变
`relational-program` 的公开引用语法，不改变原子操作、执行顺序验证、provenance 或
未来 process credit 粒度。

## 结论

`relational-program-v4` 得到 **10/20 = 50%**，合法结束 **19/20 = 95%**。相对 v3：

- process error 从 16 降到 8；
- 模型轮次从 131 降到 119；
- 原子动作从 169 降到 149；
- 总 token 从 561,665 降到 508,471；
- 但正确数从 12 降到 10。

因此类型化引用对接口稳定性和成本有正向作用，但没有通过终局正确率门。它不能替代
语义规划训练，也不能据此进入固定 50/200、SFT 或 RL。当前证据支持保留这条工程
边界作为后续小门控基础，不支持声称工具方向已经提升了能力。

## 唯一实验变量

v4 保持三个排他顶层工具：

- `observe`
- `relational_program`
- `answer_from_context`

公开 prompt 中仍不包含 atomic 工具定义、`action_block` 或旧 `$id` 语法。v4 引入
`typed-relational-reference-v1`：

```json
{"source_table":"orders"}
{"resident_table":"filter_002"}
{"resident_step":"step_7"}
{"node":"filtered"}
{"node":"filtered","column":"customer_id"}
```

只有 `node` 引用参与本轮 DAG 推导。源表、前轮 resident table、前轮 producing step
不会因为名称碰撞而成为本轮依赖。harness 将类型化公开语法确定性地降为私有执行
载体，并在 `work_graph.authored_references` 中记录原始引用类型和位置；这不是参数
修复或语义猜测。

`observe.rows` 仍严格不接受过滤条件。v4 对错误的 `condition(s)` / `where` 参数返回
明确的事实性 schema 反馈，而不代替模型执行过滤。

## 固定 20 题结果

| 方案 | 正确 | 合法结束 | 模型轮次 | 原子动作 | 过程错误 | blocked | 总 token | 耗时 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| relational-program v3 | 12/20 | 19/20 | 131 | 169 | 16 | 1 | 561,665 | 335.4s |
| **relational-program v4** | **10/20** | **19/20** | **119** | **149** | **8** | **3** | **508,471** | **264.4s** |
| atomic version24 | 16/20 | 20/20 | 127 | 127 | 1 | 不适用 | 690,165 | — |
| action-block v18 | 17/20 | 19/20 | 100 | 158 | 5 | 0 | 448,405 | — |

v4 相对 v3：

- 模型轮次 -9.2%；
- 原子动作 -11.8%；
- submitted calls -10.6%；
- process error -50.0%；
- token -9.5%；
- 墙钟时间 -21.2%；
- 正确率 -10 个百分点。

所有 119 个 API 请求均无 transport、context、carrier 或 completion retry。

## 配对翻转

v4 相对 v3：

- v4 独有正确：`00541`、`05149`；
- v3 独有正确：`02935`、`03390`、`04189`、`06171`；
- 共同正确：8；
- 净变化：-2；
- 六个 discordant pair 的精确双侧检验 `p=0.6875`。

样本很小且 DeepSeek v4 Flash 在相同题、`temperature=0` 的独立请求间仍出现单题
翻转：3 题 smoke 为 3/3，而完整运行中其中一题翻转为错误。因此不能把每个单题变化
全部归因于引用语法；可稳定解释的是聚合后的结构错误、动作数和 token 变化。

## 剩余过程错误

8 个过程错误分为：

- scalar 命名 cell 的类型化引用形状错误：3；
- join 的当前逻辑列/历史 resident namespace 错误：3；
- 运行时投影列名错误：1；
- `observe.rows` 错带 `filter` 键：1，随后恢复并答对。

v3 的三次跨程序局部 id 误用和三次 `observe.rows.limit > 20` 在 v4 中均未出现。说明
类型区分确实消除了原目标错误，但 scalar 引用仍保留了多余的 `value_ref` 嵌套，
模型产生了两种自然但当前不合法的等价形状。另一个公开反馈缺陷是底层错误消息仍会
出现私有 `$node` 和 atomic 操作名；后续版本必须在模型可见错误中反向映射为公开
类型化引用和公开 operation 名。

## 零过程错误的语义失败

7/20 在没有任何 process error 时仍终端错误。只审计公开问题和合法轨迹，主要模式是：

- 最终列集合不精确：保留了排序辅助列或问题未要求的标识列；
- 标识符与人类可读标签选择错误；
- 只查询一个区域/分表，未覆盖问题要求的完整 population；
- 把坐标匹配理解为点等值，而非对象区域包含关系；
- “最年轻获奖者”等时间语义被简化为出生日期排序；
- 已计算最大值但没有通过 grounded scalar reference 供后续过滤，随后使用了错误字面量。

这些问题中只有最后一项直接指向 scalar 引用接口；其余是 population、输出语义和
问题理解，不应通过 harness 猜测或自动改写来掩盖。

## 下一小门控

不扩大 v4。下一版只验证一个更一致的 scalar 引用表面：

1. scalar operand 直接接受类型化 `node` / `resident_step` 引用及可选 `column`，不再
   额外嵌套 `value_ref`；
2. predicate 的计算值使用一个明确的 typed value-source 字段并确定性降为底层
   `value_ref`；
3. 模型可见错误必须反向映射私有 `$node` 和 atomic 名称；
4. 先在 scalar 失败题和零错误控制题组成的冻结小集合验证；未恢复目标或回退控制时
   立即停止，不进入 20/50/200。

## 产物

- 3 题 smoke：
  `data/results/relational_program_v4_typed_refs_smoke3_deepseek_v4_flash_20260727/all.jsonl`
- 20 题轨迹：
  `data/results/relational_program_v4_typed_refs_pilot20_deepseek_v4_flash_20260727/all.jsonl`
- 20 题 manifest：
  `data/results/relational_program_v4_typed_refs_pilot20_deepseek_v4_flash_20260727/all.manifest.json`

所有产物均为 diagnostic-only，未写入任何训练 mixture。
