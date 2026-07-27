# BIRD relational-program v3 排他工具协议 20 题诊断

日期：2026-07-27
模型：DeepSeek v4 Flash，thinking enabled，reasoning effort high，temperature 0
指标：`bird-set`
固定样本：`bird_train_tool_interface_validation200_version4.jsonl` 的前 20 题
数据准入：`diagnostic_only_pending_protocol_scale_gate`

运行前 fresh preflight 记录的底层 atomic 执行契约为 `version28`、action carrier 为
`think-json-v1`、tool-scheme registry 为 `tool-scheme-registry-v3`；仓库相对
`origin/master` ahead 39、behind 0。工作区存在并保留了其他未提交变更。

## 结论

`relational-program-v3` 得到 **12/20 = 60%**，合法结束 **19/20 = 95%**。相对最初
`relational-program-v1` 的 10/20 和 15/20，排他工具协议有明确的接口收益：

- 过程错误从 43 降到 16；
- blocked 节点从 36 降到 1；
- 提交节点从 215 降到 170；
- 3 个 v1 失败被恢复，1 个 v1 正确题回退，净增 2 题。

但 v3 仍低于同一固定题组上的 atomic version24（16/20）和 action-block v18
（17/20），而且 v3 的 12 个正确题全部也是 atomic 和 v18 的正确题，没有发现独有
能力。因此当前结论是：

1. 旧 prompt 同时展开新程序和旧原子工具定义，确实造成了可避免的接口混淆；
2. v3 已验证“模型声明参数依赖、harness 自动形成 DAG、节点独立执行/归因”在工程上
   可行；
3. 排他命名只解决了接口层问题，尚未解决复杂题的语义规划、观察边界和收敛问题；
4. v3 不通过扩展或训练准入门，不得用于构造 SFT 或启用 RL。

## v3 协议

模型可见 prompt 只有三个顶层工具：

- `observe`：`schema`、`column`、`rows` 三个观察变体；
- `relational_program`：只允许 `filter`、`select`、`scalar`、`join`、`aggregate`、
  `rank`、`combine` 七种节点操作；
- `answer_from_context`：引用一个已驻留的精确结果表。

prompt 不包含 atomic 工具名、action-block 工具名或它们的独立定义。执行器内部才把
节点操作映射到现有原子执行函数。每个节点继续单独执行、单独返回错误、单独记录
provenance，并保持未来 process reward 的原子粒度。

模型只给出 `$id` / `$id.column` 参数引用。harness 校验未定义引用、环、孤立节点和
result/exports 根，并计算稳定拓扑序。20 题中形成了 32 个关系程序，平均宽度 2.56：

| 程序宽度 | 数量 |
|---:|---:|
| 1 | 6 |
| 2 | 8 |
| 3 | 14 |
| 4 | 3 |
| 6 | 1 |

其中 1 个程序的模型书写顺序不是拓扑序；harness 正确重排后执行成功。这直接验证了
“模型声明参数依赖、环境自动建 DAG”的目标，而不需要模型显式维护 edge 或 schedule。

## 固定 20 题结果

| 方案 | 正确 | 合法结束 | 模型轮次 | 原子动作 | 过程错误 | blocked | 总 token |
|---|---:|---:|---:|---:|---:|---:|---:|
| relational-program v1 | 10/20 | 15/20 | 122 | 179 | 43 | 36 | 505,003 |
| **relational-program v3** | **12/20** | **19/20** | **131** | **169** | **16** | **1** | **561,665** |
| atomic version24 | 16/20 | 20/20 | 127 | 127 | 1 | 不适用 | 690,165 |
| action-block v18 | 17/20 | 19/20 | 100 | 158 | 5 | 0 | 448,405 |

v3 相对 v1：

- 正确率 +10 个百分点；
- 合法率 +20 个百分点；
- 过程错误 -62.8%；
- blocked -97.2%；
- 原子动作 -5.6%；
- token +11.2%，模型轮次 +7.4%。

与 atomic 相比，v3 少用 18.6% token，但模型轮次没有减少（131 对 127），说明多轮
`observe` 抵消了程序合并带来的往返节省。与 action-block v18 相比，v3 同时更慢、
token 更多且准确率更低。

## 配对结果

### v3 对 v1

- v3 独有正确：`03390`、`05952`、`04189`；
- v1 独有正确：`00541`；
- 共同正确：9；
- 净变化：+2；
- 4 个 discordant pair 的精确双侧检验 `p=0.625`，样本不足以证明稳定提升。

### v3 对 atomic version24

- v3 独有正确：0；
- atomic 独有正确：`06489`、`06336`、`05149`、`00541`；
- 共同正确：12；
- 净变化：-4。

### v3 对 action-block v18

- v3 独有正确：0；
- v18 独有正确：`06489`、`06336`、`05149`、`00541`、`05440`；
- 共同正确：12；
- 净变化：-5。

这些 task id 只用于配对审计；报告不包含 gold SQL、gold 行或隐藏答案。

## 接口改善与剩余错误

v1 的主要错误是旧原子工具形状被错误地搬进程序节点，包括：

- 7 次 join `on` 不是列表；
- 6 次 join `left` 不是当前可用的精确逻辑列；
- 3 次 join `right` 不是新表的 bare column；
- 多次把旧工具当成程序顶层调用、跨程序复用局部 id，或提交孤立分支。

v3 不再出现上述主导性的旧工具/新程序载体混用。剩余 16 个过程错误集中为：

- `observe.rows` 错带 `condition(s)`：5；
- `observe.rows.limit` 超出 1..20：3；
- 引用不存在的前轮局部 id：3；
- 运行时列名错误：2；
- rank 参数形状、scalar grounding、局部列引用形状：各 1；
- 另有 1 个被依赖错误阻塞的节点。

其中 7/20 是“零过程错误但终端语义错误”。这说明进一步改错误文案最多只能修复一部分
接口失败，不能解释当前与 atomic 的全部准确率差距。

## 为什么仍然落后

1. **观察与变换的边界仍不稳定。** 模型会把过滤条件塞给 `observe.rows`，说明统一
   `observe` 虽然消除了旧工具名竞争，却还没有形成稳定的“观察事实 vs 派生关系”
   心智模型。
2. **跨程序局部引用仍有记忆负担。** `$id` 只在一个程序内有效；后续程序必须复制
   resident handle。模型仍有 3 次把前轮 handle 当作局部 id。
3. **程序化并未自动提高语义规划。** 7 条失败在没有过程错误时仍给出错误答案；DAG
   能保证执行顺序正确，但不能保证模型选对 population、grain、列和最终结果形状。
4. **复杂题上容易过度探索。** 最长失败用了 22 个模型轮次、33 个原子动作和 2 次
   completion retry，抵消了批量执行的速度优势。
5. **当前节点参数仍继承原子工具的高熵结构。** 新名称消除了工具选择冲突，但 join、
   aggregate、predicate 和 scalar 的嵌套参数复杂度没有下降。

## 下一步建议

不应继续给 prompt 堆更多旧工具说明。下一轮只做独立小门控：

1. 让 `observe.rows` 的 schema 成为真正的只读切片，错误反馈明确写出“不能过滤；
   过滤只能用程序的 filter 节点”，并保留严格拒绝；
2. 在 model-visible resident state 中给每个 handle 增加一个简短、稳定的
   `use_as` 字段，区分“跨轮 exact handle”和“本程序 `$id`”，但不做隐式引用修复；
3. 针对零过程错误的 7 条失败做只看合法轨迹前缀的语义分类，决定是否需要新的原子
   操作，而不是再做 carrier 修补；
4. 只有独立小门控同时提高语义正确率且不增加回退，才进入固定 50/200。

## 产物

- v1 轨迹：
  `data/results/relational_program_v1_deepseek_v4_flash_pilot20_20260727/all.jsonl`
- v1 manifest：
  `data/results/relational_program_v1_deepseek_v4_flash_pilot20_20260727/all.manifest.json`
- 中止的 v2 审计片段（9 条，不得作为完整配对结果）：
  `data/results/relational_program_v2_deepseek_v4_flash_pilot20_20260727/all.jsonl`
- v3 smoke：
  `data/results/relational_program_v3_deepseek_v4_flash_smoke1_20260727/all.jsonl`
- v3 完整轨迹：
  `data/results/relational_program_v3_deepseek_v4_flash_pilot20_20260727/all.jsonl`
- v3 manifest：
  `data/results/relational_program_v3_deepseek_v4_flash_pilot20_20260727/all.manifest.json`

所有 v1/v2/v3 结果均为 diagnostic-only，未写入任何训练 mixture。
