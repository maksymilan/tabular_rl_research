# checkpoint-relalg Semantic-v3-v24 差距修补

日期：2026-08-11
状态：**实现与离线审计完成；diagnostic-only；尚无行为提升结论或 SFT/RL 准入**

## 1. 为什么修补

同一 historical fixed-200 上，当前 checkpoint-relalg Atomic semantic-v2/v6 为 142/200，
历史 Atomic version24 为 145/200。两者逐题 13 gain、16 regression，差异不显著；但当前实现
的可靠性和成本明显较差：legal 188 对 197，tool/process errors 96 对 29，model turns 1635
对 1490，provider tokens 28,614,486 对 8,420,861（3.40 倍）。checkpoint 在 107 题上形成
130 次 commit，但该子集只得 70/107，历史 v24 同题为 71/107。因此优先问题不是扩大
checkpoint 次数，而是让模型更稳定、低成本地使用 Atomic 面。

该比较不是同时期单变量实验，不能把所有差异归因于下面任一项；它只用于定位工程差距。

## 2. 可观察差距

| 边界 | 当前 semantic-v2 | Atomic v24 的有效做法 | 后果 |
|---|---|---|---|
| schema delivery | Text-JSON 每轮重复递归展开的完整 JSON Schema，teacher prompt 39,701 字符 | 简洁 prose signatures、少量高熵工具规范调用 | 输入成本与形状选择负担过高 |
| provider history | 一个 phase 内累计完整历史，只有 commit 后清空 | rolling recent-4 causal history | 长轨迹 prompt 反复增长 |
| 输出约束 | 通用 exact-artifact 规则为主 | 明确约束 ID/code 与 label、独立输出槽、helper 列、top-k、重复度 | 当前仍容易形成表示或输出列错误 |
| 参数教学 | validator 精确，但常用复杂形状缺少集中示例 | 对 filter/join/aggregate/scalar/rank/output 给合法示例 | 当前参数错误更集中 |
| 执行安全 | strict canonical types、闭合 schema、typed AST、原子物化 | 较多 SQLite/旧接口兼容语义 | 当前更严格，但不能为追求旧分数退回隐式转换 |

当前 96 个工具错误中，25 个是 `invalid_text_json_content`、15 个 `unknown_column`、13 个
`unexpected_field`、10 个 `invalid_arguments`、9 个 `canonical_type_mismatch`。其中 31 个
属于 carrier/envelope 层；`read_rows` 和 `filter_rows` 又承担了大部分参数形状错误。相反，
version24 fixed-200 只有 29 个 process errors。该分布支持先修模型接口和历史成本，不支持放宽
Harness 类型、状态或 grounding 规则。

## 3. 已实现的隔离修补

新增 `atomic_operator_profile=semantic-v3-v24`，不修改冻结的 `semantic-v2`：

1. **执行面保持相同。** v3 与 v2 的工具定义和 tool-schema SHA256 完全相同：
   `dc6baa7f9df1cc578da825bd66043dd25146464b003947cc543cb20bcea932d8`。
   typed validator、SQLite executor、artifact、checkpoint、terminal answer 都不变。
2. **紧凑操作契约。** Text-JSON teacher prompt 从 39,701 字符降至 8,885 字符，减少
   77.6%。完整可执行 schema 仍用于 validation/hash/audit，但不再逐轮抄入 prompt。
3. **恢复 version24 的高价值约束。** compact contract 集中列出常用 typed predicate、工具
   精确签名和八个可由真实 validator 接受的规范调用，并明确 ID/code、输出槽、helper 列、
   duplicate 与列顺序边界。
4. **recent-4 provider history。** 同一 phase 只发送最近四个完整 Text-JSON 因果 turn；
   EnvironmentState、完整不可变轨迹和 fresh replay 仍保留全部事实。裁剪规则进入 capability 和
   protocol identity，审计器按同一规则重建请求历史。
5. **carrier 隔离。** v3 当前只允许 diagnostic A/Text-JSON；不静默改变 native carrier。

标准 checkpoint guidance 下的 v3 teacher prompt SHA256 为
`f7ca4a3d8f533062ce5d9f96adaaa22d61921b2445bb3d168a98a8a67c35fa05`；v2 与 v3 的 prompt、
history policy 和 protocol identity 不同，结果目录不得混用。

## 4. 刻意没有照搬 v24 的部分

- 不恢复 SQLite 的偶然类型转换或宽松输出命名；canonical type mismatch 仍结构化失败。
- 不把 gold SQL、reference schema/rows 或模型 reasoning 当作执行权威。
- 不删除 checkpoint，也不改变 checkpoint 快照、最多 8 次、distinct-goal 或所选 guidance。
- 不把 malformed visible JSON 自动修成一个行动；是否把纯 carrier 形状失败改为 bounded
  same-turn retry，需要单独的因果和计费实验。
- 不声称 compact prompt 或 recent-4 各自贡献了多少；当前 v3 是一个修补 package，后续若通过
  行为 gate，再做单变量拆分。

## 5. 离线验证与下一门

本次完整 checkpoint-relalg 回归为 `259 passed, 34 subtests passed`；新增端到端测试证明六轮
v3 Text-JSON 轨迹在第五轮后保持 recent-4 请求窗口，结构审计和 fresh replay 均通过。规范示例
逐条经过 semantic-v3 executable validator。

这些结果只证明实现正确，不能证明准确率提高。下一步应在同一冻结题目、同一 Flash 身份和
预算下 fresh paired 比较 semantic-v2 与 semantic-v3-v24，同时分别报告：correct/legal、strict
artifact、carrier/argument/type errors、turns、prompt/cache/completion tokens、checkpoint
coverage/commits。若 v3 不能显著降低 tokens 和 carrier/shape errors且不保正确率，应冻结而不
promote。
