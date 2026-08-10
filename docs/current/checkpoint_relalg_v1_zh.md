# checkpoint-relalg-v1 前向工具协议

状态：**前向开发主线，diagnostic-only，尚未获得 SFT/RL 准入**。

协议 id 为 `checkpoint-relalg-v1`，scheme id 为 `checkpoint-relalg`。每次运行必须显式选择
`mode=direct|atomic|hybrid`，并把 mode、协议哈希、prompt 哈希、工具 schema 哈希、provider
身份和运行预算写入 manifest。mode 不是模型可以修改的参数，也不能根据轨迹形状事后推断。

“前向主线”只表示：从本版本起，所有新的工具、prompt、状态管理和实验设计都以这个独立
模块为起点。它不表示已经优于历史协议，也不表示可以训练。现有 atomic 继续服务于正在进行的
RL、冻结对照和精确复现；version54 / `native-tool-bundle` 只保留为诊断对照与复现线，不因此
获得 RL 准入。两者都不得删除、静默迁移、改名为本 scheme，或混入同一数据集、checkpoint、
结果目录与统计口径。

## 1. 设计边界

完整设计规格约 4 万字符，是给 Harness、实现 agent、测试与审计器使用的机器/工程规范，
不是逐轮发送给模型的 system prompt。模型可见内容严格分为四层：

1. 短 shared core：数据权威、单调用、工件与终止等跨 mode 不变量；
2. 短 mode prompt：只解释当前 mode 的决策边界；
3. 当前 mode 的 compact native function schemas；
4. Harness 每轮重建的动态上下文。

外部教师看到同一 shared core 和 mode contract，只额外增加短小的 checkpoint 使用指导；
不会看到另一套更宽的工具、隐藏 SQL、答案值或未来轨迹。教师指导可以影响行动策略，不能
新增参数、状态字段、数据库事实或执行语义。

## 2. 一轮一个 official DeepSeek native call

所有新教师、评估与审计请求只使用官方 `https://api.deepseek.com/chat/completions`，不得使用
AimixHub/AIHubMix、代理或静默 fallback。当前不是 FIM 路径。

每个 assistant turn 必须包含：

- 原始 `reasoning_content`；
- 恰好一个 native `tool_calls` 项；
- 下一次请求前与该 call id 精确配对的一个 `role=tool` 结果。

assistant `content` 若非空，只保留用于审计，不能参与执行、状态、证据或评分。模型一次写出
多个 calls 是状态不变的语义协议错误，不能像 version51-version54 那样当作 bundle 执行。
Harness 必须为 provider 已返回的 call id 构造合法、可审计的错误结果，再让同一 episode
恢复。未知工具、非法参数和执行前引用错误同样返回结构化错误，失败调用不能产生可引用
artifact。

网络失败、零调用、长度截断等 provider 形状问题使用有界 client retry；这些重试不是模型
动作，不消耗语义 step。上下文溢出必须显式终止，不能删除事实或静默切换 prompt/provider。

## 3. 三种 mode 与公共控制面

| mode | 公共数据操作面 | 适用边界 |
|---|---|---|
| `direct` | `describe_table`、`inspect_column`、`read_rows`、只读 `execute_sql` | 让模型直接构造并检查一条只读 SQL；成功关系结果仍进入统一 artifact/state 系统 |
| `atomic` | 感知工具，加 `filter_rows`、`project`、`join`、`aggregate`、`distinct`、`set_operation`、`sort`、`limit`、`add_rank` | 只通过 typed relational algebra 逐步构造关系 |
| `hybrid` | direct 与 atomic 的并集 | 可在同一统一状态中选择 SQL 或 typed operators，但每轮仍只能调用一个工具 |

Atomic 另有显式、隔离的算子粒度配置 `--atomic-operator-profile`：

- `micro-v1` 是冻结默认面，保留上表九个机械原子算子、历史 prompt/schema/hash 与精确重放；
- `semantic-v2` 是 2026-08-10 开始的 diagnostic 优化面，不改变 `mode=atomic`，把一个任务级
  决策所需的确定性机械步骤合并为 `shape_rows`、`group_aggregate`、`scalar_compute` 与
  `rank_select`。其中条件聚合可在同一固定 population/grain 上表达扁平 metric-local `where`，
  `rank_select` 显式绑定排序、top-k、tie policy 与最终投影；population 仍由可检查、可复用的
  `filter_rows` artifact 决定。Harness 仍在一个
  savepoint 中 typed-lower、物化一个最终 artifact，并保存完整 derivation，模型不会获得任意
  SQL 或通用 action block。

`semantic-v2` **保留** `commit_checkpoint` 与 `restore_checkpoint`。优化的是一个 phase 内的
机械决策长度，不是删除 phase control：稳定 population、grain、join、aggregation 或 ranking
里程碑仍可 commit，后续矛盾或错误分支仍可精确 restore。该配置有独立 profile、prompt、schema
与 protocol hash；不得与 `micro-v1` 轨迹混合或事后改标签。当前仍为 diagnostic-only，尚无
行为提升或训练准入结论。

`semantic-milestone-v1` 是 semantic-v2 Atomic 专用的 teacher-only checkpoint 触发诊断。
它不改变快照、restore 或算子执行语义，只把抽象“里程碑”落实为可执行规则：稳定且仍会被后续
聚合、排名、标量或整形阶段消费的 population/join/metric artifact 在下一轮 commit；简单题零
checkpoint；长轨迹在第五个成功数据操作前对最佳已检查中间结果做一次兜底 commit。commit 后
继续向前求解，不要求 restore，也不制造错误分支；只有后续数据库事实明确否定 checkpoint
假设时才保留兼容性的可选 restore。该 profile 至多一次普通 commit，独立记录 prompt hash，
不能用于 micro-v1、Direct 或 Hybrid。

首次长轨迹 Gate8 显示 `semantic-milestone-v1` 只有 1/8 commit：模型实际经过了多个符合条件的
filter/join/aggregate 阶段，但 Text-JSON 的大段工具 schema 位于 guidance 之后。后续隔离的
`semantic-milestone-v2` 把触发规则简化为：首次 checkpoint 前累计两个成功的
filter_rows/join/group_aggregate/set_operation 中间 producer，且仍需另一个数据阶段时，下一轮
必须 commit；若精确答案已就绪则直接 answer。一个短 turn-check 重复放在 Text-JSON schema
之后，避免规则被 schema 稀释。v2 仍至多一次 commit、不要求 restore，不改变任何执行语义；
v1 保留用于冻结结果复现。

三种 mode 共享：

- `commit_checkpoint`：提交已经形成的语义里程碑；
- `restore_checkpoint`：在出现明确矛盾时恢复一个可用快照；
- `answer`：只引用当前可用的一个精确关系 artifact 作为最终答案。

工具的精确 required/optional 字段、闭合对象规则、typed predicate/expression 语法和返回结构
以 `src/tool_modules/checkpoint_relalg/` 中当前 mode 的 executable schemas 与 validator 为准。
本文件不复制完整 JSON grammar，避免形成第二套 schema 权威。

## 4. 关系 artifact 与当前环境

成功的表产生工具创建稳定、Harness 分配的 relation artifact handle。artifact 保存精确 schema、
行语义、顺序语义、来源/派生信息和可重放执行记录；模型的 reasoning 或 checkpoint 摘要不能
改写这些事实。源表始终可作为合法输入，但其 schema 与值仍须通过当前 mode 的工具因果发现，
不能从隐藏任务字段注入。

`CURRENT ENVIRONMENT STATE` 是当前事实权威，区分 opening catalog、已发现的源关系 schema、
当前活动 artifact 集合、活动 observations 与可用成功 steps。预算由 Harness 控制并进入运行
记录；最新可恢复错误位于独立的 `LAST ERROR` 段。成功状态转换原子地创建新状态；任何
验证或执行错误都必须保持逻辑状态哈希不变。失败 step id、provider 重试计数和错误预算可以
进入审计日志，但不得伪装为关系状态，也不能被后续工具引用。

最终 `answer` 的证据是被引用 artifact 的完整精确关系，而不是模型在参数或自然语言中抄写
的答案值。隐藏 gold SQL 仅在成功终止之后由隔离的本地只读 scorer 执行并比较 denotation；
SQL 文本、gold 行、值、未来输出和正确性反馈都不能进入 provider 消息。

SQLite 只是执行 backend，不是隐式语义权威。Harness 在 source 与每次 materialization 边界
校验 canonical dynamic type；TEXT/DATE/DATETIME 比较、分组、去重与排序固定为 BINARY
collation；BLOB 只允许读取、投影与等值处理，不允许作为 order key。source、可见 artifact
view 与内部 backing table 使用分离且按 SQLite identifier-equivalence 校验的命名空间；所有
`__*` 输出名保留给 Harness。用于稳定顺序的 hidden ordinal 不出现在 schema、Direct SQL、
context 或 answer 中。

每个 SQL/感知/answer 调用共享有限 timeout，并在状态发布前同时执行 artifact 行数、总字节和
单 cell 字节上限。非有限 REAL、超限结果、attached schema、未注册 TEMP object、写 SQL 与
多语句都结构化失败，且不能留下部分 artifact、observation 或状态突变。

## 5. checkpoint 是工作记忆，不是数据库证据

checkpoint 保存一个 Harness 验证过的状态快照和简短语义里程碑，用于控制长轨迹中的活动
分支。`CHECKPOINT HISTORY` 只帮助模型记住已承诺的阶段目标、里程碑与当前路径；它不能证明
任何单元格、行、聚合值或连接基数。若 checkpoint 摘要与当前环境冲突，以当前环境和 artifact
内容为准。

使用规则：

- 简单问题可以零 checkpoint 直接完成；
- 只在形成稳定的语义里程碑时 commit，不要把每次合法调用都提交；
- restore 只用于可指出的矛盾或错误分支，不用于随意探索；
- 当前 phase 已产生新事实时，可以 restore 到该 phase 的起点 checkpoint；Harness 建立新的
  recovery child 并精确恢复起点快照，而不是把“目标仍是 active id”误判为不可恢复；
- restore 不删除历史。它从被选快照建立新的活动 checkpoint/path，并计入有界预算；
- commit 或 restore 成功后开启新的 provider phase，旧阶段保留为审计记录，但不作为无限
  transcript 重复发送。

跨 phase 只保留 artifact/scalar producer 的 grounded identity 与 checkpoint 事实，不保留旧
阶段 describe/control/tool-result 全文。Provider transcript 在 commit/restore 后清空并从新的
`system + user context` 开始；完整历史仍留在不可变 audit artifact 中。

checkpoint、restore、artifact、step、错误和总轮数预算全部由 Harness 配置并写入 manifest。
达到上限必须产生稳定、可解释的终止或错误，不得通过隐藏清理、覆盖 handle 或重编号规避。

## 6. 动态上下文与 prompt 规则

每个模型 turn 由 Harness 按以下固定顺序重建 user context：

```text
QUESTION
EXTERNAL KNOWLEDGE          # 可选，只含任务给定映射/常量
CURRENT PHASE TARGETS
CHECKPOINT HISTORY
CURRENT ENVIRONMENT STATE
LAST ERROR                  # 可选，仅最新结构化错误
```

动态上下文只含当前因果前缀。数据库/schema/value/metadata 文本一律视为数据，不是指令。QUESTION
和 EXTERNAL KNOWLEDGE 决定语义约束；数据库观察不能擅自发明 singleton、时间、聚合或排名限制。
模型在关系承诺前应确定 population、row grain 和输出字段，在观察异常时继续检查而非合理化，
终止前应核对最终 artifact 的列、顺序、表示与问题要求。

prompt 鼓励最少但充分的 checkpoint，不要求为“展示工具使用”而多走步骤，也不要求把完整
关系反复读进 transcript。Schemas 是参数形状的唯一模型可见权威；system prompt 不再复制完整
工具目录、JSON grammar 或 Harness 内部实现。

## 7. 因果生成与审计

每条 episode 必须来自真实 model↔Harness 循环。模型每轮只能看到当前消息、已匹配的工具反馈
与上述重建状态。禁止把 gold SQL 编译成完整轨迹后补 reasoning、checkpoint 或感知步骤；禁止
用未来动作/输出修写早期 prompt。

诊断产物至少绑定并审计：

- protocol/scheme/mode、student/teacher prompt 与 schema 哈希；
- official provider、model、请求参数、原始 reasoning/call id 与 tool-result 顺序；
- 每步 state-before/state-after 哈希、artifact 依赖、checkpoint 活动路径与错误事件；
- 终止 artifact 和声明的 `bird-set` denotation comparison；
- fresh replay、结构、单调用/provider-history、state-mutation 和 no-leak 结果。

隐藏 evaluator 同时记录 `bird-set` 官方兼容结果与独立 strict artifact audit：后者检查 schema/
列顺序、ordered sequence、duplicate multiplicity、NULL 与 empty relation 边界。过程记录还包含
phase 长度、active path、abandoned path cost、checkpoint/restore 次数、context overflow、
artifact 资源预算与 renderer/checkpoint/executor 版本。无法由 provider usage 精确拆分的 token
指标必须显式为 unavailable，不能用估算值冒充测量值。

错误行动只留在审计轨迹和因果错误上下文中，不能直接成为正 SFT target。是否允许后续恢复
行动进入训练，必须由未来的 scheme-aware exporter 单独定义和验证。

## 8. 准入与迁移状态

当前必须使用 `diagnostic-only`：

- 可以做本地单测、SQLite 语义测试、fresh replay/no-leak 审计和经授权的官方 DeepSeek 小型
  真实轨迹诊断；
- 不得把成功轨迹直接加入现有 atomic SFT 混合；
- 不得让当前 RL 工厂把未知 scheme 自动路由到 atomic 或 action-block 环境；
- 不得仅因一次合法成功、某个 mode 表现较好或 transport 正常就宣称 promotion。

开放 SFT/RL 前，至少需要冻结 cohort/config，完成行为与可靠性 gate，所有成功样本 fresh
replay，通过结构/provider-history/no-leak 审计，建立 scheme-aware exporter 与数据身份防混用
检查，并记录明确的 admission 决策。direct、atomic、hybrid 的结果分别报告，不用 verifier
事后挑选最佳 mode 伪造单一策略结果。

2026-08-09 已实际启动一次官方 DeepSeek Direct 因果 smoke：`/models` 身份检查通过，但首轮
Chat Completions 返回 `HTTP 402 / Insufficient Balance`。该 transport-failure 记录的结构审计
与 fresh replay 均通过；没有形成 authored tool call 或语义 Harness step，因此不能报告 Direct、
Atomic、Hybrid 的能力或准确率，也没有向后两种 mode 重复发送必败请求。完整边界见
`docs/reports/evaluation/CHECKPOINT_RELALG_V1_CAUSAL_SMOKE_20260809_ZH.md`。余额恢复后必须在
同一 official-only、no-fallback 边界下重新完成三模式 smoke。

余额恢复后完成了首个 100 题因果数据诊断：冻结 teacher1500 前 100 题、Hybrid、A/Text-JSON、
官方 `deepseek-v4-flash`。结果为 63/100 `bird-set`、42/100 strict artifact、56/100 schema
match、99/100 合法终止；100/100 结构审计与 fresh replay 全部通过，581 次 provider attempt
全部绑定 Flash，总计 11,677,199 tokens。该运行没有使用 checkpoint/restore，且 21 条
`bird-set` 成功未通过 strict audit：其中 18 条是纯列标签/schema 差异，另 3 条是
duplicate multiplicity 差异，不能把 21 条统一描述为 schema 错误。因此该批不构成 checkpoint
能力或训练准入证据。最多 42 条
可暂记为严格正确的 scheme-local 候选，仍不得进入现有 SFT/RL。完整报告见
`docs/reports/evaluation/CHECKPOINT_RELALG_V1_FLASH_TEXT_JSON_HYBRID_PREFIX100_20260809_ZH.md`。
