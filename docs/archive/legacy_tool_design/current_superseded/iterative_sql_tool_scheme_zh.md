# Iterative-SQL：先探索、后提交的双 SQL 工具协议

## 当前结论

`iterative-sql-v6` 是当前注册、与 atomic、native-tool-bundle、action-block、
relational-program、direct-sql-search 互斥的诊断 tool scheme。模型每轮只看到：

1. `execute_sql(sql)`：执行一条只读 SQL，返回真实但有界的结果预览，不终止 episode；
2. `submit_sql(sql)`：提交最终答案 SQL。只有 SQL 安全、已被同文执行成功且完整执行成功时
   才终止并进入隐藏判分。

这里的重点不是增加 SQL 的表达能力，而是让模型用数据库反馈逐步消除 schema、value、join、
population、grain 与输出形状的不确定性，同时保留 direct SQL 的低接口负担。冻结 15 题
开发门上，v3 为 **7/15**，根据其失败形成的 v4 为 **12/15**；两者均 15/15 legal。v4 相对
v3 为 6 gains / 1 regression、exact paired `p=0.125`，actions 97→91、process errors 4→1，
tokens 增加 1.6%。但 v4 使用了同一集合的 v3 失败信息，因此这是 in-sample 优化结果，仍为
开发集诊断而非推广结果。

v5 根据对外部 prompt 建议的逐项审核形成，只改变 prompt 分层、证据边界、任务 recency
标签、DeepSeek-facing 文案和 syntax-only data-dependency facts；公开工具与执行语义不变。
在与旧 fixed-200 完全不重叠的 active baseline300 冻结 Prefix20 上，v5 曾以 14/20 对 v4
12/20、2 gains / 0 regressions 呈正向信号。扩展到 Prefix50 后，v5 为 **36/50**、v4 为
**34/50**，配对变为 4 gains / 2 regressions，exact `p=0.6875`；legal 为 49/50 versus
50/50，process errors 为 10 versus 7，tokens 为 984,082 versus 877,576（+12.1%），actions
基本持平。准确率小幅正向但不显著，可靠性与成本门失败，因此停止继续扩量，仍不能进入
SFT/RL。DeepSeek Flash + lazy catalog 的 v5 protocol hash 为 `b6465c6e222c12c2`；prompt 为
5,945 字符，接近 v4 的 5,820 字符。

v6 是 Prefix50 后的最小 prompt-only 版本。它不改变工具、SQL 执行、错误反馈、上下文、
preview 或评分，只把关系型答案的表示边界写成显式规则：任何答案都是 SQL 结果表；标量是
1×1 表；一个映射字段对应一列；多个映射字段必须按声明顺序分别占列；即使问题使用单数
“full name”，也不能据此拼接多个字段。只有 QUESTION 或 EXTERNAL KNOWLEDGE 明确要求一个
formatted/combined/string value 时才能拼接。该修改直接针对 v5 的两条多字段表示回归，但
Prefix50 已消费，不能用其证明改进。随后在完全不重叠的冻结第 51–70 题上，v6 与 v5 均为
**12/20**、**20/20 legal**，配对 1 gain / 1 regression、exact `p=1.0`；v6 actions 97 versus
100，但 tokens 340,873 versus 331,641（+2.8%）。该切片没有多字段姓名映射目标题或明确拼接
控制题，因此只能说明一般行为没有净增益，不能直接验证目标规则。v6 仍为 diagnostic-only。
DeepSeek Flash + lazy catalog 的 v6 protocol hash 为 `176ce977b411636e`；v5 保持可复现。

为直接覆盖目标能力，另从 tool-compatible 池冻结了与完整 baseline300 和历史 fixed-200 均
零重叠的 Target Gate20。v6 为 **15/20**，v5 为 **10/20**，5 gains / 0 regressions，exact
`p=0.0625`；两臂均 20/20 legal。6 个 multi-field targets 上 v6 为 5/6 versus v5 1/6，
4 gains / 0 regressions；single-field 与 ordinary controls 共 10 题没有回归。v6 actions、
errors、tokens 分别为 99、1、339,600，均优于 v5 的 111、6、371,036。全部轨迹通过 replay
与结构审计。预注册时把 `field1+field2` 当作明确拼接控制，但 reference 实际要求分列，所以
反向控制类别失效。结论是 v6 通过目标能力门，只授权新的代表性 paired gate；仍不进入
SFT/RL。

随后按用户授权只对冻结 v6 做 baseline300 全量单臂测试：第 51–70 题复用原审计轨迹，其余
280 个 episode 为新请求。full300 为 **215/300（71.67%）**、**298/300 legal**；全部
300 条 fresh replay 与结构审计通过。前 70 题已参与设计/诊断，因此主要推广读数是未用于
v6 设计的第 71–300 题：**168/230（73.04%）**、**228/230 legal**。两个干净 115 题切片
均为 84/115。全量共 1,698 actions、72 process errors、7,000,561 tokens；两个 max-step
episode 占 610,640 tokens。自然出现的三个 multi-field full-name 映射均正确分列，但集合仍
没有真实的显式单字符串拼接控制。由于本次没有运行 clean230 v5 arm，这个 full300 结果只
建立 v6 的绝对性能与可靠性，不能证明 v6 总体优于 v5、atomic 或 native-tool-bundle，也不
改变 diagnostic-only / 禁止 SFT-RL admission 的边界。

## 模型可见协议

探索动作：

```json
{"tool":"execute_sql","arguments":{
  "sql":"SELECT department_id, COUNT(*) AS people_count FROM people GROUP BY department_id"
}}
```

最终动作：

```json
{"tool":"submit_sql","arguments":{
  "sql":"SELECT department_id, COUNT(*) AS people_count FROM people GROUP BY department_id"
}}
```

本地文本 carrier 仍使用一个非空 `<think>` 加一个严格 raw JSON action；DeepSeek
split-response transport 则只向模型描述 native reasoning 与可见 JSON，不暴露客户端内部
canonical envelope 重建细节。两个工具的 `arguments` 都必须恰好只有一个非空 `sql` 字符串，
不做 parser repair。

Prompt 明确要求模型：

- 先用 `execute_sql` 获取足够信息，不能把第一个非空结果当作答案已经正确；
- 在实质查询前固定 answer unit、eligible population、row grain、operations 与输出槽位；
- 检查真实 stored literal、join multiplicity、NULL、日期/文本表示、聚合单位与重复语义；
- 数据库事实不能创建新的 filter、deduplication、时间限制或 aggregation policy；
- bounded preview 的缺失、顺序与截断部分不能证明全集的缺失、顺序、唯一性、极值或并列；
- 只在来源含糊、概念属于历史/事件或存在多个 plausible sources 时扩展检查；
- top N 默认保持 N 行，除非任务明确要求扩展 boundary ties；
- 不凭空加入 `DISTINCT`、top-1、latest/current、rounding 等限制，也不能把预览值抄成
  literal-only final query；
- 最终逐列审计行、列、列顺序和 representation，不带诊断/helper 字段；
- 把最终答案始终视为表：标量为 1×1，一个映射字段为一列，多个映射字段按声明顺序分别
  输出；只有任务明确要求单个格式化/组合字符串时才拼接；
- 只有 exact final query 已经通过 `execute_sql` 后才原样 `submit_sql`。

v5/v6 把原样 question 与 external knowledge 作为 `TASK SPECIFICATION COPY (VERBATIM)` 放在
每轮最新上下文，明确它不是派生摘要。成功 SQL 反馈继续包含确定性的
`query_shape_audit`：除 v4 的结果列数、DISTINCT、ROUND、WHERE、IS NULL、GROUP BY、
ORDER BY、LIMIT、join 和聚合函数字段外，v5 增加 `has_from` 和 `literal_only_select` 两个
syntax facts。该 audit 不解释问题、不访问 gold，也不判断 SQL 语义正确性；literal-only
查询不会被环境无条件拒绝，因为某些显式常量任务可能合法。

## 执行与恢复边界

`execute_sql` 接受一条 `SELECT`、`WITH`、受限 schema PRAGMA 或
`EXPLAIN QUERY PLAN SELECT/WITH`。Schema PRAGMA 只允许：

- `table_info`、`table_xinfo`；
- `foreign_key_list`；
- `index_list`、`index_info`、`index_xinfo`。

连接启用 `PRAGMA query_only=ON`，生成 SQL 有 SQLite VM deadline；多语句、写操作、可写
PRAGMA、非法 EXPLAIN 和 terminal PRAGMA 都被拒绝。字符串常量内部的分号不会被误判为
多语句。

`submit_sql` 还必须满足 prior-execution grounding：忽略首尾空白和一个结尾分号后，它必须
与本 episode 中某条成功 `execute_sql` 完全一致。该约束保证最终 SQL 已看过真实执行反馈，
同时不让环境替模型修写 SQL。

以下错误都保留在同一 episode，消耗 action budget，成功 workspace 不改变，并作为结构化
`LAST SQL ERROR` 返回：

| 错误 code | 含义 | 下一步仍可做什么 |
|---|---|---|
| `read_only_sql_rejected` | 写操作、多语句、非法 PRAGMA/EXPLAIN 或 terminal 非 SELECT/WITH | 改写为合法 SQL |
| `final_sql_not_inspected` | 最终 SQL 没有先被完全相同的 `execute_sql` 成功执行 | 先执行该 SQL，再提交 |
| `read_only_execution_failed` | SQLite syntax/schema/runtime/timeout 失败 | 根据实际错误修改 SQL |
| `successful_action_repeated` | 精确重复已成功的探索 SQL | 使用 resident fact 或查询新不确定性 |
| `invalid_action_contract` | 工具名、JSON 或参数结构非法 | 按两工具 schema 重发 |

只有最终 SQL 完整执行成功才结束 episode。此后 hidden verifier 做 `bird-set` 判分；如果 SQL
可执行但语义错误，episode 直接以 wrong answer 结束，不把 gold rows、gold SQL 或 judge
差异返回给模型。隐藏 verifier 自身异常也只记为 evaluator fault，绝不进入模型反馈。

## 上下文管理

`CURRENT SQL STATE` 是成功事实的唯一权威副本：

- 最近 6 个成功动作保留 exact output；
- 再早 6 个动作压缩成 fact-only card；
- 更早动作只累计 archived count；
- recent-4 历史 observation 只指向 resident state，不重复序列化 row payload；
- 较早成功 reasoning 被压成 continuity marker，但原 action 保留；
- 最新 rejected action 只出现在 `LAST SQL ERROR`，不进入成功状态。
- v5/v6 在最新 observation 末尾逐字重复 question/external knowledge，避免任务规格被长 SQL
  state 推到注意力远端，同时明确副本无额外权威性。

这沿用了已在 direct-sql-search-v2 中实现的去重机制，但使用独立的 prompt、工具 schema、
protocol hash、interface id 和 manifest identity。

## 代码与入口

- 公共 prompt、schema、hash 与 strict parser：`src/tool_modules/iterative_sql/protocol.py`；
- causal provider loop、只读执行、结构化反馈与 hidden scoring：
  `src/tool_modules/sql_common/runner.py --interface execute-sql-submit-sql-v6`；
- fresh replay / structural / no-hidden-input-key audit：
  `src/tool_modules/iterative_sql/audit.py`；
- scheme registry：`src/tool_modules/registry.py`，当前为 `tool-scheme-registry-v11`；
- 统一 evaluation / external-teacher launcher：`src/eval/run_tool_scheme.py` 与
  `src/sft/generate_tool_scheme_rollouts.py`。

推荐诊断入口：

```bash
PYTHONPATH=src/harness:src/sft:src/eval:src/rl \
  .venv/bin/python -u src/eval/run_tool_scheme.py \
  --tool-scheme iterative-sql -- \
  --tasks-json TASKS.jsonl --result-dir RESULT_DIR \
  --n 20 --workers 4 --max-steps 30 --api-retries 10 \
  --denotation-comparison bird-set
```

新 DeepSeek 请求仍只能使用官方 `https://api.deepseek.com`；runner 会拒绝第三方 endpoint。
Manifest 固定记录 protocol/interface/tool-schema hash、context/state/preview profile、prior-exact
submission policy、`wrong_answer_feedback=hidden_none`、`gold_visible_to_model=false`、
`sft_export_eligible=false` 与 `training_admission=diagnostic_only`。

历史 v5 / `execute-sql-submit-sql-v5` 保留 DeepSeek Flash prompt hash
`b6465c6e222c12c2`；v4 / `execute-sql-submit-sql-v4` 保留 hash
`00b4e630adf2faaf`；v3 / `execute-sql-submit-sql-v3` 保留 hash `64d865970246b3c9`；更早的
v2 / `execute_sql_submit_sql_v2` 也仅供显式复现。旧结果不能重新标记为 v6。完整 Gate15 见
`docs/reports/evaluation/BIRD_ITERATIVE_SQL_V4_OPTIMIZATION_GATE15_20260805_ZH.md`；v5 的建议
取舍与非推广边界见
`docs/reports/evaluation/BIRD_ITERATIVE_SQL_PROMPT_REVIEW_V5_20260805_ZH.md`；独立 Prefix20
结果见 `docs/reports/evaluation/BIRD_ITERATIVE_SQL_V5_INDEPENDENT_PREFIX20_GATE_20260805_ZH.md`；
最终 Prefix50 扩展结论见
`docs/reports/evaluation/BIRD_ITERATIVE_SQL_V5_PREFIX50_EXPANSION_20260805_ZH.md`；v6 的最小规则
变更见
`docs/reports/evaluation/BIRD_ITERATIVE_SQL_V6_RELATIONAL_OUTPUT_SHAPE_RULE_20260805_ZH.md`；独立
Gate20 结果见
`docs/reports/evaluation/BIRD_ITERATIVE_SQL_V6_DISJOINT_GATE20_20260805_ZH.md`；目标能力门见
`docs/reports/evaluation/BIRD_ITERATIVE_SQL_V6_TARGET_GATE20_RESULT_20260805_ZH.md`；全量单臂
结果见
`docs/reports/evaluation/BIRD_ITERATIVE_SQL_V6_FLASH_FULL300_20260805_ZH.md`。
