# Iterative-SQL v5：外部 prompt 建议逐项审核

## 结论

本次审核将建议落成 `iterative-sql-v5`，但不把它描述为准确率提升版本。v5 保留 v4 的两个
公开工具、只读执行、prior-exact submission、错误恢复和 hidden scorer 边界，只修改模型可见
prompt、最新任务副本的命名、provider-facing transport 文案，并给 syntax-only
`query_shape_audit` 增加两个不解释任务语义的数据依赖事实。冻结 v3/v4 prompt 和反馈格式继续
可复现。

本报告完成时，v4 的开发 Gate15 为 12/15、15/15 legal，v5 尚无独立行为结果。随后冻结 v5
在 active baseline300 Prefix20 上得到 14/20 versus v4 12/20；最终 Prefix50 为 36/50 versus
34/50、4 gains / 2 regressions、exact `p=0.6875`，但 legal、process errors 和 tokens 回退，
因此停止扩量。详见 `BIRD_ITERATIVE_SQL_V5_PREFIX50_EXPANSION_20260805_ZH.md`。该结果不改变
本报告的 SFT/RL 禁入结论。

## 逐项审核

| 建议 | 处理 | 审核理由与实现 |
|---|---|---|
| 1. 明确信息源优先级 | 改写后采纳 | 不采用“QUESTION 无条件高于 EXTERNAL KNOWLEDGE”。本项目的 external knowledge 常承载 BIRD 的绑定字段映射、定义与公式，降级会直接改变任务。v5 明确职责边界：QUESTION 规定任务、population、measure、representation 和 order；EXTERNAL KNOWLEDGE 提供必须共同满足的映射、定义和公式；数据库只提供 schema/value facts；最新任务副本不增加解释。若二者表面冲突，模型应检查而不是静默丢弃任一约束。 |
| 2. 数据不能创造额外语义限制 | 采纳 | 删除 v4 中 “observed data explicitly requires them” 的宽松口径。v5 明确数据只能揭示如何保存已声明的区分，不能新建 filter、DISTINCT/dedup、时间限制或 aggregation policy。 |
| 3. 明确 bounded preview 的证据边界 | 采纳 | 明确 preview 缺失不等于全集缺失；无 `ORDER BY` 时预览顺序无意义；截断预览不能证明 completeness、uniqueness、extrema、frequency、duplicates 或 boundary ties；必要时用聚焦的 `COUNT`、`EXISTS`、`GROUP BY`、`MIN/MAX` 或 predicate query。 |
| 4. 禁止把预览值硬编码为最终答案 | 采纳 prompt；增加事实审计，不做无条件拒绝 | v5 禁止把数据库派生值复制到 literal-only final `SELECT`，并在 `query_shape_audit` 中增加 `has_from` 与 `literal_only_select`。Harness 不直接拒绝所有 literal-only query，因为题目明确要求常量、标签或无表 SQLite 表达式时可能合法；环境不知道题目语义，不能安全替模型裁决。 |
| 5. 全面检查改成相关不确定性检查 | 采纳 | 只在概念具有历史/事件属性、来源含糊或存在多个 plausible sources 时扩展到 relation/event/history/fact tables，避免简单题机械扫表。 |
| 6. `one unresolved fact` 放宽为紧密不确定性集合 | 采纳 | reasoning 和动作策略都改为 highest-priority fact **or tightly related uncertainty set**；允许一条聚焦 SQL 同时解决耦合问题，但禁止扫描无关数据。 |
| 7. 明确 top-N/tie 默认行为 | 采纳 | top N 默认返回 N 行；只有题目明确要求才扩展 boundary ties；次级键不能改变答案集合，只能在任意并列顺序都合法时用于稳定展示。 |
| 8. 消除 `<think>`/native reasoning 与客户端细节混杂 | 采纳并做 provider-aware 隔离 | DeepSeek-facing v5 只要求 native reasoning，不再要求 “Keep `<think>`”；客户端重建 canonical envelope 的实现细节不再发给模型。非 split/local carrier 仍保留必需的 `<think>` 契约。v3/v4 继续使用冻结文案以保持 hash。 |
| 将 statement/read-only/JSON/prior-execution/repeat/timeout 放到工具端 | 已满足，保留 | 这些约束已经由 strict parser、SQL safety validator、SQLite read-only connection、deadline、success registry 和 no-progress check 强制；v5 prompt 只简要声明 hard boundary，不重复长实现细节。 |
| `submit_sql(sql)` 改为 `submit_sql(query_id)` | 暂不采纳 | 当前 harness 已按“首尾空白 + 一个尾部分号”规范化后要求 exact prior execution，不存在已观测到的复制失败根因。改为 query id 会改变公开 schema 和终端 action，破坏与 v4 的单变量对照，也不符合“最后通过一个 SQL 语句提交答案”的既定工具方向。若以后审计证明 SQL 重写是显著错误源，再单独做版本化 ablation。 |
| 使用整段紧凑改写 | 部分采纳 | 采用其分层结构、prompt-injection 防护、candidate 成功后立即提交和语义检查原则；保留项目需要的严格错误边界、external-knowledge 权威性和 exact SQL submission。 |

## v5 的可执行变化

1. 协议身份为 `iterative-sql-v5` / `execute-sql-submit-sql-v5`，registry 为
   `tool-scheme-registry-v8`；DeepSeek Flash + lazy catalog 的 prompt protocol hash 为
   `b6465c6e222c12c2`。最终 provider-facing prompt 为 5,945 字符，接近 v4 的 5,820 字符，
   避免了首版分层改写造成的 7,181 字符膨胀。
2. 公开参数不变：`execute_sql(sql)`、`submit_sql(sql)`。
3. 最新任务副本标题改为 `TASK SPECIFICATION COPY (VERBATIM)`，明确它只是原文 recency
   copy，不是模型或 harness 生成的摘要。
4. v5 成功 SQL 观察的 `query_shape_audit` 在 v4 字段之外增加：
   - `has_from`：去除字符串、标识符引用和注释后，SQL 是否含 `FROM` token；
   - `literal_only_select`：SQL 是 `SELECT/WITH` 候选且没有 `FROM` 的 syntax fact。
   这些字段不访问 question、external knowledge、gold SQL、gold rows 或 verifier，也不直接
   判定答案对错。
5. DeepSeek v5 prompt 删除内部 canonical envelope 重建说明；v3/v4 的 provider 文案保持
   原样，避免历史 protocol hash 漂移。

## 保持不变的边界

- `submit_sql` 仍提交 SQL 文本，而不是 query id；
- 最终 SQL 仍必须先由 `execute_sql` 完整成功执行并 exact-match；
- read-only、单 statement、schema PRAGMA allowlist、timeout、重复成功调用拒绝和结构化错误
  恢复均不变；
- executable wrong answer 仍立即终止且不返回 judge feedback；
- gold SQL、gold rows 与隐藏验证器数据仍不可见；
- v5 仍为 `diagnostic_only`，没有 SFT exporter 或 RL admission。

## 验证结果

- focused prompt/provider/scheme tests：60/60 通过；
- tool-module tests：113/113 通过；
- eval tests：53/53 通过；
- SFT tests：197/197 通过；
- v3 DeepSeek Flash protocol hash 保持 `64d865970246b3c9`；
- v4 DeepSeek Flash protocol hash 保持 `00b4e630adf2faaf`；
- 冻结 v4 Gate15 fresh replay：12/15 replay correct、15/15 structural pass，与原记录一致；
- `git diff --check` 通过。

以上只证明实现与历史兼容性。新行为结论仍只能来自独立 holdout，不能复用 v4 的 15 题开发
集声称提升。
