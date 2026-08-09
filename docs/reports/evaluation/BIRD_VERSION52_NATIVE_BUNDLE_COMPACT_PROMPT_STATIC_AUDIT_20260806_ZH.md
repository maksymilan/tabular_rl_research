# BIRD Version52 Native Bundle Prompt/Token 静态审计

状态：**实现与离线静态审计完成；尚未调用外部 API，不能报告正确率或实际 provider token。**

## 目标与单变量边界

Version52 从冻结 version51 分支，保持以下内容不变：12 个 primitive functions、参数 JSON
Schema、Harness 执行、resident state、bundle pre-state 全量预校验、provider 顺序、最近 4 个
真实 assistant turns、逐 call-id tool result、最多 8 个 provider calls，以及 terminal denotation。
`answer_from_context` 仍由 Harness 硬性要求为该 turn 唯一调用。

变化只有两类：

1. prompt 去重和 bundle 调度软约束；
2. 不参与 reward 的事实型 bundle/call 统计。

Version52 仍为 diagnostic-only，且没有 scheme-aware exporter。不得把一个 multi-call turn
flatten 成多个伪 assistant turns，也不得将本实现直接作为 SFT/RL 数据源。

训练结果质量另行采用 `causal-empty-result-target-filter-v1`：中间 `row_count=0` 的调用保留为
后续 context 但不作 target；terminal 证据表为空则整轨迹排除；1x1 表内的标量零不是空结果。

## Version51 重复来源

Version51 每个请求同时发送：

- provider system prompt：21,055 字符；
- native tools JSON：9,882 字符；
- 固定合计：30,937 字符。

其中 system prompt 又包含文本工具契约约 3.4k、教师长工具说明约 7.5k、native 调用 cookbook
约 3.7k，以及三处语义重叠的 bundle/carrier 规则。函数名、required/optional 字段和嵌套参数
形状已经由 native tools JSON 提供，这些文本形成第二套重复接口说明。

## Version52 prompt

`native-schema-semantic-only-v1` 让 API-supplied native JSON Schema 成为函数名和参数形状的
唯一权威，只保留 Schema 无法表达的内容：

- CURRENT ENVIRONMENT STATE、LAST TOOL ERROR、证据与 causal history 权威边界；
- population、row grain、join logical columns、value_ref 与 exact terminal table；
- question/external knowledge 绑定、禁止无依据 singleton/time/aggregate 约束；
- 错误调用的结构化反馈与下一轮纠正；
- 简短 reasoning、空 assistant content 的 token 软约束。

调度软约束为：默认 1 call，通常不超过 3；优先 bundle 独立 perception；仅为显式且有依据的
竞争假设并列 filter；join/group/scalar/project/extreme/set-op 通常单调用。它们不改变 provider
的硬上限 8，也不把 bundle 内调用变成可依赖的顺序程序。

## 错误恢复与训练边界

教师产生的错误调用不会被删除。每个 errored assistant bundle 和对应 `role=tool` 结构化错误
继续留在 provider-native causal history；模型下一轮可以读取 type/code/message/details、原调用
与参数后纠正。训练边界明确为：

```text
错误 bundle + tool error = 输入历史，不是正向 target
下一轮纠正后的合法 bundle = 候选 target
```

这与 `error_actions_are_sft_targets=false` 一致，不会因整条 episode 最终成功而反向把错误调用
标成正例。

## 静态结果

| 指标 | version51 | version52 | 变化 |
|---|---:|---:|---:|
| teacher/provider system prompt chars | 21,055 | 6,887 | -67.3% |
| native tools JSON chars | 9,882 | 9,882 | 0 |
| fixed prompt + schema chars | 30,937 | 16,769 | -45.8% |
| fixed-200 1,414 requests message chars | 46,155,686 | 25,200,206 | -45.4% |

最后一行是把已记录 version51 请求中的 system message 反事实替换为 version52 prompt，其他
question、schema、history、tool result 和 state 全部不动。它不等于 tokenizer 或 API 账单；
native schema 仍会随每个请求发送，所以实际降幅必须由配对 live gate 验证。

## Version51 统计回放

新的 fact-only 统计器在 version51 fixed-200 上回放得到：1,412 个含调用 turns、1,650 calls、
216 个 multi-call turns；52 个 primitive errors 的结构化反馈保留率为 52/52；45 个错误 turns
都有后续调用，其中 40 个下一轮以同工具不同参数纠正。743 个产生 derived handle 的调用中，
697 个在后续参数中被引用（93.81%）。647 个 perception calls 明确标为
`observational_credit_unresolved`，不会据此自动获得正 reward。

另有 2 个 `missing_tool_calls` provider-turn errors 没有 primitive call，因此不伪装成 primitive
错误或调用级 credit。

同一 fixed-200 的 147 条 verified 轨迹中，13 条含共 21 个显式 `row_count=0` 的中间工具结果，
但 0 条最终证据表为空。新规则会去除这 21 个动作的 target loss，同时保留全部 13 条恢复轨迹
及其空结果反馈。

## 后续 live gate

后续若获数据发送授权，应在同一冻结 cohort、同一模型、同一 K、相同 completion budget、
recent-4、catalog、denotation 和 API 重试参数下配对 version51/version52。至少报告：正确率、合法
终止、过程错误、bundle size、reasoning/content 长度、prompt/completion/reasoning/cache tokens、
实际请求次数和 replay/audit。只有 token 明显下降且行为不回归，才能继续 scheme-aware exporter
与训练准入工作。
