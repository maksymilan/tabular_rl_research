# Atomic version40 精简 Prompt 与完整推理历史 Gate50

## 结论

version40 **未通过首 50 题扩量门槛，不继续运行剩余 150 题，也不准入 SFT**。

在与 version24 完全相同的冻结 200 题输入的前 50 题上，DeepSeek v4 Flash 得到
**37/50 = 74% `bird-set`**；同题 version24 基线为 **42/50 = 84%**。两版都
50/50 合法终止，但配对结果为 0 个恢复、5 个回归，净变化 -5，exact two-sided paired
binomial `p=0.0625`。这不能与 version24 的完整 200 题 145/200 = 72.5% 横向比较；
有效比较只能是同一前 50 题的 37/50 对 42/50。

五个新增回归全部已经找到正确实体或行集，最后却选择了错误的输出列、列顺序或表示形式。
因此本轮下降不是 join 执行错误、provider carrier 故障或非法终止造成的，而是终止投影
约束在精简 prompt 中权重下降。

## 变更与冻结设置

version40 在 version39 执行语义上同时测试四项模型可见变化：

1. 从公开工具中移除 `plan`；
2. 将只读行观察工具 `read_subtable` 重命名为 `inspect_rows`，执行仍确定性降低到同一
   只读操作；
3. 使用一个 7,503 字符的分层精简 prompt；
4. 保留全部成功和被拒绝的 reasoning；成功工具调用与原样 observation 只保留最近四对，
   最新错误通过结构化 `LAST TOOL ERROR` 保留。

本轮与 version24 同题基线对齐的设置：

- 输入：`bird_train_tool_interface_validation200_version4.jsonl` 的前 50 题；
- 输入 SHA-256：
  `6b469215ac96e09c1046fee539ff4c4885b58e9b7a0b6fdb2187082267dcde36`；
- 模型：`deepseek-v4-flash`；
- 每题一次语义尝试，错误答案不重试；
- `max_steps=30`，`max_tokens=2048`；
- provider：thinking enabled、reasoning effort high、JSON Output；
- 上下文：`rolling-legal-history`，`history_turns=4`；
- 判分：`bird-set`；
- 训练准入：`diagnostic_only_pending_protocol_scale_gate`。

version40 权威哈希：

- protocol：`ea08e693a5ca03c1`；
- teacher/student canonical prompt：
  `3990f469d5fb9b0af8def94271609a308572b0fec9f0fde3815f533eb730e4fd`；
- provider prompt：
  `0445f0fd9bde11b53495f1d546f14372fa7614f5635ed71474a331081433e73d`；
- public tool schema：
  `b7761922ee1164520630f4dbe66c7c766abb9257e617ba7061e4730884ae08c4`。

## 配对结果

| 指标 | version24 | version40 | 变化 |
|---|---:|---:|---:|
| 正确 | 42/50 | 37/50 | -5 |
| 正确率 | 84% | 74% | -10 pp |
| 合法终止 | 50/50 | 50/50 | 0 |
| 合法动作 | 294 | 298 | +4 |
| 平均动作 | 5.88 | 5.96 | +0.08 |
| 过程错误 | 3 | 9 | +6 |
| 有过程错误的题 | 3 | 7 | +4 |
| API request attempts | 295 | 309 | +14 |
| prompt tokens | 1,499,140 | 1,266,284 | -15.5% |
| completion tokens | 58,500 | 98,432 | +68.3% |
| reasoning tokens | 47,746 | 86,555 | +81.3% |
| total tokens | 1,557,640 | 1,364,716 | -12.4% |
| completion retries | 1 | 10 | +9 |

配对四格：

- 两者都对：37；
- 两者都错：8；
- version40 独赢：0；
- version24 独赢：5。

五个回归题为：

- `bird_train_02408`
- `bird_train_02901`
- `bird_train_03390`
- `bird_train_06336`
- `bird_train_06489`

## 错误归因

### 1. 新增五个回归全部是终止输出形态

这些轨迹均已定位正确的实体或行集，随后发生：

- 保留了未请求的 ID 或辅助列；
- 把原本独立的姓名字段拼接成一个字符串；
- 返回 country code 而不是对应的 nation name；
- 将 “word and id” 按 `id, word` 的反顺序输出；
- 将对象样本 ID 改成更自然的对象类别 label。

version24 在同题上都给出正确终止表。旧版这五题也都没有使用 `plan`，因此不能把回归归因
于移除 plan。五题均合法、无 join-call failure；`inspect_rows` 的别名降低也没有改变关系
执行值。最直接的行为差异发生在终止列选择。

### 2. 当前 13 个错误中，8 个是“行/实体正确，输出表形态错误”

按合法轨迹中的筛选、连接、排序结果和终止 evidence 审计：

- 输出字段、字段顺序或表示错误：8/13；
- population、cardinality、聚合或异常数据解释错误：5/13。

八个输出形态题为 `01152`、`02408`、`02901`、`02925`、`03390`、`05316`、`06336`、
`06489`。其余五题为 `00593`、`03688`、`05440`、`06026`、`06492`。

这 8/13 是“关系选择基本正确但最终表不匹配”的诊断口径，不等于 benchmark 接受自然语言
同义答案；`bird-set` 仍将它们正确判为失败。

### 3. 过程错误增加，但不是主要准确率原因

version40 有 8 个 `argument_validation_error` 和 1 个 `execution_error`。错误集中在：

- `extreme_value_select.order_by` 的列表/方向格式；
- `group_aggregate` columns layout 的配套字段；
- `scalar_compute.operation` 的受限枚举；
- `condition_filter` 的 scalar/table reference；
- 一个非法的 `inspect_rows.limit`。

13 个错误答案中有 10 个完全没有过程错误。接口说明退化会增加成本和恢复负担，但本轮主要
准确率损失仍是零过程错误的语义/输出决策。

## 为什么“更短”反而更差

version24 的实际 provider system prompt 为 16,400 字符；version40 为 7,503 字符，
减少 54.3%。但总 token 只减少 12.4%，因为教师生成的 reasoning token 反而增加 81.3%，
API 请求数和 completion truncation retry 也上升。当前“适中推理”文字没有把行为控制在
更短、更确定的决策链上。

精简还删除了 version24 中不重复、且与本轮回归直接相关的操作性约束：

- 保留数据库输出槽，不把 ID/code 换成 label；
- 未明确要求格式化时，不拼接独立源字段；
- top-k 后只保留请求字段；
- 终止前用 `project` 或 `return_columns` 固定精确列顺序；
- `extreme_value_select`、`scalar_compute` 和 columns-layout aggregate 的完整形状示例。

version40 仍有“exact rows/columns/order”的抽象规则，但实测说明这句概括不能替代上述少量
具体边界。这里删掉的不是冗余，而是模型用来区分几个高混淆决策的判别信息。

## 下一步建议

不要恢复 16K/21K 的整段旧 prompt，也不要在本轮基础上继续叠加通用语义文章。建议新建
version41，只改变 prompt，在 version40 的工具、历史、执行和反馈上恢复以下最小信息：

1. 新建一个短的 `OUTPUT CONTRACT`，只说一次：先列出请求输出槽及顺序；未明确要求格式化
   时保留独立字段；不得把 ID/code 换成 label；不得附带筛选、排序、计数或 join helper
   列；终止前用 `project`/`return_columns` 形成完全一致的 evidence table。
2. 只恢复三个高熵调用的各一个合法示例：
   `extreme_value_select(order_by=["col DESC"])`、`scalar_compute` operand、
   `group_aggregate(output_layout="columns")`。不恢复低熵工具的重复解释。
3. 先冻结 8 个输出形态失败题加 8 个同类正确控制做 Gate16。建议门槛：恢复至少 4/8、
   控制保留至少 7/8、过程错误不高于 version40；通过后再跑同一 Gate50。
4. 若只恢复 prompt 仍失败，再单独测试 reasoning history 策略。不要同时改工具名、plan、
   history 和 prompt，否则无法判断完整历史是否造成错误假设锚定。

## 审计产物

- version24 同题基线：
  `data/trajectories/tool_usability_20260724/version24_fixed200_first50_bird_set.all.jsonl`
  （SHA-256
  `aaa07ed99dc8da9f4f77742d8c42da38c1a8f9d4bc2598cd658a05a4ea340241`）
- version40 全记录：
  `data/trajectories/tool_usability_20260729/version40_concise_history_fixed200_first50_bird_set.all.jsonl`
  （SHA-256
  `3568ed70ef5a3b67cabdb868d2b83e1a05f6ba19aaf3b44347b4f13bd6d6a3c0`）
- version40 成功轨迹、失败轨迹和 manifest 位于同一目录，文件前缀为
  `version40_concise_history_fixed200_first50_bird_set`。

本轮没有继续发起后 150 题，避免在首批已出现严格配对净回退后继续消耗教师 API。
