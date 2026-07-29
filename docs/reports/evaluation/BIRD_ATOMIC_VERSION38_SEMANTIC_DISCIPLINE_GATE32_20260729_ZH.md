# Atomic version38 外部教师语义约束 Gate32 审计

## 结论

version38 的外部教师 prompt **未通过扩量门槛**。

在冻结的 32 题配对集合上，DeepSeek v4 Flash 经基础设施失败恢复后得到
**18/32 `bird-set`**，相对同题 version37 的 **16/32** 只有 6 个恢复、4 个回归，
净增 2，exact two-sided paired binomial `p=0.75390625`。目标恢复为 6/16，原正确控制
只保留 12/16，均未达到预注册阈值。该变化不能证明 prompt 带来稳定能力提升，也不能
用于 SFT 数据准入。

version38 的作用并非完全为零：它在若干明确的终止形态、连接总体和实体计数题上产生了
符合约束的恢复；但教师仍会在推理中明确复述约束后主动覆盖它。与此同时，四个控制回归
暴露出“自然语言实体粒度”与 benchmark 明示/隐含口径冲突时，全局约束会把模型推向另一种
同样合理但不符合标注的解释。

## 冻结设计

- 模型：`deepseek-v4-flash`
- 工具方案：atomic
- 协议：version38，`think-json-v1`
- 上下文：`rolling-legal-history`，`history_turns=4`
- provider carrier：DeepSeek thinking + JSON Output
- 计划策略：optional
- 判分：`bird-set`
- 语义尝试：每题一次；错误答案不重试
- 目标：version37 的 16 个失败题，均有一个预先验证通过的一步反事实修正
- 控制：同一次 version37 运行中全部 16 个正确题
- 执行顺序：目标与控制交错
- 训练准入：`diagnostic_only_pending_protocol_scale_gate`

version38 只改变 external-teacher generation guidance。student runtime prompt、公开工具 schema、
参数、执行、状态和反馈均与 version37 相同。权威运行哈希为：

- teacher canonical prompt：
  `677135c9a6b025e4090376051af2c5297e493d88ec4f683f4b80529478fb3571`
- teacher provider prompt：
  `a542de1833bacdc448d0557bef146972f79892797dfa42489ccdf8ae8ad1c2e6`
- student runtime prompt：
  `81329794b41050dd62526e870bda6c5df4d68e095c2576c2a75aa3758131bd54`
- public tool schema：
  `4d9e6cae7652ba958f316177367f18c8b68a48c4cc7e447e1b11eb3d13487125`

## Provider/API 结果

### 首轮：请求级重试上限 3

首轮 32 题只有 12 题形成合法终止：

- 正确：8
- 错误答案：4
- `provider_carrier_error`：19
- `api_error`：1

19 个 carrier 终止都发生在 DeepSeek 连续三次请求返回空 visible content 后；已有响应中
可以存在 reasoning tokens，但没有可供 strict JSON carrier 解析的可见动作。另一个 API
错误是 `RemoteDisconnected: Remote end closed connection without response`。

因此首轮表面上的 8/32 不是语义正确率，不能把 20 个未完成 episode 算成模型答错。

### 仅基础设施失败恢复：请求级重试上限 10

只重跑上述 20 个 provider/API 中断题，不重跑四个已经形成错误答案的题：

- 合法终止：20/20
- 正确：10/20
- 错误答案：10/20
- terminal carrier/API failure：0

这说明更大的请求级重试预算能够消除本次终止性载体故障。它不等价于语义 pass@2：
保留口径只替换没有形成语义结论的 provider/API 中断，任何错误答案都只保留第一次结果。

两次实际 API 运行合计消耗 4,167,025 tokens、646 次 request attempts、215 次 carrier
retries、29 次已记录 transport retries 和 7 次 completion retries。当前代码对最终抛出的
transport exception 不携带本次调用的累计 retry usage，因此 terminal transport retry
计数仍存在可观测性缺口。

不能仅凭本次数据把空载体上升归因于 version38 prompt：没有同时段 version37 provider
控制。已知的是 version38 provider system prompt 从 version37 的 19,020 字符增至
21,557 字符（+13.3%），而提高请求重试后 20/20 能完成。两者都应在后续并发 A/B 中控制。

## 配对结果

| 指标 | version37 | version38（基础设施恢复口径） |
|---|---:|---:|
| 正确 | 16/32 | 18/32 |
| 合法终止 | 31/32 | 32/32 |
| 过程错误 | 8 | 7 |
| 合法步骤 | 247 | 290 |
| 平均步骤 | 7.72 | 9.06 |
| 保留 episode tokens | 1,539,900 | 3,105,935 |

配对四格为：

- 两者都对：12
- 两者都错：10
- version38 恢复：6
- version38 回归：4

14 个 version38 错误答案中，10 个没有任何过程错误；主要差距仍是语义决策，而不是接口
合法性。

预注册扩量门槛：

| 门槛 | 要求 | 实际 | 结果 |
|---|---:|---:|---|
| 16 个目标中的恢复 | ≥8 | 6 | fail |
| 16 个控制中的保留 | ≥15 | 12 | fail |
| 配对净增 | ≥7 | +2 | fail |
| 合法终止 | ≥31 | 32 | pass |
| 恢复后 terminal API/infra failure | 0 | 0 | pass |

总体：**fail，不扩量，不准入 SFT。**

## 16 个目标逐题结果

| example | version38 | 轨迹判断 |
|---:|---|---|
| 6489 | 错 | 观察到 `OBJ_SAMPLE_ID=18`，仍连接类别表并输出 `paper`；显式字段映射再次被覆盖。 |
| 593 | 错 | 已计算两条均满足条件的 11/18 天疗程，仍无依据选择最早一条。 |
| 4189 | 对 | 使用 inner join，只保留有 translated review 的免费体育 App 配对。 |
| 2408 | 对 | 最终只保留请求的 `author_name`。 |
| 5440 | 错 | 看到异常年份 `800190` 和 NULL homepage 后仍用 left join 合理化异常，而未先固定可连接 Journal 的总体。 |
| 6454 | 对 | 对 rail/mail 计数做 top-1，并只输出获胜 ship mode。 |
| 3969 | 对 | 最终只输出存储的 gender 值 `F`。 |
| 2438 | 错 | 题面没有 latest/current，仍为每个订单取最新状态并附带 order ID。 |
| 2513 | 对 | 按题目实体对 `license_no` 做 `count_distinct`，得到 distinct businesses。 |
| 1050 | 错 | 修复了两个金额字段的逗号解析，却又无依据乘以 `Order Quantity`，改变了外部给定的 net-profit 公式。 |
| 4869 | 错 | 无依据增加 `EndDate IS NULL` current-only 条件，并对员工去重。 |
| 3011 | 对 | 去除 `team_count` 辅助列，只输出 first/last name。 |
| 5544 | 错 | 把 6 条 representative-term rows 当作 6 位 representatives，没有按 `bioguide` 去重。 |
| 4906 | 错 | 外部映射要求计 object samples，模型仍改成 distinct images 后求比值。 |
| 3724 | 错 | top-5 排名正确，但最终继续携带 `rental_count` 辅助列。 |
| 5161 | 错 | 外部知识指定普通 `COUNT(recipe_id)`，模型按“recipes”常识去重后得到另一人口粒度。该题本身存在明显的自然语义/benchmark 口径冲突。 |

六个恢复集中在三类：

1. 终止表字段精确化：2408、6454、3969、3011；
2. 先固定关系总体：4189；
3. 题目实体与事实行区分：2513。

但 6489、593、2438、4869、4906 等轨迹表明，prompt 中已经逐字覆盖的规则仍未成为稳定
行为约束。教师会在长 reasoning 中重新解释任务并覆盖显式映射。

## 四个控制回归

### 1692 · `simpson_episodes`

version37 在同一 joined nominee 粒度上计算 39/59，得到 66.1016949%。version38 在推理中
多次讨论 “nominee 是人还是 award row”，最终改成 distinct person 的 23/41，得到
56.097561%。这正是 prompt 要求避免的 grain 漂移，但模型在明确复述规则后仍主动覆盖它。

### 1152 · `professional_basketball`

version37 输出 Kyrie 的 first/middle/last 三列。version38 找到同一球员后只输出
`playerID=irvinky01`。该数据的 external knowledge 与 benchmark 姓名输出约定长期存在
冲突；“显式映射必须服从”的全局规则会放大这种冲突，不能靠另一条全局姓名规则安全修复。

### 3042 · `hockey`

两版都找到 Martin Brodeur。version37 保留 `firstName`,`lastName` 两个原始字段；
version38 把 “full name” 视为显式格式要求，拼接成单列 `Martin Brodeur`。这与
“没有明确格式时保留源字段”的约束边界发生冲突，属于输出表示回归。

### 2088 · `address`

version37 按 joined ZIP rows 对两个 post-office type 做普通 count，得到 `50-114=-64`。
version38 根据问题中的 “cities” 改成 distinct city，得到 `25-86=-61`。从自然语言实体
角度 distinct 很合理，但与 external/gold 规定的 row count 不一致。全局“题目实体粒度”
规则在这类题上会与 benchmark 映射发生竞争。

## 推理质量判断

version38 有局部改善，但没有整体提升：

- 改善：部分轨迹更早明确输出槽、使用 inner join、对 business 做 distinct entity count。
- 未改善：多条错误轨迹在 reasoning 中正确复述外部映射和 prompt 规则，随后仍用常识覆盖；
  说明限制不是“模型没看见规则”，而是规则在长链决策中权重不足。
- 新问题：当 question、external knowledge 和 benchmark output convention 本身不一致时，
  更强的全局约束只是改变模型偏向，不保证向 gold 收敛。
- 成本：保留 episode 的平均步骤增加 17.4%；受 provider 空载体影响，实际 API tokens
  达到历史同题 version37 的约 2.71 倍。

因此，当前 teacher 的能力上限并没有因为增加一段通用 prompt 而明显上移。它更多改变
错误分布：修复若干显式形态错误，同时引入若干合理解释回归。

## 建议

1. 不扩展 version38，不用这批轨迹构造 SFT。
2. 把 DeepSeek 空 visible content 的请求级恢复作为 client transport policy 单独处理，
   建议为该 provider 使用更高的 carrier-only retry 上限；它仍是 client event，不计入
   agent action 或 SFT target。
3. 下一版 prompt 若继续实验，应压缩成短的优先级合同，不再叠加更多解释性段落；并使用
   同时段 version37/version39 并发 A/B，隔离 provider 波动。
4. prompt gate 应把明显 benchmark 冲突题单列，不用 1152、3042、2088、5161 这类题决定
   通用语义规则。它们更适合做数据标注/外部知识一致性审计。
5. 对 10 个零过程错误的错误答案，继续增加接口反馈收益很低；若外部教师不可训练，剩余
   提升更可能来自更高质量的任务语义/输出契约，或多样采样加 verifier 选择，而不是继续
   加长 system prompt。

## 审计与产物

- 冻结选择：
  `data/eval_inputs/bird_train_version38_semantic_discipline_gate32_20260729.manifest.json`
- 首轮：
  `data/trajectories/bird_train_version38_semantic_discipline_gate32_20260729/verified_success.manifest.json`
- 基础设施恢复：
  `data/trajectories/bird_train_version38_semantic_discipline_gate32_infra_retry1_20260729/verified_success.manifest.json`
- 合并审计：
  `data/trajectories/bird_train_version38_semantic_discipline_gate32_20260729/infrastructure_recovered.manifest.json`

18 条成功轨迹全部通过 `audit_verified_rollouts.py` 的结构门和独立 fresh replay：
18/18 replay pass，判分固定为 `bird-set`。报告只依据题目、外部知识、模型推理、合法工具
调用、环境反馈和 hidden verifier 的通过/失败信号；未查看或暴露 gold SQL/answer rows。
