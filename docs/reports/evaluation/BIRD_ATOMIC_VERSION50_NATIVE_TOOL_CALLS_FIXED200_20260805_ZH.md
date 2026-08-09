# BIRD atomic version50 DeepSeek native Tool Calls fixed-200

日期：2026-08-05  
状态：**fixed-200 已完成，version50 已拒绝；diagnostic-only，不得进入 SFT/RL**

## 结论

DeepSeek 官方 native `tools`/`tool_calls` 可以驱动本项目现有 12 个原子工具完成真实
model↔harness 闭环，但在冻结 200 题上**没有比历史 version24 更好**：

| 指标 | version50 native tools | 历史 version24 | 差值 |
|---|---:|---:|---:|
| correct | **133/200 (66.5%)** | **145/200 (72.5%)** | -12 / -6.0 pp |
| legal | **183/200 (91.5%)** | **197/200 (98.5%)** | -14 / -7.0 pp |
| process errors | **166** | **29** | +137 / 5.72x |
| model actions | **1,487** | **1,490** | -3 |
| total tokens | **16,158,278** | **8,420,861** | +7,737,417 / +91.9% |

逐题配对为 120 题共同正确、13 个 version50-only gain、25 个 version24-only
regression、42 题共同错误；discordant 38 题的 exact 双侧检验 `p=0.0729514`。Legal 配对
为 2 个 gain、16 个 regression，exact 双侧 `p=0.0013123`。原生 carrier 不仅准确率下降，
合法终止也显著恶化，因此不得替换默认 JSON-Output carrier。

## 实验问题与边界

Version50 保留 version39 的 public tools、参数、执行、canonical resident state、recent-4、
teacher semantic guidance、exact-table terminal 和 `bird-set`，只把 DeepSeek-facing carrier
改为官方 native function calls。Provider 返回的 function name 与 arguments JSON 不做参数修复，
先降为 canonical `<think> + raw JSON`，再走同一 parser、状态校验、executor、scorer 和 replay。
模型从不直接执行工具，数据库和状态仍由 harness 独占。

主要参照是同一 200 题、同模型、K=1、`bird-set` 的历史 atomic version24。由于 version50
继承当前 version39 而非历史 version24，本比较回答“当前 version39 语义 + native carrier
整体是否超过历史 fixed-200”，不是严格的 carrier 单变量因果效应。负结果足以拒绝晋级；
无需再运行 fresh version39 control 来主张不存在的增益。

## 冻结 cohort 与配置

- 输入：`data/eval_inputs/bird_train_tool_interface_validation200_version4.jsonl`
- 文件 SHA-256：`6b469215ac96e09c1046fee539ff4c4885b58e9b7a0b6fdb2187082267dcde36`
- 200 个 task-id 顺序 SHA-256：
  `a163f201d5dfd6ab56f07e75c74436f9eb9a7fe82a76533d136bd754fc7118c0`
- provider：官方 `https://api.deepseek.com`，无 fallback
- model：`deepseek-v4-flash`
- protocol/carrier：`version50` / `deepseek-native-function-call-v1`
- canonical stored carrier：`think-json-v1`
- thinking：enabled；reasoning effort：high；temperature：0
- `tool_choice="auto"`；本地强制每个 atomic turn 恰好一个调用
- context：rolling legal recent-4；full prompt；plan optional；catalog-v1
- max actions / completion tokens：30 / 2,048
- attempts per task：1；max errors per type：3；provider request attempts：最多 10
- denotation：`bird-set`
- `diagnostic_only=true`、`sft_export_eligible=false`

Manifest 身份：

- protocol hash：`f0dcfad6f89ba2e1`
- canonical teacher prompt：
  `677135c9a6b025e4090376051af2c5297e493d88ec4f683f4b80529478fb3571`
- provider prompt：
  `4c30487257e3e4eab81e0bab13c4e91c7ff102a24f58d62d094fb17ac25c2c24`
- student prompt：
  `81329794b41050dd62526e870bda6c5df4d68e095c2576c2a75aa3758131bd54`
- public tool schema：
  `4d9e6cae7652ba958f316177367f18c8b68a48c4cc7e447e1b11eb3d13487125`
- native tools schema：
  `ece0de3dbcbc7b6017abdf49a733468bf3ca1d51d8582513b9440ca78f6cdadd`

## 扩展门覆盖与基础设施审计

预注册要求 Pilot20 correct≥15、legal≥19、过程错误≤4 且无未解决 provider failure。早期
Pilot 暴露了两项客户端边界缺陷：多调用曾被误作同轮 transport retry；原生 carrier 错误又
因文本含 `arguments` 被误归类为参数错误。这两项均只修改错误归属，不执行或修复模型调用；
旧 Pilot 分目录冻结，不用于结果。一次沙箱网络拒绝批次同样作废。

修正后的 Pilot r4 在完成 19 题时为 14 correct、19 legal，且过程错误已超过上限。用户随后
明确要求取消 Pilot、覆盖扩展门并直接运行 200 题。本文 fixed-200 因此是用户授权的
diagnostic scale-up，不是门禁通过或协议晋级。

最终边界是：零调用、缺失 call id、空 reasoning 等 provider-shape 缺陷进行有界客户端重试；
多调用、非空 assistant content、未知函数或损坏 arguments 是模型已创作的 semantic action，
作为 state-preserving `protocol_error` 反馈给下一轮，不执行其中任何工具。长度截断在完成
预算耗尽后也成为可见协议错误。

## 完整结果

### 结果与可靠性

- correct：133；其中 clean success 86，recovered success 47
- legal：183
- failure：50 wrong answer，17 protocol error
- actions：1,487；历史 version24 为 1,490
- wall time：1,839.026 秒；历史两个 version24 分片合计 1,087.953 秒

过程错误分布：

| 类型 | version50 | version24 |
|---|---:|---:|
| protocol error | 124 | 3 |
| argument validation error | 41 | 9 |
| execution error | 1 | 17 |
| 合计 | **166** | **29** |

124 个 version50 protocol error 与 provider 审计完全对齐：

- 71 次一次返回 2 个函数调用
- 4 次一次返回 3 个函数调用
- 44 次 native call 同时带非空 assistant content
- 5 次长度预算耗尽后仍没有可执行调用

17 个 protocol-error 终止中，有 11 题是 version24 正确题、6 题是 version24 错误题。
25 个 paired regressions 中 11 个直接终止于 protocol error，另 14 个合法终止但答案错误。

### Token 与 API

| 指标 | version50 | version24 |
|---|---:|---:|
| prompt tokens | 15,565,362 | 8,003,435 |
| completion tokens | 592,916 | 417,426 |
| reasoning tokens | 455,378 | 358,117 |
| total tokens | **16,158,278** | **8,420,861** |
| cached prompt tokens | 13,301,376 | 5,943,040 |
| prompt cache miss tokens | 2,263,986 | 2,060,395 |
| API request attempts | 1,530 | 1,509 |
| completion retries | 42 | 8 |
| transport retries | 0 | 11 |
| context retries | 0 | 0 |
| native carrier retries | 1 | n/a |

动作数几乎不变但 token 近乎翻倍。该差值是 version50 整体 package 对历史 version24 的
比较，不能全部归因于 native schema；但它明确否定了“当前方案更省 token”。

### 正确轨迹中的工具分布

Manifest 的 `tool_hist` 只统计进入 verified JSONL 的正确轨迹：

| tool | version50 | version24 |
|---|---:|---:|
| condition_filter | 189 | 248 |
| describe_table | 140 | 160 |
| answer_from_context | 133 | 145 |
| read_subtable | 104 | 109 |
| group_aggregate | 71 | 89 |
| join_tables | 66 | 69 |
| project | 51 | 51 |
| inspect_column | 28 | 46 |
| extreme_value_select | 22 | 24 |
| scalar_compute | 17 | 16 |
| plan | 5 | 11 |
| set_op | 3 | 1 |

## 逐题配对

Version50-only gains（13）：`bird_train_00582`、`bird_train_00593`、`bird_train_02088`、
`bird_train_02512`、`bird_train_02868`、`bird_train_03636`、`bird_train_03688`、
`bird_train_03969`、`bird_train_03977`、`bird_train_04869`、`bird_train_05316`、
`bird_train_05544`、`bird_train_06324`。

Version24-only regressions（25）：`bird_train_00362`、`bird_train_00454`、
`bird_train_00541`、`bird_train_00600`、`bird_train_00858`、`bird_train_01002`、
`bird_train_01018`、`bird_train_01148`、`bird_train_01535`、`bird_train_01730`、
`bird_train_01787`、`bird_train_01926`、`bird_train_02682`、`bird_train_02901`、
`bird_train_02935`、`bird_train_03131`、`bird_train_03216`、`bird_train_03312`、
`bird_train_03505`、`bird_train_03925`、`bird_train_04244`、`bird_train_05558`、
`bird_train_05760`、`bird_train_06454`、`bird_train_06489`。

## Replay、结构与 no-leak 审计

- fresh deterministic `bird-set` replay：**133/133 pass**
- `audit_verified_rollouts.py`：133 verified、0 structural issues、full-prompt gate pass
- 正确轨迹：829 legal step targets、57 feedback-recovery targets
- 所有 200 题的 provider model inputs：1,487 turns，gold SQL 可见泄漏 **0**
- 已解析并执行的 native calls：1,359；function name/arguments 与 canonical action mismatch **0**
- 被拒绝的 protocol actions：state mutation **0**
- provider fallback、transport failure、context failure：**0 unresolved**

最终 artifacts：

- `data/trajectories/version50_native_tool_calls_20260805/fixed200_user_override/all.jsonl`
  SHA-256 `aa34cf316125a03314b6d985135d7b16e750fbe7cfd1f21408860fa36a0abbbe`
- `.../verified.jsonl`
  SHA-256 `0f33d391ef12fddb7f58f4113fabb2167f9ff310219eda2bf2df7796498bc93f`
- `.../verified.manifest.json`
  SHA-256 `c12509e77fdc872d1024b1875864914bde75d89e4683caac2fddf8ba23b99f5a`
- `.../structural_audit.json`
  SHA-256 `2f02f7524d402813ad3305370ea577ac4c07fd320572bfd81e1c6bacd7c4e80a`

执行代码快照 SHA-256：

- `generate_teacher_rollouts.py`：
  `4948665057854d8bedd9f3ba328bbdaf4c9455149b77847f9e3215408782ff8a`
- `provider_adapter.py`：
  `a2df687148f858257c69dceeb81d61e22874d7c72688a0afc4b7d694379b93d7`
- `deepseek_native_tools.py`：
  `4cc541526eb6e2e544dd4bd71606767c8ca127ea682222312ccdc7b7e2473cfb`
- `atomic_version50.py`：
  `50746281ac57829c1c337428a80db902dd6bd4e774150b709161b55e9e685700`

## 决策

1. Version50 不晋级，不替换 JSON Output 默认 carrier。
2. 133 条正确轨迹即使 replay/结构/no-leak 全部通过，也不得进入 SFT/RL。
3. 不对 fixed-200 错题做 K>1 verifier selection，也不混入冻结 version26 结果目录。
4. 若继续研究 native tools，必须登记 version51 或更高的独立诊断，先解决多调用与
   assistant-content 协议摩擦，再从小型稳定门开始；不得把本结果解释为 carrier 单变量收益。
