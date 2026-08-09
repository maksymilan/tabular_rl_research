# BIRD Version54 去 Plan：teacher1500 Prefix200 单臂验收预注册

日期：2026-08-06  
状态：**cohort、协议与绝对验收门已冻结并已获授权；首次启动因官方 API HTTP 402
`Insufficient Balance` 在 0 个模型动作处中止，Prefix200 尚未开始形成有效结果。**

## 目标

从当前冻结训练候选集 `bird_train_atomic_teacher1500_v2_nonempty` 按原顺序取前 200 题，
只运行无 `plan` 的 `version54` / `native-tool-bundle`。本实验不运行 v53，不做版本性能比较；
只判断 v54 是否满足批量生成所需的绝对正确性、可靠性、因果性和审计预期。

若所有验收门通过，在同一协议、同一模型和同一冻结 1500 集合上继续顺序生成剩余 1300 题。
不得根据 Prefix200 结果重采样或改变剩余任务顺序。

## 冻结来源与选择

- harness source：`data/eval_inputs/bird_train_atomic_teacher1500_v2_nonempty.jsonl`
- source SHA-256：`9c4995daf81f94053931fbdaa7bd11beef9157909bacde03be6cb61904952264`
- source records：1500；状态 `frozen_teacher_rollout_candidate`
- task admission：1500/1500 已通过 `gold-denotation-nonempty-task-filter-v1`
- selection：`start=0, limit=200`，原顺序前 200 条，不重采样
- unique examples：200
- unique databases：66
- external knowledge present：186/200
- example-id sequence SHA-256：
  `d3602dae513ed33bdfb5f3d696870ea354b64ac2f1588e6a0bd78928d82c0a57`
- public task identity SHA-256：
  `136dc2b4e0766049d96f4f1daaf9262eec9f39a1fc316edeb3fa1216e95ff830`
- frozen selection manifest：
  `data/eval_inputs/bird_train_atomic_teacher1500_v2_nonempty_prefix200_v54.manifest.json`

教师可见投影与 harness source 的前 200 个 task id/order 完全一致；投影禁用字段为空。私有
sampling profiles 只用于已完成的 cohort 平衡筛选，不发送给 provider，也不参与轨迹推理。

## 协议与运行配置

- protocol：`version54`
- scheme：`native-tool-bundle`
- provider functions：11 个；公开 `plan` 已移除
- provider：官方 `https://api.deepseek.com/chat/completions`，无代理、无 fallback
- model：`deepseek-v4-flash`
- attempts：K=1
- context：`rolling-legal-history` / recent 4 provider turns / `full`
- database context：`catalog-v1`
- denotation：`bird-set`
- max steps：30
- max completion tokens：2,048
- API retries：10
- timeout：300 秒
- max errors per type：3
- table output rows：0
- workers：4（有序、最多一个 worker-sized 窗口在途）
- ordered stop：`max_consecutive_semantic_failures=5`
- diagnostic-only：true

运行入口固定为：

```text
src/sft/generate_teacher_rollouts.py
--examples-file data/eval_inputs/bird_train_atomic_teacher1500_v2_nonempty.jsonl
--start 0 --limit 200
--atomic-protocol-version version54
--deepseek-carrier native-tool-bundle
--model deepseek-v4-flash
--attempts-per-example 1
--workers 4 --max-consecutive-semantic-failures 5
--context-mode rolling-legal-history --history-turns 4
--rolling-prompt-variant full --plan-policy optional
--database-context-profile catalog-v1
--denotation-comparison bird-set --diagnostic-only
```

## “符合预期”的绝对验收门

不使用对照臂。只有以下全部满足才继续剩余 1300 题：

1. **有效数据产率**：至少 140/200（70%）terminal-correct，并且这些成功全部成为
   verifier-correct 候选轨迹；
2. **合法终止**：至少 196/200（98%）形成合法 semantic completion；provider/API failure
   不超过 4 题；
3. **过程稳定性**：结构化 process errors 总数不超过 60；如果按冻结任务顺序出现五个连续
   semantic failures，立即停止提交新窗口并判定不扩展。并发实现只允许一个最多 4 题的有序
   窗口在途；触发点之后已经发出的同窗口请求仅保留审计，明确标记为 gate 外，不能用于通过
   Prefix200 验收；
4. **去 plan 生效**：官方请求 schema 的函数数严格为 11 且不含 `plan`；provider history、
   parsed calls、成功 primitive steps 中 `plan=0`；
5. **因果与结构**：全部正确轨迹 fresh replay 100% 通过；结构、grounding、provider-history、
   call/result 顺序、argument lowering、bundle-pre-state、错误状态不变和 causal-prefix 审计均
   为零 issue；
6. **隐私**：所有 model inputs 中 gold SQL、gold rows/sample/count、私有非空状态、私有
   sampling profile 和隐藏 verifier 数据泄漏均为 0；
7. **空结果训练边界**：空 intermediate 只保留为 context、不得成为正目标；terminal-empty
   轨迹不得进入 retained candidate；1x1 scalar zero 仍按非空处理；
8. **产物完整性**：manifest 必须绑定 source/cohort、teacher/student prompt、11-tool schema、
   protocol、carrier、provider controls、metric 和所有输出哈希。

工具频率、multi-call 比例、错误恢复率、turns/actions/tokens、P50/P95 和 wall time全部报告，
但不设置相对性能门，也不据此声称 v54 优于 v53。

## 条件扩展

全部 Prefix200 门通过后，才允许按 `start=200, limit=1300` 运行同一冻结来源。扩展阶段不得
改变 prompt、schema、model、carrier、history、metric 或预算；若需改变，必须建立新版本和新
pilot。扩展产物与 Prefix200 分目录保存，审计通过后才能形成一个统一候选索引。

当前 `native-tool-bundle` 仍缺少已晋级的 scheme-aware SFT exporter，因此即使轨迹本地正确，
也只能标记为 diagnostic candidate；本实验不授权把 multi-call turns 展平、直接合并进现有
atomic SFT 或启动训练。

## 外部数据边界

获得本实验的新明确授权后，允许发送的只有：自然语言问题、external knowledge、数据库
catalog/schema、当前合法 causal prefix 和逐步只读工具反馈。不得发送 gold SQL、gold rows、
gold sample、gold row count、私有非空状态、私有 sampling profile 或隐藏验证器数据。

可以把授权范围一次性覆盖 Prefix200，以及仅在 Prefix200 全部门通过后自动继续的剩余 1300
题；授权前不发出任何外部请求。

## 首次基础设施启动记录

用户已明确授权后，2026-08-06 按冻结配置启动 Prefix200。第一个有序 4 题窗口全部在首轮模型
动作之前收到官方 `https://api.deepseek.com/chat/completions` 的 HTTP 402
`Insufficient Balance`；4 条已持久化记录均为 `steps=0`、`errors=0`、`failure_type=api_error`、
token usage 为空。检测到这组一致的基础设施失败后人工中断；第二个最多 4 题的已在途窗口也被
中断，未形成 provider model response 或可评分轨迹。此次启动不是 Prefix200 语义结果，不能
计入验收或扩展门。

诊断同时暴露出 transport client 会对确定性的 402 使用完整重试预算。客户端现已改为只重试
无状态码网络错误、408/409/425/429 和 5xx；402 等确定性 4xx 首次返回即失败。对应回归测试
覆盖 402 不重试与 429 保留重试。有效 rerun 必须使用新隔离结果目录，并在官方余额恢复后从
冻结 source position 0 重新开始；不得把本次基础设施失败与有效 Prefix200 合并。

## 余额恢复后的有效 rerun 结果

2026-08-06 先用不含任务数据的最小 Chat Completions 请求检查官方服务，
`https://api.deepseek.com/chat/completions` 返回 HTTP 200。随后从冻结 source position 0 在新目录
`data/trajectories/version54_no_plan_teacher1500_20260806/prefix200_r2/` 启动有效 rerun；未读取或
合并首次 402 目录中的记录。

有序 acceptance gate 处理到第 14 题时，第 10--14 题形成五个连续 semantic failures，触发冻结
停止门。调度器停止提交新窗口；同窗口已经在途的第 15--16 题完成后被标记为
`inflight_after_semantic_stop=true`、`acceptance_gate_in_scope=false`，不得用于通过 Prefix200 门。

- gate 内：14 题，4 correct、13 legal、10 failed；正确率 28.57%，合法率 92.86%；
- gate 内失败：9 `wrong_answer`、1 `execution_error`；9 个结构化错误事件，其中
  1 `protocol_error`、7 `execution_error`、1 `argument_validation_error`；
- gate 内消耗：95 个真实 model turns、114 个 primitive calls、99 次 API request attempts、
  652,450 tokens；
- 全部落盘诊断记录：16 题，5 correct、15 legal、11 failed；其中 2 题明确为 stop 后在途记录；
- 全部落盘消耗：112 次 API request attempts、743,503 tokens；transport retry 为 0，证明本次
  失败不是余额或网络重试造成；
- static provider schema 为 11 个函数且不含 `plan`；manifest 的 `disabled_model_tools=["plan"]`；
- native bundle 审计覆盖 16 条记录、129 个 provider calls，issues=0、gate=pass；
- 正确候选结构审计覆盖 5 条记录，5/5 verified，structural issues=0、gate=pass；空 intermediate
  正目标过滤后保留 37/38 legal step targets，terminal-empty=0。

主要产物 SHA-256：

- `verified.jsonl`：`026e45f03b41c128ed7a9a0ff884a7a0ee78106ee541e16e2d25d9123f41fa2e`
- `verified.failures.jsonl`：`054389dccb34f0338719c837fe3f8cbbfddb62fcf2843bce1b22b92dff1d2a9e`
- `verified.all.jsonl`：`8ed8c17a4f76b19acb76bccea13c0d2c7dd9bfe490eec3f00d8c37f587c0e1bb`
- `verified.manifest.json`：`2919159424423f4398d7fdf960adb5df280cfbe372e90f851c79d16266412f73`
- `native_bundle_audit.json`：`beeb789c36818962478b961ef087c98a4531dcd2e12c7ed9182fcb510ff8f4a5`
- `structural_audit.json`：`3a671ce75b3f1e49e2457931a4481397234eccbbf2e2116ee1a06e6df848c940`

结论：本次是有效 provider rerun，但 Prefix200 的绝对验收门已失败。不得继续剩余 186 个
Prefix200 episode，也不得启动剩余 1300 题；这些记录继续保持 diagnostic-only，不能进入 SFT。
