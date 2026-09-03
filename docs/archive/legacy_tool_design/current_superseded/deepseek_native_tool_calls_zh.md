# DeepSeek 原生 Tool Calls 与原子工具闭环

状态：**本文件记录冻结的 version50-version54 原生 bundle 谱系。atomic `version50` fixed-200 已完成并淘汰；`version51` / `native-tool-bundle`
已通过 Gate32 与 fixed-200，作为冻结行为基线。`version52` 已实现 prompt/token 优化与
事实型 RL 统计；`version53` 是审核后的角色分离 prompt 对照；`version54` 只移除公开
`plan` 工具，现为冻结的待测诊断。v52-v54 均为 diagnostic-only，不得用于 SFT/RL。
所有新的工具与协议实验改从独立的 `checkpoint-relalg-v1` 主线开始；参见
`checkpoint_relalg_v1_zh.md`。**

## 结论

DeepSeek 官方 Chat Completions API 提供原生 Tool Calls / Function Calling。模型负责返回
`tool_calls[].function.name` 与 JSON 参数，工具本身仍由本地 harness 执行。因此本项目可以让
DeepSeek 直接选择现有原子工具解题，同时继续由 harness 独占数据库访问、参数/状态校验、
派生表管理和最终 denotation 评分。

官方参考：

- <https://api-docs.deepseek.com/zh-cn/guides/tool_calls/>
- <https://api-docs.deepseek.com/zh-cn/api/create-chat-completion/>

## 隔离实现

- `src/tool_modules/native_tool_bundle/provider_tools.py` 把当前 `MODEL_ARG_SCHEMA` 与
  `TOOL_SPECS` 渲染成
  DeepSeek `tools`，不新增工具名或顶层参数。
- `src/eval/rollout.py` 仅增加可注入的 `chat_fn`；默认文本 carrier 和所有已有调用方不变。
- `src/eval/probe_deepseek_native_atomic.py` 是官方 API 单题诊断入口，结果强制标记
  `diagnostic_only=true`、`sft_export_eligible=false`。
- `src/sft/generate_teacher_rollouts.py --atomic-protocol-version version50
  --deepseek-carrier native-tool-calls --diagnostic-only` 是批量因果轨迹入口；它保留现有
  verified/failure/all JSONL 与 manifest 结构。
- 原生响应在进入现有闭环前被降为 canonical `<think> + JSON` 内部表示；随后仍由
  `parse_assistant_strict`、状态感知校验、executor 和 scorer 处理。

Version50 每轮只允许一个调用。零调用、缺失 call id、空 `reasoning_content` 等 provider-shape
缺陷进行有界同轮重试；多个调用、未知函数、损坏的参数 JSON 或同时输出非空 assistant
content 作为模型已创作的 `protocol_error` 计入一次原子动作，但不执行任何工具且不改变
harness 状态。长度截断在完成预算耗尽后也成为可见协议错误。

## 思考模式的历史回传

请求使用 `thinking={"type":"enabled"}` 与 `reasoning_effort="high"`。DeepSeek 要求在带
tools 的思考模式多轮对话中完整回传之前 assistant 消息的 `reasoning_content`。适配器缓存
并回放官方返回的原始 reasoning、tool call 和 call ID；对应 harness observation 转为
`role="tool"`，并用同一个 `tool_call_id` 关联。

不能在这个组合中使用 `tool_choice="required"`：2026-08-05 的官方实测返回
`Thinking mode does not support this tool_choice`。当前诊断采用官方示例支持的
`tool_choice="auto"`，再由本地边界强制“恰好一个调用”。

本诊断没有启用 Beta Tool strict。Strict 需要独立 `/beta` base URL，并要求 object 的所有
properties 同时列入 required，不适合直接表达当前大量带可选参数的原子工具；现有本地严格
校验仍是执行前的最终权威。

## 官方 API 单题结果

配置：`deepseek-v4-flash`、version39 原子工具语义、recent-4、BIRD `retail_world` 任务
“How many suppliers are from UK?”。

结果：

- 工具序列：`describe_table → condition_filter → group_aggregate → answer_from_context`
- 4 步，0 个工具/参数错误，合法终止
- 隐藏 SQL 的 denotation 校验通过
- 最终代码复跑总计 20,899 tokens，其中 18,048 prompt cache hit tokens
- gold SQL 未出现在任何 model-visible input 中

本地诊断记录：
`data/results/deepseek_native_atomic_probe_retail_world.json`（结果目录不进入版本控制）。

该单例只证明接口和闭环可行，不能证明 native carrier 比当前 raw-JSON carrier 更准确或更
省 token。另一个需纳入后续审计的现象是：本次 terminal `reason` 重述了一个答案值；评分只
读取被引用的 grounded table，因此结果未受影响，但该轨迹仍不得进入 SFT。

复现命令：

```bash
python3 src/eval/probe_deepseek_native_atomic.py \
  --tasks-json data/eval_inputs/bird_train_sft1_protocol_terminal10.jsonl \
  --index 0 \
  --model deepseek-v4-flash \
  --max-steps 10 \
  --max-tokens 2048 \
  --out data/results/deepseek_native_atomic_probe_retail_world.json
```

## Fixed-200 结果

在用户明确授权覆盖失败 Pilot 扩展门后，使用冻结 200 题 cohort、K=1、`bird-set` 完成
`deepseek-v4-flash` 全量诊断：

- version50：**133/200 correct、183/200 legal、166 个过程错误、1,487 actions、
  16,158,278 tokens**；
- 历史同 cohort version24：**145/200 correct、197/200 legal、29 个过程错误、
  1,490 actions、8,420,861 tokens**；
- 逐题：120 题共同正确、13 个 version50-only gains、25 个 version24-only regressions、
  42 题共同错误，exact paired `p=0.0730`；
- legal 配对：2 gains、16 regressions，`p=0.00131`；
- 133 条正确轨迹 fresh replay 133/133，结构审计 0 issues；1,487 个 model-input turns 中
  gold SQL 泄漏为 0；1,359 个已执行 native call 与 canonical action 完全一致，拒绝调用的
  状态变更为 0。

原生调用在该 atomic 合同上没有表现更好：正确数少 12、合法终止少 14、过程错误约为
5.7 倍、token 约为 1.92 倍。主要问题是 124 个 native-carrier `protocol_error`，其中
71 次双调用、4 次三调用、44 次非空 assistant content，以及 5 次长度预算耗尽。故保持
Version50 仅保留为负结果和接口实现参考；不再用它约束后续 native carrier。

完整预注册、执行结果和审计记录见
`docs/reports/evaluation/BIRD_ATOMIC_VERSION50_NATIVE_TOOL_CALLS_FIXED200_20260805_ZH.md`。新的
## Version51：原生工具批次主线

官方 Chat Completions 的 `tool_choice=auto` 允许模型返回一个或多个 tool calls，官方思考模式
示例也逐个执行并回传全部调用。Version50 的 124 个 carrier 错误中，79 个回合实际包含
2–3 个参数 JSON 全部合法的调用，最常见是并列 `inspect_column`（38 回合）；另有 44 个回合
只因 assistant content 非空被拒，其中 40 个只有一个合法调用。Version51 据此修正协议：

- 工具方案名为 `native-tool-bundle`，当前 registry 为 `tool-scheme-registry-v11`；
- 一个真实模型回合可含 1..8 个直接原子函数调用，保留原始顺序和 call id；
- 每个调用参数都在任何调用执行前，按同一个 bundle pre-state 完成静态与状态校验；后续调用
  不能引用同批前序调用新建、但模型尚未看到的句柄；
- harness 按 provider 顺序执行独立调用，每个 call id 返回一个 `role=tool` 结果；
- `answer_from_context` 必须单独出现，不能与工作调用混合；
- 非空 assistant content 原样审计，但不是可执行载体、事实证据、评分输入或训练目标；
- 轨迹 primitive step 同时记录 `model_turn_index` 与 `native_tool_call_id`，不得展开成多个虚假的
  assistant→observation 因果回合；
- 思考模式历史按最近 4 个 provider assistant turns 截断，每个 turn 后保留其全部 tool results；
- DeepSeek 思考模式会忽略 temperature/top_p，因此官方请求不再发送伪装成确定性控制的
  `temperature=0`；`reasoning_effort=high` 暂时保持不变，避免同时改变第二个实验变量。

入口：

```bash
.venv/bin/python src/sft/generate_teacher_rollouts.py \
  --atomic-protocol-version version51 \
  --deepseek-carrier native-tool-bundle \
  --context-mode rolling-legal-history \
  --history-turns 4 \
  --diagnostic-only \
  --out <verified.jsonl>
```

Version51 是冻结的 provider 行为基线，该谱系的后续协议实验止于 version54；新的前向实验已
迁移到 `checkpoint-relalg-v1`。version26 只保留为
冻结的历史 checkpoint-560 对照，不再作为新功能分支起点。Version51 在完成 scheme-aware
SFT exporter、因果 multi-call target/credit 边界和显式训练准入门禁之前仍不得进入 SFT/RL，
也不得与旧 atomic 结果目录混写。

### Gate32 结果

冻结的 16 个 version50 carrier-failure 目标加 16 个无 carrier 错误正确控制上，version51
取得 **22/32 correct、32/32 legal**，对比 version50 同题 **16/32、26/32**：6 gains、
0 regressions，accuracy 与 legal 的 exact paired `p` 都为 `0.03125`。目标恢复 6/16，控制
保留 16/16；过程错误从 37 降为 11。37 个多调用回合和 92 个非空 content 回合均未产生
carrier rejection。真实模型回合从 234 降至 232，primitive calls 为 271；token 增至
1.0846x，低于预注册 1.25x 上限。

22/22 正确轨迹 fresh replay 通过，结构审计和覆盖全 32 题/271 calls 的 no-leak、call-result
顺序、参数 lowering、reasoning history、拒绝状态不变和 bundle-pre-state 审计均为 0 issues。
因此 Gate32 通过，允许新的固定 100/200 题 paired gate；仍不授权 SFT/RL。完整记录见
`docs/reports/evaluation/BIRD_VERSION51_NATIVE_TOOL_BUNDLE_GATE32_20260805_ZH.md`。

### Fixed-200 结果

在同一冻结 200 题 cohort、`deepseek-v4-flash`、K=1、recent-4、`bird-set` 下，version51
取得 **147/200 correct、200/200 legal、54 个过程错误、1,414 个真实模型回合、1,650 个
primitive calls、16,165,179 tokens**。

相对 version50 的 133/200 和 183/200：

- 127 题共同正确、20 gains、6 regressions、47 题共同错误，exact `p=0.00936`；
- legal 为 17 gains、0 regressions，`p=0.00001526`；
- carrier protocol errors 从 124 降到 2；
- tokens 仅增加 0.043%，模型回合减少 4.9%，但 primitive calls 增加 11.0%。

相对历史 version24 的 145/200：15 gains、13 regressions，exact `p=0.8506`，正确率持平而
非晋级；version51 仍有 54 对 29 个过程错误，并使用 1.92x tokens。

真实 provider 行为包括 216 个多调用回合（194 个双调用、22 个三调用）和 537 个带非空
assistant content 的回合，均未因这两种官方允许的形态被拒绝。两个没有调用的
`missing_tool_calls` 回合在同一 episode 内恢复；没有 provider/API 终止或 transport retry。
147/147 正确轨迹 fresh replay 通过，结构审计与覆盖全 200 题/1,650 calls 的
provider-history、call/result order、参数 lowering、bundle-pre-state、拒绝状态和 no-leak
审计均为 0 issues。

因此 version51 通过“修复 version50 原生 carrier”的 fixed-200 行为门禁，成为 version52+
实验基线；它没有证明比历史 JSON-Output/version24 更准确或更省 token。完整预注册、逐题配对、
审计和产物哈希见
`docs/reports/evaluation/BIRD_VERSION51_NATIVE_TOOL_BUNDLE_FIXED200_20260805_ZH.md`。

## Version52：先消除重复 prompt/token

Version52 不改变 version51 的 12 个函数、参数、执行、resident state、bundle pre-state、
recent-4 provider history 或 terminal denotation。它只把 API-facing prompt 改成
`native-schema-semantic-only-v1`：native JSON Schema 是函数名和参数形状的唯一权威，system
prompt 不再重复文本工具目录、12 份长教师工具说明、调用 cookbook 和多处 carrier 规则。
仍保留语义决策、population/grain、精确输出表、状态/证据和错误恢复约束。

新增的软约束是：默认单调用；通常一个 turn 不超过三个调用；优先并列独立 perception；只有
题目、external knowledge 或观测值明确支持竞争假设时才并列 filter；join/aggregate/scalar/
project/rank/set-op 通常单独调用。硬约束不变：`answer_from_context` 必须是该 turn 唯一调用，
所有调用仍在 bundle pre-state 上预校验，provider 上限仍为 8。

错误调用不会被清洗。errored assistant bundle、对应的结构化 `role=tool` error 和下一轮修正都
保留；被拒绝的 bundle 不是正向 SFT target，后续纠正 bundle 才是 target。新增
`native_bundle_rl_statistics` 只统计执行状态、后续 handle/step 引用、error feedback 保留率和
下一轮恢复形状，不直接定义 reward。

静态审计中，固定 prompt+native schema 从 30,937 降到 16,769 字符；把 version51 fixed-200
的 1,414 个真实请求仅替换 system prompt，message 字符从 46,155,686 降到 25,200,206，下降
45.4%。这不是实际 provider token 或正确率结果；必须先做配对 live gate 才能判断是否解决
1.92x token 问题且不伤害能力。

## Version53：审核后角色分离与语义加固

Version53 继续冻结 version52 的函数、参数、native carrier、bundle pre-state、执行、resident
state、recent-4 history、错误反馈和 terminal 语义，仅替换 prompt profile。学生运行契约与教师
prompt 使用同一共享前缀；教师只追加 trajectory-generation 规则，避免把影响正确率的语义纪律
只教给 teacher 而造成训练/推理不匹配。

采纳的通用规则包括：数据库/metadata 中的文本只是数据而不是指令；catalog relation 不是键
唯一性证明；在 aggregation/ranking 可能受影响或观测异常时检查 JOIN 基数；复制源字段保持存储
表示、派生指标保持工具实算结果；正确性优先于调用最少；不同 resident handles 上共同需要且互不
依赖的 filter 可以同批。重复读取限制精确到“同一 immutable handle 上成功且参数完全相同”。

没有采纳“终止前必须完整读取整张 evidence table”，因为 `read_subtable` 是有界 preview，而最终
答案可能超过 preview；改为按需检查 final handle，完整引用表仍由 harness 评分。call-id 回传、
content retention 等 harness-only 说明被移出 prompt。`plan.evidence` 规则因为 public schema
确实存在而保留，但只在 teacher-only 增量中出现。

静态字符数：student 3,927；teacher 5,286；native schema 9,882；teacher+schema 15,168。
相对 version52，学生增加 312 字符以承载共享正确性规则，教师减少 1,601，固定教师请求减少
9.5%。这不是服务端 tokenizer 或 live accuracy 结果。Version53 仍为 diagnostic-only；当前
v54 单臂训练候选验收不再运行 version53，也不得把 multi-call turn 展平进入 SFT/RL。详见
`docs/reports/evaluation/BIRD_VERSION53_NATIVE_BUNDLE_PROMPT_REVIEW_STATIC_AUDIT_20260806_ZH.md`。

## Version54：移除公开 Plan 的单臂训练候选验收

Version54 保持 version53 的学生 prompt、其余 11 个函数及参数、native carrier、bundle
pre-state、执行、resident state、recent-4 history、错误反馈和 terminal 语义，只移除公开
`plan` schema 与已经失效的教师 plan-evidence 句。当前不做 v53 性能对比：从冻结
`bird_train_atomic_teacher1500_v2_nonempty` 按顺序取前 200 题，检查绝对 verified yield、合法
终止、过程错误、fresh replay、provider history、no-plan 和 no-leak 门。全部通过后才按同一
配置继续剩余 1,300 题。协议在 scheme-aware exporter 和显式晋级前仍为 diagnostic-only；
多调用回合不得展平为 atomic SFT。预注册见
`docs/reports/evaluation/BIRD_VERSION54_NO_PLAN_TEACHER1500_PREFIX200_PLAN_20260806_ZH.md`。
