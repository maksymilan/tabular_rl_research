# checkpoint-relalg-v1 Flash Text-JSON Hybrid Prefix100

日期：2026-08-09  
状态：**完成；diagnostic-only，不具有 SFT/RL 准入**

## 1. 目的与冻结配置

本次运行使用真实模型↔Harness 因果循环，为冻结的 1,500 题 teacher-rollout 候选集前
100 题生成 checkpoint-relalg 轨迹。模型每轮只能看到合法前缀、当前 Harness 状态和上一轮
反馈；gold SQL/结果仅由本地隐藏 evaluator 使用。

- protocol/scheme：`checkpoint-relalg-v1` / `checkpoint-relalg`
- mode：`hybrid`
- carrier：A，`text-json`
- provider/model：官方 DeepSeek / `deepseek-v4-flash`
- source SHA256：`9c4995daf81f94053931fbdaa7bd11beef9157909bacde03be6cb61904952264`
- source manifest SHA256：`4c6b96e523d02b45c01eaa60df627abac4a6c3720a89f6573f3eec92740537fe`
- slice：positions `0..99`，无重采样
- selection identity：`708b7ad671ecdfbae4eed8441a16577f3a699581d8db1831219c06b5c882298c`
- implementation commit：`3f922cd`
- runner：单 worker；上限为 1,200 attempts、15,000,000 provider tokens、7,200 秒、
  连续 5 个语义失败，以及 provider 连续 2 次/累计 3 次失败

结果目录：
`data/results/checkpoint_relalg_v1_flash_text_json_hybrid_teacher1500_v2_nonempty_prefix100_20260809/`

## 2. 结果

| 指标 | 结果 |
|---|---:|
| 完整 episode | 100/100 |
| `bird-set` 正确 | 63/100 |
| strict artifact 正确 | 42/100 |
| schema match | 56/100 |
| 合法终止 | 99/100 |
| wrong answer | 36 |
| provider failure | 1 |
| model turns | 562 |
| provider attempts | 581 |
| 工具错误 | 15 |
| 累计 episode elapsed | 3,794.835 秒 |
| 最长 episode | 608.982 秒 |

唯一 provider failure 是某一 episode 在 4,096→8,192 的有界 completion 扩容后仍以
`finish_reason=length` 截断；它没有被伪装成语义答案。581/581 个响应/attempt identity 均为
`deepseek-v4-flash`。

`bird-set` 的 63 条成功中，42 条同时满足严格 artifact 与 schema；其余 21 条值集合兼容，
但输出 schema 不精确。因此，若未来建立 scheme-aware exporter，当前最多只有这 42 条可以
进入第一轮高质量候选，不能把 63 条全部当成精确监督。

## 3. Token 与载体表现

| token 字段 | 数量 |
|---|---:|
| prompt | 11,301,306 |
| prompt cache hit | 10,683,904 |
| prompt cache miss | 617,402 |
| completion | 375,893 |
| reasoning | 354,006 |
| total | 11,677,199 |

prompt cache hit 占 prompt tokens 的约 94.5%。载体 attempt 中有 561 次正常响应、18 次
`completion_length` 和 2 次 `empty_text`；除上述最终截断外，其余均由有界同 turn retry
恢复。没有模型身份漂移、第三方 endpoint 或 native multi-call 问题。

## 4. 工具行为

Hybrid 实际主要退化为 Direct SQL 路径：171 次 `execute_sql`、37 次 atomic operator；按
phase 统计为 81 个 `sql_only`、16 个 `sql_to_atomic`、3 个 `mixed`。感知调用主要是
108 次 `describe_table` 与 125 次 `read_rows`。

100 个 episode 中 `commit_checkpoint` 与 `restore_checkpoint` 均为 0。此次数据可以评估
Hybrid/Text-JSON 的因果执行与输出质量，但不能证明 checkpoint 策略已被模型学会，也不能
提供 checkpoint/restore 的正监督覆盖。

## 5. 审计与准入结论

以下门全部通过且 issue count 为 0：

- 100/100 结构与 provider-history 审计；
- 100/100 fresh replay；
- source manifest、dataset SHA、positions `0..99` 和 selection identity；
- manifest↔record、prompt/schema/protocol/model identity；
- provider attempt/token progression 与批级资源上限；
- no-leak 与隐藏 evaluator 隔离。

结论仍是 `diagnostic-only`：现有 atomic SFT/RL 管线不能读取或重标这些记录。42 条严格正确
轨迹只是 scheme-local 候选；必须先实现 checkpoint-relalg 专用 exporter、过滤 audit-only raw
provider 字段、定义恢复错误 turn 的保留策略，并通过明确准入门，才能进入训练。

## 6. 后续执行建议

本次 `workers=1`，个别 15–20 turn episode 阻塞了整个队列。后续固定大批运行应按任务位置
切成 4 个互不重叠、各自有独立 manifest/result-dir 的 shard 并发；每个 episode 内仍严格串行，
不能并行生成依赖尚未返回 Harness 反馈的 turns。聚合时只合并各 shard 审计通过的记录。
