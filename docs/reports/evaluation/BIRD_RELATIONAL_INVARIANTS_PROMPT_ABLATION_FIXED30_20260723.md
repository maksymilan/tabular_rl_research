# BIRD relational-invariants prompt 固定 30 题消融

日期：2026-07-23

## 问题

本实验不修改 version19 工具、resident state、optional plan 或执行器，只在 canonical
system prompt 后增加一段 807 字符的通用关系推理约束：

- 在 aggregation/ranking 前确认输入 population 与一行代表的 grain；
- 若答案资格或输出列依赖另一关系，先 join 再 ranking/aggregation；
- 默认 inner join，只有任务明确要求保留未匹配 base rows 时才用 left join；
- 输出 question / external knowledge 指定的实体和 slots，不以 code/label 互换；
- 除非任务要求，不对重复观测做 distinct 或 aggregation。

目标是判断这类 prompt 是否能恢复模型本来能够表达、但因 operator ordering、grain、
join type 或输出实体选择而失败的题。Gold SQL 只供 harness 做 strict-multiset verifier 和
离线审计，没有进入模型输入。

## 严格控制

P0 和 P1 使用相同的：

- 固定 cohort：
  `data/eval_inputs/bird_train_tool_interface_validation200_version4.jsonl` 的前 30 题；
- 模型：`deepseek-v4-flash`；
- temperature=0，max tokens=2048；
- rolling legal history=4，full rolling prompt；
- thinking enabled，reasoning effort=high；
- max steps=30，max recoverable errors/type=3；
- one semantic attempt/task，table output rows=0；
- optional plan，strict no-repair parser；
- `tool-call` provider carrier；
- strict-multiset denotation。

唯一实验变量是：

- P0：`--policy-prompt-variant canonical`；
- P1：`--policy-prompt-variant relational-invariants`。

API-facing system prompt 从 15,019 增加到 15,826 字符，增加 807 字符（5.37%）。两组均
没有调用 `plan`，因此结果不是由强制规划或更长 resident plan 造成的。

结果：

- P0：
  `data/trajectories/tool_usability_20260723/prompt_ablation_tool_call_p0_smoke5.*`
- P1：
  `data/trajectories/tool_usability_20260723/prompt_ablation_tool_call_p1_smoke5.*`
- P1 的 00593 transport-only 补跑：
  `data/trajectories/tool_usability_20260723/prompt_ablation_tool_call_p1_retry_00593_exact.*`

上述 P0/P1 文件名保留了最初的 `smoke5`，但两者都从相同前 5 题通过 `--resume` 扩展到
了完整固定前 30 题；manifest 的 source_count 与 unique_examples 均为 30。

## Carrier 预检与异常隔离

最初尝试沿用 version19 fixed-30 的 JSON Output carrier。该运行前 14 题中有 12 题因
protocol error 终止，79 个回合里 37 次出现空 visible `content`；原始 action JSON 被 API
放入了 `reasoning_content`。旧 P0 fixed-30 仅有 1/204 个回合出现相同问题。

为验证是否由 P1 触发，随后在同一时段对前三题同时运行 JSON Output P0/P1：

| 条件 | 空 visible content | 结果 |
| --- | ---: | --- |
| P0 | 9/10 turns | 0/3 legal |
| P1 | 9/13 turns | 0/3 legal |

当前 P0 同样失败，排除了“relational prompt 导致 carrier 崩溃”的解释。响应 metadata
同时从旧运行常见的小写 `deepseek-v4-flash` 和多种 request ID/指纹，变为统一的
`DeepSeek-V4-Flash` 与另一种 ID 形态，说明当前 API 路由/字段映射发生了外部漂移。

同一时段改用不请求 JSON Output 的 `tool-call` carrier 后，单题预检为 1/1 correct、
9 steps、0 errors；随后 P0/P1 相同 5 题均为 5/5 legal、0 protocol/execution errors。
因此正式 paired-30 将两组统一为 `tool-call`。这使 P0/P1 内部对比有效，但不能把本次
P1 直接与旧 JSON Output version19 的 provider-level carrier 指标混为一谈。值得注意的
是，新的 tool-call P0 仍复现了旧 version19 的 21/30 accuracy。

## 结果

P1 的 00593 首次完整批次在最后请求遇到
`RemoteDisconnected: Remote end closed connection without response`。这被单独记为 API
transport failure，不当作模型语义失败。对同一题做一次 transport-only 全轨迹补跑后，
得到合法但错误的终局，因此无论保守记 0 还是用补跑结果，P1 最终分数都是 24/30。

| 指标 | P0 canonical | P1 relational invariants | 差值 |
| --- | ---: | ---: | ---: |
| Strict-multiset correct | 21/30 | **24/30** | **+3** |
| Accuracy | 70.00% | **80.00%** | **+10.00 pp** |
| Legal terminal | 30/30 | 30/30 | 0 |
| Wrong answer | 9 | **6** | -3 |
| 共同正确 | 21 | 21 | — |
| P1 gains | — | **3** | — |
| P1 regressions | — | **0** | — |
| 共同错误 | 6 | 6 | — |
| Recoverable action errors | 2 | **1** | -1 |
| 平均 actions/task | 6.50 | **6.27** | -0.23 |
| 总 model turns | 195 | **188** | -7 |

P0 的两个 recoverable errors 是一个 provider carrier protocol error 和一个错误 scalar
reference；P1 的一个 error 是 provider carrier protocol error。P1 没有新增工具执行错误。

P1 恰好达到此前预设的 fixed-30 扩展线 24/30，但 30 题 paired McNemar discordant
counts 为 3 gains / 0 regressions，双侧 exact p=0.25。方向一致且效果有实际意义，但样本
仍不足以宣称稳定提升，更不能由此宣称固定 200 已达到 75%。

## 三个增益的轨迹证据

| ID | P0 错误 | P1 修正 | 对应 invariant |
| --- | --- | --- | --- |
| 03390 | 只过滤 `language` 并返回 `Country` code，如 `AG`、`AUS`；gold 要 nation name。 | 先把过滤结果 join `country`，再投影 `country.Name`。 | 若输出实体依赖另一关系，必须形成 join；code 不能替代 label。 |
| 05440 | 先在全体 `Paper` 上取异常最大 Year=800190，再 left join 得到 4 个 NULL homepage。 | 先 inner join `Paper`/`Journal`，再按 `Paper.Year DESC` top-1，返回正确 title/homepage。 | join 前后 population 不同；应先 join 再 ranking，默认 inner。 |
| 05952 | 已得到两条 Sankee rows，最后额外 `distinct=true`，把两个相同 county 压成一行。 | 保留 join 的两行，只投影 county，严格匹配 gold multiplicity。 | 不要在任务未要求时改变 grain 或折叠重复观测。 |

三条 gain 都是新增 prompt 明确针对的决策，而且 P1 仍通过相同原子工具完成，没有新增
复合工具或 harness 猜测。这支持“提示帮助模型正确使用现有工具”的解释，而不是工具表达
能力发生变化。

## 仍然失败的六题

| ID | P1 失败原因 | 判断 |
| --- | --- | --- |
| 00593 | 看见两次 11/18 天疗程，却自行固定第一次 encounter，只返回 11。 | grain / multiplicity 规则仍未被稳定遵守。 |
| 01152 | external knowledge 把 player name 定义为 `playerID`，模型按其返回 ID；gold 要 first/middle/last 三列。 | prompt/gold conflict。 |
| 03688 | 先找所有最长影片再返回它们的全部 inventory rows；gold 在无 tie rule 下任取一行 `LIMIT 1`。 | benchmark tie/multiplicity ambiguity。 |
| 06026 | 误查 west 与 south，并把 per-row profit 跨区域 sum；gold 只取 south 的 distinct Profit。 | relation choice 与不必要 aggregation；P1 未修复。 |
| 06489 | 正确找到 `OBJ_SAMPLE_ID=18`，却额外 join class 并输出 `[18, "paper"]` 两列。 | exact output slots 规则仍未稳定遵守。 |
| 06492 | 按问题字面先逐 image 计 object samples 再筛 `<15`；gold 实际是逐行筛 `OBJ_SAMPLE_ID<15` 后 count。 | prompt/gold conflict。 |

其中 3 题主要是模型仍未遵守通用 invariants，3 题含明显 gold conflict 或未声明的
tie/multiplicity 偏好。P1 改善了三类错误，但不是对这类错误的确定性约束。

## 成本与轨迹长度

P1 的静态 system prompt 增长 5.37%，但因平均轨迹从 6.50 降到 6.27 actions，固定 30
的 model turns 反而减少 7。用 00593 的合法补跑替代 transport-failed 记录后：

- P0：953,851 total tokens；
- P1：961,102 total tokens；
- P1 总 token 增加约 0.76%。

00593 补跑自身有两次 transport retry，因而 token/cost 只能作为本次运行实测，不应被
解释为 prompt 的稳定成本效应。可靠的静态开销是增加 807 字符；可靠的轨迹信号是本次
paired cohort 中少 7 个 model turns。

## 结论与下一步

1. 在固定 30 题上，P1 从 21/30 提升到 24/30，无 regression，并恰好通过预设扩展线。
2. 三个 gain 与 prompt 的 join-before-rank、exact entity 和 preserve-grain 规则逐一对应，
   因而不是随机的 verifier 数字改善；但 n=30 的统计证据仍弱。
3. 这段 prompt 是 policy guidance，不是新的工具能力。论文/实验应继续把 version19
   canonical tools 与 P1 prompt 分开命名和消融，不能把 P1 的 +3 归因给工具设计或 RL。
4. 若后续 SFT/RL 采用 P1，训练数据生成、SFT inference、Base control 和 RL evaluation
   必须使用同一 P1；只在 RL eval 增加它会构成不公平变量。
5. 下一步可以按原 gate 扩到冻结 200 题做 P0/P1 paired 验证，继续单列 API/transport
   failures；在 200 题达到至少 150/200 前，不应声称 75% 目标已经实现，也不应据此开始
   新 SFT 数据构造。
