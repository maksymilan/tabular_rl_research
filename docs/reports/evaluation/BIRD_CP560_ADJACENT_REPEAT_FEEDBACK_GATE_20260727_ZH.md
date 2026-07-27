# BIRD checkpoint-560 相邻重复调用与错误反馈门（2026-07-27）

## 结论

本轮只拦截相邻两个模型回合中规范化后完全相同的 `tool + arguments`。不比较
`<think>`，不受 JSON 键顺序影响，不搜索更早历史；工具或任一参数变化、非相邻重复
均放行。

过程指标明显改善，但 48 题 fresh A/B 的最终正确率没有提高，因此不扩大到 120/240
题，也不重跑全量 greedy：

| 运行 | 正确 | 合法结束 | 平均步数 | 错误事件 |
|---|---:|---:|---:|---:|
| fresh version26 control | 13/48 | 21/48 | 11.000 | 74 |
| version27 | 12/48 | 24/48 | 7.583 | 99 |
| version28 | 12/48 | 25/48 | 8.062 | 91 |

version28 相对 fresh version26 的分类结果：

| 冻结类别（各 12 题） | v26 正确 | v28 正确 | v26 合法 | v28 合法 |
|---|---:|---:|---:|---:|
| 相邻重复循环 | 1 | 0 | 5 | 3 |
| 协议格式错误 | 1 | 1 | 2 | 5 |
| 参数/执行错误 | 1 | 3 | 2 | 6 |
| 原干净正确对照 | 10 | 8 | 12 | 11 |

反馈对参数/执行错误有局部正信号，也把部分 `max_steps` 变成了更早、可审计的
`no_progress_error` 或合法终止；但这些改善没有稳定转化为答案正确率。

## 实现

- `ActionCarrierError` 现在提供稳定错误码和结构化细节，区分未闭合/缺失/多重
  `<think>`、旧 `<tool_call>`、无效 JSON、错误 action 键和非对象 arguments。
- 可解析错误反馈携带原始 `attempted_action`，参数验证反馈携带该工具的 required /
  optional argument contract。
- 完全相同的相邻调用在执行前被拒绝，状态哈希保持不变，错误类型为
  `no_progress_error`，错误码为 `adjacent_identical_action`。
- version28 在重复动作紧跟被拒动作时保留原始拒绝根因，避免重复错误覆盖真正的
  参数/执行错误。
- 对完全相同的 `read_subtable` 明确反馈：该工具没有 offset/cursor，相同调用只会
  读取同一行前缀，不会翻页。
- 拒绝动作仍不进入 legal history，也不成为 SFT target。

## 冻结选择与因果对照

冻结 48 题由四类各 12 题组成：相邻重复循环、协议错误、参数/执行错误、干净正确
对照。选择只使用 `example_index`、`db_id`、failure/correct/error 元数据和解析后的
工具调用；没有用 gold SQL 或 gold denotation。

历史全量 greedy 在不同时间使用过不同并发档位，且 vLLM 动态批处理下 temperature=0
跨服务重启仍会产生少量轨迹差异。因此最终门使用同一 24 并发、同一 runner、相同
请求参数的新鲜 version26 control，而不是直接把历史轨迹当作确定性反事实。

## 验证

- SFT/protocol：129 项通过。
- RL：88 项通过，9 项按环境条件跳过。
- harness：91 项通过。
- eval：30 项通过。
- 集成测试确认第二个完全相同调用不会进入执行器，反馈包含 attempted action，且
  state-before/state-after 哈希相同。

冻结 cohort manifest：
`data/eval_inputs/cp560_feedback_gate_v27/manifest.json`

结果目录：

- `data/results/qwen25_coder7b_cp560_tool_feedback_v26_control_stage1_48_r2`
- `data/results/qwen25_coder7b_cp560_tool_feedback_v27_stage1_48_matched`
- `data/results/qwen25_coder7b_cp560_tool_feedback_v28_stage1_48`

原 version26 K=4 评测在门结束后从 353/1534 原断点恢复，没有混入 version27/28
结果。
