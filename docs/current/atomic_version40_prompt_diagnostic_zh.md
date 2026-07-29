# Atomic version40 Prompt 诊断设计

## 状态

`version40` 是独立的 external-teacher 诊断配置，不替代 version39，也不能用于 SFT/RL。
入口是 `src/sft/generate_teacher_rollouts.py`：

```bash
.venv/bin/python -u src/sft/generate_teacher_rollouts.py \
  --atomic-protocol-version version40 \
  --diagnostic-only \
  --context-mode rolling-legal-history \
  --history-turns 4 \
  --rolling-prompt-variant full \
  --database-context-profile catalog-v1 \
  --out <独立结果目录>/verified.jsonl
```

不要把 version40 结果写入 version24、version26 或 version39 的目录。

## 固定变更

- 公开行观察工具从 `read_subtable` 改名为 `inspect_rows`；执行仍使用同一个只读实现。
- 从公开工具、prompt 和严格解析器中移除 `plan`。
- `join_tables(base, joins[], base_role?)` 的参数、校验和执行语义完全不变。
- 所有成功动作的历史推理完整保留；原始调用和未压缩工具结果只保留最近四对。
- 被拒绝的 assistant 文本仍不进入历史，只保留结构化 `LAST TOOL ERROR`。
- DeepSeek 继续使用 native reasoning + JSON Output，但不再同时看到 canonical carrier、
  teacher one-action suffix 和 provider carrier 三份响应规则。

## Prompt 精简结果

在 DeepSeek v4 Flash JSON Output 配置下：

| Prompt | 字符数 |
|---|---:|
| version39 当前 external-teacher provider prompt | 21,170 |
| version40 provider prompt | 7,458 |
| 减少 | 64.8% |

version40 仍保留背景、任务、环境、完整工具语义、复杂参数示例、推理连续性、操作规则和
provider 响应合同。它不是只保留工具签名的极简 prompt。

明确删除的是重复信息：

- student tool spec 后再次逐工具展开的 teacher tool guidance；
- canonical response、teacher one-action 和 DeepSeek response carrier 的重复约束；
- data-generation suffix 中重复的推理、重复读取和计划说明；
- `plan` 的语义、调用示例和 resident-plan 规则；
- 多处重复出现的 external knowledge、最终表形状、population/grain 和异常检查长段落。

## 暂时保留、需要下一轮单独确认的候选项

以下内容可能仍有 token 冗余，但这次没有删除，因为同时删除会让失败原因不可归因：

1. **五个复杂参数 JSON 示例。** 可测试只留 `inspect_rows`、`join_tables` 和
   `group_aggregate` 三个示例；风险是条件树和 terminal evidence 的格式错误回升。
2. **每个工具的完整一行语义。** 可只压缩 `describe_table`、`inspect_column`、
   `set_op` 等低熵工具；风险是外部模型不能访问独立 JSON schema，只靠 prompt 理解接口。
3. **异常结果检查规则。** 可删除 empty/multiplicity/NULL/date/arithmetic 的统一提醒；
   风险是模型更容易合理化异常结果，而 version38 的失败审计表明这是实际错误源。
4. **最终表形状的双层表达。** 当前在 TASK 中定义评分边界，在 RULES 中说明如何移除
   helper columns；可合并成一处，但可能再次增加“语义正确、最终表格式错误”。
5. **推理长度的“两到六句”软范围。** 这是本轮按要求新增的行为约束；若观察到模型机械
   凑句子，再单独测试只保留 “focused, medium-length continuation”。

建议先用当前 version40 做固定 target/control 小门，再根据错误分型只选择一个候选项做
paired ablation。至少分别报告：`bird-set`、合法终止、process errors、provider/transport
失败、最终表 shape-only 错误、总 token 和平均 action 数。
