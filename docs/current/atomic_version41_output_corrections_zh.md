# Atomic version41 输出约束与错误纠正 Prompt

## 状态

`version41` 是 version40 之上的 **prompt-only external-teacher 诊断**。它不改变工具、
参数、执行、状态、反馈、历史策略或 DeepSeek carrier，也不能用于 SFT/RL。入口：

```bash
.venv/bin/python -u src/sft/generate_teacher_rollouts.py \
  --atomic-protocol-version version41 \
  --diagnostic-only \
  --context-mode rolling-legal-history \
  --history-turns 4 \
  --rolling-prompt-variant full \
  --database-context-profile catalog-v1 \
  --out <独立结果目录>/verified.jsonl
```

不要把 version41 结果写入 version24、version26、version39 或 version40 的结果目录。

## 唯一变更

Prompt 变更集中在 `src/sft/atomic_version41_prompt.py`，由两个独立段落组成：

1. `OUTPUT CONTRACT`：合并 version40 TASK/RULES 中重叠的终止形态说明，并恢复 version24
   中被误删的非冗余边界：
   - 输出槽和顺序在关系操作前固定；
   - 未明确要求格式化时，独立源字段保持独立；
   - 不把 ID/code 换成 label/name；
   - 不擅自翻译、大小写归一或舍入；
   - 去除 filter、join、count、ranking helper 列；
   - 用 `return_columns` 或 `project` 形成精确终止表。
2. `ERROR-CORRECTION EXAMPLES`：只覆盖 version40 Gate50 实际发生过程错误的工具：
   - `condition_filter`：区分 table membership 的 `in_table` 与 scalar comparison 的
     producing-step `value_ref`；
   - `extreme_value_select`：非空 `order_by`，方向和列写在同一个字符串中；
   - `group_aggregate`：columns layout 的配套参数；
   - `scalar_compute`：合法 operation 和 producing-step operands；
   - `inspect_rows`：`limit=1..20`，正 offset 必须带确定性 `order_by`。

只展示纠正后的合法动作，不把非法动作作为可复制 JSON 示例。每个示例都通过 version41
严格 parser/argument validator 测试。

## 不变量

- public tool schema hash 与 version40 完全相同；
- `plan` 仍不可见，行观察名称仍为 `inspect_rows`；
- multi-edge `join_tables(base, joins[], base_role?)` 不变；
- 全部成功/拒绝 reasoning 继续保留；
- exact successful call + unabridged observation 仍为 recent-4；
- 最新 rejected call 和完整结构化错误仍在 `LAST TOOL ERROR`；
- canonical resident state、SQLite 执行、grounding、replay 和 `bird-set` 判分不变。

DeepSeek v4 Flash provider prompt 从 version40 的 7,503 字符增至 10,572 字符，仍明显小于
旧 external-teacher prompt。新增 3,069 字符全部属于已观察错误的判别信息，不恢复旧 prompt
中的逐工具重复说明。

当前确定性审计值：

- DeepSeek provider prompt SHA-256：
  `55207dae78fb4a953ed948e14565d45f59c3ae669e809f5d49ec7fe80ea5d87b`；
- protocol hash：`35295fde1ec3b140`；
- 与 version40 相同的 public tool schema SHA-256：
  `b7761922ee1164520630f4dbe66c7c766abb9257e617ba7061e4730884ae08c4`。

## 验证顺序

先使用 version40 Gate50 的 8 个 output-shape 错误加 8 个同类正确控制建立冻结 Gate16。
建议预先固定：

- output-shape 目标恢复至少 4/8；
- 正确控制保留至少 7/8；
- 过程错误不高于 version40 同题；
- 16/16 形成语义终止；
- 通过后才在原固定前 50 题上配对 version40/version24。

如果 version41 仍不能恢复终止列选择，下一步才单独测试 reasoning-history 策略；不要同时
改工具名、plan 或执行反馈。
