# Atomic version42 显式终止列诊断

## 状态

`version42` 是 version41 之上的独立工具接口诊断，只改变 terminal evidence carrier：

```json
{
  "tool": "answer_from_context",
  "arguments": {
    "evidence": {
      "table": "top_002",
      "columns": ["label"]
    },
    "reason": "label is the only requested output slot."
  }
}
```

它是 diagnostic-only，不能用于 SFT/RL。入口：

```bash
.venv/bin/python -u src/sft/generate_teacher_rollouts.py \
  --atomic-protocol-version version42 \
  --diagnostic-only \
  --context-mode rolling-legal-history \
  --history-turns 4 \
  --rolling-prompt-variant full \
  --database-context-profile catalog-v1 \
  --out <独立结果目录>/verified.jsonl
```

## 设计动机

version41 Gate16 的六个错误全部合法终止，其中三题的 reasoning 已明确说出正确输出槽，却
仍引用带辅助列或错误列顺序的整张表。继续重复 prompt 约束不能把这一意图稳定转换成最终
关系操作。

version42 因此要求模型在 terminal call 中结构化声明已有 grounded columns。Harness：

1. 验证 evidence table 是当前可用 source/handle；
2. 验证 `columns` 是非空、无重复、精确存在的列名列表；
3. 不读取 question、external knowledge、gold SQL 或 model reason；
4. 按声明顺序确定性执行普通 projection；
5. 只对该投影表做 `bird-set` 评分；
6. 记录 source table、requested/resolved columns、projected handle、row count 和输出列。

该 lowering 不能计算、改名、拼接、聚合、去重、改变行集或生成值。需要这些操作时，模型
仍须先调用原关系工具。模型仍负责选择正确列；harness 不做语义猜测。

## 与 version41 的不变量

- `describe_table`、`inspect_column`、`inspect_rows` 和全部关系工具不变；
- multi-edge `join_tables(base, joins[], base_role?)` 不变；
- plan 仍不可见；
- 全部成功/rejected reasoning 保留；
- exact successful calls + unabridged observations 仍为 recent-4；
- 最新 rejected call 和完整错误仍在 `LAST TOOL ERROR`；
- canonical resident state、SQLite 关系执行、provider carrier 和 `bird-set` 不变；
- diagnostic-only，当前 canonical replay/SFT exporter 不接受该终止 carrier。

## 确定性审计值

- DeepSeek provider prompt 字符数：10,741
- provider prompt SHA-256：
  `ef2b67da23dd89a6a9adc765ac33cbb01d188634bbf5db03e1a0f5781def214c`
- protocol hash：`b7d06cb82fddaf13`
- public tool schema SHA-256：
  `7489eb4d084af5572a6c7ca7ca0488f13fb25d1fb483637affd7dc5cb55770e5`

除 `answer_from_context` 的 evidence 语义外，所有非终止 tool specs 与 version41 逐项相同。

## 预注册 Gate16

复用
`data/eval_inputs/bird_train_version41_output_correction_gate16_20260729.jsonl` 的原始
8 个 output-shape 目标和 8 个控制，不重新选题。

扩量到原固定前 50 的门槛：

- output-shape 目标正确至少 4/8；
- version41 正确控制保留至少 7/8；
- 16/16 形成语义终止；
- `terminal_projection_error=0`；
- 总过程错误不超过 2。

任一失败即停止，不运行固定前 50。

## Gate16 结果

version42 得到 **12/16 `bird-set`**，相对 version41 的 10/16 有 2 个恢复、0 个回退；
目标准确 4/8、控制保持 8/8、合法终止 16/16、terminal projection errors 0。它达到
准确率相关门槛，但总过程错误为 4，超过预注册上限 2，因此总体判定 **fail**，没有启动
固定前 50。

其中两次错误来自连接结果只接受完整 dotted column，而模型使用了当前表中可唯一解析的
裸列名；下一版本只诊断这一确定性解析边界。完整结果见
`docs/reports/evaluation/BIRD_ATOMIC_VERSION42_TERMINAL_COLUMNS_GATE16_20260729_ZH.md`。
