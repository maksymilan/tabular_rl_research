# Atomic version43 唯一裸终止列诊断

## 状态

`version43` 是 version42 之上的最小接口修正，只改变 terminal column name resolution。
模型仍提交：

```json
{
  "tool": "answer_from_context",
  "arguments": {
    "evidence": {
      "table": "join_002",
      "columns": ["n_name"]
    },
    "reason": "Return the requested nation name only."
  }
}
```

Harness 按以下固定顺序解析每个列名：

1. 先做大小写不敏感的完整逻辑列名匹配；
2. 仅当输入不含 `.` 且完整匹配为零时，比较 dotted logical columns 的末段；
3. suffix 候选恰好一个时解析为该完整列名；
4. 候选为零或多个时返回结构化 `unknown_column`；
5. 已限定但错误的 `wrong.column` 不会按末段修复。

例如，在只含 `players.playerID` 的结果上，`playerID` 可确定性解析；在同时含
`players.playerID` 和 `appearances.playerID` 的结果上，`playerID` 必须被拒绝。

version43 是 diagnostic-only，不能用于 SFT/RL。

## 变更边界

除上述列名解析外，version43 保持 version42 的全部边界：

- 显式、有序、非空 terminal columns；
- Harness 只做普通 projection，不改变行、值、顺序，不计算、不改名、不去重；
- 不读取 question、external knowledge、gold SQL 或 model reason；
- 所有非终止工具、参数、execution、state、feedback、join rule 不变；
- plan 不可见；
- 全部 successful/rejected reasoning 保留；
- exact successful calls + unabridged observations 固定 recent-4；
- provider native thinking + JSON Output 不变；
- `bird-set` 不变；
- canonical SFT/replay 不接受该实验 carrier。

审计记录升级为 `terminal-column-projection-v2`，逐列保存：

- `requested`
- `resolved`
- `mode`：`exact` 或 `unique-bare`

## 确定性审计值

- DeepSeek provider prompt 字符数：10,840
- provider prompt SHA-256：
  `ea93167c66a545e1ef1f7fd94f2fb07aef6253c70f51103c2392ef254244a380`
- protocol hash：`2175e3e5e812c1a1`
- public tool schema SHA-256：
  `35ad95252681fb9ac3a58a32d5dd24fd8f771e8beacb809e582780598649d9f8`

## 预注册 Gate16

复用同一冻结输入
`data/eval_inputs/bird_train_version41_output_correction_gate16_20260729.jsonl`
（SHA-256
`4dcd9146195e0658fccea5b8775ccd233290d8175529f4a72bcdc4f7d4071f40`）。
不重新选题，不改变 version42 的 8 个目标和 8 个控制。

通过条件：

- 目标准确至少 4/8；
- version41 正确控制保持至少 7/8；
- 16/16 形成语义终止；
- `terminal_projection_error=0`；
- 总过程错误不超过 2。

全部通过后才能运行原冻结 first-50；任一失败即停止 version43。由于本轮只修复 deterministic
column resolution，Gate16 的四个 version42 错误答案不作为定向选择或 prompt 内容。

## 实验结果

Gate16 得到 **13/16**：目标准确 5/8、控制保持 8/8、合法终止 16/16、过程错误 1、
terminal projection errors 0，通过全部预注册门槛。

随后原冻结 first-50 得到 **39/50 = 78%**，相对 version24 的 42/50 有 1 个恢复、4 个
回退；过程错误 9，超过预注册上限 5。50 个终止投影全部成功，6 题共 12 列使用
unique-bare resolution，均未发生 resolver 错误。

因此 version43 停止扩量，不运行其余 150，不用于 SFT/RL。完整配对和错误归因见
`docs/reports/evaluation/BIRD_ATOMIC_VERSION43_UNIQUE_BARE_TERMINAL_COLUMNS_20260729_ZH.md`。

first-50 的 11 个失败随后各做一次 fresh causal retry：恢复 3 题，合并
verifier-selected pass@2 为 42/50，但 fresh retry 只有 10/11 合法终止、产生 7 次过程
错误。两轮合计 1,588,917 tokens，略高于 version24 单次用 1,557,640 tokens 得到同样
42/50。该 K=2 诊断也停止扩量。
