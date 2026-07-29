# Atomic version41 输出约束与错误纠正 Gate16

## 结论

version41 **未通过预注册的输出形态恢复门槛，不扩跑固定前 50 题，不准入 SFT/RL**。

在冻结的 8 个 version40 output-shape 错误和 8 个同类正确控制上，DeepSeek v4 Flash
version41 得到 **10/16 `bird-set`**，相对同题 version40 的 **8/16** 有 2 个恢复、0 个
回退，exact two-sided paired binomial `p=0.5`。8/8 控制全部保留，16/16 形成合法语义
终止，过程错误从 6 降为 0；但目标只恢复 **2/8**，低于预注册的至少 4/8。

与更早的同题 version24 相比，version41 仍是 10/16 对 13/16：0 个独赢、3 个回退，
paired `p=0.25`。因此不能把局部的 +2 宣称为恢复了旧 prompt 的能力。

## 冻结设计

目标题是 version40 Gate50 中“实体/行集基本正确但输出字段、顺序或表示错误”的全部八题：

- `bird_train_01152`
- `bird_train_02408`
- `bird_train_02901`
- `bird_train_02925`
- `bird_train_03390`
- `bird_train_05316`
- `bird_train_06336`
- `bird_train_06489`

每题匹配一个 version40 正确控制，分别覆盖三列输出、单一名称列、独立字段、ID 表示、
key-to-label join、排名辅助列移除、两列顺序和 identifier 输出。目标/控制交错执行。

固定运行参数：

- 模型：`deepseek-v4-flash`
- 协议：atomic version41，`think-json-v1`
- 输入：`bird_train_version41_output_correction_gate16_20260729.jsonl`
- 输入 SHA-256：
  `4dcd9146195e0658fccea5b8775ccd233290d8175529f4a72bcdc4f7d4071f40`
- 上下文：`rolling-legal-history`，`history_turns=4`
- reasoning：全部成功和 rejected reasoning 保留
- exact successful calls + unabridged observations：recent-4
- plan：disabled by protocol
- provider：native thinking + JSON Output，reasoning effort high
- 每题一次语义尝试，错误答案不重试
- `max_steps=30`，`max_tokens=2048`，request retries=3
- 判分：`bird-set`
- 训练准入：`diagnostic_only_pending_protocol_scale_gate`

预注册扩量门槛：

| 门槛 | 要求 | 实际 | 结果 |
|---|---:|---:|---|
| 目标恢复 | ≥4/8 | 2/8 | fail |
| 控制保留 | ≥7/8 | 8/8 | pass |
| 语义终止 | 16/16 | 16/16 | pass |
| 过程错误 | ≤6 | 0 | pass |

总体：**fail，停止扩量。**

## 配对结果

| 指标 | version24 | version40 | version41 |
|---|---:|---:|---:|
| 正确 | 13/16 | 8/16 | 10/16 |
| 合法终止 | 16/16 | 16/16 | 16/16 |
| 过程错误 | 0 | 6 | 0 |
| 有过程错误的题 | 0 | 4 | 0 |
| 合法动作 | 86 | 96 | 87 |
| 平均动作 | 5.38 | 6.00 | 5.44 |
| API requests | 86 | 98 | 88 |
| prompt tokens | 460,125 | 419,733 | 407,986 |
| completion tokens | 15,757 | 22,517 | 22,739 |
| reasoning tokens | 12,571 | 18,711 | 19,373 |
| total tokens | 475,882 | 442,250 | 430,725 |

version41 对 version40：

- 两者都对：8
- 两者都错：6
- version41 独赢：2（`02408`、`03390`）
- version40 独赢：0

version41 对 version24：

- 两者都对：10
- 两者都错：3
- version41 独赢：0
- version24 独赢：3（`02901`、`06336`、`06489`）

## 两个恢复

### `02408`：作者名

version40 保留 `author_id, author_name`；version41 直接形成只含 `author_name` 的关系并终止。
这符合恢复后的单一输出槽约束。

### `03390`：国家名

version40 在 language 表中过滤正确行后直接返回 country code；version41 继续连接 country
表并只投影国家名。这里恢复了“key/code 不是自然语言 nation label”的区别。

## 六个仍错目标

### 明确知道输出要求但未执行最终投影

- `05316`：reasoning 明确说只需要 restaurant `label`，却直接引用同时含 `label, review`
  的排名表。
- `06336`：请求顺序是 word、id，模型仍以 `wid, word` 终止。
- `06489`：external knowledge 明确把 object 映射到 `OBJ_SAMPLE_ID`，模型也观察到 ID=18，
  随后仍按自然语言常识连接 class 表并输出 `paper`。

这三题说明模型不是没有看见规则；它在终止决策中主动覆盖了规则。继续重复相同的全局
输出说明，预期收益很低。

### question/external knowledge 与 benchmark 输出槽存在冲突或歧义

- `01152`：external knowledge 明写 “player name refers to playerID”，而 benchmark 终止槽
  是 first/middle/last name。严格服从 external knowledge 会稳定地产生 `playerID`。
- `02901`：external knowledge 写 “full name = FirstName+MiddleName+LastName”，模型将其理解
  为拼接单列；benchmark 保留三个独立字段。它与“显式格式要求优先”和“默认保留独立字段”
  的边界冲突。
- `02925`：问题只说 “Which product”，external knowledge 只定义 cost，没有说明返回
  ProductID 还是 product name。模型找到正确 ProductID=873 后加入自然名称，并引用两列。

这些题不适合继续用更多通用 prompt 规则解决；应优先修正 external knowledge/output-field
映射，或把 benchmark 隐含输出槽显式提供给模型。

## 错误示例的效果

version40 在这 16 题上出现六个过程错误：

- `condition_filter`：2
- `group_aggregate`：1
- `extreme_value_select`：3

version41 同题为 0，且动作从 96 降至 87、API requests 从 98 降至 88。对应的合法纠错
示例具有正向接口信号。不过 `scalar_compute` 和 `inspect_rows` 的纠错示例没有在本 Gate
中被实际触发，不能据此宣称这两个工具的错误已被解决。

## 下一步

1. 不扩跑 version41 固定前 50，不用于 SFT/RL。
2. 保留纠错示例作为接口候选，但将“输出合同”和“错误示例”拆开做因子验证；当前 +2 无法
   判断来自哪一部分。
3. 下一轮优先做 **task output specification** 诊断，而不是继续添加 prompt：
   - 为问题提供显式、按顺序的 output fields；
   - 先排除 `01152`、`02901`、`02925` 这类 external/gold 冲突题；
   - 在 `05316`、`06336`、`06489` 上检查显式 output fields 是否能覆盖自然语言偏好。
4. 若不允许暴露 output fields，剩余错误更接近外部教师能力上限，应测试多样采样 +
   verifier 选择，而不是继续扩大工具说明。

## 审计产物

- 冻结选择：
  `data/eval_inputs/bird_train_version41_output_correction_gate16_20260729.manifest.json`
- version41 全记录：
  `data/trajectories/tool_usability_20260729/version41_output_correction_gate16_bird_set.all.jsonl`
  （SHA-256
  `354588404de14baf68d94431ebf88582a129cada341642bbcc651d866c278139`）
- version41 manifest：
  `data/trajectories/tool_usability_20260729/version41_output_correction_gate16_bird_set.manifest.json`
  （SHA-256
  `50ef8b3879463b669fcaf79d04a107882e02f180e7505bab38fa2d9ca3fb6a2d`）

本轮没有启动固定前 50 题扩跑。
