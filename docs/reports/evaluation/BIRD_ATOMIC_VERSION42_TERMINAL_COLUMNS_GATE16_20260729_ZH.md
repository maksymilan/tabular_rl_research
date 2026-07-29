# Atomic version42 显式终止列 Gate16

## 结论

version42 **通过了准确率、控制保持、合法终止和投影执行门槛，但未通过总过程错误门槛**。
按照预注册规则，停止 version42，不扩跑固定前 50，也不准入 SFT/RL。

在冻结的 8 个 output-shape 目标和 8 个正确控制上，DeepSeek v4 Flash version42 得到
**12/16 `bird-set`**，相对同题 version41 的 **10/16** 有 2 个恢复、0 个回退，
exact two-sided paired binomial `p=0.5`。8/8 控制全部保持，16/16 合法终止，目标达到
预注册的 **4/8**；但出现 4 次过程错误，超过预注册上限 2。

显式终止列恢复了两类“reasoning 已知正确输出槽、terminal 却引用整表”的错误：

- `bird_train_05316`：从带 `label, review` 的排名结果中只提交 `label`；
- `bird_train_06489`：严格服从 external knowledge，只提交 `OBJ_SAMPLE_ID`，没有再按
  自然语言偏好连接 class label。

这说明结构化 terminal columns 比继续重复 prompt 输出说明更能把已形成的语义意图转换为
最终关系操作。不过 version42 对连接结果只接受完整逻辑列名，造成两次可避免的纠错回合，
所以它不能按当前接口扩量。

## 冻结设计

复用 version41 Gate16 的原始输入，不重新选题：

- 输入：
  `data/eval_inputs/bird_train_version41_output_correction_gate16_20260729.jsonl`
- SHA-256：
  `4dcd9146195e0658fccea5b8775ccd233290d8175529f4a72bcdc4f7d4071f40`
- 8 个目标：`01152, 02408, 02901, 02925, 03390, 05316, 06336, 06489`
- 8 个控制：`03275, 04636, 05543, 02437, 02605, 06454, 04377, 03786`

固定运行参数：

- 模型：`deepseek-v4-flash`
- 协议：atomic version42，`think-json-v1`
- 上下文：`rolling-legal-history`，`history_turns=4`
- reasoning：全部 successful/rejected reasoning 保留
- exact successful calls + unabridged observations：recent-4
- plan：disabled by protocol
- provider：native thinking + JSON Output，reasoning effort high
- 每题一次语义尝试，错误答案不重试
- `max_steps=30`，`max_tokens=2048`，request retries=3
- 判分：`bird-set`
- 训练准入：`diagnostic_only_pending_protocol_scale_gate`

预注册门槛：

| 门槛 | 要求 | 实际 | 结果 |
|---|---:|---:|---|
| 目标准确 | ≥4/8 | 4/8 | pass |
| 控制保持 | ≥7/8 | 8/8 | pass |
| 语义终止 | 16/16 | 16/16 | pass |
| terminal projection errors | 0 | 0 | pass |
| 总过程错误 | ≤2 | 4 | **fail** |

总体：**fail，停止 version42 扩量。**

## 配对结果

| 指标 | version24 | version40 | version41 | version42 |
|---|---:|---:|---:|---:|
| 正确 | 13/16 | 8/16 | 10/16 | **12/16** |
| 合法终止 | 16/16 | 16/16 | 16/16 | **16/16** |
| 过程错误 | 0 | 6 | 0 | **4** |
| 有过程错误的题 | 0 | 4 | 0 | **4** |
| 合法动作 | 86 | 96 | 87 | **82** |
| 平均动作 | 5.38 | 6.00 | 5.44 | **5.13** |
| API requests | 86 | 98 | 88 | **82** |
| prompt tokens | 460,125 | 419,733 | 407,986 | **373,013** |
| completion tokens | 15,757 | 22,517 | 22,739 | **27,112** |
| reasoning tokens | 12,571 | 18,711 | 19,373 | **23,212** |
| total tokens | 475,882 | 442,250 | 430,725 | **400,125** |

version42 对 version41：

- 两者都对：10
- 两者都错：4
- version42 独赢：2（`05316`、`06489`）
- version41 独赢：0

version42 对 version24：

- 两者都对：11
- 两者都错：2
- version42 独赢：1（`05316`）
- version24 独赢：2（`02901`、`06336`）
- exact two-sided paired binomial `p=1.0`

因此 version42 的局部恢复是真实配对增益，但还没有超过同题 version24。

## 四次过程错误

### 新终止接口的精确列名错误：2

1. `02605`：模型对 `join_002` 提交裸列名 `n_name`，实际唯一列为
   `nation.n_name`；看到完整错误后恢复，最终正确。
2. `01152`：模型对 `top_003` 提交裸列名 `playerID`，实际唯一列为
   `players.playerID`；看到完整错误后恢复，但最终仍因任务输出语义冲突而错误。

这两次都不需要 question、gold 或 reason 即可确定性解析：在当前表中按末段名称匹配时
恰好只有一个候选。下一个版本可以沿用现有关系工具的 unique-bare-column 规则；若候选
不唯一则继续拒绝。

### 与新终止投影无关：2

1. `02925`：一次 DeepSeek visible content 不是合法 JSON action；结构化 provider 错误完整
   返回，模型随后恢复。
2. `06454`：`condition_filter.conditions.and[1].in_table="step_4"` 不是表/结果 handle；
   模型随后改用正确 handle，并得到正确答案。

没有发生 `terminal_projection_error`；16 个最终 terminal declarations 都被成功降低并
评分。

## 四个仍错目标

- `01152`：external knowledge 明写 “player name refers to playerID”，而 benchmark 要求
  first/middle/last。模型严格按 external knowledge 返回 `players.playerID`。这是任务规范
  冲突，不应靠工具猜测修复。
- `02925`：模型找到正确 ProductID=873，但问题只说 “Which product”，最终选择产品
  `Name`。这是隐含输出槽歧义。
- `06336`：terminal carrier 已显式表达顺序，但模型仍声明 `["wid","word"]`，而 gold 是
  `word,wid`。接口使错误可观测，但不能替模型决定语义顺序。
- `02901`：external knowledge 定义 full name 为三个字段相加，模型形成无空格的单列
  `FullName`；gold 保留三个独立字段。这同样是规范/表示边界冲突。

剩余四题不应通过 harness 自动重排、自动拆列或读取 gold 修复。

## 下一步

1. 不扩跑 version42 固定前 50，不用于 SFT/RL。
2. 新建 version43，只把 terminal column resolution 从“完整精确列名”改为：
   - 先匹配完整列名；
   - 无完整匹配时，允许唯一的 dotted-column suffix；
   - 0 个或多个候选仍返回结构化 `unknown_column`。
3. version43 不改变 prompt 纪律、非终止工具、行集、值、列顺序或评分；仍不读取 question、
   external knowledge、gold SQL 或 model reason。
4. 在同一冻结 Gate16 上复测。准确率/控制门槛保持不变，并继续要求总过程错误不超过 2；
   未通过则停止该方向。

## 审计产物

- version42 全记录：
  `data/trajectories/tool_usability_20260729/version42_terminal_columns_gate16_bird_set.all.jsonl`
  （SHA-256
  `1c88abd65f2984e3d6b94fd9ed50774fdc2a4e5fc9c6942ab1e79b355c638262`）
- version42 manifest：
  `data/trajectories/tool_usability_20260729/version42_terminal_columns_gate16_bird_set.manifest.json`
  （SHA-256
  `ff4d7c5693849ac341fc4c9f3e1798d5ff7ab076d691fb082ed773cb4187d9b1`）

本轮没有启动固定前 50 题扩跑。
