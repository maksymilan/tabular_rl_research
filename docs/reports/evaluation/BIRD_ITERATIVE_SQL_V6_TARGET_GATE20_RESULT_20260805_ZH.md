# BIRD iterative-SQL v6 Target Gate20 结果

日期：2026-08-05  
状态：数值推广门全部通过；反向控制类别失效，只授权新的代表性 paired gate，不进入 SFT/RL

## 结论

在冻结的 output-shape Target Gate20 上，官方 DeepSeek v4 Flash 的 `iterative-sql-v6` 得到
**15/20**，冻结 v5 为 **10/20**；两臂均为 **20/20 legal**。逐题结果为 5 gains、0
regressions、10 both correct、5 both wrong，exact two-sided McNemar `p=0.0625`。

v6 把 actions 从 111 降到 99（-10.8%），process errors 从 6 降到 1，tokens 从 371,036
降到 339,600（-8.5%）。两臂全部 20 条通过 fresh replay、结构和 no-hidden-input-key 审计。
预注册的八项数值门全部通过。

但是，预注册时把 external knowledge 中的 `FirstName+LastName` 解释为明确的组合字符串控制。
运行后本地检查 reference 才发现，这四题仍要求多个独立列；`+` 在该数据集中是字段枚举，
不是字符串拼接指令。该类别不能验证“明确要求单字符串时仍会拼接”的反向能力。因此本结果
足以把 v6 提升为值得进入新代表性 paired gate 的候选，不足以直接推广到 SFT/RL，也不能
声称已经验证防止过度分列的能力。

## 冻结设置与授权边界

- cohort：`data/eval_inputs/iterative_sql_v6_output_shape_target_gate20_v1.jsonl`；
- SHA-256：`12c5aa8611a321a17c73ca9dd66b51c81cbb1e3e2bd57e32c35ba3c8439f8a65`；
- 20 tasks、16 databases；与完整 baseline300 和历史 fixed-200 overlap 均为 0；
- model/provider：`deepseek-v4-flash`，官方 `https://api.deepseek.com`；
- K=1；lazy catalog；recent-4；30 steps；每类最多 3 errors；
- 2,048 max tokens；20 preview rows；20 秒 SQLite deadline；`bird-set`；
- 两臂除 protocol/interface/prompt 外的任务、顺序、数据库、预算、并发和 scorer 相同；
- 用户明确授权发送 question、external knowledge、catalog/schema 和逐步只读反馈；gold SQL、
  gold rows、gold sample、gold row count 与隐藏 verifier 数据未进入外部模型输入。

## 总体统计

| 指标 | v5 | v6 | 变化 |
|---|---:|---:|---:|
| correct | 10/20 | **15/20** | +5 / +25 pp |
| legal terminal | 20/20 | 20/20 | 0 |
| actions | 111 | **99** | -10.8% |
| mean / median / max actions | 5.55 / 5 / 13 | **4.95** / 5 / **8** | -0.60 / 0 / -5 |
| parsed `execute_sql` attempts | 89 | **79** | -10 |
| successful `execute_sql` | 85 | **78** | -7 |
| `submit_sql` attempts | 20 | 20 | 0 |
| process errors | 6 | **1** | -83.3% |
| prompt tokens | 343,919 | **312,371** | -9.2% |
| completion tokens | **27,117** | 27,229 | +0.4% |
| reasoning tokens | **22,261** | 22,877 | +2.8% |
| total tokens | 371,036 | **339,600** | -8.5% |
| API request attempts | 111 | **100** | -11 |
| transport / carrier retries | 0 / 0 | 0 / 0 | 0 |
| completion retries | **0** | 1 | +1 |
| fresh replay correct | 10 | **15** | +5 |
| structural pass | 20/20 | 20/20 | 0 |

## 分类结果

| 预注册类别 | v5 | v6 | gains / regressions | 解释 |
|---|---:|---:|---:|---|
| 6 multi-field separate targets | 1/6 | **5/6** | **4 / 0** | 目标规则产生稳定恢复 |
| 4 `+` mappings，原标为 combined controls | 0/4 | **1/4** | 1 / 0 | reference 实际要求分列，反向控制标签失效 |
| 4 single-field name controls | 4/4 | 4/4 | 0 / 0 | 全部保持 |
| 6 ordinary controls | 5/6 | 5/6 | 0 / 0 | 全部保持 |

### 四个直接目标恢复

- `bird_train_00083`：v5 把 first/middle/last 拼成一列；v6 返回三列；
- `bird_train_01150`：v5 返回拼接 full name + age；v6 返回 first/middle/last + age；
- `bird_train_01886`：v5 返回 id + 拼接 name；v6 返回 id + first + last；
- `bird_train_02499`：v5 拼接 sanitarian name；v6 返回 first_name、last_name 两列。

第五个 gain `bird_train_02806` 来自 `FirstName+MiddleName+LastName` 任务：v5 拼接，v6 分列，
而 reference 也要求三列。这证明 `+` 不能在当前 BIRD external knowledge 中自动等同于字符串
拼接。

## 五个共同失败

- `bird_train_02789`：v6 按 external knowledge 的 `FirstName, LastName, MiddleName` 顺序返回，
  reference 使用 `FirstName, MiddleName, LastName`；这是公开契约与 reference 的字段顺序冲突；
- `bird_train_00641`、`bird_train_02747`：两臂仍把 `+` 映射拼成字符串，reference 要求分列；
- `bird_train_03067`：v6 已分列，但 external knowledge 指定 `nameGiven + lastName`，reference
  使用另一个 first-name 字段；属于字段映射冲突；
- `bird_train_01454`：两臂返回存储的 `number_of_compliments`，reference 重新按行 `COUNT`；与
  本次输出形状规则无关。

因此 15/20 低估了 v6 对可满足 multi-field contract 的服从：唯一 separate target failure
本身存在 external/reference 顺序冲突。但本报告仍只按冻结 scorer 记 15/20，不修改分数。

## Process errors

v5 共 6 个：2 protocol errors、2 SQLite execution errors、2 no-progress errors。v6 只有
`bird_train_01454` 的 1 个 no-progress error，随后恢复为合法但错误的提交。所有过程错误均
保持 successful SQL state，不产生危险 SQL 或隐藏 verifier 反馈。

## 预注册门审核

| 门 | 结果 |
|---|---|
| separate targets 至少 2 gains、0 regressions | 4 / 0，通过 |
| 原 combined 类不低于 v5并保留 v5-correct | 1/4 vs 0/4，数值通过但保留条件为空，语义无效 |
| single/ordinary controls 最多 1 regression、≥90% retention | 0 regressions、9/9 retained，通过 |
| overall net gain ≥2 | +5，通过 |
| legal 不低于 v5且 ≥19/20 | 20/20 vs 20/20，通过 |
| errors ≤ v5+2 | 1 vs 6，通过 |
| token ratio ≤1.10 | 0.915，通过 |
| replay/structure | 全部通过 |

严格地说，所有“数值门”通过；但 combined-control 的语义前提被 reference 推翻，不能用其
宣称反向能力已验证。

## 决策

1. v6 通过 target capability gate，授权一个新的、未见任务上的代表性 v5/v6 paired gate。
2. v6 仍是 diagnostic-only；不进入 SFT/RL，也不据此替换其他 tool scheme。
3. 不在本 Target Gate20 上继续修改 prompt 后复测。
4. 下一代表性 gate 应使用未消费任务、保持同一 v6 prompt，并把“真实单字符串控制”单独
   标为数据稀缺边界；不能再把 `field1+field2` 自动当作拼接语义。
5. 只有代表性 gate 同时保持 accuracy、legal、errors 和 tokens，才考虑更大规模扩展。

## 产物

- v5：`data/trajectories/iterative_sql_20260805/output_shape_target_gate20_v5_flash_r1/`；
- v6：`data/trajectories/iterative_sql_20260805/output_shape_target_gate20_v6_flash_r1/`；
- paired summary：
  `data/trajectories/iterative_sql_20260805/paired_output_shape_target_gate20_v5_v6_summary.json`；
- 两臂目录内均有 `replay_structural_audit.json`。
