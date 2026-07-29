# BIRD version37 行读取、日期操作与历史策略冻结 7 题诊断

日期：2026-07-28
指标：`bird-set`
教师：DeepSeek v4 Flash
准入：`diagnostic_only_pending_protocol_scale_gate`

## 目的

Version37 在 version36 的原子工具协议上只增加两类能力：

1. `read_subtable` 可通过类型化 `conditions`、确定性的 `order_by` 和要求排序的
   `offset` 直接读取指定行或后续页，但仍然不创建筛选表；
2. `project` 增加两种类型化逐行日期表达式：
   `date_diff_days(start,end)` 与 `extract_year(date)`。

同时冻结比较两种只包含合法 assistant/observation 对的历史策略：

- A：最近 4 对（`recent-4`）；
- B：最早 5 对加最近 5 对（`head-5+tail-5`），重叠时去重。

两组使用同一 7 个恢复锚点、同一教师、同一最大步数和解码设置。Gold SQL 对教师不可见。
每个旧锚点的拒绝动作均在 version37 下重新校验；旧 version36 的文字错误反馈不进入最终
实验。

## 实现与安全边界

- 行条件读取只是感知，不返回新的关系句柄；需要派生可复用子表时仍使用
  `condition_filter`。
- 正 `offset` 必须同时给出 `order_by`，避免不稳定分页。
- 日期表达式只接受显式 `{column: ...}` 或 `{value: ...}` 操作数，执行器确定性 lowering
  到 SQLite；模型不能提交任意日期程序。
- 质量门只拒绝相邻两步中完全相同的 `tool + arguments`。非相邻重复不作废。
- 学生前缀与所有拒绝动作均为 context/audit only；只有教师合法续写可作为 target。
- 所有产物均为 diagnostic，`sft_export_eligible=false`。

## 配对结果

| example_id | recent-4 | head-5+tail-5 |
|---|---:|---:|
| bird_train_00106 | 正确 | 正确 |
| bird_train_02962 | 错误答案 | 参数校验终止 |
| bird_train_03311 | 正确 | 正确 |
| bird_train_03750 | 正确 | 正确 |
| bird_train_04127 | 参数校验终止 | 错误答案 |
| bird_train_04857 | 正确 | 正确 |
| bird_train_05897 | 正确 | 错误答案 |
| **合计** | **5/7** | **4/7** |

配对上 recent-4 独占一题，head-5+tail-5 没有独占正确题。

| 统计 | recent-4 | head-5+tail-5 |
|---|---:|---:|
| 合法终止 | 6/7 | 6/7 |
| 教师合法动作 | 76 | 93 |
| 教师错误动作 | 3 | 7 |
| 总 token | 709,739 | 1,051,252 |
| 累计墙钟时间 | 885.9 s | 1,248.9 s |
| 新行读取调用 | 9 | 25 |
| 类型化日期表达式 | 1 | 1 |
| 教师相邻完全重复尝试 | 0 | 0 |
| 被接受的非相邻重复 | 1 | 4 |

只看最终答对的轨迹，recent-4 为 33 个合法动作、0 个教师错误、253,546 token；
head-5+tail-5 为 31 个合法动作、2 个教师错误、269,279 token。扩大历史没有带来正确率或
token 收益。

## 能力与失败审计

- 日期题 `bird_train_00106` 在两组都使用了类型化 `date_diff_days`，并通过
  `bird-set` terminal verification；日期能力被真实调用，而不是只存在于 schema。
- 新行读取参数在两组分别使用 9 次和 25 次。条件读取、排序和有序 offset 均被实际调用。
  `read_subtable` 保持只读，关系派生仍由 `condition_filter` 完成。
- `bird_train_05897` 是 recent-4 的唯一配对增益。长历史在并列最高值的语义处理中选择了
  一个不完整分支；这不是缺少行读取能力。
- `bird_train_04127` 两组都表现出“最年轻”方向与城市关系解释不稳定。recent-4 最终又提交
  了超出公开上限的 `limit=35`；反馈明确且环境状态未改变。主要瓶颈是语义/关系路径判断，
  不是执行器缺字段。
- `bird_train_02962` 两组都完成了大量合法关系操作，但对复合属性含义的解释不稳定；
  这是无错误或低错误的语义失败，继续扩大历史不能解决。
- 教师续写没有相邻完全重复调用；非相邻重复按当前政策保留，验证实现没有扩大拒绝范围。

## 审计与结论

recent-4 的 5 条成功轨迹均通过 `audit_verified_rollouts.py` 的 fresh replay、结构门与 full
prompt 变体门。head-5+tail-5 的成功轨迹在同一审计器中仅因当前准入契约固定
`history_turns=4` 而不具备晋级资格；诊断记录本身未发现重复策略污染。

因此：

1. 保留并明确推广现有 `recent-4` 历史策略，不采用 head-5+tail-5；
2. 保留 version37 的行读取与类型化日期能力，进入更大但仍独立的 paired diagnostic；
3. 不把本 7 题 pilot 直接导出为 SFT 数据，也不据此启动训练；
4. 后续能力提升应针对关系语义与任务解释，而不是继续堆叠历史窗口。

## 产物

- recent-4 主运行：
  `data/trajectories/sft2_step1682_repeat_recovery_20260728/version37_history_ab/recent4_revalidated_anchor_final`
- recent-4 单题基础设施补齐：
  `data/trajectories/sft2_step1682_repeat_recovery_20260728/version37_history_ab/recent4_revalidated_anchor_final_retry03311`
- head-5+tail-5 主运行：
  `data/trajectories/sft2_step1682_repeat_recovery_20260728/version37_history_ab/head5_tail5_revalidated_anchor_final`
- head-5+tail-5 单题基础设施补齐：
  `data/trajectories/sft2_step1682_repeat_recovery_20260728/version37_history_ab/head5_tail5_revalidated_anchor_final_retry03311`
