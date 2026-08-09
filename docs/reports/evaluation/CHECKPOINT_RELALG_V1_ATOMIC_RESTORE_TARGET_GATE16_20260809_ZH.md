# Checkpoint-RelAlg V1 Atomic Restore-Target-v2 Gate16（2026-08-09）

## 结论

本轮是诊断性验证，不构成准确率或训练协议晋级。

- `restore-target-v2` 在 16 条纯 Atomic 轨迹中产生 5 次合法 `commit_checkpoint`，但产生 **0 次 `restore_checkpoint`**。
- 原因不是 Harness 拒绝 restore，而是预注册触发条件没有真正成立：checkpoint 后只出现孤立错误；唯一含两个后续错误的轨迹中间有成功动作，并且首个错误属于规则明确排除的 carrier/shape 错误。
- 因此本轮验证了 profile 的保守性（没有 root restore、没有无依据 restore），**没有验证 restore 执行后的行为或收益**。
- 与同题 Atomic Prefix100 基线逐题配对：官方 `bird-set` 均为 **10/16**，10 题双方都对、6 题双方都错，0 gain、0 regression。

## 冻结配置

- 模式：`atomic`
- carrier：`text-json`（arm A）
- 教师模型：官方 `deepseek-v4-flash`
- checkpoint guidance：`restore-target-v2`
- 数据源：冻结 teacher1500 v2 nonempty；位置 `38-47, 54-57, 62-63`
- episode：16；四个独立分片并行，单 episode 内保持因果串行
- 总 token 硬上限：4,500,000
- 每 episode：最多 20 model turns、30 primitive calls、2 checkpoints、1 restore
- teacher prompt SHA256：`f3db69caa7e6d26ded0feb08166ddb403ce72df82d18efce917e3980461dca91`

`restore-target-v2` 的关键限制是：只有成功 commit 后才允许 restore；禁止 root restore；只能恢复到当前 active path 上最近的 commit checkpoint；在新 phase 已产出至少一个答案相关 relation artifact 后，遇到明确矛盾或连续两个符合条件的类型/schema/执行错误才立即 restore。carrier/shape/missing-field 错误不计入触发，成功动作会清除连续错误计数。

## 结果

| 指标 | restore-target-v2 | 同题 Atomic 基线 |
|---|---:|---:|
| episodes | 16 | 16 |
| official correct (`bird-set`) | 10 | 10 |
| legal termination | 15 | 15 |
| strict artifact correct | 7 | 7 |
| schema match | 9 | 9 |
| model turns | 150 | 143 |
| tool errors | 8 | 8 |
| provider total tokens | 3,009,688 | 2,944,838 |
| `commit_checkpoint` | 5 | 0 |
| `restore_checkpoint` | 0 | 0 |

paired outcome：both-correct 10、v2-only 0、baseline-only 0、both-wrong 6。v2 比基线多 7 turns（+4.9%）和 64,850 tokens（+2.2%），没有准确率变化。

四个分片均完成；结构审计、manifest/record identity、cohort、provider budget 与 fresh replay 对全部 16 条均通过。无 provider failure。实际 token 使用为硬上限的 66.9%。

## Checkpoint / Restore 行为审计

5 个 commit 均成功创建 `checkpoint_001` 并从 `phase_000` 进入 `phase_001`：

| task position | commit turn | commit 后错误 | 终局 | 为什么没有 restore |
|---:|---:|---|---|---|
| 40 | 5 | 无 | wrong answer | 无触发证据 |
| 43 | 5 | turn 6 `unknown_column`，turn 7 成功 | correct | 单次错误后立即修正 |
| 47 | 7 | turn 10 `invalid_shape` | wrong answer | carrier/shape 错误被规则明确排除 |
| 54 | 13 | turn 18 `missing_required_field`；turn 19 成功；turn 20 `sqlite_execution_failed` | max turns | 首错被排除，且成功动作切断连续性；末次错误后已无下一 turn |
| 62 | 6 | turn 7 `unknown_column`，turn 8 成功 | correct | 单次错误后立即修正 |

其余 11 条没有 commit，因此按 profile 不具备 restore 权限。全批次没有 root restore、错误 checkpoint target 或 restore rejection。

## 对设计预期的判断

本轮支持两个较弱结论：

1. semantic checkpoint 本身可正常建立新 phase，状态/审计/fresh replay 都稳定；profile 成功禁止了早期 Gate8 中的 root restore 与无效目标选择。
2. 仅用自然模型错误加“连续两个错误”触发条件，restore 事件过于稀疏，不能作为 restore 有效性的实验设计。长轨迹或最终答错并不自动意味着早期状态已被证伪；Harness 在答案评分前也不会向模型泄漏 correctness，因此错误答案不能事后触发 restore。

本轮不支持“restore 有效”或“restore 无效”的结论，因为真正的 treatment（restore）从未发生。

## 下一步

若目标是验证 restore 机制而不是教师自发策略，下一轮应使用可审计的确定性 fault-injection / counterfactual gate：在 commit 后构造一个已知可恢复、不会泄漏 gold 的状态破坏或连续执行错误，随机配对 `restore` 与 `no-restore` 两臂，再比较恢复后的 active membership、abandoned path、fresh replay、合法终止与正确率。这样能保证 treatment exposure，而不依赖自然错误偶然连续出现。

如果仍坚持自然教师策略，触发规则应改成“checkpoint 后首个符合条件的 type/schema/execution error 即 restore”，并只选历史上已知 commit 后出现该类错误的任务；这验证的是激进策略，不能与本轮 v2 直接合并。

## Artifact

结果根目录：

`data/results/checkpoint_relalg_v1_flash_text_json_atomic_restore_target_gate16_20260809/`

分片：`shard_038_041`、`shard_042_047`、`shard_054_057`、`shard_062_063`。每个分片包含 manifest、逐题 record、batch status 与审计结果。
