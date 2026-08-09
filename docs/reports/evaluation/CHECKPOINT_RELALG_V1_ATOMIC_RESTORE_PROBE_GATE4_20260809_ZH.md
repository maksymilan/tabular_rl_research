# Checkpoint-RelAlg V1 Atomic Restore-Probe Gate4（2026-08-09）

## 结论

本轮首次在真实官方 `deepseek-v4-flash` 因果轨迹中验证了非 root checkpoint 的精确 restore：

- 3 条 treatment-exposed 轨迹均成功执行 `commit → disposable artifact → unknown_column probe → restore`。
- 三次 restore 后的 `environment_state_hash` 与各自目标 checkpoint snapshot hash **完全相等**。
- 三个 probe 分支 artifact 在 recovery snapshot 中均已消失，恢复前的 committed artifacts 保持 active。
- 三条均通过 record 结构审计与 fresh replay，并能从 recovery phase 继续到合法终止。

因此，`restore_checkpoint` 的状态回滚、active membership 恢复、新 recovery phase 和确定性重放机制是有效的。该结论只覆盖机制正确性，不证明 restore 能提高任务准确率。

## 诊断设计

- mode：纯 `atomic`
- carrier：Text-JSON / arm A
- model：官方 `deepseek-v4-flash`
- teacher profile：`restore-probe-v3`
- tasks：teacher1500 v2 nonempty positions 40、43、47、62
- probe：checkpoint 后创建 disposable relation，随后故意对它 `project` 保留的不存在列 `__restore_probe_missing_column__`，预期得到 state-preserving `unknown_column`，下一 turn restore 最近 checkpoint
- admission：diagnostic-only；probe 错误是人为控制流，全部轨迹禁止进入 SFT/RL

离线端到端测试先验证了同一序列：恢复后 snapshot 只保留 commit 时的 artifact、丢弃 disposable artifact，终端答案正确，结构审计与 fresh replay 通过。完整 checkpoint 测试为 212 passed + 34 subtests。

## 首轮预算失效 pilot

首轮使用 `max_checkpoints=2`。4/4 模型都成功制造 probe 错误并立即调用 restore，但模型在 probe 前额外 commit，使 commit 节点耗尽总 checkpoint 节点预算；4/4 restore 均被 `checkpoint_limit_reached` 拒绝。

首轮因此冻结为 **treatment-not-delivered / infrastructure-invalid**，不计入 restore 机制结论。它使用 1,195,526 tokens，结果为 2/4 correct、4/4 legal；所有 record 本身可重放，但没有成功 restore。

## Retry1 结果

Retry1 只把 `max_checkpoints` 从 2 提高到 4，其余 cohort、prompt、模型、工具、episode 限制和 carrier 不变。

| position | correct | commits | probe error | restore | 精确 hash 恢复 | 丢弃的 artifact | record replay/audit |
|---:|---:|---:|---|---|---|---|---|
| 40 | 0 | 3 | `unknown_column` | success，target `checkpoint_003` | yes | `project_006` | pass |
| 43 | 1 | 3 | `unknown_column` | success，target `checkpoint_003` | yes | `project_005` | pass |
| 47 | 0 | 2 | `unknown_column` | success，target `checkpoint_002` | yes | `aggregate_006` | pass |
| 62 | 1 | 4 | `unknown_column` | rejected：`checkpoint_limit_reached` | n/a | n/a | record/replay pass；batch budget fail |

三次成功 restore 均满足：

1. target 是当前 active path 上最新成功 commit checkpoint；不是 root。
2. probe artifact 在 target snapshot 之后创建。
3. restore result hash 与 target snapshot hash 精确相等。
4. recovery checkpoint 的 active artifacts 不含 probe artifact。
5. 恢复后使用新的 artifact id 继续执行，没有复用已废弃 handle。

Retry1 汇总：2/4 correct、4/4 legal、3/4 successful restore，1,274,128 tokens。总批次低于 1,500,000 token 上限。position 62 的单分片最后一个在途 response 使其从 400,000 轻微超到 406,633 tokens，因此 batch-control audit 按设计失败；它不进入 accepted mechanism set。

accepted mechanism set 为 positions 40/43/47：3/3 restore、3/3 exact hash、3/3 discarded membership、3/3 fresh replay，合计 867,495 tokens。

## 与同题 Atomic 基线

四题 Retry1 与原 Atomic Prefix100 结果逐题完全相同：双方均为 2/4 correct；treatment-exposed 三题双方均为 1/3 correct。没有 gain 或 regression。

这不能解释为“restore 对准确率无效”：probe 是人为制造的无信息错误，restore 只是删除一个已知 disposable branch；样本也只有三条。该配对只说明 restore 没有在这三条上破坏原有终局结果。

## 发现的策略/预算问题

`restore-probe-v3` 明确要求只 commit 一次，但 Flash 实际在 Retry1 中进行了 2、3、2、4 次成功 commit。教师提示不足以可靠执行机械 checkpoint 调度。继续单纯提高 `max_checkpoints` 只会掩盖这一问题。

下一步若继续做 restore policy 实验，应在诊断 runner 层采用可审计的控制约束：

- 为一次允许的 restore 预留一个 recovery checkpoint slot，普通 commit 不得消费该保留槽；或
- 为 profile 增加独立 `max_commit_checkpoints=1`，与 `max_restores=1` 分开计数；
- 将 profile compliance（一次 commit、一次 probe、下一 turn restore、恢复后不复用废弃 handle）作为独立 gate，而不只依赖通用结构审计。

生产协议仍应允许自然多阶段 commit；上述限制只属于 restore 机制诊断，不应成为默认 agent policy。

## Artifacts

- treatment 未送达首轮：`data/results/checkpoint_relalg_v1_flash_text_json_atomic_restore_probe_gate4_20260809/`
- Retry1：`data/results/checkpoint_relalg_v1_flash_text_json_atomic_restore_probe_gate4_retry1_20260809/`
- profile implementation commit：`d47145d`
