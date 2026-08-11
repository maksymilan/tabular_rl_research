# checkpoint-relalg 模型自主 Checkpoint vs 禁用 Checkpoint Gate4（2026-08-11）

## 结论

本轮没有证明 checkpoint 提高或降低答案正确率，因为实验操纵没有形成：模型自主臂在 4/4 个 episode 中均未尝试 `commit_checkpoint`，包括 14–15 turn 的长轨迹。因而两臂实际都走了零-checkpoint 路径，差异只剩 teacher guidance 与采样随机性。

这重新确认了最初引入更强 checkpoint 触发策略的原因：当 checkpoint 完全是可选动作时，教师模型倾向于持续执行关系算子直至回答或耗尽预算，而不会主动为后续阶段压缩历史。Harness 的 producer quota 虽能制造 checkpoint exposure，却把“何时分阶段”的判断从模型转移给了机械计数；两者都不是最终解。

`restore_checkpoint` 在两臂的运行预算均为 0，8/8 episode 的 restore count 为 0。新实验可将 restore 视为废弃能力；工具定义暂时保留只为历史 replay 兼容，不代表 forward policy 仍允许调用。

## 冻结实现

- 实现提交：`1ff03b60f60caf8b4c48b3d5ade6cbd1eacd2ff9`（`feat: add model-choice checkpoint ablation`）。
- 自主臂 guidance：`model-choice-commit-v1`；Harness eligibility 为 `none`，最多 8 次 checkpoint，restore 上限为 0。
- 禁用臂 guidance：`checkpoint-disabled-v1`；checkpoint/restore 上限均为 0。
- 两臂均使用 Atomic `semantic-v2`、Text-JSON carrier、官方 `deepseek-v4-flash`、相同工具 schema 与相同 frozen teacher1500 数据位置。
- 自主臂不使用 producer quota、不强制首次或后续 commit；是否为简单题、是否存在稳定子结果与后续阶段，完全由模型判断。

## Gate4 结果

冻结位置为 1、14、42、47。每题两臂相邻串行，题间并行；position 1/42 为自主臂先运行，14/47 为禁用臂先运行。

| arm | official correct | strict artifact | legal | turns | provider attempts | provider tokens | checkpoints | restores | tool errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| model-choice | 1/4 | 1/4 | 3/4 | 48 | 50 | 1,061,144 | 0 | 0 | 3 |
| checkpoint-disabled | 3/4 | 1/4 | 4/4 | 47 | 48 | 987,788 | 0 | 0 | 3 |

自主臂比禁用臂多 1 turn、2 次 provider attempt 和 73,356 tokens（约 7.4%）。该差异不能归因于 checkpoint，因为自主臂没有任何 checkpoint action。

逐题内容盲结果：

| position | model-choice | checkpoint-disabled | checkpoint exposure |
|---:|---|---|---|
| 1 | correct / legal / 10 turns / 171,650 tokens | correct / legal / 8 turns / 124,016 tokens | 两臂均 0 |
| 14 | wrong / legal / 9 turns / 188,680 tokens | official correct / legal / 12 turns / 234,179 tokens | 两臂均 0 |
| 42 | 14 turns 后触发 425k token stop / 425,866 tokens | official correct / legal / 15 turns / 430,609 tokens | 两臂均 0 |
| 47 | wrong / legal / 15 turns / 274,948 tokens | wrong / legal / 12 turns / 198,984 tokens | 两臂均 0 |

position 42 两臂都略越过预注册的 425,000 token 上限：自主臂在返回后以 `max_provider_tokens` 停止；禁用臂形成正确语义终止但总计 430,609 tokens。两条 record 的结构与 fresh replay 均通过，但 provider-budget/batch-control audit 按冻结阈值失败。因此整体 audit 为 6/8 通过；record structure、manifest/cohort binding 与 fresh replay 为 8/8 通过。

一次早期 position 14 启动卡在 `/models`/preflight 边界，目录中保持 0 provider attempts、0 tokens。进程终止后该目录改名为 `position_014_infrastructure_stale_preflight_hang` 并排除；正式重跑使用全新 `position_014`，没有混入旧记录。

## 对机制的判断

1. **恢复工具可以退出 forward policy。** 本轮明确将 restore 预算设为 0，未影响 Harness replay；以后可先在 prompt/runtime 中保持禁用，等 checkpoint 本身证明有增益后再做公共 schema 的物理删除。
2. **纯 optional checkpoint 不足。** 模型即使已经产生多个 answer-relevant artifacts、轨迹达到 15 turns，也没有主动 commit。当前 prompt 中的“may commit”没有形成可测 exposure。
3. **不能把 1/4 vs 3/4 解释为 checkpoint 回退。** 没有 episode 实际执行 checkpoint；这是 guidance/sample 对比，不是 checkpoint treatment 对比。
4. **producer quota 也不是最终答案。** 它能保证 exposure，但机械 producer 数不等于语义阶段完成，并在先前 Gate8 中出现过过度压缩或过度抑制。

## 下一步

下一轮应继续保留模型对阶段边界的所有权，但把可选建议改成可审计的模型决策：当模型认为题目是多阶段任务并完成一个稳定子目标时，下一步必须在“commit 当前阶段”与“直接 answer”之间二选一；Harness 只验证 max-8、goal distinct、快照一致性，不用 producer 数决定时机。简单题仍可一路执行并直接 answer。先用同一 Gate4 验证是否形成至少 2/4 checkpoint exposure，再谈准确率扩展。

本报告和所有轨迹继续为 `diagnostic-only`，不得进入 SFT/RL。
