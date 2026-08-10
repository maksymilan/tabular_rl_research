# checkpoint-relalg Atomic checkpoint-effect Gate12

日期：2026-08-10  
状态：diagnostic-only，不进入 SFT/RL

## 目的

本实验检验 checkpoint 是否比持续累积 Atomic 工具历史更有用，而不再只检验模型是否会调用
`commit_checkpoint`。`restore_checkpoint` 保持可用，但不是本实验目标。

## 冻结设计

- 模型：官方 `deepseek-v4-flash`
- 模式：`atomic`
- Atomic profile：`semantic-v2`
- carrier：`text-json`
- 对照：`adaptive-v1`
- 处理：`semantic-milestone-v3`
- cohort：12 个既有长轨迹任务；按旧 rollout 的公开统计字段分层为 6 个既往正确、6 个既往错误
- 每道题两臂均全新运行，共 24 个付费 episode
- 单 episode provider token 闸：400,000；最多 20 model turns、30 primitive calls
- 运行：同题两臂相邻提交，最多 8 个 episode 并行

该 cohort 是针对长轨迹的机制诊断，不代表总体 BIRD 准确率。选题没有读取 question、gold SQL、
gold rows 或模型 reasoning。

## 结果

| 指标 | adaptive-v1 | milestone-v3 |
|---|---:|---:|
| bird-set correct | 5/12 | 6/12 |
| legal termination | 10/12 | 12/12 |
| strict artifact correct | 3/12 | 3/12 |
| model turns | 127 | 134 |
| provider total tokens | 2,572,050 | 2,456,623 |
| tool errors | 9 | 13 |
| checkpoint episodes | 0/12 | 8/12 |
| checkpoint calls | 0 | 10 |
| restore calls | 0 | 0 |

正确率配对表：both correct=5、adaptive-only=0、milestone-only=1、both wrong=6。净增益为
+1/12，但 discordant pair 只有一个，不能据此声称统计显著的准确率提升。

adaptive 的 position 42 和 54 在一次在途响应后分别累计 409,419 和 441,711 tokens，按冻结的
400,000 token 闸停止。两条记录自身的结构审计和 fresh replay 均通过；目录总体审计按设计因超预算
标为失败。其余 22 个目录总体审计通过。

## checkpoint 的机械效果

milestone-v3 实际 commit 的 8 个任务中：

- adaptive：92 turns，1,964,865 tokens，平均 21,357 tokens/turn
- milestone-v3：102 turns，1,834,198 tokens，平均 17,982 tokens/turn
- milestone-v3 多执行 10 turns，但总 token 少 130,667（-6.7%）
- 每 turn token 少 15.8%

这表明 checkpoint 后清理旧 phase history 的机制确实在降低后续上下文成本，并非只增加一个控制
工具。全 12 对合计 token 下降 115,427（-4.5%），turns 增加 7（+5.5%）。

## 触发策略问题

`semantic-milestone-v3` 尚未稳定满足“每 episode 最多一次 commit”：

- 6/12 恰好一次 commit
- 2/12 两次 commit
- 4/12 没有 commit

两次 commit 分别出现在 position 42 和 54。模型可见提示虽然要求检查 checkpoint history 并禁止
第二次 commit，但提示约束仍会被忽略。因此不能把当前 profile 直接扩大为正式主实验。

## 判断

1. **checkpoint 状态压缩机制有效。** commit 后 phase reset 显著降低每 turn 上下文 token，并帮助
   两条对照超预算的长轨迹在处理臂内完成。
2. **准确率增益尚未证实。** 本 Gate12 只有 1 gain / 0 regression，样本不足，且处理臂工具错误更多。
3. **当前主要瓶颈是触发控制，而不是 checkpoint store。** restore 未被调用不影响本结论；本实验
   只检验 commit。
4. 下一步应把“最多一次 commit”变成 Harness 可执行约束或动态 capability，而不是继续依赖提示；
   完成后再用同一冻结 cohort 做 K>=2 配对复跑，再决定是否扩到更大长轨迹集合。

## Artifact

结果根目录：
`data/results/checkpoint_relalg_v1_flash_text_json_atomic_semantic_v2_checkpoint_effect_gate12_20260810/`

