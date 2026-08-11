# Checkpoint-RelAlg Atomic Model-Choice v6 vs v5 Gate4（2026-08-11）

## 结论

v6 在四个新任务上保持与 v5 完全相同的结果（2/4 `bird-set`、1/4 strict、4/4 legal），同时把 accepted commits 从 8 降到 4、turns 从 53 降到 44、provider tokens 从 860,854 降到 717,127（−16.7%）。这证明“后续 checkpoint 必须摊销 phase reset”是正确的优化方向。

但 v6 仍有一题出现 `commit_checkpoint -> commit_checkpoint`，中间没有任何新 relation 或观察动作。模型无视了明确的 “Never commit twice consecutively” 指导；因为 Harness eligibility=`none`，第二次 commit 仍被接受。因此 v6 通过行为保持和总体效率，却没有通过预注册的无连续 checkpoint 机制门，不能扩大或与 disabled 做更大比较。

## 冻结设置

- implementation commit：`a416e0900d5b7c5f1007b6f0502de9da5aa08fb3`
- official DeepSeek / `deepseek-v4-flash`
- Atomic `semantic-v2`，A/Text-JSON，restore 0，max checkpoints 8
- positions：`25, 76, 83, 92`
- v6 与 v5 的 Harness eligibility 均为 `none`
- 每题 provider token cap 500,000；8 episodes 名义总上限 4,000,000
- 选择仅依据冻结 Atomic Prefix100 的历史 model-turn 长度；未读取 question、gold、reasoning、goal 或 outcome。

## 结果

| 指标 | v6 | v5 |
|---|---:|---:|
| correct | 2/4 | 2/4 |
| strict artifact | 1/4 | 1/4 |
| schema match | 2/4 | 2/4 |
| legal | 4/4 | 4/4 |
| model turns | 44 | 53 |
| provider attempts | 45 | 53 |
| provider tokens | 717,127 | 860,854 |
| tool errors | 5 | 5 |
| checkpoint coverage | 3/4 | 3/4 |
| accepted commits | 4 | 8 |
| restore | 0 | 0 |

配对结果逐题完全一致：both correct 2、v6-only 0、v5-only 0、both wrong 2。v6 相对 v5 少 9 turns、8 attempts、143,727 tokens；三项 checkpoint-exposed 任务均保持相同 correctness。

## 阶段审计

- position 76：v6 2 commits、v5 3 commits；两臂都正确且 strict。v6 仍出现一次连续 commit，违反 teacher 指导。
- position 83：v6 1 commit、v5 3 commits；两臂都错。v5 出现一次连续 commit，v6 没有。
- position 92：v6 1 commit、v5 2 commits；两臂都正确。v5 出现一次连续 commit，v6 没有。
- position 25：两臂都不 commit、都错。

v6 将连续 checkpoint 从 v5 的两题降到一题，并减少一半 commits，但 prompt 不能保证结构不变量。

## 审计

- 8/8 official endpoint 与 requested/response `deepseek-v4-flash` identity 通过。
- 8/8 record structure、manifest binding、cohort identity、provider budget、batch control 与 fresh replay 通过。
- 无 token stop、provider failure、restore 或 Harness checkpoint rejection。

## 下一步

不再继续堆叠同义 prompt。新增一个 identity-bound 的最小 checkpoint 合法性策略：若自上个 accepted checkpoint/phase 起点以来没有任何成功非控制动作，则拒绝 commit。该规则只排除零进度 phase，不计 producer 数、不指定具体 turn、不替模型判断 relation 是否稳定，也不把 max-8 改为一次；模型仍自主选择所有有进度的阶段边界。用新的稳定 error code 返回后，再做 v7 小样本因果门。

## Artifact

- `data/results/checkpoint_relalg_v1_flash_text_json_atomic_model_choice_v6_vs_v5_gate4_20260811/`
- v6：`model_choice_v6/position_{025,076,083,092}/`
- v5：`model_choice_v5/position_{025,076,083,092}/`
- 预注册：`preregistered_plan.json`

