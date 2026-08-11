# Checkpoint-RelAlg Atomic Model-Choice v5 vs Disabled Gate4（2026-08-11）

## 结论

`model-choice-commit-v5` 恢复了模型自主 checkpoint 暴露：4 个新任务中 2 题使用 checkpoint，共 5 次 accepted commit；Harness eligibility 仍为 `none`，所以时机完全由模型选择。候选臂得 1/4 `bird-set`，disabled 得 2/4，配对为 both correct 1、v5-only 0、disabled-only 1、neither 2。该样本不支持 checkpoint 准确率收益。

v5 总 turns/tokens 低于对照，但这是 position 96 两臂不同失败长度造成的假象。在真正使用 checkpoint 的 positions 50/73 上，v5 为 0/2、对照 1/2；v5 使用 21 turns、333,456 tokens，对照 19 turns、326,402 tokens。checkpoint 子集同时出现一个准确率回退、+2 turns、+2.2% tokens。

结构审计发现主要问题不是首次 checkpoint，而是后续 checkpoint 过密：position 73 有一次连续 `commit_checkpoint -> commit_checkpoint`，中间没有生产新 relation；position 50 的第二次 commit 之后只剩三次工具动作，phase reset 无法摊销自身成本。下一步只收紧模型对第二次及以后 checkpoint 的策略，不恢复 Harness producer gate。

## 冻结设置

- implementation commit：`ef5d3a0454c80328436ef00a53c25cd8df14e7bf`
- provider/model：official DeepSeek / `deepseek-v4-flash`
- mode/profile：Atomic / `semantic-v2`
- carrier：A / Text-JSON
- positions：`50, 71, 73, 96`
- v5：`model-choice-commit-v5`，max checkpoints 8，restore 0，eligibility `none`
- control：`checkpoint-disabled-v1`，max checkpoints/restore 0
- 每题 provider token cap：500,000；完整 8 episodes 名义上限 4,000,000
- cohort 仅按冻结 Atomic Prefix100 的历史 `model_turns` 降序、position 稳定 tie-break 选择；没有读取 question、gold、reasoning、goal 或 outcome。

## 汇总

| 指标 | v5 | disabled |
|---|---:|---:|
| correct | 1/4 | 2/4 |
| strict artifact | 0/4 | 0/4 |
| legal | 3/4 | 3/4 |
| model turns | 42 | 48 |
| provider attempts | 46 | 54 |
| provider tokens | 845,829 | 1,123,950 |
| tool errors | 2 | 4 |
| checkpoint coverage | 2/4 | 0/4 |
| accepted commits | 5 | 0 |
| restore | 0 | 0 |

position 96 的 v5 是 provider failure（211,775 tokens），disabled 是 batch token stop（546,705 tokens），因此全体效率差异不可作为 checkpoint 节省证据。其余三题均合法终止。

## 阶段结构

- position 50：phase lengths `4/2/3`；两次 commit 分别在 filter+read 后、单次 join 后；最终错误。
- position 73：phase lengths `4/1/2/5`；第二个 phase 只有一次 commit，形成零进度连续 checkpoint；最终错误。disabled 在同题正确。
- position 71/96：v5 均未调用 checkpoint。

Text-JSON 每个新 phase 都要重新发送 system prompt 与完整工具 schema。checkpoint 会降低后续历史长度，但自身也占一轮；短 phase 或连续 commit 会净增加成本，并且把完整 provider tool-result 历史替换为 compact environment/checkpoint summary。当前证据支持保留首次语义阶段选择，同时对后续 checkpoint 增加更强的模型侧摊销判断。

## 审计

- 8/8 requested/response model identity 为 Flash，endpoint 为 official DeepSeek。
- 8/8 record structure、manifest binding、cohort identity 与 fresh replay 通过。
- 7/8 overall audit 通过；唯一失败是 disabled position 96 达到 batch token stop。其 record structure 与 fresh replay 仍通过。
- 无 restore；无 Harness eligibility rejection；5 次 commit 全部 accepted。

## 下一版边界

`model-choice-commit-v6` 保留 v5 的首次 checkpoint 正向条件。第二次及以后：禁止连续 commit；上次 checkpoint 后必须先有新的成功 relation-producing 动作；模型还必须预计至少四个工具动作才允许再次 checkpoint。该规则只在 teacher prompt 中执行，Harness eligibility 继续为 `none`，max checkpoints 继续为 8，restore 继续禁用。

## Artifact

- `data/results/checkpoint_relalg_v1_flash_text_json_atomic_model_choice_v5_exposure_then_pair_gate4_20260811/`
- 候选：`model_choice_v5/position_{050,071,073,096}/`
- 对照：`checkpoint_disabled/position_{050,071,073,096}/`
- 预注册：`preregistered_plan.json`

