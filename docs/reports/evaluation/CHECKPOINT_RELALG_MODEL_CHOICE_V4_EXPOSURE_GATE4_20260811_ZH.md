# Checkpoint-RelAlg Atomic Model-Choice v4 Exposure Gate4（2026-08-11）

## 结论

`model-choice-commit-v4` 在 4 个全新 Atomic `semantic-v2` 诊断任务上产生 **0 次 checkpoint 尝试和 0 次 accepted commit**，未达到预注册的 `coverage >= 2/4` 暴露门，因此没有启动 disabled 对照臂。v4 将 v3 的强制感知条件缩减为“未验证错误”和“仅剩一个关系算子”两条否决，但仍然过度抑制模型使用 checkpoint；这不是 Harness 拒绝 checkpoint，因为 Harness eligibility 仍为 `none`，模型根本没有调用。

本结果只说明 v4 prompt manipulation 未能提供机制暴露，不证明 checkpoint 本身无效。该批次及后续版本均保持 diagnostic-only，不进入 SFT/RL。

## 冻结设置

- 仓库 commit：`69b8442b1c825d70d5d5110273b8ad19954aa661`
- provider：official DeepSeek `https://api.deepseek.com/chat/completions`
- model：`deepseek-v4-flash`
- mode/profile：Atomic / `semantic-v2`
- carrier：A / Text-JSON
- guidance：`model-choice-commit-v4`
- Harness checkpoint eligibility：`none`
- `max_checkpoints=8`，`max_restores=0`
- positions：`46, 63, 67, 79`
- 单题 provider token cap：425,000；候选臂名义总上限：1,700,000
- 选择与门槛：见结果根目录的 `preregistered_plan.json`

## 结果

| 指标 | v4 candidate |
|---|---:|
| episodes | 4 |
| bird-set correct | 2/4 |
| strict artifact | 1/4 |
| legal termination | 3/4 |
| model turns / primitive calls | 36 / 35 |
| provider attempts | 39 |
| provider tokens | 740,755 |
| tool errors | 6 |
| checkpoint coverage | 0/4 |
| checkpoint attempts / accepted commits | 0 / 0 |
| restore | 0 |

逐题公开结果：position 46 合法错误；63 正确且 strict；67 bird-set 正确但 strict/schema 不匹配；79 在 11 turns 后 provider failure。没有读取或公开题目、gold SQL、gold rows、模型 reasoning 或 checkpoint goal。

## 审计

- 4/4 requested/response identity 为 `deepseek-v4-flash`，endpoint 为 official DeepSeek。
- 4/4 manifest binding、cohort identity、record structure、provider budget、batch control 与 fresh replay 通过。
- 所有结果目录独立；未覆盖、续跑或修改历史 v2/v3 artifact。

## 结构性观察与下一步

无内容动作序列显示，至少两个任务形成了可复用中间关系并继续执行多个关系决策，但 v4 仍未 commit。与能产生 5/8 checkpoint coverage 的 v2 相比，v4 将强正向 `MUST commit` 条件拆成更长的否定例外，并把后续 join/aggregate/rank/scalar 等具体工作泛化成模糊的 “substantial distinct relational work”。

下一版 `model-choice-commit-v5` 只改教师提示，不改工具、Harness 或 eligibility：模型在起始 reasoning 自主选定一个可复用语义阶段；该 relation 已存在且仍有至少两个不同关系决策时，下一动作必须 commit；一次错误只在尚无后续成功 producer/observation 时短暂否决；仅剩一个关系算子或答案已就绪时直接结束。restore 继续禁用。v5 必须先通过独立 checkpoint 暴露门，再启动付费对照。

## Artifact

- 根目录：`data/results/checkpoint_relalg_v1_flash_text_json_atomic_model_choice_v4_exposure_then_pair_gate4_20260811/`
- v4：`model_choice_v4/position_{046,063,067,079}/`
- 预注册：`preregistered_plan.json`

