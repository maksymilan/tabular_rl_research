# Semantic-v3-v24 vs Semantic-v4-v24-interface Target Gate22 result

日期：2026-08-12
状态：completed diagnostic；不扩大、不准入 SFT/RL

## 1. 结论摘要

窄接口修复有效，但完整 candidate 不能直接推广。

- fresh v3 的 target errors 为 `invalid_identifier=7`、`invalid_arguments=5`；
- v4 为 `invalid_identifier=0`、`invalid_arguments=1`；
- 总 public tool errors 从 34 降到 19，turns 从 231 降到 204；
- `bird-set` 完全持平 12/22，逐题 outcome 22/22 相同；
- 但 legal 从 21/22 降到 19/22，strict 从 10/22 降到 8/22，schema 从 12/22 降到 9/22。

因此，`preserve exact logical column when as omitted` 应保留为正确的 Harness 修复方向；当前
v4 prompt package 则冻结为 target diagnostic，不能替代 v3，也不应继续扩大。

## 2. 冻结身份

- provider：official `https://api.deepseek.com/chat/completions`
- model：44/44 manifest 与所有 response identity 均为 `deepseek-v4-flash`
- scheme/mode/carrier：checkpoint-relalg-v1 / Atomic / A Text-JSON
- control/candidate：semantic-v3-v24 / semantic-v4-v24-interface
- checkpoint guidance：两臂均为 `model-choice-commit-v6`
- cohort：由 Gate100 public `invalid_identifier|invalid_arguments` 选出的 22 positions
- candidate tool schema 与 v3 完全相同；prompt/protocol/output-name-policy identity 独立

完整预注册见
`CHECKPOINT_RELALG_SEMANTIC_V4_INTERFACE_TARGET_GATE22_PLAN_20260812_ZH.md`。

## 3. 指标

| 指标 | v3 | v4 | 差异 |
|---|---:|---:|---:|
| `bird-set` correct | 12/22 | 12/22 | 0 |
| strict artifact | 10/22 | 8/22 | -2 |
| schema match | 12/22 | 9/22 | -3 |
| legal | 21/22 | 19/22 | -2 |
| model turns | 231 | 204 | -27 |
| primitive calls | 230 | 202 | -28 |
| public tool errors | 34 | 19 | -15 |
| checkpoint coverage | 12/22 | 17/22 | +5 |
| accepted commits | 13 | 17 | +4 |
| restores | 0 | 0 | 0 |
| provider attempts | 239 | 219 | -20 |
| provider tokens | 1,910,450 | 1,660,318 | -13.09% |

Correctness 22/22 均为 both-correct 或 both-wrong。Strict 为 v3-only 3、v4-only 1；schema 为
v3-only 3、v4-only 0；legal 为 v3-only 2、v4-only 0。22 对中 v4 有 16 题 token 更低、6 题
更高，中位每题差值 `-7,721` tokens。

## 4. 接口目标是否达成

达成。Fresh v3 的七个 `invalid_identifier` 均为 `shape_rows` / `rank_select` 省略 `as` 后，
executor 错把已有 dotted/space logical column 当作 simple alias。V4 对同类调用全部接受，且没有
转化为 `unknown_column`、duplicate output 或 reserved-name error。

`invalid_arguments` 从五次降为一次。剩余一次与 implicit alias 无关。V4 compact contract 对
`read_rows.limit<=20`、`inspect_column.top_k<=50`、checkpoint arrays 和 explicit simple alias 的
约束也更明确。

## 5. 回退分析

V4 的两个 non-legal task 分别是：一个到 20 turns 未 terminal，另一个是 provider failure；都不
是新 output-name executor 的结构拒绝。三个 strict/schema 回退中，有两个任务在 v3 曾先触发
`invalid_identifier` 后修正为显式 alias，而 v4 直接保留 dotted logical label。其 row/value
denotation 仍正确，但 exact output schema 不匹配。这揭示了真实语义张力：

- **省略 `as` 保留 exact logical name** 是 schema/runtime 一致性的正确默认；
- 但终端 exact schema 有时需要模型主动选择 reference-compatible simple alias；
- 只消除 rejection 不会自动教会模型何时保留 dotted label、何时做语义重命名。

另一个 strict/schema 回退在两臂都有相同工具路径、无 target error，是 provider 行为方差，不能
归因给 executor 修复。V4 checkpoint coverage 提升也说明 prompt 的细微变化改变了模型 staging，
所以该 target gate 不能作为单纯 executor 单变量准确率结论。

## 6. 审计与预算

- 44/44 batch 完成；44/44 record structure 与 fresh replay 通过，issue count=0。
- 43/44 overall audit 通过。唯一失败是 v3 position 233 的在途最后 response 使 token 从 cap
  350,000 越至 360,848；其 structure/fresh replay 均通过，且没有继续 dispatch。
- 实际两臂合计 3,570,768 tokens，远低于预注册 15,400,000 总硬上限。
- 44 条均为 diagnostic artifacts；SFT/RL retained=0。

## 7. 决策

1. 冻结 v4 target package，不扩大为总体 gate，不替代 v3。
2. 保留并继续测试 output-name Harness 修复，但下一版必须把“保留 exact logical name”和
   “terminal semantic alias”分离：不应重新拒绝 dotted name，也不应靠错误反馈迫使模型猜 alias。
3. 下一次只做小型 output-schema target gate：显式比较保留源标签、请求字段标签与 simple alias
   的可观测契约；不增加 prompt 长度，不改 checkpoint。
4. `semantic-v3-v24` 仍是当前 diagnostic Atomic candidate；无 SFT/RL promotion。

## 8. 产物

`data/results/checkpoint_relalg_semantic_v3_v4_interface_target_gate22_20260812/`
