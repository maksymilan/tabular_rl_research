# Semantic-v2 vs Semantic-v3-v24 paired Gate100 result

日期：2026-08-11  
状态：completed diagnostic；无 SFT/RL 准入

## 1. 问题与边界

本实验验证：在完全相同的 Atomic semantic-v2 executable tool/schema、typed Harness、artifact、
checkpoint 和 verifier 语义下，采用借鉴 historical Atomic version24 的紧凑操作契约、精确输出
约束与 recent-4 因果 provider history，能否在不损害行为的情况下减少当前 semantic-v2 的决策
上下文与 token 成本。

candidate `semantic-v3-v24` 是一个 package change；本 gate 不把 compact contract 与 recent-4
拆成两个单变量，也不把不同 cohort 上的 historical version24 准确率直接并入比较。Gold SQL、
gold result 和未来轨迹均未进入 provider 请求；报告不输出题目、数据库值或 reasoning。

## 2. 冻结配置

- provider：official DeepSeek `https://api.deepseek.com/chat/completions`
- model：`deepseek-v4-flash`
- scheme/mode/carrier：`checkpoint-relalg-v1` / Atomic / A Text-JSON
- arms：`semantic-v2` vs `semantic-v3-v24`
- checkpoint guidance：两臂均为 `model-choice-commit-v6`
- source：teacher1500 v2 nonempty，positions 200–299，100 个一一配对任务
- Gate20：positions 200–219；expansion：positions 220–299
- max model turns 20、primitive calls 30、checkpoints 8、restores 3
- metric：`bird-set` 为主，strict artifact、schema match、legal 为并列诊断指标

Expansion 预注册与预算见
`CHECKPOINT_RELALG_SEMANTIC_V3_V24_PAIRED_GATE100_EXPANSION_PLAN_20260811_ZH.md`。

## 3. Expansion 80 pairs

| 指标 | semantic-v2 | semantic-v3-v24 | 差异 |
|---|---:|---:|---:|
| `bird-set` correct | 55/80 | 56/80 | +1 |
| strict artifact | 32/80 | 32/80 | 0 |
| schema match | 38/80 | 40/80 | +2 |
| legal | 77/80 | 77/80 | 0 |
| model turns | 638 | 629 | -9 |
| primitive calls | 636 | 627 | -9 |
| public tool errors | 46 | 43 | -3 |
| checkpoint episode coverage | 43/80 | 40/80 | -3 |
| accepted checkpoints | 47 | 49 | +2 |
| restores | 0 | 0 | 0 |
| provider attempts | 660 | 648 | -12 |
| provider tokens | 11,213,982 | 4,278,449 | -61.85% |

Expansion correctness 配对为：both correct 54、v2-only 1、v3-only 2、both wrong 23，exact
McNemar `p=1.0`。80 个配对中 v3 有 78 个 token 更低、2 个更高；中位每题差值为
`-64,693` tokens。

## 4. Combined Gate100

| 指标 | semantic-v2 | semantic-v3-v24 | 差异 |
|---|---:|---:|---:|
| `bird-set` correct | 69/100 | 70/100 | +1 |
| strict artifact | 41/100 | 41/100 | 0 |
| schema match | 49/100 | 51/100 | +2 |
| legal | 97/100 | 97/100 | 0 |
| model turns | 798 | 780 | -18 (-2.26%) |
| primitive calls | 796 | 778 | -18 |
| semantic atomic calls | 374 | 408 | +34 |
| public tool errors | 57 | 52 | -5 |
| checkpoint episode coverage | 55/100 | 52/100 | -3 |
| accepted checkpoints | 60 | 61 | +1 |
| restores | 0 | 0 | 0 |
| provider attempts | 822 | 799 | -23 |
| prompt tokens | 13,128,559 | 4,635,827 | -64.69% |
| completion tokens | 567,335 | 523,949 | -7.65% |
| total provider tokens | 13,695,894 | 5,159,776 | -62.33% |

正确率配对为：both correct 68、v2-only 1、v3-only 2、both wrong 29；exact McNemar
`p=1.0`。因此 +1 是描述性净增益，不是显著准确率提升。Strict 配对为 2 gains/2 regressions；
schema 为 3 gains/1 regression；legal 为 1 gain/1 regression。100 个配对中 v3 有 98 个 token
更低、2 个更高，中位每题节省 67,731 tokens，范围为节省 623,807 到增加 27,841。

## 5. 错误与机制

v3 总 public errors 从 57 降至 52。值得保留的变化是：

- `invalid_text_json_content`：11 -> 1；
- `unknown_column`：6 -> 3；
- `canonical_type_mismatch`：12 -> 11；
- `invalid_identifier`：5 -> 18；
- `invalid_arguments`：2 -> 8。

紧凑契约显著减少载体错误和部分列引用错误，但 identifier/argument 约束仍是当前主要接口缺口，
应该作为下一轮定向修补对象，而不是重新扩张 prompt。

Checkpoint 没有因 recent-4 消失：coverage 55/100 对 52/100，accepted commits 60 对 61。
Candidate 更集中地在少数 episode 多次 commit，但总体数量几乎一致。两臂 restore 都为 0，因而
本 gate 只能证明 checkpoint 路径被保留和实际使用，不能提供 restore 有效性证据。Checkpoint
正确率贡献也没有在本 package gate 中被单独识别；需要另做同一 v3 contract 下 checkpoint-on
vs checkpoint-off 的 fresh paired ablation。

## 6. 审计与费用边界

- Expansion 8/8 shards 均 completed，无 batch stop。
- 160/160 expansion records 的 current structure、manifest/cohort/provider identity、batch
  budget 和 fresh replay 均通过，issue count 为 0。
- 所有有 response envelope 的 attempts 均为 `deepseek-v4-flash`；没有 Pro 或 fallback 模型。
- Expansion 实际使用 15,492,431 tokens，低于预注册 19,000,000 上限。
- Gate100 两臂合计 18,855,670 tokens；该数字包含先前 Gate20，不是本次 expansion 的新增费用。
- Expansion 生成 160 条可重放诊断记录；SFT/RL retained count 仍为 0。

## 7. 结论

Gate100 支持把 `semantic-v3-v24` 保留为下一代 **diagnostic Atomic interface candidate**：它在
100 个 fresh paired tasks 上保持了相同量级的正确率、strict、legal 与 checkpoint 使用，同时把
总 token 降低 62.33%，并减少 turns、attempts 和 public errors。

这个结果不支持“v3 准确率显著更高”，也不支持 SFT/RL promotion。下一步最有价值的工作是：

1. 在 v3 内定向修补 `invalid_identifier` 和 `invalid_arguments`，保持 compact contract；
2. 用相同 v3 interface 做 checkpoint-on/off fresh paired ablation，隔离 checkpoint 对长 Atomic
   轨迹的真实因果贡献；
3. 若继续扩大行为评估，使用新的 disjoint cohort，并保持 provider、模型、预算、checkpoint
   guidance 和 schema identity 不变。

## 8. 产物

- Gate20：`data/results/checkpoint_relalg_semantic_v2_v3_v24_gate20_20260811/`
- Expansion：`data/results/checkpoint_relalg_semantic_v2_v3_v24_gate100_expansion_20260811/`
- 每个 shard 均保存 manifest、all/success/failure JSONL、batch status 与 current audit。
