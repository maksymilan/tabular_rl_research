# Semantic-v4-interface vs Semantic-v5-output-label Gate6 result

日期：2026-08-12
状态：completed diagnostic；gate failed，不扩大

## 结果

| 指标 | v4 | v5 |
|---|---:|---:|
| `bird-set` | 5/6 | 4/6 |
| strict artifact | 0/6 | 4/6 |
| schema match | 1/6 | 4/6 |
| legal | 6/6 | 5/6 |
| turns | 62 | 64 |
| errors | 5 | 6 |
| provider attempts | 66 | 68 |
| tokens | 600,100 | 642,951 |
| checkpoint coverage / commits | 4/6 / 4 | 4/6 / 4 |

V5 明显修复了 output label：strict 四个 paired gain、零回退；schema 三个 gain、零回退，且
`invalid_identifier=0`。但它没有通过预注册 gate：一条 provider failure 造成 correct 与 legal
各回退一题；errors 也高一。两个 `unexpected_field` 是 checkpoint 多写 `artifact_handle` 与把
`tables` 放在 action 顶层，与标签规则无关。另一次含空格 alias 仍违反 simple-identifier 约束，
说明继续加同类提示词收益有限。

12/12 records 的 identity、structure、batch/cohort 与 fresh replay 全部通过；总计 1,243,051
tokens，低于 4.2M 上限，全部 response model 为 official `deepseek-v4-flash`。

## 决策

冻结 v5，不扩大、不替代 v3、不进入 SFT/RL。保留两项工程结论：省略 `as` 必须合法保留 exact
logical name；终端 join role prefix 应由模型显式去掉。下一实验回到核心 checkpoint 因果问题，
使用当前 v3 compact interface 做 fresh checkpoint-on/off paired ablation。

产物：`data/results/checkpoint_relalg_semantic_v4_v5_output_label_gate6_20260812/`
