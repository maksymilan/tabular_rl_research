# Semantic-v2 vs Semantic-v3-v24 fresh paired Gate20 result

日期：2026-08-11

结论：**semantic-v3-v24 在 20/20 task pairs 上逐题保持 semantic-v2 的 correct、strict、
schema 和 legal outcome，同时把总 provider tokens 降低 64.49%。保留 v3 作为下一阶段
diagnostic Atomic 默认候选，但不据此获得 SFT/RL 准入或宣称准确率提升。**

## 1. 冻结设置

- repository HEAD：`38ff631ea2fef986e8822a3aaf7202ba19feda54`
- branch/upstream：`codex/checkpoint-relalg-v1` / `origin/master`，fresh fetch 后 ahead 146、behind 0
- provider：official `https://api.deepseek.com/chat/completions`
- model：requested、verified 与全部 313 个 response-bearing attempts 均为 `deepseek-v4-flash`
- cohort：teacher1500 v2 nonempty positions 200–219，共 20 task pairs
- source SHA256：`9c4995daf81f94053931fbdaa7bd11beef9157909bacde03be6cb61904952264`
- 共同设置：Atomic、A/Text-JSON、`model-choice-commit-v6`、同一 semantic-v2 executable schema
- control：`semantic-v2`；candidate：`semantic-v3-v24`
- 四个 10-task shards 并发；每 shard 3.25M token cap，总预注册上限 13M

完整预注册见
`CHECKPOINT_RELALG_SEMANTIC_V3_V24_PAIRED_GATE20_PLAN_20260811_ZH.md`。

## 2. Paired 行为结果

| 指标 | semantic-v2 | semantic-v3-v24 | paired 变化 |
|---|---:|---:|---:|
| `bird-set` correct | 14/20 | 14/20 | v2-only 0，v3-only 0 |
| legal termination | 20/20 | 20/20 | 20/20 相同 |
| strict artifact | 9/20 | 9/20 | v2-only 0，v3-only 0 |
| schema match | 11/20 | 11/20 | v2-only 0，v3-only 0 |
| model turns | 160 | 151 | -9 / -5.6% |
| primitive calls | 160 | 151 | -9 |
| atomic calls | 73 | 79 | +6 |
| tool/process errors | 11 | 9 | -2 |
| checkpoint tasks | 12/20 | 12/20 | 相同 coverage |
| accepted commits | 13 | 12 | -1 |
| restores | 0 | 0 | 0 |

correct、strict、schema match 四个 task-level outcome vectors 都逐题完全相同；因此该小样本没有
准确率 gain 或 regression，McNemar discordant pairs 为 0。v3 的 turns 在 9 题更少、6 题相同、
5 题更多；checkpoint 在 17 题相同，两题更少、一题更多。recent-4 没有消除 checkpoint
exposure，也没有诱发 restore。

错误方面，v3 消除了 v2 的 3 个 `unexpected_field` 和 1 个 `invalid_shape`；
`result_too_large` 从 2 降到 1。它仍有 `canonical_type_mismatch=1`、`unknown_column=1`，并有
`invalid_identifier=4`、`invalid_arguments=2`，说明 compact contract 修复了部分参数形状问题，
但没有解决所有列/标识选择错误。

## 3. Token 与请求成本

| provider usage | semantic-v2 | semantic-v3-v24 | 变化 |
|---|---:|---:|---:|
| prompt tokens | 2,400,270 | 807,199 | **-66.37%** |
| completion tokens | 81,642 | 74,128 | -9.20% |
| reasoning tokens | 74,077 | 67,397 | -9.02% |
| total tokens | 2,481,912 | 881,327 | **-64.49%** |
| provider attempts | 162 | 151 | -11 / -6.8% |
| summed episode elapsed | 797.16s | 746.12s | -6.4% |

v3 在 **20/20** 题上 total tokens 都低于 v2；paired 中位差为 -69,950.5 tokens，范围为
-169,065 到 -36,749。v3 的 cache-hit token 比例较低，但更短的首轮 prompt 与 recent-4
历史仍使总 prompt tokens 大幅下降。这说明收益不是依赖更高 cache 命中率。

40 个 episodes 实际只使用 3,363,239 tokens，远低于 13M 预注册上限；四个 shard 均正常
`completed`，没有 provider failure、model identity drift、budget stop 或 fallback。

## 4. 审计与解释边界

四个 shard 各自 10/10 records 通过：manifest/cohort binding、provider identity、batch budget、
structure、causal history、no-leak 与 fresh replay，合计 40/40，issue count 均为 0。结果目录：

- `data/results/checkpoint_relalg_semantic_v2_v3_v24_gate20_20260811/semantic-v2/`
- `data/results/checkpoint_relalg_semantic_v2_v3_v24_gate20_20260811/semantic-v3-v24/`

这是 compact operational contract + recent-4 history 的 package comparison，不能拆分两项各自
贡献。Gate20 支持“在不改变 observed task outcomes 的情况下显著修补 token/shape-error 差距”，
但样本不足以证明 v3 提高准确率，更不能把 trajectory 加入 SFT/RL。下一步若扩大，应保持同一
profile identity，在 disjoint cohort 上验证 outcome preservation、错误稳定性和 token reduction；
只有 package gate 稳定后才值得做 compact-prompt 与 recent-4 的单变量拆分。
