# Semantic-v2 vs Semantic-v3-v24 paired Gate100 expansion plan

状态：2026-08-11 已完成。该扩展与 positions 200–219 Gate20 合并后形成 100 task pairs；
本次新调用仅覆盖 disjoint positions 220–299。结果见
`CHECKPOINT_RELALG_SEMANTIC_V3_V24_PAIRED_GATE100_RESULT_20260811_ZH.md`。

## 冻结身份

- provider：official DeepSeek `https://api.deepseek.com/chat/completions`
- model：`deepseek-v4-flash`，requested、verified 与 response identity 必须精确一致
- scheme/mode/carrier：`checkpoint-relalg-v1` / Atomic / A Text-JSON
- checkpoint guidance：两臂均为 `model-choice-commit-v6`
- control：`semantic-v2`；candidate：`semantic-v3-v24`
- source：`data/eval_inputs/bird_train_atomic_teacher1500_v2_nonempty.jsonl`
- source SHA256：`9c4995daf81f94053931fbdaa7bd11beef9157909bacde03be6cb61904952264`
- source manifest SHA256：`4c6b96e523d02b45c01eaa60df627abac4a6c3720a89f6573f3eec92740537fe`
- expansion positions：220–299，80 task pairs、160 paid episodes；不与 Gate20 重叠

## 并发与预算

runner 内部保持 `workers=1`。每臂拆成 start=220/240/260/280 的四个 n=20 shards，八个独立
进程并发；单 episode 内仍严格因果串行。

- 每 episode：max model turns 20、primitive calls 30、checkpoints 8、restores 3
- provider retry 4；initial completion 4096、maximum completion 8192
- v2：每 shard attempts 300、tokens 3,250,000；四 shard 合计 13,000,000
- v3：每 shard attempts 300、tokens 1,500,000；四 shard 合计 6,000,000
- 本次 expansion 总硬上限：19,000,000 provider tokens
- 每 shard wall 7,200 秒；连续 provider failures 2、总 provider failures 3、连续 semantic
  failures 5。任何 shard stop 不把剩余额度转给新 cohort。

## 判断

先独立报告 expansion 80 pairs，再与冻结 Gate20 合并成 100 pairs。主指标为 paired `bird-set`、
strict artifact、schema match 与 legal；效率统计 prompt/cache/completion/total tokens、attempts、
turns；可靠性统计所有 public error codes；机制统计 checkpoint coverage/commits/restores。

每个 shard 必须通过 current manifest/cohort/provider identity、batch budget、structure、causal
history、no-leak 与 fresh replay。该 package gate 不拆分 compact contract 与 recent-4 的贡献，
不允许 verifier-selected retry，也不产生任何 SFT/RL 准入。

## 完成状态

- 8/8 shards、160/160 expansion records 完成；没有 batch stop。
- 8/8 独立 current audit 通过，160/160 structure 与 fresh replay 通过且 issue count 为 0。
- expansion 实际使用 15,492,431 provider tokens，低于预注册 19,000,000 上限。
- 合并 Gate100：semantic-v2 为 69/100，semantic-v3-v24 为 70/100；paired exact
  McNemar `p=1.0`，不构成准确率提升证据。v3 总 tokens 为 5,159,776，对照为
  13,695,894，降低 62.33%。
