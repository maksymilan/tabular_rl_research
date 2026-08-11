# Semantic-v2 vs Semantic-v3-v24 fresh paired Gate20

状态：2026-08-11 预注册，尚未产生结果。

## 冻结身份

- provider：official DeepSeek `https://api.deepseek.com/chat/completions`
- requested model：`deepseek-v4-flash`；每个 response 必须精确匹配
- scheme/mode/carrier：`checkpoint-relalg-v1` / Atomic / A Text-JSON
- checkpoint guidance：两臂均为 `model-choice-commit-v6`
- control：`atomic_operator_profile=semantic-v2`
- candidate：`atomic_operator_profile=semantic-v3-v24`
- source：`data/eval_inputs/bird_train_atomic_teacher1500_v2_nonempty.jsonl`
- source SHA256：`9c4995daf81f94053931fbdaa7bd11beef9157909bacde03be6cb61904952264`
- source manifest SHA256：`4c6b96e523d02b45c01eaa60df627abac4a6c3720a89f6573f3eec92740537fe`
- frozen positions：200–219，20 个 task pairs、40 个付费 episodes

## 运行预算

- runner 强制 `workers=1`，因此每臂拆成两个不重叠的 10-task shard：200–209 与 210–219；
  四个独立进程并发，总计最多四个 episode 同时在途，单 episode 内仍严格因果串行
- 每 episode：model turns 20、primitive calls 30、checkpoints 8、restores 3
- provider retry 4；initial completion 4096，最大 completion 8192
- 每 shard provider attempts 上限 500、tokens 上限 3,250,000、wall 上限 7,200 秒；两 shard
  合计仍为每臂 1000 attempts / 6,500,000 tokens
- 总 nominal 硬上限 13,000,000 provider tokens；任一臂停止后不得把余额转给另一新 cohort
- 连续 provider failures 2、总 provider failures 3、连续 semantic failures 5

## 判断口径

主指标是相同 task identity 上的 paired `bird-set` correct、legal termination、strict artifact。
可靠性分别统计 carrier、argument/schema、type、state/execution errors。效率统计 model turns、
provider attempts、prompt/cache/completion/total tokens。checkpoint coverage、accepted commits 与
阶段长度作为机制描述，不用于事后选择获胜臂。

所有 records 必须通过当前 structure、manifest/cohort binding、provider identity、causal history、
no-leak 和 fresh replay。该 Gate20 是完整修补 package（compact contract + recent-4 history）的
诊断，不把两项的贡献拆开。无论结果如何，产物保持 diagnostic-only，不进入 SFT/RL。
