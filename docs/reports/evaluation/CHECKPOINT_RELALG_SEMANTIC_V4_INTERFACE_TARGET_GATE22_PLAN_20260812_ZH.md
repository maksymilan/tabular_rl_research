# Semantic-v3-v24 vs Semantic-v4-v24-interface Target Gate22 plan

日期：2026-08-12
状态：已完成；结果见
`CHECKPOINT_RELALG_SEMANTIC_V4_INTERFACE_TARGET_GATE22_RESULT_20260812_ZH.md`

## 目标与单一修补边界

Gate100 中 semantic-v3-v24 的 100 条轨迹出现 18 个 `invalid_identifier` 与 8 个
`invalid_arguments` 事件，分布在 22 个 task positions。逐事件只读审计发现，18 个
`invalid_identifier` 中绝大多数来自同一接口矛盾：public schema 与 compact contract 允许
`shape_rows` / `rank_select` 的 `as` 省略，但 executor 把省略后继承的合法 dotted/space-containing
逻辑列名再次当作 simple alias 拒绝。

Candidate `semantic-v4-v24-interface` 保留 v3 的工具集合、provider schema、typed operators、
artifact、checkpoint、recent-4 和 exact-answer 语义，只做以下身份绑定修改：

1. `shape_rows` / `rank_select` 省略 `as` 时保留 exact existing logical column name；
2. 显式模型 authored `as` 仍必须是 simple identifier；
3. compact contract 明确 perception/checkpoint 数组与 limit 上限，不增加工具或 prompt 章节。

冻结 `semantic-v3-v24` 的历史执行行为不修改；v4 使用独立 profile/prompt/protocol identity。

## 冻结 cohort

候选完全由 Gate100 record 中的 public error code 选择，不读取 question、gold SQL、gold rows、
reasoning 或 answer：

`[200,204,209,210,212,220,223,228,233,238,242,245,252,253,254,261,263,265,278,282,293,298]`

共 22 个 positions、22 fresh pairs、44 paid episodes。该 target gate 测试接口错误恢复，不作为
无偏总体准确率估计。

## Provider、运行与预算

- official endpoint：`https://api.deepseek.com/chat/completions`
- requested/verified/response model：`deepseek-v4-flash`
- scheme/mode/carrier：checkpoint-relalg-v1 / Atomic / A Text-JSON
- control/candidate：semantic-v3-v24 / semantic-v4-v24-interface
- checkpoint guidance：两臂均为 `model-choice-commit-v6`
- source：teacher1500 v2 nonempty；保持原 position identity
- 每 episode：turns 20、primitive calls 30、checkpoints 8、restores 3
- 每 episode provider：attempts 50、tokens 350,000、wall 3,600 秒
- 44 episodes 总硬上限：15,400,000 provider tokens
- 四个动态 lane；同题两臂相邻，半数 v3->v4、半数 v4->v3；最多四个 episode 同时在途

## 判断与停止

主指标为 target public error events：`invalid_identifier`、`invalid_arguments`、总 errors。并列报告
paired `bird-set`、strict、schema、legal、turns、attempts、tokens、checkpoint coverage/commits。
V4 必须消除 dotted/space implicit-alias 的 Harness rejection，且不能增加相同类别的替代错误。

任一 provider identity/endpoint 漂移、cohort/hash mismatch、structure/fresh replay/no-leak failure
立即停止后续 dispatch。五个连续 semantic failures、两个连续 provider failures、三个总 provider
failures或单 episode cap 由 runner fail closed。所有产物保持 diagnostic-only；SFT/RL admission=0。
