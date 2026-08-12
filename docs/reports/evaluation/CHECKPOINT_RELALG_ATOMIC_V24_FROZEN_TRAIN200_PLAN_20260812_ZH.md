# Atomic v24 frozen Train200 评测计划

日期：2026-08-12  
状态：已冻结，等待启动

## 目标

在冻结的 BIRD train teacher1500 v2 nonempty 数据上，测量
`atomic-v24-frozen-v1` 的绝对正确率、严格工件正确率、schema match、合法终止率、错误率与成本。
这是 diagnostic-only 行为评测，不构成 SFT/RL 准入。

## 冻结配置

- cohort：`bird_train_atomic_teacher1500_v2_nonempty.jsonl` positions 320--519；连续 200 题；
- dataset SHA256：`9c4995daf81f94053931fbdaa7bd11beef9157909bacde03be6cb61904952264`；
- dataset manifest SHA256：`4c6b96e523d02b45c01eaa60df627abac4a6c3720a89f6573f3eec92740537fe`；
- mode：Atomic；operator profile：`atomic-v24-frozen-v1`；
- carrier：A / Text-JSON；history：recent-4；
- checkpoint guidance：`checkpoint-disabled-v1`；`max_checkpoints=0`；`max_restores=0`；
- provider：官方 `https://api.deepseek.com/chat/completions`；model：`deepseek-v4-flash`；
- metric：主指标 `bird-set`；secondary 为 strict artifact、schema match、legal；
- 每题最多 20 model turns、30 primitive calls；provider retry 4；
- 四个互不重叠的 50 题分片：320--369、370--419、420--469、470--519；每个进程串行，四进程并行；
- 每分片最多 600 provider attempts、7,500,000 tokens、7,200 秒；四分片 nominal cap 30,000,000 tokens，另留 2,000,000 tokens 给在途响应越界，整个批次停止线 32,000,000 tokens；
- provider failures 连续 2 / 总计 3 时停止；semantic failure 不用于提前截断完整固定 cohort。

## 冻结身份

- tool schema SHA256：`bc997223394700b9f87903b9ccd4497976246e801b3d8a493044641d8409f926`；
- student prompt SHA256：`dbba671ce36c8e9c131236002b39b6a2d25a073527c119409e9f4abacfb3f54d`；
- teacher prompt SHA256：`947445f4959d3ee9201742495af664caa7f13a36197817411224058767b4deda`；
- carrier protocol SHA256：`05716470b0865cbb20bca360b42afa35b4bcd607499e5a3104a7ef0ff7d69313`；
- implementation commit：`4ef19f3290a3c57f1cb8f86a36cd3f24b33123b3`。

## 验收与报告边界

必须报告完整 200 题的 attempted/correct/strict/schema/legal、provider attempts/tokens、公开错误码、
结构审计与 fresh replay。任何非 Flash response、非官方 endpoint、cohort/identity 漂移或 gold/no-leak
违规立即停止。错误轨迹保留为 audit-only，不手工修复，不重试语义失败，不导入训练集。
