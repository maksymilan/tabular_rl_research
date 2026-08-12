# Atomic v24 frozen Train200 评测结果

日期：2026-08-12  
状态：完成；diagnostic-only

## 结论

冻结 profile `atomic-v24-frozen-v1` 在 teacher1500 v2 nonempty 的 positions 320--519 上完成
200/200 个官方 DeepSeek v4 Flash causal episode。主指标 `bird-set` 为 **147/200 = 73.5%**
（Wilson 95% CI 66.98%--79.13%）。该结果验证了冻结工具面具备与历史 Atomic v24 相近量级的
绝对正确率，但不是同题 paired comparison，不能据此宣称优于或等于历史 fixed-200 的 145/200。

严格工件指标明显更低：strict artifact **63/200 = 31.5%**，schema match **80/200 = 40.0%**；
legal termination 为 **192/200 = 96.0%**。因此集合值正确不能当作精确输出工件正确，冻结版本仍为
diagnostic-only，SFT/RL admission 保持关闭。

## 冻结身份与执行

- dataset positions：320--519，200 个唯一位置；dataset SHA256
  `9c4995daf81f94053931fbdaa7bd11beef9157909bacde03be6cb61904952264`；
- implementation commit：`4ef19f3290a3c57f1cb8f86a36cd3f24b33123b3`；
- profile：`atomic-v24-frozen-v1`；mode `atomic`；carrier A/Text-JSON；recent-4；
- checkpoint guidance：`checkpoint-disabled-v1`；checkpoint/restore 均为 0；
- requested/response model：所有可解析响应均为 `deepseek-v4-flash`；官方 endpoint；
- tool schema SHA256：`bc997223394700b9f87903b9ccd4497976246e801b3d8a493044641d8409f926`；
- teacher prompt SHA256：`947445f4959d3ee9201742495af664caa7f13a36197817411224058767b4deda`；
- metric：`bird-set`；secondary：strict artifact/schema/legal。

四个 50 题分片并行、每题内部因果串行：

| positions | correct | strict | schema | legal | turns | tool errors | tokens | attempts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 320--369 | 37 | 15 | 20 | 48 | 372 | 22 | 2,458,532 | 381 |
| 370--419 | 35 | 11 | 14 | 48 | 351 | 23 | 2,019,565 | 361 |
| 420--469 | 40 | 18 | 24 | 49 | 348 | 20 | 2,021,803 | 355 |
| 470--519 | 35 | 19 | 22 | 47 | 381 | 23 | 2,844,474 | 395 |
| **合计** | **147** | **63** | **80** | **192** | **1,452** | **88** | **9,344,374** | **1,492** |

平均每题 7.26 model turns、46,721.9 tokens。总 token 包括 8,158,763 prompt tokens 与
1,185,611 completion tokens，其中 reasoning tokens 1,130,010；prompt cache hit/miss 分别为
4,328,576 / 3,830,187。实际成本远低于预注册的 32M 全局停止线。

## 失败与恢复

- 45 条 legal wrong answer；
- 5 条达到 max model turns；
- 3 条 provider completion 在有界扩容重试后仍 `finish_reason=length`；
- 49/200 episode 出现至少一个工具错误，其中 29 条最终恢复正确；
- 88 个工具错误以 canonical type mismatch、invalid arguments、SQL timeout、result too large、
  unexpected/missing field、unknown column 等为主；
- 无 checkpoint、无 restore，符合冻结 public surface。

## 审计

- 200/200 record structure、provider history、manifest binding、cohort identity、provider budget、
  batch control 通过；
- fresh replay **199/200** 通过；147 条正确轨迹全部 fresh replay 通过；
- 唯一 replay failure 为 position 403 的错误/非终止轨迹：live 第四个 `filter_rows` 在 20 秒边界
  返回 `sql_timeout`，fresh replay 同一查询在时限内成功，从而后续状态漂移。这是 wall-time 边界的
  执行非确定性，不是 provider/gold 泄漏，但足以阻止整批无条件准入；该条只能保留为 audit-only。

分片 all.jsonl SHA256：

- 320--369：`9c3c6cfae35a8e5432e30530b0883bba16bcfce2525af68a79e988685dc31f34`；
- 370--419：`c0c549fc22199a5a5e30f3ce83b62c95ff3e23d83a990d60ae5d998804dfd808`；
- 420--469：`7c768f5ab64f5a678ac78e25ec3dfaea41f854703ef607deb58a9907b4a9f2c3`；
- 470--519：`4dce51c3c0e631295228bb3c03c1ba1da23b8809c1925c58ee46a495d8cbd507`。

## 决策

保留 `atomic-v24-frozen-v1` 作为冻结 Atomic 工具基线：它在独立 Train200 上达到 73.5%
`bird-set`，接近历史 v24 的绝对量级，同时完全移除了 checkpoint/restore 实验分支。不得把该结果
解释为同题优越性，也不得把 147 条正确记录直接导入训练。若后续考虑训练准入，先解决 exact output
schema/strict gap，并明确导出只接受正确且 fresh-replay/structure/no-leak 全通过的 scheme-local 轨迹。

