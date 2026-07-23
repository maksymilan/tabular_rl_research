# BIRD version19 固定首 30 题 gate

日期：2026-07-23

## 目的与控制

version19 在针对性小样本通过后，用固定 200 题 cohort 的前 30 题做扩展 gate。除协议版本
外，控制项与 version11 fixed-200 基线一致：

- 模型：`deepseek-v4-flash`；
- temperature=0，max tokens=2048；
- rolling legal history=4，full rolling prompt；
- JSON Output carrier，thinking enabled，reasoning effort=high；
- max steps=30，max recoverable errors/type=3；
- one attempt/task，table output rows=0；
- strict no-repair parser，strict-multiset denotation。

输入：

`data/eval_inputs/bird_train_tool_interface_validation200_version4.jsonl`

version19 结果：

`data/trajectories/tool_usability_20260723/version19_fixed30_rolling_json.*`

协议实现提交：`30fcf1b`。

## 结果

预设扩展线是 **24/30**。实际结果：

| 指标 | version11 同 30 题 | version19 |
| --- | ---: | ---: |
| 正确 | 20/30 | **21/30** |
| Accuracy | 66.67% | **70.00%** |
| Legal terminal | — | **29/30** |
| Wrong answer | — | **8** |
| Final execution failure | — | **1** |
| API transport retries | — | **2** |

version19 使用 1,054,285 tokens，耗时 445.178 秒。它比 version11 多对 1 题，但没有达到
24/30，因此按预设 gate **不扩展到固定 200**。

## 配对差异

逐题配对：

- 共同正确：19；
- version19 新增正确：`02408`、`04598`；
- version19 回归：`06026`；
- 共同错误：8。

`02408` 现在在过滤 `George%` 后投影唯一请求列；`04598` 使用
`project(distinct=true)` 去除重复 alias。两者都是 version12/13 以来 exact-output 约束的
预期收益，不依赖隐藏 gold。

`06026` 已得到 south 表中的正确重复值 `33.8744, 33.8744`，但把问题改解释为跨区域总
利润并执行多次 sum/add，最终输出 116.1408。Gold 要 `SELECT DISTINCT Profit`，旧版返回
33.8744。当前工具完全能够用 `project(distinct=true)` 得到正确表；这是模型语义/聚合
选择回归，不是命名标量 cell 读取错误。

## 九个错误的逐题归因

| ID | 终局 | 归因 | 相对问题 / gold 的决定性差异 |
| --- | --- | --- | --- |
| 05952 | wrong | benchmark ambiguity | 问题问单数 county，模型返回唯一 `santa clara county`；gold 因两个 Sankee restaurant rows 要求两条完全相同 county。 |
| 04189 | wrong | model relation choice | 过滤 Free/Sports 正确，但选择 left join；gold 是 inner join。 |
| 01152 | wrong | prompt/gold conflict | External knowledge 明确 player name=`playerID`，模型返回 `irvinky01`；gold 返回 first/middle/last 三列。 |
| 06489 | wrong | model output semantics | External knowledge 明确 object=`OBJ_SAMPLE_ID`；模型已经读到 `paper,18`，却最终投影 `paper`。 |
| 06492 | wrong | prompt/gold conflict | 问题与 external knowledge 要按 IMG_ID 分组筛 `COUNT(OBJ_SAMPLE_ID)<15`，模型照做；gold 实际逐行筛 `OBJ_SAMPLE_ID<15`。 |
| 03688 | execution | tool syntax + benchmark ambiguity | 模型第三次 join 使用未引入的 `film.film_id`，应为 `filter_003.film_id`；即使 join 成功，其计划仍返回全部最长影片库存，而 gold 无 tie rule 地 `LIMIT 1`。 |
| 00593 | wrong | model grain choice | 已读到两次疗程，却固定第一个 encounter，只输出 11 天，遗漏 gold 的第二个 18 天 row。 |
| 05440 | wrong | model operator ordering | 先在全部 Paper 取异常 max Year，后来又自行按 Id 选一篇；gold 是先 inner join Journal，再按 Year 排序取一。 |
| 06026 | wrong | model aggregation choice | 已观察正确 per-row profit，却改成跨区域 sum；正确工具路径可直接 distinct project。 |

其中 4 题含明显 prompt/gold 或 tie/multiplicity 歧义；剩余 5 题是 join 类型、输出实体、
grain、operator order 或不必要 aggregation 的模型决策错误。`03688` 暴露了 derived-handle
join namespace 的易错点，但修复该参数错误也不会让当前模型计划匹配无 tie 规则的 gold，
因此不能把它计作一个可由工具改动恢复的干净样本。

如果仅做诊断、排除这 4 个 benchmark-conflict 样本，version19 在其余 26 题上是
**21/26 = 80.77%**；这不是官方 accuracy，也不能替代 strict-multiset gate。

## 结论

1. version19 相对 version11 没有出现系统性回归：19 个旧正确题保持，净增 1 题。
2. 命名标量引用在针对性题上显著缩短轨迹，但固定首 30 中没有新的纯工具阻塞可再安全
   修复；主要剩余错误是模型关系/粒度/算子顺序判断及 benchmark 歧义。
3. 为追求 strict 75% 而让 harness 猜 inner/left、distinct、tie-breaking 或 gold 输出实体，
   会泄漏 benchmark 偏好并破坏原子工具的中立性。
4. 固定 200 暂停，SFT 构造仍不启动。下一步若继续提高 strict pass@1，应把工作重点从
   扩展公共工具面转到 operator-order/grain 提示或训练，以及 gold/prompt conflict 隔离；
   任一新策略仍先走小样本 gate。
