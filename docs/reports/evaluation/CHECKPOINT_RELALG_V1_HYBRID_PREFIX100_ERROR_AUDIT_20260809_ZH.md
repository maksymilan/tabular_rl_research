# checkpoint-relalg-v1 Hybrid Prefix100 逐题错误与策略行为审计

日期：2026-08-09  
状态：`diagnostic-only`；不得作为 Atomic 主实验或训练准入证据。

## 结论摘要

这批 100 题实际使用 `mode=hybrid`、A/Text-JSON、官方 `deepseek-v4-flash`，不是计划中的
Atomic 主实验。100/100 都调用了 `execute_sql`，81 题为 `sql_only`，仅 19 题调用过任一
Atomic operator；没有纯 Atomic episode。

- `bird-set`：63/100。
- strict artifact：42/100。
- 63 条 `bird-set` 正确中，21 条未通过 strict：准确分类为 **18 条 schema/列标签问题 +
  3 条 duplicate multiplicity 问题**，而不是 21 条都属于 schema。
- 剩余 37 条：36 条合法 wrong answer，1 条 provider completion-length failure。
- 100/100 只有 `phase_000`；`commit_checkpoint=0`，`restore_checkpoint=0`。
- 因此，这批只部分满足“简单题走短路径”，没有验证“难题分阶段、出现矛盾后 restore”的
  checkpoint 设计目标。

本审计只使用 public example id、结构化 action/result、公开错误反馈和评分布尔字段。没有读取或
输出 Gold SQL、Gold rows、隐藏答案或模型 reasoning。没有利用 hidden gold 反推失败原因；逐题
“根因”均表示轨迹中最接近的可观察诊断。

## 21 条 `bird-set`/strict 差异

所有 21 条的列数与行宽均正确，且没有 tool error。18 条的值、行数以及适用时的顺序均匹配，
只有 schema/列标签不匹配；另外 3 条 schema 正确，但 strict bag multiplicity 不一致。

| 位置 | Example ID | 差异类别 | 可观察根因 |
|---:|---|---|---|
| 3 | `bird_train_03370` | schema label | 正确 1×1 SQL 结果又经 `project` 显式重命名，列标签被改坏。 |
| 8 | `bird_train_01946` | schema label | 单聚合 SQL 使用了不匹配的显式 alias。 |
| 12 | `bird_train_01391` | schema label | 单聚合 SQL 使用了不匹配的显式 alias。 |
| 16 | `bird_train_02087` | schema label | 单聚合 SQL 使用了不匹配的显式 alias。 |
| 18 | `bird_train_03202` | schema label | 单聚合 SQL 使用了不匹配的显式 alias。 |
| 27 | `bird_train_03782` | schema label | 单聚合 SQL 使用了不匹配的显式 alias。 |
| 34 | `bird_train_06262` | multiplicity | Join 与嵌套 `DISTINCT` 保留了正确 tuple set，但折叠了 reference 所需重复次数。 |
| 39 | `bird_train_05855` | schema label | 单聚合 SQL 使用了不匹配的显式 alias。 |
| 42 | `bird_train_02688` | schema label | Read 后的单聚合 SQL 使用了不匹配的显式 alias。 |
| 49 | `bird_train_02663` | schema label | 多次 SQL 探测后，最终聚合仍使用不匹配 alias。 |
| 51 | `bird_train_05140` | schema label | 正确 1×1 SQL 结果经 `project` 重命名后标签不匹配。 |
| 52 | `bird_train_00849` | schema label | Read 后的单聚合 SQL 使用了不匹配 alias。 |
| 57 | `bird_train_05902` | schema label | 单聚合 SQL alias 不匹配；后续观察没有改变终端值表。 |
| 67 | `bird_train_06566` | schema label | 两列值与顺序正确，但直接保留 raw source labels，至少一个标签未满足 reference schema。 |
| 71 | `bird_train_05169` | schema label | 多轮 SQL/read 后，最终单聚合 alias 仍不匹配。 |
| 73 | `bird_train_03747` | schema label | 单聚合 SQL 使用了不匹配 alias。 |
| 81 | `bird_train_03185` | schema label | 正确 1×1 SQL 结果经 `project` 重命名后标签不匹配。 |
| 82 | `bird_train_05849` | schema label | Read 后的单聚合 SQL 使用了不匹配 alias。 |
| 88 | `bird_train_01820` | schema label | 单聚合 SQL 使用了不匹配 alias。 |
| 92 | `bird_train_02530` | multiplicity | Schema 与 tuple set 正确，但关系粒度/重复次数未对齐；隐私保护 artifact 不足以判断具体多或少。 |
| 93 | `bird_train_00315` | multiplicity | 两次 join 后显式 `DISTINCT`，保留正确集合但折叠重复次数。 |

聚合：14 条聚合 alias、3 条 post-SQL `project` rename、1 条 raw 双列标签、3 条 duplicate
multiplicity。没有证据表明存在 NULL-only 或纯排序差异。

## 其余 37 条逐题诊断

### 输出投影或表示（7）

| Example ID | 失败阶段与改进点 |
|---|---|
| `bird_train_04245` | 目标行数正确但返回 4 个字段；终止前应只投影请求字段与顺序。 |
| `bird_train_00649` | 排序后行数正确但列数不匹配；辅助排序指标与答案槽位未分离。 |
| `bird_train_00539` | 找到极值行后把目标字段与辅助排名指标一起输出。载体错误已恢复，不是最终根因。 |
| `bird_train_01283` | 用于定位记录的上下文字段与真正答案字段一起返回。 |
| `bird_train_02230` | 极值行数正确，但携带 5 个上下文字段。 |
| `bird_train_06538` | 找到单一目标实体后只保留名称，遗漏另一个要求的输出槽位。 |
| `bird_train_05627` | 分组结果行数匹配，但排序/解释计数列也被作为答案输出。 |

### Population、filter 或 join（11）

| Example ID | 失败阶段与改进点 |
|---|---|
| `bird_train_01632` | 候选集被额外收窄到精确奖项标签和获奖状态；应先确定所需 population。 |
| `bird_train_04943` | 合并了两个对立角色的 ID；应确认目标角色及名称/ID 表示。 |
| `bird_train_02914` | 加入题目未要求的实体子类型限制，并未经观察直接使用类别编码。 |
| `bird_train_06143` | 后续探索已发现非空候选，但最终仍引用最初空 artifact。 |
| `bird_train_04633` | 未检查存储表示便使用缩写 literal；应先检查 code/full-name 映射。 |
| `bird_train_05050` | 无证据地把地理映射记录去重成“一实体一行”，改变目标粒度。 |
| `bird_train_01162` | Join 只用了实体 ID 与年份，遗漏联赛等复合身份键。 |
| `bird_train_04344` | 把类别词直接映射到近似计数字段的存储值，字段/值语义未经检查。 |
| `bird_train_04868` | 只应用 active 与 URL 条件，遗漏另一显著状态条件。 |
| `bird_train_02699` | 只应用最高评分条件，没有继续应用另一利润极值条件。 |
| `bird_train_00028` | 行列数匹配但值/schema 都不同，说明选择了同规模错误字段或类别表示。 |

### 聚合公式、粒度或单位（13）

| Example ID | 失败阶段与改进点 |
|---|---|
| `bird_train_02811` | 日期范围已修正为非空，但金额聚合值仍不匹配；应先锁定成本字段和聚合定义。 |
| `bird_train_03672` | 将按演员计数逐项除以 5 当作平均值，平均粒度错误。 |
| `bird_train_03843` | 已观察到同名实体对应多个 ID，仍把所有 crew rows 汇成单一计数。 |
| `bird_train_04029` | 日期差和二值状态采用了未经验证的定义；需确定边界和事实字段。 |
| `bird_train_05125` | 在代码字典表上计算比例，没有固定实际事件 population。 |
| `bird_train_04556` | 已验证唯一实体 population，最终却改用保留重复的并集作为分母。 |
| `bird_train_01211` | 计算的是比例，却以 percentage 字段终止，缺少乘 100。 |
| `bird_train_00899` | 分子为评分事件、分母却用不同用户数，粒度不一致。 |
| `bird_train_02325` | 计数不同物理标识，而非题目对应的起飞事件记录。 |
| `bird_train_05158` | 无充分依据使用 `COUNT(DISTINCT ...)`，改变成分/条目计数单位。 |
| `bird_train_01426` | 从预聚合州级关系切换到机构明细重新求和，存在重复聚合风险。 |
| `bird_train_05923` | SQL 失败后切换 Atomic，但把 distinct-device 口径改成 join-row count。 |
| `bird_train_06144` | 形状匹配但值不同；采用包含两端时间范围和普通行计数，边界/去重语义未绑定。 |

### 排序、极值或 tie（3）

| Example ID | 失败阶段与改进点 |
|---|---|
| `bird_train_01741` | 先在全表求最大值再与目标子集相交；应先限定子集再求极值。 |
| `bird_train_05452` | 在类别等级、行频次和单一 top 用户间反复切换，最终任意取一行。 |
| `bird_train_04983` | 最大长度条件扩展到所有 join rows，未先按实体去重或明确 tie 策略。 |

### 旧 artifact／错误分支恢复（2）

| Example ID | 失败阶段与改进点 |
|---|---|
| `bird_train_00984` | 后续已检查更多分区，最终仍回答早期只覆盖部分分区的 artifact。 |
| `bird_train_04745` | 已得到更严格的 47 行关系，却回答早期 52 行 artifact。 |

### Provider（1）

| Example ID | 失败阶段与改进点 |
|---|---|
| `bird_train_04149` | 4,096 与 8,192 completion 上限均以 `finish_reason=length` 结束，没有形成 action；属于 provider completion failure。 |

注：类别是首要诊断；少数轨迹同时涉及多个问题，因此表中交叉原因不会机械等同于互斥计数。
互斥汇总使用：输出 7、population/filter/join 11、聚合/粒度/单位 13、排序/极值/tie 3、
旧 artifact/恢复分支 2、provider 1。

## 是否符合 checkpoint-relalg 设计预期

### 满足的部分

- 26 题采用最短的 `describe_table → execute_sql → answer` 三步路径，其中 21 题正确。
- 1–5 turns 共 66 题，48 题正确（72.7%），说明简单任务通常没有被强制 checkpoint 拖长。
- 12 个出现工具错误的 episode 共 15 个 error event，12 个都随后产生了合法成功调用，说明
  state-preserving local correction 能工作；其中 6 个最终答对。

### 未满足的部分

- 6–9 turns 为 25 题，仅 11 题正确；10+ turns 为 9 题，仅 4 题正确。
- `sql_only` 81 题、52 题正确；`sql_to_atomic` 16 题、11 题正确；`mixed` 3 题、0 题正确。
- 全部 100 题都停留在 `phase_000`。没有一次 `commit_checkpoint`，因而也没有可恢复的阶段
  节点；`restore_checkpoint=0` 是必然结果。
- 36 条 wrong answer 中，30 条全程没有工具错误。主要瓶颈是 population、grain、aggregation、
  tie 和 exact-output 决策，而不是 Harness 拒绝。
- 出错后“恢复到可执行”不等于“恢复正确语义”：12 个错误 episode 中只有 6 个最终正确。

### 判定

这批轨迹符合“简单题可以直接完成”的一半设计预期，但不符合“复杂题形成语义阶段、后续矛盾时
restore”的核心预期。更根本地，它是 Hybrid 偏 Direct SQL 的对照臂，不能用于判断 Atomic 模式
是否通过缩小决策空间改善了正确性或 checkpoint 行为。该问题必须由同 cohort 的
`mode=atomic` 重跑来回答。

## Artifact

- Hybrid Prefix100：
  `data/results/checkpoint_relalg_v1_flash_text_json_hybrid_teacher1500_v2_nonempty_prefix100_20260809/`
- 原结果报告：
  `docs/reports/evaluation/CHECKPOINT_RELALG_V1_FLASH_TEXT_JSON_HYBRID_PREFIX100_20260809_ZH.md`
- 本报告不改变原始 artifact，不把其轨迹纳入 Atomic 或 SFT/RL 数据。
