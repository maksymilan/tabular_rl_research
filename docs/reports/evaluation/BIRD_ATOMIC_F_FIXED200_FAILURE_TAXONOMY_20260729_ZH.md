# Atomic F fixed-200 错误轨迹分类

日期：2026-07-29

## 摘要

F 方案在 frozen 200 上有 64 条失败：

| 顶层结果 | 数量 | 含义 |
|---|---:|---|
| 合法终止但答案错误 | **56** | 工具链完成，`bird-set` denotation 不匹配 |
| Provider carrier 失败 | 4 | 有界 same-turn carrier retry 后仍未产生可接受动作 |
| 参数验证错误终止 | 2 | 同类可恢复错误达到上限 |
| 30 步耗尽 | 2 | 未形成终止证据 |
| **总计** | **64** | |

56 条 `wrong_answer` 中，**49 条没有任何过程错误**，只有 7 条在中途发生过可恢复错误。
因此，F 的主要瓶颈不是调用格式或 SQLite 执行，而是合法轨迹里的语义决策。

本报告只根据公开题目、模型可见上下文、模型实际动作和 harness 反馈归类，不读取或利用
gold SQL、gold rows 或隐藏答案值。语义子类是互斥的“主因假设”；对于仅凭公开题意无法
安全判定的轨迹，单独标为 benchmark/输出约定歧义，不做确定性断言。

## 56 条合法错误的主因

| 主因 | 数量 | 占全部失败 | 说明 |
|---|---:|---:|---|
| Population、grain、join 或 multiplicity | **16** | 25.0% | 算子能执行，但作用在错误实体集合、行粒度或连接 population 上 |
| 公开题意合理但 benchmark/输出约定存在歧义 | **15** | 23.4% | 当前轨迹从公开规格看有合理性，不能安全转化为通用 prompt 规则 |
| 答案槽位或输出表示 | **12** | 18.8% | 多余辅助列、字段拼接、列顺序或表示形式不符合 verifier 约定 |
| 语义映射、算子、公式或字面量 | **11** | 17.2% | 选错表/列、聚合、公式方向、日期解释或 literal |
| Grounding/终止证据误用 | **2** | 3.1% | 已观察目标行，却引用未派生的整个 source table |
| **合法错误合计** | **56** | **87.5%** | |

### 1. Population、grain、join 或 multiplicity：16

`00041, 00074, 00593, 01167, 01530, 01692, 01787, 02052, 02078, 02189,
02418, 03312, 04244, 04869, 05147, 05544`

代表性证据：

- `00593`：观察到多个满足条件的 medication interval，随后自行用
  `extreme_value_select(top_k=1)` 压成一行；题目没有提供该 selector。
- `02078`：终止表包含 60 行经纬度，说明 join 后仍停留在 postal-code/地址行粒度，
  没有得到题目中的单个代表实体。
- `02189`：先计算一个全局销售总额，再把同一总额连接到六个品牌；品牌粒度在聚合前
  已经丢失。
- `03312`：筛选 GDP 后直接聚合，没有先建立“国家—山峰”关系 population。
- `04244`：先在 team 侧跨记录聚合，再与 MVP 条件组合；资格关系没有在聚合前固定。

这类错误最适合因果 SFT/RL，而不是增加工具。现有 join、group、count distinct 和
条件聚合已经能表达所需路径。

### 2. 公开题意合理但 benchmark/输出约定歧义：15

`00040, 00214, 00715, 01152, 01213, 01589, 01888, 02925, 03688, 04038,
04822, 04848, 05632, 06026, 06492`

典型情况：

- `00040` 按公开问题只返回标题，形态本身合理；
- `00214` 对复数 “keywords” 保留多行，没有公开 tie-break/limit 依据；
- `00715` 把 address 表示为 street number + street name，属于合理表示；
- `03688` 对最长影片的并列及其 inventory multiplicity 全部保留，题目没有截断规则；
- `04038` 返回一行一个 gender 及其 count，属于合理的 rows layout；
- `04848` 返回 employee identifier 和 position title，与 “employees ... along with titles”
  的公开措辞并不矛盾。

这些题不能直接产生“永远只输出名称”“并列时任取一行”“地址只保留一个字段”等全局
prompt 约束，否则会形成 benchmark 补丁并伤害其他题。

### 3. 答案槽位或输出表示：12

`00582, 01148, 02438, 02507, 02512, 02868, 02901, 03131, 04426, 05316,
05668, 06336`

代表性证据：

- `01148` 终止表含 `coachID, diff`；`diff` 是排名辅助列。
- `03131` 在 language 与 official flag 之外保留了 country name 定位列。
- `04426` 在 episode identifier 之外保留 title、season、episode number。
- `05316` 已选中最高评分餐厅，却引用含 `id_restaurant, label, review` 的整张排名结果，
  没有投影成问题所需槽位。
- `05668` 在差额和 vendor 之外保留 purchase-order helper。
- `06336` 问题要求 words 后 ids，终止列顺序为 `wid, word`。

其中姓名是否应保持 first/last 两列、拼成 full name，属于表示约定而非缺少关系算子。
当前 prompt 已有 slot-by-slot 检查，继续重复同类文字的边际收益很低。

### 4. 语义映射、算子、公式或字面量：11

`00004, 00886, 01050, 01393, 01461, 01560, 02682, 04906, 05161, 05440,
06246`

代表性证据：

- `00004`：把“创建十年后仍更新”转化成了一个自行选择的日期计算语义。
- `01461`：问题说 review stars，却直接在 business relation 上筛选，缺少 review 路径。
- `02682`：直接平均当前 product relation 的标准成本，没有利用历史成本关系。
- `04906`：公式计算链合法，但被计数实体选择错误；完整 schema 没有阻止模型把
  object occurrences 改写成 image-level count。
- `06246`：对 Delaware 的字段/literal 解释做了未经验证的替换。

这类错误说明“所有 schema 都可见”不等于“题意到关系路径的映射正确”。

### 5. Grounding 或终止证据误用：2

`00833, 02709`

- `00833` 已通过 `read_subtable` 观察 match/team 信息，最终却直接引用整个 source
  `Team` 表；read 是 observation-only，没有生成筛选后的可终止 handle。
- `02709` 读取了最近入职记录和 department 信息，但同样直接引用 source
  `Department` 表，没有派生只含目标部门的最终 relation。

这是清晰的工具边界问题：模型知道目标值，但没有把 observation 转化为 grounded
relational result。可以研究一个显式“从 observation 选择已验证行”的 typed 操作，
也可以用因果训练强化 `read -> condition_filter/project -> answer`；环境不应猜测并
自动把 read 变成 filter。

## 8 条非合法终止

### Provider carrier：4

`00421, 00454, 03664, 05053`

- `05053` 在 0 个语义动作时失败；
- 其余分别在 3、4、5 个合法动作后失败；
- 它们应计入端到端可靠性，但不能归因于 SQL/关系策略。

### 参数验证错误终止：2

`01569, 03505`

- `01569` 先把 `conditions` 放到 `group_aggregate` 顶层，之后连续混淆 derived
  namespace、`join_tables.on.left` 与 bare `right` 的边界；5 个过程错误后终止。
- `03505` 两次给 `read_subtable.limit` 传入 100/50，而合法范围是 1..20；中间还把
  `{table, conditions}` 对象当成 `join_tables.base`，没有先产生 filter handle。

### 30 步耗尽：2

`06165, 06299`

- `06165` 在 paper/author 路径中产生大量重复 read 与 join 试探，30 步仍未构造终止表。
- `06299` 围绕 menu id/uuid 关系反复读取和尝试空 join，发生一次
  `no_progress_error`，最终仍未聚合两个目标 menu 的 dish count。

## 失败轨迹内的接口错误

64 条失败轨迹共有 26 个过程错误，分布在 13 题：

| 错误 | 工具 | 数量 |
|---|---|---:|
| argument validation | `join_tables` | 12 |
| argument validation | `read_subtable` | 7 |
| argument validation | `group_aggregate` | 2 |
| protocol | provider/parser | 3 |
| execution | `join_tables` | 1 |
| no progress | `read_subtable` | 1 |

剩余 51/64 条失败完全没有过程错误。单独看 56 个合法 `wrong_answer`，49 条零过程错误、
7 条带有共 11 个可恢复过程错误。

F 移除了 `describe_table` 和 `inspect_column` 后，接口摩擦主要转移到：

1. 多边 `join_tables` 的 flat namespace 与 bare-right 规则；
2. 把 observation-only `read_subtable` 当成可输出新 handle 的关系操作；
3. 超出 `read_subtable.limit<=20` 或给 read 添加不支持的关系变换参数。

## 18 个相对 baseline 的回归

baseline 正确、F 错误的 18 题可分为：

| 主因 | 数量 | IDs |
|---|---:|---|
| Provider carrier | 3 | `00421, 00454, 05053` |
| 参数验证终止 | 1 | `03505` |
| Grounding/终止证据 | 2 | `00833, 02709` |
| 输出槽位/表示 | 5 | `01148, 02507, 02901, 03131, 06336` |
| Population/grain/join | 3 | `01787, 03312, 04244` |
| 语义映射/算子 | 4 | `01050, 01393, 02682, 04906` |

因此 18 个回归中，只有 4 个是 provider 或接口终止；另外 14 个是合法或语义层回归。
这与整体结论一致：F 的退化不能通过继续降低接口摩擦完全修复。

## 改进优先级

1. **不推广 F 全量上下文。** 49 条无过程错误的 wrong-answer 已表明静态信息量不是主
   瓶颈。
2. **优先处理 join/read 两个高频摩擦点。** 这是唯一有统一工具设计信号的部分，但需要
   小规模 paired gate，不能假设能恢复语义错误。
3. **训练 population/grain 与终止证据转换。** 这两类有公开、因果、可执行的纠错边界。
4. **输出约定与 benchmark 歧义隔离。** 不把隐含 verifier 惯例写进通用 prompt，也不把
   这些诊断失败直接构造成 SFT。
5. **Provider failures 单独重测。** 若要估计 F 的纯策略能力，可对 4 条 carrier failure
   做同设置 infrastructure-only retry，但不能把重试后的结果混回这次 frozen pass@1。

## 产物

- 分类 JSON：
  `data/trajectories/bird_train_atomic_full_bird_context_fixed200_20260729/failure_taxonomy.json`
- F fixed-200 总报告：
  `docs/reports/evaluation/BIRD_ATOMIC_F_FULL_CONTEXT_FIXED200_DEEPSEEK_V4_FLASH_20260729_ZH.md`
- 全部 200 条审计记录：
  `data/trajectories/bird_train_atomic_full_bird_context_fixed200_20260729/verified_success.all.jsonl`

这些轨迹和分类均为 diagnostic-only，不能作为 SFT 数据源。
