# BIRD action-block fixed-200 失败轨迹审计（2026-07-26）

## 结论

action-block v10 在 fixed-200、`bird-set` 指标下共有 **57 条失败轨迹**：55 条答案错误、
1 条 provider/API 失败，以及 1 条原子动作预算耗尽。55 条答案错误中，有 44 条是没有
任何执行错误的合法轨迹。因此，当前剩余的主要瓶颈是语义策略，而不是接口异常。

不应在运行时 prompt 中加入更强的通用检查清单。我们在冻结的 24 题集合上，对原始
v11 prompt 和一个简洁、与任务无关的 v12 constraint-first prompt 做了配对测试。v12
没有恢复任何题，反而让一个控制题退化，过程错误翻倍，token 增加 12.6%。因此该实验
被拒绝，活动 prompt 继续与 action-v4 保持逐字节一致。

应保留 v11 的安全解析器，这是目前得到确认的工具接口改进。它会拒绝在字面量参数位置
使用列引用，并再次修复了 `02179`。不应新增关系运算工具：本次审计中的所有语义失败都
可以由现有工具表达，而且没有发现某一个缺失算子具有可信的跨任务收益。

## 审计方法

审计输入为：

`data/trajectories/batch_plan_20260726/action_block_v10_low_friction_fixed200_r1.all.jsonl`

`src/eval/audit_action_block_failures.py` 会生成一个包含 57 条记录的紧凑审计视图，保留：

- 问题、外部知识、仅用于离线审计的 gold SQL、预测结果和 gold denotation；
- 模型原始调用参数以及环境最终执行的参数；
- 结果表句柄、列、行数、受限的样例行、执行错误和接口自动修正事件；
- 对应 version24 baseline 的结果。

紧凑视图只删除了重复出现的完整 resident state。gold SQL 只用于离线诊断，从未暴露
给 actor，也没有被用于改变工具执行。

下面的主因分类互斥，每条失败只计入一个主因：

| 主因 | 失败数 | 解释 |
|---|---:|---|
| Benchmark、问题或外部知识存在歧义/冲突 | 20 | 如果针对这些题加规则，学到的会是局部 gold 惯例，而不是通用关系推理原则。 |
| 语义映射或算子选择错误 | 13 | 在工具表达能力充分的情况下，选择了错误的表、列、聚合或计算公式。 |
| Population、grain、连接或多重性错误 | 12 | 运算本身看似合理，但作用在错误的实体集合或行粒度上。 |
| 答案槽位或输出形状错误 | 9 | 证据已经正确或接近正确，但输出字段缺失、多余、被拼接或布局错误。 |
| 接口问题或探索预算耗尽 | 2 | 一条是自动解析器的不安全行为，一条是反复尝试空连接。 |
| Provider/API | 1 | 已经完成若干成功工具调用后，外部连接重复中断。 |
| **总计** | **57** | |

## 逐条审计结果

“安全改进方向”用于区分通用干预和 benchmark 补丁。“现有契约已覆盖”表示当前 prompt
或工具已经可以表达正确行为；这类失败适合成为因果训练或评估证据，而不是继续增加
prompt 文字。

| ID | 主因 | 轨迹证据与失败原因 | 安全改进方向 |
|---|---|---|---|
| `00004` | 语义/算子 | 模型尝试构造“创建十年后”的时间戳并得到 2,252；gold 使用年份子串的严格差值，结果为 312。模型表达式在年份加法和字符串拼接之间也缺少括号。 | 现有 `project` 表达式已经足够；不能把某一种日期解释编码成全局规则。 |
| `00040` | Benchmark 歧义 | 问题要求列出标题，模型返回了四个正确标题；gold 还额外返回 `qty`，但题目只把数量用作筛选条件，没有明确把它定义为答案槽位。 | 不为隐含的 gold 输出列增加 prompt 补丁。 |
| `00041` | 输出槽位 | 推理和读取结果都已经包含商店 `7131` 以及销量最少的标题，但终止证据只选择了标题。 | 现有精确答案槽位检查已覆盖；适合作为因果训练样本。 |
| `00214` | Benchmark 歧义 | 模型找到了唯一最高人气电影及其全部关键词；gold 却在关键词连接结果上任意 `LIMIT 1`，与问题中的复数“keywords”不一致。 | 不增加无依据的连接后限一规则。 |
| `00362` | Benchmark 歧义 | 两位符合条件的教授教学能力同为 5，模型返回两人，gold 任意选择一行。 | 题目未给出 tie-break 时，不应虚构排序规则。 |
| `00582` | 输出形状 | 五位患者及年龄均正确，但模型把 first name 和 last name 拼成了一个字段；外部知识和 gold 都要求两个独立字段。 | 使用现有的“不得无要求拼接字段”规则。 |
| `00593` | Population/grain | 轨迹观察到了两条匹配的用药区间，分别是 11 天和 18 天，但模型在计算前任意选择了第一条。 | 保留全部匹配行；逐行日期计算已经可以通过 `project` 表达。 |
| `00600` | 输出布局 | 模型正确得到分类行 `F=193`、`M=180`，但 gold 要求一个按 `(male, female)` 排列的单行结果。 | 使用现有 `group_aggregate(output_layout="columns")`。 |
| `00715` | Benchmark 冲突 | 外部知识把地址明确定义为 `street_num, street_name`，模型按此返回两个字段，但 gold 只返回 `street_name`。 | 应遵循公开契约，而不是与其冲突的隐藏 gold 形状。 |
| `00796` | 语义映射 | 模型直接聚合 `Match.Man_of_the_Match`，没有采用数据集对“Man of the Series”所要求的 Season-to-Match 路径。 | 需要 schema-grounded 的因果训练，不需要新工具。 |
| `00886` | 语义映射 | 模型正确找到了 Austria，但选择了 `Indicators -> Series`；gold 使用 `CountryNotes -> Series`。 | 加强关系路径选择训练，而不是添加任务专用规则。 |
| `01152` | Benchmark 冲突 | 模型找到了正确的最年轻球员，但只返回 first/last；gold 还包含 middle name，而外部知识又声称“player name”指 `playerID`。 | 输出约定互相冲突，不能增加全局姓名组成规则。 |
| `01167` | Benchmark 冲突 | 模型把获奖教练连接到其真实执教记录，发现没有 POR 教练；gold 只按年份把 awards 和 teams 连接，得到 5。 | 作为 gold/query 语义冲突处理。 |
| `01213` | Benchmark 歧义 | 模型找到最新作品及其全部 47 个角色；gold 在完成多个连接后任意 `LIMIT 1`，只返回一个角色。 | 不增加题目未要求的限一。 |
| `01461` | 语义映射 | 模型筛选 `Business.stars`，而不是连接 `Reviews.review_stars`；前五行样例恰好一致，但完整 denotation 不同。 | 训练问题名词到精确列的映射；现有工具已经支持该连接。 |
| `01530` | Population/grain | 模型用匹配企业数除以整个 `Business` 表行数；gold/外部计算使用 attribute-assignment 行作为分母。 | 保持显式定义的分母粒度，不需要新算子。 |
| `01560` | 语义/算子 | 模型计算每种 tip length 的平均 likes；gold 要求每种长度的 likes 总和。 | 使用现有 `group_aggregate(sum)`，加强聚合类型选择训练。 |
| `01569` | Population/grain | 模型先找营业天数最多的企业，再输出其类别；gold 按类别聚合工作日记录并对类别排序。 | 聚合前先确定题目要求的实体粒度。 |
| `01589` | Benchmark 冲突 | 外部知识要求 `attribute_value='true'`，模型完全忽略 value，而 gold 反而接受 `none/no/false`。 | 不能为了得分而强化互相冲突的外部知识。 |
| `01888` | Benchmark 歧义 | 模型返回题目要求比较的两个计数；gold 把“compare”解释成一次减法，并且从不同表取得 2005 年数据。 | 不增加“compare 一律表示 subtract”的全局规则。 |
| `02052` | Provider | 完整重新执行后仍在完成 11 个成功语义动作之后断开连接。 | 单独报告为 provider/API 失败，不能改标为策略失败。 |
| `02078` | Population/grain | 模型把选区内所有邮编坐标平均为一个中心点；gold 返回每个邮编的经纬度。 | 使用现有的行粒度保持规则；题目没有要求聚合。 |
| `02088` | Population/grain | 模型统计 distinct city names；外部知识和 gold 统计两种 post-office type 的行数。 | 遵循显式 count 粒度，不需要工具改动。 |
| `02179` | 接口 | v10 把字面量 `value` 位置中的 `$filter_geo.LocationID` 转换成字符串 `"LocationID"`，静默得到零行。 | **已由 v11 修复**：拒绝列引用充当字面量，并要求读取 grounded value 后重新规划。 |
| `02189` | Population/grain | 模型先计算所有合格销售的全局总额，再把同一数值 cross join 到每个品牌；gold 要求按品牌分别求和。 | 使用现有分组聚合并保留 group key。 |
| `02418` | 语义/算子 | 模型使用 `count_distinct(order_id)`；外部知识和 gold 使用 `count(order_id)`。 | 严格保留外部知识明确给出的计数算子。 |
| `02438` | 输出/语义 | 模型在修复两个 namespace 错误后，为每个订单选择最新状态并输出订单 ID；gold 要求匹配历史中的 distinct status values。 | 现有槽位和 population 检查已经覆盖；v11 只能消除接口摩擦，不能修复该语义错误。 |
| `02507` | 输出形状 | 模型找到正确员工，但把 first name 和 last name 拼成一个字段。 | 使用现有独立答案槽位规则；v11 控制实验中该题得到恢复。 |
| `02513` | Population/grain | 模型统计通过的 inspection rows，得到 211，而不是 distinct businesses/licenses 的 203。 | 对题目要求的实体做计数；现有 `count_distinct` 已足够。 |
| `02868` | 输出形状 | 模型输出 first、middle、last 三列；gold 只要求 first 和 last。 | 需要答案槽位训练，但不同任务的姓名惯例并不一致。 |
| `02925` | Benchmark 歧义 | 模型返回可读的产品名称，gold 返回 `ProductID`，题目只说“product”。 | 不能从该题推导出全局优先 ID 或名称的规则。 |
| `03042` | 输出形状 | 模型完成了正确的 goalie 计算，但把 first/last 拼成了一个字段。 | 使用现有不得无要求拼接规则。 |
| `03131` | 输出形状 | 模型正确包含两个在 1830 年独立的国家，但在每个语言结果中额外加入 country name；gold 只选择 language 和 official flag。 | 在终止时删除只用于定位或消歧的辅助字段。 |
| `03373` | Benchmark 歧义 | 模型找到正确的 19 个沙漠名称并单独算出 19，但问题要求数量和名称，gold 却只含名称；终止工具只能引用一个矩形结果。 | 不围绕互相冲突的 gold 输出扭曲工具设计。 |
| `03664` | Benchmark 冲突 | 模型返回最大的 `rental_rate/rental_duration`，与外部知识完全一致；gold 按该比值排序，却输出原始 `rental_rate`。 | 作为输出定义冲突处理。 |
| `03688` | Benchmark 歧义 | 共有 10 部电影并列最长，对应 46 条 inventory 记录；gold 任意取第一条连接结果。 | 不增加任意截断并列项的规则。 |
| `04038` | Benchmark 冲突 | 模型按性别统计 distinct patients，得到 349/337；gold 尽管问题说的是 patients，却统计 condition rows，得到 553/572。 | 不增加全局反转 distinct 语义的规则。 |
| `04189` | Population/连接 | left join 保留没有 review 的 free sports apps，而 gold 使用 inner join。轨迹中的 namespace 错误已经恢复，并不是最终失败原因。 | 训练题目所需的配对 population，不需要新 join 工具。 |
| `04244` | Population/grain | 模型按 `tmID` 跨年份独立汇总 points，再与曾经拥有 MVP 的球队求交；gold 先构造 player/team/award 的统一连接 population，再按 team name 分组。 | 在聚合之前先建立完整、固定的连接 population。 |
| `04426` | 输出形状 | 模型找到了五个正确 episode IDs，但又输出了 season、episode number、title 和 air date。 | 终止时只选择题目要求的 episode identifier。 |
| `04822` | Benchmark 冲突 | 模型遵循问题和外部知识的 `TEAM='Avangard Omsk'`，结果为零；gold 却筛选 `TEAM='Czech Republic (all)'`。 | 不能编码隐藏且冲突的筛选值。 |
| `04848` | Benchmark 冲突 | 模型输出 employee ID、job title 和 sick-leave hours；外部知识提到 ID/title，但 gold 只有 title。 | 不增加针对该题的输出约定。 |
| `04869` | Population/grain | 模型只保留当前任职记录并统计 distinct employees；外部知识/gold 统计全部 shift-history 连接行。 | 保持显式定义的 population，不需要新工具。 |
| `04906` | 语义/算子 | 模型统计 distinct images；外部知识和 gold 要求计算 broccoli 与 tomato 的 object-sample occurrences 比值。 | 遵循外部知识明确指定的被计数列。 |
| `05053` | Benchmark 歧义 | 模型返回可读名称 `Laos`，gold 返回国家代码 `LAO`。 | 不能全局规定 nation name 必须替换为 code。 |
| `05147` | Population/grain | 模型在计算分子和分母之前对 businesses 去重；gold/外部公式统计连接后的 violation rows。 | 保持公式定义的粒度，不需要新算子。 |
| `05161` | 语义映射 | 外部知识规定 cheese 是 ingredient category；模型把条件扩展为 category 或 ingredient name，得到错误的 108 条 population。 | 严格保留显式 category predicate。 |
| `05544` | Population/grain | 模型统计了 6 条 representative-term rows，而不是 3 位 distinct representatives。 | 对题目要求的实体计数，而不是历史记录行。 |
| `05632` | Benchmark 冲突 | 模型计算“日本供应商中负债的比例”10.44%，与外部知识完全一致；gold 实际计算“所有负债供应商中日本供应商的比例”4.71%。 | 不强化与 benchmark 公式相冲突的规则。 |
| `05668` | 语义/输出 | 模型先按订单汇总每条明细的差额，并以 vendor name 在前的顺序输出，得到 290；外部知识/gold 要求单条 detail line 的最大差额，并输出 `(difference, VendorID)` = `(280,1520)`。 | 遵循显式算子以及输出字段和顺序。 |
| `06026` | Benchmark 歧义 | 同一产品同时出现在 West 和 South，利润不同；模型合并两个区域，gold 未给依据地只选择 `south_superstore`。 | prompt 无法安全推断隐藏区域。 |
| `06165` | 语义映射 | 模型已经观察到正确的 `PaperAuthor.Name` 行，随后又连接稀疏的 `Author` 表，把七行压缩成一行并改变了姓名。 | 优先使用已经 grounded 的直接答案字段，不需要新 join 能力。 |
| `06246` | 语义/字面量 | 外部知识把 Delaware 定义为 county；模型改用 state `DE`，并额外添加 `type='Post Office'`，而没有检查 county 的实际字面量 `DELAWARE`。 | 使用现有“对不确定字面量先 inspect”的行为。 |
| `06299` | 接口/预算 | 对过滤后的 `MenuPage.menu_id` 和 `Menu.id` 反复连接都得到空结果，模型不断尝试变体直到 20 个动作。gold 与问题对于两个总数是否相减也不一致。 | 预算终止是正确的；应改善因果训练样本，而不是新增专用 join 算子。 |
| `06324` | 语义映射 | 模型把唯一的 `langs` 行当作 corpus 并返回语言代码 `ca`；gold 按 `pages.words` 排序并返回 `pages.title`。 | 加强 schema 与问题的整体 grounding；现有工具足够。 |
| `06489` | 语义映射 | 外部知识明确把“object”映射到 `OBJ_SAMPLE_ID`；模型却额外连接 object classes 并返回 `"paper"`。 | 遵循显式输出列映射。 |
| `06492` | Benchmark 冲突 | 模型按 image 分组并统计 object samples 少于 15 的图片，符合问题和外部知识；gold 却统计 `OBJ_SAMPLE_ID < 15` 的行数。 | 不训练与问题冲突的行级捷径。 |

## Prompt 门控实验

被拒绝的 v12 prompt 只比 v4 多 196 个字符，不包含任务名、数据库名、具体列名、具体值
或 BIRD 专用规则。它只把现有 final check 改写为三个通用操作：

1. 保留外部知识明确给出的算子和实体粒度；
2. 枚举答案槽位并删除辅助列；
3. 避免未被要求的聚合、去重、限一、tie-break 和字段拼接。

冻结的 24 题集合在两个实验运行前已经确定，包括：

- 13 条理论上可能由上述通用原则改善的失败轨迹；
- 4 条 benchmark/外部知识冲突守卫；
- 用于验证 v11 字面量引用安全修复的 `02179`；
- 6 条先前正确的控制题。

| 指标 | v11 原始 prompt | v12 constraint-first | 变化 |
|---|---:|---:|---:|
| 正确数 | **6/24** | 5/24 | -1 |
| 配对恢复 / 退化 | — | 0 / 1 | `05440` 退化 |
| 合法终止 | 24/24 | 24/24 | 0 |
| 过程错误 | **4** | 8 | +4 |
| 被阻塞的后继调用 | **2** | 11 | +9 |
| 模型轮次 | **114** | 119 | +4.4% |
| 原子动作数 | **213** | 237 | +11.3% |
| 总 token | **575,765** | 648,244 | +12.6% |

v12 没有恢复 13 条预先声明的目标失败中的任何一条。`02179` 在两个版本中都正确，说明
该恢复来自 v11 的环境侧安全边界，而不是 prompt 文字。该 prompt 实验被拒绝，不应扩展
到 fixed-200。

## 工具设计结论

目前确认了一个真实的工具设计缺陷：v10 允许形如列引用的局部引用被转换成字面量字符串。
v11 的类型化拒绝是正确边界，因为它会暴露一个可恢复错误，而不是猜测 cell value。
该修复在最初的 `02179` 定向测试、v11 gate 和 v12 gate 中都成功。

其余失败不足以证明需要新增原子工具：

- 逐行计算已经可以通过 `project` 表达式完成；
- category-to-column 布局已经由 `group_aggregate` 支持；
- distinct 实体计数、分组指标、条件聚合、连通连接、排序以及终止投影都已经可以表达；
- 大多数错误轨迹合法且没有执行错误，因此继续增加接口自动修正不能解决其主因。

下一步更可信的方向，是生成对 entity grain、聚合类型和终止答案槽位进行对比的因果训练
数据，同时把 benchmark 冲突题排除在训练来源之外。不应继续增加运行时 prompt 文字，
不应从隐藏 gold SQL 编译完整轨迹，也不应建立任务专用映射补丁库。

## 产物

- `data/trajectories/batch_plan_20260726/action_block_v10_fixed200_failures.audit.jsonl`
- `data/trajectories/batch_plan_20260726/action_block_v11_failure_audit_gate24_control.all.jsonl`
- `data/trajectories/batch_plan_20260726/action_block_v12_constraint_first_gate24.all.jsonl`
- `data/trajectories/batch_plan_20260726/action_block_v12_vs_v11_failure_audit_gate24.paired.json`
- `src/eval/audit_action_block_failures.py`
