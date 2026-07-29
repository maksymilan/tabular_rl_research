# Atomic version37：54 条失败轨迹逐条因果审计

日期：2026-07-29
协议：`atomic` / `version37` / `think-json-v1`
教师模型：DeepSeek v4 Flash
指标：`bird-set`
性质：诊断报告，不是 SFT 数据，不改变 version26 的冻结生产链

## 结论先行

本报告逐条审计最近一次 70 道历史失败题复测中的 54 条失败轨迹。完整的模型可见动作、参数、观察和错误反馈保存在配套的[逐步轨迹报告](./BIRD_ATOMIC_VERSION37_PREVIOUS_TEACHER_FAILURES70_TRACE_AUDIT_20260729_ZH.md)中；本文专门回答每条轨迹“第一处偏离在哪里、为什么错、怎样最小修正、修正是否被验证”。

总体上，退化并不主要来自接口无法调用：

- 54 条失败中，45 条没有任何过程错误；只有 9 条出现过程错误。
- 对 27 条轨迹做了“一题一次、预先声明、只返回布尔结果”的隐藏判分反事实检查。16 条最小修正通过，11 条最直觉的修正被证伪。
- 已确认可由局部决策修复的错误，主要集中在：外部知识字段映射被模型自行改写、实体粒度与行粒度混淆、无依据添加时间/总体限制、终止表多带辅助列、以及字符串数值未完整规范化。
- 多条轨迹在可见证据上完全自洽但仍判错。这类题应优先审计题目、期望输出形态或隐藏标注，不能继续靠 prompt 猜答案。

本文使用四种结论标记：

- **反事实已证实**：只替换一个预先声明的决策，隐藏 `bird-set` 判分通过。
- **简单修复已证伪**：只测试了一种最自然的修复，但仍未通过；为避免标签探测，不再试第二种。
- **轨迹级高置信判断**：可见观察足以指出偏离，但未调用隐藏判分器。
- **未定位/标注待审计**：轨迹与题面、外部知识和可见数据一致，不能诚实地从轨迹推出具体错误。

## 逐条分析

### 1. example 6489 · `image_and_language`

- 问题：在图像 5、坐标 `(634, 468)` 处找到 object。
- 可见证据：外部知识已经把这里的 object 明确映射为 `OBJ_SAMPLE_ID`。步骤 2 的过滤得到唯一一行，同时包含 `OBJ_SAMPLE_ID` 和 `OBJ_CLASS_ID`。
- 第一处偏离：步骤 2 的推理在没有任何新观察的情况下，把目标从 `OBJ_SAMPLE_ID` 改成了 object class；随后连接 `OBJ_CLASSES`，最终输出 `"paper"`。
- 错因：这不是工具能力不足，而是模型覆盖了任务给定的语义映射。后续所有 SQL 等价操作都正确地执行了错误目标。
- 最小修正：直接从 `filter_001` 投影 `OBJ_SAMPLE_ID`，不连接类别表。
- 验证：**反事实已证实**，一列一行的 `OBJ_SAMPLE_ID` 结果通过。
- 改进方向：训练约束应强调“外部知识中的显式字段映射优先级高于自然语言再解释”；不需要新工具。

### 2. example 593 · `synthea`

- 问题：Berry Keebler 因 acute bronchitis 就诊时服用 Acetaminophen 160 MG 多久。
- 可见证据：严格过滤后存在两条合法药物记录；对应两个不同 encounter，日期差分别为 11 天和 18 天。两条 encounter 都明确显示 acute bronchitis。
- 第一处偏离：步骤 12 在没有“首次、最早、某一年”等题面条件时，任意选择较早 encounter。
- 错因：模型观察到了真实歧义，却用自创限制把多答案压成单答案。
- 最小修正：保留已经计算出的两行 duration，不再添加 encounter 过滤。
- 验证：**反事实已证实**，直接引用两行 `project_003` 通过。
- 改进方向：SFT 应包含“观察到多个同样满足条件的实体时，保留全集；不得用 earliest/latest 自行消歧”的因果样例。

### 3. example 4189 · `app_store`

- 问题：列出所有免费体育 App 及其 translated review。
- 可见证据：免费体育 App 有 360 行；左连接后有 9,091 行，并出现大量无评论 App 的 NULL 扩展行。关系派生反馈明确给出了实际 unmatched 行。
- 第一处偏离：步骤 3 选择 left join，而问题要求的是 App 与已有 translated review 的配对。
- 错因：左连接把“没有评论”也解释成“及其评论”的有效结果。
- 最小修正：把同一连接边改为 inner join，其他过滤和投影保持不变。
- 验证：**反事实已证实**，结果变为 8,800 行并通过。
- 改进方向：现有事实反馈已足够；重点是训练模型利用 unmatched-row 反馈，而不是继续改接口。

### 4. example 6492 · `image_and_language`

- 问题：有多少图像包含少于 15 个 object sample。
- 可见证据：模型只在 `IMG_OBJ` 中按 `IMG_ID` 分组，因此只能看到至少有一个 object 的图像；得到 46,171 个满足条件的已出现图像。
- 第一处偏离：从 object 事实表直接定义“全部图像”的总体。
- 错因：如果数据集中存在零 object 的图像，它们在 `IMG_OBJ` 中没有任何行，不可能被这个分组计入。当前公开 catalog 又没有清晰的完整 image 主表。
- 最小修正：必须从完整图像总体出发，左连接 object 计数，再筛 `<15`；如果 catalog 确实没有该总体，则此题对当前工具可见数据不可完备求解。
- 验证：**轨迹级高置信判断**，未做判分反事实。
- 改进方向：先审计 catalog 是否漏暴露完整 image relation；这是潜在数据/工具覆盖问题，不应靠 prompt 假装解决。

### 5. example 3688 · `movie_3`

- 问题：最长影片的 store ID 和 inventory ID。
- 可见证据：最大长度为 185；共有 10 部并列影片，连接 inventory 后得到 46 行。模型输出顺序为 inventory ID、store ID。
- 第一处不确定：题面使用单数 “the film”，而实际最大值有 10 个并列；模型选择保留全部并列及其全部 inventory。列顺序也与题面顺序相反。
- 已测试修复：只把最终列顺序改成 store ID、inventory ID。
- 验证：**简单修复已证伪**；46 行反转列序仍未通过，说明问题不只在列序。
- 剩余原因：最可能是并列选择口径、结果去重粒度或单数题目的期望总体；不能继续用判分器试探。
- 改进方向：需要 `with_ties` 语义明确的极值训练样例，并对题目/标注的并列策略做离线审计。

### 6. example 2408 · `books`

- 问题：找出拥有最多书籍的作者。
- 可见证据：模型已经按作者计数并定位正确最大作者，但终止证据表仍同时包含 `author_id` 和 `author_name`。
- 第一处偏离：终止前没有做精确投影，辅助 ID 被当作答案列。
- 最小修正：从最终过滤表只投影 `author_name`。
- 验证：**反事实已证实**，67 行作者名结果通过。
- 改进方向：加强“终止表就是答案本身，排序键、计数和 ID 不是默认答案列”的输出形态训练。

### 7. example 5440 · `authors`

- 问题：最新发表论文的标题与期刊主页。
- 可见证据：模型先在全部 `Paper` 上取最大 `Year`，得到异常值 `800190` 且 `JournalId=0`；inner join 期刊返回 0 行。模型随后改用 left join，接受 NULL homepage。
- 第一处偏离：先排序再限制“有期刊”的论文总体。
- 错因：题目要求 journal homepage，候选论文必须先能连接到 Journal；异常无期刊记录不属于可回答总体。
- 最小修正：先 inner join `Paper→Journal`，再在连接结果上按 `Paper.Year DESC` 取最新，输出标题和主页。
- 验证：**反事实已证实**，该算子顺序通过。
- 改进方向：训练“先固定满足所有关系约束的总体，再做极值”；无需新增 join 工具。

### 8. example 6454 · `retails`

- 问题：rail 和 mail 中哪种 shipping mode 的订单更多。
- 可见证据：模型已得到 rail/mail 两行分组计数，并在推理文本中判断了赢家。
- 第一处偏离：终止时引用两行计数表，没有把“哪一种更多”落实为单行结果。
- 最小修正：对两行计数做降序 top-1，仅输出 `l_shipmode`。
- 验证：**反事实已证实**。
- 改进方向：增加比较问句的终止形态样例；模型推理中的正确结论不能替代 grounded 结果表。

### 9. example 2901 · `works_cycles`

- 问题：列出满足条件的 10 个最年长员工的 full names。
- 可见证据：过滤和按生日排序路径合法；模型把 first/middle/last 拼成带空格的单一字符串。
- 第一处偏离：未经字段表示证据，把三个姓名字段改造成模型自定义文本格式。
- 已测试修复：只去掉拼接空格，保留单列 full name。
- 验证：**简单修复已证伪**，说明不是单纯分隔符问题。
- 剩余原因：可能需要三列原始姓名、需要处理 NULL middle name，或第 10 名存在并列；不能继续试探。
- 改进方向：输出表示应优先保留原始字段，除非外部知识明确规定拼接形式。

### 10. example 2925 · `works_cycles`

- 问题：2013 年最低 standard cost 的产品。
- 可见证据：模型把“in 2013”实现为成本有效区间与 2013 日历年有重叠。
- 第一处偏离：选择 overlap 语义时，没有可见证据说明应取“年内任意有效”、`StartDate` 在 2013、年末快照还是其他口径。
- 错因：工具执行正确，但时间总体定义没有被题面或外部知识固定。
- 最小修正：先用 schema/值观察确定业务日期语义；若仍不可消歧，应把它标为数据集语义问题，而不是自由选择一种。
- 验证：**轨迹级判断**，未做反事实。
- 改进方向：这类题需要时间口径的训练数据或更明确的外部知识，增加 prompt 通用文字作用有限。

### 11. example 4822 · `ice_hockey_draft`

- 问题：寻找同时满足俱乐部赛季与 International 条件的球员。
- 可见证据：把 TEAM、LEAGUE、SEASON、G 条件压到同一条 `SeasonStatus` 记录时得到 0 行；后续读取显示俱乐部事实和 International 事实存在于不同记录。
- 第一处偏离：假设所有条件必须共属于同一赛季状态行。
- 错因：同一球员需要两个语义角色不同的 `SeasonStatus` 实例，一个表达俱乐部赛季，另一个表达 International 记录。
- 最小修正：分别过滤两个 SeasonStatus 总体，按球员标识连接/求交，再连接球员信息。
- 验证：**轨迹级高置信判断**。
- 改进方向：针对同表多角色的因果 SFT 比改 join 语法更重要；现有 `base_role/role` 已具备表达能力。

### 12. example 4848 · `works_cycles`

- 问题：列出 sick leave hours 最高的前三名员工及其职位。
- 可见证据：模型 top-3 后直接输出 `BusinessEntityID`、`JobTitle`、`SickLeaveHours`。
- 已测试修复：移除排序辅助列 `SickLeaveHours`，只保留员工 ID 和职位。
- 验证：**简单修复已证伪**。
- 剩余原因：可能涉及第三名并列、期望人员姓名而非 ID、或“top three”按三个不同数值层级解释。单纯删辅助列不是完整原因。
- 改进方向：审计并列策略与 person 表示；不应再加针对该题的 prompt 补丁。

### 13. example 715 · `restaurant`

- 问题：给出 Peking duck 对应餐厅的地址。
- 可见证据：模型过滤菜名、连接 location，并输出可见地址行 `[2310, "el camino real"]`；外部知识也把 address 映射为 street number 与 street name。
- 错因判断：从模型可见轨迹看，关系路径、值和两列表示均自洽，没有证据支持再改字段或总体。
- 最小修正：无法从可见证据诚实确定。
- 验证：**未定位/标注待审计**。
- 改进方向：优先检查期望输出的列顺序、重复行规则和数据集标注；不要让模型通过更多探索去猜隐藏格式。

### 14. example 5053 · `mondial_geo`

- 问题：在 communist 国家中寻找 GDP 最低者。
- 可见证据：模型基于观察到的政治取值过滤，再连接经济表并取 GDP 最小值，最终得到一个国家。
- 错因判断：轨迹合法且表面语义完整；可见信息没有说明政治状态应绑定哪个历史时期，也没有说明多个经济记录的年份/指标口径。
- 最小修正：先固定政治状态与经济指标的共同时间粒度；若表中没有时间一致性字段，需审计标注。
- 验证：**未定位/标注待审计**。
- 改进方向：属于跨表时态语义，不是接口摩擦。

### 15. example 5668 · `works_cycles`

- 问题：找 ordered quantity 与 received quantity 差额最大的订单及 vendor。
- 可见证据：模型逐行计算 `OrderQty-ReceivedQty`、取最大、连接 vendor，输出订单 ID、差额、vendor。
- 已测试修复：移除订单 ID，只输出差额与 vendor。
- 验证：**简单修复已证伪**。
- 剩余原因：可能是绝对差、最大值并列、列顺序或订单标识是否应保留；外部映射不足以区分。
- 改进方向：保留现有算术工具，审计问题的差额方向及并列期望。

### 16. example 4 · `movie_platform`

- 问题：创建后超过 10 年才更新的记录。
- 可见证据：步骤 4 把 `date_diff_days` 的 start/end 反置，产生负数；步骤 8 一度改成 creation→update，步骤 9 又退回反向并筛 `>3652`，结果为 0。
- 第一处偏离：首次日期差调用的操作数顺序；后续虽观察到异常并短暂修正，却没有保持修正。
- 错因：明确的 operand-order 推理错误；另有用固定 3652 天近似十周年的边界风险。
- 最小修正：保持 `start=creation, end=update`，再用明确的十年边界规则过滤。
- 验证：**轨迹级高置信判断**。
- 改进方向：为有方向的 `date_diff_days/subtract/divide` 加入因果错误恢复样例；无需修改工具执行。

### 17. example 214 · `movies_4`

- 问题：最高 popularity 影片的关键词。
- 可见证据：模型按 popularity 降序直接 `top_k=1`，再连接关键词。
- 第一处偏离：`top_k=1` 在最大值存在并列时会任意裁掉其他影片。
- 最小修正：先求最大 popularity 标量，再用等值过滤保留所有并列影片，最后连接关键词。
- 验证：**轨迹级判断**。
- 改进方向：需要明确教会 `top_k=1` 与“所有最大值”不是同一语义；可考虑未来提供显式 `with_ties`，但不是当前必需。

### 18. example 4426 · `simpson_episodes`

- 问题：列出演职员获得 Oscar credit 的 episodes。
- 可见证据：过滤 credit 并连接 Episode 的路径成功；最终输出 season、episode、title。
- 已测试修复：只保留 episode title。
- 验证：**简单修复已证伪**。
- 剩余原因：期望的 episode 表示可能是编号组合、ID 或其他字段，单纯 title 不足。
- 改进方向：这是输出实体表示问题；应从数据集标注归纳统一规则，而不是追加该题专用提示。

### 19. example 2418 · `books`

- 问题：订单最多的 book。
- 可见证据：模型按书统计订单、取 top-1，输出 title 与 count。
- 已测试修复：只保留 title。
- 验证：**简单修复已证伪**。
- 剩余原因：更可能是最大订单数存在并列，或“订单数”需要不同计数粒度；辅助 count 不是唯一问题。
- 改进方向：极值并列与订单实体粒度应在训练中统一。

### 20. example 886 · `world_development_indicators`

- 问题：Austria 的 series subject。
- 可见证据：模型过滤 Austria，连接 indicator/series，并投影 distinct `Topic`，得到 73 个 subject。
- 错因判断：外部知识也把 subject 映射到 Topic、Austria 映射到 ShortName；轨迹看起来完全遵守映射。
- 最小修正：无法从可见证据确定是 Topic 字段、series 子集还是期望输出形态的问题。
- 验证：**未定位/标注待审计**。
- 改进方向：检查题目是否遗漏特定 series 限制及标注是否使用另一主题字段。

### 21. example 3969 · `talkingdata`

- 问题：最老的 device user 是男还是女。
- 可见证据：模型按 `age DESC` 取一行，观察到 `["F", 96]`，但终止表同时带 `gender` 和排序辅助列 `age`。
- 第一处偏离：终止证据没有精确到题目只问的 gender。
- 最小修正：从 top 表只投影 `gender`。
- 验证：**反事实已证实**。
- 改进方向：这是纯终止形态错误，不需要再探索，也不需要新工具。

### 22. example 2513 · `food_inspection_2`

- 问题：David Hodges 做的 Short Form Complaint inspections 中，有多少 businesses passed。
- 可见证据：严格条件过滤得到 211 条 inspection；模型只保留 `inspection_id`，随后按 inspection 行计数。
- 第一处偏离：步骤 5 丢弃 business 标识并把检查次数当成 business 数量。
- 最小修正：保留 `license_no`，对通过检查的 `license_no` 做 `count_distinct`。
- 验证：**反事实已证实**。
- 改进方向：训练中必须区分题目实体名词与事实表行数；`count_distinct` 已经足够。

### 23. example 2868 · `works_cycles`

- 问题：列出使用 `Distinguish` card type 的 person。
- 可见证据：模型遵循外部映射过滤 `CardType='Distinguish'`，最终输出 Title、FirstName、MiddleName、LastName。
- 第一处不确定：题目只说 person，终止表选择了四列姓名表示，但没有证据说明 Title 和 MiddleName 是答案的一部分。
- 最小修正：优先投影任务定义的 person 字段；若外部知识只给 first/last，应保持原始两列而非自行扩展。
- 验证：**轨迹级判断**。
- 改进方向：建立跨数据集稳定的“person/name”输出表示规则。

### 24. example 1050 · `regional_sales`

- 问题：Stephen Payne 的订单中，净利润超过 1000 的比例。
- 可见证据：`Unit Price` 和 `Unit Cost` 都是含千位逗号的字符串。初始相减产生明显错误；步骤 10 只对 Unit Price 做 `REPLACE(',', '')`，却直接 `CAST(Unit Cost AS REAL)`。例如 `"4,130.01"` 被错误解析成 4。
- 第一处偏离：错误恢复只规范化一个操作数。
- 最小修正：两个字段都先移除逗号、转 REAL，再相减；其余总体、条件计数与 percentage 保持不变。
- 验证：**反事实已证实**。
- 改进方向：这是当前自由字符串表达式的真实接口摩擦。可考虑未来加入 typed numeric parse/subtract；短期 SFT 应包含“双操作数使用同一规范化”的样例。

### 25. example 6246 · `address`

- 问题：Delaware 中实行 daylight saving 的 residential areas 数量。
- 可见证据：外部知识明确说 Delaware 是 county；模型却先按 state=`DE` 解释，并进一步把 residential area 猜成 `residential_mailboxes>0`。过程中发生 3 次参数/列错误。
- 第一处偏离：忽略外部知识，把 county 改成 state。
- 错因：后续所有代理概念和连接都建立在错误地理层级上。
- 最小修正：先定位 county 字段/关系并验证 `Delaware` 的实际值，再定义 residential area 的实体粒度和 daylight saving 条件。
- 验证：**轨迹级高置信判断**。
- 改进方向：与 6489 同属显式映射被覆盖；通用约束可修复，不应写 Delaware 专用 prompt。

### 26. example 1589 · `public_review_platform`

- 问题：具有 `music_karaoke=true` 且 inactive 的 businesses 数量。
- 可见证据：模型严格使用外部映射，先得到 21 条属性 business，再筛 inactive 得到 3，并计数。
- 错因判断：可见轨迹没有明显字段、关系或运算错误。可能存在重复 business 行或 active 字段表示问题，但轨迹没有证据支持其中任何一个。
- 最小修正：无法确定；先审计 `business_id` 唯一性与期望输出。
- 验证：**未定位/标注待审计**。
- 改进方向：不要为这种自洽轨迹追加猜测式提示。

### 27. example 40 · `book_publishing_company`

- 问题：CA 州商店中销售 quantity>20 的所有 titles。
- 可见证据：模型按单条 sale 的 `qty>20` 过滤，连接 title，输出 4 个 distinct title。
- 已测试修复：根据外部知识增加 `stor_name`，输出 store name + title。
- 验证：**简单修复已证伪**；因此缺少 store name 不是主要原因。
- 剩余原因：更可能是“sales of quantity more than 20”需要按 title/store 汇总，而非单笔 sale 行条件，或结果去重粒度不同。
- 改进方向：聚合前必须明确问题问单笔事实还是实体汇总；不要继续试输出列组合。

### 28. example 2438 · `books`

- 问题：04/10/2022 下单的 orders 有哪些 status。
- 可见证据：8 个订单连接历史后得到 21 条状态记录。模型自行把问题改成“每个订单的 latest status”，随后连续触发：错误逻辑列 `join_002.status_id`、`limit=21` 超上限、以及未使用反馈中精确 dotted 列名的 `order_by`。
- 第一处语义偏离：题面只问 status 集合，没有要求 latest/current；在此之后才发生接口错误和循环。
- 最小修正：连接状态表后直接 distinct 投影 `status_value`，不做 latest-per-order。
- 验证：**反事实已证实**，得到 5 行状态并通过。
- 改进方向：错误反馈已经精确；核心不是列名摩擦，而是先发明了不需要的子任务。应训练“最小充分查询”。

### 29. example 5632 · `retails`

- 问题：日本 suppliers 中有 debt 的百分比。
- 可见证据：模型连接 supplier/nation，过滤 Japan，得到总数 412、debt 43，并计算 10.4369%。
- 错因判断：轨迹与外部映射表面一致；可见信息不足以判断 debt 是否需按 distinct supplier、其他债务字段或比例表示。
- 最小修正：无法从轨迹确定。
- 验证：**未定位/标注待审计**。
- 改进方向：先检查标注的 debt 定义与 supplier 唯一性，避免将其误归因给 scalar tool。

### 30. example 1167 · `professional_basketball`

- 问题：1971–1975 获 Coach of the Year 的 coaches 中，有多少曾在 POR。
- 可见证据：模型通过值观察把奖项字符串从错误大小写恢复为实际值，得到 5 名获奖者；之后把 coaches 表同时按 coachID 和 award year 连接，再过滤 POR，结果为 0。
- 第一处偏离：把“曾在 POR”强制限定为获奖同一年。
- 错因：题面没有同年约束；team 经历与 award 是同一 coach 的两个不同时间事实。
- 最小修正：先取获奖年份范围内的 distinct coachID，再仅按 coachID 连接全部任职经历，筛 POR，最后 count distinct coach。
- 验证：**轨迹级高置信判断**。
- 改进方向：训练关系角色和时间条件归属，不需要改 join API。

### 31. example 582 · `synthea`

- 问题：在某观察日期满足条件的患者姓名与年龄。
- 可见证据：模型按外部知识使用 observation year - birth year，并把 first/last 拼成一个 `name` 字符串。
- 第一处不确定：年龄计算遵守了给定公式；更可疑的是姓名输出被改造成单列文本，而外部知识只说明 full name 对应 first/last。
- 最小修正：保留 first、last、age 三个 grounded 字段，并避免自定义拼接；同时明确该数据集年龄是否只按年份差。
- 验证：**轨迹级判断**。
- 改进方向：原始字段优先的终止形态训练。

### 32. example 4869 · `works_cycles`

- 问题：工作 night shift 的 employees 百分比。
- 可见证据：外部知识定义分子为 `SUM(Name='Night')`、分母为 `COUNT(ShiftID)`，没有 current-only 条件。模型步骤 4 先过滤 `EndDate IS NULL`，把总体缩成当前任职记录，得到 52/290。
- 第一处偏离：无依据添加 current assignment 限制。
- 最小修正：在全部 `EmployeeDepartmentHistory` 上连接 Shift，让分子和分母共享同一输入关系，并按外部定义计数。
- 验证：**反事实已证实**，该替换通过。
- 改进方向：训练“不得为了自然语言合理性添加外部知识未给出的当前/有效限制”。

### 33. example 3664 · `movie_3`

- 问题：Children 类别中每天租金最高的影片/数值。
- 可见证据：模型先过滤 Children，再对每行计算 `rental_rate/rental_duration`，然后按派生 daily rate 排序；这正是外部知识定义的算子顺序。
- 错因判断：可见轨迹没有“先选最贵再除天数”的错误，旧的粗分类不成立。更可能是并列、浮点表示或输出列形态。
- 最小修正：无法从可见证据确定。
- 验证：**未定位/标注待审计**。
- 改进方向：审计隐藏期望与浮点比较；不要据此修改工具。

### 34. example 1461 · `public_review_platform`

- 问题：Mesa 中 review stars 大于 3 的 businesses。
- 可见证据：模型直接在 Business 表按 city=Mesa 和 stars>3 过滤并返回 business ID，和可见 schema/任务表述一致。
- 错因判断：可能的歧义是“review stars”指 business 汇总星级还是逐 review 星级，但轨迹没有证据说明应选后者。
- 最小修正：无法确定。
- 验证：**未定位/标注待审计**。
- 改进方向：检查标注所用星级字段及期望 business 表示。

### 35. example 1530 · `public_review_platform`

- 问题：具有 `Accepts Credit Cards` 属性的 business 百分比。
- 可见证据：模型分子是 10,464 条 value=true 的属性行，分母是 Business 表 15,585 行；两者没有先固定同一 business 实体粒度。
- 第一处偏离：直接把属性行数除以 business 行数。
- 已测试修复：把分母改成全部 `Business_Attributes.business_id` 行数。
- 验证：**简单修复已证伪**，说明“只换分母表”不够。
- 剩余原因：需要同时明确分子/分母是否 count distinct business、是否只以具有该 attribute 的 business 为分母、以及重复属性记录。
- 改进方向：统一 entity grain 后在同一关系做条件聚合；先审计分母定义，不要继续试单一数字。

### 36. example 2078 · `address`

- 问题：Grayson Alan 所代表 district 的 longitude 和 latitude。
- 可见证据：模型找到该代表及 district，连接后出现 62 个 zip 的经纬度。题面没有定义 district centroid；模型却自行计算 62 行的均值并作为唯一坐标。
- 第一处偏离：步骤 8 发明 mean 聚合。
- 已测试修复：保留全部 62 个经纬度对，并按题面写出的 longitude、latitude 顺序投影。
- 验证：**简单修复已证伪**；因此“去掉均值”虽然消除了无依据聚合，但还不是完整答案。
- 剩余原因：可能是列顺序、district-to-zip 重复、特定地址角色或标注输出口径。为避免标签探测不再试第二种。
- 改进方向：应审计题目/标注如何把多 zip district 映射为坐标；当前工具不应默认均值。

### 37. example 3011 · `hockey`

- 问题：列出满足 goalie/team 条件的人员姓名。
- 可见证据：模型已按 goalie 统计 team 数并连接 Master，但终止表仍带 `team_count`。
- 第一处偏离：排序/筛选辅助指标进入最终答案。
- 最小修正：只投影 `firstName`、`lastName`。
- 验证：**反事实已证实**。
- 改进方向：典型 exact-output-shape SFT 样例。

### 38. example 5544 · `legislator`

- 问题：Michigan 有多少 female representatives。
- 可见证据：`current-terms` 过滤 MI+rep 得到 94 个任期行；连接性别后有 6 行。模型直接 count rows。
- 第一处偏离：把任期行数当作代表人数；同一 `bioguide` 可有多个 term。
- 最小修正：对过滤后的 `bioguide` 做 `count_distinct`。
- 验证：**反事实已证实**。
- 改进方向：这是最标准的“人物实体 vs 任期事实”粒度样例；应纳入因果 SFT。

### 39. example 3131 · `world`

- 问题：1830 年独立国家的 official/unofficial languages。
- 可见证据：模型找到两个国家和 8 条语言，最终输出 country name、language、IsOfficial、Percentage。
- 第一处不确定：题目主要问语言及官方状态，模型把 country 和 percentage 也带入答案；同时多国并列是否要保留国家列存在歧义。
- 最小修正：依据统一输出合同精确投影；最自然的是 language+IsOfficial，但未用隐藏判分试探。
- 验证：**轨迹级判断**。
- 改进方向：多实体上下文中的必要标识列需要数据集级规范。

### 40. example 4906 · `image_and_language`

- 问题：含 broccoli 的数量是 tomato 的多少倍。
- 可见证据：外部知识明确要求按 `COUNT(OBJ_SAMPLE_ID)`；模型却对两个类别都使用 `count_distinct IMG_ID`，计算图像存在性比率。
- 第一处偏离：步骤 5 把 object sample 粒度改成 image 粒度。
- 最小修正：分别 count `OBJ_SAMPLE_ID`，再保持原方向 broccoli/tomato 做 divide。
- 验证：**反事实已证实**。
- 改进方向：与 6489 同类，显式聚合字段映射必须锁定；不需要新聚合工具。

### 41. example 1213 · `shakespeare`

- 问题：most recent work 中的 characters。
- 可见证据：模型用作品日期找最新作品，再通过 chapter/paragraph 路径取得 character；这只覆盖有台词/段落引用的角色，也可能包含 stage-direction 伪角色。
- 第一处不确定：把“作品中的所有 characters”代理成“在 paragraph 中出现的 character IDs”，且没有验证并列最新作品。
- 最小修正：先审计 schema 是否有直接 work-character 关系；若只能从 paragraph 推导，需显式处理伪角色、去重和并列。
- 验证：**轨迹级判断**。
- 改进方向：属于实体总体路径选择，不是协议错误。

### 42. example 4244 · `professional_basketball`

- 问题：满足总得分阈值且与 MVP 条件相关的 teams 数量。
- 可见证据：模型把所有年份按 `tmID` 汇总 points，几乎把整支 franchise 历史压成一个实体；MVP 关系也没有与获奖年份/球队赛季对齐。
- 第一处偏离：group-by 只有 `tmID`，丢失 team-season 粒度。
- 错因：球队得分是赛季事实，MVP 也是球员-年份事实；跨年汇总再求交会扩大合格集合。
- 最小修正：分别在正确 team-season 粒度构造得分达标集合和 MVP 所在队集合，再按匹配键求交，最后按题目要求计数。
- 验证：**轨迹级高置信判断**。
- 改进方向：加入复杂多事实粒度的因果样例；工具已有集合操作。

### 43. example 74 · `book_publishing_company`

- 问题：求 1994 年相关书籍的 average order quantity。
- 可见证据：外部知识对该题给出了非常规但明确的定义：order quantity 使用 `ord_num`，平均值为 `SUM(ord_num)/COUNT(title_id)`。模型改用数值更自然的 `qty`，并先按 title 汇总。
- 第一处偏离：选择 `qty` 覆盖了外部给定的 `ord_num` 映射。
- 错因：模型按照常识“纠正”了数据集语义，导致后续聚合目标完全不同。
- 最小修正：严格使用外部知识指定的字段和分子/分母。
- 验证：**轨迹级高置信判断**。
- 改进方向：通用“不要纠正显式 benchmark mapping”规则；同时应人工审计这条外部知识是否合理。

### 44. example 3724 · `movie_3`

- 问题：租赁次数前五的影片。
- 可见证据：模型正确统计并取 top-5，但终止表含 title 和 `rental_count`。
- 第一处偏离：把排名辅助计数作为答案列。
- 最小修正：只投影 film title。
- 验证：**反事实已证实**。
- 改进方向：终止形态训练即可。

### 45. example 2189 · `beer_factory`

- 问题：符合条件的品牌及这些产品售出的总金额。
- 可见证据：模型找到 6 个品牌，却在全部关联交易上计算一个全局总额 3531，然后 cross join 到 6 个品牌，使每个品牌都重复同一个全局数值。
- 第一处偏离：先做无 group-by 的全局 sum，再把标量广播到品牌行。
- 错因：品牌粒度和全局总体混合；结果看似有“品牌+金额”两列，但每行语义错误。
- 最小修正：若问题要求每品牌金额，应按 brand group sum；若要求品牌列表加一个全局总额，则应使用明确的两粒度输出合同，而不能 cross join 冒充每品牌值。
- 验证：**轨迹级高置信判断**。
- 改进方向：这是典型 grain mismatch；关系派生反馈可用于训练识别 global aggregate + cross join 风险。

### 46. example 1569 · `public_review_platform`

- 问题：工作天数最多的 businesses 及 category。
- 可见证据：模型按 business 计算 Monday–Saturday 工作天数，发现 6 天最大值对应 6,082 个 business；连接多对多 category 后扩成 19,077 行。
- 已测试修复：只移除 `days_worked`，保留 business_id+category_name。
- 验证：**简单修复已证伪**。
- 剩余原因：外部表达可能要求先在 category/business 联合粒度计数，或期望另一种实体/去重布局；问题不只是多带排序列。
- 改进方向：审计 category 在聚合前还是聚合后参与语义；不要继续猜终止列。

### 47. example 6299 · `menu`

- 问题：两个给定 Menu UUID 的总 dish count。
- 可见证据：目标 UUID 的 MenuPage 行可见，但其 `menu_id` 反复无法连接到 Menu；模型不断更换 read/filter 方式，最终 40 步耗尽，并出现一次 no-progress 错误。
- 第一处偏离：在多次一致的空连接证据后，仍假设换读取方式会改变关系事实。
- 错因：这是无法建立目标证据关系时的停止策略失败；也可能暴露数据外键不完整。
- 最小修正：在确认 unmatched 后停止重复探索，转而检查是否有其他直接 dish/page 路径；若没有，应把任务标为当前 catalog 不可回答，而非循环。
- 验证：**轨迹级高置信判断**。
- 改进方向：需要“空连接不可被重复 read 修复”的恢复样例，以及 fact-only unmatched feedback；不应延长 action budget。

### 48. example 796 · `soccer_2016`

- 问题：谁获得最多 Man of the Series。
- 可见证据：catalog 同时有 Match、Player 和 Season；模型只在 Match 的 `Man_of_the_Match` 上计数，完全没有检查 Season 的 `Man_of_the_Series`。
- 第一处偏离：步骤 1 选错事实表/字段，把 Match award 当成 Series award。
- 错因：字符串相似导致 schema selection 错误，后续聚合本身无误。
- 最小修正：先描述/检查 Season 中的 series award 字段，再验证其与 Player 的标识或名称关系。
- 验证：**轨迹级高置信判断**。
- 改进方向：训练“题面实体名优先匹配 schema 语义，不以最近似字段替代”。

### 49. example 5161 · `cookbook`

- 问题：含 cheese 的 recipes 中，calories>200 的百分比。
- 可见证据：外部知识指定 `COUNT(recipe_id)`；模型在 cheese ingredient→Quantity→Nutrition 的 105 行关系上改用 `count_distinct recipe_id`，得到 77.78%。
- 第一处偏离：步骤 9 把规定的行计数改成 recipe 去重计数。
- 最小修正：在同一固定 join 总体上，分子和分母都使用普通 `count Nutrition.recipe_id`。
- 验证：**反事实已证实**。
- 改进方向：显式 COUNT 与 COUNT DISTINCT 不可凭实体常识互换；同时人工审计外部知识是否确实希望重复 ingredient 行。

### 50. example 41 · `book_publishing_company`

- 问题：sales 中最高 quantity 的 store，再求该 store 最低 quantity。
- 可见证据：外部知识把最高/最低 quantity 定义为 `MAX(qty)`/`MIN(qty)`。模型按 qty 降序 `limit=1` 选一条最高销售记录，再按该 store 求最小值。
- 第一处不确定：如果最高 qty 有多条并列，`limit=1` 会任意选择一个 store；轨迹没有检查并列。
- 最小修正：先求 max qty 标量，等值过滤所有最高记录，再按明确的并列规则处理对应 stores。
- 验证：**轨迹级判断**。
- 改进方向：不是“必须按 store 汇总”的确定错误；首先要处理极值并列。

### 51. example 1888 · `legislator`

- 问题：比较 1875 与 2005 的 legislators 数量。
- 可见证据：模型从 historical terms 取 1875，从 current terms 取 2005，各自计 distinct，并输出两个并列计数。
- 第一处不确定：`current-terms` 是当前人物的所有任期，而非 2005 年完整历史总体；两个年份来自不同人口分区。题面 “compare” 的输出又可能指两数、差值或倍数。
- 最小修正：先用同一完整历史总体表达两个年份，再按数据集统一 compare 形态输出。
- 验证：**轨迹级判断**。
- 改进方向：优先解决总体可比性，不能只改最终算术形式。

### 52. example 5147 · `food_inspection`

- 问题：某 postal 条件下，低分 business 中有 Low Risk violation 的比例。
- 可见证据：模型把低分 inspection 连接 business 后去重为 332 个 business，再把任意 Low Risk violation 的 business 去重为 316，计算 316/332。
- 第一处不确定：外部定义使用 `COUNT(business_id)`，而模型改成 entity-existence 语义；同时 score<95 与 violation 可能来自不同 inspection 事件，跨事件连接会制造“曾经低分且曾经 low risk”。
- 最小修正：固定一条 inspection/violation 对应关系和同一行粒度，再按外部指定的 count 口径做条件聚合。
- 验证：**轨迹级高置信判断**。
- 改进方向：加入事件共时性和 count-vs-distinct 的训练样例。

### 53. example 1560 · `public_review_platform`

- 问题：较长 tip 是否会带来较少或较多 likes。
- 可见证据：模型连接正确 category 后，按 `tip_length` 分组计算 mean likes，随后在终止 reason 中给出“影响”结论。
- 第一处偏离：题面没有规定 mean、相关系数、趋势比较或分桶方式；模型任意选择 mean-by-length。最终表本身也没有一个 grounded 的“more/less”结论列。
- 错因：这是统计判据缺失与终止 grounding 双重问题，不是简单 SQL 执行。
- 最小修正：先定义可验证的影响指标和输出形态；若数据集要求相关方向，当前工具面可能缺少直接 correlation/趋势操作。
- 验证：**轨迹级高置信判断**。
- 改进方向：先审计任务期望统计量，再决定是否增加 typed correlation 工具；不能用 reason 文本替代结果。

### 54. example 4038 · `synthea`

- 问题：哪种 gender 有更多目标 condition，并分别给出数量。
- 可见证据：模型按 gender 统计条件记录，得到 F=553、M=572，并以两行 `gender,count` 结束。
- 已测试修复：取 top-1，只输出赢家及数量。
- 验证：**简单修复已证伪**，说明题目不只是问赢家。
- 剩余原因：可能要求按 patient 去重，或要求一行固定列布局同时给出男女数量；“分别给出”支持后者，但未继续试探。
- 改进方向：先统一 condition row 与 patient entity 粒度，再使用 `output_layout="columns"` 表达固定男女列。

## 反事实验证汇总

共测试 27 条，每题仅允许一个由可见证据预注册的替代动作路径：

- **16 条通过**：6489、593、4189、2408、5440、6454、3969、2438、2513、1050、4869、3011、5544、4906、3724、5161。
- **11 条未通过，简单假设被证伪**：3688、2901、4848、5668、4426、2418、40、1530、2078、1569、4038。
- 未通过不代表原轨迹正确，只表示被测试的单一修复不是充分条件。为避免利用隐藏判分器爬山，本次没有为这些题尝试第二个候选。

## 对当前 atomic 工具设计的归因

### 可以靠通用训练约束改善

1. **外部知识映射不可无证据改写**：6489、6246、74、4906、5161。
2. **实体粒度必须在聚合前声明清楚**：2513、5544、4244、5147、1530。
3. **不得添加题面没有的限制**：593、2438、4869、1167。
4. **终止表只含答案列**：2408、6454、3969、3011、3724。
5. **先固定关系总体，再取极值**：5440、214，以及多条并列未处理轨迹。

这些都适合通过因果 SFT 学习：错误动作收到环境事实后，教师给出一条局部修正，而不是把完整 gold 路径倒灌到之前的思考。

### 值得考虑的通用工具改进

1. **typed numeric parsing/arithmetic**：1050 显示自由 SQL 字符串表达式容易只清洗一个操作数。可以研究受限的 `parse_number` 或 typed row arithmetic，但应先做小规模门控实验。
2. **显式极值并列语义**：多题混淆 top-1 与全部最大值。未来可研究 `with_ties`，或者让现有 max-scalar+filter 路径在训练中更常见。
3. **完整总体暴露**：6492 若确实缺少完整 image 主表，任何 prompt 都无法计入零 object 的图像。
4. **统计关系工具**：1560 若标注明确要求相关方向，当前算子面可能缺少 grounded correlation；先审计标注再设计工具。

### 不应继续通过 prompt 或接口猜测

715、5053、886、1589、5632、3664、1461 等轨迹与可见证据基本一致但仍判错。它们应进入 benchmark/output-contract 审计队列。继续增加“可能要去重、可能要排序、可能要换列”的通用 prompt，会同时制造控制题回归。

## 最终判断

当前 atomic 工具并没有到“接口能力上限”，但已经明显进入**语义决策与数据标注主导**的阶段。54 条失败中，真正可确认的接口摩擦很少；相反，16 条可以在不改工具的情况下由一个局部、可学习的决策修复。下一轮最有价值的工作不是继续堆 prompt，而是：

1. 把这 16 条已验证修复转成严格因果、可回放的教师示范候选；
2. 对 11 条简单假设已证伪的题停止判分试探，做人工语义/标注审计；
3. 对 trace-consistent 失败单独审计 benchmark；
4. 只对 numeric parsing、极值并列和缺失总体这三类通用能力做小规模工具门控。
