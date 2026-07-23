# BIRD version11 失败轨迹逐题审计与 resident-plan 实验

日期：2026-07-23

## 范围与判定方法

审计对象是固定 200 题、DeepSeek v4 Flash、rolling legal history=4 的 version11
结果：

- 输入：
  `data/eval_inputs/bird_train_tool_interface_validation200_version4.jsonl`
- 轨迹：
  `data/trajectories/bird_train_version11_join_validation200_flash_rolling_json_unambiguous_success.all.jsonl`
- strict-multiset：130/200 正确，70/200 失败。

逐题同时检查 question、external knowledge、完整合法/错误工具轨迹、预测 denotation 和
gold SQL。gold SQL 只用于离线审计和 verifier，没有进入模型 prompt。

主分类如下：

- `K-SHAPE`：模型已经找到了正确实体/集合/数值，但最终列、列顺序、表示、大小写、
  精度或一行/多行形状错误。
- `K-REL`：模型理解了目标，但关系代数、工具参数、操作顺序、join、distinct、分组粒度
  或引用方式错误。
- `M-SEM`：模型选择了错误的实体、关系、分母、统计对象或问题解释；仅修工具参数不足以
  得到正确答案。
- `G-GOLD`：问题、external knowledge 和 gold SQL 在输出槽、重复行、LIMIT tie、计数粒度
  或实体映射上明显冲突/欠规定；不能据此断言模型或工具失败。
- `P-PROTOCOL`：没有形成可判定的语义终局，主因是 provider carrier/protocol。

这是一项人工因果诊断，不是自动 scorer 标签。若一题同时有多个问题，表中给出最早且
足以造成最终错误的主因。

## 总体结论

| 主分类 | 题数 | 占 70 个失败 | 占固定 200 |
|---|---:|---:|---:|
| K-SHAPE | 21 | 30.0% | 10.5% |
| K-REL | 13 | 18.6% | 6.5% |
| M-SEM | 17 | 24.3% | 8.5% |
| G-GOLD | 18 | 25.7% | 9.0% |
| P-PROTOCOL | 1 | 1.4% | 0.5% |

按用户关心的二分法，至少 **34/70** 是“模型基本知道怎么做，但工具执行或最终证据表做错”；
**17/70** 是模型对问题、关系或统计粒度的理解本身不正确。另外 **18/70** 不应该直接归因
给模型/工具，因为 prompt 与 gold 的语义边界不一致；1 题仅凭当前轨迹无法判断。

因此，130/200 到 150/200 的主要可控路径不是继续改 join 语法，而是：

1. 消掉正确集合上的 exact-output 错误；
2. 为条件计数、比例、差值提供不丢失分母和 join 多重性的原子表达；
3. 对复杂题显式固定 grain、denominator、output slots；
4. 在训练前隔离 gold/prompt 冲突题。

## 逐题审计

### Easy（21）

| ID | 分类 | 轨迹相对 gold SQL 的决定性差异 |
|---|---|---|
| 00004 | K-REL | 知道要比较“更新时间是否超过创建后十年”，但用字符串 `REPLACE` 构造时间戳阈值；gold 比较年份差，阈值表达式改变了语义。 |
| 00593 | K-REL | 患者、药品和原因均正确；读取到两次疗程后又固定到第一个 encounter，只留下 11 天，丢失 gold 的第二个 18 天疗程。 |
| 00715 | G-GOLD | external knowledge 明确 address 包含 `street_num, street_name`，模型返回完整地址；gold 只返回 `street_name`。 |
| 01152 | K-SHAPE | 正确找到最年轻获奖者 Kyrie Andrew Irving，却把三列姓名拼成一列并遗漏 middle name。 |
| 02408 | K-SHAPE | 正确过滤 George 作者，但终局仍携带 `author_id`；gold 只要 `author_name`。 |
| 02901 | K-SHAPE | 过滤、join、年龄排序和 top-10 正确，但把 First/Middle/Last 三列拼成一列。 |
| 02925 | G-GOLD | 模型找到最小成本产品后返回产品名；问题问“which product”，gold 返回 `ProductID`，输出槽欠规定。 |
| 03688 | G-GOLD | 模型返回所有最长影片对应库存；gold 的 `ORDER BY ... LIMIT 1` 任取一条库存。问题没有规定 tie-breaking。 |
| 04163 | K-REL | 找到 `avg_revenue=4` 的导演和 genre，但用分组去重 genre；gold 保留 join 产生的重复 genre。 |
| 04189 | K-REL | Free/Sports 过滤和输出列正确，但使用 left join；gold 是 inner join，加入没有 review 的 App。 |
| 04426 | K-SHAPE | 正确定位 Oscar Cervantes 的 credited episodes，却输出 season/episode/title，而不是已有的 `episode_id`。 |
| 04598 | K-REL | 正确连接零人口 zip 和 alias，但没有 distinct，保留重复 alias。 |
| 04822 | G-GOLD | question 和 external knowledge 都指定 `TEAM='Avangard Omsk'`，模型照做得到空集；gold 却过滤 `TEAM='Czech Republic (all)'`。 |
| 04848 | K-SHAPE | top-3 和排序正确，但终局保留 Employee ID 和 SickLeaveHours；gold 只要 JobTitle。 |
| 05053 | G-GOLD | 模型返回国家名 Laos；问题问 nation，gold 返回数据库代码 `LAO`。 |
| 05440 | K-REL | 先在全部 Paper 上取 max Year，再 join Journal；gold 先 inner join 可匹配 Journal 的论文后排序，因此模型被无 Journal 的异常 Year 行带偏。 |
| 05668 | G-GOLD | 最大差值 280 正确，模型返回 vendor name；gold 返回 VendorID，问题“to which vendor”未规定 ID 还是名称。 |
| 05952 | G-GOLD | 模型返回唯一 county；gold 因 city join 返回两条完全相同 county，单数问题没有要求保留重复。 |
| 06454 | K-SHAPE | 比较结果正确为 RAIL，但模型把存储值改成小写 `rail`。 |
| 06489 | M-SEM | external knowledge 明确“object”指 `OBJ_SAMPLE_ID`；模型找到该 row 后仍输出 object class `paper` 而不是 ID 18。 |
| 06492 | G-GOLD | question/external knowledge 要按 IMG_ID 分组并筛选 `COUNT(OBJ_SAMPLE_ID)<15`，模型这样执行；gold 实际是逐行 `OBJ_SAMPLE_ID<15` 后计数。 |

### Medium（25）

| ID | 分类 | 轨迹相对 gold SQL 的决定性差异 |
|---|---|---|
| 00040 | K-SHAPE | CA、qty>20、titles/stores join 均正确，最终只投影 title，漏掉 gold 要求的 qty。 |
| 00214 | G-GOLD | 模型找出最高 popularity 电影的全部 keywords；问题使用复数，gold 却在 keyword join 后 `LIMIT 1` 只任取一个。 |
| 00582 | K-SHAPE | 病人、2011、血压 200mmHg、年龄均正确；把 first/last 拼成一列，gold 要三列 first/last/age。 |
| 00886 | M-SEM | 正确找到 Austria code 后走 `Indicators`，没有走 gold 的 `CountryNotes.Seriescode -> Series.Topic`，得到完全不同 topic 集。 |
| 01050 | K-SHAPE | numerator/denominator 正确，手写终值时把 28.975265… 四舍五入为 28.98。 |
| 01167 | M-SEM | 奖项和年份过滤正确，却 join `coaches` 并按 coach/team 过滤；gold join `teams` 只按 year 匹配 POR。 |
| 01461 | M-SEM | 把 `Business.stars>3` 当成条件；gold 要 `Reviews.review_stars>3` 后按 business 去重。 |
| 01530 | M-SEM | numerator 使用目标 attribute association，denominator 却用全部 Business；gold/external knowledge 的分母是 joined Business_Attributes 行。 |
| 01589 | G-GOLD | external knowledge 明确 `attribute_value='true'`，模型遵循；gold SQL 却要求 `none/no/false`。 |
| 01692 | K-SHAPE | 39/59 的逻辑与 gold 一致，只把 66.101694… 手写成 66.1。 |
| 02078 | M-SEM | 正确找到代表和 zip rows，却把所有经纬度聚合成均值并交换输出顺序；gold 保留每个 zip 的 latitude/longitude rows。 |
| 02418 | K-REL | 按 `book_id` 计数取 top-1；gold 按 `title` 分组，同名不同 ID 的订单必须合并。 |
| 02438 | G-GOLD | 模型把“order status”解释为每个订单的最新状态并携带 order_id；gold 返回当天订单历史中出现过的所有 distinct status。问题未说明 current/history。 |
| 02513 | K-REL | 所有过滤条件正确，但计 inspection rows=211；gold `COUNT(DISTINCT license_no)`=203。 |
| 02682 | M-SEM | 直接读取 `Product.StandardCost`；gold 对同一产品的 `ProductCostHistory.StandardCost` 求平均。 |
| 02868 | K-SHAPE | Distinguish 卡持有人集合正确，终局多出 BusinessEntityID 和 middle name；gold 只要 first/last。 |
| 03042 | K-SHAPE | NJD、sum(SA-GA)、top goalie 都正确，把 first/last 拼成 `Martin Brodeur` 一列。 |
| 03131 | K-SHAPE | 正确找到 1830 独立国家和语言，但多输出 Country.Name，且没有按 gold 的 Language/IsOfficial 形状跨国去重。 |
| 03664 | G-GOLD | external knowledge 把答案定义为 `MAX(rental_rate/rental_duration)`，模型返回该比值；gold 用该比值排序，却返回原始 `rental_rate`。 |
| 03969 | K-SHAPE | 正确找到最大年龄记录和存储值 `F`，但翻译为 `female`。 |
| 04869 | M-SEM | 改为 distinct current employees，并用 Employee 当前人数作分母；gold/external knowledge 统计全部 EmployeeDepartmentHistory shift rows。 |
| 05554 | M-SEM | 已正确过滤 state=NY、type=Post Office，又错误增加 city='New York'；gold 指纽约州。 |
| 05632 | G-GOLD | external knowledge 定义“日本供应商中负债者比例”，模型按此算 10.4369%；gold 的 WHERE 先限定负债，再算负债供应商中日本占比 4.7149%。 |
| 06165 | K-REL | Journal/Year/Paper 初始路径正确，但误把 `PaperAuthor.Name` 再解释为需 join Author，之后在空/错误 handle 间循环，最终 carrier protocol 失败。 |
| 06246 | M-SEM | external knowledge 明确 Delaware 是 county，模型仍在 state/alias 中搜索 Delaware，没有使用 gold 的 county relation；最终 carrier protocol 失败。 |

### Hard（24）

| ID | 分类 | 轨迹相对 gold SQL 的决定性差异 |
|---|---|---|
| 00041 | M-SEM | 把“销量最高的商店”理解成含最大单笔 qty 的商店；gold 是按 store `SUM(qty)` 排名，再在该 store 内按 title `SUM(qty)` 取最小。 |
| 00074 | M-SEM | 初始 plan 已把目标写成“1994 每 title 总量与这些 title 的平均比较”；gold 是 1994 每条 sale.qty 与全时期全 sales 的全局平均比较，且模型后来未真正用平均值过滤。 |
| 00128 | K-REL | 理解需要最高 priority，却直接假定 max=2，没有先 aggregate max，再用 value_ref 过滤。 |
| 00600 | K-SHAPE | male=180、female=193 两个计数正确，却编码成一个字典单元格；gold 是同一行两列。 |
| 00796 | G-GOLD | external knowledge 要统计 `Man_of_the_Match`，模型照做得到 CH Gayle；gold 用 Season.Man_of_the_Series 与 Match.Man_of_the_Match 的非常规 join 得到 SR Watson。 |
| 01213 | G-GOLD | 模型找到最新 work Henry VIII 并返回全部 character names；问题是复数，gold 在多表 join 后 `LIMIT 1` 任取 Chorus。 |
| 01560 | M-SEM | 模型按 tip_length 求 average likes 并输出自然语言因果判断；gold 要每个 tip_length 的 `SUM(likes)` 三行表。 |
| 01569 | M-SEM | 按 business 统计工作天数再列其 categories；gold 按 category_name 聚合所有 Business_Hours 并取总 count 最大 category。 |
| 01888 | M-SEM | 对 1875 和 2005 都使用 historical-terms、返回两个 distinct counts；gold 用 current-terms 的 2005 数减 historical-terms 的 1875 数。 |
| 02088 | M-SEM | external knowledge 指定两类 row count 相减；模型按问题字面改成 distinct city count，得到 -61 而不是 -64。 |
| 02189 | K-REL | 正确找到 6 个品牌和总交易额 3531，却把全局总额复制到每个品牌；gold 要按 BrandName 分组求各自总额。 |
| 02507 | K-SHAPE | 正确找到 David Hodges，把 first/last 拼成一列。 |
| 02512 | K-SHAPE | 正确找到 Ruth Noble，把 first/last 拼成一列。 |
| 03011 | K-SHAPE | 球员集合正确，把 first/last 拼成一列。 |
| 03636 | K-REL | actor、language 过滤路径正确，但把单行 read_subtable 当 scalar `value_ref` 连续调用；应使用 `in_table` 或直接 join，三次 execution error 后终止。 |
| 03724 | K-SHAPE | top-5 影片及排序正确，终局保留辅助 `num` 列；gold 只要 title。 |
| 04038 | K-SHAPE | F=553、M=572 正确，输出为两行 gender/count；gold 是一行 female_count/male_count 两列。 |
| 04244 | G-GOLD | 模型按“球队总得分>=3800 且曾有 MVP”分别求集合再求交；gold 在 MVP player join 后只累计这些 player 的 points，并跨年份按 team name 聚合。自然语言未明确这一乘法多重性。 |
| 04906 | M-SEM | external knowledge 指定计 `OBJ_SAMPLE_ID` rows，模型改为 distinct IMG_ID，因而 broccoli/tomato 比值口径错误。 |
| 05147 | G-GOLD | 模型按自然语言统计 distinct businesses，gold 对 violations×inspections join rows 作加权比例；同一 business 的重复违规行被重复计数。 |
| 05161 | G-GOLD | 模型先去重 cheese recipe 再算 70/90；gold 在 cheese ingredient join 上保留同一 recipe 的多重 ingredient rows，得到 84/105。 |
| 05544 | K-REL | state/type/gender 全正确，直接 count join rows=6；gold 按 bioguide 去重后 count representatives=3。 |
| 05873 | K-SHAPE | 找到正确两个 Repo stars，手工百分比得到 300.26；精确 grounded 算术应为 300.1500938…。 |
| 06299 | P-PROTOCOL | 已找到两个 uuid 对应 Menu rows，但在聚合/差值前连续出现 split carrier 空 visible content，未形成可判定终局；此外问题/external knowledge没有明确 gold 使用 subtraction。 |

## 错误簇对应的工具改进

### 1. 先验证 version13 的 exact-evidence 设计，不再增加一个同义 `select`

21 个 K-SHAPE 不是缺少关系算子，而是模型把正确数据手写、拼接、翻译、四舍五入，或引用
了仍带辅助列的表。当前 version13 已经做了两项直接针对性修复：

- terminal 只接受 evidence table，不接受模型手写 answer；
- prompt 明确输出槽、禁止拼名/翻译/改大小写/四舍五入，并提供 exact project 示例。

这正好覆盖大部分 K-SHAPE。此时再增加一个与 `project` 同义的 `select`，会重新引入竞争
动作而不增加表达能力。下一步应先在稳定 provider carrier 下完成同一 200 的 version13
对照，确认这 21 题中实际回收多少。

### 2. 为 `group_aggregate` 增加条件聚合，而不是增加多个专用百分比工具

失败中反复出现 `COUNT(CASE WHEN ...)`、条件 SUM、两组差值和比例。当前模型必须拆成
多次 filter + aggregate + cross join + scalar_compute，容易改变 denominator、distinct
和 join multiplicity。建议的最小扩展是让每个 aggregation 可选一个 `where`：

```json
{
  "tool": "group_aggregate",
  "arguments": {
    "table": "joined_rows",
    "group_by": [],
    "aggregations": [
      {"op": "count", "column": "*", "where": {"column": "type", "op": "=", "value": "A"}, "as": "a"},
      {"op": "count", "column": "*", "where": {"column": "type", "op": "=", "value": "B"}, "as": "b"}
    ]
  }
}
```

这仍是一个原子的“在固定输入 grain 上聚合”动作；后续只需 `scalar_compute` 做
subtract/divide/percent。它直接针对 01530、04869、02088、04906、05147、05161、01888
等失败，而不把每种 SQL CASE 模式做成独立工具。

### 3. 把 grain/multiplicity 作为计划控制项，而不是让 harness 猜 gold

distinct 与 join 多重性相关错误至少出现在 04163、04598、02513、05147、05161、05544。
harness 不能根据隐藏 gold 告诉模型“这里应该 distinct”，否则会泄漏监督。较安全的做法是
让复杂题的 resident plan 在执行前明确三个控制目标：

1. 结果统计单位是什么（row / business / representative / recipe / image）；
2. denominator 在哪个 join/filter 前后定义；
3. 最终输出有哪些独立列。

这些是控制状态，不是事实证据；仍符合 plan 不承载结果值的约束。

### 4. 不再继续重构 join 参数，先补针对性 cookbook 和错误反馈

这 70 题里，直接由 version11 `base + joins[]` 参数形状造成的终局失败很少。最明确的是
06165 的错误关系选择和 03636 的 `value_ref`/`in_table` 混淆，前者不是 join JSON
复杂度，后者也不是 join。继续改 join 会引入新的引用链风险。

更低风险的改进是：

- 增加“先 join，再 rank/filter/project”的完整示例，避免 05440 的 operator ordering；
- execution error 对非 scalar `value_ref` 给出一条可复制的 `in_table` 与 join 示例；
- 对 group top-1 提供固定的 `group_aggregate -> extreme_value_select -> project` 示例；
- 保持 derived handle 的 flat logical namespace，不恢复递归 handle 前缀。

### 5. gold/prompt 冲突题必须在训练数据构造前隔离

18 个 G-GOLD 题不应作为工具设计或模型能力的干净监督。特别是 04822、06492、01589、
05632、00796、03664，external knowledge 与 gold SQL 直接给出不同逻辑。建议在 SFT
候选入口加入：

- external-knowledge-to-gold consistency 审计；
- plural/singular 与 `LIMIT 1` 冲突审计；
- natural entity 与 ID/code 输出槽审计；
- 问题按实体计数而 gold 依赖 join 多重性的审计。

这些样本可以保留作 benchmark compatibility 分析，但不能进入“模型直觉工具调用”的
训练集。

## 强制 resident plan 实验

### 实现

`generate_teacher_rollouts.py` 新增实验开关：

- `--plan-policy optional`：现有默认行为；
- `--plan-policy required-resident`：
  - 首个合法动作必须是包含 2–4 个 create/add 子目标的 `plan`；
  - 至少一个非 plan 工具成功后，terminal 前必须用批量 `update` 更新计划；
  - 多个 item 的进度在同一个 ops 数组中更新，不允许逐 item 连续调用 plan；
  - 没有新的非 plan 成功动作时，拒绝重复的 plan update；
  - plan 仍消耗真实 action budget；
  - plan action 不写入 rolling assistant/tool history；
  - harness 的 `EnvironmentState.plan` 是唯一持久载体，每轮 CURRENT ENVIRONMENT STATE
    都完整呈现当前计划。

这样可以把“计划是否常驻”与“rolling 窗口是否偶然保留 plan 调用”分离。

第一版 smoke 还发现，canonical plan evidence 会保存被引用 step 的完整输出；如果该 step
是 `describe_table`，model-visible plan 会在后续每轮重复一整份 schema。现已只在
model-visible plan evidence 中保留 `step_id + tool`，goal/status 仍完整常驻，canonical
快照中的完整 grounded output 不变。事实数据继续从 resident tables/values 读取。

### Provider carrier 预检

首次 JSON Output 预检中，optional 与 required 两组均出现 HTTP 200、`finish_reason=stop`、
非空 `reasoning_content`，但 visible content 为空。该批次在语义动作前失败，已标记为
provider carrier 诊断，不能计入 plan 对照。

为避免将 provider 行为误判为 plan 效果，生成器新增显式
`--deepseek-carrier json-output|tool-call`。两者都固定 thinking enabled 和
reasoning_effort=high；`tool-call` 只移除 JSON Output constraint，并使用单一、带实例的
`reasoning_content + <tool_call>` carrier。carrier 写入每轮审计和 manifest。

### 配对结果

先冻结了 12 个 hard、且可能由规划改变的 version11 失败题：

`00041, 00074, 00128, 01560, 01569, 01888, 02088, 02189, 03636, 04906, 05544, 06299`。

按“小样本先行”的停止规则，先对其中 6 题做配对 pilot：
`00074, 01569, 02088, 02189, 04906, 05544`。两组统一 DeepSeek v4 Flash、
temperature=0、max tokens=2048、max steps=30、
rolling legal history=4、同一工具协议和 tool-call carrier；唯一语义控制变量是 plan
policy。

有效结果目录（前述 JSON Output 空-content 诊断目录不计入）：

- `data/trajectories/plan_ablation_20260723/smoke_optional_tool_call_success.*`
- `data/trajectories/plan_ablation_20260723/smoke_required_resident_compact_tool_call_success.*`
- `data/trajectories/plan_ablation_20260723/pilot4_optional_tool_call_success.*`
- `data/trajectories/plan_ablation_20260723/pilot4_required_resident_tool_call_success.*`

| 指标 | optional plan | required resident plan |
|---|---:|---:|
| strict-multiset correct | 0/6 | 0/6 |
| legal terminal | 5/6 | 4/6 |
| 平均 action 数 | 12.17 | 14.17 |
| 平均 recoverable error 数 | 1.83 | 2.00 |
| 总 token | 375,124 | 444,670 |
| 平均 token/题 | 62,521 | 74,112 |
| 成功 plan 调用总数 | 2（1 题自发使用） | 18 |

required 相对 optional：

- 正确率没有任何提升；
- legal rate 从 83.3% 降到 66.7%；
- 平均 action 增加 16.4%；
- 总 token 增加 18.5%。

逐题也没有出现“plan 改正了语义”的信号：

| ID | optional | required | 关键观察 |
|---|---|---|---|
| 02088 | -61，10 步 | -61，17 步 | 两组都保持错误 city/count grain。 |
| 05544 | 6，8 步 | 6，10 步 | 两组都没有按 representative 去重。 |
| 04906 | 0.77918，9 步 | 0.77918，12 步 | 两组都按 distinct IMG_ID 而非 OBJ_SAMPLE_ID rows。 |
| 01569 | 错误 business/category rows，19 步 | 同类错误 rows，16 步 | plan 没有把 grain 改成 category 聚合。 |
| 02189 | 全局总额 3531，15 步 | protocol failure，20 步 | required 反而未形成 terminal。 |
| 00074 | protocol failure，12 步 | protocol failure，10 步 | 两组都未完成全局平均比较。 |

这里的结论不是“plan 永远无效”，而是**当前强制的 model-authored plan/update 对这类失败没有
提供增量信息**：错误在初始 semantic grain/denominator 中，模型会把同一个错误理解写入
resident plan，后续持续可见只会稳定错误方向。即使修复了 plan evidence 的 prompt 膨胀，
强制 plan 仍增加动作和 provider/protocol 暴露面。

因此按预先设定的小样本停止原则，不把该策略扩到剩余 6 题或 200 题，也不把
`required-resident` 设为默认。保留它作为显式实验开关；如果未来重测，plan prompt 必须先
要求写出 `grain / denominator / exact output slots`，并应与条件聚合工具扩展一起做因子实验，
否则只是把原有误解持久化。
