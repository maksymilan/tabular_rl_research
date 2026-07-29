# Atomic 三方案 Gate32 语义与逻辑质量逐题审计

## 审计目标

本报告比较同一冻结 32 题上的三个 DeepSeek v4 Flash 方案：

- **B**：历史 atomic version38，按需 `describe_table` 与 `inspect_column`；
- **F**：version39，完整 schema + BIRD 描述 + 每列两个示例值，同时移除
  `describe_table` 和 `inspect_column`；
- **I**：version39，完整上下文，保留 `inspect_column`，只移除 `describe_table`。

原始指标为 B=18/32、F=19/32、I=17/32 `bird-set`。本次不把 verifier 分数直接当作
推理质量，而是从模型可见问题、外部知识、工具动作和环境反馈逐题检查：

1. 最终实体、数值或集合在语义上是否正确；
2. 人口、行粒度、连接、聚合、排序和公式是否一致；
3. 是否添加题目未要求的 latest、distinct、left join、有效值筛选等限制；
4. 观察结果是否真正支持下一步；
5. 最终失败是否仅为输出列数、姓名拼接或附带 ID 等形态问题。

审计没有读取或引用 gold SQL、gold rows 或隐藏答案。

## 记号

- `✓`：verifier 正确，语义路径也成立；
- `形`：核心答案正确，失败主要来自输出形态；
- `规`：路径符合模型可见题意/外部知识，但 verifier 仍不同，存在数据或标注歧义；
- `✗`：存在实质语义或关系逻辑错误；
- `—`：接口错误后没有形成合法终止答案。

“最佳”评价优先考虑语义和逻辑正确，其次才考虑动作数、观察充分性和接口稳定性。

## 逐题比较

| ID / 数据库 | B | F | I | 语义与逻辑比较 | 最佳 |
|---|---:|---:|---:|---|---|
| 128 `retail_complains` | ✓ | ✓ | ✓ | 三者都按 priority=2 连接客户并返回 first；F 省去已知 schema/value 检查。 | F |
| 593 `synthea` | ✗ | ✗ | ✗ | 环境返回两次都满足药物和病因的用药记录，三者都无题目依据地选择第一次住院。F 最终标量最干净，但 singleton 假设仍错误。 | F（仅相对） |
| 600 `synthea` | ✓ | ✓ | ✓ | 三者都保持 hypertension population 并比较男女。B 先过滤再连接且按 patient 去重，人口/实体粒度最稳。 | B |
| 1050 `regional_sales` | ✗ | ✗ | ✓ | B/F 自行把 Order Quantity 乘入净利润，违反显式 `Unit Price - Unit Cost`；I 遵守公式并共享同一订单分母。 | I |
| 1152 `professional_basketball` | 规 | 规 | 规 | 三者都按 `award` 筛选、以最大 birthDate 找 youngest，并返回外部知识明确指定的 playerID。可见规范下逻辑一致，verifier 不同不像普通格式问题。F 最短且无多余 inspect。 | F |
| 1692 `simpson_episodes` | ✗ | ✓ | ✗ | B 用 Award 上的 distinct person 作分母、joined rows 作分子，粒度漂移；I 发现 10 条 unmatched 后改 left join，把无人物信息记录计入分母；F 在同一 inner-joined population 做两个条件计数。 | F |
| 2088 `address` | ✗ | — | ✓ | B 计算 distinct city，而显式公式要求按 type 计数；F 把 `read_subtable` 当 distinct 工具后退出；I 用 inspect 确认值域并在一个 joined population 做两个计数。 | I |
| 2408 `books` | ✓ | ✓ | 形 | 三者都得到全部 `George%` 作者；I 额外返回 author_id，核心姓名集合未错。F 三步完成且输出准确。 | F |
| 2438 `books` | ✗ | — | — | B 能完成连接，却自行把“订单状态”改成“每单最新状态”；题目没有 latest。F/I 在派生列 namespace 上失败，且也沿用 latest 假设。 | B（仅相对） |
| 2507 `food_inspection_2` | ✓ | ✓ | ✓ | 三者都按 employee 分组计数后取最大。I 在连接姓名前先完成排名，避免缺失 employee 行改变 argmax，且比 B 少 schema 动作。 | I |
| 2512 `food_inspection_2` | ✓ | 形 | 形 | 三者找到同一员工；F/I 把 first/last 拼成一个 `full_name`，属于纯输出形态。B 保留题目映射要求的两个字段。 | B |
| 2513 `food_inspection_2` | ✓ | ✓ | ✓ | 都正确固定 David Hodges、inspection type 和 Pass，并以 distinct license 表示 businesses。F/I 更短。 | F=I |
| 2682 `works_cycles` | ✓ | ✗ | ✗ | B 继续进入 ProductCostHistory，对目标产品的多条历史 StandardCost 求平均；F/I 直接对 Product 当前单行成本求“平均”，信息虽齐全却选错关系。 | B |
| 3011 `hockey` | ✓ | ✓ | ✓ | 都在 2000–2005 范围按 goalie 统计 distinct teams 并筛选 >2。I 无冗余 plan/schema 动作。 | I |
| 3042 `hockey` | 形 | ✓ | 形 | 三者最终识别同一 goalie。B/F 先按 goalie 汇总 NJD 历史 saves；I 对单赛季记录直接取最大。B/I 的 verifier 失败主要是把 first/last 拼成一列，但 I 的时间粒度也更脆弱。 | F |
| 3636 `movie_3` | ✓ | ✓ | ✓ | 三者都正确锁定演员与 English language，再统计交集。F 读取 language ID 后使用，grounding 比 I 直接写入 ID 更清晰。 | F |
| 3724 `movie_3` | ✗ | ✗ | ✗ | 三者都按 rental row count 排名；显式外部知识却把 most rented 映射为 MAX(inventory_id)。这是共同的指令遵循失败，不是输出多一列。 | F（仅相对） |
| 3969 `talkingdata` | ✓ | ✓ | ✓ | F 用全局 max(age) 再过滤 gender；B/I 先读取排序样本再写回最大值。F 的关系表达更完整、更可迁移。 | F |
| 4163 `movielens` | ✓ | ✓ | ✓ | 都锁定最高 avg_revenue 的 directors 并投影 distinct genre。I 用 inspect 明确值域且五步完成；F 先算 max，却未读取标量便直接写 literal。 | I |
| 4189 `app_store` | ✓ | ✓ | ✗ | B/F 用 inner join，只保留确有 translated review 的 free sports app；I 改用 left join，额外保留无 review 的 NULL 行。B 无恢复错误，逻辑最干净。 | B |
| 4598 `address` | ✓ | ✓ | ✓ | 三者都按 population_2010=0 连接 alias；F/I 是相同的四步最短路径。 | F=I |
| 4869 `works_cycles` | ✗ | ✗ | ✗ | 显式公式要求在 ShiftID 行粒度统计 Night/全部 ShiftID。B 混用全部历史分母与当前夜班分子；F 只看当前历史行；I 混用 Employee 分母与历史夜班分子。F 至少保持同一 population，但仍偏离公式。 | F（仅相对） |
| 4906 `image_and_language` | ✗ | ✗ | ✗ | 显式公式要求按 OBJ_SAMPLE_ID 数量求比；三者都改成 distinct IMG_ID，回答了“包含对象的图片数比”，不是指定测量单位。F 最简洁但同样语义错误。 | F（仅相对） |
| 5161 `cookbook` | 规 | 规 | 规 | 三者都先得到含 cheese 的 distinct recipe，再按 calories>200 求比例，符合自然语言“recipes”。verifier 不同可能来自显式 COUNT(recipe_id) 是否保留同一 recipe 的多个 cheese ingredient 行这一粒度歧义。F 最短，I 的值域检查更充分。 | F |
| 5440 `authors` | ✗ | ✗ | ✗ | Year 存在异常极大值。B 仍按 max Year，但用 left join 输出 NULL homepage；F/I 为得到非空答案，擅自加 JournalId!=0、2013 和最大 Id。B 最接近显式 Max(Year)，三者均未形成可靠答案。 | B（仅相对） |
| 5544 `legislator` | ✗ | ✗ | ✗ | 三者连接人物与全部 terms 后 count rows；同一 representative 的多个任期会重复，应该先固定 representative 实体粒度。I 最短，但没有语义修复。 | I（仅相对） |
| 5554 `address` | ✓ | ✓ | ✓ | 都正确过滤 NY + Post Office 并计数；I 三步，无不必要读取。 | I |
| 5873 `codebase_comments` | ✓ | ✓ | ✓ | 都正确从 solution 找 repo，再以目标顺序计算 percent more。I 直接使用 percent_change 且最短。 | I |
| 5952 `restaurant` | ✓ | ✓ | ✓ | 都从 Sankee 的可见行得到 city，再映射 county。F 四步完成，观察—过滤—终止链最紧凑。 | F |
| 6165 `authors` | ✓ | ✓ | ✓ | 都得到 2007 Neoplasia paper 与 PaperAuthor.Name。B 在错误的 Author join 上长时间循环；I 有四次重复读取；F 六步直接完成。 | F |
| 6454 `retails` | ✓ | ✓ | ✓ | 都固定 DELIVER IN PERSON 与 RAIL/MAIL，在同一 population 分组计数后排名。F/I 同为五步。 | F=I |
| 6489 `image_and_language` | ✗ | ✓ | ✓ | 外部知识明确要求 OBJ_SAMPLE_ID。B 多连接 OBJ_CLASSES 后回答 object class；F/I 直接投影 OBJ_SAMPLE_ID，I 三步最短。 | I |

## 去除格式因素后的答案质量

只把可以明确识别的纯输出形态失败改判为“核心答案正确”：

- B：example 3042；
- F：example 2512；
- I：example 2408、2512、3042。

得到：

| 口径 | B | F | I |
|---|---:|---:|---:|
| 原始 `bird-set` | 18/32 | **19/32** | 17/32 |
| 忽略纯输出形态 | 19/32 | **20/32** | **20/32** |

因此，inspect 版从 17 补到 20 的三题并不是更差的核心答案，而是终止表形态更容易漂移。
但这也不构成 inspect 的语义优势：F 同样达到 20，而且原始精确答案更高。

另有两题不能简单归为格式：

- example 1152：三者都严格执行了模型可见的 playerID + max birthDate 定义；
- example 5161：三者都按 distinct recipe 解释自然语言，但外部知识中的
  `COUNT(recipe_id)` 是否保留 ingredient 重复行存在粒度歧义。

若把这两题记为“可见规范下语义成立”，则 B/F/I 分别为 **21/22/22**。无论采用严格
格式中立口径还是较宽的可见规范口径，F 与 I 的答案语义覆盖都是持平的。

## 逻辑质量对比

### F 相对 B 的提升

- example 1692：统一 numerator/denominator 的 joined population；
- example 3042：保留历史聚合并输出正确姓名槽位；
- example 6489：按显式字段映射回答 OBJ_SAMPLE_ID；
- 大多数共同正确题省去了 schema 获取和重复读取。

主要回归是 example 2682：完整 schema 已经显示 ProductCostHistory，模型却选择更短的
Product 单行路径。这说明完整上下文减少探索成本，但会增强“见到一个表面匹配列就提前
收敛”的风险。

### I 相对 F 的提升

- example 1050：观察价格存储后严格采用显式净利润公式；
- example 2088：inspect 承担真正的值域工具，避免 `read_subtable` 接口误用并修正计数
  单位；
- 过程错误 11→5，合法终止 30→31。

主要语义回归：

- example 1692：观察 unmatched 后错误地用 left join 改变百分比分母；
- example 4189：用 left join 保留无 translated review 的 app；
- example 2408/2512/3042：核心实体正确但最终列形态更易漂移。

也就是说，inspect 提升的是**局部值域探索和接口稳定性**，不是全局关系推理。模型有时
会对观察反馈过度解释，并据此改变本来正确的 population 或 join 语义。

## 推理效率

| 指标 | B | F | I |
|---|---:|---:|---:|
| 动作 | 290 | **196** | 216 |
| 过程错误 | 7 | 11 | **5** |
| reasoning tokens | 97,260 | **50,821** | 63,957 |
| 模型 `<think>` 词数 | 56,147 | **32,438** | 38,300 |
| 总 tokens | 3,105,935 | **2,094,317** | 2,251,910 |

F 的 reasoning tokens 比 B 少 47.7%，比 I 少 20.5%。逐题轨迹中，这主要来自不再重复
describe/inspect/read，而不是把必要关系步骤压缩掉。I 在 9 题调用 11 次 inspect，
其中部分有价值，但也重新引入了观察和解释分支。

## 最终判断

### 综合最优：F，完整上下文且不保留 inspect_column

理由不是它原始分数只高 1–2 题，而是：

1. 忽略纯格式后，F 与 I 同为 20/32，语义覆盖没有损失；
2. F 的精确终止答案为 19/32，高于 I 的 17/32；
3. F 在 population/grain 上保住了 example 1692 和 4189，而 I 的两次回归都是实质关系
   语义错误；
4. F 比 I 少 20 个动作、13,136 reasoning tokens 和 157,593 总 tokens；
5. B 虽然在少数需要探索历史表或异常值的题上更稳，但总体推理更长，仍没有解决共同的
   显式公式/粒度错误。

### inspect 版的正确定位

I 不是整体最优默认协议，但它证明了一个局部能力：当问题确实需要完整值域而两个示例值
不足时，inspect 能显著降低 observation-only 接口摩擦。它适合成为针对“值域不确定题”
的专门实验条件，而不是在当前通用 32 题上常驻。

如果要求**一个统一、不依赖 router 的工具方案**，本轮证据仍建议选择 F 的模型可见
工具面；继续优化应针对共同失败的语义纪律：

- 明示公式优先于自然推断；
- 固定实体/行粒度后再计数；
- 不因 NULL、unmatched 或异常值自行加入 left/latest/valid-only 限制；
- 终止前把姓名、ID、字段数逐槽核对。

这些约束比重新加入一个通用 inspect 工具更可能提高整体上限。

## 审计边界

- 仓库：`master`，HEAD `5fd5ad58c52d72205da49786454820698cca6125`；
  fresh fetch 后相对 `origin/master` ahead 46、behind 0。
- 工作区已有未提交修改，本报告未改写或清理它们。
- 活跃诊断协议：atomic version39、`think-json-v1`、recent-4、
  `bird-set`。
- 三套产物均为 diagnostic-only，不具备训练准入资格。
