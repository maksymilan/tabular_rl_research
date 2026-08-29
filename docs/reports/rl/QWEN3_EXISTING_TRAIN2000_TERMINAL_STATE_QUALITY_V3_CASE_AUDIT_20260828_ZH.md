# 现有 BIRD-train 轨迹终态质量分数 v3：12 条实例人工审计

日期：2026-08-28

## 审计目的

本审计不训练模型，也不使用 BIRD-dev 1534。它只检查现有 BIRD-train 2000
冻结轨迹中的真实实例，回答两个问题：

1. 正确/错误轨迹按当前终态质量分数分成高、中、低 Gold 重合度后，分数是否符合实际轨迹质量；
2. 错误轨迹是否满足“分数越高，实际错误越小”。

从六个格子各取两条轨迹，共 12 条：正确高/中/低，以及错误高/中/低。这里的
“高、中、低重合”首先是按 v3 分数选样，人工审计再判断它是否真的代表语义重合。
分数来自：

\[
Q_{rank}(\tau)=C(G,S_{terminal}),
\]

其中 `C` 是 source、join、predicate、grain、value、set、rank、output 等实际出现类别的
宏平均。它不读取模型思维作为事实，也不使用 backward slice。

人工判断分成三项：

- **任务语义完成度**：轨迹是否找对数据、约束、运算和目标答案；
- **终态答案契约**：输出列、顺序、值是否满足 Gold/benchmark；
- **工具过程质量**：是否有实质性工具错误、无效绕行或由模型在思维中完成了本应由工具显式完成的运算。

人工完成度区间只是审计标尺，不是新的 reward 定义，避免用伪精确数字替换一个尚未验证的评分器。

## 总览结论

| 轨迹 | 结果 | v3 分数 | 人工完成度 | 原始推理错误程度 | 对当前分数的判断 |
|---|---:|---:|---:|---|---|
| `bird_train_03566` | 正确 | 1.000 | 0.95–1.00 | 无 | 合理 |
| `bird_train_03186` | 正确 | 1.000 | 0.95–1.00 | 无 | 合理 |
| `bird_train_04911` | 正确 | 0.600 | 0.90–0.98 | 很轻 | 明显低估 |
| `bird_train_01214` | 正确 | 0.583 | 0.90–0.98 | 很轻 | 明显低估 |
| `bird_train_02317` | 正确 | 0.222 | 0.85–0.95 | 轻 | 严重低估 |
| `bird_train_03073` | 正确 | 0.117 | 0.75–0.88 | 轻到中等 | 严重低估语义；能反映严格列顺序问题 |
| `bird_train_05388` | 错误 | 0.833 | 0.75–0.88 | 轻到中等 | 语义接近度基本合理；作为正确性奖励则过高 |
| `bird_train_05745` | 错误 | 0.800 | 0.65–0.80 | 轻到中等 | 略高；漏计过程绕行，且终态多一列 |
| `bird_train_03061` | 错误 | 0.500 | 0.35–0.50 | 中等到严重 | 大体合理，略偏高 |
| `bird_train_02007` | 错误 | 0.533 | 0.78–0.90 | 轻 | 低估语义推理；终态额外列确实错误 |
| `bird_train_03371` | 错误 | 0.050 | 0.40–0.58 | 中等 | 严重低估过程，只正确惩罚了最终运算错误 |
| `bird_train_02902` | 错误 | 0.125 | 0.78–0.90 | 轻 | 严重低估 |

这 12 条中，只有高分正确轨迹以及一条确有关键谓词遗漏的中分错误轨迹，与人工判断较稳定地一致。
当前分数能够识别一部分“接近 Gold 的错误”，但不能稳定表示“整条轨迹质量”。

## 一、正确且与 Gold 高度重合

### 1. `bird_train_03566`：1.000，合理

问题是从准备时间大于 10 分钟的菜谱中找热量最高者的标题。Gold 是：

`Recipe` 与 `Nutrition` 按 `recipe_id` 连接，过滤 `prep_min > 10`，按 `calories DESC`
排序取一条，输出 `title`。

实际工具轨迹逐项完成了相同操作：描述两表、正确 inner join、正确过滤、按热量降序
`top_k=1`、只返回标题。最终答案与 Gold 都是
`Ginger-Orange Barbecued Beef Back Ribs`。

推理没有实质错误，也没有依靠模型思维代替数据库运算。1.000 合理。

### 2. `bird_train_03186`：1.000，合理

问题是 Harvard 在 2012 年的国际学生人数。Gold 计算：

`num_students * pct_international_students / 100`。

轨迹正确连接 `university` 与 `university_year`，过滤 Harvard 和 2012，读到
`20152` 与 `25`，再用两个 `scalar_compute` 依次做乘法和除以 100，得到 `5038`。

两步标量运算只是 Gold 单表达式的等价拆分。推理正确，1.000 合理。

## 二、正确但与 Gold 有一定差异

### 3. `bird_train_04911`：0.600，明显低估

问题是 `airplane` 类别占全部对象的百分比。Gold 使用：

- 条件 `SUM(CASE WHEN OBJ_CLASS='airplane' THEN 1 ELSE 0 END)`；
- 除以 `COUNT(OBJ_CLASS)`。

轨迹使用：

- 连接后对全部行做 `COUNT(*)`；
- 对 `OBJ_CLASS='airplane'` 做条件 `COUNT(*)`；
- 二者相除并乘 100。

更重要的是，轨迹先用 `inspect_column` 确认 `OBJ_CLASS has_null=false`。因此在这个确定性
数据库状态中，`COUNT(*) = COUNT(OBJ_CLASS)`；条件 `COUNT(*)` 也等价于条件 `SUM CASE`。
最终数值完全一致。

当前分数因为 value/output 的表达式树不相同而把两类都记为 0，只保留 source、join、grain，
得到 0.600。模型思维开头曾错误提到并不存在的 external knowledge，但随后明确放弃，执行路径正确。
实际错误很轻，分数明显低估。

### 4. `bird_train_01214`：0.583，明显低估

问题同时明确给出了 Act 4、Scene 5 和 description。Gold 仅按 description 过滤，并用
`RIGHT JOIN` 从 `works` 取 `LongTitle`。轨迹按 Act、Scene、description 三个条件过滤
`chapters`，再 inner join `works`，得到同一标题。

在 Gold 的 `WHERE chapters.Description=...` 约束下，RIGHT JOIN 对目标匹配行与 inner join
等价；轨迹增加的 Act/Scene 约束又直接来自问题，不是无关条件。当前评分把 join 类型记为完全不匹配，
并因三个谓词只命中一个而把 predicate 记为 1/3。

这条轨迹实际上比 Gold 更显式地落实了自然语言约束，推理几乎无错。0.583 明显偏低。

## 三、正确但与 Gold 重合度很低

### 5. `bird_train_02317`：0.222，严重低估

问题是 American Airlines Inc. 计划降落纽约的航班数。Gold 连接 `Airlines`、`Air Carriers`
和 `Airports`，按 carrier description 和 `DEST='JFK'` 约束，再用条件 `SUM CASE`。

轨迹先查询 `Air Carriers`，确定 American Airlines Inc. 的代码为 `19805`；随后直接在
`Airlines` 上过滤 `_AIRLINE_ID=19805 AND DEST='JFK'`，最后 `COUNT(*)=1667`。

这里有三种合理差异：

- carrier description 被工具查得的确定性代码替代；
- `Airports` 连接对按 `DEST='JFK'` 计数是冗余的；
- filter + count 与 Gold 的 conditional sum 在该状态上等价。

轨迹用了 `LIKE` 查 carrier 名称而非严格等号，但读回结果只有一行，风险很小。主要问题是模型把
已观察到的 `19805` 写成后续 literal，没有通过 `value_ref`/`in_table` 显式保留绑定；当前终态状态
因而无法证明这个常量从哪里来。它是**高语义质量但显式机器可验证 grounding 不完整**，不能简单当作
0.222 的差轨迹。

### 6. `bird_train_03073`：0.117，语义严重低估

问题是 1997 年 Tampa Bay Lightning 的 interim coach 的 losses 和 coach ID。Gold 连接
`Coaches` 与 `Teams`，按队名、年份、`notes='interim'` 过滤，按 coach 分组并 `SUM(l)`。

轨迹先从 `Teams` 查出 1997 年 Tampa Bay Lightning 的 `tmID='TBL'`，再直接过滤
`Coaches(year=1997, tmID='TBL', notes='interim')`。结果只有一行，所以该行的 `l=6`
与 `SUM(l)=6`，也不需要 group。

低分主要来自 Gold 中的 join/group 被工具查值后的直接过滤替代。真正存在的争议是：轨迹终态列顺序为
`coachID,l`，Gold 为 `SUM(l),coachID`；当前 `bird-set` 将其记录为正确，但严格有序答案契约下应算形状错误。
因此它不是完美轨迹，不过 0.117 远低于其语义完成度。

## 四、错误但与 Gold 高度重合

### 7. `bird_train_05388`：0.833，高分对应较小偏差，但暴露 Gold 执行细节

问题是 Weimei Corp 在 2018–2020 三年的平均年度订单数。Gold 做 `COUNT(OrderNumber) / 3`。
轨迹正确连接、过滤客户和三个年份，得到 161 条，再用 `scalar_compute` 算出
`161/3=53.6666667`。Gold 在 SQLite 中是两个整数相除，结果截断为 `53`，因此 benchmark 判错。

从自然语言数学含义看，轨迹的 53.67 比 Gold 的 53 更合理；从复现 Gold SQL 执行语义看，轨迹漏掉了
整数除法类型。轨迹还曾发出一次非法 `value_ref`，并在第一次尝试中错误地把三个年份条件写成 AND，
随后改用 join 和 OR 完整纠正。

0.833 能表示“语义非常接近”，但不能直接作为接近正确答案的无条件奖励：它没有识别整数/实数除法语义，
也完全忽略了中途工具错误。这条样本同时说明 Gold SQL 不是总能代表更合理的自然语言推理。

### 8. `bird_train_05745`：0.800，略偏高

问题要求输出资本所在省为 Distrito Federal 的国家之 population density 和 Industry GDP share。
Gold 输出两列：`Population/Area, Industry`。轨迹找到了正确四国、正确连接 `country.Code=economy.Country`、
正确计算所有数值，但额外输出了国家名，形成三列结果。

原始推理对“the nation”单数反复犹豫，先做了一次错误的 `country.Name=economy.Country` 连接，得到 0 行；
读取 economy 后发现键实际是国家代码，再成功纠正。最终主体语义正确，错误集中在多输出一列，属于轻到中等，
并非完全错误轨迹。

当前 output 类因为位置变化得到 0，但其余四类全为 1，宏平均仍为 0.8。若这是“语义接近度”，0.8 尚可；
若称为“整体轨迹质量”，则没有反映错误 join、冗长绕行和终态 shape failure，略偏高。

## 五、错误且与 Gold 有一定差异

### 9. `bird_train_03061`：0.500，大体合理

问题是 1950–1980 年平均每年入选 Hall of Fame 的 player 数。Gold 除年份外还要求
`category='Player'`，然后 `COUNT(name)/30=4.7667`。

轨迹只按年份过滤，直接对全部 202 条 HOF 记录计数并除以 30，得到 6.7333。模型多次说目标是 player，
却没有检查或过滤 `category`。这不是表达式等价问题，而是遗漏决定性群体约束。

`COUNT(hofID)` 与 `COUNT(name)` 在非空/唯一状态下可能等价，当前 value/output 因列名不同被额外记 0；
但 `category='Player'` 的遗漏本身已造成显著答案错误。0.5 大体能表示中等质量，甚至略偏高；原始推理错误为
中等到严重。

### 10. `bird_train_02007`：0.533，低估语义推理

问题是 2014 Winter 最年轻的参赛者。Gold 通过 `games` 连接 `games_competitor`，再连接 `person`，
输出最小 age 对应的 `full_name`。

轨迹先用工具查出 `games_name='2014 Winter'` 的 id 为 16，然后用 literal `games_id=16`，只连接
`games_competitor` 与 `person`，按 age 升序取一条。它找到正确人 `Polina Edmunds`，但额外输出 age=15，
因此被判错。

缺少 `games` source/join 不是推理错误，而是工具观察值替代；真正错误只是终态多一列。和
`bird_train_05745` 相比，这条被打到 0.533，说明当前宏平均对“查 ID 后常量替代”施加了额外惩罚。
作为严格答案契约它确实不正确，作为语义轨迹则明显高于 0.533。

## 六、错误且与 Gold 重合度很低

### 11. `bird_train_03371`：0.050，过低但确有决定性运算错误

问题要求 Fuenlabrada 所在国家的 service GDP。Gold 计算 `Service * GDP`。轨迹正确查到城市属于
国家代码 `E`，也正确查到 economy 行 `GDP=565000, Service=33.6`，但把 `Service` 当作最终答案，
返回 33.6，而非乘积 18,984,000。

模型其实在思维中识别出 Service 可能是比例，也明确意识到问题可能要求金额，最后仍选择只投影 Service。
这是一处局部但决定性的语义运算错误，严重度为中等；前面的实体解析和工具查值是正确的。

0.05 能强烈惩罚最终错误，但几乎抹掉了已完成的正确子问题，因此不适合作为整体质量或错误大小的线性量尺。

### 12. `bird_train_02902`：0.125，严重低估

问题是 rejected purchase orders 最多的非销售员工姓名。轨迹的有效过程是：

1. `Status=3` 过滤到 86 条；
2. 正确连接 Employee 与 Person，并过滤 `PersonType='EM'`；
3. 按员工 group/count，读到 `(250,28), (252,1), (261,57)`；
4. 选出员工 261，再查 Person，得到 `Reinout N Hillmann`。

Gold 输出 `FirstName, LastName`，轨迹输出 `FirstName, MiddleName, LastName`，因此终态多了中间名，
benchmark 判错。轨迹开始还有两次 plan schema 错误，但之后的关系推理和答案主体正确。

因为最后一步重新以 literal `BusinessEntityID=261` 查询 Person，终态 lineage 只看见 Person 与 261，
看不见此前已经正确完成的 Status 过滤、join、group/count 和最大值选择，于是只有 source/output 获得少量分数。
这说明“不做 backward slice”本身没有问题，但**仅用终态 lineage 就不能把该分数称为整轨迹质量**。

同时也要注意：最大值 261 是模型从三行观察中手工读出的，并没有调用 `extreme_value_select`。如果研究目标要求
所有关键运算都由 Harness 显式执行，那么这条轨迹应受到一定 grounding 惩罚；但从 0.8 左右直接跌到 0.125
仍不合理。

## 跨实例判断

### 1. “错误轨迹中高分者错误较小”得到局部支持

两条高分错误都不是完全跑偏：

- 0.833 是整数除法与实数除法的细节差异，且 Gold 的自然语言合理性本身可疑；
- 0.800 是主体结果完全正确但多输出一列。

它们确实比遗漏 `category='Player'` 或漏乘 `GDP` 更接近正确语义。因此，利用语义状态给错误轨迹排序的
研究方向有价值。

### 2. 但低分不等于错误更大，当前排序不满足全局单调性

0.125 的 `bird_train_02902` 实际只多输出中间名，关系推理几乎完整；0.533 的
`bird_train_02007` 也只多输出 age。它们的实质错误都小于 0.500 的 `bird_train_03061`，后者遗漏了
决定性 player 条件。

所以当前分数还不能稳定支持“分数越低，错误越大”。

### 3. 当前指标实际测量的是“终态对一个 Gold SQL 表达式的结构重合”，不是统一轨迹质量

主要误差源有四类：

1. **等价关系代数改写未归一化**：filter+count 与 conditional sum、inner/right join 的等价情形；
2. **工具观察值替代未归一化**：先查出 carrier/game/employee ID，后续以 literal 使用；
3. **Gold 冗余结构被当作义务**：对当前结果无影响的 join/group 仍造成失分；
4. **宏平均稀释关键错误**：output 或关键 predicate 完全错时，只损失一个类别，仍可能得到 0.8 以上。

此外，终态分数按设计不会反映非法调用、失败 join、冗长试探。若目标只是终态语义排序，这可以接受；若目标是
“轨迹整体质量”，名称和用途必须改变。

## 对 RL 验证的明确结论

**当前 v3 分数不能直接进入 RL。** 12 条人工审计支持“语义接近度可以排序部分错误轨迹”这一核心假设，
但否定了“当前终态宏平均就是可靠统一轨迹质量”的假设。

下一步不应恢复 backward-slice reward，也不应开始训练。应先做两个最小修正，再用同一批轨迹重算：

1. **状态等价归一化**：只使用 Harness 可验证事实，处理已观察唯一常量替代、已验证非空下的 count 等价、
   filter+aggregate 等价和明显冗余 join；不读取模型思维；
2. **关键答案契约单独保留**：把终态 value/output 的正确程度与关系语义接近度分开报告，避免一个关键答案错误
   被类别宏平均稀释，也避免额外一列把全部 output overlap 直接打成 0。

建议下一版先输出两个量，而不是再次强行混成一个未经验证的标量：

\[
Q_{relation}=\operatorname{macro}(source,join,predicate,grain,value,rank,set),
\]

\[
Q_{answer}=\operatorname{partial\_match}(value,columns,column\ order,row\ shape).
\]

这两个量仍只用于**整轨迹排序**，不做逐步骤 reward，不按工具类型校准，也不恢复依赖 credit。等同题多轨迹验证
证明二者的固定组合能稳定符合人工错误大小排序后，才定义唯一的 RL 标量。

## 审计产物

- 12 条完整轨迹与分数：
  `data/results/existing_sft2k_terminal_state_quality_v3_20260828/casebook_12.json`
- 样本提取脚本：`src/rl/diagnostics/extract_tool_state_quality_cases.py`
- 全量统计报告：
  `docs/reports/rl/QWEN3_EXISTING_TRAIN2000_TERMINAL_STATE_QUALITY_V3_20260828_ZH.md`

