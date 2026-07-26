# BIRD action-block 推理质量对比审计（2026-07-26）

## 结论

action-block 带来了明显的**执行规划压缩和成本改善**，但没有带来可验证的
**grounded relational reasoning 能力提升**。

在同一冻结 200 题、DeepSeek v4 Flash、greedy、`bird-set` 条件下：

- version24 单工具 baseline 为 **145/200**；
- action-block v10 为 **143/200**；
- 配对变化为 10 条恢复、12 条退化，双侧精确检验 `p=0.8318`。

因此，当前证据支持“同等能力下更少模型轮次”，不支持“模型能力上限被抬高”。
更准确地说，action-block 把计算从多次反馈后的闭环重规划，转移到更长的单次
open-loop 规划。它减少了部分跨工具的上下文重复，但也减少了模型利用新观察修正
population、grain、operator 和答案槽位的机会。

当前结果很像 **DeepSeek v4 Flash + greedy + prompt-only 工具策略的局部平台期**，
而不是工具系统或模型的理论上限。57 条 v10 失败中仍有 34 条明确的语义、
population/grain 或输出形状错误，而且现有工具均能表达正确路径。

## 比较边界

推理文字不是事实权威，也不保证忠实反映模型的内部计算。本报告把“推理质量”拆成
三个可观测层次：

1. **计划与执行效率**：模型轮次、原子动作、reasoning token 和 API 请求；
2. **grounded 反馈利用**：观察是否真的发生在后续参数选择之前，错误后是否能修正；
3. **任务能力**：最终 denotation，以及配对恢复/退化的具体原因。

v10 除 action-block 外还启用了确定性的低摩擦引用解析，因此 fixed-200 配对估计的是
整个可见接口和环境行为的组合效果，不是纯粹的 batch size 因果效应。

## 总体量化

| 指标 | version24 baseline | action-block v10 | 变化 |
|---|---:|---:|---:|
| 正确数（`bird-set`） | **145/200** | 143/200 | -2 |
| 合法终止 | 197/200 | **198/200** | +1 |
| 过程错误 | **29** | 30 | +1 |
| 模型轮次 | 1,490 | **918** | -38.4% |
| 原子动作 | **1,490** | 1,808 | +21.3% |
| API 请求 | 1,509 | **944** | -37.4% |
| reasoning token | 358,117 | **305,721** | -14.6% |
| 总 token | 8,420,861 | **4,567,874** | -45.8% |

action-block 每题的 reasoning token 从 1,790.6 降为 1,528.6，但每个模型轮次从
240.3 增至 333.0（+38.6%）。也就是说，每次 plan 更长、更集中，但全题能够发生的
“观察—再思考”轮次更少。

## 按配对结果分组

| 配对组 | 题数 | baseline：轮次 / 动作 / reasoning token | v10：轮次 / 动作 / reasoning token | 解释 |
|---|---:|---:|---:|---|
| 两者都正确 | 133 | 6.71 / 6.71 / 1,010 | 4.38 / 8.54 / 1,139 | v10 用更少反馈轮次完成更多动作；这是最可信的执行规划改善。 |
| 两者都错误 | 45 | 9.38 / 9.38 / 3,651 | 5.16 / 10.33 / 2,543 | v10 更快执行更多动作，但没有突破语义错误；额外观察没有转化为正确决策。 |
| 仅 v10 正确 | 10 | 8.40 / 8.40 / 3,511 | 4.40 / 8.20 / 2,026 | 部分任务避免了 baseline 的长链漂移、错误或预算问题。 |
| 仅 baseline 正确 | 12 | 7.58 / 7.58 / 2,030 | 5.00 / 10.42 / 1,630 | v10 以更少的反馈整合执行更多动作，是典型的过早承诺信号。 |

过程错误均值也没有显示普遍改善：

- 两者都正确：baseline 0.098，v10 0.128；
- 两者都错误：两者均为 0.200；
- v10 恢复题：baseline 0.500，v10 0.100；
- v10 退化题：baseline 0.167，v10 0.250。

因此 v10 确实能在一部分轨迹里消除长链接口摩擦，但这个收益没有泛化成总体更好的
过程正确性。

## 探索是否更充分

v10 明显增加了观察型调用：

| 工具 | baseline | v10 | 变化 |
|---|---:|---:|---:|
| `describe_table` 调用 | 223 | 405 | +81.6% |
| 被 describe 的表引用数 | 534 | 523 | -2.1% |
| `inspect_column` | 78 | 160 | +105.1% |
| `read_subtable` | 179 | 277 | +54.7% |

重复率很低，新增的 inspect/read 大多是在观察新的列或结果，而不是机械重复。
但是 describe 调用翻倍并没有扩大 schema 覆盖：模型只是把相近数量的表拆成了更多
调用。

这说明“观察数量上升”不等于“grounded reasoning 上升”。主要失败已经从缺少信息
转为对信息的语义解释错误。

### Hybrid block 的信息屏障问题

711 个 action block 中：

- 332 个只含观察工具；
- 200 个只含关系执行工具；
- 179 个同时含观察和执行；
- 157 个包含“先执行、后 read/观察”，这种验证结果供下一轮使用是合理的；
- 37 个 block 包含“先观察、后执行”，涉及 35 道题；
- 其中 15 个同时存在观察前置和执行后验证。

同一 block 中所有参数都在工具运行前由模型一次性写出。因此，当 `describe_table`、
`inspect_column` 或 `read_subtable` 位于某个关系动作之前时，后续动作不可能真正利用
这次尚未返回的内容。环境 DAG 只能解决句柄和 step-id 的**结构依赖**，不能解决
“看到列名/值后应该选择哪个 operator、literal 或 answer slot”的**认知依赖**。

这正是当前 action-block 范式最重要的边界：它支持执行依赖，却不能自动把未见观察
变成已 grounded 的决策。

## 22 条配对变化的人工审计

### v10 的 10 条恢复

- 5 条主要是输出形状/答案槽位恢复：
  `02512`、`03262`、`03724`、`03969`、`05316`；
- 5 条是关系或语义路径恢复：
  `00074`、`01692`、`03636`、`03977`、`05440`。

代表性改善：

- `05440`：v10 在两轮内完成 paper–journal 连接、最大年份排序和终止；baseline
  经过 12 轮后把最大年份 population 漂移成四行且 homepage 为 NULL。
- `01692`：v10 保持“nominees”作为固定分母 population，得到 66.10%；baseline
  在多轮分支中把分母换成了错误粒度，得到 10.57%。
- `03636`：v10 将已知的 actor、film_actor、film、language 路径压缩执行并在预算内
  完成；baseline 出现 3 个错误后未合法终止。

这证明 batch 对“schema 和目标已经足够明确”的链条有真实价值。

### v10 的 12 条退化

原报告把它们分为 6 条输出/槽位退化和 6 条 population/关系退化。结合逐条 gold
冲突审计后，其中：

- `00362`、`03373`、`05053` 是明显的 benchmark/输出约定歧义；
- `02179` 是 v10 的不安全字面量解析，已由 v11 类型化拒绝修复；
- 剩余 8 条是较清晰的策略退化。

去掉上述 4 条后，v10 的干净配对变化是 10 条恢复对 8 条退化，但核心关系/语义变化
仍为 **5 对 5**；小幅净收益来自答案形状，而不是关系推理。

代表性退化：

- `02513`：baseline 读取到 211 条 inspections 后重新思考“题目问 businesses”，
  改为 `count_distinct(license_no)` 得到 203；v10 在一个短链中直接数 inspection
  rows，得到 211。反馈边界帮助了 baseline 修正 grain。
- `06489`：外部知识明确把 object 映射到 `OBJ_SAMPLE_ID`。baseline 在读取匹配行后
  重新检查输出映射并返回 18；v10 在第一轮就假设“object 是 class name”，额外连接
  `OBJ_CLASSES` 后返回 `"paper"`。更多 schema 调用强化了错误假设。
- `04906`：v10 在第一轮 reasoning 中把 images 定义为 distinct `IMG_ID`，忽略了
  外部知识要求的 object-sample occurrences；后续多次 read 和 count 都只是在执行
  错误定义。
- `04244`：v10 使用 8 轮、19 个动作探索多个表，却按 `tmID` 跨年份聚合后再与 MVP
  球队求交，没有先建立统一的 player/team/award population。探索更多并未修复关系
  population。
- `02507`：v10 在首个 block 内一次写出 describe、aggregate、top、join、姓名拼接、
  read 以及一个非法嵌套终止；错误恢复后仍返回单字段 `"David Hodges"`。这是典型的
  batch 过度承诺和答案槽位退化。

## 是否已经到能力上限

### 可以支持的判断

当前已经接近一个**局部策略平台期**：

- version20 的校正 `bird-set` 重放为 143/200；
- version24 为 145/200；
- action-block v10 为 143/200；
- 多次 prompt、状态反馈和接口变体主要造成 paired swap，没有稳定净提升；
- v12 通用 constraint-first prompt 在冻结 24 题上从 6/24 降到 5/24，错误和 token
  反而增加。

所以，继续在 runtime prompt 增加“充分思考、检查 grain、检查答案槽位”等文字，
很可能只是改变错误分布。

### 不能支持的判断

这不是模型或工具表达能力的理论上限：

- v10 的 57 条失败中，20 条存在 benchmark/问题/外部知识歧义或冲突；
- 34 条是清晰的语义、population/grain 或输出形状错误；
- 除已经修复的一个接口安全问题外，没有发现必需的新关系算子；
- 44 条错误答案来自零执行错误的合法轨迹，说明瓶颈是策略学习，不是执行能力；
- greedy 单轨迹还混有 provider 轨迹方差，不能用一次 fixed-200 证明模型容量上限。

更准确的说法是：**现有模型在没有针对该协议训练、只靠 prompt 和接口重排时，已经
接近可达到的局部上限。**

## 下一步设计

保留 action-block 的效率价值，但把 block 边界定义为**信息屏障**：

1. 同一 block 并行执行所有参数已经 grounded 的独立观察；
2. 如果后续参数需要解释新的 schema、column value、row sample 或 error，则环境在
   该观察后自动 yield，不执行后继语义动作；
3. 允许已经 grounded 的关系链在一个 block 内连续执行，并允许末尾 read 验证；
4. 自动引用修正如果改变了关系列身份，应返回 canonical binding 并 defer 后继，而
   不是静默执行整条分支；
5. root error 与 blocked descendants 继续分开反馈。

教师数据应监督“当前 grounded 信息所允许的最大安全 batch frontier”，而不是监督
固定的两阶段模板，也不要求模型显式维护 DAG。环境可以构建结构 DAG，但训练目标还要
标注哪些决策跨越了认知信息屏障。

下一轮不应直接再跑一个 prompt fixed-200。先在以下冻结集合做小门控：

- 34 条无歧义策略失败；
- 22 条配对变化题；
- 133 条两者都正确的控制题中抽取稳定子集。

除 `bird-set` 外，同时报告：

- semantic/population/output 三类配对恢复与退化；
- unseen-observation-before-decision 数量；
- 新观察或错误后的有效修正率；
- 每题模型轮次、原子动作、reasoning token 和过程错误。

若要判断是 greedy 策略上限还是模型容量上限，应在无歧义困难题上补一个固定 K 的
多轨迹诊断。若 pass@K 明显高于 pass@1，瓶颈主要是策略方差和选择；只有多个轨迹都
重复同一语义错误时，才更像模型能力或训练分布限制。

## 输入产物

- `data/trajectories/tool_usability_20260724/version24_fixed200_first50_bird_set.all.jsonl`
- `data/trajectories/tool_usability_20260724/version24_fixed200_remaining150_bird_set.all.jsonl`
- `data/trajectories/batch_plan_20260726/action_block_v10_low_friction_fixed200_r1.all.jsonl`
- `data/trajectories/batch_plan_20260726/action_block_v10_vs_version24_fixed200.paired.json`
- `docs/reports/evaluation/BIRD_ACTION_BLOCK_FIXED200_FAILURE_AUDIT_20260726_ZH.md`
