# BIRD 多次 fixed-200 与 SFT fixed-1000 失败联合分析

日期：2026-07-29

## 结论

联合分析不支持“继续加 schema 文本或继续打磨调用格式”作为当前主线。当前主要瓶颈已经是**合法轨迹中的语义决策**：

- 最终 fixed-1000 为 **704/1000 `bird-set`**，296 条失败中 288 条是
  `wrong_answer`；仅 8 条没有合法终止。
- 288 条合法错误中 **254 条（88.2%）没有任何过程错误**。即使把 8 条非法终止
  全部修对，上限也只有 712/1000，提升 0.8pp。
- 五个同题 fixed-200 运行中，107 题五次全对，36 题五次全错，57 题随接口或策略
  翻转。五次结果的 oracle 并集是 164/200，但它不是可部署指标。
- 36 条稳定失败经完整上下文 F 的人工分类映射后，15 条属于公开证据下仍无法可靠
  确定 benchmark/output convention 的歧义题；剩余 21 条才是优先的能力或可靠性目标。

所以，下一轮最合理的研究单元不是“再做一个大 prompt 版本”，而是把 frozen-200
拆成三类训练与评测资产：

1. **57 条策略敏感题**：已有至少一条成功的无 gold 轨迹，是最适合做因果对比、
   divergence audit 和成功轨迹 replay 的数据。
2. **21 条非歧义稳定失败**：需要针对 population/grain、公开字段映射、公式和答案
   槽位构造新的因果教师尝试。
3. **15 条 benchmark 歧义题**：先隔离并审计，不进入通用 SFT 规则或过程奖励。

## 1. 分析口径

### 1.1 fixed-1000

SFT 教师生成的固定 1000 题由以下两部分组成：

- 原 fixed-200 core；
- 与 core 不重叠的 additional-800。

additional-800 使用两轮 provider-only retry 的最终结果覆盖原始 transport/provider
失败。合并后为 704/1000，而不是初始运行的 685/1000。所有记录都要求
`denotation_comparison == "bird-set"`。

### 1.2 多次 fixed-200

只联合完全相同 200 个 task id、且全部使用 `bird-set` 的五个现代运行：

| 运行 | 正确 | 合法终止 |
|---|---:|---:|
| atomic version24 | 145/200 | 197/200 |
| action-block v10 | 143/200 | 198/200 |
| action-block v21 | 136/200 | 194/200 |
| action-block v32 | 144/200 | 197/200 |
| atomic version39 full-context F | 136/200 | 192/200 |

历史 strict-multiset 结果没有混入本报告。

### 1.3 证据层级

本报告没有假装对 296 条失败逐条完成了人工语义标注，而是区分三个证据层级：

1. **全量确定统计**：正确、合法、过程错误、步数、工具出现、可见结果形状。
2. **重复实验稳定性**：同一题在五个协议/策略运行中的 0..5 次正确。
3. **人工因果审计**：F 的 64 条失败分类，以及 version37 的 54 条失败、27 条
   单候选反事实验证。

可见结果形状只能定位输出症状，不能直接证明语义根因。例如“行数不同”可能来自
错误 population，也可能来自 distinct、并列或输出约定。

## 2. fixed-1000 的大样本结果

### 2.1 错误几乎都不是调用失败

| 指标 | 数量 |
|---|---:|
| 正确 | 704 |
| 失败 | 296 |
| wrong_answer | 288 |
| argument_validation_error | 4 |
| execution_error | 3 |
| max_steps | 1 |
| 失败且无过程错误 | 255 |
| wrong_answer 且无过程错误 | 254 |

错误轨迹比正确轨迹更长：

| 轨迹 | 平均动作 | 中位数 | P90 |
|---|---:|---:|---:|
| 正确 | 6.97 | 6 | 10 |
| 失败 | 8.45 | 8 | 13 |

失败平均多 1.49 个动作（约 +21.3%）。这更像模型在复杂任务上探索更久但仍作出错误
关系/语义决策，而不是单一 carrier 或 parser 故障。

正确轨迹中也有 99 条经历过至少一个过程错误并最终恢复；失败轨迹中只有 41 条有
过程错误。过程错误应该继续保持局部负/零信用和恢复信用，但不能作为总体正确率的
主要代理目标。

### 2.2 难度是连续退化，不是突然失去协议能力

| 难度 | 正确率 | 合法率 |
|---|---:|---:|
| easy | 300/400 = 75.0% | 99.75% |
| medium | 208/300 = 69.33% | 99.67% |
| hard | 196/300 = 65.33% | 98.0% |

hard 题比 easy 题低 9.67pp，但仍能 98% 合法结束。这再次表明增加的主要难点是关系
组合、总体/粒度和运算语义，而不是模型不会输出工具调用。

### 2.3 external knowledge “存在”没有形成可见增益

| external knowledge | 正确率 |
|---|---:|
| 有 | 658/934 = 70.45% |
| 无 | 46/66 = 69.70% |

这不是因果消融，因为两组任务难度和组成不同；但它至少说明把自然语言 external
knowledge 放进 prompt 并没有自动解决映射问题。36 条五次稳定失败中也有 35 条带
external knowledge。结合人工轨迹中的 `OBJ_SAMPLE_ID → object class`、
`COUNT(recipe_id) → count_distinct`、`COUNT(OBJ_SAMPLE_ID) → distinct IMG_ID`
覆盖行为，问题更接近**信息没有成为可执行约束**，而不是单纯缺少信息。

### 2.4 wrong_answer 的可见结果症状

| 可见 pred/gold 症状 | 数量 | 占 288 |
|---|---:|---:|
| 同形状 numeric mismatch | 99 | 34.4% |
| 同列数但行数不同 | 41 | 14.2% |
| 多输出列 | 41 | 14.2% |
| 同形状 text mismatch | 38 | 13.2% |
| 少输出列 | 36 | 12.5% |
| 同形状 mixed mismatch | 24 | 8.3% |
| 可见 sample 相同但完整 denotation 不同 | 5 | 1.7% |
| 空预测 | 4 | 1.4% |

至少 77/288（26.7%）直接表现为答案列宽不匹配；把行数不同也计入，则有
118/288（41.0%）表现为可观察的输出槽位或总体基数差异。不过：

- 99 个 numeric mismatch 不能直接等价为“缺 numeric parser”；错误分母、错误
  distinct、错误 population 也会产生 numeric mismatch。
- 41 个行数差异不能直接等价为“漏了 top-k/ties”；它也可能是 join multiplicity
  或错误实体粒度。

因此输出症状适合做分层抽样，不适合直接生成全局 prompt 规则。

### 2.5 工具出现与失败率

下表是“使用过该工具的任务”的条件失败率，不是因果效应：

| 工具 | 使用任务 | 失败率 | 未使用时失败率 |
|---|---:|---:|---:|
| join_tables | 511 | 33.46% | 25.56% |
| project | 444 | 35.59% | 24.82% |
| read_subtable | 488 | 32.79% | 26.56% |
| extreme_value_select | 188 | 37.23% | 27.83% |
| scalar_compute | 116 | 37.93% | 28.51% |
| group_aggregate | 528 | 31.06% | 27.97% |

这些工具主要出现在更复杂的题上，不能据此判定工具导致失败。它们共同指向的事实是：
**需要关系组合、投影、极值或算术的题更难**。因此单边 join 的目标应是减少调用错误
和认知负担，而不应预先声称能恢复全部 join 相关语义错误。

## 3. 五次 fixed-200 的稳定性

| 五次中正确次数 | 题数 | 解释 |
|---:|---:|---|
| 0 | 36 | 稳定失败 |
| 1 | 12 | 高度脆弱 |
| 2 | 6 | 策略敏感 |
| 3 | 11 | 策略敏感 |
| 4 | 28 | 轻度脆弱 |
| 5 | 107 | 稳定正确控制 |

五次 oracle 并集为 164/200，说明不同策略已经覆盖了 82% 的题，但 verifier oracle
选择不能作为部署成绩。它真正提供的价值是识别训练数据：

- 57 条中间态任务至少有一次成功和一次失败，可以在不从 gold SQL 编译路径的前提下，
  比较最早的有事实依据的决策分歧。
- 107 条稳定正确题可作为 no-regression control。
- 36 条稳定失败不应继续依靠“多采样后只收成功”解决，因为五种较大策略变化都没有
  找到成功轨迹。

以当前最好 atomic version24 的 145/200 为起点，它的 55 条失败中：

- 36 条是五次稳定失败；
- 19 条已被至少一个其他运行恢复。

要到 160/200，必须在这 19 条已有成功证据的题中净恢复至少 15 条，同时几乎不能让
原正确题回归。这解释了为什么继续做宽泛 prompt 改动很难达到 80%。

## 4. 36 条稳定失败的人工根因

使用 full-context F 的人工分类映射：

| 主类 | 数量 | 处理建议 |
|---|---:|---|
| benchmark/output convention 歧义 | 15 | 隔离、人工审计，不训练通用规则 |
| population/grain/join/multiplicity | 9 | 最高优先级因果训练 |
| 字段映射/算子/公式/literal | 6 | 公开任务约束 + 因果训练 |
| 答案槽位/表示 | 3 | 精确终止投影训练 |
| F 中表现为非合法终止 | 3 | 重跑并做语义审计，不能把 F 的表象当稳定根因 |

15/36（41.7%）稳定失败是公开证据下仍与 benchmark 约定存在冲突或歧义的题。这批题
如果直接进入 SFT 或 RL：

- 可能奖励模型学习隐藏 benchmark 习惯，而不是可泛化数据库推理；
- 可能把“总要 distinct / 总要 top-1 / 总要换成 label”等偶然规则传播到控制题；
- 会污染过程信用，因为合法、公开证据一致的轨迹仍得到终端 0。

因此应同时报告官方全集成绩和审计后的 clean-subset 诊断，但不能用 clean subset
替换官方 BIRD 分数。

## 5. 54 条人工因果审计对大样本结论的验证

version37 的 54 条失败中，45 条没有过程错误。预注册 27 条“一题一个候选修复”的
反事实验证结果为：

- 16 条单一局部修复通过；
- 11 条直觉修复被证伪，没有继续利用 verifier 爬山。

已通过的修复集中在五类：

1. 遵守显式 external 字段或 COUNT 映射；
2. 聚合前固定实体粒度；
3. 不增加题面没有的时间、唯一性或 population 限制；
4. 终止前只保留请求的答案列；
5. 先建立正确关系总体，再做极值。

这 16 条证明：相当一部分失败不需要新工具，正确路径已在当前 atomic 的表达空间内，
缺的是对局部语义决策的学习。11 条反事实失败则证明不能把每个错误都用简单 heuristics
修复；它们应转入人工语义或标注审计。

## 6. 对上一轮改进建议的重新排序

### P0：公开任务约束的离线抽取与审计

比普通 column description 更值得优先验证。只从 question 和已有 external knowledge
抽取：

```json
{
  "population_filters": [],
  "measure": {"op": "count", "column": "recipe_id", "distinct": false},
  "formula": null,
  "output_slots": []
}
```

约束必须记录来源片段与置信度，禁止从 gold SQL 反编译。第一阶段只离线审计，不改变
模型上下文。建议分层：

- A：显式字段、运算、公式或 literal，允许进入后续门控；
- B：需要自然语言推断，审计用；
- C：与数据/标注冲突或有多种合理解释，隔离。

只有 A 层在人工样本上达到至少 95% precision，且对稳定正确控制零冲突，才测试
lineage 上的 `task_constraint_violation`。约束错误的危害大于缺少约束。

### P0：population/grain 的因果 SFT 与过程信用

优先覆盖：

- 一行代表事实、实体还是聚合组；
- join 前后的实体 key 和 multiplicity；
- `count(*)`、`count(column)`、`count_distinct(entity)`；
- 同一固定 population 上的多个条件指标；
- join 后再 extreme，而不是先在错误总体上取 extreme；
- 最终只投影请求槽位。

训练目标不应是完整 gold 路径，而应来自真实 model↔harness episode，并对最早的错误
语义分支和后续恢复分开记信用。当前 16 条反事实通过记录只能作为候选；仍需 fresh
causal replay、执行验证、质量与 no-leak gate，且在现行工具可用性门控通过前不得直接
晋升为 SFT 数据。

### P0：按稳定性重组教师数据选择

推荐次序：

1. 57 条策略敏感题：优先复用已经存在的 verifier-correct 因果轨迹，并审计成功/失败
   轨迹的最早分歧。
2. 18 条已分类的非歧义稳定语义失败（9 population + 6 mapping + 3 slots）：
   外部教师定向重试，但不暴露 gold。
3. 3 条 F 中非合法、其他运行也失败的题：先重跑确定稳定根因。
4. 15 条歧义题：不进入通用训练。
5. 107 条稳定成功题：作为 replay 与 no-regression control。

这比按“有没有 process error”或按难度随机采样更接近当前真正的能力边界。

### P1：单边、对称 join，仅作为可靠性改进

action-block v34 已在小门控中消除观察到的 join 调用错误，但 fixed-1000 表明绝大多数
失败是无过程错误的合法答案。因此建议继续做 atomic 单边 join 独立门控，但目标应写成：

- join argument/process error 显著下降；
- self-join 和 multi-hop 控制不回归；
- token/action 成本不显著增加；
- 正确率有净增才扩容。

不要把“失败轨迹中出现 join”直接解释为 join API 导致失败。

### P1：终止槽位的因果训练，高于继续扩展 schema 描述

77/288 的 wrong answer 直接出现列宽不一致，人工反事实也多次由最终精确投影恢复。
优先增加：

- requested slots 的公开解析；
- terminal 前按槽位顺序审计；
- helper/ranking/key 列必须移除；
- 不把 ID 自动替换为 label，也不擅自拼接字段。

仍应通过训练和高置信公开约束实现，不能从 gold sample 得到输出列数后回灌。

### P2：`read_subtable` → `inspect_rows`

改名并显式返回 `produces_handle:false` 是清晰、低风险的工程改进，但 F 的 56 条合法
错误中只有 2 条被归为 grounding/terminal evidence misuse，且 36 条稳定失败中没有
这一主类。它不应排在 population/grain 或任务约束之前。

### P2：typed numeric parsing 和 tie policy 只做 micro-gate

大样本中有 99 个 numeric mismatch，但这包含错误总体、错误 distinct 和错误公式，
不能据此给 numeric parser 估算 9.9pp 收益。当前只有少数人工案例确认字符串数值清洗
或 operand 方向问题。先做 8–16 题确定性 micro-gate。

并列策略同样要先审计 benchmark；显式 `include_all`/`truncate_exact_k` 只能表达约定，
不能决定隐藏标注究竟采用哪种约定。

### 暂停：普通静态 column semantic_name 扩充

已有 hard-60 和 full-context F 结果表明，低信息 semantic_name 增加 token 和策略扰动，
没有形成稳定增益。只有真正人工提供、来源明确的 description/data format，或高置信
task constraint，才值得重新做三臂消融。

## 7. 建议的实验门控

### 7.1 严格 pass@1 主指标

- 官方全集：`bird-set` pass@1；
- paired net gain / regression；
- legal termination；
- process error events；
- tokens、model turns、attempted primitives；
- stable-failure、variant-sensitive、stable-control 三个切片。

任何协议升级仍以 frozen-200 至少 150/200 为当前 promotion gate；不能把 oracle union
或 clean-subset 当正式分数。

### 7.2 教师数据生成

pass@2 可以提高找到成功因果轨迹的概率，但只报告为 teacher-data yield：

- 每个 episode fresh 开始；
- actor/teacher 不看 gold；
- 终端只由 `bird-set` verifier 接受；
- 成功轨迹 fresh replay；
- 质量、no-leak 和公开约束一致性全部通过。

pass@2 不能替代严格 pass@1 模型能力。

### 7.3 小门控阈值

对 task constraints、single-edge join、terminal slots 分开做 paired gate：

- 16 个 target + 16 个 stable control；
- 净恢复至少 3；
- control 回归不超过 1；
- legal rate 不下降；
- token 增长不超过 5%；
- 无显著新增 process-error 类型。

门控失败就停止该方向，不把多个机制合并后再猜归因。

## 8. 可复跑产物

分析脚本：

```bash
python3 src/eval/analyze_large_failure_corpus.py \
  --artifact-root /Users/hudou/Research/tabular_rl_research \
  --output-dir /Users/hudou/Research/tabular_rl_research/data/results/failure_corpus_scale_20260729
```

输出：

- `summary.json`：全量聚合、五次运行稳定性和人工分类交集；
- `fixed1000_failures.jsonl`：296 条失败的结构化特征；
- `modern_fixed200_stability.jsonl`：200 题的五次结果与稳定性标签。

本分析不修改任何原轨迹，不执行 gold SQL，也不把 gold 路径用于训练数据构造。
