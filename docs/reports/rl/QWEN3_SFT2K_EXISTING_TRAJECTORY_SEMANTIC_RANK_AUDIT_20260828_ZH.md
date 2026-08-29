# Qwen3 SFT2K 现有轨迹语义排序审计（2026-08-28）

## 结论

当前 `Q = 0.10 schema + 0.45 semantic + 0.45 answer` 不能作为已经验证通过的语义轨迹奖励。

它对正确/错误轨迹的总体分离很强，但消融表明主要判别力来自 `answer`。去掉
`answer` 后，`semantic + schema` 的 AUC 约为 0.64；在错误轨迹内部，它与答案误差几乎
不相关，并且不同工具实现存在明显的平均分偏置。因此当前结果支持“部分答案相似度可用于
失败轨迹排序”，不支持“Gold-state semantic 已能可靠评价工具过程”。

正确轨迹的优化奖励固定为 `+1` 是必要保护。若直接用 raw quality 优化所有轨迹，会把一批
完全正确但未被规范化器识别的实现打低分。

## 数据边界

- 只读取两个既有的 BIRD-train purebird1000 生成批次；没有模型调用、没有新 rollout、没有
  优化器更新，也没有使用 1534 条评测集。
- 两批共有 2000 个唯一 task，每题恰好一条已有轨迹：1436 条 verifier-correct，564 条错误。
- 最终进入当前 SFT 合并数据的是其中 1308 条正确轨迹（671 + 637）；原始 SFT1 的 678 条
  正确轨迹因没有同批失败对照，不进入主分离统计。
- 校准版完成 1997/2000 条评分：1434 correct、563 incorrect。排除一条 API failure，以及两条
  超过 60 秒的 SQLite 回放；1308 条 SFT-admitted correct 中评分成功 1307 条。

输入 pool SHA256：
`315be3047843a5516f9e2098e5610a7830b8e75ff4b3b9144bd281c34668159e`。

## 评分定义

对 Gold SQL 和终端证据表的递归 relation lineage 编译共享语义 IR：join、predicate、grain、
aggregate/compute、set/distinct、rank/limit。语义重叠使用 weighted Jaccard：

```text
Qsemantic = TPw / (TPw + FPw + FNw)
Qraw      = weighted_mean(0.10 Qschema, 0.45 Qsemantic, 0.45 Qanswer)
R         = +1                         if verifier-correct
            -1 + 0.4 Qraw              otherwise
```

`Qanswer` 使用 BIRD-set 兼容的行 Jaccard、列宽匹配和结果行数比例。缺失分量会在
`weighted_mean` 中被重新归一化。因此 semantic 不可用时，answer 的有效占比会从 45% 上升到
81.8%，这是解释总体 AUC 时必须注意的事实。

## 总体分离

| 指标 | Correct | Incorrect |
|---|---:|---:|
| 数量 | 1434 | 563 |
| raw quality 均值 | 0.7464 | 0.2250 |
| raw quality P10 / P90 | 0.4746 | 0.4059 |

- mean gap：0.5215；raw-quality AUC：0.9786。
- SFT-admitted correct 均值 0.7467，对全部 incorrect 的 AUC 同样为 0.9786。
- 只看完整回放与原标签一致的 1971 条，AUC 仍为 0.9815，粗粒度分离不是少量回放差异造成的。

但这个结论不能单独证明 semantic 有效，因为 correct 的 answer 基本为 1，incorrect 的 answer
通常很低。

## 分量消融

| 分数 | AUC | 统计范围 |
|---|---:|---|
| raw combined | 0.9786 | 1997 条 |
| answer only | **0.9968** | 1997 条 |
| schema only | 0.5958 | 1997 条，缺失记 0 |
| semantic only | 0.6415 | semantic 可计算子集 |
| semantic + schema | 0.6397 | semantic 可计算子集 |
| semantic + schema with fallback | 0.6173 | 1997 条 |

只有 1172/1997 = 58.69% 的轨迹同时通过 Gold 与 candidate 的 conservative semantic 编译。
172 条存在 Gold 编译原因，697 条存在 candidate lineage 编译原因，二者有重叠。

因此 raw AUC 的主要来源是 answer，而不是 semantic state matching。

## “高质量错误是否错误更小”

用 raw combined 排序时，incorrect 的 quality 与 answer error 的 Spearman 为 -0.488；raw
quality 四分位从低到高时，平均 answer error 为 0.860、0.842、0.821、0.645。表面上满足
“高质量错误更小”。

但 `answer` 本身就是 raw quality 的 45%（semantic 缺失时有效占比 81.8%），所以该相关性
部分是定义产生的。去掉 answer 后：

- eligible failures 上 `semantic + schema` 与 answer score 的 Spearman 为 -0.036；
- semantic-only 与 answer score 的 Spearman 为 0.005；
- process-only 四分位从低到高的平均 answer error 为
  0.811、0.824、0.768、0.796，没有单调关系。

所以目前只能说 partial denotation 能识别较小答案错误，不能说 semantic 过程分数能识别较小
语义错误。

## 正确等价路径被低估

现有数据每题只有一条轨迹，因此无法做同题两个正确工具路径的成对 path-invariance 检验。
不能拿不同问题中偶然相同的结果冒充等价路径。

但单路径 false-low 检验已经发现问题。在 1415 条“记录标签与完整回放一致”的正确轨迹中：

- 215 条 raw quality < 0.5；
- 247 条 semantic score = 0；
- 572 条 semantic score 不可用。

多条完全正确的 `scalar_compute` 路径得到 `answer=1、semantic=0、schema=0、raw=0.45`。例如
`bird_train_02617`、`bird_train_04911`、`bird_train_03186`。这说明当前 IR 没有把跨
`group_aggregate`、`scalar_compute` 等多阶段工具形成的表达式 DAG，折叠到 Gold SQL 的等价
代数表达式。

错误轨迹的最高 raw quality 为 0.951，高于大量正确轨迹的 0.45。虽然阶段一把所有正确轨迹
固定为 `+1`，避免了直接抑制，但 raw quality 仍不具备跨任务的可靠序意义。

## 工具/实现形状偏置

在 correct 且 semantic 可计算的轨迹中，包含不同工具时的平均 semantic score 为：

| 工具 | 平均 semantic |
|---|---:|
| `extreme_value_select` | 0.489 |
| `join_tables` | 0.411 |
| `project` | 0.334 |
| `group_aggregate` | 0.272 |
| `read_subtable` | 0.209 |
| `set_op` | 0.096 |
| `scalar_compute` | **0.036** |

这些是 tool-presence 条件均值，仍混有任务结构差异，不能解释成工具本身的因果效果；但
0.489 对 0.036 的跨度足以否定“当前分数已经实现工具无关的路径归一化”。直接做全局分数
标准化也不能修复这个问题，因为错误来自 IR 对等价组合的识别能力。

## 对 RL 目标的建议

当前可以安全验证的统一目标应退回到：

```text
R(trajectory) = +1                         if exact BIRD-set correct
                -1 + alpha * Qanswer       otherwise
```

其中 `alpha` 保持较小并确保所有错误奖励严格小于所有正确奖励。这个目标验证的是
result-aware trajectory ranking，不应命名为 semantic process credit。

`Qsemantic` 暂时只做审计特征，不进入优化。它至少需要完成以下门槛后再加入：

1. 修复真实 relation lineage 的列解析覆盖，特别是 join 后 filter/project 的逻辑列名；
2. 对 aggregate/compute 多阶段组合、等价 predicate、别名和代数分解做规范化；
3. semantic coverage 明显提高，并在 failures 内对独立的 answer error 呈稳定单调关系；
4. 构造同题多条正确轨迹的独立 path-invariance 集，确认不同合法路径不会被系统性打低；
5. 按 Gold SQL shape 分层检查工具条件均值，避免一种工具天然获得更高 baseline。

在这些门槛之前，不应启动 semantic-reward RL；更不应进行逐步骤 credit 分配。
