# Qwen3-8B Atomic v26：BIRD-dev 全部错误与 BIRD-train 错误一致性审核

日期：2026-08-28

## 结论

逐题审核了冻结 BIRD-dev 1534 greedy 结果中的全部 696 道错误，并与既有 BIRD-train 2000
轨迹中的全部 564 道错误对照。结论必须拆成两句：

1. **错误机制集合高度重合。** 666/696 = 95.69% 的 dev 错误在 train 中存在完全相同的终止或
   语义签名；3 条只有同一宽类、没有相同尾部组合；另外 27 条的最终失败形式没在 train 出现，
   但其前置工具错误机制在 train 中出现过。
2. **错误分布明显不一致。** dev 有 217/696 = 31.18% 的错误不能合法作答，train 只有
   11/564 = 1.95%，相差 29.23 个百分点，dev 是 train 的 15.99 倍。

因此不能说“dev 错误和 train 错误一致”。更准确的判断是：**合法结束后的语义错误结构基本
复现，但 dev 显著放大了协议、执行、参数和步数失败。**

本轮是只读评测诊断。BIRD-dev 记录用于训练、reward 设计或 task selection 的数量均为 0；没有
模型调用、没有新 rollout、没有 optimizer update。

## 一、审核对象与逐题证据

dev 身份固定为：

- 模型：`qwen3-8b-atomic-v26-sft1-qlora`；
- 协议：Atomic `version26`，`think-json-v1`，protocol hash `4da19387399bd3a5`；
- greedy：temperature 0、top-p 1、每题一条轨迹、最多 30 steps；
- verifier：`bird-set`；
- 结果：838/1534 = 54.63%，错误 696。

每道 dev 错题都生成一条独立审核记录，包含：

- 终止是否合法、最终 failure type、全部 Harness error event 和错误消息族；
- 实际工具序列与终态 evidence handle；
- Gold SQL 与终态 relation-derivation 的 source、join、predicate、grain、value、set、rank、output
  逐类 matched/missing/extra；
- 明确定义的结构距离：无合法答案、路线级、多决策、两类局部差异、一类局部差异、评分器盲点；
- train 中完全相同签名的数量、同一宽类数量和工具错误事件支持数。

逐题索引恰好 696 行、696 个唯一 example index；详细 JSONL 也恰好 696 条。这里的“逐题审核”是
基于 Harness 状态和 Gold SQL 的确定性审核，不把模型自己写的 reasoning 当证据。

## 二、最大的分布差异：dev 的合法性失败

| 失败结果 | dev 错误 | train 错误 |
|---|---:|---:|
| 合法结束但答案错 | 479/696 = 68.82% | 553/564 = 98.05% |
| 无合法答案 | 217/696 = 31.18% | 11/564 = 1.95% |

dev 的 217 条无合法答案进一步分为：

| 最终 failure type | 数量 | 占全部 dev 错误 |
|---|---:|---:|
| protocol error | 131 | 18.82% |
| execution error | 47 | 6.75% |
| argument validation error | 24 | 3.45% |
| max steps | 14 | 2.01% |
| context overflow | 1 | 0.14% |

这一差异不是由单个数据库造成的。11 个 dev DB 都出现无合法答案；它占各 DB 错误的比例从
17.1% 到 53.7%。

工具错误事件也明显更多：

- dev：380/696 = 54.60% 的错题至少有一次 error event；
- train：96/564 = 17.02%；
- 即使只看合法 wrong answer，dev 仍是 166/479 = 34.66%，train 是 86/553 = 15.55%。

dev 最常见的确定性错误消息族为：carrier shape 239 题、scalar grounding 71 题、unknown column
45 题、无 offset 的 `read_subtable` 被错误当作可分页工具 37 题。一个题可同时出现多个消息族。
其中 24 条以 argument validation 最终失败，而 train 中 argument validation 只作为 18 个错误
episode 的中间事件出现，从未成为最终 failure type。这是 dev 明确放大的终止机制。

这说明现有 train 错误集严重低估了当前 greedy Qwen 的“不能稳定完成合法工具轨迹”问题。这个差异
不能靠 predicate/value 等语义 reward 解释，也不应把每次格式错误都伪装成语义 credit。

## 三、条件化到合法 wrong answer 后，语义结构基本一致

排除无合法答案，仅比较 dev 的 479 条与 train 的 553 条合法 wrong answer：

| 语义 cohort | dev | train | dev-train |
|---|---:|---:|---:|
| source 或 join 路线错误（含 partial join） | 194/479 = 40.50% | 241/553 = 43.58% | -3.08 pp |
| source/join 正确，只差 1–2 个尾部语义族 | 92/479 = 19.21% | 110/553 = 19.89% | -0.68 pp |
| source/join 正确，但差 3 个以上尾部语义族 | 94/479 = 19.62% | 97/553 = 17.54% | +2.08 pp |
| 当前编译器不可评分 | 96/479 = 20.04% | 100/553 = 18.08% | +1.96 pp |
| 语义完全重合但 denotation 错 | 1/479 = 0.21% | 5/553 = 0.90% | -0.70 pp |
| 终态 evidence 无法形成 grounded relation | 2/479 = 0.42% | 0 | +0.42 pp |

这些比例相当接近。特别是严格 near-miss 占比为 19.21% 对 19.89%，source/join 路线错为
40.50% 对 43.58%。所以 train 中观察到的“约五分之一合法错误已经走对 route、只差一两个尾部
决策”确实在 dev 上复现。

但严格 near-miss 的具体尾部构成不完全相同。以各 split 的严格 near-miss 数为分母：

| 尾部族 | dev 92 条 | train 110 条 | 变化 |
|---|---:|---:|---:|
| output | 61/92 = 66.30% | 66/110 = 60.00% | +6.30 pp |
| predicate | 38/92 = 41.30% | 39/110 = 35.45% | +5.85 pp |
| set/distinct | 13/92 = 14.13% | 10/110 = 9.09% | +5.04 pp |
| value/formula | 11/92 = 11.96% | 35/110 = 31.82% | -19.86 pp |
| rank/limit | 12/92 = 13.04% | 25/110 = 22.73% | -9.68 pp |
| grain | 4/92 = 4.35% | 13/110 = 11.82% | -7.47 pp |

因此只能把“尾部局部错误”作为可迁移宽类，不能把 train 中各工具或各尾部族的频率直接当成 dev
的 reward 权重。

## 四、30 条没有相同 train 精确签名的题

逐题复核后，30 条分成四组：

1. 24 条 `terminal:argument_validation_error`：train 中有 argument-validation 中间事件，但没有
   以它最终失败的轨迹；常见原因是使用不存在的 `read_subtable(offset=...)`，或 join 参数不符合
   v26 schema。
2. 1 条 `terminal:context_overflow`：`bird_dev_00412`；train 没有相同终止失败，只有其前面的
   carrier-shape error 机制。
3. 2 条 `semantic:no_grounded_terminal`：`bird_dev_01433` 直接把 perception step 当 evidence，
   `bird_dev_01524` 直接引用错误 base table。它们最初会被“空 category 视为共同为空”的规则误标
   为 semantic exact；本次加了明确 guard，终态 overlap 为空且总分为 0 时不再算 exact。
4. 3 条只有新的尾部组合，但宽类在 train 存在：`bird_dev_00129`、`bird_dev_00992`、
   `bird_dev_01496`。其中 `01496` 只错 grain：模型按 Segment 聚合，而 Gold 按 CustomerID 聚合，
   最终返回 LAM 而不是 KAM。

所以没有证据表明 dev 出现了全新的“语义能力族”；真正的新现象主要是既有工具错误在 dev 中更常
成为不可恢复的最终失败。

## 五、评分器盲点和三条“预览相同但判错”

dev 只有一条真正的 semantic-exact-but-wrong：`bird_dev_00013`。Gold 使用
`CAST(NumGE1500 AS REAL) / NumTstTakr`，轨迹直接使用整数列相除，导致 top-3 完全不同；当前语义
IR 把二者归一成同一除法表达式。这与 train 中已经发现的整数除法盲点一致，证明语义满分不能替代
结果 verifier。

另有 3 条 benchmark-wrong 的 `gold_sample` 与 `pred_sample` 存储预览完全相同：

- `bird_dev_00004`：前五行相同，但模型把 FundingType 条件放在 `schools`，Gold 放在 `frpm`；
- `bird_dev_00049`：前五行相同，Gold 是 subquery + DISTINCT，完整结果仍不等价；
- `bird_dev_00142`：前五行相同，但模型走 `order` 表，Gold 走 `trans` 表。

这些字段最多保存五行预览，不能用“预览相同”推翻完整 `bird-set` verifier，也不能作为 RL reward。

96 条合法错误不可评分中，主要是 Gold subquery 48 条、重复物理表 23 条、window 3 条，其余多为
trajectory column lineage 歧义。`unscorable` 表示编译覆盖不足，不能解释为模型错误轻或重。

## 六、对 RL 方法的直接含义

这次 dev 审核不改变此前的统一目标：第一阶段仍应在 **BIRD-train 新鲜 on-policy K-way rollout**
上验证 Frontier Result-only GRPO，而不是用 dev 的错误类型或 Gold overlap 选择训练题。

但对实验设计有两个明确约束：

1. **先把“可比较轨迹”与“协议失败”分开报告。** 结果 RL 的主分析应报告所有题，同时单列合法
   rollout 率。若训练前后提升主要来自 carrier/schema 合法率，而合法 wrong answer 的正确率不变，
   不能宣称语义推理提升。
2. **Frontier 必须从当前 SFT policy 的 train rollout 重新挖掘。** 现有 train2000 错误虽然能
   复现合法语义错误宽类，却严重低估当前 Qwen 的协议/执行失败；不能把既有生成批次的
   错误频率当成当前 policy 的采样分布。

对于 479 条合法 wrong answer，dev 与 train 的 19% 严格 near-miss 复现支持继续验证
frontier-sampled result-only GRPO；但它只支持“任务采样方向可能迁移”，不支持增加逐工具 dense
reward。对于 protocol、unsupported argument、重复读取到 max-steps 等失败，优先手段应是协议
约束、causal SFT 覆盖和 on-policy 数据，而不是按工具类型设固定奖励。

## 产物

- 全量摘要：
  `data/results/qwen3_8b_sft1_v26_dev1534_error_train_consistency_20260828/summary.json`
- 696 条详细审核记录：
  `data/results/qwen3_8b_sft1_v26_dev1534_error_train_consistency_20260828/dev_incorrect_casebook.jsonl`
- 696 条可读索引：
  `data/results/qwen3_8b_sft1_v26_dev1534_error_train_consistency_20260828/dev_incorrect_casebook_index.md`
- 逐题 Gold—终态语义证据：
  `data/results/qwen3_8b_sft1_v26_dev1534_error_train_consistency_20260828/dev_incorrect_semantic_scores.jsonl`
- 输入输出哈希与 dev 禁止训练声明：
  `data/results/qwen3_8b_sft1_v26_dev1534_error_train_consistency_20260828/manifest.json`
- 审核代码与测试：
  `src/rl/diagnostics/audit_dev_incorrect_train_consistency.py`、
  `src/rl/diagnostics/test_audit_dev_incorrect_train_consistency.py`

回归测试 10/10 通过。
