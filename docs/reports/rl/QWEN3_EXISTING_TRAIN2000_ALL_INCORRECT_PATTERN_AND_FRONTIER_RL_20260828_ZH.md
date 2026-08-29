# 现有 BIRD-train 2000 全部错误轨迹规律与 Frontier RL 目标

日期：2026-08-28

## 结论

本轮分析了冻结 BIRD-train 2000 轨迹中的全部 564 条错误记录，而不只分析能够被语义评分器覆盖的
462 条。数据支持“RL 应优先把大致做对、但最后局部决策不稳定的题推成正确”这一方向，但需要把
“大致做对”的定义从跨题 Gold 重合分数改成**同题 SFT 策略是否同时产生正确和错误 rollout**。

最适合首先验证的不是新的 dense reward，而是 **Frontier Result-only GRPO**：

1. 对同一训练问题从冻结 SFT checkpoint 采样 (K) 条 rollout；
2. 只有同时出现正确和错误结果的题进入 RL，即 (0<c_q<K)；
3. reward 仍只有确定性的终态正确性 (y_{qi}\in\{0,1\})；
4. GRPO 对整条轨迹做组内优势，不加入逐工具奖励或 Gold 语义分数；
5. 全错题先增加探索或回到 causal SFT，不能靠 result-only GRPO 学习；
6. Gold 不可靠的题必须先被独立可靠性门排除。

这一定义直接对应“RL 让 SFT 已经学到的行为更显著”：只有当 SFT 自己曾采到正确轨迹时，才有
证据说明正确行为已存在于策略分布中。单条轨迹的高语义分不能提供这个证据。

## 数据边界

- 输入：冻结 BIRD-train 2000 既有轨迹，1436 条正确、564 条错误；
- 原始输入 SHA-256：
  `315be3047843a5516f9e2098e5610a7830b8e75ff4b3b9144bd281c34668159e`；
- 语义评分记录覆盖 1999 条，唯一缺失记录为 API failure；
- 本轮模型调用 0、optimizer update 0、BIRD-dev 1534 使用 0；
- Gold SQL、正确性、答案相似度和人工严重度只用于离线验证，不是候选 reward 输入。

## 一、564 条错误的完整分解

| 自动 cohort | 数量 | 占全部错误 | 含义 |
|---|---:|---:|---|
| source/join 正确，只差 1 个尾部语义族 | 32 | 5.67% | 最严格 near-miss |
| source/join 正确，只差 2 个尾部语义族 | 78 | 13.83% | 较宽 near-miss |
| source/join 正确，但差 3 个以上尾部语义族 | 97 | 17.20% | 路线已对，后半段仍不稳定 |
| 语义义务完全重合但结果错误 | 5 | 0.89% | 当前评分器盲点 |
| source 正确、join 部分正确 | 7 | 1.24% | 局部路线错误 |
| source 或 join 路线错误 | 234 | 41.49% | 更像能力/知识缺失 |
| 当前语义编译器不可评分 | 100 | 17.73% | 评分覆盖问题，不等于模型能力问题 |
| 无合法终止 | 10 | 1.77% | 执行、协议或 max-step 失败 |
| API/记录缺失 | 1 | 0.18% | 基础设施失败 |

前四类中 source 与 join 都精确匹配的共有 212/564 = **37.59%**。其中最严格的一到两个尾部
不一致共有 110/564 = **19.50%**。这 110 条是当前单轨迹数据能提供的 near-miss 预筛集合，
但不是可以直接进入 RL 的训练集合。

## 二、110 条严格 near-miss 的错误集中在哪里

尾部不一致族的出现次数如下；一条轨迹可以同时出现两个族：

| 语义族 | 出现次数 |
|---|---:|
| output | 66 |
| predicate | 39 |
| value / formula | 35 |
| rank / limit | 25 |
| grain | 13 |
| set / distinct | 10 |

最常见的联合模式为：

| 模式 | 数量 |
|---|---:|
| value + output | 35 |
| predicate | 14 |
| output | 12 |
| predicate + output | 10 |
| grain + rank | 8 |
| predicate + rank | 7 |
| predicate + set | 5 |
| rank | 5 |

这说明高价值错误并不平均分布在所有工具上，而主要集中在最后的计算表达式、过滤条件、投影列和
排序范围。不能据此给 `scalar_compute`、`condition_filter` 或 `project` 定义固定负奖励；这些工具
只是承载了问题语义，工具出现频率仍受题型混杂。

110 条 near-miss 中只有 20 条出现过工具错误，90 条完全没有工具错误；平均约 6.84 个 turn。
因此强化格式合法性、统一惩罚 tool error 或奖励更短轨迹，都不会直接解决大部分 near-miss。

## 三、全量题型规律

不能只看某类在错误数据中的数量，因为高频题型自然产生更多错误。应同时看其在全部 2000 条中的
出现次数和正确率。

| Gold SQL/问题特征 | 总数 | 错误数 | 正确率 | 判断 |
|---|---:|---:|---:|---|
| join | 1569 | 441 | 71.89% | 错误数高主要来自高暴露，不能说明 join 工具本身差 |
| aggregate | 1044 | 304 | 70.88% | 已有较强基础，但仍是最大误差来源之一 |
| percentage/ratio 问题 | 184 | 70 | 61.96% | 明显处于可优化边界 |
| average 问题 | 145 | 55 | 62.07% | 明显处于可优化边界 |
| date/time 问题 | 206 | 73 | 64.56% | 单位、极值和日期表达容易局部出错 |
| conditional SQL | 261 | 93 | 64.37% | 条件聚合与范围选择不稳定 |
| group by | 264 | 90 | 65.91% | grain 是重要尾部决策 |
| limit | 494 | 167 | 66.19% | rank/limit 是重要尾部决策 |
| order by | 462 | 148 | 67.97% | 与 limit/grain 联合出现错误 |
| distinct | 298 | 96 | 67.79% | entity/row grain 仍不稳定 |
| set operation | 8 | 7 | 12.50% | 样本极少且基础能力不足，更适合补 SFT |
| LIMIT 无 ORDER BY | 40 | 22 | 45.00% | 高 Gold/非唯一答案风险，应从 reward 中排除 |

最适合 RL frontier mining 的题型不是正确率最低的稀有能力，而是 percentage、average、date、
conditional、group、limit 等已经达到约 62%–68% 正确率、同时仍有足够失败样本的能力。set
operation 只有 8 条且 7 条错误，更像 SFT 覆盖缺口；直接用 RL 很难从全错 group 得到信号。

## 四、为什么 110 条不能直接按语义分数训练

新增 72 条分层人工审计中，有 21 条落入自动 near-miss cohort：

- 12/21 存在问题、Gold SQL、非唯一答案或输出契约冲突；
- 去掉 Gold 冲突后只剩 9 条，其平均真实错误严重度为 1.89；
- 对比之下，错误 source/join 路线且 Gold 相对可信的 16 条，平均严重度为 2.31。

这说明 near-miss cohort 的方向是对的：在 Gold 可信条件下，它确实比错路线的失败更接近正确。
但它没有解决 Gold 污染，不能直接用于 reward。典型对照包括：

- 真实 near-miss：`bird_train_00934` 漏 top-k、`01105` 选错 offense 指标、`03523` 分组键错误、
  `05166` 把格式化工资按字符串比较；
- Gold/答案契约冲突：`bird_train_03846` 的“任意十个”、`06021` 的 at-least 边界、
  `01240` 的车名与内部 ID、`01853` 的 Gold 漏输出电影标题。

还有 5 条轨迹的八类终态语义义务全部重合但答案仍错误，其中 `bird_train_03211`、
`bird_train_05015` 是整数除法错误，`bird_train_00860` 是先 top-k 再一对多 join 的作用域错误。
这直接证明即使语义分数为 1，也不能替代执行结果 verifier。

## 五、建议的统一 RL 目标

### 第一阶段：Frontier Result-only GRPO

对每个训练问题 (q) 从同一个冻结 SFT policy 采样 (K) 条轨迹，令：

\[
y_{qi}=\mathbf{1}[\text{trajectory }i\text{ 的终态 denotation 正确}],
\qquad c_q=\sum_{i=1}^{K}y_{qi}.
\]

定义真正的 SFT frontier：

\[
F(q)=\mathbf{1}[0<c_q<K].
\]

只在通过 Gold 可靠性门且 (F(q)=1) 的问题上更新，组内优势保持标准 result-only GRPO：

\[
A_{qi}=\frac{y_{qi}-\bar y_q}{\sigma_q+\epsilon}.
\]

这里没有额外语义奖励。正确轨迹整条提升，错误轨迹整条抑制；任务采样聚焦于 frontier，而不是用
Gold 重合度改变 reward 大小。这样既符合 GRPO，也准确实现“让 SFT 已经会但不稳定的内容变显著”。

任务状态有明确解释：

- (c_q=K)：已经稳定做对，不需要 result-only policy update；
- (0<c_q<K)：策略已有正确行为但概率不足，是 RL 的主要对象；
- (c_q=0)：策略当前采样不到正确行为，result-only GRPO 没有组内优势；应增加探索，仍全错则补
  causal SFT/teacher 数据，而不是伪造 dense reward。

### 第二阶段候选：工具状态配对的 frontier

在第一阶段验证后，可以进一步只优先采样“正确与错误 rollout 使用完全相同 source 表集合和 join
edge 集合，但尾部动作不同”的 frontier group。这一判断来自 Harness 的确定性 relation state，
正确 rollout 是同题行为锚点；不需要把 Gold SQL 编译成唯一工具路径，也不按工具类型校准分数。

如果之后研究 credit assignment，可以把正确/错误配对中完全相同的 Harness state/action 前缀屏蔽，
只从第一个分歧动作开始优化。但这是第二阶段实验；当前基础验证仍应使用整轨迹 GRPO。

## 六、下一实验应验证什么

建议做一个不训练的 BIRD-train rollout gate，然后才决定是否训练：

1. 对 110 条单轨迹 near-miss 预筛题及匹配控制题各生成固定 (K) 条 SFT rollout；
2. 统计每题 (c_q)，检验 near-miss 预筛是否真的富集 (0<c_q<K) 的 frontier；
3. 对 frontier 中正确/错误轨迹比较确定性 source/join state，计算 route-paired 覆盖；
4. 将 (c_q=0) 的题单独归为 SFT/data 缺口，不送进 result-only RL；
5. Gold 可靠性门未通过的题全部排除；特别是无 ORDER BY 的 LIMIT、any-N 和边界冲突；
6. 只有 frontier 富集门通过后，才比较 matched uniform result-only GRPO 与 frontier-sampled
   result-only GRPO。两臂保持 checkpoint、协议、rollout 数、optimizer update 和 token budget 一致。

主要验证指标应是：frontier 题正确概率是否上升、BIRD-train 不相交留出题是否同步提升、已稳定正确
能力是否回退。不能只看训练 cohort 的 reward。

## 产物

- 全量汇总：
  `data/results/existing_sft2k_all_incorrect_patterns_20260828/summary.json`
- 564 条逐轨迹确定性特征：
  `data/results/existing_sft2k_all_incorrect_patterns_20260828/incorrect_rows.jsonl`
- 分析代码：
  `src/rl/diagnostics/analyze_all_incorrect_trajectory_patterns.py`

本轮没有启动训练，也没有生成新 rollout；相关分析与既有评分回归测试 24/24 通过。
