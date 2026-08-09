# Exp16–Exp18 冻结轨迹策略更新诊断

日期：2026-08-08

## 问题与测量边界

本诊断检验：Exp16、Exp17、Exp18 的完整 BIRD-dev 性能几乎不变，是否对应于训练后策略状态也几乎没有变化。

测量使用训练池中保存的精确 `prompt_ids + response_ids`，不重新 rollout、不重渲染 prompt，也不访问 gold SQL。SFT2 与各 RL checkpoint 在同一 Qwen2.5-Coder-7B-Instruct、BF16 Hugging Face 前向实现下，对完全相同的已选 token 计算条件 logprob。由于 Exp16–Exp18 训练的是 full response，整段 response 是主指标；可解析 JSON tool token 是次指标。参数侧比较 196 个 LoRA 模块的有效增量 `B @ A`，而不仅是原始 adapter 参数。

这是训练后机制诊断，不是提高学习率或改变奖励稀疏度的因果消融。因此它可以确认“实际策略是否移动”，但不能单独证明哪一个超参数是根因。

## 完整性

| 数据 | 预期行数 | 实际行数 | 唯一键 | 非有限 response 分数 | 完整 SHA256 |
|---|---:|---:|---:|---:|---|
| Exp16/17 mixed60 | 2,252 | 2,252 | 2,252 | 0 | `cad0a3346760dabee7268a1576656741aa07f2825df6159178da0f5fbb6eb1c1` |
| Exp18 mixed120 | 4,425 | 4,425 | 4,425 | 0 | `b3debbf9db0f48cc9d9953de55fac56becb4228ada939dcc7e644806e756d96d` |

Exp16/17 共计 437,845 个 response token，Exp18 共计 803,911 个 response token。57/2,252 与 108/4,425 条历史 response 的 JSON tool span 不可解析；这些行只缺少次要 tool-only 指标，完整 response 指标全部保留。

## 完整 BIRD-dev 结果

| 模型 | 正确/1,534 | Accuracy | 相对 SFT2 | Valid | Avg steps |
|---|---:|---:|---:|---:|---:|
| SFT2 | 762 | 49.674% | — | — | — |
| Exp16 dense uniform 60 | 764 | 49.804% | +2 | 1,222 | 8.1056 |
| Exp17 dense strategic 60 | 760 | 49.544% | -2 | 1,199 | 8.2855 |
| Exp18 dense uniform 120 | 764 | 49.804% | +2 | 1,196 | 8.3960 |

Exp18 是单 seed 探索性结果，不代表跨 seed 稳定性。

## 策略移动主结果

`奖励方向一致率` 定义为 `advantage × logprob_delta > 0` 的 transition 比例：正奖励期望提高原动作 logprob，负奖励期望降低原动作 logprob。

| 实验 | transitions | 有效 LoRA 位移 / SFT2 | response 平均绝对 Δlogp | 中位绝对 Δlogp | `|Δlogp| > 0.01` | tool 平均绝对 Δlogp | 奖励方向一致率 | advantage–Δlogp Pearson |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Exp16 | 2,252 | 0.1602% | 0.001229 | 0.000914 | 0/2,252 | 0.000489 | 49.60% | 0.0314 |
| Exp17 | 2,252 | 0.1612% | 0.001235 | 0.000940 | 0/2,252 | 0.000495 | 49.51% | 0.0236 |
| Exp18 | 4,425 | 0.1724% | 0.001204 | 0.000874 | 0/4,425 | 0.000503 | 51.39% | 0.0461 |

三个实验都产生了非零更新，但行为分布移动处于约 `1e-3` 的量级，没有任何 transition 的 response 平均绝对变化超过 0.01。把题目数和 optimizer steps 从 60 扩大到 120，最终有效 LoRA 相对位移只从 0.1602% 增至 0.1724%，而不是接近翻倍。

## 正负奖励与关键路径

| 实验 | 正奖励 transition 的期望方向比例 | 负奖励 transition 的期望方向比例 | backslice 关键步期望方向比例 | evidence 关键步期望方向比例 |
|---|---:|---:|---:|---:|
| Exp16 | 50.49% | 48.98% | 49.68% | 53.61% |
| Exp17 | 47.79% | 50.72% | 44.37% | 53.61% |
| Exp18 | 50.03% | 52.34% | 49.45% | 53.73% |

最明显的问题不是单纯“更新小”，而是更新方向与所赋奖励只有接近随机的一致性。Exp17 特意加强的 backslice 正奖励反而只有 44.37% 的原动作 logprob 向期望方向变化；其 backslice 平均 Δlogp 为 `-1.305e-4`。

按 transition advantage 直接求和的奖励质量为：

| 实验 | 正质量 | 负质量 | 净质量 |
|---|---:|---:|---:|
| Exp16 | +117.485 | -117.586 | -0.101 |
| Exp17 | +158.910 | -117.586 | +41.324 |
| Exp18 | +236.937 | -228.379 | +8.557 |

因此，“Exp16 的正负标量质量近乎抵消”成立；但它不能解释全部现象。Exp17 已有明显正净质量，最终位移和奖励方向一致性仍与 Exp16 几乎相同。奖励标量和参数梯度不是同一个量；full-response 平均、不同 prefix/token 的梯度几何、LoRA/学习率限制及共同监督方向都可能使净策略更新远小于标量奖励和。

## Checkpoint 与跨实验方向

| 比较 | 有效 LoRA 更新余弦 |
|---|---:|
| Exp16 step60 vs Exp17 step60 | 0.97525 |
| Exp16 step60 vs Exp18 step120 | 0.83176 |
| Exp17 step60 vs Exp18 step120 | 0.83092 |

Exp16/17 虽然奖励权重不同，最终更新方向高度相似。各实验后段保存点相对 SFT2 的累计更新方向也稳定：Exp16 step40/50/60 两两余弦为 0.995–0.997，Exp17 为 0.997–1.000，Exp18 step100/110/120 为 0.992–0.995。后续 step 主要在一个已经形成的很小方向上缓慢增长，没有形成明显更强的策略分离。

位移大小也不是性能的充分条件。作为只读对照，Exp15 Action-DPO、Teacher Union60、Strict120 的有效 LoRA 位移分别为 0.2833%、0.2237%、0.2756%，完整 dev 分别为 776、785、765。Strict120 移动更多但效果仍接近 SFT2，说明更新方向和训练数据选择至少与更新幅度同样重要。

## 结论

1. **“训练后状态几乎没变”得到直接支持。** 参数空间只有 0.160%–0.172% 的有效 LoRA 相对位移；冻结轨迹上的 response 平均绝对 Δlogp 只有约 0.0012；完整 dev 只变化 -2 到 +2 题。
2. **这很可能是性能没有明显变化的直接机制之一。** 训练没有把已奖励和已惩罚动作稳定推向相反方向，尤其 backslice 信号没有可靠落到对应行为上。
3. **但不能把根因简化成“正负奖励恰好 1:1 抵消”。** Exp16 符合这一描述，Exp17 不符合，却仍得到同样微弱且高度相似的更新。更准确的描述是：当前 dense full-response 目标在现有学习率、归一化和 LoRA 约束下，产生了很小、共同成分占主导、与 transition 奖励符号弱相关的策略移动。
4. **扩大同一数据构造并未修复信用分配。** Exp18 将规模翻倍后只略增参数位移，奖励方向一致率仍约 51%，因此 scale120 的退化/持平不能只归因于样本数不足。

若继续做因果验证，最小的三个对照应是：固定同一 batch 的正负样本分别反传并测梯度余弦；key-step-only 与 full-response dense 的匹配对照；在不跑完整 dev 前，先用冻结轨迹 `Δlogp` 阈值决定学习率/损失权重是否值得扩展。

## 产物

远端根目录：

`/home/dengyan/tabular_rl_outputs/diagnostics/dense_policy_update_20260806`

核心文件：

- `exp16_exp17_scores.jsonl`：2,252 条逐 transition 完整评分。
- `exp18_scores.jsonl`：4,425 条逐 transition 完整评分。
- `exp16_exp17_scores.summary.json`、`exp18_scores.summary.json`：全量分类统计。
- `exp16_exp17_scores.manifest.json`、`exp18_scores.manifest.json`：输入、adapter SHA256 与测量契约。
- `exp16_exp17_policy_movement.analysis.json`、`exp18_policy_movement.analysis.json`：奖励方向、分位数和分组分析。
- `lora_update_comparison.json`：Exp16–Exp18 各保存点的原始 adapter 与有效 `B @ A` 位移。
- `exp15_lora_update_comparison.json`、`teacher_union_lora_update_comparison.json`：旁证实验的参数位移对照。

