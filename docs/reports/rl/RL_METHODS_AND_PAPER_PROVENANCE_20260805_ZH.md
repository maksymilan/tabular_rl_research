# 当前 RL 实验方法与论文来源：客观记录

更新日期：2026-08-05  
范围：SFT2 之后已经实际训练或形成正式诊断的 RL、preference optimization 与
distillation 实验，包括 Exp0–Exp18、Teacher-union60 和 Strict120。只记录实验做法、
数据规模、监督信号、评测数字和方法来源；未实际训练的设想不计入。

## 1. 实验信号分类

当前实验使用过以下六类训练信号：

1. terminal denotation 二值奖励；
2. harness 重放产生的局部过程信用；
3. 同题正确/错误轨迹的序列排序；
4. 同一精确 prefix 下的 DPO；
5. 固定轨迹池上的 dense outcome shaping；
6. 当前 student rollout 上的多教师 on-policy distillation，并叠加 verified repair DPO。

PPO/GRPO、DPO 与 on-policy distillation 使用了对应论文中的训练目标或基本框架。
BackSlice、确定性局部惩罚、action-token mask、dense uniform/strategic、K=4 教师资格
路由、四锚点并行修复与 verifier 阈值是本项目的具体数据或信用分配实现。

### 1.1 PPO 与 GRPO 的实际使用口径

`TransitionGRPOTrainer` 继承 TRL `GRPOTrainer`，但 trainer 类名不等于每个实验都在使用
标准 GRPO。实际损失由传入的 advantage 与启用的 loss 分支决定：

| 实验 | advantage / loss 的实际来源 | PPO clip | group-relative advantage | 准确描述 |
|---|---|---:|---:|---|
| Exp1 | 同题 K=4 的 terminal `{0,1}` reward 做组内总体标准化 | 是，`epsilon=0.2` | 是 | GRPO-style group advantage + PPO-clipped surrogate |
| Exp2 | 每个 assistant turn 的 replay process reward 直接乘 logprob | 否 | 否 | 历史 Accelerate REINFORCE-style process update |
| Exp3–11 | 每个 turn 的 `step_rewards[turn_index]` 直接作为 advantage | 是，`epsilon=0.2` | 否 | critic-free PPO-clipped process policy gradient；部分实验另加 Rank |
| Exp12、Exp14、Exp16–18 | 固定离线 K=4 轨迹池的 step reward 作为 advantage | 是，`epsilon=0.2` | 否 | offline PPO-clipped process loss；另按配置叠加 Rank |
| Exp13 | 只启用 Rank，`policy_loss_coefficient=0` | 否 | 否 | reference-free pairwise Rank-only |
| Exp15 | frozen SFT2 reference 下的 exact-prefix chosen/rejected loss | 否 | 否 | Action-DPO |
| Union60、Strict120 | `clip(log p_teacher-log p_old)` 的 OPD 与 repair DPO | 否 | 否 | on-policy distillation + DPO |

因此，若必须用一个短标签概括主 TRL 系列：**Exp1 是 GRPO-style advantage 配上 PPO
clipping；Process 系列不是标准 GRPO，而是没有 value critic 的 PPO-clipped policy
gradient。** 它们也不是完整标准 PPO，因为没有训练 value model，也没有 GAE。

tokenwise policy 分支的形式为：

```text
ratio_t = exp(log pi_theta(y_t|s_t) - log pi_old(y_t|s_t))
L_policy = -mean_t min(ratio_t*A_t, clip(ratio_t, 1-epsilon, 1+epsilon)*A_t)
```

Exp1 的 `A_t` 来自 K=4 组内标准化并在轨迹各 turn 共享；Process 系列的 `A_t` 就是该
turn 的 harness reward。两者均没有 learned value baseline。

### 1.2 Rank 的精确定义

这里的 Rank 不是评测时的 reranker，也不是“把四条候选按分数取第一条”。它是训练时
加入的 reference-free pairwise loss：对同一道题 K=4 中的全部
`correct trajectory × incorrect trajectory` 组合，计算

```text
L_rank = lambda * softplus[-beta * (S_correct - S_incorrect)]
lambda = 0.5
beta = 0.1
```

`S` 是当前策略在选定 token/action 上的 logprob 聚合。它不减 frozen reference
logprob，也不要求两条轨迹共享 exact prefix，所以与 Exp15 的 DPO 不同。各版本为：

| 实验 | score token | score 范围 | 聚合 | 实际接收 Rank 梯度的范围 |
|---|---|---|---|---|
| Exp5 | 全 response | 全轨迹 | token logprob 求和 | 全 response |
| Exp8 | JSON action | 全轨迹 | token logprob 求和 | 全部 JSON action |
| Exp9 | JSON action | 全轨迹 | token logprob 求和 | correct 的 clean legal action；incorrect 的确定性坏 action |
| Exp10 | JSON action | conservative mask | token logprob 求和 | 与 score 相同的 conservative mask |
| Exp11、Exp13、Exp14 | JSON action | conservative mask | action 内 token mean，再跨 action mean | conservative mask |
| Exp16–18 | 全 response | dense-outcome mask | response 内 mean，再跨 action mean | dense-outcome mask |

`mean_action` 的两级平均用于去除长 JSON 参数和动作数量对 trajectory score 的直接尺度
影响。Rank 的一阶导数先按 trajectory pair 计算，再作为 detached coefficient 路由回
选中的 transition token；其梯度与上式一致。

## 2. 方法谱系与实测数字

| 方法族 | 已运行实验 | 实际监督信号与信用粒度 | 与论文的对应关系 | 实测数字 |
|---|---|---|---|---|
| REINFORCE 工程基线 | simple-process pilot、Exp2 历史单卡 QLoRA backend | 采样轨迹的 scalar/turn return 乘策略 logprob | 优化器对应 Williams 1992；表格 harness、reward 与 QLoRA 为项目实现 | Exp2 equal-300：pass@1 37.67%，pass@4 55.33% |
| Result-only group-relative clipped PG | Exp1 | terminal denotation `{0,1}`；K=4 组内标准化 advantage；tokenwise PPO clip | advantage 对应 GRPO；clipped surrogate 对应 PPO；二值 reward 为项目定义 | equal-300：pass@1 41.33%，pass@4 58.33%；SFT2 为 42.33% / 55.67% |
| Replay-grounded Process-RL | Exp2–4、6–7、12，以及 Exp5/8–11/14 的 process 分支 | harness 重放产生 BackSlice、search reduction、局部错误/重复/no-op 惩罚；turn-level credit | process supervision 对应 Uesato/Lightman 的问题设定；BackSlice 在概念上对应 program slicing，具体 reward 为项目实现 | Exp2 pass@1/pass@4 37.67%/55.33%；Exp12 full-dev 761/1534 |
| Reference-free trajectory Rank | Exp5、Exp8–11、Exp13–14 | 同题 K=4 的 correct × incorrect pair；比较完整 response、action token 或 action-mean logprob | 与 RRHF/SimPO 的 response ranking 和平均 logprob 有关联；实现中没有 frozen reference subtraction，也不要求 same-prefix | Exp5 44.67%/56.67%；Exp8 42.67%/59.00%；Exp13 762/1534；Exp14 771/1534 |
| Exact-prefix Action-DPO | Exp15 | 同一精确 resident-state prefix 下，verified-correct suffix 与原 wrong suffix 构成偏好对；共同 prefix mask | pairwise reference-corrected loss 对应 DPO；prefix 构造、工具 token mask 与 verifier 数据筛选为项目实现 | 776/1534；相对 SFT2 为 +14 |
| Offline dense outcome shaping | Exp16–18 | 正确 clean legal action `+1`，错误 clean legal `-0.5`，确定性坏动作 `-2`；Exp17 另加 evidence/BackSlice `+0.5` | outcome/process shaping 与 rank loss 的项目组合，没有一篇论文与整套规则一一对应 | Exp16 764/1534；Exp17 760/1534；Exp18 764/1534 |
| Streaming multi-teacher OPD + repair DPO | Teacher-union60、Strict120 | 当前 student 每批真实 rollout；冻结 SFT2/Exp15 对精确 student token prefill，`clip(log p_T-log p_old)` 形成 action-token signal；失败轨迹另做四锚点 same-prefix repair DPO | on-policy distillation 对应 GKD/MiniLLM/MOPD；repair pair loss 对应 DPO；双教师路由、四锚点与 verifier 阈值为项目实现 | Union60 785/1534，vs SFT2 `p=0.073`；Strict120 765/1534，vs SFT2 `p=0.872` |

## 3. Exp0–Exp18 逐项记录

### 3.1 冻结训练配置

所有正式新实验均以 `Qwen2.5-Coder-7B-Instruct + SFT2 checkpoint-1682` 初始化；
SFT2 adapter SHA-256 为
`d880e2d7cc3203fdb0d11a7c188d8f607fd297b174eff23f741b6fe73cc3ce6e`。

| 实验组 | 训练数据/采样 | optimizer | 更新次数 | 其余固定配置 |
|---|---|---|---:|---|
| Exp1、Exp3–11 | 23 题；当前 policy 每题在线 K=4，共 92 条多步轨迹 | AdamW，LR `1e-6`，WD `0.01`，cosine，warmup `0.03` | 23 | PPO iterations `1`，clip `0.2`，KL `0`；T `0.7`，top-p `0.95` |
| Exp2 | 同 23 题、K=4；历史 Accelerate backend，`denotation-nonempty` admission | AdamW，LR `1e-6`；REINFORCE-style | 23 | 训练发生 3 次 gradient OOM；无 TRL PPO clip |
| Exp12–14 | 冻结 balanced mixed60 K=4 pool，共 240 条轨迹；训练时不重新 rollout | AdamW，LR `1e-6`，WD `0.01`，constant | 60 | PPO iterations `1`，clip `0.2`，KL `0` |
| Exp15 | 32 题、36 个 verified exact-prefix pair；question-balanced | AdamW，LR `1e-6`，1 epoch，grad clip `1` | 32 | DPO beta `0.1`；tool-token only；frozen SFT2 reference |
| Exp16–17 | 同一 frozen mixed60 K=4 pool，共 240 条轨迹 | AdamW，LR `1e-6`，WD `0.01`，constant | 60 | process coefficient `1`；Rank `lambda=0.5, beta=0.1`；clip `0.2`，KL `0` |
| Exp18 | frozen mixed120 K=4 pool，共 480 条轨迹；train seed `202` | AdamW，LR `1e-6`，WD `0.01`，constant | 120 | 与 Exp16 相同的 dense uniform + Rank；clip `0.2`，KL `0` |
| Union60 | 60 题，batch `2`；每批用更新后的 student 重新 rollout | AdamW，LR `1e-6`，WD `0.01`，grad clip `1` | 30 | seed `101`；advantage clip `5`；repair beta `0.1`、lambda `1` |
| Strict120 | 120 题，batch `2`；60 个 batch | AdamW，LR `1e-6`，WD `0.01`，grad clip `1` | 53 | seed `120`；7 个 batch 无可训练信号；每 group 最多 2 条 positive repair |

除 Exp15 外，表中含 rollout 的实验统一使用 max steps `30`、max new tokens `1024`、
max context `8192`、history turns `4`。Union60/Strict120 的 student 只在 run 开头从
SFT2 初始化一次，batch 之间不重置。

### 3.2 每个实验的 objective 与 mask

| 实验 | backend / objective | process / dense 分支 | Rank / DPO 分支 |
|---|---|---|---|
| Exp0 | 无训练，SFT2 对照 | — | — |
| Exp1 | TRL clipped PG | terminal `{0,1}` 的 K=4 group-relative advantage；全 response | 无 |
| Exp2 | Accelerate REINFORCE-style | BackSlice + search reduction + 局部惩罚；全 response | 无 |
| Exp3 | TRL PPO-clipped process | NoBackSlice；全 response | 无 |
| Exp4 | TRL PPO-clipped process | NoBackSlice；JSON action token only | 无 |
| Exp5 | TRL clipped process + Rank | NoBackSlice；全 response | full-response / sum / full trajectory |
| Exp6 | TRL PPO-clipped process | NoBackSlice；取消 positive-mass 归一化；全 response | 无 |
| Exp7 | TRL PPO-clipped process | NoBackSlice；error/repeat/no-op 惩罚 ×3；全 response | 无 |
| Exp8 | TRL clipped process + Rank | 与 Exp3 相同 | action-only score / sum / full trajectory update |
| Exp9 | TRL clipped process + Rank | 与 Exp3 相同 | action-only full score / conservative update |
| Exp10 | TRL clipped process + Rank | 与 Exp3 相同 | action-only conservative score+update / sum |
| Exp11 | TRL clipped process + Rank | 与 Exp3 相同 | 同 Exp10，但 mean-action score |
| Exp12 | offline PPO-clipped process | frozen mixed60 NoBackSlice；action-only | 无 |
| Exp13 | Rank-only | policy coefficient `0` | action-only / conservative / mean-action |
| Exp14 | offline clipped process + Rank | 与 Exp12 相同 | 与 Exp13 相同，`lambda=0.5` |
| Exp15 | standalone DPO | — | exact-prefix Action-DPO，tool-token only，beta `0.1` |
| Exp16 | offline clipped dense + Rank | uniform：correct legal `+1`，wrong legal `-0.5`，bad `-2`；全 response | dense-outcome / mean-action，`lambda=0.5` |
| Exp17 | offline clipped dense + Rank | Exp16 + evidence/BackSlice `+0.5`；全 response | 同 Exp16 |
| Exp18 | offline clipped dense + Rank | Exp16 扩至 120 题 | 同 Exp16 |
| Union60 | streaming OPD + DPO | action/terminal token MOPD；`clip(logp_T-logp_old, ±5)` | verified four-anchor repair DPO |
| Strict120 | streaming OPD + task-balanced DPO v2 | 与 Union60 相同 | all strong anchor groups，per-task/group normalization，最多 2 positive repairs/group |

### 3.3 Equal-300、K=4：Exp0–Exp11

| 实验 | 单变量改动 | 论文/方法归属 | pass@1 | pass@4 | 记录 |
|---|---|---|---:|---:|---|
| Exp0 | SFT2，无 RL | 对照 | 42.33% | 55.67% | 初始化 checkpoint-1682 |
| Exp1 | terminal `{0,1}` result-only | PPO/GRPO backend + 项目 reward | 41.33% | 58.33% | 相对 Exp0：pass@1 −1.00 pp，pass@4 +2.66 pp |
| Exp2 | BackSlice + search reduction + 局部惩罚，旧 backend | process supervision + program slicing 概念 + 项目实现 | 37.67% | 55.33% | 相对 Exp0：−4.66 pp / −0.34 pp |
| Exp3 | 去掉 BackSlice | 项目消融 | 38.67% | 52.67% | 相对 Exp2：+1.00 pp / −2.66 pp |
| Exp4 | process loss 只训练 JSON action token | 项目 token-mask 消融 | 40.33% | 56.00% | 相对 Exp3：+1.66 pp / +3.33 pp |
| Exp5 | 加 full-response trajectory Rank | RRHF 相关；非标准 DPO 数据与目标 | 44.67% | 56.67% | 相对 Exp3：+6.00 pp / +4.00 pp |
| Exp6 | 取消正信用轨迹归一化 | 项目消融 | 40.00% | 56.67% | 相对 Exp0：−2.33 pp / +1.00 pp |
| Exp7 | 三类确定性局部惩罚 ×3 | 项目消融 | 42.00% | 57.33% | 相对 Exp0：−0.33 pp / +1.66 pp |
| Exp8 | Rank 只看 action token | RRHF 相关 + 项目 mask | 42.67% | 59.00% | 相对 Exp0：+0.34 pp / +3.33 pp |
| Exp9 | 错误轨迹仅压明确坏动作，合法探索中性 | 项目 conservative routing | 41.00% | 55.33% | 相对 Exp0：−1.33 pp / −0.34 pp |
| Exp10 | score 与 gradient 使用同一 conservative mask | 项目一致性实现 | 44.33% | 56.33% | 相对 Exp9：+3.33 pp / +1.00 pp |
| Exp11 | action 内 token mean，再跨 action mean | 与 SimPO 平均 logprob 有关联；层级 action mean 为项目实现 | 未列入冻结排名 | 未列入冻结排名 | 已记录 grad norm 17.125→0.539；正式报告无最终 K=4 排名 |

Exp5/8–11 的 Rank loss 使用同题正负轨迹的 pairwise softplus。其实现没有 DPO 的冻结
reference 项，也不要求两条轨迹共享同一状态。本表因此将它与 Exp15 的 exact-prefix
Action-DPO 分开记录。

### 3.4 完整 BIRD-dev1534：Exp12–Exp18 与 Teacher-union

| 实验 | 方法 | 训练题数 | Correct | Accuracy | 相对 SFT2 |
|---|---|---:|---:|---:|---:|
| SFT2 | checkpoint-1682 | 0 | 762/1534 | 49.67% | 0 |
| Exp12 | fixed process-only | 60 | 761/1534 | 49.61% | −1 |
| Exp13 | fixed rank-only action-mean | 60 | 762/1534 | 49.67% | 0 |
| Exp14 | process + rank action-mean | 60 | 771/1534 | 50.26% | +9 |
| Exp15 | exact-prefix Action-DPO | 32（36 pairs） | 776/1534 | 50.59% | +14 |
| Exp16 | dense uniform full-response | 60 | 764/1534 | 49.80% | +2 |
| Exp17 | dense strategic / BackSlice | 60 | 760/1534 | 49.54% | −2 |
| Exp18 | dense uniform scale120，单 seed | 120 | 764/1534 | 49.80% | +2 |
| Teacher-union60 | SFT2+Exp15 OPD + repair DPO | 60 | 785/1534 | 51.17% | +23；exact paired `p=0.073` |
| Strict120 | task-balanced multi-repair v2 | 120 | 765/1534 | 49.87% | +3；exact paired `p=0.872` |

## 4. Teacher-union 的训练组成

Teacher-union 同时包含 on-policy distillation 与 preference optimization：

- student 在每个 batch 更新后生成下一批数据；
- frozen teacher 在精确 student token 上计算 logprob；
- dense signal 为 teacher 与 rollout policy 对该 token 的 logprob 差；
- repair 分支使用 DPO 式 chosen/rejected 相对目标；
- terminal verifier 用于记录最终正确性并筛选 repair pair；
- dense 分支不直接把 terminal `{0,1}` 写成每个 token 的奖励。

训练日志中可分别统计 student rollout、dense-teacher task、repair generation、verified
repair、strong DPO task、weak repair task、optimizer update 和 active policy token。

## 5. 论文来源与实现对应

| 论文 | 本项目采用的部分 | 本项目中另外实现的部分 |
|---|---|---|
| Williams, *Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning* (1992), DOI: [10.1007/BF00992696](https://doi.org/10.1007/BF00992696) | 早期 Accelerate REINFORCE policy-gradient 目标 | 多轮工具环境、reward、QLoRA 与 verifier |
| Schulman et al., *Proximal Policy Optimization Algorithms* (2017), [arXiv:1707.06347](https://arxiv.org/abs/1707.06347) | tokenwise clipped policy update | turn 构造、reward 与工具 credit |
| Shao et al., *DeepSeekMath* (2024), [arXiv:2402.03300](https://arxiv.org/abs/2402.03300) | group-relative advantage / GRPO backend | 表格工具任务、reward 与训练数据协议 |
| Uesato et al., *Solving Math Word Problems with Process- and Outcome-Based Feedback* (2022), [arXiv:2211.14275](https://arxiv.org/abs/2211.14275) | process 与 outcome supervision 的问题划分 | 本项目的执行重放信用，不使用人工推理步骤标签 |
| Lightman et al., *Let's Verify Step by Step* (2023), [arXiv:2305.20050](https://arxiv.org/abs/2305.20050) | 中间步骤监督的问题设定 | 本项目不训练 PRM，也不使用 PRM800K step labels |
| Weiser, *Program Slicing* (1984), DOI: [10.1109/TSE.1984.5010248](https://doi.org/10.1109/TSE.1984.5010248) | terminal evidence 依赖追踪概念 | 工具图、SQLite 重放、grounding edge 与 reward 权重 |
| Yuan et al., *RRHF* (2023), [arXiv:2304.05302](https://arxiv.org/abs/2304.05302) | response logprob 排序 | Exp5–11 的 pair 来源、mask 与 process-loss 组合 |
| Meng et al., *SimPO* (2024), [arXiv:2405.14734](https://arxiv.org/abs/2405.14734) | 平均 logprob 与长度归一化相关设计 | Exp11 的 token→action→trajectory 两级 mean |
| Rafailov et al., *Direct Preference Optimization* (2023), [arXiv:2305.18290](https://arxiv.org/abs/2305.18290) | Exp15 与 repair branch 的 reference-corrected pairwise objective | same-prefix 工具分支、verifier 与 repair 数据生成 |
| Agarwal et al., *On-Policy Distillation of Language Models / GKD* (2023), [arXiv:2306.13649](https://arxiv.org/abs/2306.13649) | student 自生成数据上的 teacher feedback | 工具 token、双教师路由与在线小批次更新 |
| Gu et al., *MiniLLM* (2023), [arXiv:2306.08543](https://arxiv.org/abs/2306.08543) | reverse-KL generative distillation | 同规模教师谱系与 verifier repair |
| Ma et al., *MOPD: Multi-Teacher On-Policy Distillation* (2026), [arXiv:2606.30406](https://arxiv.org/abs/2606.30406)；前身见 *MiMo-V2-Flash Technical Report*, [arXiv:2601.02780](https://arxiv.org/abs/2601.02780) | 多 teacher 对 student rollout 提供 dense token signal | SFT2/Exp15 双教师、K=4 任务路由、四锚点 repair DPO |

## 6. 实验与复现入口

- `docs/reports/rl/RL_EXPERIMENTS_EXP0_EXP11_COMPLETE_REPORT_20260801_ZH.md`
- `docs/current/rl_pipeline.md`
- `docs/reports/TWO_WEEK_EXPERIMENT_DATA_INDEX_20260723_20260805_ZH.md`
