# Qwen3-8B Atomic v26 SAAM：同一步骤 credit 震荡审计

日期：2026-08-28

## 1. 验证问题

SAAM v1 的第一研究目标不是立即提高 SQL 准确率，而是检验：

> 同一题、同一确定性执行前缀、同一完整工具动作是否能避免被终局结果同时强化和抑制，
> 并降低重复采样时 credit 符号反转。

严格保证只对 batch 内成立：若同一个 semantic state-action 在同一 K 组中同时出现在正确和
错误轨迹，SAAM 将其轨迹 advantage 全部置零。跨 optimizer batch 如果该动作一次只出现在
正确轨迹、另一次只出现在错误轨迹，v1 仍可能反转；当前没有历史记忆或跨 batch 平滑。

## 2. 指标

对同一个 semantic key (k=(x,s,a))，令正向与负向绝对 credit 质量为：

\[
P_k=\sum_j\max(c_j,0),\qquad
N_k=\sum_j\max(-c_j,0).
\]

定义 batch 内两侧直接冲突质量：

\[
C_k=2\min(P_k,N_k).
\]

它表示同一 semantic action 同时存在的正、负 credit 两侧。它是 credit 冲突代理，不宣称
不同 `<think>` 文本产生的参数梯度向量严格共线。

跨重复 batch 对同一 semantic key 汇总应用后的有符号 advantage，并统计相邻出现的符号
反转。由于冻结 rollout 日志没有保存 trainer 的逐 turn token 权重，该部分使用 raw
group-standardized GRPO advantage，只解释 credit 符号稳定性，不解释精确梯度范数。

## 3. 冻结 193×K8 池：batch 内冲突

生产同构 `trajectory_token_mean` 审计得到：

| 指标 | Vanilla | SAAM |
|---|---:|---:|
| mixed state-action 组 | 362 | 362 |
| ambiguous 正向系数质量 | 694.912 | 0（不应用） |
| ambiguous 负向系数质量 | 802.108 | 0（不应用） |
| two-sided direct conflict | 1,021.980 | **0** |
| ambiguous 多数方向残余 | 475.039 | 0（v1 一并删除） |

two-sided conflict 占 ambiguous 总质量 1,497.019 的 68.27%。因此 SAAM v1 的 batch 内
机制门通过：它确实消除了大量同一工具动作的直接正负 credit，而不只是形式上的重命名。

代价也被明确量化：v1 同时删除 475.039 的多数方向残余。后续“SAAM 内部正确性”可以研究
置信度阈值、历史证据或 soft mask 是否能够安全保留这部分信号；当前第一实验不引入这些
自由度。

## 4. 两次初始策略 K8 重采样：跨 batch 稳定性代理

使用既有的相同初始 SFT1 策略 first30 K8 两次重采样作为两个伪 policy step。两次运行使用
相同 seed，不能视为独立重复实验；其用途只是检查同一 semantic key 在采样扰动下的 credit
稳定性。

合并后共有：

- 60 个 question-batch；
- 455 条可审计轨迹；
- 18 个 mixed-reward batch；
- 53 个 mixed semantic-key occurrence，覆盖 222 个动作事件；
- 188 个在两次运行中重复出现的 semantic key。

结果：

| 指标 | Vanilla | SAAM |
|---|---:|---:|
| batch 内应用后的 direct conflict | 153.635 | **0** |
| 重复 key 的非零可比较转移 | 39 | 22 |
| 非零转移中的 sign flip | 13/39（33.33%） | 3/22（13.64%） |
| 所有188个重复-key转移中的 sign flip | 13/188（6.91%） | **3/188（1.60%）** |
| 涉及零 credit 的转移 | 149 | 166 |

按固定的188个重复-key分母，符号反转从13次降到3次，减少76.9%。这是支持“SAAM 降低
同一步骤反复强化/抑制”的初步证据。

但剩余3次反转也证明：batch-local SAAM 不能保证跨 batch 永不震荡。当一个 key 在某次 K8
中只有成功样本、另一次只有失败样本时，v1 不会将其识别为 mixed。这应当作为后续深入研究，
不能在第一阶段偷偷加入历史 replay 或 moving-average reward。

## 5. 当前结论

第一假设通过两个层次的验证：

1. **定义级保证**：同一 batch 内 mixed semantic state-action 的直接终局 credit 冲突严格为0；
2. **冻结重采样证据**：跨两次初始策略采样的重复-key符号反转由6.91%降到1.60%。

尚未验证：

1. 参数更新后同一动作 log-prob 的 checkpoint-to-checkpoint 反转是否下降；
2. 更稳定的 credit 是否提高 held-out `bird-set`；
3. 删除多数方向残余是否过于保守；
4. 对 `<think> + tool call` 整段置零是否优于只屏蔽 tool tokens。

因此下一步采用两遍 train60 配对 Gate：同一 SFT1 checkpoint-560、同一60个 BIRD-train
boundary 身份、K8、相同 optimizer/seed，两臂只改变 `trajectory` 与 `saam-strict`。两遍训练
让同一任务和 semantic key 有机会跨 optimizer step 重现；每一步用
`policy_global_step` 审计 batch 内冲突和跨 step sign flip。BIRD-dev 1534 只在训练结束后做
held-out evaluation，绝不进入训练或 cohort 选择。

实现：

- `src/rl/diagnostics/audit_saam_credit_oscillation.py`
- `src/rl/diagnostics/test_audit_saam_credit_oscillation.py`
- `src/rl/diagnostics/audit_saam_training_mask.py`

## 6. 在线 Gate60 启动记录

2026-08-28 17:23（Asia/Shanghai）在 `table_rl` 的两张 RTX 3090 上启动顺序配对队列，
远端 launcher PID 为 `1671644`。GPU0 专用于 QLoRA trainer，GPU1 专用于 TRL vLLM rollout；
两臂不能并行，因此先运行 trajectory control，再从原始 checkpoint-560 独立启动
`saam-strict`。

冻结训练 cohort：

- 只从前述 193 个 BIRD-train K8 mixed group 中选择；
- 选择规则为按 `SHA256(seed + NUL + task_id)` 升序取前60个，seed 为
  `qwen3-v26-saam-gate60-v1-20260828`；
- reward 只用于要求每个候选 K8 同时含0/1，不参与60题之间的排序；
- cohort SHA-256：`47e9d369bf6506d13b2432ac92bb971bf52912cb759c133846801f5411ad79d5`；
- manifest SHA-256：`5bd96f416edc6fc4845db867cb6171d9bb3b7a28a74d11fedc9cf4ae88f72758`；
- `dev1534_used=false`，没有 dev 题参与选择、训练或在线 reward。

两臂共同设置为 K8、30 prompts/update、4 optimizer updates、两次完整 train60 pass、
temperature 0.8、LR 8e-7、`trajectory_token_mean`、binary `bird-set`、KL=0、seed
20260828，并逐 update 保存 checkpoint。配置静态比较确认除实验名和
`credit_assignment=trajectory|saam-strict` 外完全相同。

远端独立 runtime：

- `/home/dengyan/tabular_rl_outputs/rl_runtime_qwen3_8b_v26_saam_gate60_20260828`；
- 运行根：`/home/dengyan/tabular_rl_outputs/qwen3_8b_atomic_v26_saam_gate60_20260828`；
- launcher：`src/rl/experiments/run_qwen3_8b_atomic_v26_saam_gate60_table_rl.sh`；
- cohort builder：`src/rl/diagnostics/build_saam_gate60_cohort.py`；
- 每臂结束后自动运行 `audit_saam_credit_oscillation.py`。

启动前两卡显存均为2 MiB、利用率0%；远端 source/data/config SHA、Python compile、shell
syntax 和本地84项相关测试全部通过。17:27 trainer 和 vLLM 均已加载，immutable
`run_manifest.json` 与 `implementation_lock.json` 已写入，trajectory arm 开始第1/4个
online update。最终训练和 held-out evaluation 结果尚未产生，本节只是实际启动与身份记录，
不得提前写成正向训练结论。

## 7. 运行预算调整：复用历史 Vanilla，当前只跑 SAAM

17:34 用户依据预计耗时决定不重复训练 Vanilla。历史 `mixed180` vanilla result-only GRPO
已满足相同 checkpoint-560、Atomic version26、K8、两遍、30 prompts/update、temperature
0.8、LR 8e-7、clip 0.2、KL=0 和 trajectory-token-mean 合同；其180个训练任务与本次
Gate60 重合58个。历史12 updates 实际约耗时15小时17分，线性折算本次4 updates 单臂约
5小时，两臂约10小时。

历史 Vanilla 的 fresh matched BIRD-dev1534 结果为：

- SFT1：832/1534 correct，1311 legal；
- Vanilla final：842/1534 correct，1306 legal；
- accuracy `+10/1534`（+0.6519pp，McNemar p=0.4876），legal `-5`；
- 结论为未通过晋级门，不是显著提升。

因此 PID `1671644` 的新 trajectory arm 在第一个 update 尚未完成、没有 checkpoint 时收到
TERM；trainer/vLLM 被 launcher 所有权清理，两卡回到2 MiB。其 immutable manifest 和部分
启动证据保留为 audit-only，不作为训练结果。随后以 mode=`saam`、新 PID `1697283` 从原始
checkpoint-560 独立启动 SAAM arm。

这个调整节省约5小时，但改变了结论强度：历史 Vanilla 多训练120题、8个 updates，虽然训练
合同高度相同且58/60任务重叠，仍不是“同60题、同4步、只改 credit_assignment”的严格因果
对照。当前实验可以验证 SAAM 的在线稳定性、训练可行性，并与已有 Vanilla 性能作探索性
比较；如果未来要声称 SAAM 准确率严格优于 Vanilla，仍需补跑原 matched trajectory arm。

## 8. 已完成 Vanilla 两遍训练的真实跨 update 震荡审计

为了检验“历史 0/1 GRPO 几乎没有净提升，是否可能来自共同前缀反复获得相反 credit”，又对
已经完成训练的历史 `mixed180` Vanilla 原始 `rollouts.jsonl` 做了逐 `policy_global_step`
审计。这个 run 包含12个真实 optimizer updates、每次30个 question-group、K8、两次完整
pass，因此同一训练题和 semantic state-action 会在参数更新后再次出现；它比第4节的两次
初始策略重采样更接近真正的训练震荡。

原始记录共有360个 question-batch、2,880条 rollout，其中2,845条进入优化，35条因非语义
失败被排除。审计到17,717个 semantic state-action occurrence，其中462个 occurrence 在同一
K8 batch 内同时出现在正确和错误轨迹中，共涉及1,895个动作事件。462个 mixed occurrence
按工具分布为：`describe_table` 230、`condition_filter` 96、`inspect_column` 92、
`read_subtable` 24、`group_aggregate` 7、`extreme` 5、`join` 5、`project` 2、
`scalar_compute` 1。也就是说，一半冲突来自常见的 describe 前缀，但另外232个冲突位于后续
工具，问题并不只等价于“第一步通常相同”。

在完全冻结这些历史 Vanilla rollout、不改变成功/失败标签和组优势的条件下，比较原
trajectory credit 与反事实 SAAM strict mask：

| 指标 | 历史 Vanilla credit | 同 rollout 反事实 SAAM |
|---|---:|---:|
| batch 内应用后的 direct conflict | 1,356.687 | **0** |
| 跨 update 重现的 semantic key | 1,185 | 1,185 |
| 非零可比较 credit 转移 | 486 | 316 |
| 非零转移中的 sign flip | 169/486（34.77%） | 62/316（19.62%） |
| 固定1,185个重现 key 中的 sign flip | 169/1,185（14.26%） | **62/1,185（5.23%）** |
| 涉及至少一次零 credit 的转移 | 699 | 869 |

按固定的1,185个重现-key分母，反复强化/抑制从169次降到62次，减少63.31%。这说明在一个
真实完成的两遍 Vanilla 训练中，所怀疑的 credit 符号震荡确实大量存在；SAAM 的规则在同一
批固定数据上能够删除其中约三分之二，并严格删除所有 batch 内可观测的双向 direct
conflict。剩余62次来自不同 batch 的单侧观测，符合 v1 只使用当前 K8 证据的设计边界。

同时，历史 Vanilla 并非没有发生参数或行为变化：12个 optimizer step 的 `grad_norm` 均
非零，约为0.034--0.044，252个 QLoRA 层完成同步；最终 dev1534 上899题的工具序列发生变化，
其中89题从错变对、79题从对变错，净结果只有+10题。这组“显著行为 churn、正负变化接近
抵消、净提升不显著”的现象与 credit 震荡假设一致，但单凭相关性不能把所有回归都归因于
SAAM 所定义的共同 state-action 冲突。

本节仍是**机制级反事实**，不是 SAAM 训练效果：固定使用 Vanilla 生成的 rollout 套用 mask，
无法生成“如果此前已经按 SAAM 更新，后续会采到什么轨迹”。另外，scalar advantage 的符号
冲突不等于两个样本的参数梯度向量严格相反；相同结构化工具调用可能带有不同 `<think>` 文本
和不同 token 条件，审计器明确不声称 `gradient_vector` 相消。最终因果链还需要当前在线
SAAM run 同时满足：实际 mask 有足够暴露、梯度健康、跨 update flip 下降，并在未参与训练的
dev1534 上减少 gain/regression 抵消或获得更好的净结果。

兼容历史 raw rollout schema 后，审计器相关12项测试通过，`git diff --check` 通过。结果文件：

- trajectory：`counterfactual_credit_oscillation_trajectory_20260828.json`，SHA-256
  `8f6c8fdebbf2b08fb6725fccbbb78a90f6db5a1e9913e5be0448f5dcc3ce150b`；
- SAAM：`counterfactual_credit_oscillation_saam_strict_20260828.json`，SHA-256
  `309bf2cf59cfa23fdee12f2cff85e1dde5eb6dba69701b521ad4b2feda73a96a`。

## 9. SAAM Gate60 在线训练完成

SAAM-only 队列于 2026-08-29 00:20（Asia/Shanghai）完成4/4个 optimizer updates；训练产物、
checkpoint-1--4、`final` 和完整 credit audit 均已生成，训练进程和 vLLM 已正常退出。全程
共120个 question-batch、960条 rollout，其中948条进入优化；没有轨迹被完全置零。

训练器记录的逐 update mask 为：

| update | rollout正确率 | 原始工具轮次 | 冲突组 | mask轮次 | mask response tokens | 初始/非初始 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.7875 | 1,789 | 49 | 190 | 33,241 | 114 / 76 |
| 2 | 0.6792 | 1,742 | 48 | 179 | 36,484 | 109 / 70 |
| 3 | 0.7750 | 1,834 | 39 | 162 | 30,642 | 93 / 69 |
| 4 | 0.7875 | 1,771 | 40 | 154 | 31,771 | 100 / 54 |
| **累计** | — | **7,136** | **176** | **685** | **132,138** | **416 / 269** |

因此累计 mask 比例为 `685/7,136 = 9.60%`，不是过低的无效触发，也没有达到删除大部分
训练信号的程度。每步删除的绝对 advantage 比例为15.37%--19.22%，`grad_norm` 为
0.0330--0.0395，252个 QLoRA 层持续同步，说明训练过程没有梯度死亡或全轨迹失信号。

完整120组的轨迹级 credit 审计得到：176个 batch-local ambiguous state-action groups，
685个实际 ambiguous events，Vanilla direct conflict mass 509.974，应用 SAAM 后为0。跨
update 方面共有398个重现-key转移，其中299次触及零 credit、29次发生可比较的符号翻转；
固定398个转移分母的 flip rate 为 `29/398 = 7.29%`，只看两端均非零的99次则为
`29/99 = 29.29%`。这再次表明 SAAM 主要是把证据不足的共享步骤置零，而不是让所有剩余
非零信号都自动变稳定。

训练阶段结论：SAAM 的在线实现和预期的 batch-local 去冲突机制均通过；它减少了可识别的
credit 推拉，同时保留了非零梯度和完整轨迹信号。**目前还没有 BIRD-dev1534 结果**，因此
不能据此声称 SAAM 提高了最终任务准确率。下一步必须使用 `final` 做冻结协议的 held-out
greedy 评测，并与同一 evaluator 下的 SFT1 结果逐题配对；1534 评测集仍未参与训练。

## 10. SAAM final fresh matched dev1534 评测启动

训练完成后，已启动一次严格的 fresh matched 评测，运行目录为：

`/home/dengyan/tabular_rl_outputs/evaluations/qwen3_8b_atomic_v26_saam_gate60_formal_20260829`

评测身份固定为 version26、Atomic、`think-json-v1`、rolling legal history 4、greedy@1、
temperature 0、top_p 1、`bird-set`，先评 SFT1，再评 SAAM `final`，1534题逐题配对。SAAM
final adapter SHA-256 为
`4f46fe9d08784b4e1f8dcdf1ccabe8d00b4743c1f9a31013a99d7c7d33249bba`；评测启动前训练 manifest、
adapter、模型、数据库和 version26 runtime 身份均已校验通过。评测进程当前状态为
`evaluating_sft1`，GPU0 已加载模型，尚未生成完整 `all.jsonl`，因此本节不提前报告正确数或
方法有效性结论。
