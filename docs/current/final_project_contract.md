# 当前项目最终契约

## 项目记录要求

项目级时序索引位于根目录 [`project_records/`](../../project_records/)。任何新的 RL 讨论形成可执行想法、配置变更、实验启动/结束/中止或结果结论时，必须同步更新其中的实验记录或决策记录，并链接配置、immutable manifest 或报告。记录保持简洁；详细门禁和当前方案仍以本文件、`decision_register.md` 及 run manifest 为准。进行中、诊断和中止不得表述为已验证提升。

更新时间：2026-09-08（Asia/Shanghai）

这是项目当前唯一的收敛说明。`docs/current/` 中其他文档只补充这里定义的契约；历史分支、
失败诊断和旧 checkpoint 仍可审计，但不再是新实验入口。

## 1. 已确定的工具版本

训练、评测和 RL 全部固定为 **Atomic version26**：

| 项目 | 固定值 |
|---|---|
| runtime | `4cd47c957fc6ae791e76a10594c8cd22f4d3b6de` 导出树 |
| carrier | `think-json-v1`：非空 `<think>...</think>` + 一个 raw JSON action |
| prompt | rolling-full，SHA-256 `848598074e653648d52b58dfa1a54dd777beabae16cb91d83c4ba1a54b5d7316` |
| protocol | `version26`，hash `4da19387399bd3a5` |
| context | causal legal history 最近 4 轮 + 当前完整 resident state |
| limits | 每 episode 最多 30 semantic steps；每轮最多 2,048 new tokens |
| model | Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218` |
| terminal scorer | `bird-set`，结果必须引用 Harness 产生的 exact artifact |

Gold SQL、gold result 和任何完整 gold path 只在 Harness 内部用于执行兼容和 denotation
验证；不能进入模型 prompt、teacher request、trajectory 或 reward 解释。

## 2. 已确定的训练路线

```text
version26 causal SFT
  → 同 runtime/prompt/carrier 的 matched BIRD-dev greedy
  → A100 单卡 replicated trainer + 独立单卡 online vLLM RL
  → 在 table_rl/NewGNN 做 matched evaluation 与行为审计
```

冻结 SFT1 的小样本 RL 可行性验证模型是 `checkpoint-560`，BIRD-dev greedy 为 838/1534 =
54.63%；当前正式 RL 固定从 cumulative SFT `checkpoint-6380` 作为初始 adapter。两者
用途不同，报告中不得混称。

用户于 2026-09-04 明确要求新增一个非主线模型族对照：在 NewGNN 用四张 RTX 3090 训练
`Qwen/Qwen2.5-7B-Instruct` revision `a09a35458c702b33eeacc393d103063234e8bc28`，复用上述
Qwen3 SFT1 完全相同的 4,471-record model-visible training view（SHA-256
`c19742ec55d99980075e51a52e79285e4322f89a1d1991dea592748c279e9a14`）。Qwen2.5 token gate
为 4,471/4,471 full-prefix 保留、零 target 截断；训练保持 4-bit all-linear QLoRA、global
batch 16、两轮和 560 optimizer steps，只把模型及 chat template 改为 Qwen2.5/`qwen`。
四卡 smoke 已通过，formal run
`qwen25_7b_atomic_v26_sft1_same4471_4gpu_full_20260904_203529` 已以 exit 0 完成，global step
`560/560`，最终 train loss `0.5573`，final adapter SHA-256 为
`4ea1db0b17338e1c14c72af2254ddf9c3ca9241e57105080416f789a2156ac87`。该对照不替换 Qwen3
SFT anchor，也不得作为当前 RL 起点；完成结果和 matched evaluation 必须单独报告。

## 3. 唯一 RL 方案（当前最终方案）

当前新 RL 只允许使用：

- result-only `four-level` reward：
  - correct + no Harness error：`+1.5`
  - correct + Harness error：`+1.0`
  - incorrect + no Harness error：`-0.5`
  - incorrect + Harness error：`-1.0`
- advantage adapter：`correctness-primary-clean-secondary`，`clean_advantage_weight=0.25`。K=8 组内正确性决定 advantage 符号；clean/error 只按有界倍率调节幅度。terminal `four-level` 数值仍完整记录，但不直接作为组内 advantage 的符号来源。
- `saam-asymmetric-error` credit：共享 `(state, full action)` 在正确轨迹保留正优势、在
  错误轨迹置零；局部错误 action 使用 `-max(|A|, 1.0)`；timeout 视为 policy error，
  timeout action 同样使用局部负向惩罚。
- `trajectory_token_mean` policy reduction；reason span 和 tool span 各占 0.5 的加权梯度
  （`span_balance_alpha=0.5`）；不使用 tool-only reward、PCGrad 或 reasoning 作为事实。
- 当前运行 `kl_beta=0`；后续必须做独立匹配对照验证 KL 是否带来增益，再决定是否启用。
  对照固定同一 `checkpoint-6380`、screened cohort、decode/runtime、GPU 拓扑和更新预算，
  只改变 `kl_beta`，使用独立 output root；具体系数/调度在对照登记前保持未定。
- Qwen3-8B 全参数 actor，AdamW，learning rate `4e-7`，weight decay `0.1`，clip `0.2`，
  200 optimizer updates，ppo iterations `1`。
- 新 cohort 目标约 500 道题：每题 K=8，筛选后 terminal correct 数为 2–6；每 update 使用 30 道题。只使用已经完成筛选的 RL 数据，和 SFT 4k 轨迹严格隔离。
- A100 使用一张卡做 replicated trainer、另一张不同卡做 online vLLM；两张卡均只能从 GPU 0–3 中动态选择，显式传入其他 GPU 必须 fail-closed。

此前双卡 FSDP 的动态 token-budget gate 和 NCCL 对齐修复仅作为历史实现记录；它们不再是当前 A100 入口。当前单卡拓扑使用独立进程，不继承双卡 world-size 约束。

正式入口：

- `src/rl/configs/experiments/qwen3_8b_atomic_v26_saam_screened500_single_gpu.yaml`
- `src/rl/scenarios/rl_main/run_qwen3_8b_atomic_v26_saam_fourlevel_spanbalanced_700_single_gpu_a100.sh`

用户于 2026-09-06 另外注册三项仅用于审计/对照的 version26 降级 RL，不属于正式 RL 主线：
原版 SMC 在原始 `saam_gate60` cohort 上的概率选择审计；原版 SAAM+four-level+
span-balanced 在新版 `smc_balanced60` 60 题 cohort 上的批次方差对照；以及在前两项审计
后预注册的 SMC 优化诊断版本。三项均从 `checkpoint-6380` 启动，使用独立 output root，完成
后在 NewGNN 做 matched greedy 评测；它们不得改变正式主线 config、cohort 或 admission，
也不得在没有 decision register 更新和匹配评测的情况下宣称 accuracy promotion。

2026-09-08 用户批准追加 alpha=0.25 诊断：table_rl 相同新版60题、30题/update、K=8、
4 updates，从 checkpoint-6380 启动；训练完成后同机双卡评测 BIRD-dev1534 并释放 vLLM。
该对照固定上一轮 alpha=0.5 的远端训练实现及实际运行参数，只改变 span 权重，正式主线
仍保持 alpha=0.5。详情和实验身份见 `decision_register.md` 及
`docs/reports/rl/SAAM_ALPHA025_TABLE_RL_20260908.md`。

2026-09-07 审计修正：旧诊断 runtime 的 SMC 概率日志缓存按题组错误重置，导致历史 audit3
和原始 cohort 重跑每个 update 只落盘 1 个题组，而不是应有的 30 个。该问题不改变 SMC
优势选择逻辑，但历史 4 行日志不能作为完整概率证据；修复后必须通过 30 题组/update 的
覆盖率检查，并重新运行需要完整概率比较的 SMC 审计。

2026-09-07 修复后的原始 cohort 重跑已完成 120 组、960 个候选的概率字段与 rollout 身份/
选择一致性检查，完整概率审计门禁已通过；训练进程仍需完成最后 checkpoint 保存和资源释放。
SAAM 新版60题训练已完成，NewGNN 评测继续；资源释放后下一训练为新版 cohort 的完整 SMC
审计补跑，不等待评测。
中间结果及分母见 `docs/reports/rl/SCM_AUDIT_PROGRESS_20260907_ZH.md`，不构成效果准入。

2026-09-07 注册 Harness-conditioned SMDP/IQL 可行性诊断：第一阶段仅从完整 Atomic v26
rollout 构造 semantic-step transition 并审计数据是否包含同一精确 model-visible state
下的 action/outcome variation；transition 只保留 policy 可见 state、typed action 和
Harness four-level terminal reward，gold/ref 字段 fail-closed。IQL 的 `Q` 和 expectile `V`
是候选 critic，不是独立 reward model；在可识别性门禁通过前不训练 actor、不替换 SAAM/GRPO，
所有产物必须标记 `diagnostic_only`。实现与入口见
`src/rl/frameworks/trl/{mdp_critic,iql}.py` 和
`src/rl/scenarios/diagnostics/audit_harness_smdp_iql.py`。

A100 已有的 update-40 运行从 `checkpoint-6380` 启动，`global_step=40` 可审计，随后在
step49 因 OOM 退出；该运行使用 700 题、每 update 14 题，因此只作为历史诊断。同步盘点见
`docs/reports/rl/REMOTE_RL_SCREEN_INVENTORY_20260906_ZH.md`。新约 500 题 cohort 尚未注册，
完成前不启动正式 RL，也不宣称 accuracy promotion。

## 4. 服务器分工

| 服务器 | 硬件 | 允许的主要工作 |
|---|---|---|
| `a100` | 8 × A100 PCIe 40GB | RL 主实验；仅允许 GPU 0–3，单卡 trainer + 单卡 vLLM |
| `table_rl` | 2 × RTX 3090 24GB | matched evaluation、候选评测、行为诊断；A100 不可用时的降级 RL |
| `NewGNN` | 8 × RTX 3090 24GB | SFT、评测、数据准备、行为诊断；A100 不可用时的降级 RL |

GPU id、端口和具体输出目录由每次 launcher 动态记录；A100 launcher 的允许集合固定为
`{0,1,2,3}`，不要使用 4–7，也不要把当前快照当成永久资源分配。
详见 `server_resources.md`。

若 `a100` 不可用，目标是在 `table_rl` 或 `NewGNN` 使用两张独立 3090 运行同一 version26 RL
流程：一张 replicated trainer、一张 online vLLM。3090 运行从 1 row / 8,192 transition
tokens 起步，必要时显式启用 4-bit base 或 8-bit AdamW；使用独立 launcher、output root、
manifest 和 matched evaluation，不与 A100 结果直接合并。当前 A100 launcher 不自动接受
3090；3090 launcher 通过同等 preflight 和 live gate 后才能启动。无法获得两张空闲卡时必须
fail-closed。

## 5. 当前状态和停止线

- 2026-09-08：按用户要求先对 a100/table_rl 既有 K=8 rollout 做大样本静态审计。
  `distance_credit` 为未通过语义准入的诊断原型，不能接入 reward/actor 或自动筛题。
  统计与人工复核必须分开，跨 run 必须处理题号重编号，按题隔离未读保留集；不改变
  正式 SAAM/four-level/span-balanced 方案。报告见
  `docs/reports/rl/ROLLOUT_CORPUS_SEMANTIC_AUDIT_20260908_ZH.md`。

- 外部 DeepSeek 生成暂停；恢复前不发新 teacher batch。
- 新数据必须真实 model↔Harness 因果生成、fresh replay、结构和 no-leak 全部通过。
- checkpoint-relalg、Atomic v24/v39/v51/v54、Direct/Hybrid/iterative-SQL、projection/
  rewrite/delete、binary-only、execution-ladder、tool-only/fixed-span/PCGrad 均冻结为历史
  诊断，不得进入当前 SFT/RL 或默认配置。
- 旧代码按 `archive/code/legacy_migration.md` 迁入 `archive/`，先不删除 checkpoint、trajectory
  和 report；保留原始 identity 与复现说明。`src/` 不保留旧路线 compatibility stub，历史
  replay 必须显式恢复 archive 路径。

待用户确认的事项集中在 `decision_register.md`，确认后只更新该登记表和本契约，不再恢复一
套平行路线。
