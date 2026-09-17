# 当前项目最终契约

## 项目记录要求

项目级时序索引位于根目录 [`project_records/`](../../project_records/)。任何新的 RL 讨论形成可执行想法、配置变更、实验启动/结束/中止或结果结论时，必须同步更新其中的实验记录或决策记录，并链接配置、immutable manifest 或报告。记录保持简洁；详细门禁和当前方案仍以本文件、`decision_register.md` 及 run manifest 为准。进行中、诊断和中止不得表述为已验证提升。

更新时间：2026-09-12（Asia/Shanghai）

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
| limits | 每 episode 最多 30 semantic steps；每轮默认最多 4,096 new tokens |
| model | Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218` |
| terminal scorer | `bird-set`，结果必须引用 Harness 产生的 exact artifact |

正式实验启动前必须先向用户列出并提交确认：配置及哈希、模型与 checkpoint、cohort/数据哈希、
protocol/prompt/runtime identity、reward/credit、rollout、optimizer、GPU 拓扑、输出目录和
评测计划。未获得用户明确确认不得启动；历史实验仍按其 immutable manifest 中记录的生成预算复现。

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

性能执行约束见[`rl_performance.md`](rl_performance.md)，所有新建/恢复RL启动前必读。
CUDA Graph的4B单组性能证据不代表8B已验证；核对launcher覆盖和隔离runtime后再使用。
正式保留actor old-policy scoring和replicated BF16/4-bit checkpointing；sampling-score
复用属于改变训练目标的独立ablation，不因追求速度自动进入正式方案。

2026-09-11模型规模对照更新：用户授权的Qwen3-4B累计SFT已完成4 epochs/6380步。
epoch3→epoch4使用table_rl两张3090分别加载完整模型（每卡TP=1/并发32），767+767题，
actionable-error-v1/2048；新epoch3已通过preflight并启动。其结果单独对照8B epoch4，
不替换当前RL起点。见[评测报告](../reports/sft/QWEN3_4B_CUMULATIVE_DP_EVAL_20260911.md)。

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
- 新 cohort 目标约 500 道题：每题 K=8，筛选后 terminal correct 数为 1–6（2026-09-16 起，
  取代原 2–6）；每 update 使用 30 道题。只使用已经完成筛选的 RL 数据，和 SFT 4k 轨迹严格隔离。

候选筛选的探索性扩展（2026-09-12）：用户允许另建一个正确数1–7的K=8候选库存，以增加
困难题和边界题的观察覆盖；该库存与当时的正式2–6 cohort分离，必须有独立版本、哈希和
admission记录，不能在没有新确认的情况下替换正式训练入口。

候选口径更新（2026-09-16）：3000题K=8筛选完成后，用户把训练候选区间改为1–6。合并后的
正式候选入口为`data/inventory/rl_training_candidates_1to6_current.json`（strict 1209、
expanded 1329、fresh 776），区间外的0与7/8观察保留为
`retained_non_candidate_observations.jsonl`记录，不进入训练。旧的2–6与1–7库存目录保留作
审计，重指向前的pointer快照为
`data/inventory/rl_training_candidates_20260911/pointer_snapshot_20260916.json`。

信号密度（2026-09-16 采纳，待实现）：group-relative advantage 对全对/全错组恒为零，A100
8B运行中0/8与8/8占47.6%、31.6%的transition因零优势被丢弃，且40个update后策略未移动。
因此采纳 adaptive-K rollout：首轮K=8，仅当组为8/8全对或0/8全错时追加一轮8条合并为K=16，
最多一轮，advantage在最终组内计算，reward/SAAM/span/clip不变；对照臂必须使用同一扩展规则。
预注册与判据见`docs/reports/rl/ADAPTIVE_K_ROLLOUT_PREREG_20260916_ZH.md`，实现前不得用于
正式cohort。同时记录一项待审计问题：现行“除以组内std”的归一化会放大稀有正确组（1/8组
单条正样本系数+2.65 vs 6/8组+0.58，1/2档题目占21%却承担37%优势质量）。
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
该对照已完成 BIRD-dev1534 评测，alpha=0.25 得到 935/1534；相同 control 为 933/1534，
alpha=0.5 为 923/1534。结果仍属于诊断证据，不自动改变正式主线。

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
step49 因 OOM 退出；该运行使用 700 题、每 update 14 题，因此只作为历史诊断。当前已登记
统一筛选候选池：strict 545 题、包含历史候选的 expanded 723 题；见
`docs/reports/rl/RL_TRAINING_CANDIDATE_POOL_20260911.md`。正式约500题 cohort 仍未冻结，
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

- 2026-09-10：用户批准的60题 legal-reason hybrid 属于独立诊断，细节以decision register为准。
  首次启动在TRL权重同步时404退出，未产生有效更新；r2已部署，补充清理修复后远端105项
  测试、CPU身份和GPU2实际CUDA核验通过。20:08后训练启动前双空卡检查exit75，未启动
  trainer/vLLM；等待两卡同时空闲，半小时接续任务已更新。在线权重同步和有效update尚未验证。
  20:42 heartbeat核实NewGNN八卡均有compute PID，本实验仍未训练，不占卡、不改变筛选。
  23:46–23:47最新进展：GPU2/7空闲后r2启动流程实际运行，TRL服务加载和HTTP门禁通过，
  状态starting_trainer；后续日志读取受平台审核容量故障阻塞，首个有效update仍未核验。
  不得重复启动或据此宣称训练稳定，先检查现有run；GPU4/6筛选保持原样。
  后续已核实：trainer exit0、4/4 updates及checkpoint-4完成，rollout/precision/权重同步
  审计通过；整体rollout正确率84.06%、合法率99.17%。当前正在用checkpoint-4在
  actionable-error-v1下评测BIRD-dev1534，dev结果尚未准入。
  评测已于2026-09-11完成，两个逻辑 shard 分别为455/767和447/767，合计902/1534；
  owned-process cleanup已完成并释放本任务vLLM。匹配的checkpoint-6380 SFT为928/1534；
  同题号49 gain / 75 loss，净-26（-1.69 pp，exact McNemar p=0.02437）。合法终止仅
  1380→1378且process errors减少41，故该`legal_reason_only_hybrid`诊断已否决，不准入
  当前主线，也不通过增加KL继续挽救；该结论不单独否决signed-binary或SAAM。
  使用signed-binary落实+1/-1，不更改历史binary及正式默认配置。
  checkpoint-6380对旧60题的K=8复筛已完成：27题全对、32题混合、1题全错，仅14题满足
  2–6，故旧cohort判为偏易。已从checkpoint-6380另一批完整848题筛选池中，仅按正确数
  冻结新60题（2至6各12题、与旧60题零重叠）。用户随后指定先运行Qwen3-4B baseline：
  从4B cumulative SFT epoch4 checkpoint-6380启动，signed-binary、trajectory/uniform、
  4updates、LR4e-7、KL0。2026-09-12训练已完成：checkpoint-4/960条rollout/precision和
  四次权重同步审计通过，最终adapter SHA-256为
  `e394e9d36684a85e2a078d572cd762985ef17eb7cff9f7a2fd2a90b5d5b9517c`；训练进程及online
  vLLM已释放。其table_rl GPU0/1双卡BIRD-dev1534评测使用actionable-error-v1、greedy、
  2048和767+767分片，对照同协议4B SFT epoch4的903/1534。
  新60题的2至6分档来自8B筛选，4B实际难度只以本次online rollout为准。checkpoint-4
  已完成BIRD-dev1534双卡独立评测，结果890/1534（58.018%）、1369合法；同协议4B SFT
  epoch4为903/1534（58.866%）、1372合法，逐题配对净-13题（79 gain、92 regression，
  McNemar p=0.3588）。
  见[`QWEN3_4B_VANILLA_GRPO_BALANCED60_20260911.md`](../reports/rl/QWEN3_4B_VANILLA_GRPO_BALANCED60_20260911.md)。
  用户随后授权独立的4B `SAAM + three-level-clean-weighted + span-balanced`诊断：原运行根为
  `/home/dengyan/tabular_rl_outputs/rl_runtime_qwen3_4b_saam_threelevel_spanbalanced60_table_rl_20260912`，
  reward为正确且完全合法1.25、正确但有Harness错误0.75、错误统一-1，使用
  `saam-asymmetric-error`、全response `span_balance_alpha=0.5`、uniform routing、KL=0。
  原运行在第一批240条rollout后的eager old-policy attention阶段OOM，未提交checkpoint；
  已建立独立的`..._r2_sdpa`根，从checkpoint-6380重跑，唯一执行变化是trainer改用SDPA；现已完成
  `global_step=4/4`、checkpoint-4、960条rollout和precision审计并释放训练资源。checkpoint-4已在
  table_rl GPU0/1完成`actionable-error-v1`、greedy、2048、BIRD-dev1534评测：897/1534、1358合法；
  对同协议4B SFT 903/1534、1372合法为77 gain / 83 regression，净-6题（-0.391 pp，exact McNemar
  p=0.6928）；对4B vanilla RL 890/1534、1369合法为88 gain / 81 regression，净+7题（+0.456 pp，
  exact McNemar p=0.6445），均未达到准入门禁。评测资源已释放，也不改变8B唯一RL主线。
  后续 correctness-only `binary + SAAM` 4B 对照沿用相同checkpoint/cohort、span alpha=0.5、
  K=8、30题/update和4 updates，完成checkpoint-4及actionable-error-v1 BIRD-dev1534评测：
  909/1534正确、1359合法。相对4B SFT为77 gain / 71 regression，净+6题（+0.391 pp，
  exact McNemar p=0.6812），合法性净-13；相对three-level为净+12（74/62，p=0.3456），
  相对vanilla为净+19（84/65，p=0.1401）。准确率差异均不显著，未通过能力准入；完整
  fresh replay仍是剩余审计门禁，评测owned资源已释放。
  随后用户授权的 `saam-first-error-capped` 4B 对照也已完成训练、评测和 fresh replay：训练根为
  `/home/dengyan/tabular_rl_outputs/overnight_first_error_capped_20260914_r3`，checkpoint-4 adapter
  SHA-256 为 `5730d15dd91500f096a5343ec15b054d776cde586ea1474f968dae48d99490f2`；评测根为
  `/home/dengyan/tabular_rl_outputs/evaluations/qwen3_4b_first_error_capped_saam60_checkpoint4_graph_actionable_20260914_r2`。
  双卡 767/767 分片已合并为 877/1534 正确（57.1708%）、1344/1534 合法（87.6141%），
  `fresh_runtime_replay_admission` gate 为 true 且 owned cleanup 完成；底层 fresh runtime audit 仍有
  112 个 step divergence 和 4 个 timeout timing divergence，fresh execution audit gate 为 false
  （9 个 replay exception、100 个 derived-legal disagreement，但 correctness disagreement 为 0），
  因此 admission 通过不等于 raw replay diagnostics 全部清零。五臂配对 provenance 确认1534个唯一
  example index、question/db_id/split零错配及 config/runtime/protocol/feedback/decode一致。相对
  correctness-only SAAM 为63 gain / 95 regression，净-32题（-2.086 pp，exact McNemar p=0.0134012）；
  相对4B SFT epoch4为64/90，净-26题（-1.6949 pp，p=0.0435999）；相对three-level SAAM净-20题，
  相对vanilla净-13题。该单次对照不支持推广 first-error-capped，不改变8B唯一RL主线。
  2026-09-14用户继续授权独立4B `saam-later-error-half`：首个确定性非timeout错误全惩罚，
  后续确定性错误半惩罚；其余保持原始checkpoint-6380、60题、binary result、4 updates及执行配置。
  此为预注册诊断；训练和matched BIRD-dev1534已完成（902/1534正确、1359/1534合法），但fresh runtime与execution replay两个gate均未通过：example 7332/sample 4发生timeout后连锁执行差异，execution replay有14条异常、100条derived-legal disagreement，correctness disagreement为0。该结果只能作为诊断，不能形成能力准入或迁移到8B；详见decision register同日条目。实现测试与CPU preflight已通过，GPU资源已释放。
  同期NewGNN上的8B signed vanilla-GRPO恢复已生成checkpoint-3（global_step=3/4、累计960条
  rollout），但最后update的eager backward再次OOM；原run及manifest保留，checkpoint-3审计
  完整。已建立`...resume_step3_sdpa`独立根并从checkpoint-3续跑，唯一执行变化是trainer
  改用SDPA；8B checkpoint-4的同协议`actionable-error-v1` BIRD-dev1534评测已完成：931/1534，
  对照checkpoint-6380 SFT的928/1534仅净+3题（66 gain / 63 regression，exact McNemar p=0.8603），
  不构成accuracy promotion，评测资源已释放；4B匹配评测也已完成，结果见上段，未达到能力准入。

- 后续新启动的评测统一使用 `actionable-error-v1`；`legacy` 仅保留为已完成历史审计。
  评测 manifest 必须显式记录 feedback identity，禁止把不同 feedback 版本的结果作为
  algorithm-only 对照。当前已在途的 legacy checkpoint-3 评测按原身份收尾，不中断。

- 2026-09-09 17:10核查：table_rl correctness-primary+SAAM/span0.5四次更新诊断训练已完成，
  legacy反馈评测checkpoint-1/2/4分别为924/915/917（分母1534），checkpoint-3已自动接续双卡评测。
  新反馈SFT的928/1534属于actionable-error-v1，不能当作同反馈的纯RL对照；不构成accuracy promotion。
  实际参数偏差、旧反馈/2048评测身份及运行路径见
  [`CORRECTNESS_PRIMARY_TABLE_RL_20260909.md`](../reports/rl/CORRECTNESS_PRIMARY_TABLE_RL_20260909.md)。

- 2026-09-08：用户批准 `actionable-error-v1` 反馈增强及 checkpoint-6380 的
  BIRD-dev1534 greedy 回归。保持 v26 action/system prompt/执行/评分不变，记录独立
  feedback 与 implementation identity；冻结 runtime 使用隔离 overlay，不能把本地
  重构差异混入。候选未通过配对评测前不替换正式 RL 默认；启动配置须先展示确认。
  用户已确认table_rl双3090、2048 tokens、历史完整baseline对照；首次服务启动因
  64位CUDA库路径缺失失败并清理，独立重试恢复已有进程环境，不改变软件版本或评测预算。

- 2026-09-08：按用户要求先对 a100/table_rl 既有 K=8 rollout 做大样本静态审计。
  `distance_credit` 为未通过语义准入的诊断原型，不能接入 reward/actor 或自动筛题。
  统计与人工复核必须分开，跨 run 必须处理题号重编号，按题隔离未读保留集；不改变
  正式 SAAM/four-level/span-balanced 方案。报告见
  `docs/reports/rl/ROLLOUT_CORPUS_SEMANTIC_AUDIT_20260908_ZH.md`。

- 外部 DeepSeek 因果数据生成仍暂停；2026-09-09 用户单独授权至多16次既有轨迹的
  事后离散过程评分API试验（不做程序化闭包，timeout为policy error）。这是诊断而非SFT
  生成或新RL，未通过分类/归因人工复核前不得扩大或进入actor；详见decision register。
- 新数据必须真实 model↔Harness 因果生成、fresh replay、结构和 no-leak 全部通过。
- checkpoint-relalg、Atomic v24/v39/v51/v54、Direct/Hybrid/iterative-SQL、projection/
  rewrite/delete、binary-only、execution-ladder、tool-only/fixed-span/PCGrad 均冻结为历史
  诊断，不得进入当前 SFT/RL 或默认配置。
- 旧代码按 `archive/code/legacy_migration.md` 迁入 `archive/`，先不删除 checkpoint、trajectory
  和 report；保留原始 identity 与复现说明。`src/` 不保留旧路线 compatibility stub，历史
  replay 必须显式恢复 archive 路径。

待用户确认的事项集中在 `decision_register.md`，确认后只更新该登记表和本契约，不再恢复一
套平行路线。
