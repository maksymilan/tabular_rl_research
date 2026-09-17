# 实验记录

## 2026-09-16

- **逐题 K=8 正确轨迹变化趋势（只读，无GPU）**：对"同一题在一轮训练内被重复采样"的臂做逐题配对趋势（每题恰好出现 2 次，K=8 对 K=8）。**per-question first→last**：909 SAAM60 3.500→3.383（Δ=−0.117，t=−0.62，提升/持平/下降=19/19/22）、cap1.0 3.167→3.500（Δ=+0.333，t=+1.83，23/24/13）、cap1.5 3.517→3.350（Δ=−0.167，t=−0.74，22/10/28）、SAAM 无 span（NewGNN）3.200→3.150（Δ=−0.050，t=−0.22，18/14/28）；**四个臂都没有净提升，Δ 都在 ±0.35/8（≈±4pp per-sample）以内且不显著**。churn（|Δ|≥1）占 60–83%，而"每题成功率的纯抽样噪声"预测约 **78.8%**（K=8→K=8 复测，p=0.25–0.75），因此观察到的波动幅度与"每题成功率不变、只是二项噪声"一致。**unpaired per-update 序列会误导**：909 的 u0→u1 从 2.80 跳到 4.20，但该臂 u0 抽到的 30 题本身更难（首现均值 2.800 vs u1 子集 4.200），按子队列配对的 Δ 分别是 0→2:0.000、0→3:−0.222、1→2:−0.444、1→3:+0.417，仍为 ~0。**180 题臂每题只被采样一次，逐题趋势不可识别**（这是 one-pass 设计的固有局限）。脚本 `/private/tmp/per_question_trend.py`（本地 + table_rl/NewGNN `/tmp`）。

- **SFT期筛选 vs RL在线：0/8 组的来源审计（只读，无GPU；含一次口径更正）**：用每题在 checkpoint-6380（SFT）的 K=8 筛选作 pre-RL 基线，与 RL 在线**按 `(update, question)` 的严格 K=8**配对（每条臂每组恰好 8 行）。结果：**180 题臂**在线 0/8 = 30/180 组，其 SFT 筛选为 2:12、3:9、4:7、5:1、6:1（28/30 在 SFT 时 ≤4/8），配对差值 **−0.300/8（t=−1.75）**；**60 题臂**在线 0/8 = 17/120 组，筛选 2:4、3:8、4:4、5:1，配对差值 **−0.558/8（t=−2.78）**；**SAAM 无 span 臂** 0/8 = 25/120 组、8/8 = 6 组。三条臂一致：0/8 组主要来自 **SFT 时就已经是边际难度（2–4/8）** 的题在该次 K=8 采样中落空，而不是"把会做的题练坏"；全臂均值存在 0.3–0.6/8 的小幅负移，需 fresh re-screen 或 dev 评测判定是否为真实退化。**此前记录的"+2.88/8、筛选波次不可比"已作废**：那是把 60 题臂跨 2 个 update 合并的 K=16 计数与 K=8 筛选相减造成的口径错误（用户指出硬件不可能决定模型性能，复核确认是分析 bug，非现象）。脚本 `group_audit.py`、`screen_vs_online_k8.py`（`/private/tmp` 本地 + table_rl `/tmp`）。

- **训练数据中的"全错组"数量（只读统计）**：候选池本身在筛选阶段不含 0/8（1–6 带排除了它们，落在 `retained_non_candidate_observations.jsonl`），但**在线重采样后每个 update 仍会出现全错组**：180 题臂 6 个 update 共 179 个组中 **29 个 0/8（16.2%）**、6 个 8/8；60 题（909）臂 120 组中 **17 个 0/8（14.2%）**、5 个 8/8；SAAM 无 span 臂 119 组中 **24 个 0/8（20.2%）**、6 个 8/8。对照组构成直方图见 decision register。这些组在 group-relative advantage 下恒为零信号，正是 adaptive-K 的目标对象。

- **adaptive-K 实现完成（shared layer，未启动 run）**：按用户指令把 adaptive sampling 上限设为 **32/题**（8→16→24→32 逐轮扩展，混合即停）。改动：`rollout.py` 新增 `RolloutSettings.adaptive_group_size_max` 与纯函数 `adaptive_extension_counts()`；原单轮 `collect()` 改名 `_collect_pass()`，新增多轮 `collect()`（只对全对/全错组按 initial K 补采样，per-prompt sample-index offset 防 id 冲突）；`run_transition_grpo.py` 新增 `--adaptive-group-size-max`（≥group-size 且整数倍）并写入 manifest；`experiment_config.py` 接受/校验 `rollout.adaptive_group_size_max`；`transition_grpo.py` 记录 `adaptive_k/*` 指标。校验：本地 compile、远端 helper/`RolloutSettings` 断言全过；新配置 `qwen3_4b_atomic_v26_correctness_only_saam180_adaptive_k_table_rl.yaml`（180 题、6 updates、adaptive 32）CPU preflight 通过（`rollout_settings.adaptive_group_size_max=32`、runtime/cohort identity pin 住）。隔离实现根：table_rl `qwen3_4b_saam_adaptive_k_20260917_r1/project`。**尚未启动 GPU run。**

- **训练侧 `rollout/correct_rate` 序列的分辨力审计（只读，无GPU）**：对三条已完成臂的 rollout 做"按题聚类的每 update 准确率"分析，回答"为什么训练曲线看不出学习趋势"（脚本 `/private/tmp/per_update_stats.py`，逻辑：按 update 分组→每题 8 条的正确比例→题级 sd/SE→对 update 序号做趋势 t 检验与 80% 功效 MDE）。结果：**① 180 题臂（6 updates、每 update 30 题）**：update 均值 0.510/0.496/0.554/0.464/0.500/0.508（整体 0.5055），题级 sd 0.28–0.36（每题正确比例 min 0.00 / max 1.00），单 update 聚类 SE **5.89pp**，update 间观测 sd 仅 **2.65pp**（低于噪声地板），趋势 −0.25pp/update（t=−0.33），**MDE ≈ 8.97pp**。**② SAAM 无 span 臂（4 updates）**：聚类 SE 5.98pp、观测 sd 7.48pp、趋势 +1.79pp/update（t=0.39，dof=2）、**MDE ≈ 28.5pp**。**③ 8B A100 40-updates 运行**：均值 0.481、sd 12.57pp、14 题/update 对应聚类 SE ≈9.35pp、趋势 +0.038pp/update（t=0.21）、**MDE ≈ 36.1pp**。结论：训练侧曲线的 MDE 为 9–36pp，而目标效应是 1–2pp，因此"看不到趋势"只能读作**不能检出**，不能读作"没有学习"；要测学习必须用固定探针集（或让同一批题跨 update 重复采样，当前 TRL sampler 在一遍内不重复，做不到），这正是 matched BIRD-dev1534 评测的职责。

- **两条 4B 训练臂均正常完成（无异常中断）**：① **180 题数据扩展臂**（table_rl）`trainer exit 0`，6/6 updates、`train_runtime=34,076s`（9h28m）、epoch 1.0、1440 rollouts（6×240）、checkpoint-1..6 + final、precision audit 全 FP32（33,030,144 trainable params / 1,008 moment tensors）、checkpoint-6 adapter SHA-256 `3a48c6b89a4a59cb25d830b752f6ea6c9f5f9e54442cd6e032337f12b965919f`；训练侧 `rollout/correct_rate` 逐步 0.458/0.450/0.513/0.433/0.417/0.463（均值≈0.455，无趋势），`nonzero_advantage_fraction` 0.66–0.92，SAAM 每 update 置零 52–113 个 transition。② **SAAM 无 span-balanced 臂**（NewGNN）同样 `trainer exit 0`，4/4 updates、`train_runtime=26,743s`（7h26m）、epoch 2.0、960 rollouts、checkpoint-4 adapter SHA-256 `8d65bded47291a63a091b2c53384fafac94e2d086f70dadd12078dcb5d92a293`，训练侧 correct_rate 0.3125/0.4875/0.3583/0.4292（无趋势）。两条的 `status` 均为 `trained_pending_audit`——这是 launcher 的**设计终态**（trainer 退出后自行停掉自己的 vLLM 并退出），不是异常；因此"GPU 空了、进程没了"是正常的资源释放。table_rl GPU0/1 与 NewGNN GPU6/7 均已释放（NewGNN 6/7 随后被其他用户作业占用，与本任务无关）。两条臂目前都**未做** checkpoint/effective-update 审计、fresh replay 与 matched 评测。

- **cap 影响量级离线分析（只读，无GPU）：cap 不动 reward，只削掉 20–32% 的训练信号，且优先削掉承重通道**：对 cap=1.0 那次的 858 条可训练轨迹 / 7415 个 transition 的**未裁剪**系数做分布与裁剪曲线分析（脚本 `archive/diagnostics/20260916_cap_scale_analysis.py`）。系数分布 `p50=0.531, p75=1.060, p90=1.913, p95=2.679, p99=4.464, max=13.13`，正负均值几乎相同（0.848 vs 0.847）。裁剪结果：**c=1.0 → 1471 transition / 569 轨迹（66.3%）被裁、削掉总质量 32.0%（正 16.0% / 负 15.9%）、error 通道只保留 55.3%**；c=1.5 → 848 / 417（48.6%）、削 19.9%（9.8/10.1）、error 通道保留 71.8%；c=2.0 → 12.9%；**c=3.0 → 仅 5.8%、error 通道保留 92.6%**；c=5.0 → 1.6%。按工具看被削最多的是关系动作 `project` 41%、`group_aggregate` 40%、`join_tables` 37%，最少的是 `read_subtable` 18%、`describe_table` 19%——**选择性由序列化 token 长度决定，而非语义重要性**（系数 = |A| × 该 turn token 占比 × transitions/trajectories）。结论：(1) cap 对 reward 零影响，是纯幅度衰减；(2) c=1.0/1.5 名义上"温和"，实际落在系数分布 p73/p80，属于重裁剪；(3) 它优先削掉"已验证错误通道"和"决定答案的关系动作"，与剂量-反应结论（削弱已验证错误惩罚会掉分）方向一致；(4) accuracy 在 20% 质量处即饱和（cap1.5 885 与 cap1.0 886，p=1.0），合法性却随 cap 变温和而改善（1340→1362，p=0.088），说明 accuracy 损失来自被切进的尾部通道而非合法性机制。

- **4B SAAM 数据扩展臂：180 题 cohort 构造完成并已启动训练（table_rl）**：为检验"更大训练数据量能否进一步提升"，在冻结候选库存上构造 180 题 cohort：**60 题沿用 2026-09-14 correctness-only SAAM 成功臂的原 cohort（逐字保留、排在最前）+ 120 题新候选**。新候选取自 `data/inventory/rl_training_candidates_1to6_20260916/fresh_candidate_tasks.jsonl`（fresh 视图，已排除 SFT training view 与旧 700 cohort），限定 **correct_count 2–6（与成功臂同带）、dataset=bird-sql（与 60 题同数据集，且 table_rl 无 SynSQL 库）**，按 `(db_id, question_sha256)` 排除与 60 题重叠的 31 题；每档配额用最大余额法分配（供给 37/25/21/27/30 → 选中 32/21/18/23/26），档内按 `sha256(seed+NUL+example_id)` 排序取前 N。产物：`data/rl_inputs/qwen3_v26_saam180_20260916/data_new180.table_rl.jsonl`（180 题，dataset 全为 bird-sql，example_index 0–179，db_path 指向 `/home/dengyan/tabular_rl_project/...`，180/180 库文件在 table_rl 上均存在），SHA-256 `d55ccccdf05b55cf597b1dba29e16040f5da218e95d45bae88305953a9416432`；选择清单 `manifest.table_rl.json`；构造入口 [`build_saam180_cohort.py`](../src/rl/scenarios/data/build_saam180_cohort.py)（含单测 `src/rl/tests/test_build_saam180_cohort.py`，4 passed）。训练臂：config [`qwen3_4b_atomic_v26_correctness_only_saam180_table_rl.yaml`](../src/rl/configs/experiments/qwen3_4b_atomic_v26_correctness_only_saam180_table_rl.yaml)，launcher [`run_qwen3_4b_saam180_table_rl.sh`](../src/rl/scenarios/diagnostics/run_qwen3_4b_saam180_table_rl.sh)，run root `/home/dengyan/tabular_rl_outputs/qwen3_4b_correctness_only_saam180_table_rl_20260916_r1`，GPU0 trainer / GPU1 vLLM（18382/51482），preflight 通过（records=180、cohort sha 一致）。manifest 与成功臂一致（seed 20260914、K=8、30 题/update、binary、`saam-asymmetric-error`、`span_balance_alpha=0.5`、LR 4e-7、error_penalty 1.0、protocol hash `4da19387399bd3a5`、initial adapter `0bc7b8b6…`），**并已核对实现差异**：Sep-15 project 与成功臂冻结快照相比只有 `state_action_ambiguity.py`/`transition_grpo.py` 的**纯新增**（first-error/later-error 变体与其指标），`saam-asymmetric-error` 路径行为不变。两处需要显式记录的差异：① updates 4→6（6×30=180，即 180 题一遍；成功臂是 60 题两遍），② 实现文件为纯新增差异。预计约 10–11 小时；完成后对 checkpoint-6 做同一协议 matched BIRD-dev1534 并与 909（SAAM60）/903（SFT）配对。

- **4B advantage-cap c=1.5 评测与配对分析完成（结论：与 c=1.0 无差异，仍不如不加 cap 的 SAAM）**：双卡评测 1534/1534 完成并 merge，paired 分析 `paired_analysis_cap15_vs_controls.json`（四臂：sft4b / saam4b / cap10 / cap15，均 BIRD-dev1534 greedy、actionable-error-v1、temperature 0、max_tokens 2048）。结果：**cap15 = 885/1534（57.69%），legal 1362（88.79%）**、mean_steps 7.165；对照 cap10 = 886（57.76%）/ legal 1340、saam4b（correctness-only SAAM，span .5）= 909（59.26%）/ 1359、sft4b = 903（58.87%）/ 1372。**cap15 vs cap10：净 -1（82 gain/83 regression，exact McNemar p=1.0），action sequence 变化 1178 题；合法净 +22（p=0.088）**——即 milder cap 只改善合法性、不改善准确率。**cap15 vs saam4b：净 -24（57/81），p=0.0498**；cap15 vs sft4b：净 -18（68/86），p=0.171，legal -10（p=0.453）；cap10 vs sft4b：-17（76/93），p=0.218，legal -32（p=0.0128）。据此 magnitude-cap 方向可收口：c=1.0 与 c=1.5 两档都未超过不加 cap 的 SAAM 臂，与静态审计"cap 是 variance control 而非 reward shaping、且同时削弱正负信号"的预测一致。该臂的 fresh replay / admission 门禁未跑，不得作为能力准入结论。

- **4B advantage-cap c=1.5 训练完成 + 双卡 matched 评测已启动（table_rl）**：cap=1.5 单臂在 table_rl 完成 `global_step=4/4`（960 条 rollout = 4×240），run root `/home/dengyan/tabular_rl_outputs/qwen3_4b_advantage_cap15_saam60_20260916_r1`，`status=trained_pending_audit`；checkpoint-4 adapter SHA-256 `5e5e94eb357075a2255bec55356ca79bc4db8ee2c695e178f6c449b39e62b270`。manifest 身份：seed 20260916、cohort SHA `1a6cb257…`（与 cap1/SAAM 同一 60 题）、K=8、30 prompts/update、binary、`saam-asymmetric-error`、`span_balance_alpha=0.5`、**`advantage_magnitude_cap=1.5`**、LR 4e-7、error_penalty 1.0、protocol hash `4da19387399bd3a5`、initial adapter `0bc7b8b6…`。随后在 table_rl GPU0/GPU1 启动双卡 data-parallel matched 评测：run dir `/home/dengyan/tabular_rl_outputs/evaluations/qwen3_4b_advantage_cap15_saam60_checkpoint4_actionable_20260916_r1`，端口 18450/18451，`example_index` 偶/奇分片 767+767，输入派生哈希 `636e096b…`（BIRD-dev 1534 冻结身份），temperature 0、max_tokens 2048、max_steps 30、n_samples 1，`error_feedback_version=actionable-error-v1`，与 cap1 / correctness-only SAAM / SFT 基线同协议。启动时遇到并解决了两个环境故障：① 共享 `~/.cache/vllm/torch_compile_cache` 里 9 月旧 artifact 损坏（`CompiledArtifact.load` 报 checksum mismatch），改为 per-run `VLLM_CACHE_ROOT`/`TORCHINDUCTOR_CACHE_ROOT`/`TRITON_CACHE_DIR` 后消失；② 清空 Triton cache 后重编 `cuda_utils.c` 需要 `TRITON_LIBCUDA_PATH=/home/dengyan/miniconda3/envs/trl-table/var/triton-libcuda`，否则 `ld` 找不到 `-lcuda`。评测进度约 13 题/分钟，预计约 2 小时完成，之后需 merge 两分片并做与 SFT(903)/correctness-only SAAM(909)/cap1.0(886) 的 paired McNemar。

- **180 题臂继续训练第二遍 epoch（NewGNN，用户要求，已启动）**：从 180 题臂自己的 checkpoint-6 续训到 12 updates（180 题 × 2 epochs），运行根 `qwen3_4b_saam180_epoch2_20260917_r1`（NewGNN GPU3 trainer / GPU7 vLLM，端口 18430/51530），起点为原始 1-epoch run 的 checkpoint-6（adapter SHA `3a48c6b8…`，trainer_state global_step=6、epoch=1.0）。
  - **resume 契约（实测确认，三条都满足）**：① checkpoint 必须是输出目录的直属子目录，且该目录已含上一轮 run 的 `run_manifest.json`+`implementation_lock.json`（`require_resume_base_model_identity`）→ 在新 root 里 staged 了 checkpoint-6、manifest、lock、implementation_source_snapshot、rollouts.jsonl（669MB），原始 1-epoch run root 未改动；② lock 绑定 `experiment_config_sha256`，所以必须用**逐字节相同**的 180 题 config（改配置会被 "implementation lock changed" 拒绝），扩展预算走 CLI `--optimizer-steps 12`（config 只提供默认值）；③ `initial_adapter_sha256` 必须保持 SFT adapter `0bc7b8b6…`。  - **实现身份**：resume 必须使用与 lock 完全一致的源码树；在 table_rl 上逐个比对后确认 Sep-15 project 快照 31/31 pinned 文件哈希一致（另外两个候选快照分别有 4 处不符被排除），已整份复制到 NewGNN 作为该 run 的 project。
  - 已在 NewGNN 通过 CPU-only preflight（base model identity、adapter、cohort sha、checkpoint 命名全部通过），manifest 确认 `resume_from_checkpoint=<新root>/train/checkpoint-6`、`optimizer_steps=12`、cohort `d55ccccd…`、K=8、30 题/update、`saam-asymmetric-error`、span 0.5、LR 4e-7、protocol hash `4da19387399bd3a5`；训练进度条显示 `0/12`。预计约 11h（→约次日 02:00）。
  - 训练结束后自动跑 checkpoint-12 的 matched BIRD-dev1534 评测（identity 与其它臂一致），输出 `evaluations/qwen3_4b_saam180_epoch2_checkpoint12_actionable_20260917_newgnn`；队列日志 `/home/dengyan/tabular_rl_outputs/saam180_epoch2_eval_after_train.log`。该臂与第一轮 896 的配对比较回答"同一批 180 题再过一遍是否有效"。
- **adaptive-K 与 carrier 格式兼容实现完成 + 固定60题/6updates 队列已启动（2026-09-17，用户要求）**：
  - **adaptive-K**：框架层已具备完整实现（`RolloutSettings.adaptive_group_size_max`、`adaptive_extension_counts()`、`TableAgentRolloutCollector.collect()` 的多轮扩展与 `last_adaptive_stats`、trainer 的 `--adaptive-group-size-max` 开关、mainfest 字段、`adaptive_k/*` 训练指标、`src/rl/tests/test_adaptive_k_rollout.py`）；本次确认可用性并纳入新臂。语义：首轮 K=8，仅当组为 8/8 或 0/8 时按 group-size 追加采样至上限（本臂 16），混合组不动，advantage 在合并后的组内计算。
  - **carrier 格式兼容（runtime-only，默认关）**：新增 `action_carrier.repair_action_carrier()` 与 `protocol.parse_assistant_strict_with_repair()`，只修复三种无歧义传输笔误：think 标签未闭合、action 被代码围栏包裹、action 对象前后有多余文本；修复后的文本必须重新通过同一套 strict carrier 与工具参数校验，工具/参数语义不变。旧 tool_call carrier、重复或空 think、缺 action、JSON 损坏仍是硬错误。运行时用 `--carrier-repair`（或 `CARRIER_REPAIR=1`）显式开启，逐 turn 记录 `carrier_repair{kind, original_error_code}`，manifest 记录 `carrier_repair` 布尔；默认关闭，所有历史臂与 matched 评测保持严格 v26 语义。新增单测 `src/rl/tests/test_carrier_repair.py`（6 passed）。
  - **已启动队列（table_rl GPU0/1，端口 18420/18422）**：arm A = 固定 60 题 + 6 updates（相对 909 臂只改 update 数，对消 180 题臂的数据/update 混淆），`adaptive_group_size_max=None`、`carrier_repair=False`；arm B = 同 arm A 再加 `--adaptive-group-size-max 16 --carrier-repair`（组合效应臂，非单变量）。两个臂训练后都用严格口径（无 adaptive-K、无 repair）做 matched BIRD-dev1534 评测，与其它五臂一致。arm A manifest 已核验：seed 20260914、cohort 1a6cb257、K=8、30 题/update、binary、saam-asymmetric-error、span 0.5、LR 4e-7、error_penalty 1.0、protocol hash 4da19387399bd3a5、`optimizer.steps=6`、`carrier_repair=False`。队列日志 `/home/dengyan/tabular_rl_outputs/saam60_updates6_queue_20260917/{queue.log,status}`；训练约 11h/臂，预计 arm A 训练 约 01:30、arm A 评测约 02:50、arm B 训练约 13:50、arm B 评测约 15:15（次日）。
  - 实现身份提醒：新快照基于 Sep-15 project 叠加当前 repo 的 `rollout.py`/`transition_grpo.py`（diff 核对为纯 adaptive-K 改动 + 移除 perf-trace 埋点）以及本次 repair 改动；arm A 与 909 臂的唯一机制差异仍是 update 数 4→6。
- **过夜流水线：GRPO+span 臂训练 + 两臂 matched 评测（NewGNN，已启动，用户安排）**：用户要求“先跑 GRPO+span-balanced 对比实验，然后把两个实验结果都评测完”。已按单条流水线启动（`src/rl/scenarios/diagnostics/run_overnight_saam_decomposition_newgnn.sh`）：① 训练 `qwen3_4b_v26_correctness_only_grpo_span60_newgnn`（`credit_assignment=trajectory` 无 SAAM + `span_balance_alpha=0.5`，其余与 correctness-only SAAM 臂逐字段相同：cohort `1a6cb257…`、seed 20260914、binary、K=8、30 题/update、LR 4e-7、SDPA、4-bit、gradient checkpointing、actor old-policy），运行根 `qwen3_4b_grpo_span_newgnn_20260916_r1`，launcher [`run_qwen3_4b_grpo_span_newgnn.sh`](../src/rl/scenarios/diagnostics/run_qwen3_4b_grpo_span_newgnn.sh)，配置 [`qwen3_4b_atomic_v26_correctness_only_grpo_span60_newgnn.yaml`](../src/rl/configs/experiments/qwen3_4b_atomic_v26_correctness_only_grpo_span60_newgnn.yaml)；② 该臂 checkpoint-4 的 matched BIRD-dev1534 greedy 评测（`ERROR_FEEDBACK_VERSION=actionable-error-v1`、2 卡 even/odd 分片、24 workers、T=0、max_tokens 2048、max_steps 30，与 909 baseline 同 identity）；③ 已完成臂 `qwen3_4b_saam_nospan_newgnn_20260916_r1` 的 checkpoint-4 同协议评测。阶段③在阶段①失败时仍会执行。GPU6/7 专用，端口 18381/51481（训练）、18382/18383（评测）；日志 `/home/dengyan/tabular_rl_outputs/overnight_saam_decomposition_20260916/{pipeline.log,status}`。估计训练约 7.4h（→约 07:10）、两次评测各约 1.5h（→约 10:10 完成）。该流水线把 890→909 的三变量混淆拆成 SAAM×span 的 2×2：已有 SAAM+span(909)、本次 GRPO+span、SAAM nospan。
- **过夜流水线结果与评测阻塞排查（2026-09-17）**：
  - **GRPO+span 评测完成（2026-09-17 11:09）**：`qwen3_4b_grpo_span60_checkpoint4_actionable_20260917_table_rl_r4`，1534/1534 覆盖、`error_feedback_version=actionable-error-v1`、`vllm.enforce_eager=0`、T=0/max_tokens 2048/max_steps 30、2 卡 even/odd 分片、24 workers，与 909 baseline 同 identity。结果 **886/1534（57.76%）**。配对比较（同一 example_index）：相对 correctness-only SAAM+span（909/1534，59.26%）= both 831、neither 570、gain 55、loss 78，**净 -23 题（-1.499pp），exact McNemar p=0.0560**。这是第一次单变量证据：在数据、seed、reward、budget、span 权重全部相同的前提下关掉 SAAM （`credit_assignment=trajectory`）会掉约 1.5pp（边界显著）。脚本 `archive/diagnostics/20260917_paired_eval_compare.py`。SAAM-nospan 评测仍在跑（约 681/767 分片），完成后补 span 侧配对。
  - **SAAM-nospan 评测完成（2026-09-17 12:31）与 2×2 配对结论**：`qwen3_4b_saam_nospan60_checkpoint4_actionable_20260917_table_rl_r3`，1534/1534，`actionable-error-v1`、`enforce_eager=0`、T=0/2048/30，与 909 baseline 同 identity，结果 **894/1534（58.28%）**。三方配对（同一 example_index，脚本 `archive/diagnostics/20260917_paired_eval_compare.py`）：
  - **180 题数据扩展臂（checkpoint-6）评测已启动（2026-09-17 12:41，用户要求）**：对 `qwen3_4b_correctness_only_saam180_table_rl_20260916_r1/train/checkpoint-6`（adapter SHA-256 `3a48c6b89a4a59cb25d830b752f6ea6c9f5f9e54442cd6e032337f12b965919f`）跑同协议 matched BIRD-dev1534 评测，输出目录 `evaluations/qwen3_4b_saam180_checkpoint6_actionable_20260917_table_rl`，GPU0/1、端口 18410/18411，identity 与其它三臂一致（`actionable-error-v1`、`enforce_eager=0`、T=0/2048/30、2 卡 even/odd、24 workers、输入 SHA `8bf5a8bf…`）。使用新增的可复用单臂评测入口 [`run_single_matched_eval_table_rl.sh`](../src/rl/scenarios/diagnostics/run_single_matched_eval_table_rl.sh)（内置 PYTHONPATH、TRITON_LIBCUDA_PATH、fresh 目录与 GPU 空闲等待），并挂了 [`compare_saam180_after_eval_table_rl.sh`](../src/rl/scenarios/diagnostics/compare_saam180_after_eval_table_rl.sh)在评测完成后自动输出与 SAAM60+span(909)、GRPO+span(886)、SAAM-nospan(894) 的配对比较。预计约 80 分钟（→约 14:05）。**注意这不是单变量对比**：180 题臂相对 60 题臂同时改了两件事——cohort 180 vs 60、`optimizer.steps` 6 vs 4（180 题一遍 vs 60 题两遍）；结论只能读作"数据扩展+更多 update 的联合效应"。
  - **180 题数据扩展臂评测完成（2026-09-17 14:13）：无提升**：checkpoint-6（adapter `3a48c6b8…`）matched BIRD-dev1534 = **896/1534（58.41%）**，identity 与其它臂一致。配对比较（脚本 `archive/diagnostics/20260917_paired_eval_compare.py`，输出见 `/home/dengyan/tabular_rl_outputs/single_matched_evals_20260917/saam180_paired_comparison.txt`）：
    - vs SAAM60+span（909）：gain 60 / loss 73，**净 -13 题 = -0.847pp，p=0.2981**（数据扩展+6 updates 没有带来提升）；
    - vs GRPO+span（886）：gain 75 / loss 65，净 +10 题（+0.652pp，p=0.4470）；
    - vs SAAM-nospan（894）：gain 82 / loss 80，净 +2 题（+0.130pp，p=0.9374）——与去掉 span 的 894 完全不可区分。
    注意该臂非单变量（cohort 180 vs 60 与 `optimizer.steps` 6 vs 4 同时改变），只能读作联合效应。
  - **当前五臂同协议总览（BIRD-dev1534 greedy、`actionable-error-v1`）**：SAAM60+span **909**（59.26%）、4B SFT **903**（58.87%）、SAAM180+span **896**（58.41%）、SAAM60-nospan **894**（58.28%）、GRPO60+span **886**（57.76%）、vanilla signed-binary(60,nospan,历史) **890**。全部落在 886–909（1.5pp）区间内：单变量证据是 SAAM ≈ +1.5pp（p=0.056，边界）、span ≈ +1.0pp（p=0.245，不显著）；把 cohort 从 60 扩到 180（并 4→6 updates）没有提升（-13，p=0.30）。仍不得宣称任何臂相对 SFT 有可靠提升。
    - 去掉 SAAM（保留 span）：909 → 886，gain 55 / loss 78，**净 -23 题 = -1.499pp，exact McNemar p=0.0560**（边界）；
    - 去掉 span（保留 SAAM）：909 → 894，gain 65 / loss 80，**净 -15 题 = -0.978pp，p=0.2449**（不显著）；
    - GRPO+span vs SAAM-nospan：886 vs 894，gain 78 / loss 70，净 +8 题（+0.522pp，p=0.5652）——两者在统计上不可区分。
    结论：这次单变量实验首次给出 SAAM 有正贡献的证据（约 +1.5pp，边界显著），span-balanced 同向但更弱（约 +1.0pp，不显著），    两者大致可加地构成 909 这个点；所有差值都落在本设计（60 题 / 4 updates，MDE≈2.3pp）的分辨极限附近，    不能据此宣称任一组件"显著提升"。第四格 vanilla signed-binary+nospan 历史为 890，与本次 GRPO+span(886)/SAAM-nospan(894) 同量级，    说明 reward profile 从 signed-binary 改为 binary 本身也没有可测增益。
  - **训练两个臂都成功**：GRPO+span（NewGNN，`qwen3_4b_grpo_span_newgnn_20260916_r1`，07:42 完成，960/960 rollout，checkpoint-4 adapter SHA-256 `8375ef938f7d917c7f831dc3a452965d423786674f8e37b70fa8299de4d8cdbe`，训练侧 `correct_rate` 0.2958/0.5333/0.4792/0.3917，`saam/*` 全 0 证明 SAAM 确实关闭，`trainable_token_fraction`≈0.003 证明 span 权重生效；第2个 update `grad_norm`=0.83 明显高于其他三步的 ~0.045）；SAAM-nospan（`qwen3_4b_saam_nospan_newgnn_20260916_r1`，前一晚 23:19 完成）。
  - **评测阶段连续三次失败，共三层原因**：① 我的流水线调用评测控制器时没有导出 `PYTHONPATH=<project>/src`，controller 直接 `ModuleNotFoundError: No module named 'rl'`（07:42 两次秒退）；② 评测 launcher `run_qwen3_8b_v26_checkpoint_dataparallel_table_rl.sh` **从未导出 `TRITON_LIBCUDA_PATH`**：triton 运行期要 gcc 编译 `cuda_utils.c` shim 并 `-lcuda` 链接，而本机驱动只提供 `libcuda.so.1`（无链接可见的 `libcuda.so`），于是先用缓存里损坏的 shim 报 `RuntimeError: Bytes object is corrupted, checksum does not match`、后在重建时报 `CalledProcessError: gcc ... -lcuda`；训练路径经 `start_vllm_server.sh` 设置了该变量所以从未触发。反证：同环境下 `torch.compile` 正常、`triton_backend()` 正常，且同样的 vLLM 命令加上 `TRITON_LIBCUDA_PATH=/home/dengyan/miniconda3/envs/trl-table/var/triton-libcuda` 后完整捕获 CUDA graph 并 `Application startup complete`（health OK）；③ launcher 的 fail-closed 保护在输出目录已存在时直接拒绝（`results directory already exists; resume is forbidden`），使前两次失败的目录让后续重试秒退。另：NewGNN 8 张卡整晚被其他用户占满，评测改到 table_rl（两卡空闲，且 909 baseline 也是在该机产出，identity 更一致）。
  - **修复**：给评测 launcher 增加 `PYTHON_ENV`/`TRITON_LIBCUDA_PATH` 推导与 shim 存在性检查（[`run_qwen3_8b_v26_checkpoint_dataparallel_table_rl.sh`](../src/rl/scenarios/evaluation/run_qwen3_8b_v26_checkpoint_dataparallel_table_rl.sh)）；评测链改用全新输出目录 `..._table_rl_r4`（grpo_span）与 `..._table_rl_r3`（saam_nospan），端口 18406/18407、18408/18409，并顺序启动两个 shard server。09:46 起 grpo_span 评测正常运行（vLLM ready，正在生成），随后自动跑 saam_nospan。


- **4B "SAAM 但不使用 span-balanced" 对照（NewGNN，已启动，用户要求）**：在新机 NewGNN 上用完全相同的数据与其余全部配置，只去掉 span 加权，观察训练后的能力变化。运行根 `/home/dengyan/tabular_rl_outputs/qwen3_4b_saam_nospan_newgnn_20260916_r1`（`project/` 为实现快照，`data_new60.jsonl` SHA-256 `1a6cb257081da58ce66e09ab4e0783f296e3fdc4aa5674da178ae4c86756b4ca`）；配置 [`qwen3_4b_atomic_v26_correctness_only_saam60_nospan_newgnn.yaml`](../src/rl/configs/experiments/qwen3_4b_atomic_v26_correctness_only_saam60_nospan_newgnn.yaml)，launcher [`run_qwen3_4b_saam_nospan_newgnn.sh`](../src/rl/scenarios/diagnostics/run_qwen3_4b_saam_nospan_newgnn.sh)。资源：GPU7=online vLLM（端口 18380/51480）、GPU6=trainer，只使用这两张空闲卡，未触碰其他用户进程；launcher 自带 gpu_idle 门与锁文件，preflight 已通过并写出 `logs/preflight.json`。与 table_rl correctness-only SAAM 臂逐字段一致（seed 20260914、K=8、30 prompts/update、binary、`saam-asymmetric-error`、`error_penalty=1.0`、LR 4e-7、clip 0.2、SDPA、4-bit、gradient checkpointing、actor old-policy、protocol hash `4da19387399bd3a5`、student prompt SHA、initial adapter SHA `0bc7b8b6…`），**唯一机制差异是 `span_balance_alpha` 0.5→None**；另有 `save_steps` 10→1（仅影响 checkpoint 落盘频率与逐 update 评测，不影响训练数学）。4 updates，按同规模 table_rl 臂估计约 7 小时；完成后对 checkpoint-4 做 matched BIRD-dev1534 greedy 评测，与 correctness-only SAAM（909）与 4B SFT（903）配对比较。
  - **完成更新（2026-09-16 23:19 CST）**：训练 4/4 updates 正常结束（`train_runtime=26,742.7s≈7.43h`），状态 `trained_pending_audit`，960/960 条 rollout 落盘，`checkpoint-4` 与 `final` 的 `adapter_model.safetensors` SHA-256 相同（`8d65bded47291a63a091b2c53384fafac94e2d086f70dadd12078dcb5d92a293`），`training_precision.json` 显示 504 个可训练张量与 1008 个 optimizer moment 全为 FP32，GPU6/7 已释放。训练侧指标（nospan vs table_rl span=0.5）：`rollout/correct_rate` 0.3125/0.4875/0.3583/0.4292（均值 0.397）vs 0.350/0.525/0.4167/0.4292（均值 0.430），差异在 30 题/update 的采样噪声内；`entropy` 0.377–0.408 vs 0.255–0.264；`rollout/trainable_token_fraction` **1.0 vs 0.0028**（span 模式下每个 turn 的总权重被归一化为 1.0 并在 reason/tool 两段间按 α 分配，非 span 模式按 token 数加权，故该指标不是"被训练 token 比例"而是权重归一化口径）；SAAM 指标同量级（ambiguous groups 33–38 vs 40–50、removed mass 4.9–9.4% vs 6.0–8.0%）。**能力变化尚未测量**：checkpoint-4 的 matched BIRD-dev1534 greedy 评测未执行，需在 GPU6/7 空闲时补做并与 correctness-only SAAM 909 / 4B SFT 903 配对比较。

- **3000题K=8筛选完成 + 候选区间改为1–6（数据筛选，完成；cohort未冻结）**：NewGNN GPU4/GPU6 两个1500题shard分别在 `2026-09-15 23:49` 和 `2026-09-16 09:01`（CST）结束，3000/3000题、每题8条轨迹，shard pass@8 为 0.698/0.677；本任务vLLM已停止，GPU4/6已释放。正确数直方图 `0:937, 1:168, 2:102, 3:83, 4:87, 5:109, 6:142, 7:250, 8:1122`。用户决定后续训练候选区间由2–6改为 **1–6**，区间外（0、7/8）不作训练但显式保留记录。合并后库存为 strict 1209 / expanded 1329 / fresh 776，入口 [`rl_training_candidates_1to6_current.json`](../data/inventory/rl_training_candidates_1to6_current.json)（并重指向正式 `rl_training_candidates_current.json`），区间外观察保留在 `retained_non_candidate_observations.jsonl`（3667条）。报告见 [`RL_TRAINING_CANDIDATE_POOL_1TO6_20260916.md`](../docs/reports/rl/RL_TRAINING_CANDIDATE_POOL_1TO6_20260916.md)，决策见 [`decisions.md`](decisions.md)。仍未冻结cohort，不得宣称accuracy提升。

- **8B历史评测补录（只读核查，此前未入库）**：A100 700题/14题每update运行 `qwen3_8b_atomic_v26_saam_fourlevel_batch14_700_single_gpu_a100_20260904_save10_keepall_r1` 在 table_rl 上有 matched BIRD-dev1534 greedy 评测：checkpoint-10 为 `925/1534`、checkpoint-30 为 `920/1534`，对照8B `checkpoint-6380` SFT `928/1534`，即 -3 / -8 题，与“性能几乎没变”一致。评测根为 `table_rl:/home/dengyan/tabular_rl_outputs/evaluations/qwen3_8b_atomic_v26_checkpoint10_dataparallel_table_rl_20260904_r3` 与 `..._checkpoint30_dataparallel_table_rl_20260905_r1`；`evaluation_config.json` 的 `source_checkpoint` 指向同一次 A100 运行的 `checkpoint-30`。身份提醒：该批RL评测为 legacy 反馈、SFT 928 为 actionable-error-v1，属跨反馈比较；checkpoint-40 的评测产物未在三台机器 `evaluations/` 命名中找到。

- **A100 8B 运行训练侧诊断（只读，本地产物）**：对 `data/inventory/a100/update40/` 的 40 个 update 日志与 5600 条 rollout 汇总做统计。cohort 组构成 `0:164, 1:69, 2:80, 3:45, 4:38, 5:47, 6:41, 7:47, 8:169`，**0/8与8/8合计333/700=47.6%恒零优势**；累计 `zero_advantage_transitions_dropped` 11047/34964=**31.6%**；`trainable_token_fraction` 均值 **7.7%**；`rollout/correct_rate` 均值0.481、单步sd0.126，前5步0.580→后5步0.536；entropy 0.253→0.249、`grad_norm` 均值0.043。结论：该run在训练侧没有移动，约束是信号密度而非credit规则。

- **优势权重按组档位的离线审计 + 自一致性离线审计（只读，无GPU）**：① 按 n=8 标准归一化，1/8组单条正样本系数+2.65、6/8组+0.58；按A100组构成，1/2档题目占21%却承担37%的优势质量，属需要重视的权重偏移。② 在3000题K=8语料（temperature 0.8、只读、未读gold）上：单样本正确率 55.0%、多数簇 57.2%、pass@8 oracle 68.8%，每题平均只有2.04个不同答案；1–6带内单样本43.7%→多数簇49.9%（+6.2pp）。该口径与greedy评测不可直接比较，需补同题greedy基线。脚本归档于 `archive/diagnostics/20260916_self_consistency_audit.py` 与 `archive/diagnostics/20260916_screen3000_trajectory_stats.py`。

- **D：终止承诺与分岔点审计（只读，无GPU，验证完成）**：对3000题K=8语料（24000条轨迹、未读gold）做两项测量。① **步级分岔点稀缺**：一对一错配对中56.2%在第1个可执行动作就不同，仅4.8%共享≥3步；只有222/941=23.6%的混合题存在共享≥3步的一对一错配对；把派生表句柄归一化后分布不变。同时99.3%的题里8条轨迹共享第一个**工具**（工具序列0.3%的配对在第1步就换工具），说明分歧在参数层而非计划层。② **承诺缺口真实**：同一批候选上单样本55.00%、多数投票57.23%、干净轨迹优先56.63%、clean_modal（干净子集内投票）**58.20%**、oracle上界68.77%（2063/3000）；"无Harness error且合法终止"是强无gold信号（全语料64.5% vs 22.1%；混合题内60.7% vs 38.5%）。93.5%的题至少有一条干净轨迹、3.7%恰好只有一条。剩余测量项：同题greedy基线（当前55%为T=0.8的sample 0）。报告 [`COMMITMENT_AND_BRANCH_POINT_AUDIT_20260916_ZH.md`](../docs/reports/rl/COMMITMENT_AND_BRANCH_POINT_AUDIT_20260916_ZH.md)，脚本 `archive/diagnostics/20260916_branch_point_audit.py`。

- **RL策略改进三合一审计（只读，无GPU，完成）**：对 4B `correctness-only binary + SAAM` run（`rl_runtime_qwen3_4b_correctness_only_saam60_table_rl_20260913_graph_r1`，960轨迹/4 updates）用其 frozen 实现重建 transition 系数。① **credit落点**：负系数质量分布 earlier_action 70.0% / last_relational 14.1% / terminal 13.5% / plan 2.3%；按答案 lineage 链拆分，合法但错轨迹的负质量仅 **41.5% 在链上、58.5% 在链外**（带error的错轨迹 on-chain 仅20.2%）。② **lineage状态覆盖**：错误turn中21.9%处于同组正确轨迹到过的状态（12.0%动作相同=SAAM屏蔽目标，**9.9%动作不同**=未使用的决定性分岔），77.9%无对比。③ **归一化**：组内std→常数除数后各档质量占比几乎不变（1档13.3%→10.5%），总质量570.8→252.6，属全局缩放，**弱杠杆，不单独开臂**；4.6×偏斜来自均值中心化本身。报告 [`RL_IMPROVEMENT_AUDIT_20260916_ZH.md`](../docs/reports/rl/RL_IMPROVEMENT_AUDIT_20260916_ZH.md)，摘要 `docs/reports/rl/rl_improvement_audit_20260916/summary.json`，脚本 `archive/diagnostics/20260916_rl_improvement_audit.py`。

- **分岔点审计（同 run，只读，追加）**：对 941 对"错误轨迹 vs 同组正确轨迹"按 lineage 身份逐 depth 定位第一处分岔：depth 0 占 39.9%、depth 1 占 45.8%、depth 2 占 9.1%、≥3 占 5.2%；**只有 14.7% 的首处分岔落在错误轨迹的答案推导链上**（正确轨迹链上 15.0%），分岔 turn 恰为错误侧 Harness error 的仅 0.2%。由此**证伪**"链外负 credit 是信号稀释"的解释：决定对错的分岔本来就在链外（主要是 depth 0–1 的早期探查）。负 credit 来源拆分：轨迹级链外 55.9%、轨迹级链上 37.0%、局部 error 惩罚（必然链外）7.1%。状态对比供给：状态匹配且正确侧动作不同 402 turn/4 updates，其中严格"正确侧≥2条一致"137 turn。

- **位置分布审计（同 run，只读，追加；修正上一条的 turn 数口径）**：按 lineage 状态（含前序动作与观测）分 depth 统计错误 turn。**SAAM 同动作屏蔽的 353 turn 中 85% 在 depth 0–1**（depth0 217、depth1 83），工具以 `describe_table` 223（63%）、`inspect_column` 45、`condition_filter` 45 为主，位置中位数 0.00；**对比候选（同状态不同动作、且未被 SAAM 屏蔽）283 turn**（严格 g=2 为 95），携带 269.94 负质量（11.7%），其中 depth0 107、depth1 97、depth2 47、depth3 22，62–92% 落在 depth 0–1；严格子集以 `describe_table` 57、`plan` 15、`condition_filter` 13 为主，宽松子集以 `condition_filter` 59、`describe_table` 46 为主。`no_match` 2272 turn 携带 1878.7 负质量（81.6%），位置中位数 0.60，工具以 `condition_filter` 527、`answer_from_context` 306、`group_aggregate` 208 为主。原因：lineage 状态包含前序动作与观测，depth d 匹配要求前缀完全一致，故状态级机制**机械地只作用于前 2 步**。修正口径：此前"不同动作 402 turn / 严格 137 turn"把已被 SAAM 置零（质量为 0）的 turn 一并计入。

- **状态级对比惩罚抽样复核（同 run，只读，结论：默认方案不执行）**：全量分类 283 个 contrast 候选的动作关系 + 抽读 80 条（严格 40/宽松 40，覆盖 27/26 道题）。`describe_set_variation`（`describe_table` 表格集合子集/超集）81（28.6%）、`projection_or_limit_only`（只差 `return_columns`/`limit`）20（7.1%）、`material_change` 182（64.3%）；分支后续仍被使用仅 80（28.3%）、位于错误轨迹答案链上仅 44（15.5%），其余 102 是丢弃的死分支。抽样实例见决策记录（ex=850/206 纯投影差异；ex=7557/1851 策略差异且分支弃用；ex=379 AND/OR 语义错误但分支弃用）。**严格护栏 g=2 更差**：严格组 62%（25/40）为表格集合差异 vs 宽松组 15%。修正定义后供给约 20 turn/update（≈总负质量 3%），只能作微调。抽样文件 `docs/reports/rl/rl_improvement_audit_20260916/contrast_samples.jsonl`。

- **4B advantage-magnitude-cap c=1.5 对照（已注册，待启动）**：固定 checkpoint-6380、同一 60 题 cohort、binary result-only、SAAM asymmetric-error、span alpha=0.5、KL=0、训练 4 updates；唯一变量为 post-reduction symmetric advantage cap `c=1.5`。配置为 [`qwen3_4b_atomic_v26_advantage_cap15_saam60_table_rl.yaml`](../src/rl/configs/experiments/qwen3_4b_atomic_v26_advantage_cap15_saam60_table_rl.yaml)，启动脚本为 [`run_qwen3_4b_advantage_cap15_table_rl.sh`](../src/rl/scenarios/diagnostics/run_qwen3_4b_advantage_cap15_table_rl.sh)。完成后必须做 exact cap-hit 审计、fresh replay 和 matched BIRD-dev1534。

## 2026-09-16

- **4B advantage cap post-reduction 审计（CPU-only，完成）**：对 cap=1 run 与历史 correctness-only SAAM 无 cap run 逐条重建实际 post-reduction 系数，统计 cap=1/1.5/2.0，并输出逐轨迹 action/tool-output 清单；不读取 gold SQL、不生成、不占 GPU。cap=1 实际命中569/858可训练轨迹、1471 transitions；cap=1.5命中417/858、848 transitions。292条正向命中轨迹全部最终 correct，261条无 Harness error；其中127条含 exploratory positive hit。详细证据见 [`CAP_POST_REDUCTION_AUDIT_20260916.md`](../reports/rl/CAP_POST_REDUCTION_AUDIT_20260916.md)。

## 2026-09-15

- **8B table_rl 性能 gate（取消，未进入 GPU）**：因把用户要求错误理解为加速验证，曾准备 checkpoint-6380 + gate60 的 8B 单 update 性能 gate；只运行到 CPU identity preflight，未启动 vLLM/trainer、未占用 GPU、无 checkpoint/结果。该草案不计入实验结果。

- **4B later-error-quarter 机制验证（已注册，待启动）**：在 frozen 960 条 rollout 的 alpha=.25/.5/.75 离线审计后，注册同一4B checkpoint-6380/60题 cohort 的4-update单臂；唯一机制变化是后续 deterministic Harness error 使用 `-0.25*max(|A|,1)`，首错和 timeout 规则保持不变。完成后执行同题 matched BIRD-dev1534 与 fresh replay，未通过门禁不推广。

- **启动状态**：已通过远端 preflight，GPU 实验已在 table_rl 启动。运行根为 `/home/dengyan/tabular_rl_outputs/qwen3_4b_later_error_quarter_saam60_20260915_r1`；GPU0 trainer、GPU1 vLLM 均已进入工作态，当前仍在第一个 update 的 rollout/优化阶段，尚未生成 checkpoint-1。

- **后续错误重复审计（只读）**：当前 quarter run 尚未写出 `rollouts.jsonl`，因此先审计最近完成的同 cohort 4B later-error-half 960 条轨迹。29 条轨迹含至少两个 Harness error；35 个后续错误中，28 个（80.0%）与前一个错误类型相同，但按“相同 attempted tool+arguments”严格定义只有 2/25（8.0%）重复。按轨迹计，23/29（79.3%）出现同类型重复，2/29（6.9%）出现严格相同动作重复。该结果说明后续错误大多是同一错误类别的连续链条，但很少是模型逐字重复同一个动作；机制应区分 error-type cascade 与 exact action repetition。

- **组构成与 LOO 影响审计（只读）**：按 `example_index + policy_global_step` 形成 120 个 K=8 组（不是把四次 update 的同题合并）。当前已完成 later-error-half rollout 中，6/120（5.0%）全对、20/120（16.7%）全错、94/120（78.3%）混合；正确数为7的单错组12个、正确数为1的单对组21个，共33条 minority trajectory。当前 quarter 运行尚无 rollout；这些数字不代表 quarter 结果，也不否定此前已完成的 2–6 筛选，它们是新 rollout 的实际组结果。

- **timeout-barrier replay audit v1 完成（CPU-only，promotion gate仍关闭）**：对 fresh replay 唯一 outcome mismatch 的 `rl_7332_sample_4` 单独做 5/10/15/30 秒重放，10 秒重复三次；同一 recorded model outputs 在目标复测中均恢复为 recorded `wrong_answer+legal`，说明先前 timeout→join cascade 是 timing-sensitive，而非稳定 Harness 语义差异。5 条已标记 timeout 的轨迹也完成 10 秒三次重复，正确性/合法性均稳定。版本化诊断报告将首个 fresh timeout 后缀标为不可重放、保留原始 outcome/credit：`timeout_barrier_replay_audit.json`，diagnostic gate=true、strict all-exact gate=false、promotion gate=false。该结果支持先做 barrier-aware 离线 credit 审计；暂不启动新的长RL。

- **4B later-error-half timeout敏感性验证（已登记，CPU-only）**：针对 fresh replay 唯一 outcome mismatch 的 `example_index=7332 / rl_7332_sample_4`，固定 recorded model outputs、任务、Atomic v26 runtime、rolling-legal-history、30 steps，仅改变 replay timeout 为 5/10/15/30 秒，并在10秒设置重复三次。记录每个 turn 的 parse、Harness error、tool output digest、最终 correctness/legal/failure_type；不读取或输出 gold SQL，不启动GPU。判据：若10秒重复结果不稳定或提高timeout后恢复 recorded outcome，归因为timeout timing sensitivity并修订replay admission；若各timeout仍稳定复现同一状态差异，则转为Harness deterministic replay bug，暂停新的RL训练。输出根为远端 later-error-half `train/timeout_sensitivity_7332_audit.json`。

## 2026-09-14

- **4B `saam-later-error-half` fresh replay 收尾（未通过 admission）**：匹配评测保持 **902/1534 正确（58.8005%）、1359/1534 合法（88.5919%）**，1534题全覆盖；训练本身已完成4/4 updates。fresh runtime replay 覆盖960条，102条 generation-truncated prefix精确、852条普通记录精确、无 parse divergence/重放异常，但 example 7332 的一条普通轨迹在第9步 timeout 后发生连锁执行差异，由记录的 `wrong_answer+legal` 变为 fresh `execution_error+illegal`，另有2个tool-output divergence，因此 gate=false。fresh execution replay 960条全计入、14条 replay exception、100条 derived-legal disagreement，但 correctness disagreement=0、action-count disagreement=0，gate=false。该结果只作为诊断证据，不进入能力准入或8B主线推广；后续若继续，应先修复/隔离该 timeout-sensitive Harness 重放稳定性，再注册 matched credit 对照，不重复启动当前4B half run。详见远端 `train/fresh_runtime_replay_audit.json`、`fresh_execution_replay_audit.json`、`fresh_runtime_replay_admission.json`。

- **4B `saam-later-error-half` 状态补充（训练、评测、fresh replay完成；admission未通过）**：训练根
  `/home/dengyan/tabular_rl_outputs/qwen3_4b_later_error_half_saam60_20260914_r1`
  已完成 `4/4` updates 并生成 `checkpoint-4`；四个 rollout correct rate 为
  `0.408333/0.429167/0.370833/0.437500`。第一次评测因控制器缺少远端 Python 包路径退出，
  第二次因共享 vLLM 编译 artifact checksum 损坏退出；两次均保留且未改动训练结果。第三次
  评测在独立根
  `/home/dengyan/tabular_rl_outputs/evaluations/qwen3_4b_later_error_half_saam60_checkpoint4_actionable_20260914_r3`
  使用同一模型、checkpoint、cohort、protocol、feedback 和 decode，已完成双卡分片；合并为
  902/1534正确、1359/1534合法，1534题全覆盖，owned cleanup完成。相对同题 correctness-only
  SAAM为60/67（净-7，p=0.5946），相对 first-error-capped为84/59（净+25，p=0.0444）；
  fresh replay 随后已完成但 `fresh_exact_runtime_replay_gate_v1` 未通过：960 条任务身份和协议
  均一致，853 条普通记录中 852 条逐步一致，102 条截断前缀一致，5 条 timeout 仅 1 条稳定；
  共109个 step divergence、1条 outcome mismatch，未有 parse divergence；随后 admission gate 仍为
  false，execution audit记录100个 derived-legal disagreement和14个 replay exception，但
  correctness disagreement为0。timeout-barrier audit 已确认 example 7332 的 cascade 是 timing-
  sensitive，diagnostic gate通过但 strict/promotion gate仍关闭。该限制不改变902/1534评测数字，
  但在 barrier-aware credit 审计完成前不能作完整可复现或能力准入结论。

- **4B `saam-later-error-half`（训练、评测完成；fresh replay admission未通过）**：固定后续确定性错误半惩罚，其他训练设置沿用已完成
  first-error-capped对照；配置为
  `src/rl/configs/experiments/qwen3_4b_atomic_v26_later_error_half_saam60_table_rl.yaml`。
  新根 `/home/dengyan/tabular_rl_outputs/qwen3_4b_later_error_half_saam60_20260914_r1`，
  训练后自动评测根 `/home/dengyan/tabular_rl_outputs/evaluations/qwen3_4b_later_error_half_saam60_checkpoint4_actionable_20260914_r1`。
  复制原始frozen runtime和实际eval controller，保留原输出；81项本地/17项远端测试、CPU preflight及
  handler与离线alpha=.5一致性审计通过。第三次受控启动已进入 GPU 生命周期：GPU0 trainer、GPU1
  vLLM 均已加载并完成 `4/4` 更新，训练约7小时14分48秒；此前两次审批超时及两次评测基础设施失败
  均不计入模型结果。fresh replay/admission已完成但未通过；当前不启动新GPU实验。

- **4B first-error-capped SAAM 训练、评测与 replay admission 完成（未准入）**：对 frozen
  correctness-only SAAM 的 960 条 rollout 做 gold-free 静态重评分。A 当前 binary+SAAM
  的正确轨迹负向 coefficient mass 为 104.14；B 首个非 timeout deterministic Harness
  error 封顶后为 95.08（下降 8.7%），正确负向 transition 68→61，封顶 30 个后续 error
  transition（其中 7 个来自最终正确轨迹），错误轨迹正向 credit 未增加。静态 gate 通过后，
  已在 `table_rl` 启动独立 B arm，训练根为
  `/home/dengyan/tabular_rl_outputs/overnight_first_error_capped_20260914_r3`；同一
  checkpoint-6380、60题 cohort、K=8、4 updates、4-bit/SDPA、CUDA Graph，4/4 updates
  完成并生成 checkpoint-4（训练约6小时39分37秒）。匹配评测根
  `/home/dengyan/tabular_rl_outputs/evaluations/qwen3_4b_first_error_capped_saam60_checkpoint4_graph_actionable_20260914_r2`
  的双卡767/767分片已合并为877/1534正确（57.1708%）、1344/1534合法（87.6141%）；
  adapter SHA-256为`5730d15dd91500f096a5343ec15b054d776cde586ea1474f968dae48d99490f2`。
  provenance确认五臂1534个唯一example index、question/db_id/split零错配及身份一致，
  `fresh_runtime_replay_admission` gate为true（`train/fresh_runtime_replay_admission.json`），但底层
  fresh runtime audit仍有112个step divergence和4个timeout timing divergence，fresh execution audit
  gate为false（9个replay exception、100个derived-legal disagreement，correctness disagreement为0）；
  admission通过不等于raw replay diagnostics全部清零，owned cleanup完成。
  相对 correctness-only SAAM 为63/95（净-32，-2.086 pp，p=0.0134012），相对4B SFT epoch4
  为64/90（净-26，-1.6949 pp，p=0.0435999），相对three-level SAAM净-20题，相对vanilla净-13题；
  单次结果显示准确率下降，不支持推广该 credit 规则，也不改变8B主线。r1/r2为已停止的实现/启动诊断根。

- **后续 Harness error 软衰减离线审计（完成、无 GPU）**：同一 frozen 960 条 rollout、7,389 个
  transition 上扫描 alpha=0.25/0.5/0.75。A 的正确轨迹负向 coefficient mass 为104.1416（68个
  transition），硬截断 alpha=0 为95.0796（61个，30个后续错误清零，其中7个最终正确）；
  alpha=.25/.5/.75 分别为97.3451/99.6106/101.8761，三者均保留68个正确负向 transition，错误
  轨迹正向 transition均为0；后续错误共30个（7个最终正确、23个最终错误）。结果仅说明离线
  credit 端点差异，不构成准确率结论。结果根：
  `/home/dengyan/tabular_rl_outputs/overnight_first_error_capped_20260914_r3/train/saam_soft_attenuation_audit_20260914_final.json`。
  当前建议优先 alpha=.5 单独4B/table_rl验证，先核对 raw replay divergence，不与8B规模验证并行。

- **first-error replay divergence 拆解（完成、只读）**：112 个 step divergence 可由 108 条外部调度边界
  前缀（107 generation_length + 1 context_overflow）和 4 条 timeout timing divergence 完整解释；没有
  tool-output、parse、task identity 或 correctness outcome mismatch。6 条 timeout-sensitive 轨迹的
  outcome/parse 均保持稳定，fresh runtime admission gate 通过。底层 execution replay 的 9 条 exception
  是重放原始 malformed/error action 时重新抛出的 `ProtocolError`/`ToolExecutionTimeoutError`，另有
  100 条 derived-legal disagreement（52 条 wrong_answer、48 条最终 outcome 正确）；action-count 与
  correctness disagreement 均为 0。因此不能宣称 raw execution replay 全清，但当前差异集中在记录边界、
  原始错误动作和 derived-legal 重算层，没有证据表明训练 credit 的轨迹级正确性被改写。

- **下一步验证（预注册、未启动）**：先不启动新的长 RL。针对后续 deterministic Harness error 做
  alpha=0.25/0.5/0.75 的离线软衰减重评分，保留首错和 timeout 规则，并同步核对 replay divergence
  边界、错误/恢复后缀及正确轨迹负向 credit。只有某一 alpha 相对 correctness-only SAAM 不增加
  错误轨迹正向 credit 且不牺牲正确率/合法率，才在同一 checkpoint/cohort/4B table_rl 条件下注册
  独立单臂训练；随后执行 matched BIRD-dev1534 和 fresh replay。

- **4B correctness-only SAAM BIRD-dev1534 评测完成（actionable-error-v1）**：checkpoint-4
  的 r4 根
  `/home/dengyan/tabular_rl_outputs/evaluations/qwen3_4b_correctness_only_saam60_checkpoint4_graph_actionable_20260914_r4`
  已完成双卡 767/767 分片，合并结果为 **909/1534 正确（59.2568%）**、**1359/1534 合法
  （88.592%）**，1534 个 example index 全覆盖；GPU 和 vLLM owned processes 已清理。前两个
  启动失败根与 r3 `legacy` 部分结果均保留为失败/诊断证据，不计入准确率。
  同题配对相对 4B SFT epoch4 为 77 gain / 71 regression（净 +6，exact McNemar
  p=0.6812），相对 three-level SAAM 为 74/62（净 +12，p=0.3456），相对 vanilla 为
  84/65（净 +19，p=0.1401）。合法性相对 SFT 为 62 gain / 75 regression（净 -13，
  p=0.3052），相对 three-level 为 70/69（净 +1，p=1.0），相对 vanilla 为 66/76
  （净 -10，p=0.4502）。配对产物为 `results/merged/paired_comparison.json`；准确率增量均
  不显著，且相对 SFT 的合法性下降，因此尚未达到能力准入门禁。四臂均为1534个唯一
  example index，question/db_id/split 零错配；哈希证据见同目录
  `paired_comparison_provenance.json`。

- **4B correctness-only SAAM 训练完成，评测完成但仍待 fresh replay**：新 binary+SAAM 对照已完成
  `global_step=4/4`，生成 960 episodes、训练前后合计 6,996 transitions（每 update
  1,193–1,667 条保留 transition，零优势丢弃417–506条）。`train_runtime=26,093.77s`
  （约7小时14分53秒），GPU0/1 owned resources 已释放；checkpoint-4、manifest、
  implementation lock 和 precision audit 均存在。四个 update 的 rollout correct rate
  为 0.35/0.525/0.4167/0.4292，尚未据此判断能力；当前状态为 `trained_pending_audit`，
  checkpoint 与同题 BIRD-dev1534 双卡评测已经完成；只读 lineage audit 不等同于完整
  fresh replay，因此后者仍是进入能力结论前的剩余门禁。

- **4B correctness-only SAAM 新对照（训练/评测完成，未通过能力准入）**：按已登记的离线优先方向
  使用同一 checkpoint-6380、同一 60 题 cohort、Atomic v26、binary result、SAAM asymmetric-error、
  span alpha=0.5、K=8、30题/update、4 updates；采用 4-bit、SDPA 和 vLLM CUDA Graph。
  输出根为 `/home/dengyan/tabular_rl_outputs/rl_runtime_qwen3_4b_correctness_only_saam60_table_rl_20260913_graph_r1`，
  `global_step=4/4`、960 episodes、`train_runtime=26,093.77s`；checkpoint-4、implementation
  lock、precision audit 均存在，训练 GPU 已释放。已完成只读 SAAM lineage/advantage audit（960
  episodes、856 eligible、7,389 events；39条含问题、5条未解析 lineage；未读取 gold SQL），证据为
  `train/saam_lineage_replay_audit.json`；它不等同于完整 fresh replay。matched BIRD-dev1534
  为909/1534，相对同题SFT仅净+6且不显著，因此未通过准入。

- **重复续跑已停止**：此前为验证加速而重新启动的 4B three-level 同 identity 运行在首个
  update 尚未完成前由 owned lifecycle wrapper 收到 SIGTERM，状态为 `failed exit=143`，GPU0/1
  已清理；保留启动、CUDA Graph capture 和日志作为性能证据，不作为训练结果。

- **4B SAAM 三等级重复续跑（已停止）**：该重复根
  `/home/dengyan/tabular_rl_outputs/rl_runtime_qwen3_4b_saam_threelevel_spanbalanced60_table_rl_20260913_graph_r2`
  在首个 update 前收到 owned lifecycle SIGTERM，GPU 已清理；只保留性能/启动证据，不作为训练或能力结果。

按时间倒序；“诊断/中止”不代表正式方案已提升。

## 2026-09-13

- **固定 transition 性能门禁完成（仅诊断）**：`table_rl` 对同一条 1,957-token transition
  完成 4-bit/BF16 old/current 单前向融合对照。4-bit 两次 forward→融合的 steady median
  为 1.8167→1.3895s（1.307x），BF16 为 1.5220→1.1659s（1.306x）；token log-prob
  完全一致、梯度余弦相似度1.0。BF16单前向约快16.1%，但峰值分配约为4-bit的1.68倍。
  无 optimizer、非完整 GRPO update，未切换正式 trainer；下一步仅做真实 batch 集成对齐和
  OOM/checkpoint 门禁。

- **CUDA Graph 固定 example 复测完成（仅诊断）**：`table_rl` 以 `example_index=7365`、4-bit、
  SDPA、checkpointing、actor old-policy、KL=0和`VLLM_ENFORCE_EAGER=0`完成一组8轨迹/47
  transition审计；rollout 74.42s、old-policy 28.38s、policy前反向93.73s、step 198.55s，
  vLLM日志确认CUDA Graph capture，owned资源已释放。采样轨迹不同，不能与eager总时长直接
  相减；未切换正式trainer。根为`/home/dengyan/tabular_rl_outputs/perf_gate_cuda_graph_20260913`。

## 2026-09-12

- **4B PyTorch profiler（2026-09-13，正式精度路径完成）**：初次脚本缺少BF16 autocast，
  FP32/math attention和4096 OOM不能外推正式trainer；正式 r2 使用BF16 autocast、SDPA、
  4-bit/FP32 LoRA和checkpointing完成。2048/4096/8192 无profiler中位 forward+backward 为
  1.481/3.016/6.505s，peak allocated 4.92/6.12/8.90GiB，LoRA梯度覆盖504/504，4096/8192
  均未OOM。8192 CUDA主要为mm、FlashAttention、copy和mul；Command Buffer Full需结合
  profiler扰动解释。远端根 `/home/dengyan/tabular_rl_outputs/perf_profiler_4b_20260913_r2_autocast`，
  无optimizer更新。身份正确的 BF16 paired profiler 随后完成：2048/4096/8192 中位数
  1.257/2.591/5.751s，相对4-bit快15.2%/14.1%/11.6%；12288/14700快10.7%/9.5%，
  长度14700 peak allocated 14.52GiB，504/504梯度覆盖。BF16 暂作为独立 execution arm，
  尚未切换正式训练；需完整 trainer update 的数值和OOM门禁。

- **4B/旧8B耗时核对与速度优化门禁（2026-09-13）**：已核对4B vanilla四步训练计算
  29,842.95s、SAAM-SDPA 29,942.40s，二者只差0.33%；SAAM并未因 mask/有效优势数量
  增加而显著变慢。当前SAAM时间拆分为 rollout 7,307.46s、old-policy 5,154.32s、
  policy前反向17,160.86s，主要成本是同步生成、长序列和单卡训练。旧8B可检索到的
  6h05m是历史规划/运行口径，尚无与当前完全同身份的完整四步实测，暂不作为严格基线。
  已更新 heartbeat `table-rl`：在 table_rl 空闲资源上先做固定 transition 的 CUDA
  Graph复测、4B BF16/4-bit隔离和 actor old/current log-prob 单前向融合的 token/梯度
  对齐；对齐前不启动新的长RL。

- **8B SDPA恢复及匹配评测完成（无显著增益）**：NewGNN隔离根
  `/home/dengyan/tabular_rl_outputs/rl_runtime_qwen3_v26_signed_vanilla_grpo_c2to6_balanced60_20260912_resume_step3_sdpa`
  已从checkpoint-3完成至`global_step=4/4`，checkpoint-4完整且训练资源已释放；唯一执行变化为trainer使用SDPA，
  reward/cohort/rollout/optimizer identity保持不变。checkpoint-4 adapter SHA-256为
  `207a2b59d25f21122a3dd5d164e4734be7e0f3b8c0933e3a53f2b3909a11d272`。最后一批240条rollout的离线审计为
  209条可训练trajectory、1777个transition，最大总长9771 tokens；未发现新的OOM。已在NewGNN GPU2/3
  完成独立的`actionable-error-v1` BIRD-dev1534双卡匹配评测，结果为931/1534（60.691%）、
  1387/1534合法；同协议checkpoint-6380 SFT为928/1534（60.495%）、1380/1534合法。同题号为
  66 gain / 63 regression，净+3题（+0.196 pp），exact McNemar p=0.8603，不能视为可复现提升。
  评测根为
  `/home/dengyan/tabular_rl_outputs/evaluations/qwen3_v26_signed_vanilla_grpo_c2to6_balanced60_checkpoint4_sdpa_actionable_20260912`，
  owned-process cleanup状态为completed，GPU2/3和本任务vLLM均已释放。

- **4B SAAM三等级SDPA恢复及匹配评测完成（未超过SFT）**：`..._r2_sdpa`已完成`global_step=4/4`、
  checkpoint-4、960条rollout和FP32 LoRA/AdamW precision审计，训练进程与online vLLM均已释放。训练rollout为
  428/960正确、842/960合法；reward计数为`1.25:383 / 0.75:45 / -1:529 / 0:3`，最后3条是明确排除的
  `process_update=false`。checkpoint-4 adapter SHA-256为
  `dfa7db27f9d619e777d4a43efa7941b77f7e783de985817585510686840aa3ab`。已在table_rl完成
  `actionable-error-v1`、greedy、2048、BIRD-dev1534双卡匹配评测：897/1534（58.475%）、1358合法；
  同题号对checkpoint-6380 SFT（903/1534、1372合法）为77 gain / 83 regression，净-6题（-0.391 pp，
  exact McNemar p=0.6928）；对4B vanilla RL（890/1534、1369合法）为88 gain / 81 regression，净+7题
  （+0.456 pp，exact McNemar p=0.6445），两项均未达到显著性门禁。评测根为
  `/home/dengyan/tabular_rl_outputs/evaluations/qwen3_4b_saam_threelevel_spanbalanced60_checkpoint4_sdpa_actionable_20260913`，
  owned-process cleanup状态为completed。

- **1–7 K=8筛选继续运行**：NewGNN GPU4/6的两个shard仍健康，已完成1270/3000题（最近核验shard-0 670、shard-1 600；正确875题），未重启或干预。

- **4B SAAM 三等级恢复进度更新（历史快照，已由上条完成记录取代）**：原 eager 运行根 `/home/dengyan/tabular_rl_outputs/rl_runtime_qwen3_4b_saam_threelevel_spanbalanced60_table_rl_20260912` 在首个 update 的 backward 阶段 OOM（尝试分配9.29GiB，GPU0仅9.13GiB可用），无 checkpoint。隔离恢复根 `/home/dengyan/tabular_rl_outputs/rl_runtime_qwen3_4b_saam_threelevel_spanbalanced60_table_rl_20260912_r2_sdpa` 后续已完成第4 update和匹配评测，最终状态见上条。


- **OOM恢复与SDPA续跑（进行中）**：8B signed vanilla-GRPO 的 checkpoint-3→4 恢复在最后
  update 的 eager backward 再次 OOM。失败批次审计为1,847个可训练 transition，其中只有同一
  条错误轨迹的4个 transition达到14.3k–15.5k tokens；4096是多行packing预算，不能限制单行。
  已保留原证据并建立 `...resume_step3_sdpa` 隔离根；唯一训练实现变化是 trainer eager→SDPA，
  reward/cohort/rollout/context/micro-batch不变。preflight已通过，GPU2 trainer/GPU3 vLLM运行中，
  尚未形成checkpoint-4。

- **4B SAAM三等级重跑（进行中）**：首轮在第一批240条rollout后的 eager old-policy attention
  阶段失败，最大13,212-token transition触发9.29GiB softmax申请，未提交checkpoint。新根
  `...table_rl_20260912_r2_sdpa` 从checkpoint-6380重新开始，唯一执行变化为trainer eager→SDPA；
  已写入240条rollout并进入GPU0计算，尚未再次OOM。`generation_length`按该三等级 profile
  的预注册规则继续作为可训练负样本。

- **Qwen3-4B SAAM 三等级 reward + span-balanced baseline（首轮已失败）**：使用 4B
  cumulative SFT epoch4 `checkpoint-6380`、新60题 cohort、Atomic v26、K=8、30题/update、4 updates。
  新 profile `three-level-clean-weighted` 固定为 clean correct=1.25、error correct=0.75、
  incorrect=-1.0；credit=`saam-asymmetric-error`；全response span balance alpha=0.5、
  uniform routing；KL=0。该运行是独立 4B 诊断，不替换8B主线。启动前记录 immutable manifest、
  implementation lock、precision audit 和双卡资源证据；首轮未完成optimizer update，不能作能力结论。


- **Qwen3-8B table_rl 软件加速对照（诊断，已完成）**：双3090分别运行trainer/vLLM，checkpoint-6380、新60题、Atomic v26、单题K=8、1 update。Eager 152.67秒/13,903 tok（91.1 tok/s，但仅1个transition有效）；CUDA Graph 166.88秒/22,749 tok（136.3 tok/s），sampling复用 449.24秒总训练并省约84.7秒 old-policy forward；三者有效训练量不同，不能作端到端速度排名。`max_num_batched_tokens=32768` 为449.69秒，和默认无差异；microbatch=2 backward OOM。A100未测试。证据根：`table_rl:/home/dengyan/tabular_rl_outputs/perf_audit_qwen3_8b_v26_onegroup_20260912_graph_sampling_mbt32768`。


- **性能优化适用范围核查与共享记录（未新启GPU实验）**：已确认共享server和本地8B launcher默认
  `VLLM_ENFORCE_EAGER=0`；8B已在table_rl完成单组CUDA Graph诊断，A100本轮未测。修正replicated BF16 checkpointing默认
  回归，3个CPU测试通过。详见[启动前性能记录](../docs/current/rl_performance.md)，明确区分
  episode/训练transition、step_time/train_runtime及sampling-score的算法影响。

- **RL update 性能优化 smoke（诊断，已完成）**：在独立 source copy 上加入 checkpointing/
  attention 配置记录并比较 vLLM CUDA Graph。`--no-gradient-checkpointing` 长轨迹 OOM；
  checkpointing+SDPA 总306.49秒；`VLLM_ENFORCE_EAGER=0` 总199.43秒，rollout 74.67秒，
  相比 retry1 总303.55秒约快34.3%。server 默认已切到 CUDA Graph，训练默认仍启用
  checkpointing；结果不用于 RL 能力判断。

- **old-policy sampling-score smoke（诊断，已完成）**：CUDA Graph 下启用显式 sampling
  logprob 复用，单组 update 171.86秒，old-policy forward 约0秒，较 actor-score 199.95秒
  再快14.0%。importance ratio恒为1，属于改变 correction 目标的速度 ablation；正式默认仍
  使用 actor old-policy scoring。

- **table_rl 双卡性能审计（packing 对照已完成）**：上一轮 Qwen3-4B checkpoint-4 的
  `actionable-error-v1` 双卡评测已完成并释放 GPU0/1，合并结果为 890/1534。随后在
  table_rl GPU0 trainer + GPU1 vLLM 启动隔离的一组性能 smoke；审计不改变已完成训练根目录，
  使用 checkpoint-6380 SFT adapter、同一 Atomic v26/60题数据、K=8、1个 optimizer update、
  `transition_micro_batch_size=1`、4096 token packing cap，并在隔离 source copy 中加入
  rollout generation、Harness apply、old-policy microbatch、training microbatch/backward
  的 JSONL 埋点。首个 instrumented smoke 因埋点变量未在训练循环初始化而退出，已保留失败日志；
  修正后的 retry1 已完成：一个8轨迹组37个transition的update总计303.55秒，其中rollout
  203.85秒、old-policy 22.92秒、policy backward阶段74.87秒，37个transition在4096 cap下
  产生37个microbatch。随后完成`retry2_pack8192`同题packing对照：microbatch降为18个但
  update反而342.73秒、policy阶段113.27秒，峰值reserved约11.77GiB；因此不把简单增大
  token cap作为正式修复。两者都只能作为性能诊断，不能作为RL accuracy结果。

- **c=2–6 signed vanilla-GRPO OOM恢复（首个恢复副本已失败）**：原隔离运行
  `/home/dengyan/tabular_rl_outputs/rl_runtime_qwen3_v26_signed_vanilla_grpo_c2to6_balanced60_20260911`
  在第1个已提交update之后的policy backward因CUDA OOM退出；checkpoint-1有效，原rollout
  480条中前240条已提交、后240条未提交，已由`resume_preparation_20260912.json`审计并移除。
  未修改原manifest/implementation lock；已建立逐字复制的隔离恢复根，并将micro-batch上限降为
  1 row/4096 padded tokens、启用expandable segments。外部生命周期wrapper已通过语法和
  preflight。2026-09-12 05:52（服务器时间）复核NewGNN GPU2/3均无compute PID、显存近空且
  独占锁可获得后，已用GPU2 trainer + GPU3 online vLLM从checkpoint-1接续；TRL服务
  `18312`完成模型加载并通过health/serving-contract检查，trainer已加载基座并进入`0/4`
  恢复进度。后续已核验checkpoint-2（`global_step=2/4`、`epoch=1.0`）和累计480条rollout，
  未见OOM。当前进入下一批rollout时，GPU2 trainer暂时等待GPU3 online vLLM；因此GPU2利用率
  可为0但显存仍由trainer占用约22.3GiB，GPU3承担生成计算，不代表GPU2空闲或训练退出。
  当前已核验checkpoint-3（`global_step=3/4`、`epoch=1.5`）和累计960条rollout；最后update
  在eager backward再次OOM，原根状态为failed，GPU2/3已释放。该副本不作为完成结果。

- **1–7候选库存重建（已完成静态整合，未冻结正式cohort）**：复用已有Atomic v26 K=8筛选
  观察，按稳定`example_id`去重并保留筛选来源。strict 896题、expanded 1025题、fresh
  353题；候选直方图为1:179、2:128、3:92、4:85、5:107、6:114、7:191。输出及哈希见
  `data/inventory/rl_training_candidates_1to7_20260912/summary.json`和对应报告。

- **3000题K=8筛选（2026-09-12启动；2026-09-16完成，见2026-09-16记录）**：NewGNN GPU4/6分别运行1500题shard，checkpoint-6380、
  Atomic v26、temperature0.8、max_tokens2048、max_steps30、stop_on_success=false；vLLM
  端口8205/8206，筛选父进程321249/321250，evaluator进程328646/328647。两个结果manifest
  已成功创建且health=200；最近复核结果文件分别有73/1500和68/1500个去重完成题目（summary
  的`total`只表示已写入完成记录，不能误读为输入规模），两个vLLM端点仍健康且有活动请求，
  尚不能统计新批次的1–7入选数量。完成后合并
  结果并追加到版本化候选库存，不覆盖原2–6库存。
  只读吞吐核查显示并非停滞：已写入记录的单题耗时中位数约605秒/403秒、轨迹步数中位数
  8/7；当前约10–14题/小时/ shard，按此速率3000题可能需要约4–6天。当前不重启正在运行
  的筛选，避免破坏已完成结果；后续若资源和配置允许再做独立吞吐优化。
  2026-09-16更新：两个shard均已跑满1500/1500，GPU4/6已释放；结果已合并为1–6候选库存
  （strict 1209 / expanded 1329 / fresh 776），区间外观察另行保留。上面“进行中”的描述
  只代表2026-09-12的状态。

## 2026-09-11

- **RL筛选候选池统一整合（已完成，未冻结正式cohort）**：按稳定`example_id`整合原始700题、
  新600题、剩余848题和历史候选索引；完整K=8且正确数2–6的strict候选为545题，扩展历史池
  为723题，已登记逐次筛选观测2603条。旧A100 700题cohort重叠423题，SFT training view重叠
  98题，均保留标记；当前60题复筛只作diagnostic。产物与复现入口见
  [`RL_TRAINING_CANDIDATE_POOL_20260911.md`](../docs/reports/rl/RL_TRAINING_CANDIDATE_POOL_20260911.md)。

- **Qwen3-4B c=2–6 balanced60 signed vanilla GRPO（训练与评测完成）**：旧60题复筛确认
  27/60为8/8全对、仅14/60满足2–6，不能继续作为当前起点的边界题。从checkpoint-6380
  已完成848题池的204道2–6候选中，仅按正确数每档固定12题；新60题SHA为
  `1a6cb257081da58ce66e09ab4e0783f296e3fdc4aa5674da178ae4c86756b4ca`。预注册signed-binary
  vanilla GRPO、trajectory/uniform、4 updates、LR4e-7、KL0。从4B cumulative SFT epoch4
  `checkpoint-6380`启动；2026-09-11 17:49在table_rl以GPU0 4-bit trainer、GPU1 online vLLM
  启动，CPU身份门禁、服务能力门禁和首次在线权重同步均通过。step-1已完成并核验（240条
  rollout、98正确、221条可训练，step总计5632s）；因rows=1导致micro-batch 1153个，已封存
  后从checkpoint-1续跑rows=8/8192。step-2也已完成（global_step=2、step约8200s，平均
  1.63行/micro-batch）；step-3也已完成（global_step=3、step约8351s，平均1.74行/micro-batch），
  step-4完成后训练正常exit 0；960条rollout共390正确、850合法、861条进入优化，precision
  audit和4次QLoRA权重同步均通过，GPU已释放。checkpoint-4随后以双卡独立服务完成
  actionable-error-v1 BIRD-dev1534：890/1534（58.018%）、1369合法；对照4B SFT epoch4
  为903/1534（58.866%）、1372合法，逐题配对净-13题（79 gain、92 regression），
  McNemar p=0.3588。分层结果为challenging +7、moderate -5、simple -15，整体合法率仅
  -0.20个百分点；1190/1534题action sequence变化但工具边际JSD仅7.77e-05，支持“小cohort+
  低预算vanilla统一轨迹credit导致局部高方差改写”的诊断，尚不足以称为显著退化。题目的
  2–6分档来自8B筛选池，4B实际难度以本次online rollout为准。
  见
  [运行报告](../docs/reports/rl/QWEN3_4B_VANILLA_GRPO_BALANCED60_20260911.md)。

- **60题 checkpoint-6380 起点 K=8 复筛（已完成）**：前置 checkpoint-4 的
  actionable-error-v1 双卡评测已完成（GPU0 455/767，GPU1 447/767；合计 902/1534），
  本任务 vLLM 已释放。随后对字节级一致的60题 cohort（SHA-256
  `b377ab3f7cbec8745a5342f3428435f91a77b8ada5f35aafb3465c780a07bbb4`）启动 checkpoint-6380
  SFT 起点复筛；每题8条、temperature 0.8、top_p 1、max_tokens 2048、max_steps 30、
  pass-k 1/2/4/8、stop_on_success=false、Atomic v26/protocol hash `4da19387399bd3a5`。
  两个 evaluator 分别运行于 NewGNN GPU1/3（端口18410/18411），输出目录为
  `/home/dengyan/tabular_rl_outputs/evaluations/qwen3_v26_checkpoint6380_rescreen_current60_k8_20260911`；
  复筛完整生成480条轨迹：27题全对、32题混合、1题全错，477/480合法；本任务vLLM已释放。

## 2026-09-10

- **后续筛选审计与RL接续任务（分支已纠正，等待当前双卡评测完成）**：先确认当前60题
  的原始筛选 checkpoint。若为 checkpoint-6380，必须对完全相同的60题用 checkpoint-6380
  SFT 起点重新做一次 K=8 筛选，统计每题正确数、全对/混合/全错组和合法率；若为
  checkpoint-560，则不重复筛选，直接把它作为旧模型已验证的简单题证据。当前 cohort hash
  与 checkpoint-6380 历史运行链一致、与 checkpoint-560 历史60题不一致，因此暂按前一分支
  排队。复筛结果出来后，全对组高才从现有池重新固定2–6题并先做 vanilla GRPO；不高则
  排查未更新参数时的正确率跃升。所有筛选仍是一次性固定，不在 RL epoch 中动态重筛题。
  由 heartbeat `table-rl` 接续推进。

- **固定 cohort 筛选口径复核（2026-09-11）**：确认 RL 训练不应按 epoch 重新筛题；正确
  流程是 checkpoint-6380 SFT 上一次性 K=8 筛选，随后固定题目池，仅在 RL 中重新采样
  轨迹。当前 hybrid r2 第一个 update 的 30 题出现17个全对组，不能直接解释为题目过于
  简单；远端60题 cohort 当前只含题目/通用来源元数据，未含筛选阶段正确数与 decode
  manifest，暂不能验证2–6筛选条件。后续核对原始 SFT screening 产物和同题配置身份。

- **Legal-reason hybrid 完成并否决（2026-09-11）**：4/4 updates、checkpoint-4
  `global_step=4`；训练rollout为807/960正确、952/960合法。actionable-error-v1全量评测为
  902/1534，低于同协议checkpoint-6380 SFT的928/1534，净退化26题（-1.69 pp）。同题号
  49 gain / 75 loss，exact McNemar p=0.02437；合法率1380→1378，54 gain / 56 loss，
  p=0.92410。process errors反而784→743，说明主要问题不是整体合法率崩溃，而是reason-only
  粗粒度credit未定位语义决策且屏蔽了合法tool参数载体。该hybrid routing停止推广；当前
  checkpoint-6380 K=8复筛完成前不启动新RL。见[完整训练与配对审计](../docs/reports/rl/LEGAL_REASON_HYBRID_RESTART_20260910.md)。

- **Hybrid r2已实际启动流程（23:46–23:47，未核验有效update）**：确认GPU2/7空闲后以
  2训练/7推理启动，控制PID3694416；新CPU preflight和TRL服务HTTP能力检查通过，模型
  加载完成，状态进入starting_trainer。后续两次日志查询被平台审核容量故障拒绝，不能
  宣称训练健康或有效更新。GPU4/6筛选保留；下次先核查，禁止重复启动。
  见[进程/状态证据](../docs/reports/rl/LEGAL_REASON_HYBRID_RESTART_20260910.md)。

- **Hybrid 20:42定时检查：无空卡，未训练**：NewGNN八卡均有compute PID，本实验仍为
  waiting_resources，无trainer_state/checkpoint/rollout。本轮只读检查，无新GPU进程；
  GPU4/6筛选保持不变，继续半小时接续。见[监控证据](../docs/reports/rl/LEGAL_REASON_HYBRID_RESTART_20260910.md)。

- **Hybrid r2全部启动前门禁通过，等待双空卡（20:08后，未训练）**：清理修复部署后远端
  105项测试通过，CPU身份preflight及GPU2真实CUDA BF16核验通过并释放。
  GPU0短任务占用反复变化，实际启动前检查exit75，未启动trainer/vLLM；半小时heartbeat
  将按已确认配置继续尝试，保护GPU4/6筛选。见[证据和接续入口](../docs/reports/rl/LEGAL_REASON_HYBRID_RESTART_20260910.md)。

- **Hybrid CPU门禁通过，补充清理修复（未GPU启动）**：r2真实远端101项测试通过、1项跳过，
  CLI完整模型/cohort/冻结runtime身份preflight通过；均有保存的日志及exit0。
  另修复退出回调作用域与信号退出码，正在部署增量冻结包并复测；不改算法参数。
  见[接续证据](../docs/reports/rl/LEGAL_REASON_HYBRID_RESTART_20260910.md)。

- **Hybrid r2部署完成，监控更新生效（未GPU启动）**：19:35后代码包及同60题cohort已上传/
  哈希验证并部署到隔离r2。官方接口确认现有半小时heartbeat已更新为60题hybrid接续，
  不再运行旧120题待确认逻辑。真实远端回归/preflight仍进行中，未宣称有效update。
  1483题筛选保持运行；GPU0/2曾空闲但启动前需重新核验。见[续查记录](../docs/reports/rl/LEGAL_REASON_HYBRID_RESTART_20260910.md)。

- **Hybrid远端清理推进（未重启）**：用户明确“允许”后，NewGNN状态及旧服务归属核验
  完成，向本任务PID2750458发送TERM返回exit0；显存释放复查和r2创建再次被平台审核
  容量错误拦住。16:30快照没有双空卡，1483题K8筛选未触碰。见[续查记录](../docs/reports/rl/LEGAL_REASON_HYBRID_RESTART_20260910.md)。

- **Hybrid接续阻塞**：代码修复及144项本地测试完成，冻结包和接续流程已记录；远端清理/
  部署与heartbeat更新均遇授权服务故障，未完成且未冒称生效。原半小时任务仍为旧120题
  只读监控，需批准/恢复后更新为本60题诊断。见[续查记录](../docs/reports/rl/LEGAL_REASON_HYBRID_RESTART_20260910.md)。

## 2026-09-09

- **KL 静态审计（完成，未启动 KL 训练）**：对 NewGNN 已完成的 checkpoint-4 legal-reason
  hybrid run 核对训练指标、960 条 rollout 字段和 checkpoint-6380→checkpoint-4 的 LoRA
  位移。四次 update 的 sampler→trainer importance `log_ratio_abs_mean` 为
  0.01880–0.01994、cap exceeded 均为0；504个 LoRA 张量相对位移为0.1663%。这排除了
  明显的采样/训练错位，但不是 reference-policy KL；rollout未保存 reference 逐token
  logprob，故不能声称真实KL很小或已证明KL无效。静态结论是KL最多作为稳定器，当前没有
  证据把它当主要增益来源；后续仅做独立 matched beta 对照。详见
  [`KL_STATIC_AUDIT_LEGAL_REASON_HYBRID_20260911_ZH.md`](../docs/reports/rl/KL_STATIC_AUDIT_LEGAL_REASON_HYBRID_20260911_ZH.md)。

- **Legal reason-only hybrid NewGNN validation（2026-09-10更正：启动失败，非训练中）**：远端 train.log 证实14:00模型加载后，在 TRL `init_communicator` 请求普通 vLLM 服务时收到404，未进入有效更新。原 GPU3 已有其他用户任务；GPU4/6筛选保持不动。修正版使用独立 signed-binary(+1/-1)、正确的逐turn错误标记和 TRL 服务门禁；144项本地测试通过，GPU重启/修正版远端部署尚未完成，不能宣称效果。精确配置、冻结代码包、启动与接续门禁见[续查记录](../docs/reports/rl/LEGAL_REASON_HYBRID_RESTART_20260910.md)。

- **Legal reason-only hybrid static audit（诊断完成，待用户确认后启动）**：对 checkpoint-560 与 checkpoint-6380 已保存的 K=8 rollout 做合法/错误类型和组内 advantage 静态审计。6380 的错误轨迹中约92.5%–96.0%为合法轨迹，支持降低合法错误的 tool span 权重；但合法语义错误常由工具参数决定，不能把 tool span 完全置零当作已验证方案。按用户确认，新实验采用wrong-uniform/binary reward，以排除旧four-level的全错组错误正优势和全对组正确负优势；拟在 NewGNN 用同一 checkpoint-6380、60题、K=8、4 updates、30题/update、binary/SAAM、alpha=0.5 非错误turn与错误turn路由对照；启动前需确认配置，详见 [`REASON_ONLY_LEGAL_HYBRID_STATIC_AUDIT_20260910_ZH.md`](../reports/rl/REASON_ONLY_LEGAL_HYBRID_STATIC_AUDIT_20260910_ZH.md)。

- **Wrong-uniform SAAM 120题 NewGNN validation（已冻结，待启动）**：使用已完成600+848筛选池中正确轨迹数1–6的120题，固定配额30/25/25/20/10/10；checkpoint-6380、Atomic v26、binary result（全部错误轨迹-1）、SAAM asymmetric-error、trajectory-token-mean、reason/tool span alpha=0.5、4 optimizer updates、30题/update、K=8、max_new_tokens=2048。标准JSONL任务SHA为 `4f37f5c717e534c9715af14e1da91c7f8a73081c26ed3bf7c33d4804454ba901`；配置已部署，GPU preflight通过前不启动训练。

- **Qwen3-4B Atomic v26 SFT scale control（旧运行中止，已按 checkpoint-6380 规模重启）**：先前 4,471-record/2-epoch/560-step 运行已因与当前 Qwen3-8B `checkpoint-6380` 不同规模而中止，保留 `exit_status=143` 记录；核对 8B immutable manifest 后确认当前 cumulative 数据为 25,513 records、4 epochs、effective global batch 16、1,595 steps/epoch、6,380 steps。现已在 `table_rl` 使用同一 training view（SHA-256 `f664a7c77210f34e6f202dc21313fecd43e7044b3ce25bd5398f8691b7a491d9`）、Qwen3-4B 固定 revision `1cfa9a7208912126459214e8b04321603b3df60c`、双3090、gradient accumulation 8 重启 cumulative 对照；训练成功后再启动 base/SFT BIRD-dev1534 双卡评测。

- **Reward 机制多代理审查（诊断完成）**：基于Qwen3-8B checkpoint-6380既有A100/table_rl/SMC/SAAM rollout，第一轮4个独立子任务后又追加5个独立脑暴子任务；排除通用操作bonus、local-Q、Gold-path、provenance/closure、critic、树搜索和SMC top-1。A `wrong-uniform + correctness-only SAAM`确定为唯一主训练方向，C窄范围format-only span保留为离线门禁，B first-error capped仅作后续小消融；recovery-preservation、conservation-budget ledger、mixedness-gated weighting暂不进训练。未改代码、未启动训练、未占用GPU。详见 [`REWARD_BRAINSTORM_SYNTHESIS_20260909_ZH.md`](../reports/rl/REWARD_BRAINSTORM_SYNTHESIS_20260909_ZH.md)。

- **Reward 变体离线重算（完成）**：扩展 boundary mixed 池1516条可用轨迹/193组上，four-level的错误正向/正确负向为8/68；binary与sign-preserving为0/0。仅验证信号方向，不作accuracy结论，不启动RL。详见 [`REWARD_BRAINSTORM_SYNTHESIS_20260909_ZH.md`](../reports/rl/REWARD_BRAINSTORM_SYNTHESIS_20260909_ZH.md)。

- **First-error capped 静态质量计算（完成）**：boundary池652个确定性error events中，首错封顶移除约49.95%的局部负向mass，包含67个最终正确轨迹的后续error事件；仅作候选诊断，不启动RL。

- **State-conditional contrast 覆盖率审计（诊断完成）**：first32/K8 与 boundary mixed 池的严格同状态动作对比覆盖分别为0.80%和2.00%；leave-one-out预测未证明动作条件值稳定优于状态均值。不接入reward、不启动RL。详见 [`STATE_CONDITIONAL_CONTRAST_COVERAGE_AUDIT_20260909_ZH.md`](../reports/rl/STATE_CONDITIONAL_CONTRAST_COVERAGE_AUDIT_20260909_ZH.md)。

- **观察工具时序分布（全量只读诊断完成）**：8,480条rollout中98.48%首turn为观察；前/中/后1/3的观察占比76.67%/29.46%/18.59%。同时含观察/操作的8,214条中46.80%操作后再次观察，正确轨迹也有40.51%；read_subtable的77.63%位于中后段。不能按阶段判观察无效或据此证明最短路径最优。来源哈希重验、计数及实例见 [观察时序报告](../docs/reports/rl/OBSERVATION_TOOL_POSITION_AUDIT_20260909_ZH.md)。未改reward/未启动RL。

- **图外操作效用核查（全量分母+80例逐条诊断）**：31,378个操作中6,992个图外：694执行失败、2,459成功但无终点、3,839成功且有终点。对非保留分区后者抽查80个：46有实际信息传递、14重复/无新增任务信息、10放弃尝试、9不确定、1错误语义传值；不是全量语义比例。全量另检出138个再次执行相同成功调用，和抽查有交集。未改reward/未启动RL。详见 [图外操作审计](../docs/reports/rl/HARNESS_OUTSIDE_OPERATION_AUDIT_20260909_ZH.md)。

- **闭包覆盖聚合更正/bonus效用复核（诊断完成）**：原按trajectory_id连接覆盖了table_rl跨update同名记录，旧71.209%撤回；8480条逐记录重算操作闭包24,386/31,378=77.717%，正确操作16,014/18,692=85.673%。这不是semantic precision或实际非零梯度覆盖；保留V6，新增逐源哈希/分母[V7](../docs/reports/rl/HARNESS_CAUSAL_CLOSURE_AUDIT_20260909_V7.json)。复核5150样本确认Ingredient过滤通过自动预览提供literal6，未被闭包连接；不改reward、不启动RL。详见 [报告](../docs/reports/rl/HARNESS_CAUSAL_CLOSURE_AUDIT_20260909_ZH.md)。

- **纯最终格式错误逐条人工审计（诊断完成）**：逐条核对158条合法错误轨迹：33条format-only、118条semantic_wrong、7条no-terminal。以当前人工确认的33条format-only为分母，包含关系识别precision/recall均为100%；不再使用20.9%/24.4%作为format recall。未改正式reward，建议只对最后format action附加惩罚。详见 [`FORMAT_ERROR_MANUAL_EXHAUSTIVE_AUDIT_20260909_ZH.md`](../docs/reports/rl/FORMAT_ERROR_MANUAL_EXHAUSTIVE_AUDIT_20260909_ZH.md)。
- **Project位置边界修正（诊断完成）**：437个`project`中仅281个紧邻answer；严格结果链352个project中114个是中间语义步骤，不能统一归为format。33条format-only里9条是最终project，24条是最终filter/top的`return_columns`差异，后者只在支持span级mask时惩罚format字段。未改正式reward。

- **格式错误包含关系审计（诊断完成）**：同题正确轨迹reference与错误terminal结果做保守包含检查；158条合法错误中33条为高置信format-only候选。20.9%/24.4%只是候选在全部/可比较错误中的占比，不是format recall；真实format-only总数尚未独立标注。建议`project/answer`不进入core reward，确认format-only时只惩罚最后格式动作；不改变当前RL配置。详见 [`FORMAT_ERROR_CONTAINMENT_AUDIT_20260909_ZH.md`](../docs/reports/rl/FORMAT_ERROR_CONTAINMENT_AUDIT_20260909_ZH.md)。

- **Process credit 人工 precision 审计（诊断完成）**：在新版60题/K=8的960条rollout上，对严格终点data/value闭包候选做50个分层人工核对：41/50（82%）是真正改变问题语义的core动作，6/50（12%）是答案表示层projection，3/50（6%）是可选或冗余动作；将projection算作任务相关信号为47/50（94%）。确认观测结果经普通literal传递会形成闭包漏标，当前不能声称全量semantic recall。未改reward、未启动RL。详见 [`PROCESS_CREDIT_MANUAL_PRECISION_AUDIT_20260909_ZH.md`](../docs/reports/rl/PROCESS_CREDIT_MANUAL_PRECISION_AUDIT_20260909_ZH.md)。

- **Process credit coverage audit（诊断完成）**：新版60题960条rollout静态重建 provenance；严格结果链候选覆盖正确轨迹动作52.63%，含grounding候选76.52%；高置信错误动作只覆盖15/173条错误轨迹。无独立语义标注，precision/recall只对显式结构/event一致性报告，不改变 reward 或启动RL。详见 [`PROCESS_CREDIT_STATIC_COVERAGE_AUDIT_20260909_ZH.md`](../docs/reports/rl/PROCESS_CREDIT_STATIC_COVERAGE_AUDIT_20260909_ZH.md)。

- **评测框架更新（已确认）**：后续新评测统一使用`actionable-error-v1`并做feedback identity preflight；当前checkpoint-3 legacy在途评测按原身份收尾，不中断。定时任务不再在空卡时自动启动旧RL。

- **Action counterfactual fixed-suffix audit（诊断完成）**：Atomic v26 frozen runtime；30个非保留混合题组、60条A100历史轨迹、383个有效 skip branches；数据库前后哈希不变。最终结果和保守标签见 [`ACTION_COUNTERFACTUAL_AUDIT_20260909_FINAL_ZH.md`](../docs/reports/rl/ACTION_COUNTERFACTUAL_AUDIT_20260909_FINAL_ZH.md)；不启动RL、不改变SAAM/GRPO reward，后续若要回答policy effect需屏蔽观察后重新rollout。

- **Correctness-primary checkpoint 顺序评测（17:10核查，诊断）**：checkpoint-1/2分别为924/1534、915/1534；checkpoint-2正常清理后已自动接续checkpoint-3（worker日志110+107条），checkpoint-4此前为917/1534。实际RL评测全部为`legacy`反馈；928/1534的SFT为`actionable-error-v1`，此前聊天将两者称为同反馈对照有误，不能据此归因纯RL退化。见 [运行记录及纠正](../docs/reports/rl/CORRECTNESS_PRIMARY_TABLE_RL_20260909.md)。不重启训练，不改变在途队列配置。

- **外部教师过程credit pilot（已完成，仅诊断）**：使用当前table_rl既有轨迹和官方DeepSeek完成22次实际请求（含停止时已发出的并发请求），v4-pro/flash分别暴露长轨迹输出耗尽和错误归因违反终局事实的问题；10条通过当前门禁、12条拒绝。未更新actor，不改变进行中的实验。见 [报告](../docs/reports/rl/TEACHER_PROCESS_CREDIT_PILOT_20260909.md)。
- **当前依赖解析静态审计（已完成，仅诊断）**：table_rl 960条轨迹中，6487条合法action的显式 table/value dependency 全部解析到已知source/producer；3个 `in_table` source edge 是记录器漏写，15条unparsed和10条无terminal来自轨迹错误/截断。旧unresolved主要是普通literal grounding和终端slice，未接入reward。见 [报告](../docs/reports/rl/PROCESS_CREDIT_DEPENDENCY_PARSE_AUDIT_20260909.md)。
- **Action-level counterfactual 静态/人工审计（已完成，仅诊断）**：对既有 A100 rollout 做固定后缀 skip replay；6条完整 legal trajectory 的原始终局 replay 全部一致，抽查中观察/探索 action 可与当前后缀依赖步骤区分。但固定后缀无法识别删除 observation 后的 policy-mediated effect，error action 还需纳入统一 intervention node。未更新 actor、reward 或正式配置。见 [报告](../docs/reports/rl/ACTION_LEVEL_COUNTERFACTUAL_STATIC_MANUAL_AUDIT_20260909_ZH.md)。

- **Correctness-primary SAAM 诊断（训练完成，评测进行中）**：table_rl 以 checkpoint-6380、同新版60题、30×8、4 updates、clean weight0.25/span0.5 完成 960 条 rollout；每个 update 的 rollout 正确率为 81.25%、84.17%、91.25%、79.17%，训练日志显示有效 transition 比例约42.9%–53.2%，有效 trainable-token fraction 约1.42%–2.66%。120 个组中63组全对、2组全错，表明主要剩余问题是结果对比组不足而非正负方向错误。r6 已按旧 feedback/2048 匹配启动，检查时已写入约1280/1534条，最终同题结果待完整收尾。详见 [运行记录](../docs/reports/rl/CORRECTNESS_PRIMARY_TABLE_RL_20260909.md)。

## 2026-09-08

- **SAAM span 三组配对审计**：同一 checkpoint-6380/新版 60 题/K=8/4-update protocol 下核对 OFF（`alpha=None`）、alpha=0.5、alpha=0.25。全量 dev1534 分别为 OFF 933、alpha=0.5 923、alpha=0.25 935；A025 相对 A05 为 66/54 gain/loss，A025 相对 OFF 为 70/68，OFF 相对 A05 为 51/61。A025 已完成评测并释放 vLLM；轨迹级 reward/span 审计见 [`SAAM_SPAN_THREE_ARM_AUDIT_20260908_ZH.md`](../docs/reports/rl/SAAM_SPAN_THREE_ARM_AUDIT_20260908_ZH.md)。
- **SFT反馈回归**：`actionable-error-v1`，checkpoint-6380，用户确认固定2048、dev1534 greedy单次新反馈对历史924/1534；table_rl双卡。首次23:49启动因Triton库搜索路径缺失失败（未答题，已清理）；已有库路径的编译/BF16内核检查通过，独立r2重新preflight后于09日00:05:59启动，两服务和评测器已ready，实际manifest确认2048与历史评测相同。正式结果待出；详见 [`SFT_FEEDBACK_REGRESSION_20260908_ZH.md`](../docs/reports/rl/SFT_FEEDBACK_REGRESSION_20260908_ZH.md)。

- **K=8 rollout 静态语义审计**：覆盖 a100/table_rl 既有语料；见 `docs/reports/rl/ROLLOUT_CORPUS_SEMANTIC_AUDIT_20260908_ZH.md`。仅为诊断证据，未改变 reward、cohort 或正式评测。
- **span alpha=0.25 对照**：table_rl，`checkpoint-6380`，新版 60 题，K=8、4 updates、4-bit frozen base + FP32 LoRA/AdamW；训练后 matched BIRD-dev1534。与 alpha=0.5 独立比较，不能替代正式 arm。见 `docs/reports/rl/SAAM_ALPHA025_TABLE_RL_20260908.md`。
- **A100 主线**：update-40 checkpoint 可审计；step49 因 OOM 退出。旧 700 题/14 题每 update，仅作历史诊断。

## 2026-09-07

- **SAAM 新版 60 题训练**：从 `checkpoint-6380` 启动，four-level + span-balanced，训练完成；NewGNN matched 评测仍是衔接证据，尚未形成正式全量准入结论。
- **SMC 审计修复**：原始 cohort 完成 120 题组/960 候选的完整字段、连接和选择一致性检查；新版 cohort 仍需完整概率审计。
- **SMDP/IQL 离线诊断**：已实现 transition、Q/V 与可识别性审计入口；`diagnostic_only`，不更新 actor。

## 当前准入状态

正式 RL 仍待约 500 题 cohort 冻结、immutable manifest、fresh replay/audit 和独立 matched BIRD-dev1534 证据。历史报告和旧实验只能通过 `docs/reports/`、`docs/archive/` 审计，不能自动恢复为当前入口。
# 2026-09-09 Qwen3-4B cumulative SFT throughput calibration and formal restart

- **状态：训练完成，epoch3评测进行中、epoch4排队（2026-09-11）。** 用户定义的“4k 数据”固定为 4,284 条完整轨迹、25,513 条 model-visible turn records，数据 SHA-256 `f664a7c77210f34e6f202dc21313fecd43e7044b3ce25bd5398f8691b7a491d9`，从 checkpoint-6380 对应 cumulative 数据重训 Qwen3-4B。
- 在 table_rl 双 RTX3090、相同 global batch=16、同一 25,513-record 数据上完成 10-update 隔离吞吐短测：batch=1/grad-acc=8 为约 19.3 s/update，batch=2/4 为约 26.7 s/update，batch=4/2 为约 33.2 s/update；batch=4 峰值显存约 11.3 GiB。保留 batch=1，避免以空闲显存换取更慢的长序列 attention。
- 正式 run 已启动：`qwen3_4b_atomic_v26_cumulative_formal_b1_20260910`，配置 `formal_b1.yaml`，GPU 0,1，4 epochs，6,380 optimizer updates，输出 `/home/dengyan/tabular_rl_outputs/checkpoints/qwen3-4b-atomic-v26-cumulative-fresh4ep-table-rl-formal-b1`。短测和正式 run 使用独立输出目录；状态以远端 launch/status manifest 为准。
- **吞吐解释**：上述 batch=1/2/4 短测均固定 global batch=16，因此 update 数相同，只比较同一优化预算下的 micro-batch/累积实现。若改为 global batch=32，更新数约降至 3,190，但训练噪声、学习率调度和总墙钟时间需作为独立 matched run 实测，不能直接替换当前正式 run。
- **双卡评测纠正（2026-09-11）**：用户所指双卡为两份完整模型，已停止原TP=2 legacy运行并保留输出；一次错误分片尝试在preflight即失败、未占GPU。新队列复用共享data-parallel runner，每卡TP=1、并发32，总64，actionable-error-v1/2048/BIRD-dev1534（767+767）。epoch3通过preflight并启动，完整合并和释放后才评epoch4，对照8B epoch4新反馈928/1534。详见[报告及manifest](../docs/reports/sft/QWEN3_4B_CUMULATIVE_DP_EVAL_20260911.md)。
# 2026-09-15 8B table_rl single-update GPU gate（已注册，启动中）

- Cohort：冻结 60 题 `data/rl_inputs/qwen3_8b_atomic_v26_saam_gate60_train_v1.jsonl`，SHA-256 `47e9d369bf6506d13b2432ac92bb971bf52912cb759c133846801f5411ad79d5`。
- Model：Qwen3-8B checkpoint-6380，adapter SHA-256 `8900e4c482f4624ff05059a6fecc2811fa99552cddea7550d44ce8cde853abe6`；Atomic v26；binary result + SAAM asymmetric-error；1 optimizer update，K=8，30 prompts/update。
- Runtime：table_rl 两张 RTX3090，trainer/vLLM 分卡；4-bit、gradient checkpointing、SDPA、old-policy-logprob-source=actor、micro-batch 1、4096 transition tokens、KL=0。输出根目录和 immutable manifest 由远端启动器生成。
- Admission：GPU/显存/吞吐 gate；结果隔离，未通过 gate 或未完成 replay/audit 前不得晋升主线。

# 2026-09-15 4B credit concentration audit（诊断，首个完整 update）

- Active arm：Qwen3-4B `later_error_quarter`，table_rl 双 3090，首个完整 update 30 组/240 条 rollout，201 条 eligible trajectory、1,751 transitions。
- Exact coefficient-mass audit：effective extreme minority 5 条，合计占绝对系数质量 8.62%、负系数质量 17.64%；单条最大占比 2.16%，阈值 >=1% 为 5 条、>=2% 为 2 条、>=5% 为 0 条。multi-error 5 条，合计绝对系数质量 2.79%、负系数质量 4.49%，单条最大 0.97%。
- Direct error replacement affects 2.98% of absolute coefficient mass; its added negative mass is 2.2783. This is a coefficient proxy, not measured PPO loss/gradient; current run does not record per-trajectory gradients. Report: remote `.../train/credit_concentration_audit.json`; code: [`archive/diagnostics/20260915_credit_concentration.py`](../archive/diagnostics/20260915_credit_concentration.py).

# 2026-09-15 persistent-SAAM recurrence audit（离线完成，未接入训练）

- 在已完成 binary+SAAM 的 960 条 rollout（856 条 eligible）上，用封存的 LineageReplay 统计相邻 optimizer update 的 exact `(example, canonical state, canonical full action)` 重现：0→1 为 0/1,642，1→2 为 81/1,388（5.84%），2→3 为 0/1,281。上一轮成功 key 在下一轮错误轨迹出现为 0/701、22/791（2.78%）、0/616；只忽略 state 的 action-only 上界为 0%、14.97%、0%。相邻题目集合重叠为0、18、0个，说明零重现主要受采样调度限制。
- 全四轮任意 update 至少重现一次的 strict key 为287个，102个相邻非零系数转移中有32次符号翻转（31.37%）。persistent-SAAM 假设覆盖窄且证据方向不稳定，只保留为离线小消融；报告见 [`PERSISTENT_SAAM_RECURRENCE_AUDIT_20260915.md`](../docs/reports/rl/PERSISTENT_SAAM_RECURRENCE_AUDIT_20260915.md)。

# 2026-09-15 advantage magnitude cap 静态审计（完成，未启动）

- 当前 binary+SAAM 960 条 rollout 的 reward 分布为413 correct/547 wrong；856条 eligible中413/443，104条排除。cap 不改变 reward、组构成或 eligibility，只改变 advantage 幅度。
- 轨迹系数质量：无cap绝对质量570.76；cap=2.0为556.84（25条被截断）；cap=1.5为536.81（68条，35正/33负）；cap=1.0为479.38（147条，61正/86负）。SAAM mask后 cap=1.0 的负event mass削弱约17.3%、正event mass削弱约15.8%，不能静态保证 regression减少。报告见 [`ADVANTAGE_MAGNITUDE_CAP_AUDIT_20260915.md`](../docs/reports/rl/ADVANTAGE_MAGNITUDE_CAP_AUDIT_20260915.md)。

# 2026-09-15 4B later-error-quarter 训练完成（评测待做）

- table_rl 运行根 `/home/dengyan/tabular_rl_outputs/qwen3_4b_later_error_quarter_saam60_20260915_r1` 已正常完成 `global_step=4/4`，960 条 rollout、checkpoint-1/2/3/4 和 final adapter 均存在；训练耗时 `26,093.37s`（约7小时14分53秒），GPU0/1 已释放。
- 四轮 rollout correct rate 为 `0.4125/0.4375/0.4750/0.4042`；训练中每轮 SAAM 共享错误动作抑制 `86/74/86/104` 条，correct-error positive flip `10/14/10/13` 条。当前只有训练完成和系数诊断证据，matched BIRD-dev1534 评测与 fresh replay/admission 尚未完成，不能宣称能力提升。

# 2026-09-15 4B later-error-quarter checkpoint-4 匹配评测（资源阻塞）

- BIRD-dev1534 配对评测 preflight 已通过，固定 checkpoint-4 adapter、Atomic v26/actionable-error-v1、2048 tokens、双卡端口 18366/18367，并复用同一 merged baseline（SHA-256 `740d6a723a5fcccdc456baeb2a404a7f2e37fc396de3d1dec8b99a195e69dfe2`）。
- 正式启动被资源门禁拒绝：table_rl GPU 1 由其他用户的 `web_mspg_predictor.py` 占用 936 MiB；未抢占、未开始采样，待两卡同时空闲后重试。

# 2026-09-15 4B later-error-quarter 单卡匹配评测（进行中）

- 因 GPU 1 持续被占用，改用 pinned Atomic v26 runtime 的底层 vLLM/rollout wrapper，将完整 BIRD-dev1534 题目集中到空闲 GPU 0；单独标记为 single-GPU diagnostic，不伪造双卡身份。
- 评测根 `/home/dengyan/tabular_rl_outputs/evaluations/qwen3_4b_later_error_quarter_saam60_checkpoint4_single_gpu_actionable_20260915_r1`，已完成 vLLM 启动并处理前 11/1534 条；GPU 1 外部进程未触碰，paired analysis 待完成。

# 2026-09-15 4B later-error-quarter 单卡匹配评测（完成，诊断）

- 完整 BIRD-dev1534 已完成，GPU 0 单卡，结果和配对分析落盘；GPU 1 外部进程未触碰。
- Candidate（Qwen3-4B later-error-quarter）：900/1534（58.67%）correct、1361/1534（88.72%）legal；正确的 Qwen3-4B epoch4 SFT baseline：903/1534（58.87%）correct、1372/1534（89.44%）legal。原 paired analysis 曾误引用 Qwen3-8B checkpoint-6380 baseline（928/1534、1380 legal），已改用 4B baseline 重新生成 `paired_analysis_4b_sft_baseline.json`：77 gain / 80 regression，净 -3（-0.196 pp，p=0.8732），合法净 -11（-0.717 pp，p=0.3998）。

# 2026-09-16 4B advantage-magnitude-cap c=1.0（训练完成，评测待做）

- 独立 matched ablation：Qwen3-4B checkpoint-6380、同一 60 题 cohort、binary result-only + SAAM asymmetric-error、span alpha=0.5、唯一机制变量为 advantage magnitude symmetric cap `c=1.0`，作用于 credit assignment 和 policy reduction 之后；不混入 KL 或其他 credit 变化。
- 远端根 `/home/dengyan/tabular_rl_outputs/qwen3_4b_advantage_cap1_saam60_20260915_r1` 已正常完成 `global_step=4/4`，checkpoint-1..4、final adapter、precision audit 和 immutable manifest 均存在；训练耗时 `25027.25s`（约 6 小时 57 分），GPU0/1 已释放。
- 四轮每 update 240 episodes，rollout correct rate 为 `33.75%/45.42%/41.67%/45.83%`；manifest 确认 `advantage_magnitude_cap=1.0`。日志中的 `saam/capped_*` 是 SAAM 错误 credit 的独立计数，不能替代 magnitude-cap 命中审计；需先确认 cap 按 post-reduction advantage 实际生效，再做 matched BIRD-dev1534、fresh replay 和 admission。
- 当前没有能力结论；此前条目中的 paired 数字不属于本次 cap run，不能引用。

# 2026-09-16 4B advantage-magnitude-cap c=1.0 匹配评测完成

- 使用 Atomic v26/actionable-error-v1、temperature=0、BIRD-dev1534、双卡独立模型分片完成；评测根为 `/home/dengyan/tabular_rl_outputs/evaluations/qwen3_4b_advantage_cap1_saam60_checkpoint4_actionable_20260916_r4`，两张 GPU 已由 owned-process cleanup 释放。
- Candidate 为 `886/1534` correct（57.7575%）、`1340/1534` legal（87.3533%）；candidate adapter SHA-256 为 `bf62b50d9bcb84856323e56900a087a63c756afca01dcddee1e9213b7a79d692`。
- 相对正确的 Qwen3-4B epoch4 SFT baseline `903/1534`、`1372/1534`：76 gain / 93 regression，净 `-17`（-1.108 pp，exact McNemar `p=0.2183`）；合法性 62/94，净 `-32`（-2.086 pp，`p=0.0128`）。相对 matched correctness-only SAAM `909/1534`、`1359/1534`：64/87，净 `-23`（`p=0.0730`）；合法性 70/89，净 `-19`（`p=0.1532`）。
- paired artifacts：`paired_analysis_4b_sft_baseline.json`、`paired_analysis_4b_controls.json`。fresh runtime replay/admission 尚未完成，结果暂不晋升主线；该单次结果不支持继续采用 c=1.0。
- **RFT C 档正式跑已启动 + 吞吐实测（2026-09-17）**：8B `checkpoint-6380`（base `/home/dengyan/models/Qwen3-8B-TrustSQL-baseline`，adapter sha256 `8900e4c4…`）在 table_rl GPU0/1 从 `qwen3_atomic_v26_rft_c_tier_12288_20260917`（2,361 记录 / 287 episodes，jsonl sha256 `a298d304…`）做 on-policy 自训练，`cutoff_len 12288`、LR `2e-5`、4 epochs = 592 步，run root `/home/dengyan/tabular_rl_outputs/rft_20260917/rft_c_tier_12288_4ep_b1_20260917`。**吞吐实测**：per-device batch 1 / accum 8（有效 16）峰值显存 13.9/13.2 GB、**66 s/步**；per-device batch 2 / accum 4（有效 batch 同为 16）峰值升至 19.0/21.5 GB、**87–100 s/步**，变长记录按最长 padding 反而变慢，已放弃并删除该实验根。20 步 smoke（`rft_smoke_20260917`，stage 标记为 smoke）已验证显存与 loss 正常，不作结果。已建每小时检查并推进评测的自动化 `rft-c`。
- **RFT B 档数据与配置已就绪（2026-09-17，待 C 评测后自动启动）**：为把"数据量"与"优化量"解耦为可对比的一对，B 档固定与 C 相同的阶段参数（`cutoff_len 12288`、LR `2e-5`、从 8B `checkpoint-6380` 续训），只扩大 cohort：band 1–6、每题限 2 条 → 1,033 条 verified episode（1,382 候选中 `ProtocolError` 102 / `ScalarGroundingError` 13 / `ValueError` 4 / `terminal_denotation_mismatch` 60 / `tool_output_mismatch` 2 被拒）→ ShareGPT 8,257 条记录（1,033/1,033 独立 replay 通过）→ 12288 prefix-complete 审计后 **8,197 条记录 / 1,029 episodes**（仅 60 条记录因 `incomplete_oldest_history_pair` 丢弃），`feedback_recovery_targets 184`。配置 [`bird_rft_qwen3_8b_atomic_v26_b_tier_12288_qlora.yaml`](../src/sft/configs/bird_rft_qwen3_8b_atomic_v26_b_tier_12288_qlora.yaml)，**2 epochs = 1,024 步**（约 19 h），使 B 的优化预算与 C 的 592 步同量级，从而把 C↔B 的差异归因到 episode 数（287 → 1,029，3.6×）。用户已授权 C 评测完成后自动进入 B，无需另行确认；编排由每小时自动化 `rft-c` 承担。
