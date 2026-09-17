# 决策记录

## 2026-09-17

- **流水线/评测启动检查清单入库（防止重犯，用户要求）**：把本次「训练成功但评测全灭」的三层原因固化为可执行清单，写入 `docs/current/rl_pipeline.md` 的「流水线与评测启动检查清单」，并在 `AGENTS.md` 加启动前指向：① 评测 controller 必须带 `PYTHONPATH=<project>/src`；② vLLM 必须带 `TRITON_LIBCUDA_PATH`（缺失时的两个错误签名 `Bytes object is corrupted, checksum does not match` 与 `CalledProcessError ... cuda_utils.c ... -lcuda` 是同一原因；不得用 `--enforce-eager` 绕过，会破坏 `vllm.enforce_eager` 的 matched identity）；③ 评测输出目录必须全新（launcher 对已存在目录 fail-closed，重试要换后缀）；④ 阶段之间轮询等待 GPU ≤512 MiB；⑤ 不要用会匹配自身命令行的 `pkill -f`。新增一条命令的预检 `src/rl/scenarios/diagnostics/preflight_eval_environment.sh <project_src> <gpu0> <gpu1> [port0] [port1]`（检查 triton shim、`triton_backend()`、带 PYTHONPATH 的 `import rl`、GPU 空闲、端口空闲）；`run_overnight_saam_decomposition_newgnn.sh` 已按清单补齐（PYTHONPATH、TRITON_LIBCUDA_PATH、顺序启动、拒绝复用目录）。

- **修复评测 launcher 的 triton libcuda 缺失（实现修复，已生效）**：`run_qwen3_8b_v26_checkpoint_dataparallel_table_rl.sh` 现在从 `PYTHON_BIN` 推导 `PYTHON_ENV` 并导出 `TRITON_LIBCUDA_PATH`（默认 `$PYTHON_ENV/var/triton-libcuda`，缺失则 fail-closed 退出）。此前该变量只在训练路径（`start_vllm_server.sh`）设置，评测路径缺失导致 triton 无法编译 `cuda_utils.c` shim（`-lcuda` 链接失败 / 缓存 shim 校验失败），使所有 checkpoint 评测在 vLLM engine core 初始化时崩溃。诊断反证：同环境 `torch.compile` 与 `triton_backend()` 正常；设置该变量后 vLLM 完整捕获 CUDA graph 并启动成功。该修复不改变评测 identity（`enforce_eager` 仍为 0，与 909 baseline 一致）。同时确认另外两个非代码原因：我的流水线缺 `PYTHONPATH=<project>/src`（controller 立即 ImportError），以及 launcher 对已存在输出目录 fail-closed（`resume is forbidden`）让重试秒退，已改用全新输出目录。

## 2026-09-16

- **逐题正确轨迹趋势的读法（结论：无净提升，看 unpaired 曲线会误判）**：在四个"每题被采样 2 次"的臂上做 K=8→K=8 的逐题配对：909 −0.117、cap1.0 +0.333、cap1.5 −0.167、无 span −0.050（均不显著），提升/持平/下降近似三分；churn 幅度与"每题成功率不变"的二项噪声预测（≈78.8%）一致。同时确认：unpaired 的 per-update 曲线会出现 2.8→4.2 的"上升"，但那来自"该 update 抽到的题更难/更易"（909 的 u0 子集首现均值 2.80 vs u1 子集 4.20），按子队列配对后仍 ~0。**因此后续判定学习必须用固定题集（同题重复采样或探针集），不要用未配对的分 update 曲线。** 180 题 one-pass 设计无法给出逐题趋势，下一轮若要趋势应改为 ≥2 遍或加入固定探针题。

- **【已更正】"全错组是被练坏的题吗"：结论不变，但此前给出的"+2.88/8 波次不可比"是我的口径 bug，作废**。用户指出硬件不可能决定模型性能——成立。复核发现我先前把 60 题臂"同一题跨 2 个 update 合并后的 K=16 计数"直接与 K=8 筛选相减，人为放大约 +2。按 `(update, question)` 的严格 K=8 重算后：**180 题臂**（180 组 × 8 行）在线 0/8 = 30 组，其 SFT 筛选分布为 2:12、3:9、4:7、5:1、6:1（**28/30 在 SFT 时本就 ≤4/8**）；screen vs online 配对差值 **−0.300/8（t=−1.75，n=180，不显著）**。**60 题臂**（120 组 × 8 行）在线 0/8 = 17 组，screens 2:4、3:8、4:4、5:1；配对差值 **−0.558/8（t=−2.78，n=120）**。两条臂方向**一致**（都是小幅负移，0.3–0.6/8，接近 K=8 抽样噪声），不存在"筛选波次不可比"的证据。**SAAM 无 span 臂**按同口径：120 组 × 8 行，0/8 = 25 组、8/8 = 6 组（此前记的 24/119 是过滤掉 `process_update=False` 行后的口径）。结论：0/8 组来自"SFT 时就是边际难度（2–4/8）的题在这次采样中落空"，而不是"把会做的题练坏"；唯一残留信号是全臂均值有 0.3–0.6/8 的小幅下移，需要用 fresh re-screen（同 pipeline 重筛）或 dev 评测来判定是噪声还是真实退化。

- **adaptive-K 采纳为下次实验默认，上限 32（用户指令）**：用户要求"下次实验自动进行 adaptive-K，最多采样 32 次"。实现为"首轮 K=8；只对全对/全错组按 initial-K 步长补采样，逐轮 8→16→24→32，一旦变成混合组立即停止；advantage 在最终组内计算；reward/SAAM/span/clip 不变"。实现放在共享层（`rl.frameworks.trl.rollout` + 配置/CLI/manifest 管道 + `adaptive_k/*` 指标），并已通过 CPU preflight 校验。**注意它引入的两个口径变化**：每 update 的 rollout 数不再恒等于 `prompts × group_size`（manifest 记录 `adaptive_group_size_max` 与逐 update 的 `final_group_size_*` 直方图），以及成本变为 `1 + 补采样比例`（按已测零信号率 14–20% 估计约 1.2–1.4×）。因此与固定 K=8 的历史臂比较时属于单变量（采样规则）对照，必须同时报告 `adaptive_k/*` 与实际 rollout 成本。

- **"训练侧看不出学习趋势"的正确读法（分辨力审计结论）**：`rollout/correct_rate` 是"该 update 抽到的 30 题 × K=8 的 on-policy 采样准确率"，而每个 update 换一批新题、题间正确比例从 0 到 1 全铺开（题级 sd 0.28–0.36），导致单 update 的聚类标准误约 5.9pp；180 题臂 6 点的观测 sd 只有 2.65pp（小于噪声）、趋势 −0.25pp/update（t=−0.33），80% 功效下的 MDE ≈9.0pp；4 点的 nospan 臂 MDE ≈28.5pp；8B 的 40 点序列因为每 update 只有 14 题，MDE 反而 ≈36pp。**因此该指标的 MDE（9–36pp）比我们要找的效应（1–2pp）大一个量级**：序列"平"是预期现象，既不能证明没学到、也不能证明学到。要判定学习，必须用固定探针集在每个 checkpoint 上重复测量——即 matched BIRD-dev1534 评测（已经预注册为每条臂的判据）；训练侧若要有分辨力，只能固定少量探针题长期参与抽样，或把每 update 题数从 30 显著提高。

- **两条 4B 臂正常结束，不等同于"中断"**：180 题臂（table_rl，6/6、9h28m、ckpt-6 adapter `3a48c6b8…`）与 SAAM 无 span 臂（NewGNN，4/4、7h26m、ckpt-4 adapter `8d65bded…`）都是 `trainer exit 0` + `trained_pending_audit`。这个状态串的含义是"训练完成、等待 checkpoint/有效 update 审计与评测"，launcher 会在此状态停止自己启动的 vLLM 并退出，所以 GPU 变空、进程消失是预期行为。审计 + fresh replay + matched 评测仍需显式执行，未做之前不得报告能力结论。

- **过夜流水线：拆解 SAAM×span 的 2×2（用户安排，已启动）**：用户要求"先跑 GRPO+span-balanced 对比实验，然后把两个实验结果都评测完"。设计为单条无人值守流水线（`run_overnight_saam_decomposition_newgnn.sh`）：① 训练 GRPO+span 臂（`credit_assignment=trajectory` 无 SAAM、`span_balance_alpha=0.5`、binary reward，其余与 correctness-only SAAM 臂完全一致）；② 该臂 checkpoint-4 的 matched BIRD-dev1534 评测；③ 已完成 SAAM-nospan 臂的 checkpoint-4 同协议评测；阶段③在①失败时仍执行。这样把此前 890→909 的三变量混淆（reward profile / credit assignment / span）拆成可解释的 2×2：SAAM+span（已有 909）、GRPO+span（本次）、SAAM nospan（本次）；vanilla signed-binary 保留为第四格的历史证据。评测 identity 与 909 baseline 匹配：`actionable-error-v1`、2 卡 even/odd 分片、24 workers、T=0、max_tokens 2048、max_steps 30、同一 BIRD-dev 输入（SHA `8bf5a8bf…`）。GPU6/7 专用，其余用户进程未触碰。

- **cap 的"信息"：它不是无效，而是尺度选错了（离线分析结论）**：cap 对 reward 的影响是 0，它只作用在 reduction 之后的逐 transition 系数上；但实测 c=1.0 削掉 32.0% 的信号质量、c=1.5 削掉 19.9%，而系数分布 p50=0.53、p75=1.06、p90=1.91 —— 也就是说 c=1.0/1.5 落在分布的 p73/p80，根本不是"只裁异常值"。更关键的是被裁掉的部分有明确语义：确定性 error 通道只剩 55.3%（c=1.0）/71.8%（c=1.5），且 `project`/`group_aggregate`/`join_tables` 这些决定答案的关系动作被削 37–41%，而 `describe_table`/`read_subtable` 只被削 18–19%（**选择性由 token 长度决定**）。因此这两次实验否定的是"深裁剪"，不是"variance control"。若要继续该方向，应预注册 c=3.0（只削 5.8%、error 通道保留 92.6%）作为真正的温和 cap 单变量对照；或者把 cap 改到 trajectory advantage 尺度（两者相差约 transitions/trajectories ≈ 8.6 倍）。accuracy 在 20% 质量处已饱和（885 vs 886，p=1.0），合法性却随 cap 变温和改善（1340→1362，p=0.088），说明掉分来自尾部通道而不是合法性机制。

- **数据扩展实验：180 题 cohort，沿用 909 成功臂配置（用户指令）**：目标是把"correctness-only binary + SAAM + span0.5"这套成功配置扩到 3× 数据量，观察是否还有提升。设计上刻意只改数据规模：cohort = 原 60 题（逐字保留）+ 120 题新候选（fresh 视图、2–6 带、bird-sql、按 identity 排除与 60 题重叠的 31 题、每档最大余额配额）。唯一被显式接受的次要差异是 **updates 4→6**（6×30=180 即一遍数据；若坚持 4 updates 则 120 次抽样只覆盖 ~2/3 数据，作为数据扩展测试更弱），而实现差异经 diff 核对为**纯新增**（first-error/later-error 分支），不影响 `saam-asymmetric-error` 路径。选择理由：保持 dataset（bird）、正确数带（2–6）、reward、credit、span、LR、K、prompts/update、optimizer、precision 全部与成功臂一致，使"是否提升"可以归因到数据量。备用方案（未采用）：固定算力的 4-update 版本、以及包含 spider 的混合数据集版本（table_rl 缺 SynSQL 库，混合数据集会同时改变两个变量）。

- **magnitude-cap 方向收口：c=1.5 与 c=1.0 无差异，且均低于不加 cap 的 SAAM（用户指令跑完评测后判定）**：paired 结果 cap15 885 vs cap10 886（净 -1，p=1.0；legal +22，p=0.088），cap15 vs correctness-only SAAM 909（净 -24，p=0.0498），cap15 vs 4B SFT 903（净 -18，p=0.171）。也就是说全局 magnitude cap 无论取 1.0 还是 1.5 都不能恢复准确率，只把合法性往 SFT 方向挪了一点。结合静态审计（cap 同时削弱正/负信号、无选择性）与剂量-反应（削弱错误惩罚越狠越差），**不再为该方向开新臂**；后续若要动 credit 规模，应改在组内归一化或错误惩罚的定向维度上（见同日 adaptive-K / 状态对比条目），且必须单变量。

- **cap=1.5 arm 训练完成后立即启动双卡 matched 评测（用户指令）**：table_rl 上 cap=1.5 单臂已完成 4/4（checkpoint-4 adapter `5e5e94eb…`，960 rollouts），随即在 GPU0/GPU1 以 data-parallel 分片（767+767）启动 `actionable-error-v1` BIRD-dev1534 greedy 评测，run dir `evaluations/qwen3_4b_advantage_cap15_saam60_checkpoint4_actionable_20260916_r1`，端口 18450/18451，协议身份与 cap1/SAAM/SFT 完全一致。选择该入口而非新建脚本：复用仓库既有的 `scenarios/evaluation/run_qwen3_8b_v26_checkpoint_dataparallel_table_rl.sh` 与 cap1 评测冻结的 controller 三个脚本（sha 已核对），保证与对照臂同评测实现。启动故障处理记录在 `experiments.md`（共享 torch_compile_cache 损坏 → per-run cache；Triton 需 `TRITON_LIBCUDA_PATH`）。评测完成后做 merge + paired 比较。

- **在 NewGNN 启动"SAAM 但不使用 span-balanced"对照（用户指令）**：用相同数据（`data_new60.jsonl` SHA-256 `1a6cb257…`）、相同 seed（20260914）、相同 reward/budget/优化/rollout/SDPA/4-bit/gradient-checkpointing/actor old-policy，**只把 `span_balance_alpha` 从 0.5 改为 None**，在 NewGNN GPU6（trainer）+ GPU7（vLLM 18380）运行 4 updates，运行根 `qwen3_4b_saam_nospan_newgnn_20260916_r1`。目的：在 SAAM 族内隔离 span-balanced 梯度的贡献——此前的 890→909 对比混了 reward profile、credit assignment 与 span 权重三个变量。次要差异仅 `save_steps` 10→1（不影响训练数学）。完成后对 checkpoint-4 做 matched BIRD-dev1534 greedy 评测，与 correctness-only SAAM 909 和 4B SFT 903 配对。

- **采纳 adaptive-K rollout 作为下一步信号密度手段（用户裁决，设计已定/未实现）**：规则为“首轮 K=8；仅当组为 8/8 全对或 0/8 全错时追加一轮 8 条并合并为 K=16 单组；最多一轮；混合组不扩展；advantage 在最终组内计算；reward/SAAM/span/clip 不变”。预注册见 [`ADAPTIVE_K_ROLLOUT_PREREG_20260916_ZH.md`](../docs/reports/rl/ADAPTIVE_K_ROLLOUT_PREREG_20260916_ZH.md)。依据：A100 8B cohort 中 0/8与8/8占47.6%、31.6%的transition因零优势被丢弃、40个update训练侧无移动；4B 60题cohort online重采样后仍有21.7%零信号组。实现须落在 `rl.frameworks` 共享层，并把 `manifest_trajectories == manifest_tasks * group_size` 改为记录变长 `final_group_size`。

- **组内归一化权重偏移列为需要重视的问题（用户裁决，待离线审计）**：n=8 标准归一化下 1/8 组单条正样本系数 +2.65 vs 6/8 组 +0.58；按 A100 组构成，1/2 档题目占 21% 却承担 37% 的优势质量，而新 1–6 口径把 1/8 档从被排除变为池内约 25%。下一步只读审计：在冻结 rollout 上对比“组内 std”与“固定除数”两种归一化的系数质量与分档占比；必须与已测且有害的 cap（reduction 后逐 transition 裁剪）区分开。

- **SAAM 影响力读法更正（用户指出）**：A100 运行中 SAAM 只移除约 6.3% 的优势质量，但 4B 同 cohort 对照中 correctness-only SAAM 相对 vanilla signed-binary 为 84 gain/65 regression、净 +19 题（p=0.1401）。因此不能用“移除质量占比小”判定机制影响小：SAAM 改变的是哪些 token 获得梯度，修的是系统性符号错误。后续该方向以共享 `(state, action)` 对的识别精度/覆盖率为指标，而非移除质量。

- **补录 8B 历史评测事实（只读核查）**：A100 700题运行 checkpoint-10 为 925/1534、checkpoint-30 为 920/1534，对照 8B SFT 928/1534（-3/-8），与“几乎没变”一致；该批 RL 评测为 legacy 反馈，与 SFT 的 actionable-error-v1 属跨反馈比较。checkpoint-40 的评测产物未在三台机器 `evaluations/` 命名中找到，若存在其他路径需补录。这些数字此前未进仓库文档，是本轮补录的直接原因。

- **D 验证结果：步级分岔点不可行，终止承诺不可弃（只读审计完成）**：① 用户对"找不到中间分岔点"的判断成立——一对一错配对中 56.2% 在第 1 个可执行动作即分岔、仅 4.8% 共享 ≥3 步，只有 23.6% 的混合题存在共享 ≥3 步的配对，句柄归一化后不变；因此"在同一 state 上做 action 对比"（GiGPO-lite / 分歧点偏好对）在当前语料没有供给，维持冻结。② 但承诺缺口真实：同一批候选上单样本 55.00% → 多数投票 57.23% → clean_modal 58.20%（上界 oracle 68.77%），"无 Harness error 且合法终止"在混合题内是 60.7% vs 38.5% 的强无 gold 判别信号。③ 结论：训练侧若要利用 D，应做"在自采候选集上训练选择/复核"（新机制，需登记；不得恢复已冻结的 Direct/Hybrid/iterative-SQL 入口），而不是步级偏好；推理侧部署收益仍需同题 greedy 基线才能主张。报告 [`COMMITMENT_AND_BRANCH_POINT_AUDIT_20260916_ZH.md`](../docs/reports/rl/COMMITMENT_AND_BRANCH_POINT_AUDIT_20260916_ZH.md)。

- **RL 策略改进方向（三合一审计 + 追加的分岔点审计，用户要求先审计再定方向）**：审计 4B correctness-only SAAM 冻结 rollout 后，**先提出后证伪了"按答案推导链重定向负 credit"**：首轮测得合法但错轨迹的负质量 58.5% 在答案链外，但追加的分岔点审计（941 对错误/正确轨迹按 lineage 身份逐 depth 比较）显示**第一处分岔 85.7% 在 depth 0–1、且只有 14.7% 落在错误轨迹的答案链上**——决定对错的分岔本来就在链外，因此"链外衰减"等于移走学习信号，**撤回不做**；负 credit 来源拆分为轨迹级链外 55.9%、链上 37.0%、局部 error 惩罚仅 7.1%，而此前只拿掉局部惩罚（first-error-capped）已显著 -26 题（p=0.044），进一步佐证减弱惩罚方向有害。修正后的优先级：① **同一状态、不同动作的局部对比惩罚**（供给 402 turn/4 updates，严格"正确侧≥2 条一致"137 turn）；② 早期 turn 权重（85.7% 分岔在 depth 0–1；需按新证据单独登记，此前"位置衰减"为暂停状态）；③ 采样密度（adaptive-K + 更大 cohort）；④ harness 格式兼容（已批准）；⑤ 归一化换除数（弱杠杆，不单独开臂）。报告 [`RL_IMPROVEMENT_AUDIT_20260916_ZH.md`](../docs/reports/rl/RL_IMPROVEMENT_AUDIT_20260916_ZH.md)。

- **状态级对比惩罚方案定型（已预注册，未实现）+ 位置分布口径修正**：规则为"按 `(example_index, lineage state_signature)` 分组；错误轨迹的某个 turn 若在同一状态下存在正确轨迹选择过**不同动作**，且该 turn 未被 SAAM 屏蔽，则该 turn 的负向 advantage 乘 `(1+λ)`，λ=0.5；正向 credit 与正确侧完全不动；优先级为确定性 error 覆盖 > SAAM 共享屏蔽 > 本对比放大"。供给与量级：**宽松 283 turn / 269.94 负质量（占总负质量 11.7%）；严格 g=2（正确侧 ≥2 条一致替代动作）95 turn / 98.45（4.3%）**；λ=0.5 时新增负质量约 135（5.9%）/ 49（2.1%）。此前记录的 402/137 为错误口径（把已被 SAAM 置零、质量为 0 的 turn 计入"不同动作"）。**位置分布**：SAAM 屏蔽的 353 turn 中 85% 在 depth 0–1（`describe_table` 占 63%），对比候选 62–92% 也在 depth 0–1；原因是 lineage 状态包含前序动作与观测，深度匹配要求前缀完全一致。81.6% 的负质量（1878.7）落在"状态从未被正确轨迹访问"的中后段，状态级机制对其无能为力。预注册见 [`STATE_CONTRAST_CREDIT_PREREG_20260916_ZH.md`](../docs/reports/rl/STATE_CONTRAST_CREDIT_PREREG_20260916_ZH.md)。

- **状态级对比惩罚抽样复核（用户要求；结论：默认方案不执行）**：对 283 个 contrast 候选做全量动作关系分类并抽读 80 条（严格 40 + 宽松 40，覆盖 27/26 道不同题）。分类结果：`describe_set_variation`（两侧 `describe_table` 表格集合为子集/超集）81 个（28.6%）、`projection_or_limit_only`（只差 `return_columns`/`limit`）20 个（7.1%）、`material_change` 182 个（64.3%）；其中"分支后续仍被使用"仅 80 个（28.3%）、"位于错误轨迹答案链上"仅 44 个（15.5%），其余 102 个是探索后丢弃的死分支。抽样实例：ex=850/206 是同一 filter 仅 `return_columns` 不同；ex=7557（先 join vs 先 filter）、ex=1851（先 filter Rating=5 vs 先 read）属策略差异且分支被丢弃；只有 ex=379（"RUTH 或 LINDA"被写成只过滤 RUTH）是真实语义错误，但该分支也已弃用。**严格护栏 g=2 反而更差**：严格组 62%（25/40）是表格集合差异，宽松组仅 15%——"≥2 条正确轨迹一致"筛出的是表格列举习惯而非正确做法。修正定义（排除表格集合/投影差异 + 要求分支后续被使用，可选 on-chain）后供给仅约 20 turn/update，约占总负质量 3%，只能作微调。证据见 `docs/reports/rl/rl_improvement_audit_20260916/contrast_samples.jsonl`。

- **SAAM 归因澄清（用户提问触发，配置核查）**："SAAM 相对 vanilla 净 +19"并非单变量对比：890 臂与 909 臂至少三处不同——`result_reward_profile`（`signed-binary` vs `binary`）、`credit_assignment`（`trajectory` vs `saam-asymmetric-error`）、`span_balance_alpha`（未指定 vs 0.5）。因此不能把 +19 归因给 SAAM。**隔离 SAAM 的臂（binary + trajectory 无 SAAM + span0.5）从未训练过**，而 SAAM 族内 reward 形式差异量级相当（three-level SAAM 897 vs correctness-only SAAM 909）。SAAM 已验证的机制是 sign hygiene：direct conflict mass 1356.687→0（另 1021.98→0、153.635→0），固定重现 key 的跨 update sign flip 14.26%→5.23%。结论：要回答"SAAM 为什么有效"，必须补跑隔离臂；当前只能说点估计回升且机制层面确实消除了 credit 自相矛盾。

- **训练候选区间改为1–6，区间外观察保留（用户决定，已生效）**：3000题K=8筛选完成后，用户决定后续训练只选择正确轨迹数1–6的题目；0与7/8不作训练，但必须显式保留记录以备后续用途。合并库存入口为 `data/inventory/rl_training_candidates_1to6_current.json`（strict 1209 / expanded 1329 / fresh 776），并同步重指向正式 `data/inventory/rl_training_candidates_current.json`；区间外观察写入 `retained_non_candidate_observations.jsonl`（3667条，含 `retention_reason=never_solved/always_solved/incomplete_screen`）。旧2–6与1–7库存目录不覆盖，重指向前pointer另存为 `data/inventory/rl_training_candidates_20260911/pointer_snapshot_20260916.json`。资格按观察逐条判定：同一题若旧观察在1–6内而新批次观察在区间外，仍保留为候选并同时记录两次观察。报告见 [`RL_TRAINING_CANDIDATE_POOL_1TO6_20260916.md`](../docs/reports/rl/RL_TRAINING_CANDIDATE_POOL_1TO6_20260916.md)。该库存仍非冻结cohort，启动正式RL前需按admission policy冻结manifest。

- **释放筛选GPU**：3000题批次两个shard均完成后，NewGNN GPU4/GPU6上的本任务vLLM进程（PID 321249/321250）已停止，显存回到空闲；未触碰其他用户进程。

- **注册 4B cap=1.5 matched ablation（待启动）**：基于 cap=1.0 逐轨迹审计中“全局截断同时削弱有效正向信号”的结果，启动同一 checkpoint-6380/60 题 cohort 的独立 `advantage_magnitude_cap=1.5` 对照。只改变 cap，其余 reward、SAAM、span balancing、optimizer、KL 和 runtime identity 固定；不将其视为主线候选，须完成 cap-hit、fresh replay、matched BIRD-dev1534 后再判断。

## 2026-09-16

- **Advantage cap post-reduction 逐轨迹审计（完成，未推广）**：按两个 run 各自 frozen credit 实现重建；cap=1.0 实际命中 cap run 569/858 可训练轨迹、1471 transitions（761 正/710 负），而此前 147 是 raw trajectory coefficient 口径。292 条正向命中轨迹全部最终 correct，其中261条无 Harness error；499/761 正向命中 transition 在最终答案表祖先链上，262个是 exploratory 分支。cap=1.5 为417条轨迹/848 transitions；无 cap 对照反事实为436/867。显式 error 的负向命中为75条（26条来自最终正确轨迹），不能靠全局 cap 实现更大 error penalty。报告见 [`CAP_POST_REDUCTION_AUDIT_20260916.md`](../reports/rl/CAP_POST_REDUCTION_AUDIT_20260916.md)。

## 2026-09-15

- **Vanilla 与当前最佳 binary+SAAM 的机制复盘（诊断结论，未启动新实验）**：matched BIRD-dev1534 为 vanilla 890/1534、correctness-only SAAM 909/1534；SAAM 相对 vanilla 为84 gain/65 regression、净+19，但 exact McNemar `p=0.1401`，因此只称点估计回升。vanilla 的主要问题是 signed-binary trajectory credit 将终局方向广播到共享 state/action 前缀，导致局部 credit sign churn；不是少数极端轨迹支配。历史反事实审计中 direct conflict mass 为1356.687→0，跨 update sign flip 固定key为14.26%→5.23%。SAAM 的下一优化方向是减少 strict mask 同时删除的多数方向残余（离线估计475.039），先验证 net-preserving/置信度 soft-mask，再决定是否训练；不继续堆 Harness error penalty、observation bonus 或猜测性 semantic error 惩罚。

- **P/N 解释纠正**：`P`/`N` 只是某个精确 `(state, full action)` 上正确侧正系数质量与错误侧负系数绝对值的诊断记号，不是当前 SAAM 的新决策规则。当前 asymmetric SAAM 的语义是：只要同一 key 在正确轨迹出现，就保留正确侧正 credit，并将错误轨迹上的共享负 credit 置零；不按 `P-N` 重新缩放，也不让错误侧重新获得负 credit。上一条提出的 net-preserving cancellation 不作为当前推荐方向，避免违背“成功轨迹出现过的动作至少不应被错误终局惩罚”的动机。

- **纠正后的下一步机制候选（预注册设计，待离线 gate）**：优先审计 persistent-SAAM：跨 optimizer update 保存精确 `(example, state signature, full action signature)` 的成功证据；后续同 key 在错误轨迹出现时继续只抑制错误侧负 credit、保留成功侧正 credit。必须先验证 state/action identity 不漂移、错误轨迹正 credit=0、正确轨迹负向质量不增加；通过后再与当前 batch-local SAAM 和 vanilla 做同 cohort matched 训练。不得用 P/N 净值给成功动作重新扣分。

- **persistent-SAAM 离线审查结果（未接入训练）**：同一 960 条 binary+SAAM rollout 的 exact lineage key 在相邻 update 中重现率为 0→1 `0/1642=0%`、1→2 `81/1388=5.84%`、2→3 `0/1281=0%`；上一轮成功 key 在下一轮错误轨迹出现为 `0/701`、`22/791=2.78%`、`0/616`。相邻题目重叠仅0、18、0个，且全四轮 287 个 recurrent key 中32/102个非零比较发生符号翻转，说明持久证据覆盖窄且不稳定。persistent-SAAM 不作为默认强假设；保留为后续小型离线消融。

- **跨 update 方向否决后的候选收敛（只讨论，未启动）**：后续候选限定为同一 update 内的机制或稳定化： (A) exact state 下不同 full action 的成功/失败对比，只对已观测到的分叉动作做局部 credit；(B) 保留 SAAM asymmetric 语义的 advantage variance cap，减少 K=8 组构成造成的幅度差异；(C) binary+SAAM 配低强度、独立预注册的 KL 稳定化。暂不采用跨 update ledger、全局 Harness error 加罚、dense observation bonus 或猜测性 semantic error credit；先对 A/B 做 gold-free 离线覆盖和方向门禁。

- **Advantage magnitude cap 静态结果（完成，未启动）**：cap 不改变当前 reward 分布（413/547 raw，413/443 eligible）或 eligibility。`c=1.5` 截断68/856条，`c=1.0`截断147/856条；c=1.0对SAAM后负event mass削弱17.3%、正event mass削弱15.8%，因此不能保证减少 loss/regression。若继续验证，cap必须作为单独 matched ablation预注册，不能与KL或新credit同时改变。

- **4B later-error-quarter 训练完成，评测待做**：table_rl 运行正常完成 `global_step=4/4`，960 条 rollout，训练耗时 `26,093.37s`，checkpoint-4/final 和 precision/implementation 产物存在，GPU资源已释放。四轮 correct rate 为 `41.25%/43.75%/47.50%/40.42%`；matched BIRD-dev1534、fresh replay 和 admission 尚未完成，不作能力结论。

- **GPU 机制实验纠偏**：曾误将本轮 GPU 请求收窄为 8B 性能 gate；该方案只执行了 CPU preflight，随后在用户澄清“验证 RL 机制”后取消，未进入 GPU、未产生训练结果。下一实验改回机制主线：4B `saam-later-error-quarter`。

- **4B later-error-quarter 机制对照（已注册，待启动）**：在 alpha=.25/.5/.75 离线软衰减审计后，固定 4B checkpoint-6380、同一60题 cohort、Atomic v26、binary result-only、SAAM asymmetric-error、span alpha=0.5、K=8、30题/update、4 updates、LR4e-7、KL0；唯一算法变量是首个非 timeout 确定性 Harness error 后续错误的负向 credit 为 `-0.25*max(|A|,1)`，首错、timeout、共享动作和其他规则不变。table_rl 仅提供已验证的 trainer/vLLM 分卡硬件条件；完成后做 matched BIRD-dev1534 与 fresh replay。

- **启动确认**：上述 4B quarter-credit 机制实验已通过 preflight 并在 table_rl 运行根 `/home/dengyan/tabular_rl_outputs/qwen3_4b_later_error_quarter_saam60_20260915_r1` 启动；GPU0 trainer/GPU1 vLLM 已工作，首个 update 尚未完成。

- **后续错误重复审计**：quarter 当前运行尚无可读 rollout；在最近完成的同 cohort 960 条 later-error-half 轨迹中，后续错误按 error type 重复率为 28/35=80.0%，但按完整 attempted tool+arguments 严格重复仅 2/25=8.0%（轨迹级 2/29=6.9%）。因此“后续错误”主要表现为同类错误级联，不是简单重复同一个 action；下一步恢复感知机制应优先识别错误链条和是否响应反馈，而不是只惩罚 exact action duplicate。

- **组构成与 leave-one-out 风险**：同一 completed later-error-half rollout 按题目和 update 分成120个 K=8 组：全对6组、全错20组、混合94组；7正确/1错误的组12个，1正确/7错误的组21个，共33条 minority trajectory。LOO baseline 在这些单条 minority 组中会放大少数轨迹优势，需先做离线系数审计，不能直接接入训练。

- **timeout-barrier replay audit v1 完成（CPU-only，promotion gate仍关闭）**：对 fresh replay 唯一 outcome mismatch 的 `rl_7332_sample_4` 单独做 5/10/15/30 秒重放，10 秒重复三次；同一 recorded model outputs 在目标复测中均恢复为 recorded `wrong_answer+legal`，说明先前 timeout→join cascade 是 timing-sensitive，而非稳定 Harness 语义差异。5 条已标记 timeout 的轨迹也完成 10 秒三次重复，正确性/合法性均稳定。版本化诊断报告将首个 fresh timeout 后缀标为不可重放、保留原始 outcome/credit：`timeout_barrier_replay_audit.json`，diagnostic gate=true、strict all-exact gate=false、promotion gate=false。该结果支持先做 barrier-aware 离线 credit 审计；暂不启动新的长RL。

- **4B later-error-half timeout敏感性验证（已登记，CPU-only）**：针对 fresh replay 唯一 outcome mismatch 的 `example_index=7332 / rl_7332_sample_4`，固定 recorded model outputs、任务、Atomic v26 runtime、rolling-legal-history、30 steps，仅改变 replay timeout 为 5/10/15/30 秒，并在10秒设置重复三次。记录每个 turn 的 parse、Harness error、tool output digest、最终 correctness/legal/failure_type；不读取或输出 gold SQL，不启动GPU。判据：若10秒重复结果不稳定或提高timeout后恢复 recorded outcome，归因为timeout timing sensitivity并修订replay admission；若各timeout仍稳定复现同一状态差异，则转为Harness deterministic replay bug，暂停新的RL训练。输出根为远端 later-error-half `train/timeout_sensitivity_7332_audit.json`。

## 2026-09-14

- **4B `saam-later-error-half` fresh replay admission 未通过**：训练与BIRD-dev1534匹配评测已完成（902/1534正确、1359/1534合法、1534题全覆盖），但 fresh runtime replay 在 example 7332/sample 4 的普通轨迹出现 timeout 后连锁状态差异，记录的 `wrong_answer+legal` 变成 fresh `execution_error+illegal`；execution replay 另有14条异常和100条派生 legal 不一致，虽然 correctness disagreement=0。两个 fresh gate 均为 false，当前结果只能作为诊断，不能 promotion 或迁移到8B。下一步先修复/隔离 timeout-sensitive replay 稳定性，再做预注册的 matched credit 对照；不重复当前4B训练。

- **后续错误半权重对照状态补充**：4B `saam-later-error-half` 已完成 4/4 updates，当前四轮
  rollout correct rate 为 `40.83%/42.92%/37.08%/43.75%`，不能解释为逐轮单调下降。checkpoint-4
  的 BIRD-dev1534 匹配评测第三次独立根已完成，结果902/1534正确、1359合法、1534题全覆盖；
  前两次仅为导入环境/共享编译缓存的基础设施失败，未计入模型结果。相对correctness-only SAAM
  同题为60/67（净-7，p=0.5946），相对first-error-capped为84/59（净+25，p=0.0444）。
  fresh replay 已完成但 strict gate 未通过：853 条普通记录中 852 条逐步一致，102 条截断前缀一致，
  5 条 timeout 仅 1 条稳定；有 109 个 step divergence 和 1 条 outcome mismatch，未发现身份、
  解析或 runtime transport 问题。随后生成的 admission 仍为 false：execution audit 有100个
  derived-legal disagreement、14个 replay exception，但 correctness disagreement为0；拒绝原因
  锁定为 example 7332 的 timeout-cascade；随后 timeout-barrier audit 已确认该 cascade 是 timing-
  sensitive，diagnostic gate通过但 strict/promotion gate仍关闭。下一步完成统一 paired artifact
  和 barrier-aware credit 审计，不启动新的长RL。

- **后续错误半权重对照（用户已授权）**：固定 `saam-later-error-half`，首个确定性非timeout错误
  使用全负向credit，后续错误使用0.5倍；保持timeout和其他规则。独立从4B SFT checkpoint-6380
  开始，同一60题、seed20260912、K8、30题/update、4 updates、binary result、span0.5、KL0，
  保留4-bit/SDPA/checkpointing/actor old-policy/1row-4096/CUDA Graph。先验证实现与离线alpha=.5
  数值一致，再训练并自动衔接双卡BIRD-dev1534；结果未出前不宣称改善，详见decision register。
  实现/81项本地测试/17项远端测试/数值一致性/preflight均通过；训练与评测已完成，fresh replay两个gate均未通过，当前不启动新的长RL。

- **SSH 恢复与执行约定纠正（已验证）**：09:55 同一 `table_rl` 只读命令在普通沙盒中
  exit 255，在 `exec_command(sandbox_permissions="require_escalated")` 审批通过后 exit 0。
  之前还存在独立的自动审批超时；未更改远端或凭据。已有用户授权不反复口头确认，遇到本地
  限制主动申请工具权限并按审批范围执行，不循环重试普通 SSH；更底层超时原因未知。
  详细证据和后续操作约定见 [服务器资源记录](../docs/current/server_resources.md#codex-远端连接边界)。

- **first-error-capped 对照收尾（远端核验）**：checkpoint-4 的双卡 BIRD-dev1534 已合并为
  877/1534 正确、1344 合法，replay admission gate通过但raw replay diagnostics仍有 divergence，owned 评测进程已清理。相对同题
  correctness-only SAAM 为 63 gain / 95 regression（净 -32，exact McNemar p=0.0134），
  本次结果不支持推广首错截断，不改变正式主线；见 [实验记录](experiments.md) 与评测根
  `/home/dengyan/tabular_rl_outputs/evaluations/qwen3_4b_first_error_capped_saam60_checkpoint4_graph_actionable_20260914_r2/results/merged/paired_comparison_provenance.json`。

- **后续 Harness error 软衰减离线审计（完成、无 GPU）**：同一 frozen 960 条 rollout、7,389 个
  transition 上扫描 alpha=0.25/0.5/0.75。正确轨迹负向 coefficient mass 从全惩罚的104.1416，
  变为97.3451/99.6106/101.8761；硬截断 alpha=0 为95.0796。三种软衰减均保留68个正确负向
  transition，后续错误共30个（7个最终正确、23个最终错误），错误轨迹正向 transition均为0。
  这是离线 credit 端点审计，不构成准确率结论。结果根：
  `/home/dengyan/tabular_rl_outputs/overnight_first_error_capped_20260914_r3/train/saam_soft_attenuation_audit_20260914_final.json`。
  当前建议优先用 alpha=.5 做独立4B/table_rl单臂，先核对 raw replay divergence，不与8B规模验证并行。

- **first-error replay divergence 拆解（完成、只读）**：112 个 step divergence 与 108 条外部调度边界
  前缀（107 generation_length、1 context_overflow）及 4 条 timeout timing divergence 数量完全对应；
  无 tool-output、parse、identity 或 correctness outcome mismatch。6 条 timeout-sensitive 轨迹的
  outcome/parse 均稳定。execution replay 的 9 条 exception 来自重放原始 malformed/error action 时
  重新抛出的 `ProtocolError`/`ToolExecutionTimeoutError`；100 条 derived-legal disagreement（52 条
  wrong_answer、48 条最终正确）只发生在派生 legal 重算层，action-count/correctness disagreement
  均为0。结论：admission gate 保持通过，但 raw execution replay 不能称为全清；当前没有证据表明
  轨迹级 correctness 或训练 credit 被语义重放改写，后续 GPU 实验仍应保留该审计限制。

- **静态 gate 通过后启动 first-error-capped SAAM 对照**：同一 frozen 960-rollout corpus
  的 A/B 重评分显示，首错封顶将正确轨迹负向 coefficient mass 从 104.14 降至 95.08，
  正确负向 transition 从 68 降至 61，封顶 30 个后续 deterministic error（其中 7 个为
  最终正确轨迹）；错误轨迹正向 credit 未增加。该结果只证明 credit 拓扑符合预注册方向，
  不证明 accuracy 增益。已在 table_rl 独立启动 B arm：
  `/home/dengyan/tabular_rl_outputs/overnight_first_error_capped_20260914_r3`，保持
  checkpoint/cohort/reward/rollout/optimizer 不变，仅新增 `saam-first-error-capped`；完成后
  必须通过 fresh replay 和 matched BIRD-dev1534，失败根 r1/r2 不合并。

- **下一步实验建议（待执行，先离线验证 credit，再训练）**：不立即扩大 cohort 或启动新的长 RL。先对已冻结的 960 条 Atomic v26 rollout 做无 GPU 的三臂重评分：(A) 当前 correctness-only wrong-uniform + SAAM，(B) 只对首个可验证 Harness error 做负向 credit 并封顶后续错误的 `first-error-capped`，(C) 已完成的 three-level SAAM control。固定原始 trajectory、Harness feedback、example index 和 checkpoint 身份，比较正确轨迹被错误压负、错误轨迹获得正优势、共享动作翻转、被丢弃 transition 比例，以及 credit 与最终 denotation 的一致性；不奖励 observation 频率，也不猜测合法语义错误的局部责任。
  只有 B 相对 A 明确减少正确轨迹负优势且不增加错误轨迹正优势时，才在同一 60 题、K=8、30题/update、4 updates 条件下训练一个独立 B arm；随后用同一 BIRD-dev1534 index 做 matched greedy 评测。若单次结果仍是小幅净增，则转向更大冻结 cohort/多 seed 验证，而不是继续叠加 reward 规则。完整 fresh replay 是该训练结果进入能力结论前的必需门禁。

- **correctness-only SAAM 评测完成但未通过能力准入**：checkpoint-4 的 BIRD-dev1534
  双卡 r4 评测（`actionable-error-v1`）完成 1534/1534，909 正确（59.2568%）、1359 合法
  （88.592%）。相对同题 4B SFT epoch4 为 77/71（净 +6，p=0.6812），相对 three-level
  SAAM 为 74/62（净 +12，p=0.3456），相对 vanilla 为 84/65（净 +19，p=0.1401），均未
  达到准入门禁。合法性相对 SFT 为 62/75（净 -13，p=0.3052），相对 three-level 为
  70/69（净 +1，p=1.0），相对 vanilla 为 66/76（净 -10，p=0.4502）；配对产物为
  `results/merged/paired_comparison.json`。r4 根为
  `/home/dengyan/tabular_rl_outputs/evaluations/qwen3_4b_correctness_only_saam60_checkpoint4_graph_actionable_20260914_r4`。
  前两个启动失败根和 r3 `legacy` 部分结果不计入结论。

- **correctness-only SAAM 训练完成，暂不下能力结论**：4/4 updates 已完成，实际训练耗时
  26,093.77 秒（约7小时14分53秒），相对此前重复 three-level 续跑的首 update 吞吐验证
  有效，但速度仍需绑定相同有效 transition 量比较。checkpoint-4 和 precision/implementation
  artifacts 已生成，GPU资源已清理；已补做只读 lineage/advantage audit（960 episodes、856 eligible、
  7,389 events，未读取 gold SQL），但该审计不替代完整 fresh replay。matched BIRD-dev1534
  已完成且未通过能力准入；完整 fresh replay 仍是剩余审计门禁。

- **改为推进新 reward 对照（训练已完成）**：用户确认重复已完成的 4B three-level SAAM 没有信息增益；
  correctness-only `binary + SAAM` 对照已完成 `global_step=4/4`，训练耗时 26,093.77 秒，
  checkpoint-4、precision/implementation artifacts 已生成且 GPU 已清理。同题 BIRD-dev1534
  matched 评测已经完成但增量不显著；当前只剩完整 fresh replay 审计，不据此推广算法能力。

- **完成性能门禁后恢复同 identity 4B SAAM 训练**：固定 transition 融合仍未进入正式
  trainer；为保持算法可比性，本轮采用已验证且不改变 reward/old-policy 的 CUDA Graph
  serving 路径（4-bit、SDPA、actor old-policy），并用新的隔离根续跑原 60 题、4-update
  实验。远端根为
  `/home/dengyan/tabular_rl_outputs/rl_runtime_qwen3_4b_saam_threelevel_spanbalanced60_table_rl_20260913_graph_r2`；
  训练完成前不解读能力变化，必须通过 checkpoint、precision、fresh replay 和匹配
  BIRD-dev1534 评测门禁。

按时间倒序；详细当前门禁见 [`docs/current/decision_register.md`](../docs/current/decision_register.md)。

## 2026-09-13

- **固定 transition 融合与精度门禁通过，但暂不采用**：在 `table_rl` 同一
  `example_index=7365`、1,957-token transition 上，4-bit old/current 两次 forward 的
  steady 1.8167s 与单前向1.3895s，BF16 1.5220s 与1.1659s；两者 token log-prob 完全对齐、
  梯度余弦相似度均为1.0。BF16约快16.1%，峰值显存约1.68倍。该证据只覆盖无 optimizer 的
  单 transition；融合必须通过真实 batch、checkpoint/OOM 和 manifest 门禁后才能进入正式
  trainer，当前仍保留 actor old-policy scoring和4-bit默认。

- **CUDA Graph 复测通过，仅作为性能诊断**：固定`example_index=7365`的一组4-bit online
  audit在`VLLM_ENFORCE_EAGER=0`下确认vLLM mixed prefill/decode和decode graph capture；
  8 episodes/47 transitions的rollout、old-policy、policy前反向分别为74.42/28.38/93.73s，
  step为198.55s，owned资源已释放。由于采样轨迹不同，不把它与eager总时长直接相减，也不
  改正式trainer；8B/A100仍需独立验证。

## 2026-09-12

- **4B/旧8B耗时差异与速度门禁（2026-09-13）**：4B vanilla 四步训练计算约
  29,842.95s（8h17m23s），SAAM-SDPA 约29,942.40s（8h19m02s），差0.33%，不支持把
  耗时增加归因于 SAAM 或“有效优势变多”。SAAM 拆分为 rollout 7,307.46s、old-policy
  5,154.32s、policy 前反向17,160.86s；6,682条保留 transition 和约2,317万 padded
  tokens 才是主要工作量。旧8B文档的6h05m是历史规划/运行口径，尚未取得同身份的完整
  四步实测，不能与当前8h直接作因果比较，也不能称当前已达理论上限。下一步在table_rl
  用固定 transition 做 CUDA Graph复测、4B BF16对4-bit隔离比较，以及严格条件下的
  actor old/current log-prob 单前向融合对齐测试；未通过数值/梯度对齐前不启用正式训练。

- **4B vanilla 与 SAAM 退化/回升分析（2026-09-13，已登记，未启动新训练）**：纯 GRPO 的续跑
  `train_runtime=24210.68s`只覆盖 checkpoint-1 后的3个 update；加上首个 update `5632.27s`，
  四步训练计算时间约 `29842.95s`（8小时17分23秒）。SAAM SDPA 恢复为 `29942.40s`
  （8小时19分02秒），差约100秒（0.33%），不能把此前的6小时43分与8小时19分直接比较，
  也不能把差异归因于SAAM算法本身。
- **信号解释**：vanilla 在同60题/4 update短预算下用整条trajectory的signed-binary方向，
  120个K=8组中只有93组混合、27组全同质；简单题相对SFT净损失15题而challenging题净增7题，
  支持“局部高方差更新扰动已学会行为”的解释，但McNemar `p=0.3588`，不能称显著退化。SAAM
  训练rollout相对vanilla为428/960 vs 390/960正确、414 vs 460合法错误、118 vs 110非法/失败；
  其记录的共享错误动作抑制408条、共享正确动作保留394条、correct-error正向翻转38条，说明
  SAAM确实修正了部分共享前缀的方向污染；但独立采样和评测净增7题（88/81，`p=0.6445`）均不足以
  证明能力增益。
- **错误轨迹定位**：vanilla/SAAM合法错误中只有34/460（7.4%）和42/414（10.1%）带显式
  Harness错误，其余主要是合法但语义错误；几乎所有轨迹都使用观察工具，错误轨迹平均步数仍更长，
  因而问题不是“没有观察”，而是观察后的实体/列/连接/聚合语义选择及恢复后缀。不能对所有观察或
  合法错误动作统一加减分。
- **下一步注册为离线优先的三臂验证**：固定这960条rollout先重算 (A) correctness-only
  `wrong-uniform + SAAM`，(B) A 加首个显式Harness错误封顶的 `first-error-capped`，(C) 当前
  three-level SAAM control；比较全错正优势、正确负优势、恢复轨迹负向质量和span质量后再训练。
  若A/B通过，再用更大混合cohort和更长预算验证，并预注册低强度KL/alpha=.25对照；不引入
  observation/provenance bonus、Gold-path或对合法语义错误的猜测性局部惩罚。

- **8B SDPA恢复评测完成，未达到准入增益**：`resume_step3_sdpa`已核验`global_step=4/4`、完整checkpoint-4
  和资源释放；trainer仅切换eager→SDPA，不能与原eager失败根合并。同协议`actionable-error-v1`、greedy、
  2048、BIRD-dev1534结果为931/1534，对照checkpoint-6380 SFT的928/1534仅净+3题；66 gain / 63 regression，
  exact McNemar p=0.8603。合法终止1380→1387。该结果不构成accuracy promotion，也不证明SDPA改变能力；
  评测owned-process cleanup已完成。

- **4B恢复及匹配评测完成，未达到准入增益**：`..._r2_sdpa`已核验`global_step=4/4`、960条rollout、完整
  checkpoint-4、FP32 LoRA/AdamW precision与资源释放；原eager失败根不重启。table_rl GPU0/1的
  `actionable-error-v1`、greedy、2048、BIRD-dev1534双卡评测已完成，结果897/1534、1358合法。
  对同题号4B SFT 903/1534、1372合法为77 gain / 83 regression，净-6题（-0.391 pp，exact McNemar
  p=0.6928）；对vanilla RL 890/1534、1369合法为88 gain / 81 regression，净+7题（+0.456 pp，
  exact McNemar p=0.6445）。两项均未通过能力准入，评测owned-process cleanup已完成。

- **4B 三等级 SAAM 运行 OOM 后 SDPA 恢复**：原 eager 根在首个 backward OOM，保留失败日志且不复用半成品；新根使用 SDPA、microbatch=1/4096、其余 reward/SAAM/span/data/optimizer identity 全部保持不变。恢复已完成并通过完整审计；匹配评测结果已单独登记于上条，未通过能力准入。


- **两项OOM恢复改用SDPA（进行中）**：8B checkpoint-3之后的最后update确认是单条超长
  transition绕过packing预算导致的eager backward OOM；1,847个训练transition中仅4个落在同一
  错误轨迹、长度14.3k–15.5k。4B首轮则在13,212-token正确轨迹的eager attention softmax
  申请9.29GiB时失败。两项均保留数据/reward/rollout/context/micro-batch，只切换trainer
  attention为SDPA；8B从checkpoint-3续跑，4B从头重跑。禁止把SDPA恢复结果与原失败根混合，
  必须分别审计manifest、checkpoint和BIRD-dev评测。

- **恢复证据与备用边界**：8B原恢复根保持failed和只读；新的`resume_step3_sdpa`在preflight
  中复制并锁定checkpoint-3及更新后的implementation hash。若SDPA仍OOM，先停止并保留证据，
  不自动改变context；历史审计表明`max_context_tokens=14336`只会影响失败批次中1/215条可训练
  轨迹，但任何采用该边界的续跑必须另立命名根并重新审计。

- **4B SAAM + 三等级 reward + span-balanced baseline（用户授权，待启动）**：table_rl 独立诊断，
  4B SFT epoch4 checkpoint-6380、新60题、K=8、30题/update、4 updates、KL=0。
  reward 为正确且完全合法1.25、正确但有Harness错误0.75、错误轨迹统一-1；
  `saam-asymmetric-error`、`trajectory_token_mean`、全response `span_balance_alpha=0.5`。
  输出与8B主线隔离，配置见 [`qwen3_4b_atomic_v26_saam_threelevel_spanbalanced60_table_rl.yaml`](../src/rl/configs/experiments/qwen3_4b_atomic_v26_saam_threelevel_spanbalanced60_table_rl.yaml)。


- **8B table_rl 加速验证完成（诊断）**：在双3090隔离测试 Qwen3-8B checkpoint-6380。CUDA Graph 将归一化 rollout 吞吐从 eager 91.1 提升到约136.3 response tok/s（约+50%），但有效训练量随机不同，不能比较两次总 update；sampling-score 省约84.7秒但改变 old-policy correction，仍不作默认。`max_num_batched_tokens=32768` 无可见收益；microbatch=2 backward OOM。A100因被占用未测。


- **RL性能优化纳入全项目启动记忆**：按用户要求新增
  [`rl_performance.md`](../docs/current/rl_performance.md)，AGENTS/README/最终契约/RL运行入口均指向它。
  明确8B目前仅在table_rl完成单组诊断、4B单组结果不能外推8B、sampling-score为算法ablation；
  同时恢复replicated BF16/4-bit均启用checkpointing的历史默认，避免8B意外显存回归。

- **RL update 性能优化完成第一轮 smoke**：训练入口新增显式 gradient-checkpointing 和
  attention backend 选项，manifest/preflight 会记录实际值。关闭 checkpointing 在长轨迹上
  达到23.25GiB并 OOM，故不作为默认；4-bit+checkpointing+SDPA 相对原 smoke 无稳定收益。
  将 vLLM `VLLM_ENFORCE_EAGER` 默认改为0以启用 CUDA Graph，同规模隔离 smoke 的 rollout
  约206.70→74.67秒、update 约303.55→199.43秒（阶段级约34.3%加速），未触发 OOM。该运行
  仅验证性能，不产生 accuracy 结论；后续正式 RL 需沿用新 server 默认并重新做短 update/OOM
  门禁，兼容调试时可显式设为1。

- **old-policy sampling-score 快速模式已验证（仅 ablation）**：显式启用
  `--old-policy-logprob-source sampling` 后跳过 actor old-policy forward，CUDA Graph 同组
  smoke 总时长171.86秒（rollout 74.70秒、policy 93.94秒），较 actor-score 的199.95秒再快
  14.0%，较原 retry1快43.4%。该模式令 importance ratio恒为1并改变正式 correction 目标，
  默认仍为`actor`，不得直接替换正式RL arm；manifest和训练日志会记录复用状态。

- **性能审计启动**：table_rl 的双卡评测释放后，按用户要求优先做 update 变慢的代码层诊断。
  采用独立 source copy 和外部资源清理 wrapper，不触碰已完成的4B训练、评测或共享主线实现；
  trace 记录 generation/Harness/old-policy/backward 的分段耗时、microbatch rows/padded
  tokens 和 CUDA memory。等待一组完整 smoke 后，才决定是否提出正式优化；当前不能把诊断运行
  的训练结果解释为能力变化。

- **packing 对照暂不采纳**：同一8轨迹组上将 transition packing 从1 row/4096提升到
  8 rows/8192后，microbatch 37→18，但 update 303.55s→342.73s、policy阶段74.87s→113.27s，
  峰值显存约9.97→11.77GiB。该证据否决“单纯扩大token cap即可加速”的修改；后续保持当前
  稳定配置，优先分析rollout多轮generation及policy算子实现，任何正式改动须另做matched
  smoke和OOM门禁。

- **OOM恢复门禁（第一副本已完成至checkpoint-3后失败）**：NewGNN上的c=2–6诊断在step-1
  backward发生CUDA OOM，不能用旧launcher
  直接重启。保留原运行证据和checkpoint-1，恢复副本保持trainer源码、model/cohort/reward/
  K/update/evaluation identity不变，仅将transition token上限由8192降至4096，并由独立
  lifecycle wrapper负责清理；只有两张真实空闲卡同时通过进程级门禁才允许接续。2026-09-12
  05:52复核GPU2/3无compute PID且锁可获得后，已按GPU2 trainer + GPU3 vLLM启动；18312服务
  health和TRL capability门禁通过，trainer从checkpoint-1进入`0/4`恢复进度。后续已核验
  checkpoint-2（global_step=2/4、epoch=1.0）和累计480条rollout；当前GPU2利用率为0是因为
  trainer在等待GPU3 vLLM完成下一批240条生成，GPU2仍被其进程占用约22.3GiB，不得视为空闲卡。
  当前已生成checkpoint-3（global_step=3/4、epoch=1.5），累计960条rollout；最后update再次
  在eager backward OOM，已释放GPU2/3。原根不作完成结果，SDPA续跑另立隔离根。

- **4B新60题vanilla训练与评测完成**：4/4 updates、checkpoint-4、960条rollout、
  precision和每步252层权重同步均已核验，最终adapter SHA-256为
  `e394e9d36684a85e2a078d572cd762985ef17eb7cff9f7a2fd2a90b5d5b9517c`。训练rollout正确率
  390/960只描述采样分布，不是dev结果。现以table_rl两张卡各加载一份完整4B，按767+767题
  运行actionable-error-v1 BIRD-dev1534，结果为890/1534（58.018%），baseline为同协议4B
  SFT epoch4的903/1534（58.866%）；逐题配对净-13题（79 gain、92 regression，McNemar
  p=0.3588），两张卡进程均已清理并释放。分层为challenging +7、moderate -5、simple -15，
  合法率仅下降0.20个百分点；诊断优先指向60题/4-update小预算下signed-binary trajectory
  uniform credit的高方差局部改写，暂不修改评测或服务配置。

- **RL筛选条件扩展为1–7（探索性库存）**：用户要求在当前已完成筛选证据的基础上，将候选
  条件从每题 K=8、正确轨迹数2–6放宽为1–7。按稳定`example_id`重建的版本化库存为
  strict 896题、expanded 1025题、已知排除后fresh 353题；strict正确数直方图为
  1/2/3/4/5/6/7 = 179/128/92/85/107/114/191。新库存入口为
  `data/inventory/rl_training_candidates_1to7_current.json`，详细报告见
  [`RL_TRAINING_CANDIDATE_POOL_1TO7_20260912.md`](../docs/reports/rl/RL_TRAINING_CANDIDATE_POOL_1TO7_20260912.md)。
  该范围当时是探索性数据筛选条件，正式RL主线的2–6准入合同和旧指针保留不变，不能自动替换
  正式cohort。2026-09-16起训练候选区间改为1–6，本条只作历史审计。

- **新增K=8筛选任务已启动**：已确认旧`remaining_k8_screen848`为848/848题完成，释放其遗留
  vLLM；在NewGNN GPU4/6启动隔离的3000题任务，输出根为
  `/home/dengyan/tabular_rl_outputs/evaluations/qwen3_v26_k8_screen3000_1to7_20260912`。
  两个shard各1500题，输入tasks SHA-256为
  `0b1efa4dfde66b449bf0d1eb30dbb1447ca6694d2425988ca70784a75ee8e646`；vLLM端口为8205/8206，
  启动时健康检查通过；最近复核shard-0/shard-1分别完成550/1500和450/1500题（合计1000/3000），任务仍在运行。该任务只使用checkpoint-6380/Atomic v26/K=8，
  不触碰GPU2/3上的在途RL。半小时heartbeat `table-rl` 已加入该任务的续跑、完成后释放和
  1–7库存重建逻辑。

## 2026-09-11

- **统一RL候选池并保留重叠标记**：原始700、新600、剩余848三批完整K=8筛选按稳定题目ID去重，
  strict候选545题；加入历史候选后723题。旧A100 cohort和checkpoint-6380 SFT training view的
  重叠分别为423/98，不在本次整合中静默删除；正式训练仍需按admission policy另行冻结约500题。
  复现和哈希见[`RL_TRAINING_CANDIDATE_POOL_20260911.md`](../docs/reports/rl/RL_TRAINING_CANDIDATE_POOL_20260911.md)。

- **4B新60题vanilla baseline已启动**：checkpoint-6380复筛中27/60题为
  8/8全对，只有14/60满足2–6，因此不再用旧cohort解释算法效果。从既有848题完成池中
  仅按每题K=8正确数选择60题，2至6各12题、与旧60题零重叠。先跑signed-binary vanilla
  GRPO（trajectory/uniform、4updates、LR4e-7、KL0），再决定是否恢复SAAM消融。用户指定
  本轮模型为Qwen3-4B，从4B SFT epoch4 checkpoint-6380启动；table_rl双3090分别承担trainer
  和online vLLM。step-1已完成（global_step=1，240条rollout、98正确、221条可训练），实测
  rows=1单步约94分钟；已封存后从checkpoint-1续跑rows=8但保持8192 token上限。2–6标签是8B
  筛选统计，不能表述为4B的预知正确数。见[运行报告](../docs/reports/rl/QWEN3_4B_VANILLA_GRPO_BALANCED60_20260911.md)。

- **4B vanilla step-2已完成**：`checkpoint-2` 的 `global_step=2`、`epoch=1.0` 已核验；step-2
  总计约8200秒，rows=8在8192 padded-token上限下平均1.63行/micro-batch，说明瓶颈主要是
  长上下文前后向计算。当前进入后续rollout，未发现OOM或资源越界。

- **4B vanilla step-3已完成**：`checkpoint-3` 的 `global_step=3`、`epoch=1.5` 已核验；step-3
  总计约8351秒，平均1.74行/micro-batch，显存约20GB且无OOM。当前进入最后一批rollout，
  仍需checkpoint-4和完整precision/rollout审计。

- **否决 legal-reason-only hybrid（全量配对证据）**：checkpoint-4在完全一致的
  actionable-error-v1 BIRD-dev1534协议下为902/1534，checkpoint-6380 SFT为928/1534；
  同题号49 gain / 75 loss，净-26（-1.69 pp），exact McNemar p=0.02437。合法终止仅
  1380→1378、54 gain / 56 loss，process errors还减少41，故不能把退化归因于整体工具
  合法性下降。2831个有效transition中98.02%走reason-only，但轨迹级优势并不能定位错误
  reason，且合法语义决策常由tool参数承载。停止该routing，不添加KL挽救；本结果不单独
  否决signed-binary或SAAM，先完成当前60题复筛并跑合格cohort的vanilla baseline。
  见[配对审计](../docs/reports/rl/LEGAL_REASON_HYBRID_RESTART_20260910.md)。

- **4B双卡评测拓扑纠正（已授权执行）**：两张3090分别加载完整4B、TP=1，按767+767题分片；每卡并发32，epoch3→epoch4串行。统一actionable-error-v1并对照8B epoch4，原TP=2/legacy结果仅保留诊断。见[报告](../docs/reports/sft/QWEN3_4B_CUMULATIVE_DP_EVAL_20260911.md)。

- **60题筛选来源分支逻辑纠正（用户确认，待执行）**：这里要判断的是“这60题最初由哪个
  checkpoint 做 K=8 筛选”，不是只看后续 RL 的起点。如果来源是 checkpoint-6380，必须
  用 checkpoint-6380 SFT 起点对完全相同的60题重新生成每题8条轨迹，统计每题正确数以及
  全对/混合/全错组；如果来源是 checkpoint-560，则这60题已经被旧模型筛过，直接视为
  简单题，不重复筛选。当前精确 cohort hash
  `b377ab3f7cbec8745a5342f3428435f91a77b8ada5f35aafb3465c780a07bbb4` 出现在
  checkpoint-6380 的 `smc_balanced60`、`smc_filtered2to6_60` 及 span/primary 运行中，
  而 checkpoint-560 的历史60题使用另一 cohort hash `47e9...d79d5`；因此当前60题走
  checkpoint-6380 分支。现有文件没有保存筛选阶段的逐题正确数和 decode manifest，当前
  双卡评测释放后必须补做这次精确 K=8 复筛，不能因已有 RL rollout 或训练 manifest 而跳过。

- **60题 checkpoint-6380 K=8 复筛已启动（2026-09-11）**：前置 checkpoint-4 的
  actionable-error-v1 双卡评测已完成并释放 GPU。对字节级一致的60题 cohort
  （SHA-256 `b377ab3f7cbec8745a5342f3428435f91a77b8ada5f35aafb3465c780a07bbb4`）启动
  checkpoint-6380 SFT 起点复筛；每题8条、temperature 0.8、max_tokens 2048、max_steps 30、
  Atomic v26/protocol hash `4da19387399bd3a5`，输出目录为
  `/home/dengyan/tabular_rl_outputs/evaluations/qwen3_v26_checkpoint6380_rescreen_current60_k8_20260911`。
  使用 NewGNN GPU1/3，当前只做筛选审计，不启动 RL；两个 shard 完成后按逐题正确数决定后续
  2–6题 vanilla GRPO 分支。

## 2026-09-09

- **固定 SFT 筛选、评测后分支验证（2026-09-11）**：用户确认所有 RL 题目均从
  checkpoint-6380 SFT 一次性筛选，训练期间固定 cohort，不按 epoch 重筛。当前 hybrid
  run 的 adapter_path 已核实为 checkpoint-6380，输入 cohort hash 为
  `b377ab3f7cbec8745a5342f3428435f91a77b8ada5f35aafb3465c780a07bbb4`，不是 checkpoint-560
  数据；但现有 cohort 文件未保存筛选阶段每题 K=8 正确数。当前双卡评测完成后，先补齐
  筛选证据；证据不足则对当前题池做 checkpoint-6380 SFT 起点 K=8 控制并按全对组比例
  分支：高则重新固定2–6题目先做 vanilla GRPO，低则排查未更新参数时的正确率跃升。任务
  已写入 table-rl heartbeat，评测完成前不占用 GPU。

- **筛选口径纠正（2026-09-11）**：RL 题目应从 checkpoint-6380 SFT 一次性完成 K=8
  筛选，训练过程中固定 cohort，不要求每个 epoch 重筛。当前 legal-reason hybrid 的
  960 条训练 rollout 在第一个 update 出现 17/30 个全对组，只说明新的随机 rollout
  分布，不等于筛选题目过于简单。现有60题 cohort 文件未保存筛选阶段每题正确数、筛选
  decode 配置和原始结果，故暂不能验证其是否严格满足2–6；后续先补齐 SFT 筛选审计
  manifest，再决定固定池构成，不改变“只从SFT筛选一次”的流程。

- **KL 静态门禁（2026-09-11）**：当前 legal-reason hybrid checkpoint-4 的 importance
  ratio 稳定（`log_ratio_abs_mean` 0.01880–0.01994，cap=0），但该指标不是相对
  checkpoint-6380 reference 的 KL；LoRA 相对位移仅0.1663%，且 rollout 未保存 reference
  逐token logprob。暂不据此启用KL或宣称无效；后续只允许预注册 beta、同 cohort/seed/
  reward/更新预算的独立 matched KL arm，并要求记录 masked KL、梯度量级和同题号1534评测。
  详见 [`KL_STATIC_AUDIT_LEGAL_REASON_HYBRID_20260911_ZH.md`](../docs/reports/rl/KL_STATIC_AUDIT_LEGAL_REASON_HYBRID_20260911_ZH.md)。

- **Legal reason-only hybrid NewGNN run（用户已确认，2026-09-10首启失败更正）**：加载模型不等于训练启动；远端 TRL 权重同步404已证实。改正本轮错误与上一轮恢复上下文混淆；缺失audit不静默当合法；修复canonical factory精确路径门禁，保留冻结protocol路径检查。旧binary实际+1/0，独立signed-binary落实已批准的+1/-1及timeout负惩罚，截断仍排除。60题、4updates、2048预算不变；不增加其他算法臂、不覆盖历史runtime，恢复远端授权和空闲双卡后继续。详见[续查记录](../docs/reports/rl/LEGAL_REASON_HYBRID_RESTART_20260910.md)。

- **合法动作 reason-only 混合 span 静态审计（完成，训练未启动）**：checkpoint-560 的错误轨迹中非法/未合法终止约14.1%，checkpoint-6380 三个 span 对照约4.0%–8.8%，支持“6380的失败更多是合法语义选择而非工具协议崩溃”的趋势；合法错误的决定性差异常在工具参数，故tool span置零仍只是诊断假设。用户确认新方案采用wrong-uniform/binary reward，旧four-level的全错组错误正优势和全对组正确负优势已从本次实验排除。详见[`REASON_ONLY_LEGAL_HYBRID_STATIC_AUDIT_20260910_ZH.md`](../docs/reports/rl/REASON_ONLY_LEGAL_HYBRID_STATIC_AUDIT_20260910_ZH.md)。

- **120题 cohort 输入修复并冻结（实验前置完成）**：复核发现初始 `tasks_120.jsonl` 使用字面量 `\\n` 分隔，不能作为标准 JSONL 直接训练；原文件保留，已生成并校验 `tasks_120_fixed.jsonl`，共120条，SHA-256 `4f37f5c717e534c9715af14e1da91c7f8a73081c26ed3bf7c33d4804454ba901`，同步生成 `cohort_manifest_fixed.json`。独立实验配置 `qwen3_8b_atomic_v26_wrong_uniform_saam_120_newgnn.yaml` 已部署到 NewGNN；训练仍待 preflight/启动，不使用当前1483题在途筛选。

- **修正 Qwen3-4B 对照规模**：撤回 4,471-record/560-step 的早期 SFT1 对齐解释；`checkpoint-6380` 的真实 cumulative manifest 是 25,513 records、4 epochs、6,380 optimizer steps。按用户要求停止旧规模运行并保留中止证据，改用 cumulative training view SHA `f664a7c77210f34e6f202dc21313fecd43e7044b3ce25bd5398f8691b7a491d9`、同 QLoRA/optimizer/数据轮数和 global batch，仅因 table_rl 双卡将 gradient accumulation 设为 8 以保持 effective batch 16。

- **多代理 reward 复盘与候选收敛（诊断完成，未启动）**：第一轮4个高强度只读子任务加第二轮5个独立脑暴子任务一致认为当前首要问题是 four-level/clean-secondary 的组内方向污染与错误惩罚预算，而不是缺少 dense process bonus。基于960条rollout、严格state-action覆盖0.8%–2.0%、显式错误覆盖8.67%，A `wrong-uniform + correctness-only SAAM` 确认为唯一主训练方向，C 窄范围format-only span列为必须保留但先做离线门禁；B `first-error capped SAAM`仅作A之后小消融。新增的 recovery-preservation、conservation-budget ledger、mixedness-gated weighting均不直接进入训练；GiGPO-lite/provenance/closure bonus、Gold-path、critic、树搜索、SMC top-1和位置衰减继续暂停。离线矩阵和停止线见 [`REWARD_BRAINSTORM_SYNTHESIS_20260909_ZH.md`](../docs/reports/rl/REWARD_BRAINSTORM_SYNTHESIS_20260909_ZH.md)。

- **Reward 变体首次实际重算（离线完成，未启动）**：在扩展 boundary mixed 池的1516条可用轨迹/193组上，原始four-level产生8条错误轨迹正向advantage和68条正确轨迹负向advantage；binary/correctness-only与sign-preserving均为0/0。该结果支持先修复方向性，但不代表accuracy提升；输入池按mixed outcome选择，不能作为最终效果评测。详见同一 [`REWARD_BRAINSTORM_SYNTHESIS_20260909_ZH.md`](../docs/reports/rl/REWARD_BRAINSTORM_SYNTHESIS_20260909_ZH.md)。

- **First-error capped 静态质量计算（完成，未接入）**：同一boundary池中652个确定性error events的局部负向absolute mass为944.82；只保留每条轨迹首个error后为472.91，移除49.95%，其中包含67个最终正确轨迹的后续error事件。该候选可减少恢复惩罚扩散，但可能漏掉错误轨迹中的独立后续错误，仅进入小型对照。

- **State-conditional contrast 覆盖率审计（完成，否决直接接入）**：对 first32/K8 与扩展 boundary mixed 池重跑严格 policy-state/action 审计。宽松的重复状态约覆盖20%，但同时要求同状态不同完整action、terminal结果混合、两个action各至少出现2次且成功率不同后，仅为0.80%（first32）和2.00%（boundary，233/11,645事件）。leave-one-out预测未证明动作条件值稳定优于状态均值，当前不把局部成功率差接入reward。详见 [`STATE_CONDITIONAL_CONTRAST_COVERAGE_AUDIT_20260909_ZH.md`](../docs/reports/rl/STATE_CONDITIONAL_CONTRAST_COVERAGE_AUDIT_20260909_ZH.md)。

- **不推进闭包bonus，转向行为证据（诊断结论）**：现有结构图不足以判定增量奖励价值，不因高覆盖继续加bonus；reward hacking属于设计风险，尚非该未上线bonus的观测事实。观察呈前重后轻但并非固定两段，正确轨迹40.51%在操作后再次观察；不新增按位置奖惩，不宣称已证明最短路径。详见 [时序审计](../docs/reports/rl/OBSERVATION_TOOL_POSITION_AUDIT_20260909_ZH.md)。

- **图外不等于无效（离线证据更新）**：80个有终点的成功图外操作中46个提供后续literal所需信息，14个为已记录路径的重复/冗余；不将这些抽样比例外推为6,992个图外动作的全量语义精度。观察/操作工具名不足以决定是否需要信息依赖；重复计算还可能是“第一次图外、第二次图内”，故不能仅依闭包决定奖惩。保持当前算法不变。详见 [报告及逐条定位](../docs/reports/rl/HARNESS_OUTSIDE_OPERATION_AUDIT_20260909_ZH.md)。

- **闭包不等于 bonus（诊断结论，未改算法）**：观察类保留基础结果/SAAM信号，不强求artifact图；操作类闭包只证明结果依赖，不证明额外奖励价值。正确操作候选覆盖85.673%，67.894%的正确轨迹全部操作命中；既有50动作人工分层仍包含表示/可选动作，literal还会漏标有用操作。暂不加入统一闭包bonus；若验证须隔离图选点与单纯操作加权的效果。详见 [覆盖率更正及效用分析](../docs/reports/rl/HARNESS_CAUSAL_CLOSURE_AUDIT_20260909_ZH.md)。

- **纯最终格式错误逐条人工审计（诊断完成，未接入训练）**：逐条核对当前158条合法错误轨迹，确认33条为纯format-only、118条为语义/结果错误、7条未完成terminal。以33条真实format-only为分母，包含关系识别的precision/recall均为100%（当前语料内）；此前的20.9%/24.4%只是错误轨迹占比，已纠正。`project/answer`不进入core reward，格式错误只惩罚最后format action。详见 [`FORMAT_ERROR_MANUAL_EXHAUSTIVE_AUDIT_20260909_ZH.md`](../docs/reports/rl/FORMAT_ERROR_MANUAL_EXHAUSTIVE_AUDIT_20260909_ZH.md)。
- **Project位置边界修正（诊断完成，未接入训练）**：960条rollout共有437个`project`，281个紧邻answer、156个后接语义动作；正确轨迹严格结果链中的352个project又分为238个终端前和114个中间project。33条format-only中仅9条是最终project，其余24条是最终filter/top的`return_columns`差异；后者不能惩罚整个核心action，只能做token/span级format惩罚，否则保持neutral。详见同一 [`FORMAT_ERROR_MANUAL_EXHAUSTIVE_AUDIT_20260909_ZH.md`](../docs/reports/rl/FORMAT_ERROR_MANUAL_EXHAUSTIVE_AUDIT_20260909_ZH.md)。

- **格式错误包含关系审计（诊断完成，未接入训练）**：在新版60题/K=8 rollout中，158条合法错误里有135条同题存在正确reference且有terminal可比较；其中33条满足“错误答案表包含正确答案表、行数相同”的高置信格式候选。20.9%/24.4%只是候选占全部/可比较错误的比例，不是format recall；真实format-only分母尚未完成独立人工标注。`project/answer`从核心路径剥离，条件/排序动作只把`return_columns`视为format。详见 [`FORMAT_ERROR_CONTAINMENT_AUDIT_20260909_ZH.md`](../docs/reports/rl/FORMAT_ERROR_CONTAINMENT_AUDIT_20260909_ZH.md)。

- **Process credit 人工 precision 审计（诊断完成，未接入训练）**：对新版60题/K=8的严格终点data/value闭包候选分层抽样50个动作，41个是改变题目语义的core动作（82%），6个是答案列/形状的projection，3个是可省略或冗余动作；若把projection算作任务相关信号则为94%。另发现`read_subtable`观测值落成普通literal时会被闭包漏掉，故不能把结构覆盖率直接当semantic recall。详见 [`PROCESS_CREDIT_MANUAL_PRECISION_AUDIT_20260909_ZH.md`](../docs/reports/rl/PROCESS_CREDIT_MANUAL_PRECISION_AUDIT_20260909_ZH.md)。不改变当前reward，不启动RL。

- **Process credit 静态覆盖率审计（完成，未接入训练）**：新版60题/K=8的960条rollout上，正确轨迹的终端data/value结构链覆盖4494个tool action中的2365个（52.63%）；把grounding并入候选为3439个（76.52%），但不等价于应奖励。错误轨迹173条中仅15条有明确Harness error（8.67%），其余158条是合法语义错误，不能可靠定位额外惩罚。显式table/value结构引用在当前审计内全部解析到已知source/producer，但没有独立semantic ground truth，不能宣称semantic precision/recall；详见 [`PROCESS_CREDIT_STATIC_COVERAGE_AUDIT_20260909_ZH.md`](../docs/reports/rl/PROCESS_CREDIT_STATIC_COVERAGE_AUDIT_20260909_ZH.md)。

- **评测反馈框架收敛（已确认）**：后续新启动评测统一使用`actionable-error-v1`，`legacy`只保留历史审计；当前在途checkpoint-3的legacy评测不强行中断。定时任务已取消空卡后自动启动旧SAAM/SMC RL的规则，待`wrong-uniform-four-level`离线统计并重新确认后再启动训练。

- **SAAM主方向收敛（已确认，待离线登记）**：保留SAAM asymmetric-error、four-level框架和reason/tool span-balanced，摒弃SMC；所有incorrect轨迹统一负向信号，不因无Harness过程错误而减罚。推荐最小profile为`wrong-uniform-four-level`（correct-clean +1.5、correct-error +1.0、incorrect -1.0），先离线统计再启动训练，不叠加semantic distance、gold-path、critic或树搜索。

- **Harness causal closure 的 schema 版本（否决）与弱替代（待验证）**：新增 `source_ref` 会改变 action schema/调用分布，需要重新训练，且 Harness 不能证明模型据此决策，因此不加字段、不改 prompt。允许离线验证不改 schema 的 `conservative-read-support`：只保留 `read_subtable` 唯一值、后续成功、terminal artifact closure 且不在题目文本中的候选；现有数据覆盖 759/8480 条轨迹，其中正确 533/5105。它是弱支持候选，不是确定因果，先做 reward simulation 和人工精度审计。

- **Action-level fixed-suffix replay扩展审计（诊断完成）**：新增共享 `rl.runtime.action_counterfactual` 与薄 CLI，固定原始 step/value_ref/句柄编号，做 baseline fidelity、数据库只读 SHA 门禁和逐轨迹预算。A100 rollout 中确定性抽取30个混合 K=8 题组的正负配对，共60条；51条完整、9条截断不纳入分支统计。383个有效删除分支中，正确原轨迹115个保持终局、90个出现新结构执行错误、27个终端证据句柄消失；错误原轨迹69个仍错、61个出现新结构执行错误、21个终端证据句柄消失；没有wrong-to-correct。结论仅支持固定后缀结构必要性，不支持policy-level credit，不接入reward。详见 [`ACTION_COUNTERFACTUAL_AUDIT_20260909_FINAL_ZH.md`](../docs/reports/rl/ACTION_COUNTERFACTUAL_AUDIT_20260909_FINAL_ZH.md)。

- 用户批准用官方DeepSeek先验证完整轨迹的离散过程分类/依赖识别：不做程序化闭包；timeout明确负惩罚，类别倍率预注册，不让教师输出连续分数。只恢复至多16次事后审计API请求，不恢复SFT生成、不启动RL；小样本复核后再决定扩大。见 [门禁](../docs/current/decision_register.md)。
- 外部教师pilot实际收到22个请求（停止时已发出的并发请求继续完成，超过预登记16的过程控制缺陷已记录）；10个通过规则门禁，12个被输出耗尽或终局/归因一致性门禁拒绝。暂不把教师接入reward；先增加本地终局事实锁定和单一first-error约束。
- 当前table_rl 960条rollout的静态依赖审计显示：6487条合法action中，3371个派生产出对应的3890个table input和509个value input全部能解析到已知source/producer；15条不可解析action和10条无terminal均由截断/执行/参数错误造成。旧unresolved主要是普通literal的观测grounding、schema/control edge和终端causal slice，不是原表/派生表句柄解析失败。详见 [依赖审计](../docs/reports/rl/PROCESS_CREDIT_DEPENDENCY_PARSE_AUDIT_20260909.md)。
- action-level counterfactual 静态/人工审计完成：固定后缀 skip replay 在抽查的 6 条完整 legal trajectory 上与原始终局 6/6 一致；但其结论只能表示 `necessity_for_suffix`，不能直接表示任务级因果，因为删除 observation 后没有重新生成策略后缀。error action 已有 state hash/attempted arguments，可作为独立 regression node，但当前 normalized legal-step replay 尚未保留为统一 intervention API。详见 [审计报告](../docs/reports/rl/ACTION_LEVEL_COUNTERFACTUAL_STATIC_MANUAL_AUDIT_20260909_ZH.md)。

- 用户确认table_rl空闲后自动执行correctness-primary-clean-secondary（0.25）+four-level+SAAM+span0.5的4-update诊断，先核验首个有效update，再于正常训练完成后自动双卡评测并释放本任务vLLM。已启动，不重复起任务；不提前宣称效果提升。实际运行参数/偏差及衔接门禁见 [运行记录](../docs/reports/rl/CORRECTNESS_PRIMARY_TABLE_RL_20260909.md)。
- 复盘纠正：上述 correctness-primary、全同质组零结果优势、SAAM 共享错误动作抑制和局部 Harness-error 惩罚已经在本次实现中生效，不再作为“新优化”重复提出。960 条训练 rollout 的 120 个组中 63 组全对、2 组全错，仅 55 组提供正负结果对比；下一步优先隔离信号覆盖率和 span 影响，不先改 reward 语义。当前 r6 评测仍未完成，部分结果不作准确率结论。

## 2026-09-08

- 三组 span 轨迹审计完成：OFF（关闭 span）933/1534，alpha=0.5 为 923/1534，alpha=0.25 为 935/1534。A025 相对 OFF 仅净增 2 题（70 gains/68 regressions），相对 A05 净增 12 题（66/54）；当前不足以证明 alpha=0.25 稳定优于 OFF。原因证据仍指向 tool span 对错误/恢复轨迹的同步放大，以及 all-correct 组内 `+1.0` 轨迹的负向 clean-secondary 信号；详见 [`SAAM_SPAN_THREE_ARM_AUDIT_20260908_ZH.md`](../docs/reports/rl/SAAM_SPAN_THREE_ARM_AUDIT_20260908_ZH.md)。在固定 rollout counterfactual 前，不改变正式主线配置。
- 上述“下一步提案”已被 2026-09-09 的 correctness-primary 诊断实际覆盖；不再把它当作未启动方案或重复实验。
- 用户批准实现 `actionable-error-v1`（失败 action + 可见 Harness 元数据），拒绝嵌套 value/join 自动兼容；以 checkpoint-6380 做 dev1534 greedy 回归。用户已确认本次固定2048 tokens，只评新反馈，与历史924/1534基线比较；冻结 runtime 隔离加载，不提前改正式 RL 默认。详见 [`decision_register.md`](../docs/current/decision_register.md)。

- 继续以 Atomic v26、`checkpoint-6380`、four-level reward、SAAM asymmetric-error、reason/tool 各 0.5 为唯一正式 RL 入口；约 500 题 cohort 冻结和 matched BIRD-dev 评测前，不宣称 accuracy promotion。
- 单轮 rollout 预算统一为 `max_new_tokens=4096`；历史运行保留原 manifest 值。
- 任何正式实验启动前，必须先列出配置及哈希、模型/checkpoint、cohort/数据哈希、protocol/prompt/runtime identity、reward/credit、rollout、optimizer、GPU 拓扑、输出目录和评测计划，并获得用户明确确认；未确认不得启动。
- 先做 a100/table_rl K=8 rollout 的静态语义审计；distance/format 仅作诊断，不能进入 reward、actor 或自动筛题。
- 批准 table_rl 的 `span_balance_alpha=0.25` 独立对照；不改变正式 alpha=0.5，使用相同 checkpoint/cohort/runtime 并完成同机 BIRD-dev1534 评测。

## 2026-09-07

- SMC 概率审计修复后要求每个 4-update run 落盘 120 个题组；新版 cohort 在完整审计前不作 SMC 优化结论。
- Harness-conditioned SMDP/IQL 先做离线可识别性审计；未同时观察到精确 state 的 action 和 outcome 变化前，不训练 actor、不替换 SAAM/GRPO。

## 2026-09-06

- A100 采用一张卡 replicated BF16 trainer 加另一张卡 online vLLM，只使用 GPU 0–3；3090 仅作为隔离的降级入口。
- 新 RL cohort 目标约 500 题，每题 K=8、正确轨迹数 2–6、每 update 30 题，从 `checkpoint-6380` 启动。

## 2026-08-29

- 采用 `saam-asymmetric-error` credit 和 four-level result reward；timeout 按 policy error 处理。历史 binary-only、tool-only、PCGrad 等方案冻结为诊断。
# 2026-09-15 GPU arm: 8B table_rl single-update gate

- 用户要求立即启动 GPU 验证。因 table_rl 远端未找到已冻结的 120 题文件，本次不使用 120 题配置，改用已冻结 8B gate60 cohort（60 条，SHA-256 `47e9d369bf6506d13b2432ac92bb971bf52912cb759c133846801f5411ad79d5`）。
- 固定 checkpoint-6380（adapter SHA-256 `8900e4c482f4624ff05059a6fecc2811fa99552cddea7550d44ce8cde853abe6`），Atomic v26、binary result、SAAM asymmetric-error、K=8、30 prompts/update；table_rl 双 3090 采用 4-bit、gradient checkpointing、SDPA、old-policy actor、micro-batch 1/4096 transition tokens、KL=0，先跑 1 update 作为 GPU/显存/吞吐 gate。
- 这是降级硬件上的隔离诊断，不替换 A100 正式主线，也不据此宣称 accuracy promotion；完成后再决定是否扩展到 4 updates 和 matched BIRD-dev1534 评测。

# 2026-09-15 credit concentration interpretation

- For the active 4B quarter-credit arm, report trajectory-normalized coefficient mass separately from actual PPO loss/gradient. The first complete update has 5 effective extreme-minority trajectories (>=1% share: 5; >=2%: 2; >=5%: 0) and 5 multi-error trajectories, with no single multi-error trajectory reaching 1% share. Direct Harness error credit is replacement (`-max(|A|,1)` / half / quarter), not additive stacking; trajectory-token-mean bounds per-trajectory coefficient mass. Do not call this proof of gradient dominance without per-trajectory gradient instrumentation.
- **4B/8B baseline 身份纠正及后续 Harness error 软衰减结论（2026-09-15）**：`928/1534` 是 **Qwen3-8B checkpoint-6380 SFT** baseline；**Qwen3-4B epoch4 SFT** baseline 是 `903/1534`（合法 `1372/1534`）。因此 4B later-error-quarter 的正确配对结果是 `900/1534`，相对 4B SFT 为 77 gain / 80 regression、净 `-3`（-0.196 pp，exact McNemar `p=0.8732`），合法净 `-11`（-0.717 pp，`p=0.3998`）。不能把 4B 结果与 8B 的 928 直接比较。将原始 4B SAAM、half、quarter、hard-truncation 的点估计约记为 `909/902/900/877`；这只是跨运行趋势，不构成严格单调定律，但没有可靠增益，因此不将后续错误软衰减引入主线或迁移到 8B，保留原始 SAAM 作为控制。
- **Advantage magnitude cap c=1.0 已启动（2026-09-15）**：针对已完成静态审计的唯一预注册 matched ablation，固定 Qwen3-4B checkpoint-6380、同一60题 cohort、binary result-only、SAAM asymmetric-error、span alpha=0.5、KL=0；仅在 credit assignment 和 policy reduction 后对 advantage 做对称 `[-1,1]` 截断。table_rl 根为 `/home/dengyan/tabular_rl_outputs/qwen3_4b_advantage_cap1_saam60_20260915_r1`，preflight 通过，GPU0 trainer/GPU1 vLLM 已启动；完成 matched BIRD-dev1534 与 replay/audit 后再判断。
- **Advantage magnitude cap c=1.0 训练完成（2026-09-16，评测待做）**：同一远端根已完成 `4/4` update，耗时 `25027.25s`，checkpoint-1..4 和 final adapter 均存在，GPU 已释放。四轮 rollout correct rate 为 `33.75%/45.42%/41.67%/45.83%`。manifest 确认 cap=1.0；训练日志未提供独立 magnitude-cap 命中计数，`saam/capped_*` 不可用于替代该审计。下一步先核对 cap 是否按 post-reduction advantage 实际截断，再做 matched BIRD-dev1534 和 fresh replay；在此之前不作能力结论。
- **Advantage magnitude cap c=1.0 匹配评测完成（2026-09-16，未准入）**：双卡 BIRD-dev1534/actionable-error-v1 完成，candidate `886/1534` correct、`1340/1534` legal。相对正确的 4B SFT `903/1534` 为 76/93、净 `-17`（`p=0.2183`），合法性 62/94、净 `-32`（`p=0.0128`）；相对 matched correctness-only SAAM `909/1534` 为 64/87、净 `-23`（`p=0.0730`），合法性 70/89、净 `-19`（`p=0.1532`）。paired 根为 `/home/dengyan/tabular_rl_outputs/evaluations/qwen3_4b_advantage_cap1_saam60_checkpoint4_actionable_20260916_r4`；fresh replay/admission 尚未完成。该结果不支持把 c=1.0 引入主线。
- **RFT（on-policy 自训练）C 档正式臂注册（2026-09-17，预注册）**：动机来自已测事实——checkpoint-6380 自身 K=8 筛选中 pass@8 oracle `68.8%` 而单样本仅 `55.0%`，即"正确轨迹在支持集里、但不在模式上"；而 4B 上五种 credit 变体全部落在 1.5pp 内、8B update-40 相对 SFT 为 -3/-8 题，说明在线 RL 的 credit assignment 不是当前瓶颈。假设：把**同一模型自己采到并被 Harness 判正确**的轨迹做离线 on-policy 自训练，可以直接把支持集里的正确轨迹推成高概率模式。**判据（预注册，先于启动）**：matched BIRD-dev1534 greedy，对照 **8B checkpoint-6380 SFT = 928/1534（60.495%）**；单 seed 按题配对 MDE ≈ 2.1pp，即需要 **≥32 题净增益**才算收益；副判据为 T=0.8 K=4 采样平均（本阶段在采样分布上优化）。**数据**：`data/rl_inputs/qwen3_v26_rft_20260917/`，`qwen3_atomic_v26_rft_c_tier_12288_20260917` = 2,361 action-level 记录 / 287 episodes（源自 289 条 verified episode、353 个 band-1–3 候选、每题限 1 条），`protocol_hash 4da19387399bd3a5`，与 checkpoint-6380 的 SFT 训练集 episode **零重叠**（屏幕 input manifest 记 `candidate_pool_overlap_with_sft_training_episode_ids: 0`）。**显式偏离 checkpoint-6380 schedule 的两项**：`cutoff_len 12288`（6400 下 prefix-complete 只存活 141/289 episode；精确 LF tokenizer 审计在 12288 下保留 2361/2397 记录、287/289 episode）与 `learning_rate 2e-5`（1e-4 会覆盖已有最优）。**启动前已排除**：`replay_execute_failed` 42 条（ProtocolError 33 / ScalarGroundingError 9）、`terminal_denotation_mismatch` 21 条、`ValueError` 1 条，共 64 个候选不作为训练数据。配置 [`bird_rft_qwen3_8b_atomic_v26_c_tier_12288_qlora.yaml`](../src/sft/configs/bird_rft_qwen3_8b_atomic_v26_c_tier_12288_qlora.yaml)，launcher [`train_rft_qwen3_8b_atomic_v26_table_rl.sh`](../src/sft/train_rft_qwen3_8b_atomic_v26_table_rl.sh)。**4 epochs**（592 步，smoke 实测 ~66 s/步、显存峰值 13.9/13.2 GB，预估 ~10.9 h）。未通过 fresh replay 与 matched 评测前不得宣称任何提升。
