# 当前决策登记表

更新时间：2026-09-14。这里把已确认事项和仍需用户拍板的事项分开，避免文档再出现多套
“当前方案”。用户回复后，只修改本表和 `final_project_contract.md`，再同步对应 manifest。

## 已确认

### 2026-09-16 4B SAAM 数据扩展臂（180 题）已启动

**训练侧曲线分辨力（同日离线审计，读曲线的纪律）**：`rollout/correct_rate` 是"该 update 抽到的
30 题 × K=8 的 on-policy 采样准确率"，不是固定题集上的学习曲线。按题聚类后：单 update 的
题级 sd 0.28–0.36（每题正确比例 min 0.00 / max 1.00），聚类 SE ≈5.9pp；180 题臂 6 个点的
观测 sd 只有 2.65pp（低于噪声）、趋势 −0.25pp/update（t=−0.33），**80% 功效 MDE ≈9.0pp**；
4 点的 SAAM 无 span 臂 MDE ≈28.5pp；8B 的 40 点序列因每 update 仅 14 题，MDE ≈36.1pp。
即训练侧指标的 MDE（9–36pp）比目标效应（1–2pp）大一个量级，**"看不到趋势"只能读作不能检出，
不能读作没有学习**。判定学习必须依赖固定题集的 matched BIRD-dev1534 评测。

**2026-09-17 状态：训练已正常完成（6/6），等评测。** `trainer exit 0`，`train_runtime=34,076s`
（9h28m），epoch 1.0，1440 rollouts（6×240），checkpoint-1..6 + final，precision audit 全 FP32
（33,030,144 trainable params、1,008 moment tensors），checkpoint-6 adapter SHA-256
`3a48c6b89a4a59cb25d830b752f6ea6c9f5f9e54442cd6e032337f12b965919f`。训练侧
`rollout/correct_rate` 逐步 0.458/0.450/0.513/0.433/0.417/0.463（无趋势），
`nonzero_advantage_fraction` 0.66–0.92。**status 为 `trained_pending_audit` 是 launcher 的设计
终态**（trainer 退出后自动停掉自身 vLLM 并释放 GPU），不是异常中断；checkpoint/effective-update
审计、fresh replay 与 matched 评测尚未执行，故仍不得报告能力结论。

用户要求把当前的 909 成功配置扩到更大训练数据（180 题）并观察是否进一步提升。已构造 cohort 并
启动训练：

- cohort：`data/rl_inputs/qwen3_v26_saam180_20260916/data_new180.table_rl.jsonl`（180 题），
  SHA-256 `d55ccccdf05b55cf597b1dba29e16040f5da218e95d45bae88305953a9416432`。组成 = 原 60 题
  （成功臂 cohort，逐字保留）+ 120 题新候选（fresh 视图、correct_count 2–6、dataset=bird-sql、
  剔除与 60 题重叠的 31 题；每档配额 32/21/18/23/26）。选择清单见同目录 `manifest.table_rl.json`，
  构造入口 `src/rl/scenarios/data/build_saam180_cohort.py`（单测 4 passed）。
- 训练：config `qwen3_4b_atomic_v26_correctness_only_saam180_table_rl.yaml`，launcher
  `src/rl/scenarios/diagnostics/run_qwen3_4b_saam180_table_rl.sh`，run root
  `/home/dengyan/tabular_rl_outputs/qwen3_4b_correctness_only_saam180_table_rl_20260916_r1`，
  table_rl GPU0 trainer / GPU1 vLLM（18382/51482）。manifest 与成功臂一致（seed 20260914、K=8、
  30 题/update、binary、`saam-asymmetric-error`、`span_balance_alpha=0.5`、LR 4e-7、
  error_penalty 1.0、protocol hash `4da19387399bd3a5`、initial adapter `0bc7b8b6…`）。
- **显式记录的两处差异**：① `optimizer.steps` 4→6（6×30=180 题一遍；成功臂为 60 题两遍），
  ② 实现为 Sep-15 project（与成功臂冻结快照的差异经 diff 核对为纯新增的 first-error/later-error
  分支及其指标，`saam-asymmetric-error` 路径行为不变）。
- 完成后必须用同一协议（greedy、BIRD-dev1534、`actionable-error-v1`、1534 全覆盖）对
  checkpoint-6 做评测，并与 909（SAAM60）/903（4B SFT）配对；未完成前不得声称数据扩展带来提升。

### 2026-09-16 4B advantage-cap c=1.5 训练完成并已启动双卡评测

**cap 影响量级分析（离线，同日）**：cap 对 reward 的影响为 **0**——reward 值、组构成、eligible
判定与 SAAM mask 都不变，cap 只作用在 credit assignment + `trajectory_token_mean` reduction
**之后**的逐 transition 系数。在 cap=1.0 那次的 858 条可训练轨迹/7415 个 transition 上测量
（未裁剪系数的分布 `p50=0.531, p75=1.060, p90=1.913, p95=2.679, p99=4.464, max=13.13`，正负
均值 0.848 vs 0.847）：

| cap | 被裁 transition | 被裁轨迹 | 削掉总质量 | 正/负 | error 通道保留 |
|---:|---:|---:|---:|---:|---:|
| 1.0 | 1471 | 569（66.3%） | **32.0%** | 16.0% / 15.9% | **55.3%** |
| 1.5 | 848 | 417（48.6%） | **19.9%** | 9.8% / 10.1% | 71.8% |
| 2.0 | 517 | 301 | 12.9% | 6.4% / 6.5% | 82.4% |
| 3.0 | 210 | 148 | 5.8% | 2.9% / 2.9% | 92.6% |
| 5.0 | 41 | 36 | 1.6% | 0.9% / 0.7% | 98.7% |

结论：① c=1.0/1.5 落在系数分布的 p73/p80，属重裁剪而非"只裁异常值"；② 被裁质量有明确语义——
已验证 error 通道损失 44.7%/28.2%，且 `project`(41%)/`group_aggregate`(40%)/`join_tables`(37%)
等决定答案的关系动作被削最多，而 `describe_table`(19%)/`read_subtable`(18%) 最少，**选择性由
序列化 token 长度决定**；③ accuracy 在约 20% 质量处饱和（cap1.5 885 vs cap1.0 886，p=1.0），
合法性却随 cap 变温和改善（1340→1362，p=0.088），说明掉分来自尾部通道而不是合法性机制。
因此若要继续该方向，应预注册 **c=3.0**（只削 5.8%、error 通道保留 92.6%）作为真正的温和 cap
对照，或把 cap 移到 trajectory advantage 尺度（两者相差约 `transitions/trajectories`≈8.6 倍）。
脚本 `archive/diagnostics/20260916_cap_scale_analysis.py`。

**2026-09-16 评测与配对分析已完成（结论）**：四臂 matched BIRD-dev1534 greedy（`actionable-error-v1`、
temperature 0、max_tokens 2048、1534/1534 全覆盖）结果如下：

| arm | correct/1534 | accuracy | legal | mean steps |
|---|---:|---:|---:|---:|
| 4B SFT epoch4（sft4b） | 903 | 58.87% | 1372 | 7.223 |
| correctness-only SAAM（saam4b，span .5） | **909** | **59.26%** | 1359 | 7.077 |
| advantage cap=1.0（cap10） | 886 | 57.76% | 1340 | 7.188 |
| advantage cap=1.5（cap15） | 885 | 57.69% | 1362 | 7.165 |

配对比较：**cap15 vs cap10 净 -1（82 gain/83 regression，exact McNemar p=1.0；legal 净 +22，p=0.088）**；
cap15 vs saam4b 净 -24（57/81，**p=0.0498**）；cap15 vs sft4b 净 -18（68/86，p=0.171，legal -10）；
cap10 vs sft4b 净 -17（76/93，p=0.218，legal -32，p=0.0128）。结论：**magnitude cap 在 1.0 与 1.5 两档
都不能恢复准确率，c=1.5 只把合法性向 SFT 挪了一点；该方向不再开新臂。** 该臂 fresh replay/admission
门禁未跑，不能作为能力准入结论。paired 产物：
`evaluations/qwen3_4b_advantage_cap15_saam60_checkpoint4_actionable_20260916_r1/paired_analysis_cap15_vs_controls.json`。

cap=1.5 单臂在 table_rl 完成 `4/4` updates（960 rollouts），run root
`/home/dengyan/tabular_rl_outputs/qwen3_4b_advantage_cap15_saam60_20260916_r1`，状态
`trained_pending_audit`；checkpoint-4 adapter SHA-256 `5e5e94eb…`。身份与 cap1/SAAM 臂一致
（同一 60 题 cohort `1a6cb257…`、K=8、30 prompts/update、binary、`saam-asymmetric-error`、
`span_balance_alpha=0.5`、LR 4e-7、error_penalty 1.0、protocol hash `4da19387399bd3a5`），
唯一变量是 `advantage_magnitude_cap=1.5`。

已在 table_rl GPU0/GPU1 启动双卡 data-parallel matched 评测：run dir
`/home/dengyan/tabular_rl_outputs/evaluations/qwen3_4b_advantage_cap15_saam60_checkpoint4_actionable_20260916_r1`，
端口 18450/18451，767+767 分片，BIRD-dev 1534 派生输入哈希 `636e096b…`，temperature 0、
max_tokens 2048、max_steps 30、n_samples 1、`error_feedback_version=actionable-error-v1`。
该 run 尚未 merge，也未做 paired 分析，因此**不得报告准确率结论**；完成后需与 4B SFT 903、
correctness-only SAAM 909、cap1.0 886 做 exact McNemar 配对比较。

### 2026-09-16 过夜流水线：GRPO+span 训练与两臂评测（NewGNN）

用户安排“先运行 GRPO+span-balanced 对比实验，然后完成两个实验结果的评测”。已启动单条无人值守流水线`src/rl/scenarios/diagnostics/run_overnight_saam_decomposition_newgnn.sh`：

1. 训练`qwen3_4b_v26_correctness_only_grpo_span60_newgnn`：`credit_assignment=trajectory`（SAAM关闭）、`span_balance_alpha=0.5`、binary reward；cohort/seed/reward/budget/优化/rollout/SDPA/4-bit/gradient-checkpointing/actor old-policy与correctness-only SAAM臂完全一致（manifest已核验cohort `1a6cb257…`、seed 20260914、LR 4e-7、protocol hash `4da19387399bd3a5`）。运行根`qwen3_4b_grpo_span_newgnn_20260916_r1`，launcher`src/rl/scenarios/diagnostics/run_qwen3_4b_grpo_span_newgnn.sh`，配置`qwen3_4b_atomic_v26_correctness_only_grpo_span60_newgnn.yaml`。
2. 该臂checkpoint-4的matched BIRD-dev1534 greedy评测，identity与909 baseline一致（`actionable-error-v1`、2卡even/odd分片、24 workers、T=0、max_tokens 2048、max_steps 30、输入SHA `8bf5a8bf…`），输出`evaluations/qwen3_4b_grpo_span60_checkpoint4_actionable_20260916_newgnn`。
3. 已完成臂`qwen3_4b_saam_nospan_newgnn_20260916_r1`的checkpoint-4同协议评测，输出`evaluations/qwen3_4b_saam_nospan60_checkpoint4_actionable_20260916_newgnn`；阶段3在阶段1失败时仍会执行。

该流水线把此前890→909的三变量混淆拆成SAAM×span的2×2（SAAM+span=已测909、GRPO+span=本次、SAAM nospan=本次；vanilla signed-binary为第四格历史证据）。资源只用NewGNN GPU6/7，端口18381/51481（训练）与18382/18383（评测）；日志与状态`/home/dengyan/tabular_rl_outputs/overnight_saam_decomposition_20260916/`。预计约10.5小时完成。

### 2026-09-16 NewGNN 4B "SAAM 无 span-balanced" 对照已启动

用户要求在 NewGNN 上用相同数据起一个"只用 SAAM、不使用 span-balanced"的实验，观察训练后的能力
变化。已启动，运行根
`/home/dengyan/tabular_rl_outputs/qwen3_4b_saam_nospan_newgnn_20260916_r1`；配置
`qwen3_4b_atomic_v26_correctness_only_saam60_nospan_newgnn.yaml`，launcher
`src/rl/scenarios/diagnostics/run_qwen3_4b_saam_nospan_newgnn.sh`。与 table_rl 的
correctness-only SAAM 臂逐字段一致（cohort SHA `1a6cb257…`、seed 20260914、K=8、30
prompts/update、binary、`saam-asymmetric-error`、`error_penalty=1.0`、LR 4e-7、clip 0.2、
SDPA、4-bit、gradient checkpointing、`old-policy-logprob-source=actor`、protocol hash
`4da19387399bd3a5`），**唯一机制差异是 `span_balance_alpha` 0.5→None**（另有 `save_steps`
10→1，只影响 checkpoint 落盘，不影响训练数学）。资源为 NewGNN GPU6=trainer、GPU7=vLLM
（18380/51480），只使用这两张空闲卡；preflight 通过。训练 4 updates，估计约 7 小时，完成后对
checkpoint-4 做 matched BIRD-dev1534 greedy 评测并与 correctness-only SAAM（909）和 4B SFT
（903）配对比较。该臂不改变 Atomic v26 contract。

### 2026-09-16 8B历史评测补录、信号密度诊断与adaptive-K采纳

**补录的8B历史评测（此前未进仓库文档）**：A100 700题/14题每update运行
`qwen3_8b_atomic_v26_saam_fourlevel_batch14_700_single_gpu_a100_20260904_save10_keepall_r1`
在table_rl上有matched BIRD-dev1534 greedy评测：checkpoint-10为925/1534、checkpoint-30为
920/1534，对照8B checkpoint-6380 SFT的928/1534，即-3/-8题，与“性能几乎没变”一致。评测根
为`table_rl:/home/dengyan/tabular_rl_outputs/evaluations/qwen3_8b_atomic_v26_checkpoint10_dataparallel_table_rl_20260904_r3`
与`..._checkpoint30_dataparallel_table_rl_20260905_r1`。身份提醒：该批RL评测使用legacy反馈，
SFT 928使用actionable-error-v1，严格属跨反馈比较；checkpoint-40的评测产物未在三台机器的
`evaluations/`命名中找到，若存在其他路径需补录。

**同一次运行的训练侧诊断（本地`data/inventory/a100/update40/`）**：40个update、700题覆盖一遍
（0.8 epoch，14题/update）后，`rollout/correct_rate`均值0.481、单步sd0.126，前5步0.580→后5
步0.536；entropy 0.253→0.249；`grad_norm`均值0.043；`trainable_token_fraction`均值7.7%；
`zero_advantage_transitions_dropped`累计31.6%。组构成：0/8与8/8合计333/700（47.6%）恒零
优势；按2–6口径仅251题合格，按1–6口径320题。结论：该run的约束是信号密度与体量，不是
credit规则；RL在8B上“没动”，而不是“变差”。

**用户裁决（2026-09-16）**：

1. **采纳 adaptive-K rollout（A）**：规则冻结为“首轮K=8；若组为8/8全对或0/8全错，追加
   一轮8条并合并为K=16单组；混合组不扩展；advantage在最终组内计算；reward/
   SAAM/span/clip不变”。预注册见
   [`ADAPTIVE_K_ROLLOUT_PREREG_20260916_ZH.md`](../reports/rl/ADAPTIVE_K_ROLLOUT_PREREG_20260916_ZH.md)。
   当前**未实现、未启动**；实现须落在`rl.frameworks`共享层，并修改
   `manifest_trajectories == manifest_tasks * group_size`这一在变长K下失效的identity断言。
  **2026-09-17 更新（用户指令）**：上限改为**32/题**（8→16→24→32 逐轮补采样，混合即停），并
  **已在共享层实现**：`rl.frameworks.trl.rollout`（`adaptive_extension_counts()` + 多轮
  `collect()`/`_collect_pass()`、`RolloutSettings.adaptive_group_size_max`）、
  `--adaptive-group-size-max` 与 `run_manifest.adaptive_group_size_max`、
  `rollout.adaptive_group_size_max` 配置校验、`adaptive_k/*` 训练指标。校验：本地 compile、远端
  helper/设置断言、以及 180 题新配置（6 updates、adaptive 32）的 CPU preflight 全过。隔离实现根为
  table_rl `qwen3_4b_saam_adaptive_k_20260917_r1/project`；配置
  `qwen3_4b_atomic_v26_correctness_only_saam180_adaptive_k_table_rl.yaml`。**尚未启动 GPU run**；
  启动后每 update 的 rollout 数不再恒等于 `prompts × group_size`，须以 `adaptive_k/*` 与成本倍率
  一并报告。数据侧参考值：在线 0/8 组比例为 180 题臂 29/179（16.2%）、60 题臂 17/120（14.2%）、
  SAAM 无 span 臂 24/119（20.2%）。
2. **组内归一化的权重偏移（B）已审计完成：实测为弱杠杆，不单独开臂**（报告见
   [`RL_IMPROVEMENT_AUDIT_20260916_ZH.md`](../reports/rl/RL_IMPROVEMENT_AUDIT_20260916_ZH.md)）。
   在4B correctness-only SAAM的冻结rollout上，把除数从组内population std换成常数后，各难度
   档的质量占比几乎不变（1档13.3%→10.5%、3档16.5%→18.2%、7档8.3%→6.2%），总质量
   570.8→252.6（×0.44），本质是全局缩放（AdamW近似吸收），档间相对权重变化≤1.5×。此前
   “1/8组+2.65 vs 6/8组+0.58”的4.6×偏斜来自**均值中心化本身**而非std，换除数不解决，只能靠
   更多样本（adaptive-K）或组级优势上限缓解。cap（reduction后逐transition裁剪）仍然是另一件
   已证伪的事，不可与归一化混为一谈。
3. **C（SAAM覆盖面）的读法更正**：A100运行中SAAM只移除约6.3%的优势质量
   （`saam/ambiguous_state_action_groups`均值16.5/update），但4B同cohort对照里
   correctness-only SAAM相对vanilla signed binary为84 gain/65 regression、净+19题
   （p=0.1401）。因此“移除质量占比小”不等于影响小：SAAM改变的是**哪些token获得梯度**，
   修的是系统性符号错误而非信号体量。后续若继续该方向，应以“共享(state, action)对识别
   的precision/coverage”为指标，而不是以移除质量为指标。
4. **D（终止承诺/自一致性）验证完成，结论分两半**（报告见
   [`COMMITMENT_AND_BRANCH_POINT_AUDIT_20260916_ZH.md`](../reports/rl/COMMITMENT_AND_BRANCH_POINT_AUDIT_20260916_ZH.md)）：
   ① **步级分岔点确实不可行**（用户判断成立）：一对"一对一错"轨迹里56.2%在第1个可执行
   动作就不同，只有4.8%共享≥3步；只有222/941=23.6%的混合题存在共享≥3步的一对一错配对；
   把派生表句柄归一化后分布完全不变，说明不是编号假象。但99.3%的题里8条轨迹共享第一个
   *工具*，即"计划几乎一样、落点立刻不同"，所以这是参数层而非计划层的分歧。
   ② **终止承诺的可达空间真实存在**：同一批候选上，单样本55.00%、多数投票57.23%、
   干净轨迹优先56.63%、**干净子集内投票(clean_modal)58.20%**、oracle上界68.77%（2063/3000）；
   "无Harness error且合法终止"是强无gold信号（混合题内干净样本正确率60.7% vs 非干净38.5%）。
   93.5%的题至少有一条干净轨迹，但只有3.7%的题恰好只有一条，故有效形式是"干净子集内投票"
   而非"唯一干净"硬规则。
   ③ 剩余测量项：本文55.00%是temperature0.8的sample 0，不是greedy；要主张推理侧部署收益，
   必须在同一批题上补一次greedy（每题1条、T=0），再比较greedy/单样本/多数投票/clean_modal。
   在补齐前，D只登记为**已验证存在可达空间、尚不可部署**；训练侧若要使用，应做成"在自采
   候选集上训练选择/复核"的新机制（需登记），不得恢复已冻结的Direct/Hybrid/iterative-SQL入口。

上述四项均不修改当前Atomic v26 contract、reward、工具集或Harness；adaptive-K在实现前
不得用于任何正式cohort。

### 2026-09-16 RL策略改进审计（归一化/credit落点/lineage覆盖）与下一步方向

用户要求先跑只读审计再决定下一步实验方向。审计对象为4B `correctness-only binary + SAAM`运行
`rl_runtime_qwen3_4b_correctness_only_saam60_table_rl_20260913_graph_r1`（960轨迹/4 updates），
使用该run自己的frozen实现重建，CPU-only、未读gold、未占GPU；报告
[`RL_IMPROVEMENT_AUDIT_20260916_ZH.md`](../reports/rl/RL_IMPROVEMENT_AUDIT_20260916_ZH.md)，
脚本`archive/diagnostics/20260916_rl_improvement_audit.py`。

三项结果：

1. **credit落点（最大发现）**：负系数质量按角色分布为earlier_action 70.0%、last_relational
   14.1%、terminal 13.5%、plan 2.3%；按“是否在最终答案的lineage推导链上”拆分，**合法但错**
   轨迹的负质量只有41.5%在链上、**58.5%在链外**（schema探查/读取/废弃分支），带Harness error的
   错误轨迹更差（on-chain 20.2%）。即近六成负向信号施加在不可能直接决定答案的token上。
2. **lineage状态覆盖**：错误轨迹turn中21.9%处于“同组正确轨迹也到过的状态”，其中12.0%动作也
   相同（=SAAM当前屏蔽目标）、**9.9%动作不同**（同一状态、正确轨迹选了别的动作的决定性分岔，
   当前完全未使用）；77.9%的状态从未被正确轨迹访问，无对比信号可用。因身份为lineage解析，
   该覆盖率比按文本严格匹配的0.8–2.0%高约5–10倍。
3. **归一化**：见上一节第2条，弱杠杆。

**追加审计（同日）证伪了“链外衰减”这条思路**：对941对“错误轨迹 vs 同组正确轨迹”按lineage身份
逐depth定位第一处分岔，结果是depth 0占39.9%、depth 1占45.8%（合计85.7%），而**只有14.7%的
首处分岔落在错误轨迹的答案推导链上**，15.0%在正确轨迹链上。即决定最终对错的分岔85%发生在
答案链之外（主要是depth 0–1的早期探查）。因此§B的“58.5%负质量在链外=信号稀释”这一解释
不成立：那些位置正是轨迹真正分岔之处，减弱它们等于移走学习信号。负credit来源拆分为
轨迹级链外55.9%、轨迹级链上37.0%、局部error惩罚（必然链外）仅7.1%——而此前4B
first-error-capped只拿掉那7.1%量级的局部惩罚就已显著掉-26题（p=0.044）。

修正后的下一步候选与优先级（尚未启动任何run）：① 同一状态、不同动作的局部对比惩罚
（宽松供给**283** turn/4 updates、携带269.94负质量=总负质量11.7%，严格“正确侧≥2条一致”
**95** turn、98.45=4.3%；λ=0.5乘法放大时新增负质量约5.9%/2.1%。注：此前写402/137是错误
口径，把已被SAAM置零的turn计入“不同动作”，质量不受影响但turn数偏大119）；位置分布上
SAAM屏蔽的353 turn有85%在depth 0–1（`describe_table`占63%），对比候选62–92%也在depth 0–1，
即两类机制都只能作用于前2步；81.6%的负质量落在“状态从未被正确轨迹访问”的中后段，无状态级
信号可用。预注册见
[`STATE_CONTRAST_CREDIT_PREREG_20260916_ZH.md`](../reports/rl/STATE_CONTRAST_CREDIT_PREREG_20260916_ZH.md)，
状态为**已登记、未实现、未启动**。用户要求抽样复核"这些动作是否真的是错的"，复核结果是
**默认方案（严格g=2）不建议执行**：全量283个候选中，81个（28.6%）只是`describe_table`的表格
集合子集/超集差异、20个（7.1%）只差`return_columns/limit`、102个（36.0%）是探索后丢弃的
死分支；只有**80个（28%，≈20/update）**同时满足"语义确实不同且分支后续仍在使用"，其中位于
错误轨迹答案链上的仅44个（≈11/update，约占总负质量3%）。更关键的是严格护栏g=2**反而更差**：
抽样中严格组62%（25/40）是表格集合差异（因为多数正确轨迹恰好描述了同一组表），宽松组仅15%，
即"≥2条正确轨迹一致"筛出的常是"表格列举习惯"而非正确做法。修正后的候选定义应为：排除
表格集合与投影差异、要求错误侧动作产出在后续被使用、可选再加on-chain，并把护栏改为"正确侧
替代动作在语义上也不同"。按此定义供给仅约20 turn/update，预期效应很小，只能作微调不能作
主杠杆。抽样证据：`docs/reports/rl/rl_improvement_audit_20260916/contrast_samples.jsonl`。

**SAAM归因澄清（同日核查，用户提问触发）**：此前记录的"SAAM相对vanilla为84 gain/65
regression、净+19"并不是单变量对比。比对两个配置文件后确认，被比较的890臂与909臂至少有三处
不同：`result_reward_profile`（`signed-binary` vs `binary`）、`credit_assignment`
（`trajectory` vs `saam-asymmetric-error`）、以及`span_balance_alpha`（未指定 vs 0.5）。因此
不能把+19归因给SAAM；其中可能包含reward profile（去掉signed-binary的符号广播）与span权重的
贡献。截至目前**"binary + trajectory（无SAAM）+ span0.5"这一隔离SAAM的臂从未训练过**，而
SAAM族内的reward形式差异本身就量级相当（three-level SAAM 897 vs correctness-only SAAM 909）。
SAAM自身已验证的机制是sign hygiene而非错误识别：反事实审计显示direct conflict mass
1356.687→0（另有batch 1021.98→0、153.635→0），固定重现key的跨update sign flip
14.26%→5.23%。要回答"SAAM为什么有效"必须补跑缺的隔离臂。

② 早期turn权重（把credit集中
到depth 0–2，依据是85.7%分岔在depth 0–1；注意此前“位置衰减”处于暂停状态，须按新证据单独
登记）；③ 采样密度（已采纳的adaptive-K+更大cohort，决定①②能否被测量）；④ harness格式兼容
（用户已批准）；⑤ 归一化换除数（弱杠杆，不单独开臂）。任何臂仍须遵守单变量对照、预注册门禁
与fresh replay。

### 2026-09-16 3000题K=8筛选完成与候选区间改为1–6

NewGNN GPU4/GPU6 上的3000题K=8筛选已全部完成：两个1500题shard分别在
`2026-09-15 23:49`和`2026-09-16 09:01`（CST）结束，3000/3000题、每题8条轨迹，无缺题；
shard pass@8 分别为0.698/0.677。任务完成后本任务vLLM进程已停止，GPU4/GPU6已释放。
正确数直方图为`0:937, 1:168, 2:102, 3:83, 4:87, 5:109, 6:142, 7:250, 8:1122`。

用户同时决定：后续训练候选区间由2–6改为**1–6**；区间外的0与7/8观察不作训练，但必须显式
保留记录以备后续用途。该决定取代2026-09-12条目中“正式RL主线仍为2–6”的表述，并已同步
`final_project_contract.md`、`data_generation.md`与`AGENTS.md`。

合并后的正式候选库存为strict 1209题、expanded 1329题、known-exclusion fresh 776题；
strict直方图`1/2/3/4/5/6`为299/194/146/144/189/237。入口为
`data/inventory/rl_training_candidates_1to6_current.json`，并同步重指向
`data/inventory/rl_training_candidates_current.json`；旧2–6库存、1–7探索库存保持原样，重
指向前的pointer内容另存为
`data/inventory/rl_training_candidates_20260911/pointer_snapshot_20260916.json`。区间外
观察记录在`retained_non_candidate_observations.jsonl`（3667条）；报告为
`docs/reports/rl/RL_TRAINING_CANDIDATE_POOL_1TO6_20260916.md`。

这仍只是候选池而非冻结cohort：正式RL启动前必须按admission policy选scope、应用排除策略、
完成fresh replay/manual audit并冻结独立cohort manifest，不能宣称accuracy提升。另有100题
在本批次中`legal_samples=0`且correct全为0，属于Harness合法性异常，需单独审计后再决定用途。

### 2026-09-16 Advantage magnitude cap c=1.0 评测结论

- 4B cap=1.0 单臂已完成 `4/4` update 和 matched BIRD-dev1534；实现审计确认截断顺序为 `policy_reduction_advantages()` 之后的对称 `[-1,1]` clamp，manifest 锁定 `advantage_magnitude_cap=1.0`。训练 rollout 未保存独立 cap-hit 计数，`saam/capped_*` 仅是 SAAM 错误 credit 计数，不能替代该数值审计。
- Candidate 为 `886/1534` correct、`1340/1534` legal。相对正确的 4B epoch4 SFT `903/1534`，paired 为 76 gain / 93 regression，净 `-17`（`p=0.2183`）；合法性 62/94，净 `-32`（`p=0.0128`）。相对 matched correctness-only SAAM `909/1534`，净 `-23`（`p=0.0730`）；合法性净 `-19`（`p=0.1532`）。
- 结论：单次 matched 结果不支持把 c=1.0 引入主线或迁移到 8B；fresh replay/admission 尚未完成，根目录 `/home/dengyan/tabular_rl_outputs/evaluations/qwen3_4b_advantage_cap1_saam60_checkpoint4_actionable_20260916_r4` 仅保留为隔离诊断证据。

### 2026-09-15 后续验证

- **GPU 机制实验纠偏**：曾误将本轮 GPU 请求收窄为 8B 性能 gate；该方案只执行了 CPU preflight，随后在用户澄清当前目标是 RL 机制验证后取消，未进入 GPU、未产生训练结果。

- **4B later-error-quarter（当前待启动）**：注册 alpha=.25 后续确定性 Harness error 软衰减单臂。固定 4B checkpoint-6380、同一60题 cohort、Atomic v26、binary result-only、SAAM asymmetric-error、span alpha=0.5、K=8、30题/update、4 updates、LR4e-7、KL0；唯一算法变量是首错之后的后续错误负向 credit `-0.25*max(|A|,1)`，首错、timeout、共享动作和其他 credit 规则不变。table_rl 使用 trainer/vLLM 分卡、4-bit、gradient checkpointing、SDPA、actor、micro-batch 1/4096。训练后做 matched BIRD-dev1534 和 fresh replay，结果未通过门禁前不得推广。

- **启动状态**：已通过远端 preflight，实验已在 table_rl 运行根 `/home/dengyan/tabular_rl_outputs/qwen3_4b_later_error_quarter_saam60_20260915_r1` 启动；GPU0 trainer/GPU1 vLLM 均已进入工作态，checkpoint-1 尚未生成。

- **timeout-barrier replay audit v1 完成（CPU-only，promotion gate仍关闭）**：对 fresh replay 唯一 outcome mismatch 的 `rl_7332_sample_4` 单独做 5/10/15/30 秒重放，10 秒重复三次；同一 recorded model outputs 在目标复测中均恢复为 recorded `wrong_answer+legal`，说明先前 timeout→join cascade 是 timing-sensitive，而非稳定 Harness 语义差异。5 条已标记 timeout 的轨迹也完成 10 秒三次重复，正确性/合法性均稳定。版本化诊断报告将首个 fresh timeout 后缀标为不可重放、保留原始 outcome/credit：`timeout_barrier_replay_audit.json`，diagnostic gate=true、strict all-exact gate=false、promotion gate=false。该结果支持先做 barrier-aware 离线 credit 审计；暂不启动新的长RL。

- **4B later-error-half timeout敏感性验证（已登记，CPU-only）**：针对 fresh replay 唯一 outcome mismatch 的 `example_index=7332 / rl_7332_sample_4`，固定 recorded model outputs、任务、Atomic v26 runtime、rolling-legal-history、30 steps，仅改变 replay timeout 为 5/10/15/30 秒，并在10秒设置重复三次。记录每个 turn 的 parse、Harness error、tool output digest、最终 correctness/legal/failure_type；不读取或输出 gold SQL，不启动GPU。判据：若10秒重复结果不稳定或提高timeout后恢复 recorded outcome，归因为timeout timing sensitivity并修订replay admission；若各timeout仍稳定复现同一状态差异，则转为Harness deterministic replay bug，暂停新的RL训练。输出根为远端 later-error-half `train/timeout_sensitivity_7332_audit.json`。


### 2026-09-14 状态补充

4B `saam-later-error-half` 已完成 `4/4` updates 并生成 checkpoint-4；训练 rollout 的
`correct_rate` 为 `0.408333/0.429167/0.370833/0.437500`，不是单调下降曲线。checkpoint-4
的 actionable-error-v1、BIRD-dev1534 双卡评测前两次分别因 Python 导入路径和共享编译缓存
损坏退出，均未触碰训练产物；第三次独立评测已完成：902/1534 正确、1359/1534 合法，1534
题全覆盖，GPU 已清理。fresh replay 已完成但 admission gate 未通过，详见下文。

同题配对结果：相对 correctness-only SAAM 为60 gain / 67 regression，净-7题（-0.456 pp，
exact McNemar p=0.5946）；相对 first-error-capped 为84/59，净+25题（+1.630 pp，p=0.0444）；
相对 vanilla 为75/63，净+12题（+0.782 pp，p=0.3491）；相对 three-level 为71/66，净+5题
（+0.326 pp，p=0.7327）。相对同一4B SFT epoch4总数为902 vs 903，正式配对显著性仍待统一
paired artifact；这些结果暂不支持继续扩大 alpha 扫描或宣称能力提升。

fresh replay 已完成但未通过 `fresh_exact_runtime_replay_gate_v1`：960/960 条记录的任务身份、
Atomic v26 模块路径和解析均一致，无 replay exception 或 runtime transport failure；853 条普通
轨迹中 852 条逐步一致，102 条 generation-truncated 前缀一致，5 条 timeout 中仅 1 条满足稳定
负向条件。共 109 个 step divergence、1 条 outcome mismatch（example 7332，重放出现新的
`condition_filter` timeout 及后续 `join_tables` execution error），因此严格 raw-exact gate
未通过；随后完成的 timeout-barrier audit 确认该 cascade 是 timing-sensitive，非稳定 Harness
语义差异，并保留原始 outcome/credit。该诊断 gate 已通过，但 strict all-exact 与 promotion
gate 仍关闭；该差异不应直接解释为模型能力退化或提升。

### 2026-09-14 后续错误半权重 SAAM（训练、评测、fresh replay完成；admission未通过）

用户授权继续下一步实验，固定使用 `saam-later-error-half`：首个非 timeout 确定性 Harness
错误保留 `-max(|A|,1)`，同一轨迹后续确定性错误使用 `-0.5*max(|A|,1)`；不是只处理相邻错误。
timeout、共享动作处理和其他动作的轨迹优势继承匹配对照的实际实现，不同时修订 reward。
从原始 Qwen3-4B SFT checkpoint-6380 独立启动，保持同一60题 cohort、binary result-only、
span_balance_alpha=0.5、seed=20260912、K=8、30题/update、4 updates、LR4e-7、KL0。
沿用 table_rl 单卡4-bit/SDPA trainer + 单卡 online vLLM、gradient checkpointing、actor old-policy、
microbatch 1/4096、CUDA Graph。训练结束后对 checkpoint-4 做双卡独立模型分片的
actionable-error-v1 BIRD-dev1534 matched greedy评测，并补 fresh replay；不根据中途结果回挑alpha。该门禁现已完成但未通过：example 7332/sample 4 出现 timeout 后连锁执行差异，两个 fresh gate 均为 false。
离线审计显示alpha=.5保留全部68个正确负向transition，正确负向coefficient mass为99.6106，
错误轨迹正向transition为0；这不预先证明准确率/合法率改善，二者由训练后的同题配对判断。
该授权为独立4B诊断，不改变8B主线。原始replay审计限制保留，不能把admission通过写成全量无差异。
已通过81项本地测试、17项远端SAAM测试与CPU preflight；实际训练handler在冻结960条rollout上
与离线alpha=.5完全一致（68个正确负向transition、mass=99.610618272092、无错误轨迹正向credit）。
模型/cohort/runtime/采样/执行参数与原控制一致，仅5个共享实现文件发生credit接入变更。
新根 `/home/dengyan/tabular_rl_outputs/qwen3_4b_later_error_half_saam60_20260914_r1` 的
`launch_preflight_receipt.json` 为启动检查证据。两次远端启动命令均在自动权限审批阶段超时，
工具返回CreateProcess拒绝，命令未执行；第三次受控启动已通过并完成训练与第三次独立评测。
训练约7小时14分48秒，评测根为
`/home/dengyan/tabular_rl_outputs/evaluations/qwen3_4b_later_error_half_saam60_checkpoint4_actionable_20260914_r3`。
fresh replay 与 timeout-barrier 诊断已完成，但 strict all-exact/promotion gate 仍关闭；在
统一 paired artifact 和 barrier-aware credit 审计完成前不注册下一条长RL。

### 2026-09-14 first-error-capped SAAM（训练、评测与 fresh replay 完成，未准入）

对 correctness-only SAAM 的 frozen 960 条 rollout 做 gold-free 静态重评分。当前 A 的正确
轨迹负向 coefficient mass 为 104.14，B 首个非 timeout deterministic Harness error 封顶为
95.08；正确负向 transition 为 68→61，封顶 30 个后续 error transition（其中 7 个最终正确），
错误轨迹正向 credit 未增加，静态 gate 通过。B arm 在 `table_rl` 独立根
`/home/dengyan/tabular_rl_outputs/overnight_first_error_capped_20260914_r3` 完成训练，使用
同一 checkpoint-6380、60题 cohort、K=8、4 updates、4-bit/SDPA、actor old-policy、KL=0和
CUDA Graph；唯一算法变化是 `credit_assignment=saam-first-error-capped`。训练 checkpoint-4
adapter SHA-256 为 `5730d15dd91500f096a5343ec15b054d776cde586ea1474f968dae48d99490f2`。

对应 `actionable-error-v1` BIRD-dev1534 评测根为
`/home/dengyan/tabular_rl_outputs/evaluations/qwen3_4b_first_error_capped_saam60_checkpoint4_graph_actionable_20260914_r2`；
双卡 767/767 分片已完成，合并结果为 877/1534 正确（57.1708%）、1344/1534 合法（87.6141%）。
`merged/paired_comparison_provenance.json` 确认五臂均覆盖1534个唯一 example index，question/db_id/split
零错配，config/runtime/protocol/feedback/decode匹配；`fresh_runtime_replay_admission` gate 为 true，
但底层 `fresh_runtime_replay_audit` 仍记录112个 step divergence、4个 timeout timing divergence，
`fresh_execution_replay_audit` gate 为 false（9个 replay exception、100个 derived-legal disagreement，
但 correctness disagreement 为0）。因此 admission 通过不等于 raw replay diagnostics 全部清零；
owned cleanup 已完成。相对 correctness-only SAAM
（909）为63 gain / 95 regression，净-32题（-2.086 pp，exact McNemar p=0.0134012）；相对
4B SFT epoch4（903）为64/90，净-26题（-1.6949 pp，p=0.0435999）；相对 three-level SAAM
（897）净-20题，相对 vanilla（890）净-13题。该单次对照结果显示准确率下降，不支持推广
first-error-capped，也不改变8B主线；r1/r2失败根不合并。

下一步不直接启动新的长 RL：先对该 960 条 frozen rollout 做后续 deterministic Harness error 的
alpha=0.25/0.5/0.75 离线软衰减重评分，保留首错和 timeout 规则，并核对上述 replay divergence
边界、错误/恢复后缀及正确轨迹负向 credit。只有某一 alpha 相对 correctness-only SAAM 不增加
错误轨迹正向 credit 且不牺牲正确率/合法率，才注册一个独立的 4B/table_rl 单臂；随后做 matched
BIRD-dev1534 和 fresh replay。

### 2026-09-14 correctness-only binary + SAAM 对照（训练与评测完成，能力准入未通过）

已登记的 A 方向对照已在 `table_rl` 启动，未重复已完成的 three-level 结果。运行根为
`/home/dengyan/tabular_rl_outputs/rl_runtime_qwen3_4b_correctness_only_saam60_table_rl_20260913_graph_r1`；
其 manifest 已锁定 Qwen3-4B `checkpoint-6380`、同一60题 cohort（SHA-256
`1a6cb257081da58ce66e09ab4e0783f296e3fdc4aa5674da178ae4c86756b4ca`）、Atomic v26、binary
result、SAAM asymmetric-error、span alpha=0.5、K=8、30题/update、4 updates、4-bit、SDPA、
actor old-policy、micro-batch 1/4096、KL=0和vLLM CUDA Graph。已完成 `global_step=4/4`、960
episodes，`train_runtime=26,093.77s`；checkpoint-4、implementation lock 和 precision audit
存在，训练资源已释放。已补做只读 lineage/advantage audit（960 episodes、856 eligible、7,389
events、未读取 gold SQL），但不替代完整 fresh replay。checkpoint 完整性已经通过，完整
fresh replay 是唯一剩余审计门禁。同题
BIRD-dev1534 matched greedy评测也已完成，不能把当前运行与已停止的重复续跑合并。

当前评测根为
`/home/dengyan/tabular_rl_outputs/evaluations/qwen3_4b_correctness_only_saam60_checkpoint4_graph_actionable_20260914_r4`，
其 `evaluation_config.json` 已锁定 `error_feedback_version=actionable-error-v1`；双卡 767/767
分片已完成，合并为 909/1534 正确（59.2568%）、1359/1534 合法（88.592%），1534 个
example index 全覆盖，owned GPU/vLLM 进程已清理。相对 4B SFT epoch4 为 77 gain / 71
regression（净 +6，exact McNemar p=0.6812），相对 three-level SAAM 为 74/62（净 +12，
p=0.3456），相对 vanilla 为 84/65（净 +19，p=0.1401），均未达到能力准入；只读
lineage audit 仍不替代完整 fresh replay。合法性相对 SFT 为 62 gain / 75 regression（净 -13，
p=0.3052），相对 three-level 为 70/69（净 +1，p=1.0），相对 vanilla 为 66/76（净 -10，
p=0.4502）。规范配对产物为 r4 根下 `results/merged/paired_comparison.json`（SHA-256
`12f92760ae294311c2c617f97e09a8031c21e936bb1b1bed2af52ab4ea210d09`），文件哈希与
question/db_id/split 的1534题零错配核验见同目录 `paired_comparison_provenance.json`
（SHA-256 `4bc0bfdf7948abb4dcbe93de750ad2769252b41752f00c1feefa6a4796020146`）。
此前 r3 根的 manifest 为 `legacy`，已在完成前停止，不得作为本次结果。

此前同identity的 three-level Graph 续跑根
`/home/dengyan/tabular_rl_outputs/rl_runtime_qwen3_4b_saam_threelevel_spanbalanced60_table_rl_20260913_graph_r2`
在首个 update 前收到 owned lifecycle SIGTERM，状态为`failed exit=143`；仅保留性能/启动证据，
不作为训练结果，也不再重复启动。

### 2026-09-13 4B profiler 精度校正与结果

首份独立 profiler 因缺少 BF16 autocast 而出现 FP32/math attention 和假性 4096-token OOM，
仅保留为失配反例。按正式 4-bit NF4、FP32 LoRA、SDPA、gradient checkpointing、BF16 autocast
重测后，2048/4096/8192 总序列的无 profiler forward+backward 中位数为 1.481/3.016/6.505
秒，peak allocated 为 4.92/6.12/8.90 GiB，504/504 LoRA 参数有有限梯度，4096/8192 均未 OOM。
8192 trace 使用 FlashAttention；CUDA self time 主要为 mm、FlashAttention、copy、mul。该
synthetic last-token loss 不是完整 GRPO update，不能直接换算正式训练吞吐；固定 transition 的
trainer-stage BF16/4-bit paired benchmark随后已完成，当前仍需真实 transition batch 的集成门禁。

随后完成 BF16 paired profiler：2048/4096/8192 中位数为 1.257/2.591/5.751 秒，相对
4-bit 快 15.2%/14.1%/11.6%；12288/14700 长度快 10.7%/9.5%，BF16 14700 peak allocated
14.52 GiB，504/504 LoRA 梯度覆盖。真实 Atomic v26 单组 BF16 stage 为211.24秒（rollout
84.91、old-policy28.54、policy95.36），但 transition 数与历史4-bit smoke不同，不能
宣称端到端加速；BF16仍是独立 execution arm。

### 2026-09-13 固定 transition 融合门禁

在 `table_rl` 对同一 `row_index=3/example_index=7365/turn_index=0`（总长1,957 tokens）完成
5次稳定计时。4-bit 两次 forward 与单前向融合的 steady median 为1.8167s与1.3895s，
1.307x；BF16为1.5220s与1.1659s，1.306x。两种模式的 token-level current/old log-prob
最大绝对差均为0，梯度余弦相似度均为1.0（梯度差属于舍入范围）。BF16单前向约快16.1%，
但 peak allocated 为9,074,672,128B，4-bit为5,398,452,224B，约1.68倍。证据根为
`table_rl:/home/dengyan/tabular_rl_outputs/perf_gate_fixed_transition_20260913/run5/`。

这通过的是固定单 transition、无 optimizer 的数值门禁，不是完整 GRPO throughput 或 accuracy
证据。只有在 PPO iteration=1、dropout=0、KL=0、无 ranking 且 optimizer 前权重相同的条件下，
old/current 融合才有等价前提；正式 trainer 尚未切换。下一步是用真实 transition batch 做
集成 token/gradient 对齐、checkpoint/OOM 回归并生成新的 implementation lock/manifest；正式
arm 继续使用 actor old-policy scoring，BF16继续作为独立对照。

### 2026-09-13 CUDA Graph 复测

在 `table_rl` 以固定 `example_index=7365` 完成一组 online one-group 复测：4-bit、SDPA、
gradient checkpointing、actor old-policy、KL=0、micro-batch 1/4096，vLLM
`VLLM_ENFORCE_EAGER=0`。日志确认 mixed prefill/decode 和 decode CUDA graph capture；8 episodes、
47 transitions、11,397 response tokens 的 rollout collect为74.42s，old-policy为28.38s，
policy前反向为93.73s，weight sync为1.72s，step为198.55s，train_runtime为199.92s。与此前
Graph smoke处于同一量级，但轨迹/有效transition不同，不能与eager总时长直接作差。状态为
`complete_audited`，owned trainer/vLLM已释放；证据根为
`table_rl:/home/dengyan/tabular_rl_outputs/perf_gate_cuda_graph_20260913/`。该诊断不改变正式
reward、old-policy或precision contract，8B/A100和完整update仍未验证。

### 2026-09-13 4B vanilla/SAAM 退化解释与下一步

纯 GRPO 的四步总训练计算时间为约 29,842.95 秒（8小时17分23秒；续跑日志中的
`train_runtime=24,210.68s`只覆盖后3步），SAAM SDPA 为 29,942.40 秒（8小时19分02秒），
差异约0.33%，不能把此前的续跑时间误认为纯 GRPO 全程时间，也不能把差异归因于 SAAM 算法。
纯 GRPO 的 simple 题相对 SFT 净损失15题、challenging题净增7题，符合小 cohort/短预算下
trajectory-level signed-binary 高方差扰动的症状，但 McNemar `p=0.3588`不显著。SAAM 相对
vanilla 的训练采样由390/960升至428/960正确，且记录到408条共享错误动作被抑制、394条共享
正确动作被保留；另有38条 correct-error 动作从原本的正优势被显式错误惩罚覆盖为负向，说明
错误惩罚可能过强。评测净增7题（88/81，`p=0.6445`）仍不足以证明能力增益。两次训练中合法错误轨迹仅约7.4%/10.1%带显式 Harness error，绝大多数是合法
语义错误；观察工具几乎普遍出现，错误轨迹更长，说明主要问题是观察后的实体/列/连接/聚合
选择及恢复后缀，而不是缺少观察。下一步先对固定960条 rollout 离线重算 correctness-only
SAAM、first-error-capped SAAM 和当前 control，再决定扩大 cohort/预算或注册 KL/span 对照；
不奖励观察频率，也不猜测合法语义错误的局部责任。

### 2026-09-12 SDPA恢复与8B匹配评测

8B `c=2–6` SDPA隔离恢复根已完成`global_step=4/4`，checkpoint-4完整、资源已释放；训练只改变trainer
attention backend（eager→SDPA），不与原eager失败根合并。同协议`actionable-error-v1`、greedy、2048、
BIRD-dev1534结果为931/1534（60.691%）和1387合法；checkpoint-6380 SFT为928/1534（60.495%）和
1380合法。同题号66 gain / 63 regression、净+3题，exact McNemar p=0.8603，未达到accuracy准入门禁；
评测owned-process cleanup已完成。4B SAAM恢复根也已完成`global_step=4/4`、960条rollout、
checkpoint-4和precision审计并释放训练资源；其独立`actionable-error-v1` BIRD-dev1534双卡评测已完成：
897/1534正确、1358合法。相对同题号4B SFT 903/1534、1372合法为77 gain / 83 regression、净-6题
（-0.391 pp，exact McNemar p=0.6928）；相对vanilla RL 890/1534、1369合法为88 gain / 81 regression、
净+7题（+0.456 pp，exact McNemar p=0.6445）。两项均未达到能力准入，table_rl GPU0/1和评测vLLM
owned-process cleanup已完成。

### 2026-09-12 Qwen3-4B SAAM 三等级 span-balanced 诊断

用户授权在 `table_rl` 启动独立 4B 诊断：从 4B cumulative SFT epoch4 `checkpoint-6380` 开始，
沿用新60题、K=8、30题/update、4 optimizer updates、Atomic v26、KL=0、
`trajectory_token_mean`。新增 result profile `three-level-clean-weighted`：正确且完全合法
（无 Harness/工具错误）= **1.25**，正确但有错误 = **0.75**，所有错误轨迹 = **-1.0**。
credit 使用 `saam-asymmetric-error`；span-balanced baseline 使用全 response 的
`span_balance_alpha=0.5`、`span_routing=uniform`，不使用 legal-reason-only hybrid。
这是 4B 独立诊断，不改变 8B 当前唯一 RL 方案；启动前必须完成 runtime/model/adapter/cohort
哈希 preflight，并将输出隔离。配置：
[`qwen3_4b_atomic_v26_saam_threelevel_spanbalanced60_table_rl.yaml`](../src/rl/configs/experiments/qwen3_4b_atomic_v26_saam_threelevel_spanbalanced60_table_rl.yaml)。

### 2026-09-12 OOM恢复与SDPA隔离续跑

两项诊断的首轮/首个恢复副本均出现了由 eager trainer attention 暴露的长序列显存问题：
8B 从 checkpoint-3 接续的最后 update 在 backward 申请716MiB时失败；失败批次的1,847个
可训练 transition 中只有同一条错误轨迹的4个 transition达到14.3k–15.5k tokens。4096是
多行 transition packing 预算，不能限制单条序列。4B SAAM首批240条 rollout 后，在一个
13,212-token正确 transition上触发9.29GiB eager softmax申请，尚未提交checkpoint。

已建立两个独立输出根并只改变 trainer attention backend 为 `sdpa`：8B 从完整复制的
checkpoint-3续跑，4B从checkpoint-6380重新开始；reward、credit、cohort、online rollout、
`max_new_tokens=2048`、`max_context_tokens=16384`、micro-batch=1/4096、gradient
checkpointing和KL均保持原值。SDPA只改变注意力实现，不把两次结果合并为同一实验。8B新的
preflight、base-model identity、implementation lock和checkpoint直接子目录门禁均已通过，
当前由NewGNN GPU2 trainer/GPU3 vLLM运行；4B由table_rl GPU0/GPU1运行。若SDPA仍OOM，先
保留失败证据，不自动降低context或删轨迹；`14336`只作为另立命名根的预注册备用边界。


### 2026-09-12 RL性能优化作为项目共享记忆

用户明确要求后续agent启动实验时必须感知已完成优化。规范入口为
[`rl_performance.md`](rl_performance.md)，并从AGENTS、当前README、最终契约和RL运行契约链接。
CUDA Graph主要优化online生成，reward/credit变更后仍可评估；8B A100 launcher当前覆盖
`VLLM_ENFORCE_EAGER=1`，8B尚未验证，不能认为共享默认0已经应用到8B或在途runtime。
`sampling`仅为改变old-policy分母及重要性校正的ablation，默认actor，当前不兼容非零KL。

随后在 table_rl 双3090完成8B单组对照：CUDA Graph rollout为166.88秒/22,749 response tokens（约136.3 tok/s），相对 eager 的91.1 tok/s约提升50%，但两次随机轨迹有效训练transition数不同，不能比较总update时间；sampling复用比actor-score少约84.7秒，但仍只作算法消融。`VLLM_MAX_NUM_BATCHED_TOKENS=32768` 对照为166.76秒，与默认几乎无差异，保留为可选参数。microbatch=2在3090 backward OOM；当前8B/3090安全起点仍为microbatch=1/4096。A100本轮因他人占用未测。
适配核查发现新resolver误改replicated BF16的checkpointing默认，已恢复BF16/4-bit均开启，
3个CPU回归测试通过。上述不变更reward/credit主线，也不启动或改写在途实验。

### 2026-09-12 NewGNN vanilla-GRPO OOM恢复

原`c=2–6 balanced60 signed vanilla-GRPO`在第1个已提交update之后的backward阶段发生
CUDA OOM；`checkpoint-1`和240条已提交rollout保留，未提交的240条由恢复审计移除。原运行的
manifest与implementation lock不修改。恢复副本逐字复制训练实现和identity artifacts，使用
外部生命周期wrapper（不改变trainer实现）从副本`checkpoint-1`接续；只把transition
micro-batch限制为1 row/4096 padded tokens，并设置`PYTORCH_ALLOC_CONF=expandable_segments:True`。
模型、checkpoint-6380、cohort、reward、K=8、4 updates和评测协议保持不变。资源门禁是两张
真正空闲且无其他用户compute PID；若仍OOM，保留失败证据，不停止他人进程。

2026-09-12 05:52（服务器时间）复核NewGNN GPU2/3无compute PID、显存近空且独占锁可获得，
已按GPU2 trainer + GPU3 online vLLM启动隔离恢复。vLLM端口18312已通过health和TRL serving
capability检查，trainer从checkpoint-1加载后进入`0/4`恢复进度；后续已生成checkpoint-2
（global_step=2/4、epoch=1.0），累计480条rollout。GPU2短时利用率为0时，进程仍占用约
22.3GiB显存并在等待GPU3 vLLM的下一批生成，不得将其判定为空闲。当前已生成checkpoint-3
（global_step=3/4、epoch=1.5），累计960条rollout；最后update在eager backward再次OOM并已
释放GPU2/3，原恢复副本不能作为效果证据。SDPA续跑见上方独立恢复登记。

### 2026-09-12 K=8筛选条件扩展与新批次

用户要求将后续筛选后的候选条件从“每题 K=8、正确轨迹数2–6”扩展为“正确轨迹数1–7”。
该变化当时记录为探索性候选库存版本，未改变正式RL主线的admission contract，也
不覆盖`data/inventory/rl_training_candidates_current.json`。基于现有2603条筛选观察重建的
1–7库存为strict 896题、expanded 1025题、known-exclusion fresh 353题；strict直方图
1/2/3/4/5/6/7为179/128/92/85/107/114/191。入口为
`data/inventory/rl_training_candidates_1to7_current.json`，报告为
`docs/reports/rl/RL_TRAINING_CANDIDATE_POOL_1TO7_20260912.md`。

旧`qwen3_v26_remaining_k8_screen848_20260906`已确认848/848完成并释放遗留GPU4/6服务。
新增批次使用已准备的3000题输入，复制到隔离输出根
`/home/dengyan/tabular_rl_outputs/evaluations/qwen3_v26_k8_screen3000_1to7_20260912`，两
个1500题shard在NewGNN GPU4/6、端口8205/8206上运行；任务输入SHA-256为
`0b1efa4dfde66b449bf0d1eb30dbb1447ca6694d2425988ca70784a75ee8e646`。启动时两个health端点
均为200，结果manifest已创建；最近复核结果文件分别有670/1500和600/1500个去重完成题目（合计1270/3000，正确875题），任务尚未完成，不得提前报告新候选数量。heartbeat `table-rl`
负责只检查本任务PID、异常时按同一manifest安全续跑、完成后释放本任务vLLM并重建1–7库存，
不得触碰GPU2/3在途RL或其他用户进程。

2026-09-16更新：该批次已完成3000/3000并释放GPU4/6，合并库存重建为1–6口径，见本表
“2026-09-16 3000题K=8筛选完成与候选区间改为1–6”条目；上面记录的“未完成”状态不再适用。

### 2026-09-12 table_rl update 性能审计

上一轮4B checkpoint-4的`actionable-error-v1`双卡评测已完成并释放GPU0/1。随后在隔离 source
copy 上做一组8轨迹、1 update的埋点 smoke：retry1（1 row/4096 tokens）总303.55秒，
rollout 203.85秒、old-policy 22.92秒、policy阶段74.87秒，37个transition形成37个
microbatch；retry2（8 rows/8192 tokens）虽降为18个microbatch，总时间反而342.73秒，
policy阶段113.27秒，峰值reserved约11.77GiB。结论是暂不把简单扩大token cap作为正式
优化；保留隔离埋点，后续若改训练实现必须另做低扰动matched smoke、OOM门禁和评测。

代码优化 smoke 已完成：训练入口新增 checkpointing 和 attention backend 的显式配置并写入
manifest。4-bit 关闭 checkpointing 在长轨迹阶段 OOM（23.25GiB），checkpointing+SDPA 总时长
306.49秒，未见相对 retry1 的稳定收益。将 vLLM `VLLM_ENFORCE_EAGER` 设为0启用 CUDA Graph
后，rollout 约206.70→74.67秒，update 约303.55→199.43秒（约34.3%），显存峰值约11.8GiB。
该 smoke 只作性能证据；默认保留 checkpointing，正式 RL 需重新做短 update/OOM 门禁，兼容
调试可设 `VLLM_ENFORCE_EAGER=1`。

另有显式 sampling-score 快速模式：`--old-policy-logprob-source sampling` 与 CUDA Graph
组合的同组 smoke 为171.86秒（rollout 74.70秒、policy 93.94秒），较 actor-score 的
199.95秒再快14.0%。它令 importance ratio 恒为1，改变正式 actor-vLLM correction 目标，
仅作为独立速度 ablation；默认仍是`actor`，manifest/log 会记录是否复用。

### 2026-09-11 RL 筛选候选池统一登记

已将完整 Atomic v26 K=8 筛选证据按稳定 `example_id` 整合。原始700题、新600题和剩余848题
的 strict 合并为545题；加入此前历史候选索引后为723题。旧 A100 700题 cohort 重叠423题，
checkpoint-6380 SFT training view 重叠98题；这些重叠保留为元数据，不静默删除。当前60题
复筛只作为diagnostic证据，不新增候选。统一入口和哈希见
`docs/reports/rl/RL_TRAINING_CANDIDATE_POOL_20260911.md`；正式训练前仍需从候选池冻结约500题，
完成 exclusion、fresh replay、manual audit 和 immutable cohort manifest。

### 2026-09-11 Qwen3-4B 新60题 vanilla GRPO baseline

用户明确授权运行4B GRPO baseline，训练数据固定为已冻结的新60题
`1a6cb257081da58ce66e09ab4e0783f296e3fdc4aa5674da178ae4c86756b4ca`，与旧简单60题零重叠。
复用8B新60题对照的预算：可训练完整carrier使用signed-binary +1/-1、trajectory/uniform、K=8、30题/update、
4updates、LR4e-7、KL0、2048 tokens/turn。从4B SFT epoch4 checkpoint-6380启动，
adapter SHA256 `0bc7b8b644ab8aba0c8f0762ff58bba66f5b7e917182bf5ada10e6b3ce522d19`。
A100 GPU0–3均被占用，使用table_rl双3090、一卡4-bit replicated trainer加一卡online vLLM，
micro-batch1/8192。该数据的2–6分档来自8B筛选，不能直接称为4B的2–6分档；4B实际混合组
比例由online rollout记录。保持数据固定，不临时换题，不启用SAAM/span奖励。
达到2048 token上限却未形成完整action的carrier沿用vanilla排除合同，不进入组归一化或反向；
其比例必须单列，不能描述为已接受-1训练信号。
属于用户授权的独立baseline诊断，不更改8B主线；训练结束后按actionable-error-v1做
BIRD-dev1534独立评测，与4B SFT epoch4的903/1534进行同题比较。

2026-09-12更新：4/4 updates已完成，checkpoint-4/960条rollout/precision/四次权重同步
审计通过，最终adapter SHA-256为
`e394e9d36684a85e2a078d572cd762985ef17eb7cff9f7a2fd2a90b5d5b9517c`；训练进程和online
vLLM已释放。checkpoint-4随后完成双卡独立服务评测，每卡767题，使用
actionable-error-v1/greedy/2048；结果为890/1534（58.018%）、1369合法，对照4B SFT
epoch4的903/1534（58.866%）、1372合法，逐题配对净-13题（79 gain、92 regression，
McNemar p=0.3588）。分层为challenging +7、moderate -5、simple -15，整体合法率仅下降
0.20个百分点；当前解释是60题/4-update小预算下vanilla trajectory-uniform credit的高方差
局部改写，结论为诊断性小幅回归，不构成显著退化。

### 2026-09-11 Qwen3-4B 双卡独立服务评测

用户明确双卡指每张卡独立加载完整模型。4B累计SFT已完成4 epochs/6380步；epoch3
checkpoint-4785、epoch4 checkpoint-6380顺序使用table_rl GPU0/1，每卡TP=1/32并发、
各767题，统一actionable-error-v1/2048。原TP=2 legacy运行停止并保留，不并入新结果。
共享runner preflight已通过；同反馈参考为8B epoch4的928/1534，不替换RL主线基座。
见[执行报告](../reports/sft/QWEN3_4B_CUMULATIVE_DP_EVAL_20260911.md)。

### 2026-09-09 后续评测统一使用新反馈框架

用户明确要求：从本决定之后新启动的评测统一使用 `actionable-error-v1` 反馈框架，并与
对应 checkpoint 的 model/protocol/prompt/runtime identity 一起记录；不得再用 `legacy`
作为后续正式评测框架。已完成的 legacy 评测只保留为历史审计，不回写或混入新反馈结果。
当前已在途的 checkpoint-3 legacy 评测不强行中断，避免改变已启动序列；后续新队列须先
通过 feedback identity preflight。定时任务已同步为不再自动启动旧 SAAM/SMC RL，待新的
`wrong-uniform-four-level` reward 完成离线统计并重新确认后才能启动训练。

### 2026-09-09 SAAM 主方向收敛与错误轨迹统一惩罚

用户决定保留 SAAM 方向的三个组件：SAAM asymmetric-error credit、four-level reward
框架和 reason/tool span-balanced；摒弃 SMC，不再将其作为后续训练或验证入口。所有最终
错误轨迹使用相同的负向结果信号，不因“没有检测到过程错误”而减轻惩罚；SAAM 的共享
state-action 抑制和局部错误 action credit 仍保留。该决定尚未启动新的训练，下一版 reward
的具体 profile 需先完成离线回放统计并登记后再启动。

现阶段推荐的最小 profile 为 `wrong-uniform-four-level`：correct+clean=`+1.5`，
correct+error=`+1.0`，任何 incorrect=`-1.0`。其中 correctness 决定符号，cleanliness
只区分正确轨迹的正向幅度，不再区分两类错误轨迹。首个对照只改变该 profile，保持 SAAM、
span alpha、cohort、decode、optimizer 和 checkpoint 不变；不得同时加入 semantic distance、
gold-path reward、critic、树搜索或新的错误分类器。

### 2026-09-09 否决 Harness causal closure reward

此前登记的需要新增 `source_ref` 字段的 `harness-causal-closure-grpo` 版本予以否决，不进入训练或后续验证。它原本利用 Atomic v26
的实际工作模式：模型通过 typed tool 观察 resident state、读取结果、产生 relation/value
artifact，最后用 `answer_from_context` 引用证据表。对正确轨迹，从终端引用的证据表沿
Harness-authored `derivation/producer/inputs` 元数据追溯 artifact 链，并沿成功 action 的
可见 observation/grounding/reference 元数据追溯“观察结果被后续 action 使用”的决策边，
得到 episode 内的候选因果闭合链；但 observation→decision 关系无法从现有记录中可靠恢复。

否决原因：让模型填写 `source_ref` 会改变既有 action schema 和模型调用分布，需要重新训练；
Harness 只能验证引用的值是否曾出现在 observation 中，不能验证模型是否真的依据该 observation
做出决策，因此仍会产生不可接受的伪因果。仅靠现有 rollout 的 literal 相等匹配同样不满足
完全准确条件。保留已完成的离线覆盖率和人工核查报告作为反例证据；后续不再实现该字段、
不修改 prompt、不接入该字段版本 reward。

在不改 action schema 的前提下，允许另行登记一个低覆盖、弱标签的 `conservative-read-support`
离线候选：只使用 `read_subtable` 的唯一可见值、后续成功 action、terminal artifact closure
和题目文本排除条件。现有 8,480 条 rollout 上该候选覆盖 759/8,480 条轨迹，其中正确轨迹
533/5,105；这些数字仍不是确定因果，只能先做离线 reward simulation 和人工精度审计，不得
直接替换主线或宣称 observation credit 已被可靠识别。

同日候选效用复核（仅诊断）：观察工具不强求 artifact 图连接，不加额外 bonus 不等于
撤掉现有结果优势/SAAM credit。原操作覆盖率聚合只按 `trajectory_id` 连接，受跨 update
ID 复用影响；逐源逐行重算为 24,386/31,378（77.717%），旧 71.209% 撤回。正确轨迹中
16,014/18,692 操作命中候选（85.673%），3,466/5,105 条正确轨迹全部操作命中（67.894%）。
这不是 action 语义精度或非零优势覆盖。已有50动作分层人工审计的41 core/6表示/3可选
标签不足以证明全部命中动作值得额外加强；literal 传值还会漏掉有用的操作 producer。
不把闭包命中直接升级为独立 reward；若继续，只讨论固定基础信号/正向系数预算下的
credit 分配对照，不按工具名硬分语义、不改变主线、不启动 RL。详见
[`HARNESS_CAUSAL_CLOSURE_AUDIT_20260909_ZH.md`](../reports/rl/HARNESS_CAUSAL_CLOSURE_AUDIT_20260909_ZH.md)。

进一步图外审计：6,992个图外操作中，694执行失败，2,459成功但整条轨迹无可解析终点，
3,839成功且有终点。对非保留分区最后一类确定性抽取80个逐条核查：46有信息传递、
14重复/无新增任务信息、10放弃尝试、9不确定、1错误语义传值。抽样不是全量语义真值；
不能将22.283%图外比例解释为无效比例。存在第一次成功计算被排除、第二次重复反而入图
的反例，不据此启动闭包bonus或mask。全量138条重复调用另列，不能与抽样计数相加。
详见 [`HARNESS_OUTSIDE_OPERATION_AUDIT_20260909_ZH.md`](../reports/rl/HARNESS_OUTSIDE_OPERATION_AUDIT_20260909_ZH.md)。

用户质疑增量bonus假设后，不继续推进“闭包命中即加奖”；当前证据不支持其价值，亦不
证明模型已经达到最短路径。观察时序全量核查显示前/中/后1/3观察占比76.67%/29.46%/
18.59%，但正确轨迹中40.51%（2,063/5,092，同时含两类动作）在开始操作后仍会观察。
保持基础算法，不按阶段额外奖惩观察；reward hacking仅作为设计风险。见
[`OBSERVATION_TOOL_POSITION_AUDIT_20260909_ZH.md`](../reports/rl/OBSERVATION_TOOL_POSITION_AUDIT_20260909_ZH.md)。

### 2026-09-09 固定后缀 action 删除重放（仅离线诊断）

继续静态/逐轨迹审计，新增共享只读 skip replay，不接入 reward、不启动 RL、不调用教师。
保留原始 action index/value_ref 和输出句柄编号，先通过逐步输出与终局的 baseline fidelity
门禁，再报告后缀新错误、终端引用失效、仍正确/仍错误等事实。后缀固定时观察步骤的
policy-mediated effect 一律未识别，不标“冗余”。从已登记的 discovery/reviewed 分区
确定性选择30个混合题组的正负配对，保留 reserved_unread；数据库及运行时记录哈希，
禁止主库写入，设置整条诊断时限。数据库缺失/基线不一致/诊断超时只影响覆盖率，不算策略错误。
不改变原运行中 SQL timeout 属于 policy error 的约定。详细结果写入扩展审计报告。

最终审计已完成：60条轨迹/30个题组中51条通过 baseline fidelity，9条因生成截断保留
`unknown_incomplete`；383个有效删除分支没有出现 `wrong_to_correct`。数据库前后哈希
不变。最终报告中的 `executable_same_result` 只表示固定后缀结构结果，不表示策略冗余，
也不接入当前 reward。

### 2026-09-09 外部教师离散过程 credit：小样本 API 审计已授权

用户明确恢复的范围仅为既有 RL 完整轨迹的事后评分，不恢复因果 SFT 生成，不启动 RL。
使用官方 DeepSeek `/chat/completions`，教师看问题、学生工具契约、全序列 action/observation
及终局 correct 标志；不看 gold SQL/result、隐藏 denotation comparison 或成功轨迹答案。
不实现程序化依赖闭包，教师输出离散类别、直接依赖步骤号、证据和离散确定性。
SQL/tool timeout 必须归 policy error，不作环境错误或零惩罚；教师 API 故障则显式中止审计。

预注册诊断规则 `teacher-process-credit-v1`：correct 中关键/有用/冗余/不确定步骤的
固定倍率分别为1.0/0.5/0.0/1.0；显式错误步骤包括已恢复的错误使用局部
`-max(abs(A),1.0)`。wrong 中仅终端表示错误惩罚answer，语义过程错误惩罚最早未恢复的
致错步骤；选中惩罚位置使用 `-max(abs(A),1.0)`、其余为0。timeout 的每次实际超时action
均保留同样局部负惩罚，不因上游归因而被mask。格式错误发生在中间action、缺失聚合/
去重、生成截断均不能自动判为终端格式错误。无法定位的wrong保留原始结果优势，不制造
正向credit。权重由固定映射计算，教师不得输出连续confidence/score/reward。

计划先做至多16次真实API调用（最多8条、两次独立判断）和逐例复核；记录prompt、来源及请求
哈希、provider identity/usage、失败和重复判断一致性。小样本不是准确率评测；未通过语义
复核不能扩大到全量、不能接入actor。具体产物见
`docs/reports/rl/TEACHER_PROCESS_CREDIT_PILOT_20260909.md`。
本轮因停止并发线程时已发出的请求继续完成，实际为22次；这被记录为预算控制缺陷，
不是新的授权，也不改变未接入actor的停止线。

| 事项 | 当前答案 | 证据/落点 |
|---|---|---|
| 工具版本 | Atomic version26 | `docs/current/final_project_contract.md` |
| A100 训练方式 | 单卡 replicated trainer + 独立单卡 online vLLM | update-40 run manifest；world size=1 |
| 服务器职责 | A100 优先做 RL 主实验；A100 不可用时，table_rl/NewGNN 可按降级拓扑做 RL；三机均可评测 | 用户于 2026-09-06 确认；`server_resources.md` |
| A100 serving | 另用一张 A100 跑 online vLLM | 当前 live gate 进程和 formal launcher |
| RL credit | `saam-asymmetric-error` | `src/rl/frameworks/trl/state_action_ambiguity.py` |
| RL reward | four-level result reward | `src/rl/runtime/terminal_reward.py` |
| RL advantage adapter | `correctness-primary-clean-secondary`, `clean_advantage_weight=0.25`; correctness determines the sign, cleanliness only changes magnitude | 2026-09-08 A100 rollout audit: raw four-level normalization produced positive advantages for 331 all-wrong legal trajectories and negative advantages for 69 correct-with-error trajectories |
| RL 梯度更新 | reason/tool 各 0.5 的加权 full-response 梯度 | `span_balance_alpha=0.5` |
| timeout | 视为 policy error，timeout action 使用局部负向惩罚 | SAAM asymmetric-error |
| RL 起点 | cumulative SFT `checkpoint-6380` | A100 formal/live launcher |
| RL 数据与批次 | 目标约500题；每题K=8且正确数1–6；30题/update；从checkpoint-6380启动；候选入口`data/inventory/rl_training_candidates_1to6_current.json` | 2026-09-06跨服务器筛选盘点；2026-09-16区间由2–6改为1–6 |
| GPU 选择 | A100 仅允许 GPU 0–3；动态选择一张 trainer 与一张不同的 vLLM 卡 | 用户确认；launcher 已加入 allowlist |
| 旧代码 | 全部移入 `archive/`，暂不删除；`src/` 不保留 compatibility stub | `archive/code/legacy_migration.md` |
| KL 验证 | 必须通过独立匹配对照验证是否有增益；当前正式 arm 仍为 `kl_beta=0` | `docs/current/rl_pipeline.md` |
| A100 资源约束 | 单卡拓扑；trainer/vLLM 必须不同卡；只允许 GPU 0–3 | 用户于 2026-09-06 确认 |
| 单卡资源入口 | 作为当前主线；默认 BF16 冻结基座、FP32 AdamW、独立 output root | `qwen3_8b_atomic_v26_saam_screened500_single_gpu.yaml` |
| rollout 生成预算 | 当前主线及后续新配置默认 `max_new_tokens=4096`（单轮）；历史配置保留其 manifest 中的显式值，不回写历史实验 | 用户于 2026-09-08 根据 A100 5,600 条 rollout 中 802 条（14.32%）触发 2,048 上限后的确认 |
| 正式实验启动确认 | 启动前必须先列出并提交用户确认：配置/配置哈希、模型与 checkpoint、cohort/数据哈希、protocol/prompt/runtime identity、reward/credit、rollout、optimizer、GPU 拓扑、输出目录和评测计划；未获确认不得启动 | 用户于 2026-09-08 明确要求 |
| 3090 降级入口 | 两张独立 3090：单卡 replicated trainer + 单卡 online vLLM；从 1 row / 8,192 tokens 起步 | 用户于 2026-09-06 确认；需独立 launcher、manifest/output root |
| Qwen2.5-7B SFT 对照 | 在 NewGNN 用四卡训练 `Qwen/Qwen2.5-7B-Instruct`，复用 Qwen3 SFT1 的同一 4,471-record model-visible 数据；只作为模型族对照，不改变 Qwen3/RL 主线 | 用户于 2026-09-04 明确要求；run `qwen25_7b_atomic_v26_sft1_same4471_4gpu_full_20260904_203529` 已以 exit 0 完成，checkpoint-560 已保存 |
| 今晚诊断对照实验 | 注册三项仅用于审计/对照的 version26 降级 RL：原版 SMC+原始 `saam_gate60` 数据、原版 SAAM+four-level+span-balanced+新版 `smc_balanced60` 数据、以及经审计后固定的 SMC 优化诊断版本；统一从 checkpoint-6380 启动，独立 output root，完成后在 NewGNN matched greedy 评测；结果不得替换正式 SAAM 主线或宣称 accuracy promotion | 用户于 2026-09-06 明确要求；table_rl 仅作诊断资源，所有配置、cohort、概率审计和评测结果分别记录 |
| Harness-conditioned SMDP/IQL 方向 | 先做离线 transition/可识别性诊断，不训练 actor；transition 只保留 model-visible state、typed action 和 Harness four-level terminal reward，拒绝 gold/ref 字段。只有“同一精确 state 存在不同 action 且 outcome 也不同”才进入 critic 小规模训练门禁 | 用户于 2026-09-07 要求尝试；实现落在 `src/rl/frameworks/trl/{mdp_critic,iql}.py` 与 `src/rl/scenarios/diagnostics/audit_harness_smdp_iql.py`，manifest 标记 `diagnostic_only` |

## 2026-09-09 已批准的 correctness-primary 诊断

用户确认table_rl双卡空闲后自动启动4-update算法验证，不再二次询问；正常完成后自动双卡
matched greedy评测并释放本任务vLLM。使用checkpoint-6380、同新版60题、30题/update、K=8、
four-level、correctness-primary-clean-secondary（clean weight0.25）、SAAM、span0.5、KL0。
独立运行已于02:46启动，训练已完成；用户随后批准同机双卡顺序评测checkpoint-1→2→3。
09日17:10核查：checkpoint-1/2/4为924/915/917（均1534题），checkpoint-3已接续并有双卡吞吐。
评测使用legacy反馈；928题的新反馈SFT为actionable-error-v1，跨反馈差值不能归因为纯RL效果。
仅监控既有队列，不在空卡时重复启动训练，不更改在途评测配置。实际沿用旧2048预算、rows1/token budget0，
不得写成已执行新默认4096或硬8192上限。见
[`CORRECTNESS_PRIMARY_TABLE_RL_20260909.md`](../reports/rl/CORRECTNESS_PRIMARY_TABLE_RL_20260909.md)。

## 已确认的隔离诊断

### 2026-09-09 已确认的 120 题 wrong-uniform 隔离诊断

用户已确认从已完成的 600+848 题筛选池中，按每题 K=8 的正确轨迹数 1–6 选择 120 题，
固定配额为 30/25/25/20/10/10；该范围有意包含正确数为1的困难题，不引入额外筛选指标。
本实验不是当前约500题、正确数2–6的正式主线，而是用于验证“所有错误轨迹统一为负向结果信号”
是否能改善训练方向的隔离诊断，结果不得替代主线或宣称正式 accuracy promotion。

实验固定为：Qwen3-8B `checkpoint-6380`、Atomic v26、binary result（correct=`+1`，
incorrect=`-1`）、`saam-asymmetric-error`、`trajectory_token_mean`、reason/tool
`span_balance_alpha=0.5`、K=8、30题/update、4 optimizer updates、LR `4e-7`、weight
decay `0.1`、clip `0.2`、`kl_beta=0`、单轮 `max_new_tokens=2048`。训练使用 NewGNN 两张
独立 3090，训练卡和 vLLM 卡动态选择，不抢占已有进程；训练完成后在新的
`actionable-error-v1` 反馈框架下进行对应 checkpoint 的 BIRD-dev1534 matched greedy
评测并释放本实验拥有的 vLLM。标准任务文件 SHA-256 为
`4f37f5c717e534c9715af14e1da91c7f8a73081c26ed3bf7c33d4804454ba901`，配置落点为
`src/rl/configs/experiments/qwen3_8b_atomic_v26_wrong_uniform_saam_120_newgnn.yaml`；
远端已完成 JSONL 读取门禁，GPU 空闲后再启动。

### 2026-09-10 已确认的 60 题 legal-reason-only hybrid 隔离诊断

用户已确认启动本次验证：不改变当前正式主线，仅验证在 checkpoint-6380 的新版 60 题
cohort 上，binary/wrong-uniform result reward 与 SAAM asymmetric-error 结合时，按
turn 的 Harness 错误状态选择 loss carrier。正确/错误轨迹均保留结果方向；没有 Harness
错误反馈的 turn 只训练 reasoning span，带 Harness/protocol 错误的 turn 使用
`span_balance_alpha=0.5` 的 reason/tool 加权。span 边界无法解析时 fail-closed 回退到完整
response，不丢弃 transition。固定为 Qwen3-8B checkpoint-6380、Atomic v26、cohort
SHA-256 `b377ab3f7cbec8745a5342f3428435f91a77b8ada5f35aafb3465c780a07bbb4`、K=8、
30题/update、4 optimizer updates、seed=20260901、LR `4e-7`、weight decay `0.1`、
clip `0.2`、`kl_beta=0`、`max_new_tokens=2048`。该 arm 是 diagnostic-only，不替代
约500题正式主线；完成后使用当前 actionable-error-v1 evaluator 做 BIRD-dev1534 matched
greedy，并保留 full manifest、implementation identity 和 rollout audit。

本次运行在 NewGNN 采用已验证 legacy runtime 的隔离副本，以保持 Atomic v26 protocol
module identity；hybrid 逻辑只加入该副本和 immutable manifest，不覆盖历史 runtime。

最终结果（2026-09-11）：训练和actionable-error-v1 BIRD-dev1534评测均完整结束。
checkpoint-4为902/1534，匹配的checkpoint-6380 SFT为928/1534；同题号49 gain、75 loss，
净-26（-1.69 pp），exact McNemar p=0.02437。合法终止1380→1378，54 gain、56 loss，
p=0.92410；process errors 784→743。该证据否决`legal_reason_only_hybrid`，不再将其作为
候选主线或通过增加KL继续挽救。由于缺少同cohort uniform/full-response control，本结果不
单独否决signed-binary或SAAM；下一步仍按60题复筛结果先做vanilla baseline。详细证据见
[`LEGAL_REASON_HYBRID_RESTART_20260910.md`](../reports/rl/LEGAL_REASON_HYBRID_RESTART_20260910.md)。

### 60题筛选来源分支（2026-09-11纠正）

后续判断必须针对“最初做 K=8 筛选的模型”。若这60题由 checkpoint-6380 筛选，必须对
完全相同的60题用 checkpoint-6380 SFT 起点重新做逐题 K=8 筛选，再统计全对/混合/全错组；
若由 checkpoint-560 筛选，则不重复筛选，并将其作为旧模型已验证的简单题。当前精确 cohort
`b377ab3f7cbec8745a5342f3428435f91a77b8ada5f35aafb3465c780a07bbb4` 出现在
checkpoint-6380 的 `smc_balanced60`、`smc_filtered2to6_60` 及 span/primary 运行链中，
而 checkpoint-560 的历史60题使用另一 cohort hash `47e9...d79d5`；因此当前60题按
checkpoint-6380 分支处理。由于当前 cohort 文件没有筛选阶段逐题正确数和 decode manifest，
当前双卡评测释放后必须补做这次 K=8 复筛，不能以已有 RL rollout 或训练起点替代筛选证据。
该复筛只做筛选审计，不在 RL epoch 中动态重筛题。

当前前置的 checkpoint-4 actionable-error-v1 双卡评测已完成，两个 shard 为 455/767 和
447/767，合计 902/1534；本任务 vLLM 已释放。已在 NewGNN GPU1/3 启动上述精确60题的
checkpoint-6380 K=8 复筛，输出根为
`/home/dengyan/tabular_rl_outputs/evaluations/qwen3_v26_checkpoint6380_rescreen_current60_k8_20260911`。
复筛已完成：60/60题、480/480样本完整，27题8/8全对、1题0/8、仅14题正确数2–6，
477/480轨迹合法；复筛服务已释放。这证明旧60题对checkpoint-6380偏易。已从另一批
checkpoint-6380完整848题筛选池的204道2–6候选中，仅按正确数固定新60题（2至6各12题，
与旧60题零重叠），任务SHA为
`1a6cb257081da58ce66e09ab4e0783f296e3fdc4aa5674da178ae4c86756b4ca`。下一项冻结为
signed-binary vanilla GRPO：trajectory/uniform、4updates、LR4e-7、KL0；CPU preflight
已通过。NewGNN当前没有两张空闲卡，所以trainer/vLLM尚未启动。完整合同见
[`VANILLA_GRPO_C2TO6_BALANCED60_20260911.md`](../reports/rl/VANILLA_GRPO_C2TO6_BALANCED60_20260911.md)。

2026-09-10 续查修正（尚无首个有效 update 证据）：旧 `binary` 实际为 +1/0，不能在
manifest 中称为 +1/-1。本诊断使用独立 `signed-binary` profile 落实已确认的正确 +1、
错误 -1；timeout 保留且对应失败 action 负惩罚。保留 binary 的 generation-length、
OOM/context-overflow 排除边界，不借本次修复扩大截断训练。成功恢复动作上的
`feedback_recovery` 描述上一轮错误，不作为本轮错误；hybrid 必须有逐 turn 对齐的
Harness audit。普通 OpenAI vLLM 服务不能替代 TRL 权重同步服务，必须核查服务接口。
canonical factory 的重构后模块名应为 `rl.runtime.tool_environment_v26`，核验其精确
源码路径并继续核验 frozen protocol/executor 路径；不以切换旧入口绕过 identity 门禁。

同日用户已明确再次允许本诊断的远端清理、部署及定时任务更新，后续相同范围无需重复
确认。已重新核验本任务旧vLLM归属并发送TERM（exit0），释放尚待核查；平台授权审核
容量故障不是用户未授权，不得用其他路径绕过。尚未观测到可用双空卡，不抢占其他任务。

同日后续核验已更新上述状态：旧本任务服务不再运行，r2独立部署及半小时heartbeat更新
成功；远端核心101项测试和CPU完整身份preflight通过。补充修复共享退出清理的作用域/
信号退出码，不改变确认的实验参数；GPU启动仍须即时空卡检查、真实在线同步和有效update。
逐次运行状态以`docs/reports/rl/LEGAL_REASON_HYBRID_RESTART_20260910.md`文首为准。
20:08后：补充清理修复部署后的远端105项测试及真实CUDA核验通过；双空卡即时检查未过，
启动exit75，等待资源而非重新请求配置授权。新训练和在线同步仍未发生。
23:47最新状态已覆盖上述资源等待：本批准诊断在空闲GPU2/7实际启动控制流程，CPU身份与
TRL HTTP门禁通过，进入trainer初始化。实际在线同步/有效update仍待核验；后续只读查询
遇平台审核容量故障，恢复后先检查现有run，不重复启动、不改变参数。

## 待用户确认

### 2026-09-08 错误反馈增强与 SFT dev 回归（用户已授权实施并确认2048预算）

用户批准实现失败 action 回传及基于 Harness 已有状态的结构化诊断，并用
Qwen3-8B cumulative SFT `checkpoint-6380` 做 BIRD-dev1534 greedy 回归。
候选 `actionable-error-v1` 只改变 rejected-action observation：不修改 system prompt、
action 接受集合、工具执行结果、错误类别/计数、reward 或终止条件；嵌套 value 和
join right 带前缀仍拒绝。共享实现落在 `rl.runtime.error_feedback`；冻结 runtime
通过显式评测 adapter 加载，不覆盖原始 runtime，不把本地重构后的其他差异带入对照。
所有 source/hint 只能使用失败 action、既有错误和已可见 resident metadata，无 gold、
额外查库或语义修复建议。先通过本地不变量/上下文回归，再展示实际配置/身份哈希并获得
确认后启动评测。默认新预算为4096；若旧 baseline 为2048，不能混淆预算影响与反馈影响，
须另作同4096旧反馈对照或经用户确认固定2048。用户随后明确选择 **2048 tokens，
只评一次新反馈并与历史完整 baseline 对比**；此次不是改变新配置默认4096。
使用历史冻结 runner 原有工具超时策略，不另加 formal10秒包装；serving 的32768上下文、
8192 batched tokens、24并发也对齐历史 launch manifest。跨服务器/时间复用 baseline
仍存在数值及运行波动，配对结果不能冒称排除这些因素。完成前不准入正式 RL 默认。
首次vLLM启动因未带现有 `TRITON_LIBCUDA_PATH` 而失败，未进入答题且已自动清理。
重试只还原进程级CUDA库路径并锁定真实库hash，不安装或升级任何推理软件/驱动。

### 2026-09-08 大样本静态语义审计（不改训练）

用户要求先观察 a100 主实验及 table_rl 既有 K=8 rollout，不凭启发式猜测错误等级。
当前 distance/format 原型仅保留为失败诊断：不能产生 actor reward、可信度或因果归因；
不得接入训练，也不自动剔除 cohort。共享统计入口为 `rl.diagnostics.rollout_corpus`，
薄 CLI 为 `scenarios/diagnostics/audit_rollout_corpus.py`。跨 run 按数据库和题干建立题目
映射，不能直接连接 cohort 内的 example_index。正式 reward、GPU 进程及评测评分不变。
观察集和按题保留集隔离，Codex 逐条审阅不冒称独立人工标注；详细证据见
`docs/reports/rl/ROLLOUT_CORPUS_SEMANTIC_AUDIT_20260908_ZH.md`。

### 2026-09-08 用户批准的 span alpha 对照

用户要求在 table_rl 启动 `span_balance_alpha=0.25`，训练完成后在同机双卡评测。
此项为独立诊断，不修改正式 arm 的 alpha=0.5。复用
`qwen3_v26_saam_fourlevel_spanbalanced_smc_balanced60_20260907_retry2` 的实际设置：
checkpoint-6380、相同新版60题（SHA-256
`b377ab3f7cbec8745a5342f3428435f91a77b8ada5f35aafb3465c780a07bbb4`）、30题/update、K=8、
4 optimizer updates、seed=20260901、LR=4e-7、four-level/SAAM、4-bit frozen base + FP32 LoRA/AdamW。
训练完成后评测 BIRD-dev1534 greedy，自动释放本次拥有的 vLLM 进程；比较对象为
checkpoint-6380 SFT 和该 alpha=0.5 run 的 table_rl 全量评测。
评测步数必须从 trainer_state 验证；历史评测的 checkpoint10 标签不能当作真实训练步数。
入口 `src/rl/scenarios/diagnostics/run_saam_span_alpha025_table_rl.sh` 显式使用已归档的远端
训练实现，单独记录控制代码与训练实现身份；不把本地重构后的 trainer 混入该对照。
全量结果已完成：alpha=0.25 为 935/1534，alpha=0.5 为 923/1534，关闭 span control 为
933/1534；A025 相对 control 的配对为 70 gains/68 regressions。该结果只作为诊断，不改变
正式 arm；本次拥有的 vLLM 已按 manifest 自动释放。

### 2026-09-07 审计修正

已发现并记录一个概率审计实现缺陷：旧诊断 runtime 在每个 question group 调用
`smc_mode_concentration_advantages` 时重置了内存中的审计缓存，因此一个 update 的 30 个
题组最终只落盘最后一个题组。历史 `audit3` 和原始 `saam_gate60` 重跑中的 4 行日志只代表
4 个 update 的各一个题组，不能作为完整概率审计。优势选择逻辑本身未改变；修复后要求每个
4-update run 写入 120 个题组记录，并通过覆盖率测试后才允许进入 SCM 优化决策。

修复后原始 cohort 已通过 120 组/960 候选的完整字段、rollout 连接和选择一致性检查；
完整概率审计门禁已通过。新版 cohort 仍需补跑完整概率审计；已完成的 SAAM 训练不重跑。
中间评测与衔接证据：`docs/reports/rl/SCM_AUDIT_PROGRESS_20260907_ZH.md`。不改变正式 RL
方案，也不预先宣称 SCM 优化版本或 accuracy promotion 已准入。

### 2026-09-07 MDP critic 预注册门禁

本方向不引入独立 reward model：reward 仍来自 Harness 的 terminal scorer，IQL 只学习
`Q(state, action)` 与 expectile `V(state)`。第一阶段只从完整 Atomic v26 rollout 构造
semantic-step SMDP transition 并做可识别性审计；不更新 actor、不替换 SAAM/GRPO。审计必须
同时观察到同一精确 model-visible state 的 action variation 和 terminal outcome variation，
否则只能说明数据可拟合 trajectory-success predictor，不能说明 critic 能把 pass@4 集中到
greedy。审计产物需记录 checkpoint-6380、protocol/prompt hash、source hash、IQL 超参和
`diagnostic_only` 状态。

1. **KL 对照系数/调度**：是否采用固定 beta（例如 `0.01`）或预注册的 beta sweep，待
   对照启动前登记；不得在看到结果后再挑选系数。

2. **新 cohort 最终规模**：当前跨服务器去重后确认 430 道满足 2–6 的题目；补充筛选完成后再冻结约500题 manifest。

## 未决前的执行规则

- 新 RL 只能从 version26 + 当前 four-level/SAAM + reason/tool 加权 config 启动；约500题
  cohort manifest 和 matched eval 完成前，不宣称 accuracy promotion。
- A100 update-40 运行已同步：checkpoint-40 可审计，step49 因 OOM 退出；它使用旧700题/14题每update配置，只保留作历史诊断。
- 不启动新的外部 DeepSeek 生成 batch；仅允许上述用户明确授权的小样本事后过程评分。
- Qwen2.5-7B 四卡 SFT 对照使用 Qwen2.5 原生 `qwen` chat template；数据内容 SHA-256
  固定为 `c19742ec55d99980075e51a52e79285e4322f89a1d1991dea592748c279e9a14`。该运行不是
  Qwen3 SFT anchor，也不得直接作为当前 RL 起点；正式 run 已完成，最终 train loss 为
  `0.5573`，global step 为 `560`。
- 评测优先放在 table_rl/NewGNN；A100 可用时承担主 RL，A100 不可用时允许在两张空闲 3090 上
  运行降级 RL 及其 rollout serving。
- KL 是否有增益已确定要通过实验回答；在对照完成前，正式 arm 继续使用 `kl_beta=0`，且
  任何 KL 对照必须与主 run 使用同一 checkpoint、cohort、decode、runtime、GPU 拓扑和预算，
  并使用独立 output root。

## 归档迁移说明

旧 scheme 的实现目标目录是 `archive/code/`，旧 launcher/config/adapter/trajectory 目标目录
是 `archive/experiments/` 或对应历史结果目录。当前 `src/tool_modules/registry.py` 已收敛为
Atomic-only；旧 scheme 包、薄别名和仅支持旧方案的 runner 已迁入 archive，不能被默认 config
或 launcher 导入。
# 2026-09-15 table_rl 8B GPU gate（当前执行）

用户要求继续推进 GPU 实验。远端 table_rl 未发现已冻结的 120 题文件，因此注册一个独立的 8B 单 update gate：使用冻结 gate60 cohort（60 条，SHA-256 `47e9d369bf6506d13b2432ac92bb971bf52912cb759c133846801f5411ad79d5`）和 checkpoint-6380（adapter SHA-256 `8900e4c482f4624ff05059a6fecc2811fa99552cddea7550d44ce8cde853abe6`），Atomic v26、binary result、SAAM asymmetric-error、K=8、30 prompts/update。table_rl 使用 trainer/vLLM 分卡、4-bit、gradient checkpointing、SDPA、actor old-policy、micro-batch 1/4096 tokens、KL=0，仅用于验证当前优化后的 GPU 链路和显存/吞吐；完成后再决定是否扩展到 4 updates 和正式评测，不能替代 A100 主线。

# 2026-09-17 RFT（on-policy 自训练）C 档正式臂（预注册，先于启动）

用户要求在 RL 主线之外启动 RFT 验证。**判据与假设在启动前写死**：把 Qwen3-8B cumulative
SFT `checkpoint-6380` 自己采样、被 Harness 判为 correct+legal 的轨迹做离线 on-policy
自训练，目标是把已进入支持集但未成为模式的正确轨迹推成高概率输出。

- 主判据：matched BIRD-dev1534 **greedy**，对照 **8B `checkpoint-6380` SFT = 928/1534
  （60.495%）**，按题配对；单 seed MDE ≈ 2.1pp，**需要 ≥32 题净增益**才可登记为收益。
- 副判据（同批并列报告）：T=0.8 K=4 采样平均准确率。本阶段在采样分布上优化，因此采样端
  是它天然的观测面；两个终点都要报，不得只报赢的那个。
- 数据身份：`qwen3_atomic_v26_rft_c_tier_12288_20260917`，2,361 action-level 记录 /
  287 episodes（353 个 band-1–3 候选每题限 1 条 → 289 verified episode），
  `protocol_hash 4da19387399bd3a5`、`carrier think-json-v1`、与 checkpoint-6380 的 SFT
  训练集 episode **零重叠**。数据 SHA-256 记录在 `preflight.json`。
- 显式声明两项偏离 checkpoint-6380 schedule 的参数：`cutoff_len 12288`（6400 下
  prefix-complete 仅存活 141/289 episode；LF 精确 token 审计在 12288 下保留
  2361/2397 记录、287/289 episode）、`learning_rate 2e-5`（沿用 1e-4 会覆盖已有最优）。
- 训练前被排除的候选：`replay_execute_failed` 42（ProtocolError 33 / ScalarGroundingError 9）、
  `terminal_denotation_mismatch` 21、`ValueError` 1，共 64 条，不作训练数据。
- 计算：table_rl GPU0/1、4-bit base + LoRA r16、per-device batch 1 / accum 8（有效 16）、
  gradient checkpointing、bf16、cosine LR、**4 epochs = 592 步**；smoke 实测 ~66 s/步、
  显存峰值 13.9/13.2 GB，预估 ~10.9 h。
- 准入：训练后必须做 matched BIRD-dev1534 greedy 与 **fresh replay**；未通过前不得宣称
  任何能力提升，也不得与 4B 臂的 903/909 直接比较。
