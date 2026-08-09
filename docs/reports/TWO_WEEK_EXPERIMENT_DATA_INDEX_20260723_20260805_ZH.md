# 2026-07-23 至 2026-08-05 实验数据总索引

更新日期：2026-08-05  
用途：导师汇报、实验复核和后续研究决策。  
范围：过去两周所有形成正式结论的 baseline、SFT、RL、atomic 工具接口、非 atomic 工具方案和 provider-native 工具调用实验。基础设施 smoke、OOM 和未完成运行只作为审计产物列出，不计入正式结果。

## 1. 存储位置

### 本地仓库

- 仓库根目录：`/Users/hudou/Research/tabular_rl_research`
- 正式报告：`/Users/hudou/Research/tabular_rl_research/docs/reports`
- 本地原始评测结果：`/Users/hudou/Research/tabular_rl_research/data/results`
- 外部教师和工具轨迹：`/Users/hudou/Research/tabular_rl_research/data/trajectories`
- 冻结评测输入：`/Users/hudou/Research/tabular_rl_research/data/eval_inputs`

### table_rl 服务器

- 总根目录：`table_rl:/home/dengyan/tabular_rl_outputs`
- RL checkpoint：`table_rl:/home/dengyan/tabular_rl_outputs/checkpoints`
- version36 全量评测：`table_rl:/home/dengyan/tabular_rl_outputs/eval_runtime_version36_20260728/data/results`
- Exp12–Exp17 训练池：`table_rl:/home/dengyan/tabular_rl_outputs/phase8_controlled_20260801`
- Exp18 dense scale120：`table_rl:/home/dengyan/tabular_rl_outputs/phase8_dense_stage2_20260802`
- teacher-union 60：`table_rl:/home/dengyan/tabular_rl_outputs/streaming_teacher_union_20260803`
- teacher-union strict120：`table_rl:/home/dengyan/tabular_rl_outputs/streaming_teacher_union_scale120_v2_20260804`

服务器路径需要先执行 `ssh table_rl` 后读取。大模型 checkpoint 和全量 RL 轨迹只保存在服务器，不复制进 Git 仓库。

## 2. 主要结果汇总

### 2.1 本地 Coder-7B Direct-SQL baseline

| 实验 | Greedy / K=4 | 结果 | 原始数据 | 报告 |
|---|---|---:|---|---|
| Qwen2.5-Coder-7B-Instruct Direct SQL | greedy | 748/1534 = 48.76% | `data/results/qwen2.5_coder_7b_bird_direct_sql_base_greedy_rp105_dev1534_bird_ex/` | `docs/reports/evaluation/BIRD_QWEN25_CODER_7B_DIRECT_SQL_BASELINE_20260723.md` |
| Qwen2.5-Coder-7B-Instruct Direct SQL | sampled K=4 pass@1 | 727/1534 = 47.39% | `data/results/qwen2.5_coder_7b_bird_direct_sql_base_passk4_rp105_dev1534_bird_ex/` | 同上 |
| Qwen2.5-Coder-7B-Instruct Direct SQL | sampled K=4 pass@2 | 820/1534 = 53.46% | 同上 | 同上 |
| Qwen2.5-Coder-7B-Instruct Direct SQL | sampled K=4 pass@4 | 904/1534 = 58.93% | 同上 | 同上 |

相关对照：

- Qwen2.5-7B-Instruct greedy：`data/results/qwen2.5_7b_bird_direct_sql_base_greedy_dev1534_bird_ex/`
- Qwen2.5-7B-Instruct K=4：`data/results/qwen2.5_7b_bird_direct_sql_base_passk4_dev1534_bird_ex/`
- OmniSQL 报告：`docs/reports/evaluation/OMNISQL_7B_BIRD_REPRODUCTION_20260723.md`
- SQL-Astra 报告：`docs/reports/evaluation/BIRD_QWEN25_7B_SQL_ASTRA_BASELINE_REPRODUCTION_20260729.md`
- 当前 SQL-Astra 复现：`docs/reports/evaluation/SQL_ASTRA_QWEN25_7B_INSTRUCT_BASELINE_REPRODUCTION_20260805_ZH.md`

### 2.2 SFT2 与完整 BIRD-dev RL 结果

共同评测：BIRD-dev 1534、greedy@1、temperature=0、top_p=1、version36、`bird-set`、max_steps=30。

| 实验 | 方法 | Correct | Valid | Avg steps | 正式评测目录 |
|---|---|---:|---:|---:|---|
| SFT2 | checkpoint-1682 对照 | 762/1534 = 49.67% | 1194 | 8.371 | `table_rl:/home/dengyan/tabular_rl_outputs/eval_runtime_version36_20260728/data/results/sft2_version36_dev1534_greedy_t0_logprobs20_20260801/` |
| Exp12 | fixed process-only | 761/1534 = 49.61% | 1205 | 8.216 | `table_rl:/home/dengyan/tabular_rl_outputs/eval_runtime_version36_20260728/data/results/stage1_exp12_fixed_process_only_version36_dev1534_greedy_t0_p1_logprobs20_bird_set_20260802/` |
| Exp13 | rank-only action-mean | 762/1534 = 49.67% | 1189 | 8.475 | `table_rl:/home/dengyan/tabular_rl_outputs/eval_runtime_version36_20260728/data/results/stage1_exp13_fixed_rank_only_action_mean_version36_dev1534_greedy_t0_p1_logprobs20_bird_set_20260802/` |
| Exp14 | process + rank action-mean | 771/1534 = 50.26% | 1207 | 8.188 | `table_rl:/home/dengyan/tabular_rl_outputs/eval_runtime_version36_20260728/data/results/stage1_exp14_fixed_process_rank_action_mean_version36_dev1534_greedy_t0_p1_logprobs20_bird_set_20260802/` |
| Exp15 | fixed-prefix Action-DPO | 776/1534 = 50.59% | 1208 | 8.610 | `table_rl:/home/dengyan/tabular_rl_outputs/eval_runtime_version36_20260728/data/results/stage1_exp15_fixed_prefix_action_dpo_version36_dev1534_greedy_t0_p1_logprobs20_bird_set_20260802/` |
| Exp16 | dense uniform full-response | 764/1534 = 49.80% | 1222 | 8.106 | `table_rl:/home/dengyan/tabular_rl_outputs/eval_runtime_version36_20260728/data/results/stage1_exp16_dense_uniform_full_response_version36_dev1534_greedy_t0_p1_logprobs20_bird_set_20260802/` |
| Exp17 | dense strategic/backslice | 760/1534 = 49.54% | 1199 | 8.286 | `table_rl:/home/dengyan/tabular_rl_outputs/eval_runtime_version36_20260728/data/results/stage1_exp17_dense_strategic_full_response_version36_dev1534_greedy_t0_p1_logprobs20_bird_set_20260802/` |
| Exp18 seed202 | dense uniform scale120，单 seed 探索 | 764/1534 = 49.80% | 1196 | 8.396 | `table_rl:/home/dengyan/tabular_rl_outputs/phase8_dense_stage2_20260802/stage2_scale120_seed202_full_dev_summary.json` |
| Teacher-union 60 | SFT2+Exp15 MOPD + repair DPO | **785/1534 = 51.17%** | **1237** | 8.306 | `table_rl:/home/dengyan/tabular_rl_outputs/eval_runtime_version36_20260728/data/results/streaming_sft2_exp15_opd_repair_dpo60_version36_dev1534_greedy_t0_p1_logprobs20_bird_set_20260803/` |
| Teacher-union strict120 | task-balanced multi-repair v2 | 765/1534 = 49.87% | 1208 | 8.296 | `table_rl:/home/dengyan/tabular_rl_outputs/eval_runtime_version36_20260728/data/results/streaming_sft2_exp15_strict120_multi_repair_v2_version36_dev1534_greedy_t0_p1_logprobs20_bird_set_20260804/` |

每个正式评测目录中的关键文件结构相同：

- `accuracy.json`：主要准确率；
- `valid_rate.json`：合法终止率；
- `avg_steps.json`：平均步数；
- `per_question_result.json`：逐题配对数据；
- `all.jsonl`：完整轨迹；
- `success.jsonl` / `failure.jsonl`：成功与失败分片；
- `failure_types.json`：失败类型；
- `action_distribution.json`：工具分布；
- `manifest.json` / `evaluation_protocol.json`：冻结协议；
- `summary.json`：评测摘要。

### 2.3 Teacher-union 60/120 配对统计

| 比较 | Gains | Regressions | Net | Exact paired p | 摘要文件 |
|---|---:|---:|---:|---:|---|
| union60 vs SFT2 | 87 | 64 | +23 | 0.07305 | `table_rl:/home/dengyan/tabular_rl_outputs/streaming_teacher_union_20260803/full_dev_summary.json` |
| strict120 vs SFT2 | 78 | 75 | +3 | 0.87162 | `table_rl:/home/dengyan/tabular_rl_outputs/streaming_teacher_union_scale120_v2_20260804/full_dev_summary.json` |
| strict120 vs union60 | 62 | 82 | -20 | 0.11303 | 由两个 `per_question_result.json` 逐题配对计算 |

## 3. RL 训练数据、checkpoint 和审计路径

### 3.1 固定初始化

- SFT2 checkpoint：`table_rl:/home/dengyan/tabular_rl_outputs/checkpoints/qwen2.5-coder-7b-bird-sft2-batch2-cp560-repeated-only-6400-single-gpu-qlora/checkpoint-1682/`
- Adapter SHA-256：`d880e2d7cc3203fdb0d11a7c188d8f607fd297b174eff23f741b6fe73cc3ce6e`

### 3.2 Exp12–Exp17 训练池

- Rank 视图：`table_rl:/home/dengyan/tabular_rl_outputs/phase8_controlled_20260801/balanced_mixed60_rank_seed101/validated_trajectories.jsonl`
- Process 视图：`table_rl:/home/dengyan/tabular_rl_outputs/phase8_controlled_20260801/balanced_mixed60_process_seed101/validated_trajectories.jsonl`
- Dense uniform：`table_rl:/home/dengyan/tabular_rl_outputs/phase8_controlled_20260801/balanced_mixed60_dense_uniform_seed101/validated_trajectories.jsonl`
- Dense strategic：`table_rl:/home/dengyan/tabular_rl_outputs/phase8_controlled_20260801/balanced_mixed60_dense_strategic_seed101/validated_trajectories.jsonl`
- Exp15 same-prefix pairs：`table_rl:/home/dengyan/tabular_rl_outputs/phase8_controlled_20260801/stage1_balanced_mixed60_v3/fixed_prefix_train_seed101/fixed_prefix_train.verified.jsonl`
- 每个目录的 `manifest.json` 是对应训练数据冻结清单。

### 3.3 Exp12–Exp17 checkpoint

- Exp12：`table_rl:/home/dengyan/tabular_rl_outputs/checkpoints/trl-transition-v26-exp12-fixed-process-only-sft2-60xk4-seed101-20260801/`
- Exp13：`table_rl:/home/dengyan/tabular_rl_outputs/checkpoints/trl-transition-v26-exp13-fixed-rank-only-action-mean-sft2-60xk4-seed101-20260801/`
- Exp14：`table_rl:/home/dengyan/tabular_rl_outputs/checkpoints/trl-transition-v26-exp14-fixed-process-rank-action-mean-sft2-60xk4-seed101-20260801/`
- Exp15：`table_rl:/home/dengyan/tabular_rl_outputs/checkpoints/trl-transition-v26-exp15-fixed-prefix-action-dpo-sft2-seed101-20260801/`
- Exp16：`table_rl:/home/dengyan/tabular_rl_outputs/checkpoints/trl-transition-v26-exp16-dense-uniform-full-response-sft2-60xk4-seed101-20260802/`
- Exp17：`table_rl:/home/dengyan/tabular_rl_outputs/checkpoints/trl-transition-v26-exp17-dense-strategic-full-response-sft2-60xk4-seed101-20260802/`

### 3.4 Exp18 dense scale120

- Uniform 120 题/480轨迹：`table_rl:/home/dengyan/tabular_rl_outputs/phase8_dense_stage2_20260802/balanced_mixed120_dense_uniform_seed101/validated_trajectories.jsonl`
- Strategic 视图：`table_rl:/home/dengyan/tabular_rl_outputs/phase8_dense_stage2_20260802/balanced_mixed120_dense_strategic_seed101/validated_trajectories.jsonl`
- Rank 视图：`table_rl:/home/dengyan/tabular_rl_outputs/phase8_dense_stage2_20260802/balanced_mixed120_rank_seed101/validated_trajectories.jsonl`
- seed202 checkpoint：`table_rl:/home/dengyan/tabular_rl_outputs/checkpoints/trl-transition-v26-exp18-dense-uniform-full-response-scale120-sft2-120xk4-trainseed202-20260802/`
- 单 seed 汇总：`table_rl:/home/dengyan/tabular_rl_outputs/phase8_dense_stage2_20260802/stage2_scale120_seed_summary.json`
- seed101 在40/120取消，seed303未启动；不得把它们计作正式完成 seed。

### 3.5 Teacher-union 60

- 教师资格：`table_rl:/home/dengyan/tabular_rl_outputs/streaming_teacher_union_20260803/teacher_eligibility_balanced120.jsonl`
- 教师资格 manifest：`table_rl:/home/dengyan/tabular_rl_outputs/streaming_teacher_union_20260803/teacher_eligibility_balanced120.manifest.json`
- 正式训练根目录：`table_rl:/home/dengyan/tabular_rl_outputs/streaming_teacher_union_20260803/formal60_batch2_repairk4_restart1_micro4/`
- 训练配置：上述目录的 `run_manifest.json`
- 逐题 rollout/teacher/repair 记录：上述目录的 `tasks.jsonl`
- 每批 optimizer 更新：上述目录的 `updates.jsonl`
- 最终 adapter：上述目录的 `final/`
- 全量评测比较：`table_rl:/home/dengyan/tabular_rl_outputs/streaming_teacher_union_20260803/full_dev_summary.json`

### 3.6 Teacher-union strict120

- 扩展教师资格：`table_rl:/home/dengyan/tabular_rl_outputs/streaming_teacher_union_scale120_v2_20260804/teacher_eligibility_mixed238.jsonl`
- 冻结 strict120 routing：`table_rl:/home/dengyan/tabular_rl_outputs/streaming_teacher_union_scale120_v2_20260804/teacher_eligibility_strict120.jsonl`
- 正式训练根目录：`table_rl:/home/dengyan/tabular_rl_outputs/streaming_teacher_union_scale120_v2_20260804/formal120_strict_batch2_repairk4_v2/`
- 训练配置：上述目录的 `run_manifest.json`
- 逐题 rollout/teacher/repair：上述目录的 `tasks.jsonl`
- 每批 optimizer 更新：上述目录的 `updates.jsonl`
- 最终 adapter：上述目录的 `final/`
- 全量评测比较：`table_rl:/home/dengyan/tabular_rl_outputs/streaming_teacher_union_scale120_v2_20260804/full_dev_summary.json`

### 3.7 RL 总报告

- RL 方法全集与论文来源：`docs/reports/rl/RL_METHODS_AND_PAPER_PROVENANCE_20260805_ZH.md`
- Exp0–Exp11：`docs/reports/rl/RL_EXPERIMENTS_EXP0_EXP11_COMPLETE_REPORT_20260801_ZH.md`
- SFT2/result-only/process matched K4：`docs/reports/rl/BIRD_SFT2_RESULT_ONLY_PROCESS_RL_MATCHED_K4_COMPARISON_20260730_ZH.md`
- 简单 process pilot：`docs/reports/rl/BIRD_SFT2_SIMPLE_PROCESS_LR5E6_STEP23_TRAJECTORY_REWARDS_20260729_ZH.md`
- Grounding shortcut 重审：`docs/reports/rl/GROUNDING_SHORTCUT_REAUDIT_20260724.md`
- Grounding change audit：`docs/reports/rl/CODEX_GROUNDING_CHANGE_AUDIT_20260730_ZH.md`
- Process reward 精度敏感性：`docs/reports/rl/ATOMIC_PROCESS_REWARD_PRECISION_SENSITIVITY_AUDIT_20260724.md`

## 4. Atomic 工具与接口实验

### 4.1 主要决策结果

| 实验 | 结果 | 原始数据根目录 | 正式报告 |
|---|---:|---|---|
| version24 Flash fixed200 | 145/200 | `data/trajectories/tool_usability_20260724/` | `docs/reports/evaluation/BIRD_VERSION24_RELATION_DERIVATION_FIXED200_20260724.md` |
| version24 Pro capability ceiling | 145/200，和 Flash 持平 | `data/trajectories/model_capability_ceiling_20260803/` | `docs/reports/evaluation/BIRD_ATOMIC_VERSION24_FLASH20_PRO200_CAPABILITY_CEILING_20260803_ZH.md` |
| version40 concise history Gate50 | 37/50 vs version24 42/50 | `data/trajectories/tool_usability_20260729/` | `docs/reports/evaluation/BIRD_ATOMIC_VERSION40_CONCISE_HISTORY_GATE50_20260729_ZH.md` |
| version41 output correction Gate16 | 10/16 | 同上 | `docs/reports/evaluation/BIRD_ATOMIC_VERSION41_OUTPUT_CORRECTION_GATE16_20260729_ZH.md` |
| version42 terminal columns Gate16 | 12/16 | 同上 | `docs/reports/evaluation/BIRD_ATOMIC_VERSION42_TERMINAL_COLUMNS_GATE16_20260729_ZH.md` |
| version43 Gate16 / first50 | 13/16；39/50 | 同上 | `docs/reports/evaluation/BIRD_ATOMIC_VERSION43_UNIQUE_BARE_TERMINAL_COLUMNS_20260729_ZH.md` |
| version44 search Gate50 | 38/50 vs version39 39/50 | `data/trajectories/version44_paired_gate50_20260730/` | `docs/reports/evaluation/BIRD_ATOMIC_VERSION44_SEARCH_VALUES_PAIRED_GATE50_20260730_ZH.md` |
| version45 bounded search Gate50 | 37/50 vs version39 39/50 | `data/trajectories/version45_paired_gate50_retry_20260730/` | `docs/reports/evaluation/BIRD_ATOMIC_VERSION45_BOUNDED_SEARCH_PAIRED_GATE50_20260730_ZH.md` |
| version46–49 context cards Gate16 | 8/15、8/15、9/15、9/15 | `data/trajectories/context_handle_card_ablation_20260731/` | `docs/reports/evaluation/BIRD_ATOMIC_CONTEXT_HANDLE_CARD_ABLATION_GATE16_20260731_ZH.md` |
| version49 K=3 stability | 与 version39 同为20/24，成本更高 | `data/trajectories/context_stability_k3_20260731/` | `docs/reports/evaluation/BIRD_ATOMIC_VERSION49_CONTEXT_STABILITY_K3_GATE8_20260731_ZH.md` |

### 4.2 Atomic 原始数据根目录全集

- `data/trajectories/plan_ablation_20260723/`
- `data/trajectories/tool_usability_20260723/`
- `data/trajectories/tool_usability_20260724/`
- `data/trajectories/batch2_cp560_rollout3000_20260726/`
- `data/trajectories/atomic_tool_gap_20260727/`
- `data/trajectories/bird_train_version11_failed70_version37_recent4_20260728/`
- `data/trajectories/sft2_step1682_repeat_recovery_20260728/`
- `data/trajectories/bird_train_version38_semantic_discipline_gate32_20260729/`
- `data/trajectories/tool_usability_20260729/`
- `data/trajectories/bird_train_atomic39_hard60_context_ablation_20260729/`
- `data/trajectories/bird_train_atomic_full_bird_context_gate32_20260729/`
- `data/trajectories/bird_train_atomic_full_bird_context_with_inspect_gate32_20260729/`
- `data/trajectories/bird_train_atomic_full_bird_context_fixed200_20260729/`
- `data/trajectories/schema_context_ablation_20260729/`
- `data/trajectories/version24_search_single_variable_20260730/`
- `data/trajectories/version24_sql_aligned_gate20_20260730/`
- `data/trajectories/version24_sql_aligned_pagination_probe_02179_20260730/`
- `data/trajectories/version44_external_interface_smoke_20260730/`
- `data/trajectories/version44_external_interface_exercise_20260730/`
- `data/trajectories/version44_paired_gate50_20260730/`
- `data/trajectories/version45_paired_gate50_20260730/`
- `data/trajectories/version45_paired_gate50_retry_20260730/`
- `data/trajectories/context_handle_card_ablation_20260731/`
- `data/trajectories/context_stability_k3_20260731/`
- `data/trajectories/model_capability_ceiling_20260803/`

每个原始轨迹目录通常包含 `all/verified/failures.jsonl`、manifest、逐任务记录以及 replay/structure/no-leak audit。结论以同名 `docs/reports/evaluation/` 报告为准。

## 5. Action-block 与 Relational-program

### Action-block

- 原始数据：`data/trajectories/batch_plan_20260726/`
- 补充结果：`data/results/action_block_v33_sequential_segment_pilot20_deepseek_v4_flash_20260727/`
- `data/results/action_block_v34_one_edge_join_gate14_deepseek_v4_flash_20260727/`
- `data/results/action_block_v35_scalar_cell_gate4_deepseek_v4_flash_20260727/`
- 主要报告：
  - `docs/reports/evaluation/BIRD_ACTION_BLOCK_PILOT20_20260726.md`
  - `docs/reports/evaluation/BIRD_ACTION_BLOCK_V18_OPTIMIZATION_PILOT20_20260726_ZH.md`
  - `docs/reports/evaluation/BIRD_ACTION_BLOCK_V21_FIXED200_20260726_ZH.md`
  - `docs/reports/evaluation/BIRD_ACTION_BLOCK_V32_FIXED200_20260726_ZH.md`
  - `docs/reports/evaluation/BIRD_ACTION_BLOCK_V34_ONE_EDGE_JOIN_PILOT20_20260727_ZH.md`

### Relational-program

- `data/results/relational_program_v1_deepseek_v4_flash_pilot20_20260727/`
- `data/results/relational_program_v2_deepseek_v4_flash_pilot20_20260727/`
- `data/results/relational_program_v3_deepseek_v4_flash_pilot20_20260727/`
- `data/results/relational_program_v4_typed_refs_pilot20_deepseek_v4_flash_20260727/`
- `data/results/relational_program_v5_typed_scalar_gate6_deepseek_v4_flash_20260727/`
- `data/results/relational_program_v6_base_role_gate4_deepseek_v4_flash_20260727/`
- 报告：`docs/reports/evaluation/BIRD_RELATIONAL_PROGRAM_V3_PILOT20_20260727_ZH.md` 至 `BIRD_RELATIONAL_PROGRAM_V6_BASE_ROLE_GATE4_20260727_ZH.md`。

## 6. Direct-SQL-search 与 Iterative-SQL

### Direct-SQL-search

- v1/早期实验：`data/trajectories/direct_sql_search_20260801/`
- v2 授权后正式 holdout：`data/trajectories/direct_sql_search_20260803/`
- v2：6/15，atomic version39 对照：10/15；Direct 使用约38.3%的 token。
- 报告：`docs/reports/evaluation/BIRD_DIRECT_SQL_SEARCH_V2_OPTIMIZATION_HOLDOUT_GATE15_20260801_ZH.md`

### Iterative-SQL

- 7月29日 Gate32：
  - `data/trajectories/bird_train_semantic_gate32_iterative_sql_v1_ds_v4_flash_20260729/`
  - `data/trajectories/bird_train_semantic_gate32_iterative_sql_v2_ds_v4_flash_20260729/`
  - `data/trajectories/bird_train_semantic_gate32_iterative_sql_full_schema_samples2_ds_v4_flash_20260729/`
- 8月5日 v4/v5/v6：`data/trajectories/iterative_sql_20260805/`
- Prefix50：v5 36/50，v4 34/50，p=0.6875；v5 legal/errors/tokens 未通过一致性要求。
- 报告：
  - `docs/reports/evaluation/BIRD_ITERATIVE_SQL_V2_GATE32_DEEPSEEK_V4_FLASH_20260729_ZH.md`
  - `docs/reports/evaluation/BIRD_ITERATIVE_SQL_FULL_SCHEMA_SAMPLES_GATE32_20260729_ZH.md`
  - `docs/reports/evaluation/BIRD_ITERATIVE_SQL_V4_OPTIMIZATION_GATE15_20260805_ZH.md`
  - `docs/reports/evaluation/BIRD_ITERATIVE_SQL_V5_INDEPENDENT_PREFIX20_GATE_20260805_ZH.md`
  - `docs/reports/evaluation/BIRD_ITERATIVE_SQL_V5_PREFIX50_EXPANSION_20260805_ZH.md`
  - `docs/reports/evaluation/BIRD_ITERATIVE_SQL_V6_DISJOINT_GATE20_20260805_ZH.md`
  - `docs/reports/evaluation/BIRD_ITERATIVE_SQL_V6_TARGET_GATE20_RESULT_20260805_ZH.md`

## 7. Native provider tool calls：version50/51

| 实验 | Correct | Legal | Process errors | 原始数据 | 报告 |
|---|---:|---:|---:|---|---|
| version50 single native fixed200 | 133/200 | 183/200 | 166 | `data/trajectories/version50_native_tool_calls_20260805/fixed200_user_override/` | `docs/reports/evaluation/BIRD_ATOMIC_VERSION50_NATIVE_TOOL_CALLS_FIXED200_20260805_ZH.md` |
| version51 native-tool-bundle Gate32 | 22/32 | 32/32 | 11 | `data/trajectories/version51_native_tool_bundle_20260805/gate32/` | `docs/reports/evaluation/BIRD_VERSION51_NATIVE_TOOL_BUNDLE_GATE32_20260805_ZH.md` |
| version51 native-tool-bundle fixed200 | **147/200** | **200/200** | 54 | `data/trajectories/version51_native_tool_bundle_20260805/fixed200/` | `docs/reports/evaluation/BIRD_VERSION51_NATIVE_TOOL_BUNDLE_FIXED200_20260805_ZH.md` |

正式 version51 fixed200 关键文件：

- `data/trajectories/version51_native_tool_bundle_20260805/fixed200/all.jsonl`
- `data/trajectories/version51_native_tool_bundle_20260805/fixed200/verified.jsonl`
- `data/trajectories/version51_native_tool_bundle_20260805/fixed200/failures.jsonl`
- `data/trajectories/version51_native_tool_bundle_20260805/fixed200/verified.manifest.json`
- `data/trajectories/version51_native_tool_bundle_20260805/fixed200/structural_audit.json`
- `data/trajectories/version51_native_tool_bundle_20260805/fixed200/native_bundle_audit.json`

## 8. SFT 与因果数据生成

- 外部教师 fixed1000：`data/trajectories/external_teacher_fixed1000_20260724/`
- 固定输入：`data/eval_inputs/bird_train_external_teacher_fixed1000.jsonl`
- SFT 报告：`docs/reports/sft/BIRD_EXTERNAL_TEACHER_FIXED1000_DUAL_7B_SFT_20260724.md`
- Student prompt 因果实验：`docs/reports/sft/BIRD_STUDENT_FORMAL_PROMPT_CAUSAL_GATE_20260725.md`
- Prompt early checkpoint：`docs/reports/sft/BIRD_STUDENT_PROMPT_EARLY_CHECKPOINT_GATE_20260725.md`
- Coder carrier repair：`docs/reports/sft/BIRD_QWEN25_CODER_CARRIER_REPAIR_20260725.md`
- OmniSQL SFT union：`docs/reports/sft/BIRD_OMNISQL_SFT1_UNION_LAUNCH_20260801_ZH.md`
- SFT2 repeat/recovery 诊断：`data/trajectories/sft2_step1682_repeat_recovery_20260728/`

## 9. 失败、OOM 和不应作为正式结果使用的产物

这些路径只用于工程审计：

- union60 首次 OOM：`table_rl:/home/dengyan/tabular_rl_outputs/streaming_teacher_union_20260803/formal60_batch2_repairk4.failed_repair_batch16_oom_8tasks_20260803T080021Z/`
- union60 早期未完成目录：`table_rl:/home/dengyan/tabular_rl_outputs/streaming_teacher_union_20260803/formal60_batch2_repairk4/`
- union60 多个 smoke/OOM 目录：`table_rl:/home/dengyan/tabular_rl_outputs/streaming_teacher_union_20260803/smoke2_batch2_repairk2.failed_*/`
- Exp13 scalar-shuffle 失败：`table_rl:/home/dengyan/tabular_rl_outputs/checkpoints/trl-transition-v26-exp13-fixed-rank-only-action-mean-sft2-60xk4-seed101-20260801.failed_scalar_shuffle_20260801T1441Z/`
- Exp15 question-expanded 多个 OOM checkpoint：`table_rl:/home/dengyan/tabular_rl_outputs/checkpoints/trl-transition-v26-exp15-question-expanded-strict-action-dpo-sft2-seed101-20260804-oom-*/`
- Exp18 seed101：40/120 后取消，仅审计，不恢复；seed303 从未启动。
- version51 `gate32_aborted_limit20/`：旧动作预算中止产物，不属于正式 Gate32。
- Direct-SQL-search `holdout15_v2_r1/`：首次 transport-only 失败，正式语义结果在 `direct_sql_search_20260803/`。

## 10. 使用建议

1. 导师汇报的主表使用第2节；不要把不同模型、不同 cohort 的 fixed200 与本地 Coder full-dev 直接排成同一准确率排名。
2. RL 统计与逐题分析从服务器正式评测目录的 `per_question_result.json` 读取。
3. Teacher-union 的监督效率从正式训练目录的 `tasks.jsonl` 和 `updates.jsonl` 读取。
4. Atomic/provider 实验的结论以对应正式 Markdown 报告为准，原始轨迹用于失败审计与复算。
5. 任何名称含 `failed`、`oom`、`smoke`、`aborted` 或 `probe` 的目录均不得作为正式模型分数。
