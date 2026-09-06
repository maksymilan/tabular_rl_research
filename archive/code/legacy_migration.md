# Legacy code migration map

更新时间：2026-09-03

旧工具/实验代码统一迁入 archive，先不删除。当前 active 入口只允许 Atomic version26；本表
记录已完成的物理迁移，保留原文件内容以保持可复现 identity。

| 原位置 | 归档位置 | 备注 |
|---|---|---|
| `src/tool_modules/checkpoint_relalg/` | `archive/code/legacy_tool_modules/checkpoint_relalg/` | checkpoint-relalg diagnostic |
| `src/tool_modules/action_block/` | `archive/code/legacy_tool_modules/action_block/` | action-block diagnostic |
| `src/tool_modules/direct_sql_search/` | `archive/code/legacy_tool_modules/direct_sql_search/` | direct-SQL diagnostic |
| `src/tool_modules/iterative_sql/` | `archive/code/legacy_tool_modules/iterative_sql/` | iterative-SQL diagnostic |
| `src/tool_modules/native_tool_bundle/` | `archive/code/legacy_tool_modules/native_tool_bundle/` | v51–v54 diagnostic |
| `src/tool_modules/relational_program/` | `archive/code/legacy_tool_modules/relational_program/` | relational-program diagnostic |
| `src/tool_modules/_bootstrap.py` | `archive/code/legacy_compatibility/tool_modules_bootstrap.py` | temporary flat-import bridge |
| thin aliases under `src/eval/` and `src/sft/` | `archive/code/legacy_compatibility/` | no active compatibility stubs remain |
| branch-only RL/SFT/eval files importing retired schemes | `archive/code/legacy_compatibility/` | replay/audit only |
| historical RL/SFT/eval launchers and configs | `archive/experiments/{rl,sft,evaluation}/` | preserve manifest and source hash |
| `src/sft/atomic_version40.py`–`atomic_version49.py` and prompt layers | `archive/code/legacy_compatibility/atomic_versions40_49/` | retired protocol chain |
| `src/sft/tests/test_atomic_version40.py`–`test_atomic_version49.py` | `archive/code/legacy_compatibility/atomic_versions40_49/` | tests for retired variants |
| `src/eval/audit_context_handle_card_ablation.py` | `archive/code/legacy_compatibility/atomic_versions40_49/` | historical context-card audit |
| root `tmp_*.py` and `tmp_run_diag_*.sh` scratch files | `archive/code/scratch/` | unreferenced one-off diagnostics, retained verbatim |
| `src/rl/experiments/run_phase0_sft2_equal300_passk4_table_rl.sh`, `run_phase1_process_current_equal300_passk4_table_rl.sh` | `archive/experiments/evaluation/legacy_version36/` | duplicate frozen version36 launchers, retained for replay |
| `src/rl/experiments/run_exp15_exact_prefix_*.sh`, `finalize*exp15_exact_prefix*.sh` | `archive/experiments/rl/legacy_exp15_exact_prefix/` | duplicate frozen Action-DPO launchers |
| `src/rl/experiments/{run,watchdog}*rank*followup*queue_table_rl.sh` | `archive/experiments/rl/legacy_exp10_11/` | frozen rank follow-up queue/watchdog group |

## 迁移原则

1. 迁移保留完整源文件；归档目录记录原路径、目标路径和依赖边界。
2. `src/tool_modules/registry.py` 现在只暴露 Atomic version26，不再包含旧 scheme builder 或
   compatibility 常量；旧代码若需重放，必须显式恢复归档路径。
3. 归档代码不得被当前 version26 SFT/RL 默认 config、launcher、runtime source snapshot
   或 output root 导入。
4. checkpoint、trajectory、report 暂不删除；物理移动后仍保留原 manifest 和独立 replay 入口。
