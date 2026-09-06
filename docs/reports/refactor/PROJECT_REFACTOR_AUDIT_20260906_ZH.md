# 项目重构与清理审计（2026-09-06）

## 本轮范围

- 全仓库扫描当前入口、脚本引用和历史协议文件。
- 旧协议 version40–49 及其专用测试/审计代码移入
  `archive/code/legacy_compatibility/atomic_versions40_49/`，文件内容保留。
- 根目录未被当前入口引用的 `tmp_*.py`、`tmp_run_diag_*.sh` 移入
  `archive/code/scratch/`，文件内容保留。
- Direct-SQL 公共模块暂留在 `src/eval/`，因为现有 `src/eval` 测试和历史评测脚本仍直接导入；
  直接迁移会破坏评测链路，列为后续逐项处理的冲突项。
- `src/rl/experiments` 中的 version26 高相似 launcher（SAAM、DDP/FSDP、gate60/diagnostic14、
  candidate-only、vanilla/earlystop）暂不合并：它们携带不同的实验身份、GPU 拓扑、cohort 或
  contract，且部分是当前未提交文件。后续应抽取只读公共 shell 函数，但保留每个正式入口。

## 当前主线检查

- Atomic version26 launcher、TRL wrapper、matched evaluator：`bash -n` 通过。
- 当前 `src/rl`、`src/harness`、`src/eval`：Python compileall 通过。
- 归档文件与迁移前 `HEAD` 内容逐一 SHA-256 对比：全部一致。
- `git diff --check`：通过。
- 本地 pytest 未能完成收集：可变 checkout 的 `src/sft/protocol.py` 声明为 version39，
  而 active RL 模块按契约拒绝非 version26。正式运行必须使用固定 version26 runtime。
- 本地 vanilla matched evaluator 的 `--plan` 被 pinned contract SHA-256 gate 拒绝；当前
  checkout 中该 contract 已有用户未提交修改，实际哈希与 launcher 内登记值不同。本轮没有
  覆盖这些修改。

## 本轮按文件夹处理结果

- `src/sft/`：version40–49 协议链已迁移；当前 version26 共享模块未改写。
- `src/rl/experiments/`：version36、Exp10–11、Exp15 历史重复 launcher 已迁移；当前 v26
  变体保留并记录身份差异。
- `src/rl/evaluation/`：没有移动当前 v26 evaluator；Direct-SQL 依赖仍待单独迁移设计。
- 根目录：未引用 scratch 已迁移；`rollout.remote.current.py` 等快照因用途不明确保留。

## 三台服务器只读 smoke

| 主机 | 结果 |
|---|---|
| `a100` | matched-eval `--plan` 成功；固定 runtime verifier 报内容树文件数不匹配（期望 101，实际 137），因此不能宣称 runtime gate 通过。该主机 checkout 没有正式三卡 RL launcher。 |
| `table_rl` | 固定 runtime 可导入，报告 `version26`、`atomic think-json-v1`；主机 checkout 没有当前 v26 RL/eval launcher。 |
| `NewGNN` | 固定 runtime 可导入，报告 `version26`、`atomic think-json-v1`；主机 checkout 没有 `src/rl`，不能从该 checkout 启动 RL preflight。 |

本轮没有启动训练、vLLM、正式评测或外部 API 请求。

## 工作树状态

本轮开始前工作树已有大量用户修改和未跟踪文件；本轮没有覆盖或丢弃它们。当前环境的
`.git/index` 为只读，无法代替用户执行 staging/commit，因此工作树仍会显示 dirty。归档
迁移会在下一次 `git add -A` 时被 Git 识别为 rename；若要得到真正 clean 的 commit，需要
先决定是否把已有用户修改一起提交。
