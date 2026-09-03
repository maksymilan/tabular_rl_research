# RL 完成后的统一评测衔接

更新时间：2026-09-03

主 RL 只在 `a100` 运行。训练完成后停止 vLLM 并封存完整 artifact，再把候选 checkpoint
交给 `table_rl` 或 `NewGNN` 的 version26 matched evaluator；不要在主 RL GPU 上临时改成
另一种协议，也不要把在线 rollout 的部分结果当成评测结果。

## 资源分工

- A100：两卡 FSDP trainer + 一卡 online vLLM，负责 RL optimizer 和 rollout。
- table_rl：2 × RTX 3090，候选/基线 matched evaluation、行为诊断。
- NewGNN：8 × RTX 3090，SFT、评测、数据准备和行为诊断。

## 标准入口

使用 `src/rl/evaluation/run_v26_matched_eval_handoff.sh`，先 `--plan` 再 `--run`。训练目录
必须是完整 run root，不是只有 `final/` adapter 的目录；显式绑定：

```bash
RUNTIME=/home/dengyan/tabular_rl_outputs/runtime/version26-4cd47c957fc6ae791e76a10594c8cd22f4d3b6de
OUTPUT_ROOT=/home/dengyan/tabular_rl_outputs/evaluations/my_version26_eval
CANDIDATE_RUN=/home/dengyan/tabular_rl_outputs/<completed_rl_run>
SAAM_RUN=/home/dengyan/tabular_rl_outputs/<matched_baseline_run>

env RUNTIME="$RUNTIME" OUTPUT_ROOT="$OUTPUT_ROOT" \
  CANDIDATE_RUN="$CANDIDATE_RUN" SAAM_RUN="$SAAM_RUN" \
  CANDIDATE_GPU=0 SAAM_GPU=1 CANDIDATE_PORT=8087 SAAM_PORT=8088 \
  bash src/rl/evaluation/run_v26_matched_eval_handoff.sh --plan
```

实际脚本变量名若要求 `SAAM_RUN` 等历史兼容名称，以脚本的 `--plan` 输出为准；不得手工
复制 detached SSH 命令。GPU id 和端口只对本次评测有效，并写入 evaluation identity。

## fail-closed 检查

启动前必须验证：

- run manifest、implementation lock、training precision 和最终 checkpoint 完整；
- protocol/runtime/prompt/carrier/history 与 version26 一致；
- candidate 与 baseline 使用同一 cohort、decode、max steps、scorer 和 evaluator；
- 输出目录为空或明确新建，GPU/端口未被占用；
- evaluator 完成后题号覆盖完整、无重复、无 API/断线污染，且存在 `status.json` 和
  `summary.json`。

只有 candidate 和 baseline 都覆盖同一合同规定的完整题集，才可计算 gain/regression、
McNemar 或 promotion 结论。candidate-only 结果只能作进度记录，不能宣称提升。

## 当前状态

2026-09-03 A100 只有 14-record/1-update live gate，尚无可交付的 700-record RL checkpoint；
因此当前 handoff 只能作为流程入口，不能提前启动“正式 RL 后评测”。KL 稳定性对照仍见
`decision_register.md`。
