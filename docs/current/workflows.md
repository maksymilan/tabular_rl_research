# 当前实验工作流

项目只有四类操作。工作流层负责选择操作及其参数，训练器、rollout、Harness、reward 和评测器负责实现细节。新实验不应复制 launcher；先用 `src/rl/workflows/cli.py` 生成计划，再由 canonical entrypoint 做 preflight。

```text
rl_verify  --questions N --prompts-per-update N   小规模机制验证（K 固定为 8）
rl_full    --questions 500 --prompts-per-update 30 全量 RL 主实验
 evaluate  --dataset-path data/eval/bird_dev.jsonl --dataset bird --split dev --gpu 0 --gpu 1  全量数据集评测
sft_frozen                                          已冻结的 SFT 基线，仅供复现
```

RL 的规模不是一套新代码：`rl_verify` 和 `rl_full` 共享同一 Atomic version26 流程，区别只在题目数、每次 update 的题目数和 update 数。机制由 `--mechanism` 传入；当前正式值为 `saam-asymmetric-error`。机制组件必须实现统一 trainer 接口，流程层不应出现机制分支。

资源优先使用 `a100`；若不可用，3090 专用 launcher 才可在同一台服务器上选择两张独立卡分别
运行 replicated trainer 和 online vLLM，并将降级参数写入 manifest。当前 A100 launcher 不
接受 3090；两张卡不足时直接失败。

评测由数据集、split、题目数和 GPU 列表参数化。评测 GPU 的具体上限由服务器 launcher 检查；A100 任务只能使用 0–3 号卡。

仅生成计划（不会启动训练或占用 GPU）：

```bash
PYTHONPATH=src python -m rl.workflows.cli rl_verify \
  --questions 30 --prompts-per-update 14 --json
PYTHONPATH=src python -m rl.workflows.cli rl_full \
  --questions 500 --prompts-per-update 30 --json
PYTHONPATH=src python -m rl.workflows.cli evaluate \
  --dataset bird --dataset-path data/eval/bird_dev.jsonl --split dev \
  --checkpoint /path/to/checkpoint-6380 --gpu 0 --gpu 1 --json
```

SFT 基线已冻结，不通过工作流层重新生成或修改；需要复现实验时直接调用历史入口并保留其 manifest。
