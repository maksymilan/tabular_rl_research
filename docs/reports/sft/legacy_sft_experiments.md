# SFT Experiments

## Qwen2.5-7B v1 QLoRA

Started: 2026-06-13 00:01 Asia/Shanghai

### Long-context smoke

- Source: `spider_v1_train.jsonl`, longest record at source index 3230.
- Longest raw content length: 9,113 Qwen2.5-7B tokenizer tokens.
- `cutoff_len=10240`, LoRA rank 32: OOM on the first backward pass.
- `cutoff_len=8192`, rank 32, standard AdamW: first update completed, second
  backward OOM after optimizer-state allocation.
- `cutoff_len=8192`, rank 32, `paged_adamw_8bit`: two updates passed, but
  GPU memory peaked at 24,250 / 24,576 MiB.
- Final smoke configuration: `cutoff_len=8192`, rank 16, alpha 32,
  `paged_adamw_8bit`. Two updates passed; loss went from 0.6580 to 0.5054
  and peak GPU memory was 23,646 / 24,576 MiB.

The smoke dataset deliberately repeats the longest record twice so the second
step includes optimizer state. A one-step smoke is insufficient for this setup.

### Full run

- Dataset: `spider_tools_v1_8k`.
- Records: 6,763 total; LLaMA-Factory split 6,627 train / 136 validation.
- Eight records over 7,900 raw content tokens were excluded. The longest kept
  record has 7,701 raw content tokens.
- Effective batch size: 16 through gradient accumulation.
- Epochs: 2; optimizer steps: 830.
- Remote PID file:
  `/home/dengyan/tabular_rl_project/logs/qwen2.5_7b_spider_v1_qlora.pid`
- Training log:
  `/home/dengyan/tabular_rl_project/logs/qwen2.5_7b_spider_v1_qlora.log`
- GPU log:
  `/home/dengyan/tabular_rl_project/logs/qwen2.5_7b_spider_v1_qlora_gpu.csv`
- Output:
  `/home/dengyan/tabular_rl_project/checkpoints/qwen2.5-7b-spider-v1-qlora`

Completed: 2026-06-13 13:03 Asia/Shanghai

- Optimizer steps: 830 / 830; epochs: 2.
- Runtime: 13:01:20.
- Train loss: 0.2495288724.
- Validation loss: 0.2270773202.

The remote SFT environment requires `bitsandbytes==0.46.1` because its
`transformers==5.6.0` rejects older versions for 4-bit loading.

### Zero-shot tool evaluation

The final LoRA adapter was served with vLLM and evaluated through the same
closed-loop tool harness used by the baseline:

```bash
.venv/bin/python src/eval/rollout.py \
  --base-url http://127.0.0.1:18000/v1 \
  --model qwen2.5-7b-spider-v1 \
  --n 1034 --few-shot 0 --workers 8 --max-steps 20 \
  --result-dir data/results/qwen2.5_7b_sft_v1/tool_zero_shot_dev1034
```

- Full Spider dev: 707 / 1,034 = 68.38%.
- Legal final answers: 985 / 1,034 = 95.26%.
- Average trajectory length: 3.70 steps.
- Tool execution errors observed across trajectories: 247.
- Failures: 276 wrong answers, 42 execution errors, 8 protocol errors,
  and 1 max-steps case.
- Verified-v1 subset: 705 / 998 = 70.64%.
- Dev examples outside verified-v1 coverage: 2 / 36 = 5.56%.
- Covered `add_to_memory` subset: 9 / 29 = 31.03%.
- Covered non-memory subset: 696 / 969 = 71.83%.

The result directory retains `all.jsonl`, `success.jsonl`, `failure.jsonl`,
and one formatted JSON file per example in `success_cases/` and
`failure_cases/`. Each record includes the initial model input, every model
output, parsed action, tool output/error, final messages, and score.

Interpretation: SFT fixed most of the base model's tool-protocol failure
(95.26% legal answers versus 78 legal answers for the two-shot base model)
and reaches the direct-SQL baseline's range. The low memory-subset result
means this v1 checkpoint should not be treated as evidence that scalar
memory/provenance is solved; regenerate corrected trajectories before the RL
stage.
