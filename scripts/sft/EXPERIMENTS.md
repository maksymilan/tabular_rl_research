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

The remote SFT environment requires `bitsandbytes==0.46.1` because its
`transformers==5.6.0` rejects older versions for 4-bit loading.

Check status:

```bash
ssh NewGNN '
pid=$(cat ~/tabular_rl_project/logs/qwen2.5_7b_spider_v1_qlora.pid)
ps -p "$pid" -o pid=,etime=,stat=,cmd=
tail -n 30 ~/tabular_rl_project/logs/qwen2.5_7b_spider_v1_qlora.log
'
```
