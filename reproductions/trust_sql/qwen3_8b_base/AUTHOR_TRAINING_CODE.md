# Author Qwen3 training code map

The released repository contains intended full-parameter Qwen3 RL machinery. It does **not** train
the paper-labeled Qwen3-8B line in Table 2; that line is an inference baseline. The public artifacts
are also insufficient to train the reported TRUST-SQL-8B from the public Qwen checkpoint because
the SFT data, SFT job, SFT-initialized checkpoint, and paper run manifest are absent.

Pinned upstream: `JaneEyre0530/TrustSQL@89df0661ad6b8e29ed8e61f7c950fbc2c1678b08`.

## Entry points

- `submit_training.sh`: paper-oriented RL launcher template. It selects Qwen3-8B, asynchronous RL,
  eight rollouts per prompt (the GRPO group size, not evaluation Majority K), schema weight 0.25,
  and the NL2SQL rollout/reward.
- `examples/nl2sql/run_qwen3_async_oversampling.sh`: parses the model family/size, loads the matching
  Megatron architecture, converts the Hugging Face checkpoint to `torch_dist`, constructs the
  Slime/SGLang/Ray arguments, and launches asynchronous GRPO.
- `examples/nl2sql/run_qwen3_unified.sh`: candidate synchronous counterpart, useful for tracing the
  same argument construction without asynchronous oversampling. Its existence does not supply a
  released paper 4B invocation.
- `examples/nl2sql/generate_sql_token.py`: four-phase model-to-database rollout and token-level
  reward placement.
- `examples/nl2sql/sql_reward_with_schema.py`: execution, format, and schema rewards.
- `scripts/models/qwen3-8B.sh`: exact Megatron architecture flags for the dense 8B model.
- `slime/backends/megatron_utils/config_mapping/predefined_config_mappers.py`: maps Hugging Face
  `model_type=qwen3` to Megatron and enables QK layer normalization.
- `tools/convert_hf_to_torch_dist.py`: the launcher's Hugging Face → Megatron `torch_dist`
  initialization path, implemented through `mbridge.AutoBridge`.
- `slime/backends/megatron_utils/megatron_to_hf/qwen2.py`: the reverse Megatron → Hugging Face
  export mapping shared by dense Qwen2/Qwen3; it contains Q/K-normalization branches.

## Qwen3-8B is not a renamed Qwen2.5-7B config

| Field | Qwen3-8B | Qwen2.5-7B |
|---|---:|---:|
| layers | 36 | 28 |
| hidden size | 4096 | 3584 |
| FFN size | 12288 | 18944 |
| attention heads | 32 | 28 |
| KV groups | 8 | 4 |
| explicit head size | `kv-channels=128` | inferred |
| Q/K normalization | required (`--qk-layernorm`) | absent |
| QKV bias | absent | required (`--add-qkv-bias`) |
| vocabulary size | 151936 | 152064 |

This is a code-level comparison of the released `qwen3-8B.sh` and `qwen2.5-7B.sh`, not a paper
ablation. Qwen3's per-head Q/K RMSNorm is a material architecture difference. Reusing Qwen2.5
flags would construct a Megatron architecture with missing Q/K normalization and extra QKV bias,
which does not match the Qwen3 checkpoint and may fail conversion or parameter validation. The
upstream mapper registers separate `qwen2` and `qwen3` model types even though the dense families
share most export name-mapping code.

The released generation code calls `tokenizer.apply_chat_template(...)` without an
`enable_thinking` argument, while the task protocol requires an explicit `<think>...</think>`
block. Because the paper does not publish its exact checkpoint/template revision, behavior cannot
be inferred from a Qwen2.5 launcher or asserted as an author-fixed Qwen3 thinking configuration;
the selected public revision's template must be pinned and audited separately.

## Why paper-scale training is not launched here

The upstream launcher is a template, not a self-contained paper job:

- model/data/save paths are placeholders;
- `utils/export_env_slime.sh` and `utils/low_gpu_utilization.py` are referenced but absent;
- the SFT trajectories and SFT-initialized Qwen3-8B checkpoint are not released;
- the launcher says two episodes and LR `1e-6`, whereas paper Table 8 says three RL epochs and
  Qwen3-8B LR `8e-7`;
- paper training uses 32 A100 GPUs; `table_rl` has two RTX 3090 GPUs.

Consequently this reimplementation evaluates the selected public Qwen3-8B checkpoint as the
paper-labeled inference baseline and preserves the author RL path as a code reference. Starting a
two-3090 QLoRA/shortened-K job would be a separate low-resource experiment, not the paper's
baseline or TRUST-SQL-8B result.
