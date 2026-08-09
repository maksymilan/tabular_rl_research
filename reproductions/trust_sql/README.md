# TRUST-SQL reproduction on `table_rl`

This directory pins and audits the official implementation of
**TRUST-SQL: Tool-Integrated Multi-Turn Reinforcement Learning for Text-to-SQL over Unknown
Schemas**.  It is an isolated external-paper reproduction, like the existing OmniSQL and SQL-R1
reproductions.  It does not modify or reinterpret the repository's active
`native-tool-bundle/version51` protocol.

## Current result

As of 2026-08-06, the reproducible part of the paper is complete on `table_rl`:

- official source pinned at commit `89df0661ad6b8e29ed8e61f7c950fbc2c1678b08`;
- source, released RL JSONL, prompt, rollout, and reward files match recorded SHA-256 hashes;
- all four action formats pass the released validator;
- the released execution reward returns exactly `1.0 / 0.2 / 0.0` for correct / executable-wrong /
  invalid SQL;
- exact schema reward and the released Dual-Track loss markers pass;
- the 11,642 released RL records resolve to all 69 BIRD and 146 Spider databases on `table_rl`;
- no released gold SQL string appears verbatim in the model-visible prompt.

The separate [`qwen3_8b_base/`](qwen3_8b_base/) directory implements the public-checkpoint
reimplementation of the paper-labeled Qwen3-8B unknown-schema greedy baseline. It uses the
author's released async evaluator and four-phase prompt, while recording the unpublished
checkpoint, greedy horizon, scorer, and runtime choices as assumptions. This does not require or
claim to reconstruct the trained TRUST-SQL-8B checkpoint.

This is **not an end-to-end numerical reproduction of the headline model scores**.  It cannot be
one with the currently public artifacts and this host:

- the paper used 8-32 A100 GPUs for RL and 16 A100 GPUs for SFT; `table_rl` has two RTX 3090 24 GiB
  GPUs with no NVLink;
- the authors did not release the approximately 9.2k SFT trajectories, the SFT warm-up checkpoints,
  or the final TRUST-SQL-4B/8B weights;
- the pinned launch script references two missing private runtime helpers and does not ship a
  locked Slime/SGLang/Megatron image;
- the public RL JSONL and the available public benchmark databases share all 215 database IDs, but
  full read-only execution finds 353 stable non-timeout SQL errors (350 BIRD, 3 Spider) across six
  databases.  Schema labels reference absent tables/columns in 382 records.  The largest mismatch
  is `retail_world` (330 records), indicating a database-release/content mismatch rather than a
  path-resolution failure;
- the recorded 16-worker audit also interrupted 124 additional BIRD queries at the explicit
  10-second limit.  This timeout subset varies with host load (126 and 129 in two preceding runs),
  so it is reported separately and is not classified as invalid SQL.

The path-rewritten JSONL is therefore an audit artifact, **not a training-ready dataset**.  Do not
silently drop the 353 SQL errors or the separately reported slow-query subset and call the result
the paper configuration.

## Pinned protocol and optimization contract

Each assistant turn emits exactly one textual action:

```text
<think>...</think>
<action>explore_schema|propose_schema|generate_sql|confirm_answer</action>
<tool_call>...</tool_call> | <schema>...</schema> | <answer>...</answer>
```

Only `explore_schema` and `generate_sql` invoke `execute_sql_query`.  `propose_schema` is the
verified-schema checkpoint, and `confirm_answer` submits final SQL.

The paper configuration samples eight trajectories per question.  The schema track ends at the
last proposal and uses sparse exact schema reward, coupled to `R_exec=1`.  The full track uses
`R_exec + R_fmt`.  The two group-relative advantages and token masks are computed separately, then
combined as:

```text
L = L_full + 0.25 * L_schema
```

Paper Table 8, rather than current launcher defaults, is the reproduction source of truth:

| Model | Stage | Hardware | LR | Batch | Epochs | K | Turns |
|---|---|---|---:|---:|---:|---:|---:|
| Qwen3-4B | SFT | 16x A100 | 1e-5 | 256 | 2 | - | - |
| Qwen3-4B | RL | 8x A100 | 1e-6 | 32 | 3 | 8 | 10 |
| Qwen3-8B | SFT | 16x A100 | 1.5e-6 | 256 | 2 | - | - |
| Qwen3-8B | RL | 32x A100 | 8e-7 | 32 | 3 | 8 | 10 |

The released `submit_training.sh` instead sets two episodes and a shared `1e-6` RL learning rate;
those defaults must not be presented as the paper's 8B configuration.

## Remote layout

```text
/home/dengyan/tabular_rl_outputs/reproductions/trust_sql/
  source/       # wrappers copied from this directory
  third_party/TrustSQL/
  data/
  logs/
  manifests/
  results/
```

The full audit manifests are:

```text
manifests/upstream_audit.json
manifests/host_preflight.json
manifests/released_rl_data_full_audit.json
manifests/reproduction_status.json
```

## Re-run

Run source, host, path, and lightweight data checks:

```bash
TRUSTSQL_PYTHON=/home/dengyan/miniconda3/envs/sft/bin/python \
  bash source/run_preflight_table_rl.sh
```

Run all 11,642 hidden gold SQL queries read-only (about two to three minutes on this host):

```bash
/home/dengyan/miniconda3/envs/sft/bin/python source/audit_released_rl_data.py \
  --input third_party/TrustSQL/data_for_sql/filtered_questions.jsonl \
  --bird-database-root /home/dengyan/tabular_rl_project/data/bird/train/train_databases \
  --spider-database-root /home/dengyan/tabular_rl_project/data/spider_data/database \
  --manifest manifests/released_rl_data_full_audit.json \
  --execute-gold --workers 16 --sql-timeout-seconds 10
```

The manifest separates non-timeout SQL errors from `OperationalError: interrupted` queries.  The
latter are progress-handler timeouts, and their exact membership is load-sensitive under parallel
execution.

`assert_paper_scale_ready.sh` deliberately refuses a full run on `table_rl`.  It requires at least
eight A100 GPUs, a supplied paper SFT checkpoint, and the Slime runtime imports before permitting a
paper-scale launcher.

## What is needed for headline-score reproduction

1. Obtain the exact SFT trajectories or warm-up checkpoint from the authors.
2. Obtain hashes for the authors' BIRD/Spider SQLite release, especially the 16 affected databases.
3. Obtain a locked container/runtime for the pinned Slime, SGLang, Megatron, Apex, and Transformer
   Engine stack.
4. Run Qwen3-4B on at least 8 A100 GPUs, or Qwen3-8B on 32 A100 GPUs, using Table 8 settings.
5. Evaluate BIRD with the named `bird-set` denotation metric and report provider/runtime failures
   separately.  The released training reward's 30-row `set(...)` comparator is not a replacement
   for benchmark evaluation.

Any QLoRA, smaller-model, shorter-context, reduced-K, or two-3090 adaptation is useful only as a
systems smoke and must be labeled a low-resource reimplementation rather than the paper result.
