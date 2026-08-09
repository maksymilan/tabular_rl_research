# Qwen3-4B/8B raw SQL interface controls

This isolated reproduction evaluates pinned raw `Qwen/Qwen3-4B` and `Qwen/Qwen3-8B` revisions on
BIRD-dev under two repository-owned controls. Neither result is relabeled as a paper-reported
TRUST-SQL number. `qwen3_model_specs.json` pins the revision, config/index hashes, tokenizer/chat
template, architecture, and every weight shard for both model sizes.

## Contracts

Both arms use the same 1,534 BIRD-dev tasks and databases, `bird-set` denotation, greedy `n=1`,
`temperature=0`, `top_p=1`, `max_tokens=2048`, explicit Qwen3 thinking, the same model shards,
four evaluator workers, and a detached localhost vLLM service on the selected GPU host.

- `direct`: one model request with the complete canonical JSON schema and no execution feedback;
  the model emits one SQL query inside `<answer>...</answer>`.
- `iterative`: current diagnostic `iterative-sql-v6` with lazy catalog, `execute_sql(sql)` and
  `submit_sql(sql)`, recent-four history, at most 30 actions, and hidden final scoring.

The direct arm is a true single-turn control. The iterative arm is an SQL-only feedback control;
it is not the paper's four-phase Explore/Propose/Generate/Confirm agent. The paper-reported raw
Qwen3-8B 47.9% is that four-phase unknown-schema agent and is reported separately.
The corresponding paper-reported raw Qwen3-4B four-phase score is 29.3%; its author-protocol
reimplementation is also reported separately.

## Local-vLLM boundary

`src/tool_modules/sql_common/runner.py` accepts `--base-url` only for loopback HTTP endpoints.
External DeepSeek requests still come from ignored `api.md` and remain restricted to the official
`https://api.deepseek.com` endpoint. `--enable-thinking` pins
`chat_template_kwargs={"enable_thinking":true}` and records it in the iterative manifest.

## Smoke on NewGNN

```bash
MODEL_SIZE=4b MODE=direct N=20 RUN_ID=qwen3_4b_direct_sql_smoke20_20260808 \
  bash reproductions/trust_sql/qwen3_8b_sql_controls/launch_detached_newgnn.sh

MODEL_SIZE=4b MODE=iterative N=20 RUN_ID=qwen3_4b_iterative_sql_v6_smoke20_20260808 \
  bash reproductions/trust_sql/qwen3_8b_sql_controls/launch_detached_newgnn.sh
```

Defaults are physical GPU 5 / port 8030 for direct and GPU 6 / port 8031 for iterative. For 8B on
one RTX 3090, set `GPU_MEMORY_UTILIZATION=0.92`; 0.90 leaves slightly insufficient KV cache for
the pinned 32,768-token serving boundary and the supervisor records the setting.

On `table_rl`, point at its existing, byte-identical BIRD input rather than uploading another copy:

```bash
REMOTE=table_rl MODEL_SIZE=4b MODE=direct GPU=0 PORT=8030 \
  REMOTE_SOURCE_INPUT=/home/dengyan/tabular_rl_project/data/eval_inputs/bird_dev_20240627.jsonl \
  N=20 RUN_ID=qwen3_4b_direct_sql_smoke20_table_rl_20260808 \
  bash reproductions/trust_sql/qwen3_8b_sql_controls/launch_detached_newgnn.sh
```

## Full BIRD-dev

Use new run IDs after both smoke runs complete without API errors:

```bash
MODEL_SIZE=4b MODE=direct N=1534 RUN_ID=qwen3_4b_direct_sql_bird_dev1534_20260808 \
  bash reproductions/trust_sql/qwen3_8b_sql_controls/launch_detached_newgnn.sh

MODEL_SIZE=4b MODE=iterative N=1534 RUN_ID=qwen3_4b_iterative_sql_v6_bird_dev1534_20260808 \
  bash reproductions/trust_sql/qwen3_8b_sql_controls/launch_detached_newgnn.sh
```

Each run owns a new directory below:

```text
/home/dengyan/tabular_rl_outputs/evaluations/qwen3_sql_controls/
```

The launcher copies the already resident BIRD source JSON only inside the same server after checking
its fixed SHA-256; it does not upload the local dataset. The supervisor writes `status.json`,
`launch_manifest.json`, `input_gate.json`, `model_gate.json`, `vllm.log`, `evaluation.log`, the
immutable source runtime tar, and `result/`. Completion requires requested unique task coverage and
zero API/connection errors.

For the 2026-08-08 two-server matrix, 4B Direct runs on table_rl GPU 1 while 4B iterative runs on
NewGNN GPU 7. The table_rl queue in
`../qwen3_4b_baselines/queue_qwen3_8b_sql_full_after_table_jobs.sh` starts the 8B Direct and
iterative full controls on table_rl GPUs 1 and 0 only after the current 4B table jobs have completed
and both devices are idle. Scores remain invalid until each supervisor writes a successful
1,534-record terminal `status.json` with `api_error_count=0`.
