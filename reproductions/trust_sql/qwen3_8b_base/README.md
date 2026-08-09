# TRUST-SQL paper-labeled Qwen3-8B baseline reimplementation

This directory implements an auditable attempt at the **paper-labeled Qwen3-8B unknown-schema
greedy baseline** and targets the 47.9% result in TRUST-SQL Table 2. It is deliberately separate
from both this repository's active tool protocols and the paper's trained TRUST-SQL-8B model.
The pinned model transfer, end-to-end validation, and full 1,534-question run are complete. The
local reimplementation scores **709/1,534 = 46.22%**, which is **1.68 percentage points below**
the paper's 47.9%. The baseline pipeline is therefore reproduced, but the unpublished numerical
result is not matched exactly.

Paper comparison points:

- Qwen3-8B, unknown schema, greedy: **47.9%** BIRD-Dev EX;
- Qwen3-8B, full schema prefill, greedy: **49.9%** BIRD-Dev EX;
- TRUST-SQL-8B after SFT+RL, unknown schema, greedy: 65.8% (not this baseline).

The paper and released repository do not identify the exact Qwen3 checkpoint revision.  This run
uses the public `Qwen/Qwen3-8B` revision
`b968826d9c46dd6066d109eabc6255188de91218` and records that as a reproduction assumption rather
than presenting it as an author-published fact.

## What is reused and what is patched

The generation core uses the author's pinned `trustsql_eval/main_batch_async.py` and released
four-phase prompt as the closest public implementation. The repository does not provide a Table 2
launcher or manifest tying that evaluator and prompt to the reported 47.9%. `prepare_bird_dev.py`
therefore creates the missing evaluation JSONL in the same two-message shape as the released
training records, with external knowledge inline and gold SQL hidden in reward-only fields. This
is an explicit reimplementation choice, not evidence of the unpublished Table 2 input format.

`official_eval_vllm019.patch` is a narrow runtime audit patch:

- adapts removed vLLM V0/engine arguments to table_rl's vLLM 0.19.1;
- exposes and records seed/penalties/GPU memory settings;
- makes shuffle deterministic, resumes incomplete/error rollouts, and replaces a prior record with
  the same rollout index instead of creating ambiguous duplicates;
- writes result JSON atomically;
- propagates fatal runner errors to the shell instead of returning exit status zero;
- explicitly shuts down modern vLLM EngineCore/workers so GPUs and log pipes are released;
- enforces the stated SQLite deadline with a progress handler;
- returns at most 30 observation rows, resolving the paper/prompt versus released-code conflict in
  favor of the paper and prompt (the released evaluator returned up to 100).

It does not rewrite generated text or terminal SQL and performs no verifier selection. The V0→V1
runtime change, explicit seed, and 30-row model-visible observation cap can affect generation, so
results from this directory must remain labeled as a reimplementation.

## Remote layout

```text
/home/dengyan/tabular_rl_outputs/reproductions/trust_sql/
  third_party/TrustSQL/                    # clean pinned source
  runtime/TrustSQL-qwen3-8b-base/          # patched evaluation clone
  data/qwen3_8b_base/
  results/qwen3_8b_base_unknown_greedy_*/
  logs/qwen3_8b_base/
```

Model path:

```text
/home/dengyan/models/Qwen3-8B-TrustSQL-baseline
```

## Current validation status

- The pinned public checkpoint was downloaded through `https://hf-mirror.com`; every regular file
  was checked by size plus Git blob ID and every LFS object by size plus SHA256.
- The four-question smoke run `qwen3_8b_base_unknown_greedy_smoke4_seed20260806` produced four
  legal terminal answers. All four final SQL strings parsed and both prediction and gold SQL
  executed without timeout; one of four denotations matched. This tiny contiguous prefix is a
  transport/protocol smoke test, not an accuracy estimate.
- A one-question shutdown regression run with seed `20260807` exited normally after explicitly
  stopping vLLM EngineCore/workers. No evaluation process or GPU allocation remained afterward.
- The full 1,534-question run `qwen3_8b_base_unknown_greedy_dev1534_seed20260806` completed in
  149.5 minutes and scored **709/1,534 = 46.2190%**. All 1,534 expected output slots were present
  with no duplicate or unexpected records. The scorer reported six parse failures, 74 prediction
  execution errors, two prediction timeouts, and two gold timeouts. At the raw runner level, 1,532
  records had `terminated=true`, two reached the round limit without termination, and one record
  contained a generation-retry-exhaustion error.

The full score is
`/home/dengyan/tabular_rl_outputs/reproductions/trust_sql/results/qwen3_8b_base_unknown_greedy_dev1534_seed20260806/score.json`
with SHA256 `745a17c573958df38424b68625d9d332380606aa517ac1d71053d69ad39b2df7`.
Smoke artifacts are retained under their matching directories in the same `results/` root.

## Run

Prepare the immutable patched runtime once:

```bash
bash source/qwen3_8b_base/prepare_runtime.sh
```

Download the pinned model on `table_rl` through the Hugging Face mirror:

```bash
bash source/qwen3_8b_base/download_model.sh
```

The default three-worker `mirror-curl` backend resumes `.partial` files and verifies every file against the
pinned revision's published size plus LFS SHA256 or Git blob ID before making it visible at the
model path.  `DOWNLOAD_BACKEND=hf-cli` remains available, but the mirror's CLI metadata route has
shown intermittent SSL EOF failures on `table_rl`; direct `curl` downloads have been stable. Set
`MIRROR_JOBS=1..5` to change download concurrency. A host lock and signal trap prevent two runs
from writing the same partial file concurrently.

Run four-question smoke evaluation:

```bash
EVAL_SCOPE=smoke SMOKE_LIMIT=4 \
  bash source/qwen3_8b_base/run_greedy_table_rl.sh
```

Run all 1,534 questions in the locally pinned BIRD-Dev snapshot:

```bash
EVAL_SCOPE=full BATCH_SIZE=16 \
  bash source/qwen3_8b_base/run_greedy_table_rl.sh
```

The scorer extracts the final `<answer>`, executes it read-only, and reports `bird-set` denotation
accuracy: full-result set equality that ignores row order and duplicate multiplicity. The paper
describes matching database results but does not release the Table 2 scorer, so `bird-set` is a
local benchmark-aligned metric choice. Generation artifacts are never verifier-selected or
silently retried semantically. The scorer's 10-second read-only SQL deadline is also a local
project choice and reports prediction/gold timeouts separately.

The launcher fixes greedy `max_rounds=15`; the paper explicitly gives 15 turns for sampled
majority/Pass@K inference, not for greedy evaluation. It also fixes `top_p=1`, which is inert at
temperature zero but was not reported by the paper. Both remain recorded assumptions.

Schema-prefill and majority voting are not labeled reproduced here: the authors did not release a
formal prefill serialization or majority K/tie-break implementation.  Those can be added only as
explicitly named reimplementations.

See `AUTHOR_TRAINING_CODE.md` for the upstream training entry points and the concrete Qwen3 versus
Qwen2.5 architecture/conversion differences.
