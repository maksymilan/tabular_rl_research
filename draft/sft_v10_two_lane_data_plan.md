# SFT v10 Two-Lane Data Generation Plan

Status: execution plan for the next data iteration. This document fixes the current decision so
Claude and Codex do not invent separate trajectory formats.

## Goal

The first Qwen3.5-9B SFT stage used about 1k high-quality observation-enriched examples. The next
stage should expand supervision without changing the model-visible tool protocol:

- keep the current `src/sft/protocol.py` contract (`PROTOCOL_VERSION = v2b`);
- do not add a reflection tool;
- do not reintroduce memory or `memory_id`;
- train the model to react to observations through ordinary `<think>` text and existing tools.

## Data Mixture

Use two data lanes.

### Lane A: clean canonical data

Purpose: enlarge the correct main-path distribution and stabilize tool formatting.

Source:

1. sample training trajectories not already used in the v9 1k set;
2. compile gold SQL to verified tool trajectories through the existing harness pipeline;
3. add observation steps and first-person reasoning with `src/sft/enrich_traj.py`;
4. replay through the harness and export only `quality_status == "ready"`.

External LLM role: rewrite reasoning and place necessary observations. It should not invent a new
answer, change the verified relational backbone, or add recovery branches in this lane.

Recommended command shape:

```bash
.venv/bin/python src/sft/select_subset.py \
  --n 1600 --per-db-cap 16 --seed 10 \
  --out-prefix subset_v10_clean_1600 \
  --exclude-ids data/trajectories/<v9_used_ids>.json

.venv/bin/python src/sft/enrich_traj.py \
  --ids data/trajectories/subset_v10_clean_1600.ids.json \
  --which subset \
  --subset-file data/trajectories/subset_v10_clean_1600.jsonl \
  --mode staged_perception \
  --max-attempts 10 \
  --out data/trajectories/spider_v10_clean_enriched.jsonl
```

Generated initial candidate files:

- `data/trajectories/subset_v10_clean_1600.jsonl` was an old v2a skeleton artifact and must not be
  used directly for current SFT. Regenerate the skeleton from `spider_train_v3.jsonl`, or use the
  current replacement below.
- `data/trajectories/subset_v10_clean_1600_v3_skeleton.jsonl`
- `data/trajectories/subset_v10_clean_1600.ids.json`
- `data/eval_inputs/subset_v10_clean_1600_train_examples.json`

These were sampled with `--exclude-ids data/trajectories/subset1000_v9.ids.json`; overlap with v9
training ids is 0.

The rollout input file is built from Spider train, not dev:

```bash
.venv/bin/python src/sft/build_rollout_examples.py \
  --ids data/trajectories/subset_v10_clean_1600.ids.json \
  --output data/eval_inputs/subset_v10_clean_1600_train_examples.json
```

Important: `select_subset.py` only selects current-protocol skeletons. The SFT-clean file is the
output of `enrich_traj.py` after quality filtering, e.g.
`data/trajectories/subset_v10_clean_1600_v3_ready.jsonl`. Do not train on the skeleton JSONL.

### Lane B: recovery data

Purpose: teach the model to notice that an observation or tool result does not support the current
path and then continue with a better existing-tool path.

Strict split rule: recovery data must come from **training-set rollouts only**. Spider dev / held-out
evaluation artifacts may be inspected for diagnosis, but they must never be converted into SFT
recovery candidates.

Source:

1. run the current SFT model on a new training subset using `--examples-json`;
2. select failed samples that are useful for recovery:
   - legal but wrong;
   - empty intermediate result;
   - execution error that can be repaired by better schema/value observation;
   - pass@k mixed cases where one sample succeeds and another fails;
3. give the failed sample, observations, question, catalog, and verified backbone to the external LLM;
4. let the LLM produce a corrected recovery trajectory;
5. replay and validate with the harness;
6. feed validation errors back to the LLM for up to 10 attempts;
7. discard examples that still fail after 10 attempts.

External LLM role: generate first-person recovery reasoning and corrected existing-tool calls. It is
not authoritative; the harness decides whether the repaired trajectory is usable.

Candidate selection command:

```bash
.venv/bin/python src/eval/rollout_passk.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model <served_sft_model> \
  --examples-json data/eval_inputs/subset_v10_clean_1600_train_examples.json \
  --n-samples 2 --pass-k 2 \
  --workers 4 --sample-workers 2 \
  --max-steps 20 --max-tokens 768 --api-retries 3 \
  --result-dir data/results/<train_subset_rollout>

.venv/bin/python src/sft/select_recovery_candidates.py \
  --input data/results/<train_subset_rollout>/all.jsonl \
  --output data/trajectories/recovery_candidates_v10.jsonl \
  --limit 400 --per-db-cap 12
```

`src/eval/rollout_passk.py` defaults to Spider dev only when `--examples-json` is absent. For recovery
data, always pass the explicit train examples file. `src/sft/select_recovery_candidates.py` refuses
dev/eval-looking inputs unless `--allow-dev-analysis` is set for temporary auditing.

The repair generator should default to `max_repair_attempts = 10`.

## Final Mixture

Initial target for the next SFT run:

- 1600-1700 clean canonical examples;
- 300-400 recovery examples.

If recovery yield is low, do not fill with unverified repairs. Increase clean canonical data instead.

## Validation Gates

Every exported trajectory must pass:

- current tool argument schema;
- full harness replay;
- final answer equals the gold SQL denotation;
- no `add_to_memory`, `memory_id`, `mem_`, or `supporting_memory_ids` residue;
- every assistant turn has non-empty first-person `<think>`;
- no schema/literal leak before the corresponding observation;
- final answer is grounded in a visible scalar result or a read evidence table.

## What Not To Do

- Do not add an `invalidate` concept.
- Do not make recovery metadata a model-visible field.
- Do not train mostly on recovery traces; that may teach the model to make a mistake first.
- Do not repair API errors or context-overflow cases as reasoning failures.
