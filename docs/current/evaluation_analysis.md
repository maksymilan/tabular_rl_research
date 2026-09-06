# Unified evaluation-result analysis

`src/rl/scenarios/diagnostics/analyze_evaluation_results.py` is the canonical analyzer for deterministic
single-sample evaluation `all.jsonl` artifacts. New experiments must use this CLI instead of adding
an experiment-named summarizer.

It owns the shared implementation of:

- per-arm accuracy, legal termination, mean steps, difficulty strata, action counts, tool counts,
  and adjacent exact-repeat counts;
- paired accuracy and legal gains/regressions/net plus exact two-sided McNemar p-values;
- baseline/candidate rates and paired net deltas as counts, proportions, and percentage points;
- exact-action and tool-sequence changes, first-action changes, argument-only changes, edit-distance
  distributions, common-prefix behavior, and trajectory-length changes;
- marginal tool-count deltas, Jensen-Shannon divergence, and outcome-group policy shifts;
- exact cohort selection and cross-arm evaluation-contract validation.

## Standard invocation

```bash
python src/rl/scenarios/diagnostics/analyze_evaluation_results.py \
  --examples data/eval_inputs/bird_dev_20240627.jsonl \
  --indices path/to/frozen_holdout.indices.json \
  --arm candidate=path/to/version26_candidate/all.jsonl \
  --arm baseline=path/to/version26_baseline/all.jsonl \
  --compare candidate:baseline \
  --expected-count 1534 \
  --protocol-version version26 \
  --protocol-hash 4da19387399bd3a5 \
  --temperature 0 --top-p 1 \
  --denotation-comparison bird-set \
  --output path/to/unified_analysis.json
```

Omit `--indices` only for a complete cohort whose order and membership come directly from
`--examples`. `--arm` uses `LABEL=PATH`; `--compare` uses `CANDIDATE:BASELINE`. Comparisons are
directional. If `--compare` is omitted, every arm after the first is compared with the first arm.
Use `--include-per-example` only when downstream diagnosis needs the complete per-example edit
records; paired gain/regression and changed-index lists are always retained.

The output schema is `unified-evaluation-analysis-v1`:

- `cohort`: exact indices, size, difficulty distribution, and source paths;
- `evaluation_contract`: observed protocol version and hash, temperature, top-p, and denotation
  values;
- `arms`: one standard metric object per named result plus its sibling
  `evaluation_identity.json` when present;
- `comparisons`: one accuracy/legal/behavior object per requested directional pair.

The analyzer refuses duplicate indices, missing rows, multiple samples per question, mismatched
cross-arm evaluation contracts, unexpected cohort sizes, and mismatches against explicitly pinned
protocol version/hash or sampling parameters. It refuses to overwrite an existing output unless
`--overwrite` is supplied.

Promotion gates must additionally use the fail-closed identity options. Pass every sidecar with
`--identity LABEL=PATH`, then use `--require-identities`, `--require-distinct-adapters`, explicit
`--adapter-sha LABEL=SHA256` values, and one `--match-identity-field` for every frozen runtime,
dataset, serving, concurrency, decode, and agent field. Identity paths are not discovered by
walking ancestor directories: a launcher that stores the sidecar above `result/all.jsonl` must
name it explicitly. Any present identity is always checked against the row-level protocol,
temperature, top-p, and denotation contract, even in historical non-strict analyses.

New full-dev runs atomically bind a previously empty result directory to the evaluated adapter
path and SHA-256, adapter-config SHA-256, base model, served-model identity, and executable
protocol version/hash in `evaluation_identity.json`. A partial directory without that identity, or
an attempted resume with another adapter, is rejected rather than assigned inferred provenance.
Historical comparator results may lack this newer sidecar, but the candidate baseline readiness
check requires it and verifies its adapter SHA against the final LoRA artifact analyzed after
training.

## Compatibility boundary

The following historical entrypoints remain only because frozen launchers and reports refer to
their old CLI or JSON schema:

- `archive/code/legacy_rl/distillation/summarize_full_dev.py`;
- `src/rl/scenarios/diagnostics/summarize_routed_coupled_behavior_smoke.py`;
- `src/rl/scenarios/diagnostics/measure_policy_behavior_shift.py`.

They contain no independent metric implementation; each delegates to the unified analyzer and
only translates arguments/output into its historical schema. Do not call them from new experiments.

Adapter-tensor analysis (`compare_lora_updates.py`) and frozen-transition log-probability analysis
(`summarize_frozen_policy_scores.py`) consume different artifact types and remain separate. A new
analyzer is justified only for a genuinely different input schema, and its ownership boundary must
be added here before use.
