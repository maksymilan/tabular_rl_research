# Process Reward v2: harness-inferred dependencies

## 1. Design correction

V1 required a final `answer_from_context.evidence` handle before it could seed `BackSlice`. That
made reward depend on a model-authored citation and assigned no credit to most perception actions.
V2 moves this responsibility into the harness:

1. The final dependency table is the first table-bearing legal action found when scanning backward
   from the answer. A last `read_subtable` uses its input handle; a table-producing action uses its
   output handle. Model `evidence` does not control reward.
2. `build_references` still creates exact data/value edges from action parameters.
3. `build_grounding_references` creates harness-authored perception edges from prior legal outputs:
   schema edges for table/columns returned by `describe_table`, domain edges for literals returned
   by `inspect_column`, and row edges for values returned by `read_subtable`.
4. `BackSlice` traverses `data`, `value`, and `grounding` edges.
5. A perception step in the slice receives `B=1`; a used schema/domain/row observation receives
   `E=1`. Unused or truncated observations do not receive automatic credit.
6. Observation errors use the existing tool-error penalty. Empty observations and legal actions
   after observation errors use the same harness-verified `F` rule as relational tools.

No rule parses model think/reason text. Future error events persist `attempted_tool` and
`attempted_arguments`, allowing observation-error recovery to be audited by tool class.

## 2. Bob Corker regression case

`bird_train_01915` previously had `evidence=null`, an empty slice, and terminal fallback `+1.0`.
The harness now reconstructs:

```text
describe people/social schema
  -> filter Bob Corker
  -> read bioguide C001071
  -> filter social-media with C001071
  -> read instagram senbobcorker
  -> answer
```

With equal pilot weights, the reward is:

| step | B | E | S | reward |
|---|---:|---:|---:|---:|
| describe_table | 1 | 1 | 0 | 0.1856 |
| first condition_filter | 1 | 0 | 0.8899 | 0.1754 |
| first read_subtable | 1 | 1 | 0 | 0.1856 |
| second condition_filter | 1 | 1 | 0.8877 | 0.2679 |
| final read_subtable | 1 | 1 | 0 | 0.1856 |
| answer_from_context | 0 | 0 | 0 | 0 |

The rewards sum to 1.0. The second filter has both table-evidence production and search reduction;
the reads are credited because their returned values are consumed later.

## 3. Scale-500 success-183 audit

Artifacts:

- `data/rl/bird_scale500_success183_process_reward_v2/scored_trajectories.jsonl`
- `data/rl/bird_scale500_success183_process_reward_v2/summary.json`
- config: `configs/process_reward_v2.json`

| metric | v1 | v2 |
|---|---:|---:|
| replay correct | 183/183 | 183/183 |
| structured grounding | 79/183 (43.2%) | 182/183 (99.45%) |
| terminal fallback | 69 | 1 |
| ungrounded non-fallback | 35 | 0 |
| positive steps | 350 | 881 |
| zero steps | 822 | 291 |
| max positive share p50 | 1.000 | 0.271 |
| max positive share p90 | 1.000 | 0.422 |

Perception allocation:

| tool | total steps | in BackSlice | E hits | positive steps |
|---|---:|---:|---:|---:|
| describe_table | 200 | 196 | 196 | 196 |
| inspect_column | 88 | 33 | 33 | 39 |
| read_subtable | 213 | 184 | 184 | 186 |

The extra positive inspect/read steps are feedback-recovery events. The distinction between used and
unused observation is visible in `bird_train_00589`: its truncated inspection did not show the
eventual literal and receives zero, while the final read and error recovery receive credit.

Reward invariants still pass: all 183 replay correctly, every episode satisfies `sum(r)=C-P`, every
correct total is positive, and reward min/mean/max are 0.30 / 0.8828 / 1.00.

## 4. Remaining risks

Structural coverage is no longer the blocker, but inferred-edge precision still needs audit:

1. The nearest prior table rule can over-attribute an unrelated last exploratory branch.
2. Row-value matching can over-connect common literals; v2 limits this to the latest matching read,
   but it is still an availability inference rather than direct causal proof.
3. `inspect_column` gets evidence credit only when a returned domain value is later used. Other
   legitimate uses, such as choosing an operator from null/domain statistics, are not yet covered.
4. `E` is binary per step and does not measure evidence amount or uniqueness.
5. Existing 183 historical error events do not contain structured attempted tools. New episodes do,
   but observation-specific error allocation still needs empirical failure-rollout tests.
6. Positive weights and penalty lambdas remain untuned.

The report therefore records:

```text
structural_grounding_gate_passed = true
grounding_precision_audit_approved = false
process_reward_ready = false
```

The next gate is a stratified manual audit of inferred edges, followed by failure-trajectory replay
and weight ablations. Only then should `--grounding-audit-approved` be used.

## 5. Reproduction

```bash
.venv/bin/python src/rl/build_process_reward_report.py \
  --input data/trajectories/bird_ds_flash_v4_scale500_rolling4_full_success.jsonl \
  --output-dir data/rl/bird_scale500_success183_process_reward_v2 \
  --config-json src/rl/configs/process_reward_v2.json \
  --quiet
```
