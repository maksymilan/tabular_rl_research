# External grounding audit (2026-07-16)

## Decision

`process_reward_ready` remains **false**. The harness-owned grounding implementation has excellent
structural coverage, but external review found reproducible false edges and incomplete final
dependency chains. External labels are audit evidence only; they are not reward targets or human
gold labels.

## Scope and method

- Inputs: all 322 replay-correct BIRD-train teacher successes from the two disjoint Scale-500 runs
  (183 + 139).
- Evidence: question, legal tool arguments, harness outputs, final-table rows, and harness-authored
  data/value/grounding provenance. Model think text is excluded.
- First review: `deepseek-v4-flash` over all 322 packages.
- Independent recheck: `deepseek-v4-pro` over every Flash fail/ambiguous case, every
  `common_literal` risk case, and a deterministic 30-case Flash-pass control (seed 20260716).
- Strict response contract: complete JSON with every requested edge id exactly once. No JSON/tag
  repair or semantic parameter repair. Reviewer transport failures remain unresolved.

The audit exporter is `review_grounding_edges_external.py`; deterministic recheck selection and
aggregation are `select_grounding_recheck.py` and `summarize_grounding_external_audit.py`.

## Audit protocol corrections

The first audit package exposed only grounding edges. Reviewers therefore reported real
filter/project/aggregate data links as missing. Those v1 labels are preserved under
`data/rl/grounding_external_audit_v1/` but are invalid for precision conclusions.

The v2 package adds trusted harness `data/value` edges as context while asking the reviewer to label
only grounding edges. Its full Flash result is:

| Level | Valid/pass | Invalid/fail | Ambiguous | Error |
|---|---:|---:|---:|---:|
| Trajectory overall | 275 | 35 | 12 | 0 |
| Final dependency | 298 | 20 | 4 | 0 |
| Individual grounding edge | 1,065 | 20 | 3 | 0 |

Among decided labels, Flash reports 98.16% local-edge precision and 93.71% final-dependency
precision. These are reviewer estimates, not calibrated human precision.

The v2 Pro selection contains 85 cases; 84 completed: 50 pass, 27 fail, 7 ambiguous, 1 unresolved.
Flash/Pro agree on 20 fails and 31 passes; 17 are ambiguous/disputed and 16 are direct pass/fail
disagreements. In the 30 Flash-pass controls, Pro gives 24 pass, 2 fail, 3 ambiguous, and 1 error
(2/26 = 7.69% fail among decided pass/fail controls).

A second exporter issue was then found: `read_subtable` evidence showed only the first five rows,
although an inferred value could occur later in the returned rows. The current v3 exporter keeps
the first three rows plus rows matching the target value, capped at ten; final-table audit samples
likewise retain rows matching the answer. Re-reviewing every case involved in the five v2
double-invalid local edges reduced the two-model confirmed local-edge errors from five to two. One
of five trajectory-level consensus failures became a model disagreement. Thus the latest confirmed
trajectory-failure lower bound is at least **19/322 = 5.90%**, but a final calibrated precision
number requires a full v3 audit or human adjudication.

## Confirmed implementation failures

### Cross-column literal collision

`bird_train_00835` asks for Mumbai Indians' 2013 win percentage. A read of the `Season` table
contains the number `7` in an unrelated field. The current row matcher flattens all cells and uses
value membership only, so it creates:

```text
read_subtable(Season) -> condition_filter(Match_Winner = 7)
```

Both reviewers mark this edge invalid. The observation never establishes that Mumbai Indians has
`Team_Id = 7`.

### Answer-value collision

`bird_train_02424` asks how many qualifying orders exist. A shipping-method observation contains
identifier `2`, while the final order count is also `2`. The matcher links that observation directly
to `answer_from_context` even though the action consumes the aggregate handle, not the shipping
identifier. Both reviewers mark the edge invalid.

### Incomplete multi-source final dependency

The nearest-last-table heuristic represents only one final handle. Cases such as
`bird_train_04133` answer with both a venue from an earlier handle and a winning team from the last
handle. The venue observation can be omitted from the final slice even though the final denotation
is correct.

### Missing observation completeness

Several replay-correct trajectories use an entity id, denominator, date condition, or schema column
without an observation establishing it. Examples include car 382 -> country id, customer name ->
customer id, total coin count for a percentage, and Trainer vs Trainee. Execution correctness alone
does not prove a causally grounded exploration trajectory.

## Required fixes before process RL

1. Make row grounding column-aware. Match `(source table, source column, value)` to the exact target
   argument role; never flatten all cells into a bag of values.
2. Do not infer direct answer grounding from scalar equality alone. Require the answer's structured
   source handle/column or an explicit harness data dependency.
3. Support multi-handle final answers, or conservatively withhold final grounding credit when the
   answer contains values not supported by the selected final handle.
4. Add deterministic completeness checks for action columns and literals, then rerun reward reports
   and the current v3 external audit.

## Artifacts

- Formal corrected full audit: `data/rl/grounding_external_audit_v2/`
- Latest target-row sampling probe: `data/rl/grounding_external_audit_v3/`
- Historical invalid package iteration: `data/rl/grounding_external_audit_v1/`

All raw response content, provider finish metadata, strict-validation errors, selection reasons,
packages, per-model reviews, consensus summaries, and confirmed-failure cases are retained.
