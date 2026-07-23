# BIRD version22 structural-feedback pilot (16 tasks, 2026-07-23)

## Decision

**Do not expand version22 to the 50-task stage.** The candidate was non-destructive but missed its
predeclared target-recovery gate.

## Frozen comparison

- Model: `deepseek-v4-flash`
- Context: `rolling-legal-history`, 4 successful turns
- Prompt: canonical
- Plan policy: optional
- Provider carrier: JSON Output with native reasoning
- Maximum actions/tokens: 30 / 2048
- Denotation metric: `bird-set`
- Candidate input:
  `data/eval_inputs/tool_usability_20260723/version22_structural_feedback_pilot16.jsonl`
- Candidate result:
  `data/trajectories/tool_usability_20260723/version22_structural_feedback_pilot16_json.all.jsonl`
- Paired control:
  `data/trajectories/tool_usability_20260723/version20_final_fixed200_json.all.jsonl`

The cohort contains seven unambiguous target failures, one benchmark-ambiguity diagnostic
(`bird_train_04038`), and eight previously correct controls. The predeclared gate was at least
3/7 target recoveries, at most one control regression, and no material action-length increase.

## Result

| Measure | version20 paired records | version22 pilot |
|---|---:|---:|
| Correct overall | 8/16 | 10/16 |
| Unambiguous targets correct | 0/7 | 2/7 |
| Controls correct | 8/8 | 8/8 |
| Diagnostic `04038` correct | 0/1 | 0/1 |
| Legal termination | 16/16 | 16/16 |
| Protocol/execution error actions | not used for the paired gate | 0 |
| Mean actions | 6.875 | 6.813 |

The two recovered targets were `bird_train_02512` and `bird_train_02507`. Both used the
aggregate-shape feedback before selecting the maximum group, but a single stochastic paired run
cannot distinguish a feedback effect from ordinary sampling variation.

## Target-level audit

| Task | Candidate | Relevant observation |
|---|---|---|
| `04189` | wrong | The left join reported 291 unmatched base rows. The next reason explicitly accepted NULL reviews and retained the left join. |
| `02901` | wrong | Projection reported collapse of first/middle/last into one output. The model quoted the warning, classified it as advisory, and kept `FullName`. |
| `05544` | wrong | Aggregate reported `row_grain=[]` and `count=input_rows`; the model still counted six joined term rows rather than distinct representatives. |
| `00582` | wrong | Projection reported collapse of first/last into one output. The model quoted the warning but treated “full name” as authorization to concatenate. |
| `02512` | correct | Grouped by employee, selected the maximum, joined employee, and kept first/last separate. |
| `02507` | correct | Grouped by employee, selected the maximum, joined employee, and kept first/last separate. |
| `02189` | wrong | Aggregate reported a single global row. The model cross-joined that global total onto every brand instead of summing per brand. |
| `04038` | wrong diagnostic | Produced one row per gender and counted distinct patients; gold uses one row with female/male conditional row counts. Question/gold layout remains ambiguous. |

Fourteen structural observations were emitted: eleven `aggregate_shape`, two `column_collapse`, and
one `left_join_match`. They did not change public tools, action arguments, execution legality,
table denotations, provenance references, or replay records.

## Interpretation

The implementation successfully exposed grounded local facts without damaging the controls, but
its action guidance was too weak. In the failed traces the model did not miss the observation; it
actively rationalized why the warning did not apply. Therefore simply scaling this exact candidate
would not establish that the tool design repairs model-capable failures.

The narrow next candidate should keep the same tools and arguments and change only the existing
feedback payload:

1. Make projection collapse explicit that labels such as “name” or “full name” do not imply a
   formatted single string when task guidance names separate stored fields.
2. Make aggregate feedback state that `count(*)` preserves join multiplicity and that
   `row_grain=[]` is one global total, not a per-entity metric suitable for cross replication.
3. Make left-join feedback state that unmatched rows have no requested right-side attribute and
   should be excluded unless missing/unmatched entities are explicitly requested.

This is still a local policy hint layered on harness facts, not a new relational primitive. It
must be reported as such in later comparisons. Reuse this exact 16-task cohort before any 50-task
run.
