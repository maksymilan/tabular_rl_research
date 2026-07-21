# BIRD dev-30 difficulty and scoring audit

Date: 2026-07-18

## Trigger

The behavior-selected Grounded R2 epoch-1 adapter scored 3/12 on `simple` and 3/9 on
`challenging` in the fixed BIRD-dev 30 gate. This audit checks whether the challenging cases are
actually difficult for the current tool environment and whether simple answers were falsely
rejected.

## Difficulty labels

The adapter does not calculate these labels. `src/harness/bird_dev_adapter.py` copies the
`difficulty` field verbatim from the released `data/bird/dev_20240627/dev.json`. The 12/9/9 sample
therefore has the official BIRD labels, with no local relabeling bug.

Official BIRD difficulty and current tool-agent difficulty are not the same axis. External evidence
is visible to the model and can collapse semantic difficulty, while apparently simple SQL may be
awkward under the atomic tool API.

Gold-SQL proxy averages in the fixed sample:

| Label | Joins | SELECTs | Aggregates | SQL chars |
|---|---:|---:|---:|---:|
| simple | 1.08 | 1.00 | 0.42 | 153.6 |
| moderate | 1.56 | 1.22 | 1.11 | 235.1 |
| challenging | 1.44 | 1.22 | 1.33 | 252.8 |

The averages preserve the expected trend, but individual cases overlap heavily.

### Challenging successes

- `513`: single table, one filter, one sort, `LIMIT 1`. The evidence directly maps the wording to
  `type='commander'` and `MAX(totalSetSize)`. This is officially challenging but easy in the actual
  model-visible task.
- `724`: two joins to the same colour lookup plus two equality filters. This is moderate tool
  difficulty.
- `1194`: two joins and four exact predicates across medical/laboratory tables. This is genuinely
  challenging relative to the other sampled tasks.

Thus only one of the three challenging successes is unambiguously hard for the harness agent.

The challenging failures include genuinely complex percentage calculations, subqueries,
conditional aggregation, and grouped top-1 selection. The label bucket itself is not wholly easy;
the observed rate is driven by sample composition and three particular successes.

### Simple bucket is not uniformly easy

- `179` requires a three-join path, year extraction, filtering, and sum aggregation.
- `1204` requires date-difference reasoning that is awkward in the current tools and reaches
  `max_steps`.
- `481` repeatedly reads an intermediate distinct-language table and reaches `max_steps`.
- Several others require joins plus aggregation or exact text/date grounding.

The selected simple bucket is structurally harder than full-dev simple on average (1.08 versus
0.75 joins and 153.6 versus 128.6 SQL characters).

## Simple-case scoring audit

The scorer is row-order insensitive, normalizes numeric cells, tolerates answer-column order, and
uses a 4-decimal numeric canonicalization. No failure is caused by row order, numeric type, or float
precision.

| Dev index | Strict result | Audit conclusion |
|---:|---|---|
| 179 | wrong | 63,050,621 versus gold 303,276; genuinely wrong |
| 187 | wrong | 0 versus gold 240; filtered the wrong concept/table |
| 326 | wrong | claims no result; gold has five molecule ids |
| 481 | max steps | no terminal answer |
| 582 | wrong | returns two of four gold titles; incomplete |
| 789 | wrong | returns sum-like 990 instead of average 123.75 |
| 1204 | max steps | no terminal answer |
| 1462 | strict wrong / semantic boundary | reason lists all four correct category/amount pairs, but evidence cites an unprojected seven-column table and structured `answer` is empty |
| 1515 | wrong | claims no transaction; gold segment is `KAM` |

For `1462`, projecting evidence columns `(category, amount)` reproduces the gold rows exactly. The
strict score is nevertheless correct under the current contract: `answer_from_context.evidence`
must identify a table holding the answer rows, and `reason` is explanatory text rather than the
answer channel. Automatically dropping arbitrary evidence columns would create false positives.
Treat it as a diagnostic semantic success, not as a change to official execution accuracy.

## Corrected interpretation

- Strict score: simple 3/12 = 25.0%, challenging 3/9 = 33.3%.
- Human semantic diagnostic counting `1462`: simple 4/12 = 33.3%, challenging 3/9 = 33.3%.
- Both buckets contain exactly three strict successes. Fisher's exact two-sided test for the strict
  2x2 table is `p=1.0`; the inversion has no statistical support.
- Approximate 95% Wilson intervals are very wide: simple 8.9%-53.2%, challenging 12.1%-64.6%.

## Decision

There is no evidence that the model truly performs better on hard tasks. Keep the official BIRD
labels for benchmark reporting, but add a separate harness/tool-complexity stratification for
diagnosis. Keep strict execution accuracy unchanged and report semantic projection matches such as
`1462` only as a secondary diagnostic. Increase the fixed gate to at least 100 tasks before making
difficulty-specific claims.

