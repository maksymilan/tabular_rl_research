# SQL-ASTRA Qwen2.5-7B-Instruct BIRD Baseline Reproduction

Date: 2026-07-29, completed 2026-07-30

## Current conclusion

The apparent gap is smaller than 47.5 versus 38.98:

| Result | Metric/input contract | BIRD dev |
| --- | --- | ---: |
| Historical local result | old `strict-multiset`, canonical JSON schema prompt | 598/1534 = 38.98% |
| Current comparable local result | `bird-set`, canonical JSON schema prompt | 650/1534 = 42.37% |
| Reproduced ASTRA-style input | `bird-set`, Appendix-F DDL/descriptions/values | **679/1534 = 44.26%** |
| SQL-ASTRA table | labelled greedy; undisclosed exact baseline implementation | 47.5% |

The reproduced disclosed input closes **1.89 percentage points** of the comparable 5.13-point gap.
It remains **3.24 points below** the paper's rounded 47.5 result, so this is a partial, not exact,
reproduction. The apparent starting gap was 5.13 points, not 8.52.
The 38.98 artifact counts duplicate multiplicity while released BIRD EX uses set equality; it must
not be compared directly with SQL-ASTRA's number.

The paired result is not statistically decisive: the ASTRA-style prompt gains 178 tasks and
regresses 149, for an exact two-sided McNemar/binomial `p=0.1214`. Its clearest measurable effect is
schema grounding: generated-SQL execution errors fall from 370 to 334, while executable
wrong-result cases rise from 514 to 521.

SQL-ASTRA cites the generic Qwen2.5 release for its 47.5 row, but the Qwen2.5 technical report does
not contain a BIRD or Text-to-SQL evaluation. The SQL-ASTRA paper and its public landing pages do
not identify an evaluation-code repository. Consequently 47.5 is not currently auditable as an
official Qwen baseline, and the exact hidden generation configuration cannot be recovered from the
citation alone.

## Strongest disclosed recipe

The paper discloses the following relevant details:

- model: `Qwen/Qwen2.5-7B-Instruct`;
- split: all BIRD dev tasks;
- table footnote: greedy decoding for BIRD and Spider;
- one-shot baseline: the same model in a single-turn SQL setting;
- Appendix F input:
  - SQLite DDL rather than a compact JSON schema;
  - original table and column identifiers;
  - BIRD natural-language column descriptions;
  - two live representative values for nearly every column;
  - primary and foreign keys;
  - BIRD external knowledge immediately before the question;
  - explicit step-by-step and exact-output instructions.

The current canonical local prompt already supplies all columns, types, keys, and BIRD external
knowledge. It does **not** supply natural-language column descriptions or representative values.
Those are the principal controlled input differences.

There is one paper-internal ambiguity: Table 1 labels BIRD evaluation as greedy, while Appendix C
lists validation temperatures `0.6 / 1.0` without mapping them unambiguously to each reported
baseline. This reproduction follows the explicit Table 1 footnote and uses true greedy decoding.

## Implemented isolated profile

`src/eval/direct_sql_prompt.py` now defines two independent input profiles:

- `canonical-json-v1`: byte-for-byte compatible with the historical direct-SQL prompt;
- `sql-astra-appendix-v1`: the closest reproducible single-turn version of Appendix F.

The prompt layer is independent of decoding, SQL execution, candidate aggregation, and denotation
scoring. Both `src/eval/text2sql.py` and `src/eval/text2sql_passk.py` accept:

```text
--prompt-profile
--schema-value-count
--schema-metadata-json
```

The SQL-ASTRA renderer deterministically samples two distinct, non-null, non-empty live values per
column, truncates strings to 40 characters, and renders BIRD descriptions and key relationships.
Its longest BIRD-dev input is approximately 16.5K characters, so the configured 8,192-token model
context plus 2,048-token output budget is sufficient.

Regression tests establish that the canonical renderer is unchanged and that the new profile
contains comments, live values, PK/FK constraints, external knowledge, and the required output
carrier. The local and deployed table_rl suites both pass 38/38 tests.

## Frozen reproduction contract

The queued run uses:

| Field | Value |
| --- | --- |
| Model | `/home/dengyan/models/Qwen2.5-7B-Instruct` |
| Tasks | BIRD dev 1,534 |
| Prompt | `sql-astra-appendix-v1` |
| Representative values | 2 per column |
| Decoding | `n=1`, temperature 0, top-p 1 |
| Repetition penalty | 1.0, explicitly pinned |
| Max output | 2,048 tokens |
| Generated-SQL timeout | 10 seconds |
| Denotation | `bird-set` |
| Thinking mode | disabled |
| Execution feedback | none |

This is intentionally a direct-SQL baseline. It does not use SQL-ASTRA's three-turn execution
agent, CSMR, ATR, training data, or RL checkpoint.

## Completed result and paired analysis

The first queued smoke reached the server after GPU 0 was released, but failed before inference
because the copied task JSONL contained macOS absolute SQLite paths. This failure is preserved in
the original directory. The v2 launcher explicitly uses the server's verified remote task export:

- tasks:
  `/home/dengyan/tabular_rl_project/data/eval_inputs/bird_dev_20240627.remote.jsonl`;
- validation: 1,534 tasks, 1,534 unique ids, zero missing database paths;
- runtime:
  `/home/dengyan/tabular_rl_outputs/sql_astra_baseline_runtime_20260729`;
- queue log:
  `/home/dengyan/tabular_rl_outputs/logs/qwen2.5-7b-instruct_sql-astra-appendix-v1_greedy_bird-set_remoteinput-v2.queue.log`;
- smoke result:
  `/home/dengyan/tabular_rl_outputs/evaluations/qwen2.5-7b-instruct_sql-astra-appendix-v1_greedy_bird-set_remoteinput-v2_smoke32`;
- full result:
  `/home/dengyan/tabular_rl_outputs/evaluations/qwen2.5-7b-instruct_sql-astra-appendix-v1_greedy_bird-set_remoteinput-v2_dev1534`.

The 32-task transport/context smoke completed with no API, context-overflow, or incomplete-response
failures and scored 6/32; the canonical control is 4/32 on the same nonrepresentative contiguous
prefix.

The complete paired result is:

| Outcome | Tasks |
| --- | ---: |
| Both correct | 501 |
| ASTRA-style only correct | 178 |
| Canonical only correct | 149 |
| Both wrong | 706 |
| Net | **+29** |

By official BIRD difficulty:

| Difficulty | ASTRA-style | Canonical | Difference |
| --- | ---: | ---: | ---: |
| Simple, n=925 | 499/925 = 53.95% | 473/925 = 51.14% | +2.81pp |
| Moderate, n=464 | 144/464 = 31.03% | 146/464 = 31.47% | -0.43pp |
| Challenging, n=145 | 36/145 = 24.83% | 31/145 = 21.38% | +3.45pp |

One infrastructure exception is fully audited. `bird_dev_00701` was the last unfinished task
because its gold SQL occupied one CPU core for more than 13 minutes; the old canonical artifact
also records 105.9 seconds for that gold query. After the other 1,533 tasks were durable, the stuck
worker was terminated. The same live model and identical greedy request regenerated q701
deterministically; its output omitted the required carrier, the unchanged parser extracted invalid
SQL, and it was recorded as an execution failure. The recovery metadata is stored directly in that
task record. Its hidden denotation was not needed to decide the failure.

The local mirrored artifacts are:

- `data/results/qwen2.5_7b_sql_astra_appendix_v1_greedy_dev1534_bird_ex_remoteinput_v2/`;
- paired analysis: `paired_analysis.json`;
- reproducible analyzer: `src/eval/analyze_sql_astra_baseline.py`.

## Interpretation

The disclosed SQL-ASTRA input recipe is genuinely useful, but it does not reproduce the paper's
47.5:

1. DDL, BIRD descriptions, representative values, and stronger output instructions improve the
   local baseline from 42.37 to 44.26 and reduce execution errors by 36.
2. The prompt also causes substantial policy churn: 149 previously correct tasks regress, moderate
   difficulty is slightly worse, and the paired gain is not significant at 0.05.
3. The remaining 3.24-point gap can depend on undisclosed prompt details, model snapshot,
   generation defaults, baseline construction, or evaluation implementation. The paper provides no
   public baseline code sufficient to distinguish them.
4. A controlled follow-up should separate descriptions from live values on the same task ids. The
   current result must not be reported as an exact 47.5 reproduction.
