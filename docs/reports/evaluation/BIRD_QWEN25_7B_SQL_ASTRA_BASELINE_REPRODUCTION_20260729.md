# SQL-ASTRA Qwen2.5-7B-Instruct BIRD Baseline Reproduction

Date: 2026-07-29

## Current conclusion

The apparent gap is smaller than 47.5 versus 38.98:

| Result | Metric/input contract | BIRD dev |
| --- | --- | ---: |
| Historical local result | old `strict-multiset`, canonical JSON schema prompt | 598/1534 = 38.98% |
| Current comparable local result | `bird-set`, canonical JSON schema prompt | 650/1534 = 42.37% |
| SQL-ASTRA table | labelled greedy; stronger disclosed schema/value prompt | 47.5% |

The current like-for-like gap to investigate is therefore **5.13 percentage points**, not 8.52.
The 38.98 artifact counts duplicate multiplicity while released BIRD EX uses set equality; it must
not be compared directly with SQL-ASTRA's number.

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

## Runtime status and artifacts

Both table_rl GPUs were occupied by independent process-RL runs when the reproduction was prepared,
so no running experiment was interrupted. An isolated runtime passed all tests and the baseline was
queued for GPU 0:

- queue PID: `3068106`;
- runtime:
  `/home/dengyan/tabular_rl_outputs/sql_astra_baseline_runtime_20260729`;
- queue log:
  `/home/dengyan/tabular_rl_outputs/logs/qwen2.5-7b-instruct_sql-astra-appendix-v1_greedy_bird-set.queue.log`;
- smoke result:
  `/home/dengyan/tabular_rl_outputs/evaluations/qwen2.5-7b-instruct_sql-astra-appendix-v1_greedy_bird-set_smoke32`;
- full result:
  `/home/dengyan/tabular_rl_outputs/evaluations/qwen2.5-7b-instruct_sql-astra-appendix-v1_greedy_bird-set_dev1534`.

The queue first runs a 32-task transport/context smoke and launches the complete 1,534-task
evaluation only if the smoke has no API, context-overflow, or incomplete-response failures.
This section must be updated with the observed score and paired task analysis after completion.

## Interpretation gate

The outcome separates three cases:

1. **Near 47.5:** most of the prior gap was schema/value prompt strength, not model weights.
2. **Near 42.4:** 47.5 depends on an undisclosed decode, prompt detail, database snapshot, or
   evaluation implementation.
3. **Between them:** run a paired ablation that adds column descriptions and live values
   independently to measure which component supplies the gain.

No 47.5 reproduction claim should be made until the full artifact is complete and every task is
checked under `bird-set`.
