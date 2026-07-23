# Version2 remaining terminal-error audit

Date: 2026-07-22

Scope: the 10 `protocol_error` and 2 `execution_error` terminal failures remaining after the
version2 replay of 110 version1 failures.

## Protocol failures

The 10 terminal cases contain 30 protocol-error events:

| DeepSeek adapter rejection | Events |
|---|---:|
| Visible reasoning/prose before `<tool_call>` | 28 |
| Incomplete or incorrectly suffixed tool-call block | 2 |

These are provider carrier failures, not relational-tool semantic failures. In almost every case the
API supplied native `reasoning_content`, but visible content repeated reasoning before a valid call.
The strict adapter correctly refused to strip that prose.

The version2 feedback was misleading: after adapter rejection, the downstream canonical parser
usually reported `expected exactly one non-empty <think> block`. DeepSeek's actual contract requires
the opposite in visible content: reasoning belongs in native `reasoning_content`, and visible content
must contain only the tool call. Repeating the canonical error encouraged the same invalid response.

Version3 now intercepts a known adapter rejection before canonical parsing and returns a precise
`LAST TOOL ERROR`, for example:

```text
DeepSeek split-response transport error: visible content contained reasoning/prose before the
tool_call; put all reasoning only in native reasoning_content. On the retry, visible content must be
exactly one complete <tool_call>...</tool_call> with nothing before or after it.
```

This does not repair, strip, or accept invalid output. The rejected action remains audit-only and
spends the normal action budget.

## Execution failure: `bird_train_06502`

Question: most populated city's name, located country, and life expectancy.

The model joined City and Country with prefixes but supplied dotted `return_columns`:

```json
["city.Name", "country.Name", "country.LifeExpectancy"]
```

Version2 collapsed the first two output aliases to duplicate `Name` columns, exposed by SQLite as
`Name` and `Name:1`. The model then correctly reused the exposed `Name:1` in `project`, but the
project renderer treated it as raw SQL and failed three times with `near ":1": syntax error`.

There were two harness defects:

1. join projection silently accepted non-contract dotted names and could even turn unresolved quoted
   names into SQLite string literals; and
2. project could not quote an exact environment-exposed duplicate name such as `Name:1`.

Version3 fixes both boundaries:

- `join_tables.return_columns` now requires exact model-facing output columns such as
  `city__Name` and rejects `city.Name` with available columns;
- projected join columns retain their unique resolved names instead of collapsing to `Name/Name:1`;
- `project` can quote an exact exposed column and alias forms such as `Name:1 AS country_name`.

Local BIRD replay confirms that valid version3 arguments produce stable
`city__Name`, `country__Name`, and `country__LifeExpectancy` columns. This task may still have a
separate semantic mismatch because its gold query returns country code rather than full country name;
the environment bug is fixed, but correctness is not assumed.

## Execution failure: `bird_train_05248`

Question: highest-budget film in each genre.

All three execution errors are invalid n-way join edges. The model repeatedly prefixed right-table
source columns (`movie__movie_id`, `G2__MG__genre_id`) or mixed source/table-qualified forms. Under the
current contract, each right key must be a bare column from the newly attached table, while later
left keys use materialized prefix names.

Version3 replaces the two-table cookbook example with a three-table path showing both folds:

```json
{
  "tables": ["orders", "customers", "regions"],
  "on": [
    [{"left": "O__customer_id", "right": "id"}],
    [{"left": "C__region_id", "right": "id"}]
  ],
  "prefixes": ["O", "C", "R"]
}
```

This is the limit of a prompt-only repair. If the model continues to confuse left materialized names
and right source names, version4 should replace the asymmetric string convention with structured
source-instance column references owned and validated by the harness. The executor must not silently
strip prefixes or rewrite an invalid edge.

## Verification and pending replay

- Harness: 49/49
- SFT tests: 46/46
- Eval tests: 16/16
- Version3 protocol hash: `ef7f3f024b18672d`
- Version3 replay input:
  `data/eval_inputs/bird_train_tool_interface_ablation12_terminal_errors_version3.jsonl`

The external 12-task Flash replay could not start because the API execution approval reported the
account usage limit. No partial version3 output was created. The code and local execution fixes are
verified; external recovery counts remain pending until API access is available.
