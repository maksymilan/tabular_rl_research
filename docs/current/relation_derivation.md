# Relation Derivation Contract

Status: active model-visible metadata contract for protocol `version24`.

Source of truth: `src/harness/relation_derivation/`.

## Boundary

A table-producing action has three separate products:

1. the executor creates a relation handle with concrete SQL denotation;
2. provenance records harness-authored data/value consumption edges;
3. relation derivation records fact-only formal semantics for the output handle.

Derivation never changes the action arguments, SQL, result rows, provenance references, terminal
evidence, or reward. It never interprets the question and never recommends a later action.

Every active table-producing tool has one derivation builder:

- `condition_filter`
- `project`
- `scalar_compute`
- `join_tables`
- `group_aggregate`
- `extreme_value_select`
- `set_op`

Historical `pivot` replay also has a builder. A coverage test fails if the active table action space
and this list drift.

## Canonical envelope

```json
{
  "schema": "relation-derivation-v1",
  "operator": "project",
  "inputs": [
    {"kind": "table", "role": "input", "ref": "join_003"}
  ],
  "semantics": {
    "row_operation": "preserve",
    "column_operation": "project",
    "column_lineage": [
      {
        "output": "full_name",
        "sources": ["person.first_name", "person.last_name"],
        "kind": "expression",
        "expression": "person.first_name || ' ' || person.last_name AS full_name"
      }
    ]
  }
}
```

Top-level keys are exactly `schema`, `operator`, `inputs`, and `semantics`. The schema validator
rejects unknown top-level shape, unsupported operators, malformed inputs, missing
operator-specific fields, incomplete project lineage, unordered join edges, and policy fields such
as `advice` or `recommendation`. Every operator declares both a `row_operation` and a
`column_operation`; this shared shape prevents one-off feedback formats.

## Operator semantics

- Filter records the exact predicate, predicate columns, scalar/table predicate dependencies, and
  whether columns are preserved or projected.
- Project records row preservation/deduplication, authored expressions, and complete ordered
  lineage, including wildcard expansion.
- Scalar compute records value/constant operands, their order, arithmetic operation, and result
  column.
- Join records ordered inputs and every join edge, including type and predicates. A left edge may
  additionally record the measured number of final output rows null-extended at that edge.
- Aggregate records output row grain, row/column layout, passthrough columns, category axis, and
  each metric's operation/source/exact condition and referenced scalar/table dependencies.
- Extreme selection records ordering, limit, and column preservation/projection.
- Set operation records operand roles, operation, and duplicate semantics.
- Historical pivot records its reshape axis and marks itself historical.

These fields describe what was executed. They do not say whether that operation was appropriate
for the user question.

## Resident state

Derivation is stored under the table that it produced:

```json
{
  "tables": {
    "project_004": {
      "created_by": "step_7",
      "columns": ["full_name"],
      "row_count": 10,
      "derivation": {"schema": "relation-derivation-v1", "...": "..."}
    }
  }
}
```

There is no global “latest feedback” slot and no feedback lifecycle independent of a table. Compact
rolling observations point to resident state rather than duplicate derivation payloads.

## Module ownership

- `src/harness/executor.py`: SQL and relation inspection primitives.
- `src/harness/relation_derivation/schema.py`: canonical schema and validation.
- `src/harness/relation_derivation/lineage.py`: expression/condition dependency analysis.
- `src/harness/relation_derivation/builders.py`: operator-specific construction.
- `src/harness/relation_derivation/__init__.py`: stable public interface.
- `src/harness/provenance.py`: data/value/grounding reference edges.
- `src/harness/environment_state.py`: artifact-bound resident storage.
- `src/eval/rollout.py`: orchestration only; it invokes the harness modules.
- `src/sft/protocol.py`: model-visible rendering only.

SFT and RL reuse the same execution entry, so the derivation contract is identical across data
generation, evaluation, replay, and training environments. The live public action schema is
unchanged: derivation is a harness-owned result contract, not another tool or another call
parameter.
