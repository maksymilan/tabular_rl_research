# Observation-Action Reflection Trajectory Plan

Status: design-only handoff for the next data iteration. This document does not claim that
reflection trajectories, decision-state fields, or their rewards are implemented.

## 1. Motivation

V2-ctx solves the context-scaling problem structurally: the opening prompt contains a lazy catalog,
table-producing tools return metadata-only handles, and rows enter the prompt only through explicit
perception tools. The full Spider evaluation nevertheless dropped from v1 **68.38%** to v2a
**66.83%** and v2-ctx **62.77%**. This does not invalidate the context design; it shows that the
current SFT data teaches how to execute a gold plan, but provides little supervision for deciding
whether the plan is justified by the observations.

The current 7,767 V2-ctx train+dev trajectories contain 6,361 `read_subtable` calls. Every one is
the penultimate step and is immediately followed by `answer_from_context`. The model therefore sees
`read_subtable` almost exclusively as a final-answer ritual, not as evidence that can confirm,
reject, or revise an unfinished plan.

A representative failure is:

> What are the names and release years for all the songs of the youngest singer?

The necessary columns (`Age`, `Song_Name`, `Song_release_year`) are all in `singer`, so the correct
path is:

```text
describe_table(singer)
-> extreme_value_select(singer, Age ascending, top_k=1)
-> project(Song_Name, Song_release_year)
-> read_subtable
-> answer
```

V2-ctx described both `singer` and `singer_in_concert`, then performed an unnecessary join and
selected `Name` rather than `Song_Name`. V2a also chose `Name` despite receiving the full schema
upfront. The failure is therefore not explained by catalog compression alone: the training data
does not teach the model to compare required columns against candidate operations and revise a
plausible but unsupported interpretation.

## 2. Action Taxonomy

Classify model-visible calls by their role:

| Role | Tools | Purpose |
| --- | --- | --- |
| Observation | `describe_table`, `inspect_column`, `read_subtable` | Acquire schema, value-domain, or row evidence without changing the relational result |
| Action | `condition_filter`, `project`, `join_tables`, `group_aggregate`, `aggregate`, `extreme_value_select`, `set_op`, `add_to_memory` | Transform data or persist a grounded scalar |
| Terminal | `answer_from_context` | Return the final answer with evidence |
| Control state | structured decision/reflection record | Record model reasoning; never treated as factual evidence |

Do not enforce a rigid observation/action alternation. Require observation only when the next
action has an unmet evidence precondition:

- a table or column must have been described before it is used;
- both join inputs must be described, and the required columns must actually span the two sides;
- a text literal must be grounded with `inspect_column` before filtering;
- the final evidence table must be read before answering;
- an intermediate result must be read when its contents, cardinality, or schema can change the next
  operation choice.

The last rule is the missing training signal. A read whose result cannot affect any later decision
is merely verification; a read followed by a changed operation is a decision-point observation.

## 3. Reflection Trajectory Types

### 3.1 Pre-action correction

The model considers a plausible operation, gathers enough evidence, then rejects it before the
operation executes. This is the safest first SFT target because all executed data operations remain
on a verified correct path.

Example:

```text
describe_table(singer, singer_in_concert)
-> decision: joining is unnecessary because singer already supplies Age, Song_Name, and
   Song_release_year
-> extreme_value_select(singer, Age ascending, top_k=1)
```

### 3.2 Post-action recovery

The model executes a legal but unnecessary or incorrect operation, observes a deterministic
contradiction, abandons that artifact, and resumes from a valid earlier handle. Use this in a small
SFT mixture and more heavily in RL or preference data.

Examples of contradictions:

- an unnecessary join increases rows or duplicates entities without contributing a required column;
- a projection exposes `Name` while the question requires `Song_Name`;
- sorting in the wrong direction yields an age inconsistent with the requested minimum;
- filtering produces zero rows after an ungrounded literal.

The incorrect artifact must never become answer evidence, and recovery must not mutate or conceal
the original failed step.

### 3.3 Error-feedback recovery

The tool returns an execution or validation error, and the model uses the error plus a new
observation to repair the call. Initial cases should cover:

- unknown or ambiguous columns;
- invalid table handles;
- zero-row filters caused by a wrong text literal;
- invalid join keys;
- incompatible set-operation schemas.

Do not synthesize arbitrary malformed JSON for this category. The failure should represent a
plausible semantic or interface mistake and have one clearly verifiable repair.

## 4. Structured Decision State

Reflection cannot live only in unconstrained `<think>` prose. Store a compact model-authored control
record alongside each decision point:

```json
{
  "candidate_tables": ["singer", "singer_in_concert"],
  "required_columns": [
    "singer.Age",
    "singer.Song_Name",
    "singer.Song_release_year"
  ],
  "selected_tables": ["singer"],
  "rejected_operations": [
    {
      "operation": "join_tables",
      "reason": "The second table contributes no required columns.",
      "evidence_step_ids": ["step_1"]
    }
  ],
  "unresolved": []
}
```

Ownership and trust boundary:

- the model authors candidate operations, purposes, and rejection reasons;
- the harness owns tool outputs, step IDs, table handles, execution status, and provenance;
- the validator checks that cited evidence steps exist and that the reason is supported by
  deterministic facts such as schemas, row counts, or tool errors;
- decision state is control information, not factual authority, answer evidence, or a valid
  `value_ref`.

This record may later become a typed `plan`/`hypothesis` memory, but the first implementation should
keep it trajectory-local so memory semantics and reflection supervision can be ablated separately.

## 5. Programmatic Construction

Build each example from an execution-verified gold Plan:

1. Compile and execute the gold Plan.
2. Identify decision points: table selection, join choice, literal grounding, sort direction,
   projection choice, and evidence sufficiency.
3. Generate one plausible perturbation from a typed library.
4. Execute the required observation or the perturbed action in an isolated branch.
5. Evaluate a deterministic contradiction oracle.
6. Emit a pre-action correction, post-action recovery, or error-recovery variant.
7. Resume a valid plan from the appropriate earlier table handle.
8. Execute the complete trajectory and require equality with the gold result.
9. Validate step references, argument schemas, evidence dependencies, and absence of answer leakage.

Initial perturbation library:

| Perturbation | Required evidence | Deterministic rejection test |
| --- | --- | --- |
| Unnecessary join | schemas of both sides + required columns | one side contributes no required column or predicate |
| `Name` vs semantic target column | described schema + question slot | selected column does not cover the requested semantic slot |
| Wrong sort direction / `top_k` | requested extremum + ordered result | result violates min/max or requested cardinality |
| Wrong related table | catalog relation + described schemas | table adds no required field or filtering role |
| Ungrounded text literal | `inspect_column` output | literal absent while a plausible canonical value exists |
| Premature projection | downstream required columns | projection removes a column needed by a later verified step |
| Empty filter | filter output + inspected domain | zero rows and predicate value is unsupported by the domain |

Question-to-column semantic alignment is the least deterministic oracle. Start with cases where the
gold SQL and schema names provide a clear mapping; reject ambiguous candidates rather than asking a
template to invent certainty. External LLMs may propose paraphrases or perturbations, but execution
and deterministic validation remain the admission gate.

## 6. Acceptance Gates

Every generated correction trajectory must satisfy:

- the perturbation is plausible and typed, not a random error;
- the observation genuinely supports the stated confirmation or contradiction;
- every reflection cites existing earlier evidence step IDs;
- no harness-owned value, row, provenance edge, or error is authored by the model;
- the recovered final result equals the gold query result;
- abandoned artifacts are not in the terminal backward slice;
- no gold answer value is leaked into decision state or `<think>`;
- equivalent longer correct plans are not mislabeled as incorrect merely for being non-minimal;
- the same protocol parser and executor accept the trajectory online;
- clean-gold and correction variants have separate labels for ablation.

For unnecessary-operation labels, prefer a precise criterion: the operation contributes no required
column, predicate, grouping key, ordering key, or answer dependency. Do not require every correct
trajectory to be globally minimal.

## 7. First Experimental Mixture

Use a conservative first mixture:

- **70%** clean execution-verified gold trajectories;
- **20%** pre-action correction trajectories;
- **10%** post-action and error-recovery trajectories.

Report separate evaluations for:

- clean answer execution accuracy;
- legal tool-call rate;
- unnecessary join rate;
- wrong-column selection rate;
- recovery success after zero-row/error feedback;
- accuracy on examples requiring a decision-point observation;
- context length and action count.

Run an ablation with clean gold only versus clean+pre-action correction before increasing recovery
data. A later RL reward may credit evidence-supported correction, penalize repeated invalid calls,
and penalize unnecessary operations, but reward work is not part of this data-generation change.

## 8. Implementation Order

1. Add an offline analyzer that marks gold Plan decision points and required columns.
2. Implement deterministic precondition checks and contradiction oracles.
3. Define the versioned decision-state schema and validator.
4. Generate pre-action correction variants first.
5. Add post-action branches that can safely resume from earlier immutable table handles.
6. Add error-feedback variants using real executor errors.
7. Rebuild SFT data with explicit variant labels and mixture manifests.
8. Replay all dev trajectories through the online rollout path.
9. Train a small smoke run, inspect success and failure trajectories, then run the 7B comparison.
10. Use verified failed-to-corrected rollouts as the first simple RL dataset.

Implementation should extend the V2-ctx protocol rather than restore full table previews. The
large-database invariant remains: only the latest requested row observation is transiently visible;
older intermediate tables persist as harness handles plus metadata and grounded memory.
