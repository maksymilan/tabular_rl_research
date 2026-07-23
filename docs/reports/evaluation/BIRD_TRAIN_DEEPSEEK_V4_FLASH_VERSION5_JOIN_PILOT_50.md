# DeepSeek V4 Flash Version5 Join Pilot — 50 Hard BIRD-Train Tasks

Date: 2026-07-23

## Scope

This pilot evaluates the version5 public join interface:

```json
{
  "base": "orders",
  "joins": [
    {
      "table": "customers",
      "on": [{"left": "orders.customer_id", "right": "id"}]
    },
    {
      "table": "regions",
      "on": [{"left": "customers.region_id", "right": "id"}]
    }
  ]
}
```

The interface keeps a connected N-way join in one action, removes model-authored prefixes and
join-local projection, uses a flat `relation.column` output namespace, and reserves semantic
`base_role`/`role` only for repeated relations. Historical version1-version4 calls remain
replay-compatible but are rejected for new model actions.

The teacher was `deepseek-v4-flash`. Generation was a real state-only model↔harness loop with
strict parsing, hidden gold SQL, one attempt per task, 30 maximum actions, three recoverable errors
per type, eight workers, and 2,048 maximum completion tokens. The formal run's protocol hash was
`6982e73230a3cca2`.

## Cohort Selection

The prior version4 200-task causal run contained 97 distinct tasks in which the model actually
attempted `join_tables` (127 calls). The pilot selected the top 50 using only:

- prior parsed join arity and edge count;
- join-local argument/execution errors;
- join retry count and total action count;
- distinct join inputs and action-tool diversity.

Selection did not read `gold_sql`, gold execution results, difficulty labels, terminal correctness,
legality, or failure type. Full source records containing verifier-only gold SQL were joined back
only after the 50 task ids were frozen. The final cohort spans 28 databases.

## Formal Run Results

| Metric | Version5 result |
| --- | ---: |
| Tasks | 50 |
| Verifier-correct | 13 (26.0%) |
| Fully legal trajectories | 27 (54.0%) |
| Final provider/protocol failure | 23 |
| Final wrong answer | 14 |
| Mean / median / p90 / max actions | 10.52 / 10 / 15 / 17 |
| Tasks with a version5 join attempt | 36 |
| Version5 join attempts | 45 |
| Structurally canonical version5 calls | 45/45 |
| Successfully executed joins | 42/45 (93.33%) |
| Join-local errors | 3 |

The low whole-episode legal/correct rate cannot be interpreted as a join-interface rate:
23 trajectories ended because DeepSeek repeatedly violated its split response carrier after already
executing multiple semantic actions. Join-local metrics isolate the interface from that provider
noise.

## N-way Versus Binary Length

Observed join arities were:

| Arity | Calls |
| ---: | ---: |
| 2 | 34 |
| 3 | 10 |
| 4 | 1 |

Eleven of 45 calls were genuinely multiway. Replacing them with binary calls would require 57 join
actions instead of 45: 12 extra calls, a 26.7% increase in join actions. Spread across the complete
50-task cohort this is 0.24 extra actions per episode, raising the observed mean trajectory length
from 10.52 to approximately 10.76 if everything else stayed fixed. Keeping a connected join
component N-way is therefore justified even though most individual joins are binary.

## The Remaining Namespace Failure

All three join-local errors had the same cause. When continuing from a derived composite handle,
the model invented a new handle namespace:

- `filter_003.BusinessEntityID` instead of one of the returned source namespaces;
- `join_001.disabled.name` instead of `disabled.name`;
- `join_003.customer_id` instead of `rental.customer_id`.

The harness correctly rejected these calls because the first two were also suffix-ambiguous.
Silently stripping the derived handle would make an arbitrary source-instance choice and weaken
grounding. The next controlled change should expose a compact `logical_namespaces` summary beside
each handle and explicitly tell the model to copy `on.left` from the handle's returned logical
columns. It should not accept ambiguous `handle.column` aliases.

## Output Width and Context Pressure

All 42 successful joins produced a flat namespace and none recursively prefixed columns with a join
handle.

| Join-output metric | Version5 |
| --- | ---: |
| Column count median / p90 / max | 14.5 / 31 / 56 |
| Total column-name characters median / p90 / max | 254.5 / 625 / 1,309 |

This confirms the context-pressure concern. The old same-id version4 artifacts had a median width
of four because join-local `return_columns` often projected early. Reintroducing `return_columns`
would again mix join and projection semantics. A cleaner remedy is compact namespace/schema
rendering or lazy column expansion in resident state while leaving `project` as the independent
relational atom.

## Replay and Reference Audit

All 13 verifier-correct trajectories replayed to the same denotation. They contained 14 executed
join steps:

- 14/14 had exactly one harness-authored data edge for every `base`/`joins[].table` input;
- 14/14 join steps appeared in the final backward dependency slice;
- 13/13 had complete action-literal grounding;
- 11/13 had complete final-value grounding and deterministic end-to-end grounding.

The two incomplete final-value cases were not join-lineage failures:

- `bird_train_05640` manually computed `2013 / 10000 * 100 = 20.13`;
- `bird_train_02088` manually computed `50 - 114 = -64`.

Both terminal numbers were verifier-correct but no tool action produced the arithmetic result.
This exposes a separate atomic-semantic gap: the tool set needs a grounded scalar arithmetic atom
(or a narrowly typed scalar expression tool) before these episodes can pass the process-credit
gate.

## Descriptive Version4 Comparison

For the same selected ids, the prior version4 artifacts had 80 join attempts, 60 successful
executions, and 20 join-local execution errors (75.0% local success). Version5 had 45 attempts,
42 successes, and three errors (93.33%).

This is not a causal A/B estimate: selection was conditioned on the prior version4 behavior, and
that run used bounded rolling context while version5 used state-only context. It is nevertheless
consistent with the hypothesis that `base + joins[]` removes substantial prefix/projection
argument friction.

## Decision

Keep the version5 N-way component and simplified arguments. Keep flat non-recursive source
namespaces and keep projection separate. Before a larger controlled evaluation:

1. add compact logical-namespace metadata to derived handle state and strengthen the continuation
   rule without adding alias repair;
2. reduce wide-handle rendering pressure without putting selection back inside join;
3. add a grounded scalar arithmetic atom;
4. isolate or fix DeepSeek split-carrier failures, then run a protocol-matched randomized A/B
   rather than using the selection-conditioned version4 comparison.

## Artifacts

- Selection: `data/eval_inputs/bird_train_join_interface_version5_hard50.jsonl`
- Selection audit: `data/eval_inputs/bird_train_join_interface_version5_hard50.jsonl.selection.json`
- Verified trajectories:
  `data/trajectories/bird_train_join_interface_version5_hard50_deepseek_v4_flash_stateonly_success.jsonl`
- All formal attempts:
  `data/trajectories/bird_train_join_interface_version5_hard50_deepseek_v4_flash_stateonly_success.all.jsonl`
- Formal manifest:
  `data/trajectories/bird_train_join_interface_version5_hard50_deepseek_v4_flash_stateonly_success.manifest.json`
- Structured audit:
  `data/trajectories/bird_train_join_interface_version5_hard50_deepseek_v4_flash_stateonly_audit.json`

An initial sandbox-only network probe attempted 20 records and failed at action zero with
`Operation not permitted`. Those files have a different name and are not included in any metric
above.
