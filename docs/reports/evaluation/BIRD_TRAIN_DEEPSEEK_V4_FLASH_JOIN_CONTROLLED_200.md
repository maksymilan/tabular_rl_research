# DeepSeek V4 Flash Join-Interface Evaluation on 200 Frozen BIRD-Train Tasks

Date: 2026-07-23

## Executive Decision

Keep the version7 semantic design:

- a connected N-way join is one action;
- the public call is `base + joins[]`;
- each `joins[]` item introduces one table with edge-local `on`;
- output columns use flat, stable `relation.column` names;
- a derived handle is a table argument, never a recursive column namespace;
- wide model-visible state groups columns under `column_namespaces`;
- downstream relational tools consume the same logical names, including inside projection
  expressions, and accept a bare suffix only when it is unique.

This is the most model-natural and compositionally complete design among the discussed alternatives.
It reduced join-local execution failures from 20/127 calls under the original version4 interface to
1/100 calls in version7, while preserving exact harness-owned input provenance for every replayed
join.

The version7 **whole-episode accuracy run is not a valid superiority result**, however. Although all
static experiment fields match, the sequential external-teacher runs experienced sharply different
DeepSeek split-carrier behavior. Version7 ended 76/200 tasks with provider/protocol failure, versus
12/200 in version4. The time-local failure burst makes the external service state an uncontrolled
variable. The version7 run validates local semantics and exposes the next experimental requirement;
it does not establish that version7 is worse than version4 as a reasoning policy.

## Compared Designs

The names below denote the join designs discussed in this study, not the older project-wide
“version1–version4” taxonomy:

1. **Original version4 join** — multi-table prefix arrays, model-authored prefixes, and optional
   join-local `return_columns`.
2. **Strictly binary atomic join** — one pair of relations per call, requiring intermediate handles
   for a larger connected component.
3. **Simplified connected N-way join** — `base + joins[]`, one introduced table per item, flat
   source namespaces, and no join-local projection.
4. **Simplified N-way plus compact namespaces** — design 3 with lossless model-visible
   `column_namespaces`.
5. **Closed logical-column semantics (version7)** — design 4 with consistent dotted-column and safe
   unique-bare resolution across filter, projection, grouping, and ordering.

The 50-task state-only version5 pilot was useful for finding namespace failures but is not treated
as a causal comparison because the original version4 run used rolling history. The 200-task runs
below all use the original rolling setting.

## Controlled Setup

All three formal 200-task runs used:

- the exact frozen input
  `data/eval_inputs/bird_train_tool_interface_validation200_version4.jsonl`;
- `deepseek-v4-flash`;
- BIRD train, 200 tasks, one attempt per task;
- strict-multiset denotation;
- `rolling-legal-history`, four history turns, full rolling prompt;
- 30 maximum semantic actions and three recoverable errors per type;
- 2,048 completion tokens, no returned table rows, and eight workers;
- strict no-repair parsing and three API transport attempts.

The structured audits report no mismatches across the recorded control fields. Protocol/tool
semantics and their corresponding prompt text are the intended treatment. Version7's manifest hash
`81697382e3fd4c2a` matches the current rolling/full DeepSeek protocol exactly.

## Whole-Episode Results

| Metric | Original v4 | Compact N-way v6 | Closed semantics v7 |
| --- | ---: | ---: | ---: |
| Correct / 200 | 125 (62.5%) | 121 (60.5%) | 89 (44.5%) |
| Legal terminal | 188 | 165 | 122 |
| Final protocol failure | 12 | 28 | 76 |
| Final wrong answer | 63 | 44 | 33 |
| Final execution / argument failure | 0 / 0 | 6 / 1 | 1 / 1 |
| Mean actions | 7.97 | 8.27 | 8.04 |
| Prompt tokens | 6,632,595 | 7,038,824 | 6,905,665 |

The cleanest end-to-end comparison is version4 versus version6:

- both correct: 106;
- version4 only: 19;
- version6 only: 15;
- neither: 60;
- paired difference: -4/200;
- exact McNemar `p = 0.607591`.

There is therefore no evidence of an overall accuracy difference at this sample size, even though
the join-local behavior changed substantially.

The raw version4-versus-version7 pairing is 79 both correct, 46 version4-only, 10 version7-only,
and 65 neither (`p = 0.000001`). That number must not be read as a join-design effect because the
provider carrier distribution changed drastically.

As a descriptive diagnostic only, among the 122 task ids for which neither run ended in a protocol
failure, version7 is +4 tasks relative to version4 (9 version7-only versus 5 version4-only,
`p = 0.423950`). This conditioning is post-treatment and is not a replacement causal estimate.

## External-Teacher Instability

Version7 recorded 326 protocol-error events:

- 190 were empty native `reasoning_content`;
- 116 put reasoning/prose in visible content;
- 20 were other strict-protocol violations.

For comparison, version4 recorded 18, 93, and 24 in those categories; version6 recorded 46, 103,
and 21.

The version7 failures were strongly clustered by completion time. Tasks completed at positions
101–120 had zero final protocol failures, while positions 181–200 had 19/20. The last bucket
contained 59 protocol-error events. This nonstationarity is incompatible with attributing the raw
whole-episode difference solely to the join treatment. Static configuration control was necessary
but insufficient; future external-teacher A/B runs must interleave the compared protocols within
the same time blocks.

## Join-Local Results

| Metric | Original v4 | Compact N-way v6 | Closed semantics v7 |
| --- | ---: | ---: | ---: |
| Tasks with a join attempt | 97 | 97 | 90 |
| Join attempts | 127 | 106 | 100 |
| Executed joins | 107 | 103 | 99 |
| Join-local success | 84.25% | 97.17% | 99.00% |
| Join-local errors | 20 | 3 | 1 |

Original version4's 20 join failures consisted of 11 join-local projection/`return_columns`
identifier errors and nine edge-identifier errors. Version7 had one remaining failure: the model
incorrectly used a derived handle as a logical column namespace. All 100 version7 join attempts
used the canonical public shape, and all 99 successful outputs had a flat non-recursive namespace.

The improvement is therefore directly aligned with the intended simplification:

- removing model-authored prefix arrays removes alias bookkeeping;
- removing `return_columns` keeps join and projection as separate atoms;
- retaining exact `relation.column` keys prevents ambiguous suffix repair;
- exposing compact logical namespaces tells the model what can legally appear in `on.left`.

## Downstream Closure

Version6 revealed that a correct flat join contract is insufficient if downstream tools interpret a
dotted logical column as a SQL table qualifier. Its execution errors included:

- `project`: 23/103 parsed calls (22.33%);
- `group_aggregate`: 6/118 parsed calls (5.08%).

After version7 added exact logical-column quoting inside expressions and safe unique-bare
resolution:

- `project`: 7/72 parsed calls (9.72%);
- `group_aggregate`: 2/102 parsed calls (1.96%).

The remaining version7 projection errors are not the repaired dotted-column class; examples include
unsupported `day`, malformed `ID` expressions, an unrecognized colon, and references to a table not
present in the operation. Unit tests separately cover dotted arithmetic/concatenation and
unique-bare aggregation, so the fixed semantic gap does not depend only on the noisy teacher run.

## N-way Versus Binary Length

Version7 attempted 90 binary, eight 3-way, one 4-way, and one 5-way connected joins. Strict binary
decomposition would require 113 join actions instead of 100: 13 extra join calls, or 13% more join
actions. On this general 200-task cohort the average whole-episode increase would be only 0.065
actions, but on the earlier hard-join cohort it was 12 extra calls over 45 observed joins (26.7%).

This supports a bounded N-way atom. Most calls remain naturally binary, while the same simple shape
avoids a chain of intermediate handles on the cases where arity is genuinely higher. A binary-only
contract would increase horizon and namespace reuse without simplifying the individual edge
description enough to compensate.

## Context and Namespace Pressure

Version7's 99 successful joins had:

- median / p90 / maximum output width: 10 / 29 / 57 columns;
- median / p90 / maximum column-name characters: 177 / 591 / 1,429.

Compact `column_namespaces` appeared in 352 model turns and saved 62,596 rendered characters:
177.83 characters on average when used, p90 397, maximum 1,916. This confirms that flat logical
names can be compressed losslessly in model-visible state.

It did not reduce total prompt tokens relative to version4 in this run. Version7 made more recovery
requests and carries a newer protocol prompt, so total tokens remain a whole-episode outcome rather
than a direct measure of namespace compression. The local rendering measurement is the appropriate
evidence for the namespace change.

## Reference-Chain Audit

Of version7's 89 verifier-correct trajectories:

- 88 replayed to the same denotation;
- 37 executed join steps were replayed;
- 37/37 had exactly one harness-authored data edge per `base`/`joins[].table` input;
- 36/37 appeared in the final answer's backward dependency slice;
- 76/89 had complete final-value grounding;
- 71/89 had complete deterministic end-to-end grounding.

The single replay exception was the known hidden handle-counter mutation after a rejected action,
which made a later `group_005` unavailable during clean replay. It is a general executor/replay
issue, not loss of join lineage. The incomplete final-value cases primarily expose the separate
grounded scalar-arithmetic gap. Neither should be folded into the join treatment.

## Final Design Ranking

| Design | Model-call fit | Trajectory length | Composition | Decision |
| --- | --- | --- | --- | --- |
| Original prefix-array + join projection | Poor: aliases, prefixes, and selection are coupled | Short | Fragile identifiers | Retire for new actions; retain replay compatibility |
| Binary-only join | Simple per call | Longest on multiway tasks | Reuses intermediate handles heavily | Reject as sole public join |
| Simplified N-way `base + joins[]` | Natural connected-component description | Bounded | Good join-local behavior | Keep as the join atom |
| N-way + compact namespaces | Same call, lower state repetition | Bounded | Downstream semantics initially incomplete | Keep compression |
| Version7 closed logical-column semantics | Best complete contract | Bounded | Join and downstream operators agree | **Recommended current design** |

The key distinction is that “atomic” should mean one semantically coherent connected join component,
not necessarily two physical inputs. Each edge remains atomic inside `joins[]`, while the model
avoids manufacturing a sequence of temporary joins.

## Remaining Validation

Before claiming an end-to-end policy improvement:

1. run version4 and version7 as randomized, interleaved task pairs against the same teacher-service
   time blocks;
2. report split-carrier failures separately and do not retry whole episodes selectively;
3. preserve strict all-task accuracy as the primary causal metric, with carrier-clean results only
   as labeled diagnostics;
4. fix the hidden handle-counter mutation and add grounded scalar arithmetic as separate,
   independently evaluated treatments.

## Artifacts

- Version4 all attempts:
  `data/trajectories/bird_train_version4_validation200_flash_all.jsonl`
- Version6 all attempts:
  `data/trajectories/bird_train_version6_join_validation200_flash_rolling_success.all.jsonl`
- Version7 verified trajectories:
  `data/trajectories/bird_train_version7_join_validation200_flash_rolling_success.jsonl`
- Version7 all attempts:
  `data/trajectories/bird_train_version7_join_validation200_flash_rolling_success.all.jsonl`
- Version7 manifest:
  `data/trajectories/bird_train_version7_join_validation200_flash_rolling_success.manifest.json`
- Version4 versus version7 controlled audit:
  `data/trajectories/bird_train_version4_vs_version7_join_controlled200_audit.json`
- Version6 versus version7 controlled audit:
  `data/trajectories/bird_train_version6_vs_version7_join_controlled200_audit.json`
- Version7 reference audit:
  `data/trajectories/bird_train_version7_join_validation200_flash_rolling_reference_audit.json`
