# DeepSeek V4 Flash Version8 Provider-Prompt Retest on 200 Frozen BIRD-Train Tasks

Date: 2026-07-23

## Decision

The provider-specific prompt ambiguity was real, but it was not the whole explanation for the
earlier gap.

Version8 removed the contradictory requirement to emit canonical `<think>` text in visible content,
converted rolling assistant history to DeepSeek's visible tool-call-only shape, and added an explicit
two-field example. Relative to version7, final protocol failures fell from 76/200 to 36/200 and
correct answers increased from 89/200 to 106/200. The paired all-task improvement was +17
(`p = 0.042957`, exact McNemar).

Version8 nevertheless remained below the original version4 run: 106/200 versus 125/200. The paired
difference was -19 (`p = 0.006609`). Therefore the new run does not support claiming an
end-to-end improvement over the original interface.

The local tool evidence points in the opposite direction from the raw episode score. Version8
executed 90/93 join calls successfully (96.77%), versus 107/127 (84.25%) for version4, and all
verified join reference chains remained intact. The remaining whole-episode deficit is dominated by
DeepSeek split-response carrier failures, not by the simplified join executor.

Keep the version8 semantic tool design, but do not treat this sequential external-service run as the
final causal A/B. A definitive comparison requires randomized interleaving of version4 and version8
requests within the same provider time blocks.

## Treatment

Version8 retains the closed logical-column semantics introduced in version7:

- one connected N-way join action using `base + joins[]`;
- one newly introduced relation per `joins[]` item;
- flat, non-recursive `relation.column` logical names;
- `column_namespaces` as a lossless compact state rendering;
- downstream filter, projection, grouping, and ordering over the same logical names;
- exact bare-suffix resolution only when the suffix is unique.

The version8 treatment changes the DeepSeek request envelope:

- canonical positive visible-`<think>` output instructions are removed from the API-facing prompt;
- successful rolling-history assistant messages contain only the visible `<tool_call>` block;
- the prompt contains exactly one DeepSeek split-response contract;
- the contract gives concrete values for `reasoning_content` and visible `content`;
- multi-step tasks are told to call `plan` early, while simple direct tasks may omit it.

The exact full rolling DeepSeek prompt hash is `387ae13edaa8ed1e`. Its audit has one split-response
contract, zero positive visible-`<think>` rules, the current `base + joins[]` example, and no legacy
prefix-array example.

## Controlled Setup

The rerun used:

- the frozen input
  `data/eval_inputs/bird_train_tool_interface_validation200_version4.jsonl`;
- `deepseek-v4-flash`;
- BIRD train, 200 tasks, one attempt per task;
- `rolling-legal-history` with four history turns and the full rolling prompt;
- 30 maximum semantic actions and three recoverable errors per type;
- 2,048 completion tokens and no table rows returned automatically;
- eight concurrent workers;
- strict no-repair parsing and up to three API transport attempts;
- strict-multiset terminal denotation.

The version4-versus-version8, version6-versus-version8, and version7-versus-version8 audit files all
report zero mismatches in their recorded control fields. Provider time is not controlled because
the runs were sequential rather than interleaved.

## Whole-Episode Results

| Metric | Original v4 | Compact N-way v6 | Closed semantics v7 | Prompt-fixed v8 |
| --- | ---: | ---: | ---: | ---: |
| Correct / 200 | 125 (62.5%) | 121 (60.5%) | 89 (44.5%) | 106 (53.0%) |
| Legal terminal | 188 | 165 | 122 | 163 |
| Final protocol failure | 12 | 28 | 76 | 36 |
| Final wrong answer | 63 | 44 | 33 | 57 |
| Final execution failure | 0 | 6 | 1 | 0 |
| Final argument failure | 0 | 1 | 1 | 1 |
| Mean actions | 7.97 | 8.27 | 8.04 | 7.86 |
| Prompt tokens | 6,632,595 | 7,038,824 | 6,905,665 | 6,113,434 |

Version8 used 6,412,885 total tokens over 1,585 API request attempts, including 14 transport
retries. It completed in 1,089.939 seconds.

### Paired all-task comparisons

| Comparison | Both correct | Baseline only | Version8 only | Neither | v8 delta | Exact McNemar p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original v4 vs v8 | 93 | 32 | 13 | 62 | -19 | 0.006609 |
| Compact N-way v6 vs v8 | 93 | 28 | 13 | 66 | -15 | 0.027533 |
| Closed semantics v7 vs v8 | 66 | 23 | 40 | 71 | +17 | 0.042957 |

The version7-to-version8 gain is consistent with the prompt correction reducing carrier ambiguity.
It is not a clean causal estimate because the external requests were sent in different time
windows.

Among the 156 task ids where neither version4 nor version8 ended in a final protocol error, the
paired result was exactly tied: 93 both correct, 11 version4-only, 11 version8-only, and 41 neither.
This is a useful diagnostic, but it conditions on a post-treatment outcome and cannot replace the
primary all-task comparison.

## Raw Response Audit

Version8 recorded 176 protocol-error events, down from 326 in version7:

| Protocol event class | Version7 | Version8 |
| --- | ---: | ---: |
| Missing native reasoning | 190 | 130 |
| Reasoning/prose in visible content | 116 | 38 |
| Other strict-protocol violation | 20 | 8 |

The reduction from 116 to 38 visible-content errors is the clearest evidence that the old prompt was
ambiguous for this provider. Missing native reasoning remained the largest class, so the prompt fix
did not stabilize the provider's two-field carrier.

The 36 final protocol-failure records break down as:

- 27 missing `reasoning_content`;
- seven visible reasoning prefixes before a tool call;
- two empty/truncated visible responses with `finish_reason=length`.

Thirty-four of the 36 ended with API `finish_reason=stop`; 31/36 raw visible responses still
contained a complete tool-call JSON object. Eight of those complete rejected calls were already
terminal `answer_from_context` actions. Under the strict protocol they must remain rejected rather
than silently repaired, but their presence confirms that these failures are not ordinary join
argument errors or API transport outages.

Final protocol failures were also temporally clustered: the ten consecutive 20-result buckets
contained 1, 1, 3, 5, 12, 0, 1, 4, 5, and 4 failures. Version4's corresponding buckets stayed
between zero and three. This remaining nonstationarity is why a sequential A/B cannot isolate the
join treatment.

## What the Higher-Scoring Original Prompt Actually Did

The local artifact inventory contains no run with 127 correct answers out of these 200 tasks. The
original high-scoring version4 artifact has 125/200 correct; `127` is its number of attempted
`join_tables` calls. If a separate 127/200 artifact exists outside this repository, it needs to be
identified before its exact prompt can be attributed.

The actual API-facing version4 prompt was not free of the split-carrier problem. It contained:

1. a positive canonical rule requiring visible
   `<think>brief reasoning</think><tool_call>...</tool_call>`;
2. another data-generation rule requiring one non-empty visible `<think>` block;
3. a final DeepSeek clause saying that those earlier rules were overridden and that reasoning must
   instead go in `reasoning_content`;
4. rolling assistant history in which every prior legal action was rendered with visible
   `<think>...</think>` followed by `<tool_call>...</tool_call>`.

The first version4 system prompt contained seven `<think>` occurrences. Across its actual requests,
all 4,266 rolling assistant-history messages contained a visible `<think>` block. Version8's system
prompt contains no positive visible-`<think>` instruction, and all 4,107 assistant-history messages
are visible tool-call-only.

This explains the direction of the error shift:

- version4 strongly demonstrated that a reason must exist, so it had only 18 missing-native-reason
  events;
- the same demonstrations conflicted with field placement, producing 93 visible-reason-prefix
  events;
- version8 reduced visible-prefix events to 38, but missing-native-reason events rose to 130;
- version4 recovered after a protocol error in 77 episodes and terminated in 12;
- version8 recovered in 57 episodes and terminated in 36;
- both runs had almost the same number of episodes with at least one protocol event: 89 versus 93.

Therefore the original run did experience the issue. Its error mode was usually a recoverable
placement mistake, while version8 more often repeated the same missing-field mistake three times
and exhausted the per-type recovery budget.

Rolling history is not the full explanation. On the first request of an episode, before any
assistant history exists, version4 had two missing-native-reason errors and version8 had 18. The
runs occurred sequentially on the same day: the version4 manifest completed at 11:14, version6 at
12:34, version7 at 12:58, and version8 at 14:06 local time. The changing first-turn distribution
and time clustering leave provider/backend state as an uncontrolled factor.

The request payload also relied on API defaults: it sent `model`, `messages`, `temperature`, and
`max_tokens`, but did not explicitly send `thinking: {"type":"enabled"}` or
`reasoning_effort:"high"`. Current DeepSeek V4 documentation supports both thinking and
non-thinking modes and documents these explicit controls. Although thinking is documented as the
default, omitting the fields makes the experimental contract less auditable and leaves room for
default or routing changes. The harness also did not record the returned response model or
`system_fingerprint`, so the artifacts cannot prove that both runs used an identical backend
snapshot.

The right lesson is not to copy version4's contradictory prompt. A more controlled next treatment
is:

- keep version8's single provider-specific field contract;
- explicitly enable thinking in the API request and fix the reasoning effort;
- keep the exact two-field example;
- record request thinking settings, returned model, response id, and system fingerprint;
- interleave version4 and version8 task pairs in time.

If explicit thinking still produces empty `reasoning_content`, then test a separately versioned
content-only canonical envelope. Do not mix that fallback into the same treatment.

## Join and Downstream Semantics

| Metric | Original v4 | Compact N-way v6 | Closed semantics v7 | Prompt-fixed v8 |
| --- | ---: | ---: | ---: | ---: |
| Join attempts | 127 | 106 | 100 | 93 |
| Executed joins | 107 | 103 | 99 | 90 |
| Join-local success | 84.25% | 97.17% | 99.00% | 96.77% |
| Join-local errors | 20 | 3 | 1 | 3 |

All 93 version8 calls used the current canonical `base + joins[]` shape. Its three join errors were
the same model mistake: using a derived handle as though it were a logical column namespace. There
were no version8 `project` or `group_aggregate` execution errors; version6 had 23 and six
respectively before downstream dotted-column semantics were closed. The number of downstream
attempts differs, so the zero count is supporting evidence rather than a standalone rate proof.

Version8 attempted 88 binary joins, four 3-way joins, and one 6-way join. Strict binary
decomposition would require 101 join actions rather than 93, an 8-action or 8.6% increase in the
join portion of these trajectories. A bounded N-way atom therefore avoids unnecessary intermediate
handles without complicating the common binary case.

Compact namespaces appeared in 297 model turns and saved 32,579 rendered characters: 109.69
characters on average when present, p90 248, maximum 733.

## Reference-Chain Verification

All 106 verifier-correct version8 trajectories replayed to the same denotation with no replay
errors.

- 34 join steps were replayed;
- 34/34 had exactly one harness-authored data edge for every join input;
- 34/34 appeared in the final answer's backward dependency slice;
- 98/106 had complete action-literal grounding;
- 91/106 had complete final-value grounding;
- 86/106 had complete deterministic end-to-end grounding.

The join refactor therefore preserves the original reference chain. The remaining grounding gaps
are mainly scalar/value-grounding issues outside the join namespace design.

## Interpretation

The experiment supports four separate conclusions:

1. **The old DeepSeek prompt was ambiguous.** Removing visible-`<think>` conflicts substantially
   reduced visible-format failures and improved version7's raw score.
2. **The corrected prompt did not close the raw gap to version4.** On the primary all-task metric,
   version4 remained better in this sequential run.
3. **The simplified join is locally better and compositionally sound.** It has a much higher
   execution rate, shorter multiway trajectories than binary-only decomposition, closed downstream
   identifiers, compact state, and intact provenance.
4. **Provider carrier state remains a confounder.** Missing native reasoning and time-clustered
   format failures dominate the residual difference. Static configuration equality alone does not
   control this variable.

Version6's 121/200 is the strongest raw result among the simplified-interface runs, but it cannot be
selected as the final design because it has known downstream semantic defects. Version8 is the
most complete tool contract; its next evaluation should change experimental scheduling, not revert
the join semantics.

## Required Next Validation

For a causal interface comparison:

1. submit version4 and version8 as randomized paired requests for each frozen task;
2. interleave both arms within small time blocks and record dispatch/completion order;
3. explicitly set `thinking: {"type":"enabled"}` and `reasoning_effort:"high"` in both arms;
4. record response model, response id, and system fingerprint when the provider returns them;
5. keep one attempt, decoding, action budget, rolling history, and strict parser identical;
6. report transport retries, split-carrier failures, semantic policy failures, join-local execution,
   and strict-multiset denotation separately;
7. do not condition the primary score on successful carrier formatting.

If split-carrier failures remain high under interleaving, evaluate a separately versioned
content-only canonical response envelope. Do not silently accept complete tool JSON with missing
native reasoning inside version8, because that would change the protocol and confound the result
again.

## Artifacts

- Version8 all attempts:
  `data/trajectories/bird_train_version8_join_validation200_flash_rolling_success.all.jsonl`
- Version8 verified trajectories:
  `data/trajectories/bird_train_version8_join_validation200_flash_rolling_success.jsonl`
- Version8 manifest:
  `data/trajectories/bird_train_version8_join_validation200_flash_rolling_success.manifest.json`
- Version4 versus version8 controlled audit:
  `data/trajectories/bird_train_version4_vs_version8_join_controlled200_audit.json`
- Version6 versus version8 controlled audit:
  `data/trajectories/bird_train_version6_vs_version8_join_controlled200_audit.json`
- Version7 versus version8 prompt audit:
  `data/trajectories/bird_train_version7_vs_version8_prompt_controlled200_audit.json`
- Version8 reference audit:
  `data/trajectories/bird_train_version8_join_validation200_flash_rolling_reference_audit.json`
