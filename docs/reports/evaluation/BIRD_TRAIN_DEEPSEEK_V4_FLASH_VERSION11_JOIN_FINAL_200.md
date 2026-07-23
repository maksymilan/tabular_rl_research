# DeepSeek V4 Flash Version11 Join and Carrier Validation on 200 Frozen BIRD-Train Tasks

Date: 2026-07-23

## Decision

Adopt **version11** as the current model-visible design:

- one connected join component is one bounded N-way `join_tables(base, joins[], base_role?)`
  action;
- every `joins[]` item introduces one relation and one edge-local `on` list;
- roles are omitted normally and used only for repeated relations such as employee/manager;
- output columns keep a flat, non-recursive `relation.column` namespace;
- derived handles are table arguments, never replacement column namespaces;
- wide state renders the same logical columns compactly as `column_namespaces`;
- projection stays a separate atom, and all downstream relational tools consume the same logical
  dotted columns;
- DeepSeek receives native reasoning plus one raw JSON action object, with no competing visible
  XML instruction.

This design reached **130/200 = 65.0%**, exceeding the requested 125/200 target and the original
version4 result of 125/200. The paired version4 comparison is +5 tasks, but is not statistically
significant (`p = 0.404873`, exact McNemar), so the correct claim is that version11 meets or exceeds
the baseline in this run—not that it has established a significant policy improvement.

Version11 is preferred because it combines the strongest observed end-to-end result with a much
simpler join call, bounded multiway horizon, closed downstream column semantics, and verified
reference chains. The original prefix-array design is not justified by its 125/200 score because
its join-local execution rate was substantially worse and it coupled join edges, aliases, and
projection in one call.

## The Final Prompt Fix

Version10 explicitly enabled DeepSeek thinking, fixed reasoning effort, constrained visible output
with JSON Output, and stored provider identity metadata. During the version10 audit, one remaining
contradiction was found in the actual API-facing system prompt:

- the generic data-generation clause still requested a visible `<tool_call>` block;
- the final DeepSeek clause requested a raw JSON object with only `tool` and `arguments`.

Version11 replaces the generic clause with provider-neutral “one action using the provider-specific
envelope” wording and changes retry feedback from “complete tool_call block” to “complete JSON
action object.” Its full rolling DeepSeek prompt has:

- protocol hash `0c0feb3f0536f5fc`;
- exactly one `DEEPSEEK SPLIT-RESPONSE JSON OUTPUT CONTRACT`;
- zero literal `<tool_call>` tags;
- one concrete raw-JSON action example;
- explicit `thinking={"type":"enabled"}`, `reasoning_effort="high"`, and
  `response_format={"type":"json_object"}` request controls.

The fixed first-20 preflight produced 14/20 correct, 20/20 legal terminals, and zero protocol-error
events. The same task ids yielded 13/20 under both original version4 and version10, so version11
passed the small-sample gate before the remaining 180 requests were issued.

## Controlled Setup

The formal comparison uses the frozen input
`data/eval_inputs/bird_train_tool_interface_validation200_version4.jsonl` and:

- `deepseek-v4-flash`;
- BIRD train, 200 tasks, one attempt per task;
- `rolling-legal-history` with four history turns and the full rolling prompt;
- 30 maximum semantic actions and three recoverable errors per type;
- 2,048 completion tokens and no automatic table rows;
- eight workers, strict no-repair parsing, and three API transport attempts;
- strict-multiset terminal denotation.

The first 20 version11 records were copied from the successful preflight and the output was resumed
for the remaining 180. The final artifact has 200 unique trajectory ids and zero duplicate attempt
records. The manifest's `elapsed_seconds=1094.81` covers only the resumed 180-task phase; token and
request totals are recomputed from all 200 persisted records.

Version10 versus version11 has zero mismatches across the audit's recorded control fields. The
version4/version6/version7/version8 comparisons differ in provider request options because those
runs relied on API defaults and did not use JSON Output. All formal 200-task runs use rolling
history; the earlier version5 state-only hard-50 pilot remains descriptive and is not mixed into
the controlled full-cohort table.

Requests were sequential rather than randomized and interleaved, so provider time remains an
uncontrolled variable in every cross-run accuracy comparison.

## Whole-Episode Results

| Metric | Original v4 | Compact N-way v6 | Closed semantics v7 | Split-prompt v8 | JSON v10 | Final v11 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Correct / 200 | 125 | 121 | 89 | 106 | 121 | **130** |
| Accuracy | 62.5% | 60.5% | 44.5% | 53.0% | 60.5% | **65.0%** |
| Legal terminal | 188 | 165 | 122 | 163 | 195 | **196** |
| Final protocol failure | 12 | 28 | 76 | 36 | 3 | **3** |
| Final wrong answer | 63 | 44 | 33 | 57 | 74 | **66** |
| Final execution / argument / max-step | 0 / 0 / 0 | 6 / 1 / 0 | 1 / 1 / 0 | 0 / 1 / 0 | 0 / 1 / 1 | 1 / 0 / 0 |
| Mean actions | 7.97 | 8.27 | 8.04 | 7.86 | 8.07 | **7.52** |

Version11 used 5,976,028 prompt tokens and 425,267 completion tokens, including 362,738 reasoning
tokens: 6,401,295 total tokens over 1,526 API request attempts. API transport retries were 22 and
are not counted as semantic actions or policy errors.

All 1,504 recorded semantic responses reported model `deepseek-v4-flash`. A provider fingerprint
was present on 898 responses and absent on 606, so the artifacts still cannot prove one backend
snapshot across all sequential runs.

### Paired all-task comparisons

| Baseline vs v11 | Both correct | Baseline only | v11 only | Neither | v11 delta | Exact McNemar p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original v4 | 116 | 9 | 14 | 61 | **+5** | 0.404873 |
| Compact N-way v6 | 108 | 13 | 22 | 57 | **+9** | 0.175465 |
| Closed semantics v7 | 80 | 9 | 50 | 61 | **+41** | <0.000001 |
| Split-prompt v8 | 98 | 8 | 32 | 62 | **+24** | 0.000182 |
| JSON v10 | 114 | 7 | 16 | 63 | **+9** | 0.093140 |

The v10-to-v11 difference is the cleanest recorded prompt comparison because its provider request
options and all other control fields match. Its +9 result is suggestive but does not cross the
conventional 0.05 threshold. Version7 and version8 have much larger raw gaps, but their provider
carrier distributions and request controls differ; those rows must not be read as isolated join
effects.

The repository has no 127/200 original artifact. The original version4 result is 125 correct; 127
is its number of attempted join calls.

## Provider-Carrier Audit

Version11 recorded 18 recoverable provider/protocol events:

| Event class | v10 | v11 |
| --- | ---: | ---: |
| Empty visible content | 12 | 12 |
| Visible content not valid JSON | 2 | 3 |
| JSON with the wrong top-level shape | 2 | 3 |
| Missing native reasoning | 1 | **0** |
| Total | 17 | 18 |
| Final protocol failures | 3 | 3 |

The three version11 final protocol failures were `bird_train_06246`, `bird_train_06165`, and
`bird_train_06299`. Two ended with length-truncated empty visible content; one exhausted its shared
protocol budget after returning JSON with an extra top-level `reason` field. Across all responses,
1,493 ended with `finish_reason=stop` and 11 with `finish_reason=length`.

Removing the last prompt contradiction did not reduce the aggregate event count relative to
version10. Its value is contractual: the request now has one unambiguous return format, so the
remaining empty/truncated/extra-key responses can be classified as provider/model behavior rather
than prompt conflict. The +9 accuracy change should not be attributed entirely to this wording.

## Join-Local and Horizon Results

| Metric | Original v4 | Compact N-way v6 | Closed v7 | v8 | v10 | v11 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Join attempts | 127 | 106 | 100 | 93 | 119 | **100** |
| Executed joins | 107 | 103 | 99 | 90 | 110 | **94** |
| Join-local success | 84.25% | 97.17% | 99.00% | 96.77% | 92.44% | **94.00%** |
| Join-local errors | 20 | 3 | 1 | 3 | 9 | **6** |

All 100 version11 attempts used the canonical `base + joins[]` shape, and all 94 successful outputs
used flat namespaces. Version11 attempted 87 binary, nine 3-way, two 4-way, and two 5-way joins.
Strict binary decomposition would require 119 join actions instead of 100: 19 extra join actions,
or a 19% increase in the join portion of these trajectories.

All six version11 join errors were the same residual model mistake: prefixing a logical column with
a derived handle that is not an introduced namespace. This is real ergonomic pressure, but it does
not justify accepting arbitrary `handle.column` aliases: a derived handle can contain multiple
source namespaces with the same suffix, making automatic repair ambiguous. The safe next treatment
is a targeted negative/positive call example or more explicit structured state rendering, not
silent alias stripping.

Version6's higher 97.17% join-local rate is not enough to select it. Its incomplete downstream
semantics produced 23 `project` and six `group_aggregate` execution errors. Version11 keeps the
closed dotted-column semantics and had two project execution errors; neither was a recurrence of
the old systematic downstream dotted-column interpretation gap.

## Namespace Pressure

Version11's successful joins had median/p90/maximum widths of 13/32/57 columns and
median/p90/maximum column-name characters of 221/590/1,328.

Compact `column_namespaces` appeared in 308 model turns and saved 49,208 rendered characters:
159.77 characters on average when used, p90 466, and maximum 948. This preserves the canonical
flat names for replay while reducing model-visible repetition.

## Reference-Chain and Replay Verification

An initial replay exposed one general recovery bug. `bird_train_05476` had a rejected project at
`step_6`; the online executor had already consumed the hidden handle suffix, so the later legal
table was `project_006`. Legal-only replay allocated `project_005`, while the recorded terminal call
still referenced `project_006`.

The replay layer now maintains a deterministic recorded-handle to replay-handle mapping. It remaps
exact table references and derived `handle.column` namespaces only during replay; model-authored
calls and canonical stored trajectories remain unchanged. A regression test covers this recovered
error/gapped-handle case.

After the fix:

- all **130/130** verifier-correct trajectories replay to the same strict-multiset denotation;
- replay has zero exceptions and zero replay-incorrect ids;
- **45/45** successful join steps have exactly one harness-authored data edge per input relation;
- **45/45** join steps appear in the final answer's backward dependency slice;
- 115/130 have complete action-literal grounding;
- 101/130 have complete final-value grounding;
- 92/130 have complete deterministic end-to-end grounding.

The join reference chain is therefore intact. The remaining grounding gaps are chiefly
scalar/value-grounding limitations outside the join namespace contract and still block a claim of
global process-credit completeness.

## Design Ranking

| Design | Call intuitiveness | Multiway horizon | Composition | Decision |
| --- | --- | --- | --- | --- |
| Original prefix arrays + join-local projection | Low; edges, aliases, prefixes, and selection are coupled | Short | Fragile identifiers; 84.25% local success | Retire for new calls; replay only |
| Strict binary join | High per call | Longest; creates intermediate handles | More reference reuse and error opportunities | Reject as the only public join |
| `base + joins[]` bounded N-way | High; mirrors a connected join path | Bounded | One explicit new relation per edge | Keep as the join atom |
| N-way + compact namespaces | Same call, lower context pressure | Bounded | Needs consistent downstream logical names | Keep compression |
| Closed semantics + unambiguous JSON carrier (v11) | Highest complete contract | Bounded | Join, downstream tools, history, and replay agree | **Recommended** |

“Atomic” should mean one semantically coherent connected join component, not necessarily exactly
two physical inputs. The list elements remain simple edge atoms, while a common binary case still
uses a one-element `joins[]` list.

## Remaining Work

1. For a causal accuracy claim, run randomized interleaved version4/version11 task pairs in the
   same provider time blocks; the current sequential +5 result is not significant.
2. Test a focused state/example treatment for the six invalid derived-handle namespace calls.
3. Close scalar arithmetic/final-value grounding before enabling process-RL optimization.
4. Consider making executor failures transactionally restore hidden view/counter state. The replay
   mapping makes existing artifacts deterministic, but it does not change the online hidden counter
   behavior.
5. Continue reporting API transport retries separately from semantic policy failures.

## Verification

After the version11 prompt and replay fixes:

- active harness checks: 57/57;
- SFT unit tests: 63/63;
- RL unit tests: 32/32;
- evaluation unit tests: 17/17.

## Artifacts

- Version11 all attempts:
  `data/trajectories/bird_train_version11_join_validation200_flash_rolling_json_unambiguous_success.all.jsonl`
- Version11 verified trajectories:
  `data/trajectories/bird_train_version11_join_validation200_flash_rolling_json_unambiguous_success.jsonl`
- Version11 manifest:
  `data/trajectories/bird_train_version11_join_validation200_flash_rolling_json_unambiguous_success.manifest.json`
- Version11 replay/reference audit:
  `data/trajectories/bird_train_version11_join_validation200_flash_rolling_json_unambiguous_reference_audit.json`
- Version4/version11 paired audit:
  `data/trajectories/bird_train_version4_vs_version11_join_controlled200_audit.json`
- Version10/version11 paired audit:
  `data/trajectories/bird_train_version10_vs_version11_join_controlled200_audit.json`
- Version6/version7/version8 comparison audits:
  `data/trajectories/bird_train_version6_vs_version11_join_controlled200_audit.json`,
  `data/trajectories/bird_train_version7_vs_version11_join_controlled200_audit.json`, and
  `data/trajectories/bird_train_version8_vs_version11_join_controlled200_audit.json`.
