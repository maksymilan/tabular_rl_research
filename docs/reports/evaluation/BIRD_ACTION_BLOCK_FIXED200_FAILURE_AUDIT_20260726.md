# BIRD action-block fixed-200 failure audit (2026-07-26)

## Decision

The action-block v10 fixed-200 result has **57 failures** under `bird-set`: 55 wrong answers,
one provider/API failure, and one exhausted atomic-action budget. Forty-four of the 55 wrong
answers are error-free legal trajectories. The main remaining limitation is therefore semantic
policy behavior, not interface exceptions.

Do not add a stronger generic checklist to the runtime prompt. A frozen 24-task gate compared the
unchanged v11 prompt with a concise, task-independent v12 constraint-first prompt. V12 recovered
zero tasks, regressed one control, doubled process errors, and used 12.6% more tokens. The
experiment is rejected and the active prompt remains byte-identical to action-v4.

Keep the v11 safe resolver as the confirmed tool-interface improvement. It rejects a column
reference in a literal-value position and again solved `02179`. Do not add a new relational atom:
every semantic failure in this audit is expressible with the current tools, and the observed
mistakes do not identify one missing operation with a credible cross-task gain.

## Audit method

The audit input is
`data/trajectories/batch_plan_20260726/action_block_v10_low_friction_fixed200_r1.all.jsonl`.
`src/eval/audit_action_block_failures.py` produces a compact 57-record view that preserves:

- the question, external knowledge, offline gold SQL, prediction, and gold denotation;
- every authored action and the environment's resolved action;
- output handles, columns, row counts, bounded row samples, errors, and interface resolutions;
- the paired version24 outcome.

Repeated resident-state snapshots are removed only from the compact audit view. Gold SQL is used
offline for diagnosis and is never rendered to the actor or used to alter execution.

Primary-cause counts are deliberately exclusive:

| Primary cause | Failures | Interpretation |
|---|---:|---|
| Benchmark/question/external-knowledge ambiguity or conflict | 20 | A score-seeking rule would encode a local gold convention rather than a general relational invariant. |
| Semantic mapping or operator choice | 13 | Wrong table/column/aggregate/formula despite sufficient tool expressivity. |
| Population, grain, join, or multiplicity | 12 | Correct-looking operations over the wrong entity population or row grain. |
| Answer slots or output shape | 9 | Correct or near-correct evidence, but missing/extra/concatenated/mislaid output fields. |
| Interface or exhausted exploration budget | 2 | One unsafe resolver behavior and one repeated empty-join search. |
| Provider/API | 1 | Repeated external disconnect after successful semantic actions. |
| **Total** | **57** | |

## Per-trajectory findings

The “safe response” column distinguishes a general intervention from a benchmark patch. “Existing
contract” means the current prompt/tool set already expresses the right behavior; the failure is
useful supervision/evaluation evidence, not a reason to add more prose.

| ID | Primary cause | Trajectory evidence and diagnosis | Safe response |
|---|---|---|---|
| `00004` | Semantic/operator | The model attempted a ten-year timestamp expression and counted 2,252; the gold uses a strict difference of year substrings and returns 312. The authored expression also lacks parentheses around year addition before concatenation. | Existing `project` expressions suffice; do not encode one date convention globally. |
| `00040` | Benchmark ambiguity | The question asks to list titles and the model returns the four correct titles; gold also returns `qty`, which is described as a predicate but not clearly an answer slot. | No prompt patch for implicit gold columns. |
| `00041` | Output slots | The reasoning and observed row contain store `7131` and the least-quantity title, but terminal evidence selects only the title. | Existing exact-answer-slot check; suitable causal training example. |
| `00214` | Benchmark ambiguity | The model finds the single maximum-popularity movie and all its keywords; gold orders joined keyword rows and arbitrarily limits to one keyword despite plural wording. | Do not add arbitrary post-join limiting. |
| `00362` | Benchmark ambiguity | Two eligible professors tie at teaching ability 5; the model returns both, while gold takes one arbitrary row. | Do not invent a tie-breaker absent from the question. |
| `00582` | Output shape | The correct five patients and ages are found, but first and last names are concatenated into one column; external knowledge and gold keep them separate. | Existing no-unrequested-concatenation rule. |
| `00593` | Population/grain | Two matching medication intervals are observed (11 and 18 days), but the model selects the first before computing a scalar. | Preserve all matching rows; row-wise calculation is already expressible through `project`. |
| `00600` | Output layout | Correct counts are produced as category rows `F=193`, `M=180`; gold expects one ordered row `(male, female)`. | Existing `group_aggregate(output_layout="columns")`. |
| `00715` | Benchmark conflict | External knowledge defines address as `street_num, street_name`; the model returns both, while gold returns only `street_name`. | Follow the declared contract, not the contradictory gold shape. |
| `00796` | Semantic mapping | The model aggregates `Match.Man_of_the_Match` directly rather than following the dataset's Season-to-Match path for “Man of the Series”. | Schema-grounded causal supervision; no new tool. |
| `00886` | Semantic mapping | Austria is found correctly, but the model uses `Indicators -> Series`; gold uses `CountryNotes -> Series`. | Stronger relation-selection training, not task-specific prose. |
| `01152` | Benchmark conflict | The correct youngest player is found, but the model returns first/last; gold includes middle name, while external knowledge says “player name” refers to `playerID`. | Conflicting output conventions; no global name-component rule. |
| `01167` | Benchmark conflict | The model joins award winners to their actual coaching rows and finds no POR coach; gold joins awards to teams only by year and returns five. | Treat as gold/query semantic conflict. |
| `01213` | Benchmark ambiguity | The model finds the most recent work and all 47 characters; gold applies `LIMIT 1` after the joins and returns one arbitrary character. | Do not add unrequested limiting. |
| `01461` | Semantic mapping | It filters `Business.stars` instead of joining `Reviews.review_stars`; the first sample happens to match but the full denotation does not. | Exact noun-to-column training; tools already support the join. |
| `01530` | Population/grain | The model divides matching businesses by all `Business` rows; gold/external calculation uses attribute-assignment rows as its denominator. | Preserve explicitly declared denominator grain; no new atom. |
| `01560` | Semantic/operator | It computes mean likes per tip length; gold asks for total likes per length. | Existing `group_aggregate(sum)`; train aggregate selection. |
| `01569` | Population/grain | It first selects businesses open the maximum number of days, then emits their categories; gold groups working-day rows by category and ranks categories. | Train entity-grain selection before aggregation. |
| `01589` | Benchmark conflict | External knowledge says `attribute_value='true'`, the model ignores value, and gold instead accepts `none/no/false`. | Do not strengthen contradictory external knowledge for score seeking. |
| `01888` | Benchmark ambiguity | The model returns the two requested counts; gold interprets “compare” as one subtraction and also draws 2005 from a different table. | No global “compare means subtract” patch. |
| `02052` | Provider | A full fresh retry again disconnects after 11 successful semantic actions. | Report as provider/API failure; no policy relabeling. |
| `02078` | Population/grain | It averages all zip coordinates in the district into one centroid; gold returns every zip's latitude/longitude. | Existing preserve-row-grain rule; no aggregate was requested. |
| `02088` | Population/grain | It counts distinct city names; external/gold count rows of each post-office type. | Follow declared count grain; no tool change. |
| `02179` | Interface | V10 converted `$filter_geo.LocationID` in a literal `value` field into the string `"LocationID"`, silently yielding zero rows. | **Fixed by v11**: reject column-as-literal and force grounded value/replan. |
| `02189` | Population/grain | A global sum of all qualifying sales is cross-joined onto every brand; gold requires one sum per brand. | Existing grouped aggregation; preserve group key. |
| `02418` | Semantic/operator | It uses `count_distinct(order_id)`; external knowledge/gold use `count(order_id)`. | Keep explicit count operator exact. |
| `02438` | Output/semantics | After recovering two namespace errors, it selects the latest status per order and outputs order IDs; gold asks for distinct status values over matching histories. | Existing slot and population checks; v11 removes friction but not this semantic error. |
| `02507` | Output shape | The correct employee is found, but first and last names are concatenated. | Existing separate answer-slot rule; v11 control happened to recover it. |
| `02513` | Population/grain | It counts passing inspection rows (211) rather than distinct businesses/licenses (203). | Entity-grain supervision; `count_distinct` already exists. |
| `02868` | Output shape | It emits first, middle, last; gold requests first and last only. | Answer-slot supervision; name conventions are inconsistent across tasks. |
| `02925` | Benchmark ambiguity | It returns the readable product name; gold returns `ProductID`, while the question only says “product”. | Do not globally prefer IDs or names from this case. |
| `03042` | Output shape | It computes the correct goalie, then concatenates first/last into one field. | Existing no-concatenation rule. |
| `03131` | Output shape | It correctly includes both countries independent in 1830 but adds country name to each language row; gold selects only language and official flag. | Omit helper/disambiguation field at terminal. |
| `03373` | Benchmark ambiguity | It finds the correct 19 desert names and separately counts 19, but the question asks for both while gold contains only names; terminal can cite only one rectangular result. | Do not contort the tool around a contradictory gold output. |
| `03664` | Benchmark conflict | The model returns the maximum `rental_rate/rental_duration`, exactly matching external knowledge; gold orders by that ratio but selects raw `rental_rate`. | Treat as output-definition conflict. |
| `03688` | Benchmark ambiguity | Ten films tie for maximum length and yield 46 inventory rows; gold takes the first joined row. | Do not add arbitrary tie truncation. |
| `04038` | Benchmark conflict | It counts distinct patients per gender (349/337); gold counts condition rows (553/572) despite asking for patients. | No global distinctness reversal. |
| `04189` | Population/join | A left join retains free sports apps without reviews; gold uses an inner join. A namespace error is recovered but is not the final cause. | Train requested-pair population; no new join tool. |
| `04244` | Population/grain | It independently totals points by `tmID` across years and intersects with teams that ever had an MVP; gold builds one joined player/team/award population and groups team names. | Establish the joined population before aggregation. |
| `04426` | Output shape | The five correct episode IDs are found, but season, episode number, title, and air date are also emitted. | Terminal should select only the requested episode identifier. |
| `04822` | Benchmark conflict | The model follows the question/external value `TEAM='Avangard Omsk'` and gets zero; gold filters `TEAM='Czech Republic (all)'`. | Do not encode the hidden conflicting value. |
| `04848` | Benchmark conflict | The model emits employee ID, job title, and sick-leave hours; external knowledge names ID/title, but gold contains title only. | No task-specific output convention. |
| `04869` | Population/grain | It restricts to current assignments and distinct employees; external/gold count all joined shift-history rows. | Preserve declared population; no new tool. |
| `04906` | Semantic/operator | It counts distinct images; external knowledge/gold divide object-sample occurrences for broccoli and tomato. | Follow the explicit counted column. |
| `05053` | Benchmark ambiguity | It returns human-readable `Laos`; gold returns country code `LAO`. | Do not globally substitute code for nation name. |
| `05147` | Population/grain | It deduplicates businesses before numerator and denominator; gold/external formula counts joined violation rows. | Preserve formula grain; no new atom. |
| `05161` | Semantic mapping | External knowledge says cheese is an ingredient category; the model broadens the filter to category or ingredient name, producing 108 rather than the gold population. | Keep explicit categorical predicate exact. |
| `05544` | Population/grain | It counts six representative-term rows instead of three distinct representatives. | Count the requested entity, not historical rows. |
| `05632` | Benchmark conflict | The model computes debt among Japanese suppliers (10.44%), exactly matching external knowledge; gold computes Japanese share among all indebted suppliers (4.71%). | Do not strengthen the contradictory benchmark formula. |
| `05668` | Semantic/output | It sums line differences per order, returns vendor name first, and gets 290; external/gold use the maximum single detail-line difference and output `(difference, VendorID)` = `(280,1520)`. | Follow explicit operator and output fields/order. |
| `06026` | Benchmark ambiguity | The product exists in West and South with different profits; the model combines both regions, while gold silently selects `south_superstore`. | No prompt rule can infer the hidden region safely. |
| `06165` | Semantic mapping | The correct `PaperAuthor.Name` rows are observed, then an unnecessary join to sparse `Author` collapses seven rows to one and alters the name. | Prefer the already grounded direct answer field; no new join capability. |
| `06246` | Semantic/literal | External knowledge defines Delaware as a county; the model switches to state `DE` and adds `type='Post Office'` instead of inspecting the county literal (`DELAWARE`). | Existing inspect-before-uncertain-literal behavior. |
| `06299` | Interface/budget | Repeated joins between filtered `MenuPage.menu_id` and `Menu.id` return empty; the model retries variants until 20 actions. The gold/question also disagree about whether the two totals are subtracted. | Budget stop is correct; improve causal examples before adding a special join atom. |
| `06324` | Semantic mapping | It treats the single `langs` row as the corpus and returns language code `ca`; gold ranks `pages.words` and returns `pages.title`. | Broader schema-to-question grounding; tools suffice. |
| `06489` | Semantic mapping | External knowledge explicitly maps “object” to `OBJ_SAMPLE_ID`; the model unnecessarily joins object classes and returns `"paper"`. | Follow explicit output-column mapping. |
| `06492` | Benchmark conflict | The model groups by image and counts images with fewer than 15 samples, matching the question and external knowledge; gold instead counts rows where `OBJ_SAMPLE_ID < 15`. | Do not train the contradictory row-level shortcut. |

## Prompt gate

The rejected v12 prompt added 196 characters to v4. It contained no task names, database names,
column names, values, or BIRD-specific language. It replaced the existing final check with three
general operations:

1. preserve explicit external-knowledge operators and entity grain;
2. enumerate answer slots and remove helper columns;
3. avoid unrequested aggregate/deduplicate/limit/tie/concatenation changes.

The frozen 24-task set was declared before either run:

- 13 failures plausibly addressable by those invariants;
- four benchmark/external-knowledge conflict guards;
- `02179` for the v11 literal-reference safety fix;
- six previously correct controls.

| Metric | v11 unchanged prompt | v12 constraint-first | Change |
|---|---:|---:|---:|
| Correct | **6/24** | 5/24 | -1 |
| Paired gains / regressions | — | 0 / 1 | regression `05440` |
| Legal termination | 24/24 | 24/24 | 0 |
| Process errors | **4** | 8 | +4 |
| Blocked descendants | **2** | 11 | +9 |
| Model turns | **114** | 119 | +4.4% |
| Atomic actions | **213** | 237 | +11.3% |
| Total tokens | **575,765** | 648,244 | +12.6% |

V12 recovered none of the 13 declared target failures. `02179` remained correct in both runs,
which attributes that recovery to v11's environment-side safety boundary rather than prompt prose.
The prompt ablation is rejected and should not be expanded to fixed-200.

## Tool-design conclusions

One confirmed design defect exists: v10 allowed a column-shaped local reference to become a
literal string. V11's typed rejection is the right boundary because it exposes a recoverable error
without guessing a cell value. It solved `02179` in the original targeted test, the v11 gate, and
the v12 gate.

The remaining failures do not justify a new atomic tool:

- row-wise arithmetic is already available through `project` expressions;
- category-to-column layout is already available in `group_aggregate`;
- distinct entity counts, grouped metrics, conditional aggregates, connected joins, ranking, and
  exact terminal projection are all expressible;
- most wrong trajectories are legal and error-free, so another interface normalization would not
  address their cause.

The next credible intervention is causal training data that contrasts entity grain, aggregate
choice, and terminal answer slots while preserving benchmark-conflict cases as audit exclusions.
It should not be more runtime prompt prose, hidden-gold trajectory compilation, or a patch library
of task-specific mappings.

## Artifacts

- `data/trajectories/batch_plan_20260726/action_block_v10_fixed200_failures.audit.jsonl`
- `data/trajectories/batch_plan_20260726/action_block_v11_failure_audit_gate24_control.all.jsonl`
- `data/trajectories/batch_plan_20260726/action_block_v12_constraint_first_gate24.all.jsonl`
- `data/trajectories/batch_plan_20260726/action_block_v12_vs_v11_failure_audit_gate24.paired.json`
- `src/eval/audit_action_block_failures.py`
