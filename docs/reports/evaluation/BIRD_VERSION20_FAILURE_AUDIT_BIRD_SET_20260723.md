# BIRD version20 fixed-200 failure audit under baseline-aligned `bird-set`

Date: 2026-07-23

## Decision

The archived version20 fixed-200 artifact was originally scored with the implicit
`strict-multiset` default and reported **138/200 = 69.0%**. Deterministic replay reproduces all
200 stored outcomes exactly.

The current direct-SQL BIRD baseline uses `bird-set`, not `strict-multiset`. Replaying the same
tool trajectories under `bird-set` raises the result to **141/200**. Five former failures differ
from gold only in duplicate-row multiplicity:

`05952, 04598, 04163, 03262, 00128`.

Two otherwise correct percentage trajectories (`01787`, `00165`) initially became false under raw
`bird-set` equality because `scalar_compute(percent)` evaluated `(part / whole) * 100`, while the
gold SQLite expressions evaluate `(part * 100) / whole`. The two orders differ by one ULP. After
making the tool use SQLite's operation order, deterministic replay is **143/200 = 71.5%**.

Therefore the baseline-aligned current error set is **57 tasks**, not the 62 failures in the
archived strict-multiset manifest.

## Classification

| Primary class | Count | Meaning |
| --- | ---: | --- |
| `C-SHAPE` | 12 | The model found the right rows/value but produced the wrong final columns, layout, representation, or casing. |
| `C-REL` | 8 | The model understood the target but chose the wrong join type, grain, aggregation placement, or operator order. |
| `M-SEM` | 17 | The model's selected entity, relation, denominator, population, or statistic was wrong. |
| `G-GOLD` | 20 | The question/external knowledge and gold SQL conflict or under-specify output slots, multiplicity, tie-breaking, or population. |
| **Total** | **57** | |

The clean “model knows what to do but the tool trajectory lands incorrectly” set is
`C-SHAPE + C-REL = 20/57`. Correcting any 7 of these would move the aligned score from 143/200 to
the requested 150/200 = 75%.

Only 9/57 failures contain any rejected tool action (10 error events total); 48/57 are fully legal
but semantically wrong terminal trajectories. Protocol permissiveness is therefore not the main
remaining bottleneck.

## Model-capability failures

### Final evidence shape or representation (`C-SHAPE`, 12)

| ID | Current trajectory versus gold |
| --- | --- |
| `01152` | Correct youngest winner Kyrie Irving; projects first/last while gold requires first/middle/last. |
| `02901` | Correct filters, join, age order, and top-10; concatenates three name slots into one string. |
| `06454` | Correct comparison result `RAIL`; rewrites the stored value as lowercase `rail`. |
| `04848` | Correct top-3 job titles; terminal evidence retains `BusinessEntityID`. |
| `04426` | Correct Oscar Cervantes credited rows; returns episode titles instead of `episode_id`. |
| `00040` | Correct CA, quantity filter, and joins; omits the requested `qty` output column. |
| `02868` | Correct Distinguish cardholder rows; adds middle name while gold requires first/last. |
| `00582` | Correct patients, year, blood pressure, and ages; concatenates first/last into one column. |
| `03131` | Correct 1830 countries and languages; retains country and does not collapse to the gold language/official grain. |
| `02512` | Correct employee Ruth Noble; concatenates first/last into one column. |
| `02507` | Correct employee David Hodges; concatenates first/last into one column. |
| `04038` | Correct female/male counts 553 and 572; returns two category rows instead of one two-column row. |

### Relational/tool landing errors (`C-REL`, 8)

| ID | Current trajectory versus gold |
| --- | --- |
| `04189` | Correct Free/Sports filters and columns; uses a left join, adding apps without review rows, instead of gold inner join. |
| `00593` | Correct patient, drug, reason, and both courses are observed; later fixes on one encounter and returns only 11 days, dropping the 18-day row. |
| `05440` | Understands “latest paper with journal,” but computes maxima/ties before the exact inner-join population and chooses a different tied paper. |
| `00004` | Understands “updated more than ten years later,” but uses elapsed days `>3652`; gold compares integer year substrings. |
| `02418` | Counts orders by `book_id`; gold groups by title, so same-title IDs must merge before ranking. |
| `06165` | Correct Journal→Paper→PaperAuthor path; unnecessarily joins Author and collapses seven author rows to one. |
| `05544` | Correct state/type/gender population; counts six term rows instead of three distinct representatives. |
| `02189` | Correct six root-beer brands and global amount; aggregates once globally instead of grouping amount by `BrandName`. |

## Model semantic failures (`M-SEM`, 17)

| ID | Current trajectory versus gold |
| --- | --- |
| `06489` | Finds the correct image-object row but returns object class `paper` instead of `OBJ_SAMPLE_ID=18`. |
| `06324` | Ranks `langs.words` and returns language `ca`; gold ranks `pages.words` and returns `pages.title`. |
| `01692` | Uses Award-row denominator and distinct-person numerator, producing 33.33%; gold counts the joined nominee grain, producing 66.10%. |
| `00886` | Uses Austria's Indicators path; gold requires CountryNotes→Series topics. |
| `01167` | Joins award winners to coaches by coach/year; gold joins to teams by year and filters `POR`. |
| `01530` | Numerator uses the target attribute association, but denominator is all Business rows rather than joined Business_Attributes rows. |
| `04869` | Counts distinct current night-shift employees over current Employee count; gold counts all EmployeeDepartmentHistory shift rows. |
| `01461` | Filters `Business.stars`; gold filters `Reviews.review_stars` and groups business IDs. |
| `02078` | Finds the correct representative and ZIP rows, then averages all coordinates instead of returning every latitude/longitude row. |
| `01425` | Treats `grad_100` as a Boolean flag and counts rows; gold sums the numeric `grad_100` values. |
| `02088` | Counts distinct city names; external knowledge/gold count joined ZIP rows by type. |
| `00074` | Compares per-title 1994 totals to their mean; gold compares each 1994 sale row to the all-period global sale-quantity average. |
| `04906` | Counts distinct images; external knowledge/gold count object-sample rows. |
| `01569` | Ranks individual businesses by days and lists their categories; gold aggregates total day rows by category name. |
| `01888` | Uses historical terms for both years and returns two counts; gold subtracts historical-1875 from current-2005. |
| `00041` | Chooses the store with the largest single sale row; gold ranks stores by total quantity, then titles within that store by total quantity. |
| `01560` | Computes mean likes per tip length; gold requires sum of likes per tip length. |

## Benchmark/gold incompatibilities (`G-GOLD`, 20)

| ID | Current trajectory versus gold |
| --- | --- |
| `06492` | Question/external knowledge specify grouping images by object count `<15`; gold instead filters `OBJ_SAMPLE_ID<15` row-wise and counts rows. |
| `06026` | Question names a product but no region; gold arbitrarily reads distinct South profit, while the product occurs across regional fact tables. |
| `03688` | Model returns all inventories tied at maximum film length; gold `LIMIT 1` chooses one without a tie rule. |
| `02925` | “Which product” permits a name; gold requires `ProductID`. |
| `00715` | External knowledge defines address as street number plus street name; gold returns street name only. |
| `04822` | Question/external knowledge require Avangard Omsk; gold filters Czech Republic (all). |
| `05668` | “To which vendor” permits vendor name; gold requires VendorID. |
| `00214` | Question asks plural keywords of the highest-popularity movie; gold takes one arbitrary keyword with `LIMIT 1`. |
| `01589` | External knowledge requires `attribute_value='true'`; gold requires `none/no/false`. |
| `02438` | Model returns each order's latest status; gold returns all distinct historical statuses, with no current/history cue in the question. |
| `05632` | Question/external knowledge ask debtors among Japanese suppliers; gold asks Japanese share among all debtors. |
| `03664` | External knowledge defines price per day as rate/duration; gold ranks by that value but returns raw rental rate. |
| `03216` | External knowledge says divide `SUM(id)` by three and the model does so; gold counts joined criteria rows instead of summing criterion IDs. |
| `02052` | Nineteen aliases share the maximum value; gold's unqualified `LIMIT 1` picks `Sconset`, while the question gives no tie rule. |
| `01213` | Question asks character names of the newest work; model returns all 47, while gold takes one arbitrary character. |
| `04244` | Natural language supports intersecting high-total teams with teams having an MVP; gold sums only MVP-player rows and merges years by team name. |
| `00796` | External knowledge points to Man-of-the-Match counts; gold uses an unusual Season.Man-of-the-Series→Match.Man-of-the-Match join. |
| `05161` | “Percentage of recipes” supports deduplicating recipes; gold weights a recipe repeatedly when it has multiple cheese ingredient rows. |
| `05147` | “Percentage of businesses” supports distinct businesses; gold weights violations×inspections join rows. |
| `06299` | Question asks total dish counts for two UUIDs, but gold subtracts one from the other; the trajectory also loops to max steps without a terminal. |

## Metric contract and implementation consequence

- `strict-multiset`: normalized cell equality, row order ignored, duplicate multiplicity preserved.
- `bird-set`: raw BIRD reference set equality, row order and duplicate multiplicity ignored.
- The direct-SQL BIRD baseline launchers explicitly use `--denotation-comparison bird-set`.
- `generate_teacher_rollouts.py` now exposes the same explicit flag and records it in every record
  and manifest. Its default remains `strict-multiset`; a `bird-set` run is marked ineligible for SFT
  export so the stricter training/replay gate does not silently change.

Future tool-agent versus direct-SQL evaluation must use `bird-set` on both sides. Training replay,
grounding audits, and SFT quality gates should continue to use `strict-multiset`.
