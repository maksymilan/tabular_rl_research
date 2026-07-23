# BIRD version23 actionable-feedback gate (50 tasks, 2026-07-23)

## Decision

**Do not run version23 on the fixed 200.** It improved the deliberately enriched 50-task cohort but
failed both recovery and net-gain expansion thresholds.

## Frozen setup

- Model: `deepseek-v4-flash`
- Context: `rolling-legal-history`, 4 successful turns
- Prompt: canonical
- Plan policy: optional
- Provider carrier: JSON Output with native reasoning
- Maximum actions/tokens: 30 / 2048
- Denotation metric: `bird-set`
- Input:
  `data/eval_inputs/tool_usability_20260723/version23_actionable_feedback_gate50.jsonl`
- Candidate:
  `data/trajectories/tool_usability_20260723/version23_actionable_feedback_gate50_json.all.jsonl`
- Original paired records:
  `data/trajectories/tool_usability_20260723/version20_final_fixed200_json.all.jsonl`

The cohort contains all 20 audit-confirmed model-capability failures and 30 original-correct
controls. The original paired score is 30/50.

## Gate result

| Measure | Required | Original | version23 | Pass |
|---|---:|---:|---:|---|
| Overall correct | net gain at least 7 | 30/50 | 35/50 | **No: +5** |
| Capability recoveries | at least 7/20 | 0/20 | 6/20 | **No** |
| Controls retained | at least 28/30 | 30/30 | 29/30 | Yes |
| Legal termination | at least 49/50 | 50/50 | 50/50 | Yes |
| Mean actions | increase at most 1 | 6.96 | 6.96 | Yes |

The candidate used six recoverable semantic error actions versus four in the original paired
records. All 50 trajectories terminated legally, but process errors did not improve.

## Paired gains and regression

Recovered failures:

- `04189`: switches the Free/Sports app-to-review join to inner.
- `06454`: preserves the stored uppercase result.
- `00582`: returns first, last, and age as three columns.
- `05544`: counts distinct representatives instead of joined term rows.
- `02512`: returns first and last as separate fields.
- `02507`: returns first and last as separate fields.

`06454` did not emit structural feedback, so its change cannot be attributed to version23 and is
consistent with ordinary rollout variation. The other five gains align with the intended feedback
boundary.

Regressed control:

- `01050`: multiplies unit profit by order quantity despite external knowledge defining net profit
  as unit price minus unit cost. The aggregate feedback is emitted later and does not cause this
  earlier semantic choice; this is also consistent with rollout variation.

## Remaining capability failures

| Failure type | Tasks | Tool-design implication |
|---|---|---|
| Existing warning ignored | `02901`, `02189` | Stronger advisory prose is unlikely to be reliable; blocking the action would remove valid concatenation/global-aggregate uses. |
| Ambiguous requested layout | `04038` | Task/gold conflict remains; a tool cannot safely infer the intended output orientation. |
| Missing/extra/wrong final fields | `01152`, `04848`, `04426`, `00040`, `02868`, `03131` | The relational tools can already express the gold output. Detecting the error requires task-aware output-schema judgment, not another SQL atom. |
| Operator/population/grain ordering | `00593`, `05440`, `00004`, `02418`, `06165` | Existing atoms express the gold path; failures are policy choices over action order, population, or grouping identity. |

## Process errors

Six rejected actions occurred:

- one provider protocol envelope error (`03275`);
- one unexpected camelCase aggregate argument and two duplicate plan creates (`02418`);
- one non-scalar `value_ref` (`02462`);
- one invalid plan op (`00582`).

All were state-preserving and recovered except that `02418` remained semantically wrong.

## Interpretation

Version23 is useful as an audited local-feedback experiment, but it does not justify a claim that
the tool interface alone reaches the 75% usability target. The remaining safe tool boundary is
important:

- Current relational atoms can express every audited capability failure.
- More advisory wording is already ignored on some traces.
- Automatically rejecting concatenation, global aggregation, left joins, or helper columns would
  create false rejections for legitimate tasks.
- Detecting missing/extra final fields requires comparing the task meaning with the evidence
  schema. That is a task-aware semantic validator or learned policy, not an atomic SQL operation.

Do not construct SFT data from these `bird-set` evaluation runs. Before further tool mutation,
choose explicitly between keeping version23 feedback and moving the remaining corrections into
causal SFT/RL, or introducing a task-aware pre-terminal validator as a separately named research
condition.
