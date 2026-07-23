# BIRD version23 actionable-feedback pilot (16 tasks, 2026-07-23)

## Decision

**Weak pass; expand only to a frozen 50-task gate.** Version23 met every predeclared pilot
threshold, but only at the target/control boundary and with no overall accuracy improvement over
version22.

## Controlled setup

The input, model, provider carrier, rolling context, prompt, plan policy, action/token budgets, and
`bird-set` metric are identical to the version22 pilot. The only implementation change is more
actionable wording inside the three existing structural-feedback payloads. Public tools,
model-authored arguments, SQL execution, provenance, resident-state boundaries, and terminal
scoring are unchanged.

- Input:
  `data/eval_inputs/tool_usability_20260723/version22_structural_feedback_pilot16.jsonl`
- Version23:
  `data/trajectories/tool_usability_20260723/version23_actionable_feedback_pilot16_json.all.jsonl`
- Version22:
  `data/trajectories/tool_usability_20260723/version22_structural_feedback_pilot16_json.all.jsonl`
- Original paired records:
  `data/trajectories/tool_usability_20260723/version20_final_fixed200_json.all.jsonl`

## Result

| Measure | version20 | version22 | version23 |
|---|---:|---:|---:|
| Correct overall | 8/16 | 10/16 | 10/16 |
| Unambiguous targets correct | 0/7 | 2/7 | 3/7 |
| Controls correct | 8/8 | 8/8 | 7/8 |
| Ambiguity diagnostic correct | 0/1 | 0/1 | 0/1 |
| Mean actions | 6.875 | 6.813 | 7.000 |
| Legal termination | 16/16 | 16/16 | 16/16 |
| Recoverable semantic error actions | archived comparison | 0 | 1 |

The declared pilot gate was target recovery at least 3/7, control regression at most 1/8, and mean
action increase at most one. Version23 met those thresholds exactly or comfortably.

## Paired changes versus original

Recovered:

- `bird_train_02901`: returns first/middle/last as three stored fields.
- `bird_train_05544`: changes joined-row count to distinct representative count after feedback.
- `bird_train_02512`: returns first/last as separate stored fields.

Regressed control:

- `bird_train_01148`: retains the helper `diff` column beside `coachID`. No structural feedback was
  emitted on this trajectory, so this is not an observed feedback reaction and may be sampling
  variation.

Other important traces:

- `00582` initially concatenates first/last, then creates the correct separate-field table after
  feedback, but terminally cites the earlier concatenated table. The relational path and correction
  are model-capable; final evidence-handle selection remains a policy/SFT issue.
- `03145` first emits an invalid empty aggregate `where`, receives a state-preserving validation
  error, recovers, and answers correctly.
- `04189`, `02507`, and `02189` explicitly ignore the actionable warning. Feedback is advisory and
  does not force a semantic correction.

## Frozen 50-task expansion gate

The expansion cohort contains all 20 audited model-capability failures and 30 original-correct
controls. Because no original-correct task in the fixed 200 used a triggering concatenation or
single-edge left join, the controls emphasize aggregation, where version23 feedback is always
emitted.

Expansion is allowed to the full 200 only if all conditions hold:

1. at least 7/20 capability failures are recovered;
2. net paired gain versus original is at least 7/50;
3. at least 28/30 controls remain correct;
4. legal termination is at least 49/50;
5. mean actions increase by no more than one.

This gate is intentionally aligned with the required fixed-200 gain from the baseline-aligned
143/200 to 150/200. Passing the selected cohort does not itself establish 75%; it only authorizes
the complete fixed-200 run.
