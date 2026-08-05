# BIRD SFT-2 On-policy Data Construction Protocol

## Goal

SFT-2 refines the frozen SFT-1 student on states that the student actually visits.  Data generation
therefore follows a fixed order:

1. run the SFT-1 student in the real BIRD-train harness with stochastic pass@K;
2. admit replay-verified student successes first;
3. retain the first legal action after a real harness error as Feedback Recovery, but only when its
   complete trajectory is verifier-correct;
4. send only tasks that still fail after pass@K to an external teacher;
5. use DeepSeek v4 Flash first, and change to v4 Pro only after a recorded Flash quality/yield gate;
6. admit a teacher correction only when its complete branch is replay- and denotation-correct.

Gold SQL is harness-only.  It may score terminal denotation but must never enter a model input,
teacher prompt, reasoning target, or correction explanation.

All new DeepSeek teacher calls use the official `https://api.deepseek.com` Chat Completions
service. AimixHub/AIHubMix is deprecated and must not be used as a provider, proxy, or fallback.
Provider configuration and the FIM boundary are defined in `provider_api.md`.

## Data lanes

The complete second-stage mixture is:

```text
D_SFT2 = D_demo_replay + D_student_success + D_decision_correction + D_feedback_recovery
```

### Student success

Every legal action in a verifier-correct SFT-1 student rollout is a candidate target.  The raw
pass@K artifact is re-executed from an empty harness to reconstruct authoritative state-before,
state-after, outputs, references, and error-conditioned recovery flags.  Exact duplicate action
sequences for the same task are collapsed before quality filtering.

### Feedback Recovery

An error action remains in `turns` and `error_events` only.  It is never an SFT target.  The first
subsequent legal action is rendered from the unchanged environment plus structured
`LAST TOOL ERROR` and tagged `feedback_recovery`.  Recovery targets are a tagged subset of student
success, not duplicated records.  Sampling weights may expose them more often.

### Teacher fallback and Decision Correction

External generation is restricted to tasks with no correct student sample after the declared K.
The teacher starts from the same task and current protocol, receives no gold SQL, and runs in the
real harness. It sees the shared student tool/state contract plus teacher-only generation
guidance and examples. A teacher-success trajectory may provide fallback demonstrations. A Decision
Correction target is the teacher action at the first divergence after an exactly matching legal
student/teacher prefix.  The state-before must match, the teacher action must differ from the
student action, and the complete teacher branch must replay to the correct denotation.  The bad
student action stays only in the paired audit record.

This first-divergence label is privileged offline supervision.  It does not claim that every later
student action is independently wrong.  When no exact shared prefix/state exists, no correction is
exported.

## Training shape

The current SFT checkpoint, evaluation, and RL use one bounded-rolling **student runtime prompt**.
SFT export re-renders the teacher's canonical executed trajectory with that prompt; it does not
copy teacher-only examples or edge-case guidance into training records. SFT-2 must keep the same
`rolling-legal-history`, `history_turns=4`, student prompt, and resident-observation contract. Each
ShareGPT record uses `mask_history: true`; only the final assistant action receives loss. Rejected
actions are absent from legal history, while their structured error is present in the current user
state when applicable.

## Admission gates

- BIRD train only; held-out BIRD dev never supplies SFT data.
- Strict parser and current argument schema.
- Fresh database replay with matching state snapshots.
- Correct final denotation for every accepted source branch.
- No gold SQL, current output, or future factual step reference in model input.
- Error actions excluded from labels.
- Exact source/protocol/context metadata, teacher/student prompt hashes, public tool-schema hash,
  and state hashes retained.
- Repeated identical calls, overlong trajectories, overlong reasoning, and target truncation are
  rejected or reported, never silently repaired.
- Deduplicate by task and canonical tool-action sequence; keep origin and transition type in a
  separate index for mixture control and ablations.

## Pilot and scale gate

The first pilot uses the existing SFT-1 K=4 pool, then selects a small difficulty-balanced set of
remaining failures for one Flash attempt each.  Report student yield, teacher success, strict
carrier failures, replay failures, correction-pair yield, token lengths, and accepted target counts.
Do not switch to Pro merely because individual tasks are hard.  Switch only if Flash's verified
branch/correction yield or protocol compliance is too low for economical scaling, and rerun the
same frozen pilot ids for a fair comparison.
