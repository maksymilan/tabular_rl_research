# BIRD causal data construction protocol

Status: the current training mainline has returned to the frozen Qwen3-8B Atomic version26 SFT1
method. New scale data must reproduce its real teacher↔Harness causal loop, hidden-gold boundary,
rolling-prefix projection, and replay/quality gates. External DeepSeek generation is paused until
the user explicitly resumes it. The SFT-2 and checkpoint-relalg material below is retained as
future on-policy/control design; it has no automatic admission into the version26 mainline.

## Retained SFT-2 design

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

All new tool/protocol generation work starts from `checkpoint-relalg-v1` with an explicit
`mode=direct|atomic|hybrid`. It runs one official DeepSeek native call per assistant turn against
the real shared state/artifact/checkpoint Harness. The provider sees only a short shared core, one
short mode prompt, compact schemas for that mode, and the current causal dynamic context. A teacher
adds only short checkpoint-use guidance. The complete design specification is implementation-side
material and must never be copied wholesale into requests. The scheme remains diagnostic-only;
generated episodes are audit artifacts, not SFT records, until scheme-specific replay, no-leak,
behavior, exporter, and admission gates pass.

The completed atomic `version50` diagnostic tested official DeepSeek native function calls as the
teacher transport. It preserved the same causal harness loop and canonical stored trajectory but
scored 133/200 versus historical version24 at 145/200, with 183/200 versus 197/200 legal
termination and 166 versus 29 process errors. It is permanently `diagnostic-only`; its successful
episodes are not SFT candidates despite passing replay and structural/no-leak audits. The batch
entry point remains
`src/sft/generate_teacher_rollouts.py`, selected with
`--atomic-protocol-version version50 --deepseek-carrier native-tool-calls --diagnostic-only`.

The frozen provider behavior baseline is `version51` with the distinct `native-tool-bundle` scheme. One real
assistant turn may contain 1..8 direct primitive calls. All calls are validated against the same
pre-turn harness state, executed and audited in provider order, and returned as one matching tool
message per call id. The record keeps the shared `model_turn_index`; it must never flatten the
bundle into several assistant turns that falsely imply intermediate observations. The entry point
is `--atomic-protocol-version version51 --deepseek-carrier native-tool-bundle
--diagnostic-only`.

The frozen compact-prompt predecessor is `version52` with the same carrier and execution contract. Its
compact prompt treats the supplied native schemas as the sole function/argument-shape authority,
adds soft one-to-three-call scheduling and explicit error recovery, and preserves every errored
assistant bundle plus structured tool feedback in causal history. A rejected bundle is never a
positive SFT target; a later corrected bundle may be, with the error prefix intact. Records also
contain `native_bundle_rl_statistics`, which is fact-only and must not be interpreted as a reward.
Launch it with `--atomic-protocol-version version52 --deepseek-carrier native-tool-bundle
--diagnostic-only` only for explicit reproduction.

The frozen reviewed prompt is `version53`. It preserves version52 execution while separating shared runtime
semantics from teacher-only trajectory-generation rules and hardening data authority, join
cardinality, output representation, independent filters, and error recovery. Launch it with
`--atomic-protocol-version version53 --deepseek-carrier native-tool-bundle --diagnostic-only`.
The frozen no-plan diagnostic is `version54`: it keeps version53's student prompt and all non-plan
semantics, while removing only the provider-visible `plan` function and stale teacher plan rule.
Launch it with `--atomic-protocol-version version54 --deepseek-carrier native-tool-bundle
--diagnostic-only`. Version51-version54 are not SFT sources until a scheme-aware exporter and
explicit training-admission gates promote them. Keep this lineage for existing compatible work and
exact reproduction; new protocol/tool work branches from `checkpoint-relalg-v1`, not version54 or
version26.

The preregistered version54 rollout gate is single-arm: use the frozen first 200 tasks of
`bird_train_atomic_teacher1500_v2_nonempty` in source order and judge absolute verified yield,
legal completion, process-error, replay, native-history, no-plan, and no-leak gates. Do not run a
v53 comparison. Only if every Prefix200 gate passes may the same configuration continue over the
remaining 1,300 frozen tasks. All resulting multi-call trajectories remain diagnostic candidates;
they are not SFT records until a scheme-aware exporter and explicit promotion exist.
This frozen plan and the version51 results below provide no admission evidence for
`checkpoint-relalg-v1`. Its first actual official request on 2026-08-09 passed model identity
verification but was transport-blocked by `HTTP 402 / Insufficient Balance` before any authored
tool call. That audited failure artifact is not a semantic pilot result; the scheme must still be
evaluated under its own protocol hashes and gates after official service availability is restored.

Its frozen Gate32 passed: 22/32 correct and 32/32 legal versus version50 at 16/32 and 26/32. The
completed fixed-200 scored 147/200 correct and 200/200 legal versus version50 at 133/200 and
183/200, with significant paired improvement and all replay/structure/provider-history/no-leak
audits passing. It remains statistically tied with version24's 145/200 and uses 1.92x its tokens;
the SFT exclusion above remains.

The separate active `iterative-sql-v6` causal loop is also diagnostic-only. Although its launcher lives
under `src/sft/` for external-teacher dispatch, its `execute_sql`/`submit_sql` episodes are not
atomic SFT candidates and cannot enter any current exporter or mixture. A future admission would
require its own frozen paired gate, fresh replay/no-leak audit, and scheme-aware exporter.

## Training-task admission before rollout

Training-task selection now applies `gold-denotation-nonempty-task-filter-v1` before any student or
external-teacher episode starts. The local harness executes hidden gold SQL against a SQLite
connection opened with URI `mode=ro` plus `PRAGMA query_only=ON` and calls only `fetchone()`:

- zero returned rows exclude the task;
- a returned 1x1 scalar whose value is `0` is nonempty and stays eligible;
- SQL execution errors, timeouts, missing databases, and invalid query inputs fail closed and are
  not training tasks;
- source `gold_exec_results` placeholders are ignored;
- neither SQL text, rows, values, nor the private empty/nonempty status may enter a model/teacher
  prompt or external-provider request.

The 2026-08-06 audit executed all 6,601 normalized BIRD-train tasks. It found 6,599 certified
nonempty tasks, zero true empty denotations, and two execution errors. The compatible rollout pool
had 5,915/5,915 certified nonempty tasks and no errors. Frozen artifacts are:

- `data/eval_inputs/bird_train_filtered_nonempty_v1.jsonl` (6,599);
- `data/eval_inputs/bird_train_tool_compatible_nonempty_v1.jsonl` (5,915);
- private proof `data/eval_inputs/bird_train_nonempty_v1.private_status.jsonl` and its manifest;
- `data/eval_inputs/bird_train_atomic_teacher1500_v2_nonempty.jsonl` plus manifest.

The 1,500-task v2 preserves every v1 task id and its exact order because all 1,500 were certified
nonempty. Its task file has the same SHA-256 as v1; v2 adds a hash-bound admission proof instead of
confounding this policy change with resampling. New teacher-data generation uses v2. The v1 cohort
is retained only for historical reproduction. Selection refuses an uncertified training cohort by
default; `--allow-unfiltered-historical-reproduction` is an explicit historical-only bypass.

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

### Empty-result filtering

`causal-empty-result-target-filter-v1` treats an empty relation like recoverable environment
feedback, not a positive demonstration. A successful intermediate call with an explicit
`row_count=0` stays in `steps` and subsequent history but is marked `sft_target_eligible=false`;
the next grounded correction may still be supervised. If `answer_from_context` cites a table whose
resident `row_count=0`, the whole trajectory is excluded from training. A scalar answer whose 1x1
table contains numeric zero is not empty and remains eligible. Manifests report empty context-only
steps and excluded terminal-empty trajectories separately.

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

The current version26 SFT checkpoint, evaluation, and RL use its frozen bounded-rolling **student
runtime prompt**. Expanded version26 data must retain that exact prompt/carrier/history identity.
Any future SFT-2 export must likewise re-render the teacher's canonical executed trajectory with
its declared student prompt; it does not copy teacher-only examples or edge-case guidance into
training records. SFT-2 must keep the same
`rolling-legal-history`, `history_turns=4`, student prompt, and resident-observation contract. Each
ShareGPT record uses `mask_history: true`; only the final assistant action receives loss. Rejected
actions are absent from legal history, while their structured error is present in the current user
state when applicable.

## Admission gates

- BIRD train only; held-out BIRD dev never supplies SFT data.
- Every source task is a member of a hash-bound nonempty-gold-denotation task pool before rollout;
  zero-row tasks and tasks that cannot be certified are excluded before provider dispatch.
- Strict parser and current argument schema.
- Fresh database replay with matching state snapshots.
- Correct final denotation for every accepted source branch.
- No gold SQL, current output, or future factual step reference in model input.
- Error actions excluded from labels.
- Empty-row intermediate results excluded from labels while retained in causal context; terminal-
  empty trajectories excluded wholesale. Grounded 1x1 scalar zero remains eligible.
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
