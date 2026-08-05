# Reinforcement-learning pipeline

## Scope

The active RL system trains a policy over typed table tools in the same closed-loop environment
used for rollout and evaluation. Gold SQL is hidden from the actor and is used only by the harness
to score terminal denotation on training tasks.

RL uses the same concise rolling student runtime prompt as SFT export and tool evaluation. It never
loads the external teacher's generation guidance or worked examples. Checkpoint metadata binds the
protocol version/hash, student prompt SHA-256, and public tool-schema SHA-256 so resume cannot
silently cross a prompt contract.

## Active modules

- `src/rl/tool_environment.py`: one causal model↔harness episode, selected exclusively as
  `atomic` or `action-block`.
- `src/rl/task_loader.py`: model-visible task records plus hidden execution labels.
- `src/rl/terminal_reward.py`: the binary result-only control.
- `src/rl/target_support.py`: bounded hidden gold T/C/R support used only on the reward side.
- `src/rl/task_support.py`: deterministic model-visible literal support and canonical rewrites.
- `src/harness/observation_binding.py`: canonical literal-slot parsing and conservative
  visible-singleton binding semantics shared by provenance and replay.
- `src/rl/process_credit.py`: replay-derived, step-local process credit.
- `src/rl/process_objective.py`: per-step policy objective.
- `src/rl/trajectory_replay.py`: path-independent replay of the same authored tool program on
  schema-compatible counterfactual databases.
- `src/rl/counterfactual_suite.py`: immutable task-suite manifest loading and artifact binding.
- `src/rl/build_sft_task_set.py`: exact BIRD task cohort from the retained SFT index.
- `src/rl/audit_sft_process_rewards.py`: exact SFT-index reward coverage and one-factor ablations.
- `src/rl/audit_reward_sensitivity.py`: matched success/failure coefficient sensitivity scan.
- `src/rl/review_grounding_edges_external.py`: strict external grounding/dependency packages and
  reviews.
- `src/rl/select_grounding_recheck.py` and `src/rl/summarize_grounding_external_audit.py`:
  deterministic second-opinion selection and consensus aggregation.
- `src/rl/frameworks/trl/`: the framework-backed transition-level GRPO/PPO backend. It retains
  one exact rolling-state prefix per assistant turn, performs batched multi-episode rollout through
  a dedicated TRL vLLM server, and delegates temperature-aware old/reference log probabilities,
  tokenwise clipping, KL, optimizer scheduling, gradient clipping, and checkpointing to TRL.
- `src/rl/frameworks/accelerate/`: the frozen single-GPU QLoRA REINFORCE baseline. It remains an
  audit/equivalence reference and must not receive new optimizer features.

The historical Verl integration is archived under `archive/code/experimental_backends/verl/` and
is not a supported training entry point. It concatenated multi-turn transcripts, which is not
equivalent to the active rolling-state contract where every assistant turn has a separately rebuilt
causal prefix.

## Result-only control

The deliberately coarse baseline uses:

```text
reward = 1  if the terminal answer has the correct executed denotation
         0  otherwise
```

It has no process shaping, error penalty, or length reward. Homogeneous groups have zero relative
policy advantage; when the controlled experiment uses a nonzero KL coefficient, they may still
receive the same frozen-SFT-reference KL update as the process condition. Because the episode-level
advantage is applied to all generated turns, recovered successful trajectories can positively train
earlier erroneous turns; this behavior is retained only as the control condition.

## Process credit

`process_credit.py` reconstructs harness-owned data, value, and grounding dependencies by replaying
the actor's own trajectory. Model-authored reasoning and plan text never determine factual credit.
Local protocol, argument, and execution errors remain assigned to the event that caused them;
recovery credit is assigned to the later legal action that uses the feedback.

The current per-turn reward is:

```text
r_t = C*c_positive_t
      + (1-C)*eta*target_potential_delta_t
      + lambda_answer*legal_answer_t
      - capped_episode_penalty*c_negative_t
```

Positive credit linearly normalizes harness-grounded back-slice membership, first-used evidence,
fixed-root search reduction, verified feedback recovery, and hidden target-support progress. Zero
feature mass stays zero; a correct episode with no grounded positive mass is excluded from process
optimization rather than receiving terminal fallback credit. Failure exploration receives only the
bounded target-potential term. Terminal failure is attached to the last assistant action, while
tool/protocol errors, unsupported guesses, repeated calls, semantic no-ops, ignored feedback, and
unverified empty filters stay local to the action that caused them.

`answer_from_context` receives only the small legal-format reward plus any independently grounded
feedback-recovery signal. `plan` is control state rather than factual evidence and therefore has no
special positive reward. Every other current tool can receive back-slice/evidence/target-support
credit when its executed output participates in the verified answer.

The active framework adapter applies one scalar advantage to the complete authored assistant turn.
Every turn retains its exact rollout prompt ids, response ids, and vLLM sampled-token log
probabilities; environment tokens are never action tokens. TRL recomputes temperature-consistent
old/reference token probabilities, applies tokenwise PPO clipping and vLLM importance correction,
and reduces the clipped loss over response tokens. Optional KL uses a frozen copy of the SFT
initialization adapter. Result-only and process conditions share the same causal environment and
`bird-set` terminal scorer; reward mode is the controlled difference.

The framework-backed default uses learning rate `3e-7`, constant scheduling, no warmup, PPO clip
`0.2`, fixed-reference KL `0.001`, gradient clipping `1.0`, sampling temperature `1.0`, and
`top_p=1.0`. Each rollout batch is reused for two clipped PPO iterations. Dropout is disabled so
rollout and gradient policies agree. A batch with no trainable
transitions aborts rather than silently advancing optimizer or scheduler state. The source-informed
backend rationale and framework equivalence requirements are recorded in
`docs/decisions/process_rl_backend.md`.

`frameworks/trl/run_atomic_transition_grpo.sh` launches the atomic condition after a dedicated
server has been started with `frameworks/trl/start_vllm_server.sh`. For a controlled ablation, pin the same
model/adapter, task-set artifact, group size, decoding settings, action/context budgets, learning
rate, update count, and KL coefficient; change only `REWARD_MODE` (and the process reward config
that is unused by the result-only control).

This backend being executable does not promote process-RL optimization. Keep production launches
on the result-only engineering control until the deterministic completeness and independent
grounding edge-precision gates in the project contract both pass. A process-mode smoke may test
adapter plumbing, but its checkpoint is not an admissible research result.

The frozen Accelerate backend also accepts `--tool-scheme atomic|action-block`. Its action-block
path supports result-only RL only. It deliberately rejects process mode because existing process
credit is atomic-action-local, while one action-block response may contain several successful, failed, and
blocked primitive calls. Assigning those local rewards to the whole authored block would violate
the credit boundary. Atomic process RL remains unchanged.

Process RL is the main research condition; result-only RL is only its matched baseline. Before the
process condition is launched, both deterministic completeness and independent edge-precision
audits must pass. Versioned reports and frozen configs live in `docs/reports/rl/` and
`src/rl/configs/` respectively.

The queued Exp7 strong-penalty ablation keeps the NoBackSlice positive-credit surface, full-turn
training, normalized positive mass, optimizer, rollout, and evaluation protocol fixed. It triples
only the three active deterministic local penalties: tool/protocol error `0.08 -> 0.24`, adjacent
exact repeat `0.06 -> 0.18`, and legal semantic no-state-change `0.03 -> 0.09`; its trajectory cap
is `0.95`, guaranteeing that a grounded correct trajectory with normalized positive mass `1.0`
retains total reward of at least `0.05`. `ignored_feedback` and `unsupported_guess` remain zero
because a July 31 audit found that
the former conflates unverified recovery with ignored feedback and the latter has systematic false
positives for valid composed date patterns such as `2017-03%`.

Two rank follow-ups are queued after Exp7. Exp8 (`phase4_process_rank_action_only`) keeps the
NoBackSlice process loss and the frozen Rank pairing/coefficient/beta fixed, but computes the rank
score and rank gradient only on the exact raw-JSON tool-action suffix; `<think>` tokens receive no
rank gradient. Exp9 (`phase5_process_rank_conservative`) keeps Exp8 fixed and changes only which
action turns receive the detached pairwise rank coefficient: every legal, locally unpenalized call
in a verifier-correct trajectory is reinforced; only calls with a deterministic positive local
penalty in an incorrect trajectory are suppressed; legal exploration in an incorrect trajectory
is rank-neutral. The ordinary NoBackSlice process-loss branch remains unchanged in both runs, so
Exp8 isolates action-token ranking and Exp9 isolates conservative rank-gradient routing.

Exp8 and Exp9 are assigned to the shared `NewGNN` host only after a fail-closed readiness audit.
Both runs initialize from the frozen SFT2 `checkpoint-1682` adapter over its exact
`Qwen2.5-Coder-7B-Instruct` base; the base checkpoint alone is never the RL initialization. The
audit verifies the SFT2 adapter hash, base-model declaration and shard sizes, exact TRL software
versions, all 23 training databases, the content-hashed counterfactual suite, and all databases for
the fixed equal-300 evaluation cohort. The shared-server queue uses only candidate GPUs 6 and 7,
requires both cards to remain below 512 MiB for five continuous minutes, rechecks immediately
before every training/evaluation phase, and never stops or replaces another user's process. A
table_rl handoff marker disables its duplicate Exp8/Exp9 queue only after NewGNN passes readiness
and a bounded CUDA/vLLM/SFT2 load smoke.

Exp10 and Exp11 are isolated table_rl follow-ups, each initialized independently from the same
frozen SFT2 `checkpoint-1682` adapter rather than from any earlier RL checkpoint. Exp10
(`phase6_process_rank_conservative_score_masked`) closes the remaining Exp9 score/update mismatch:
the pairwise score as well as its gradient includes only clean legal actions from a correct
trajectory and deterministic locally penalized actions from an incorrect trajectory. A pair is
dropped when either side has no such action. Exp11
(`phase7_process_rank_conservative_action_mean`) keeps Exp10 fixed and changes only rank-score
reduction: tool-token log probabilities are averaged within each selected action, then selected
action scores are averaged within each trajectory. This removes both argument-length and action-
count bias while retaining the ordinary NoBackSlice process-loss branch. The sequential queue and
five-minute server watchdog run from the isolated
`/home/dengyan/tabular_rl_outputs/rl_runtime_rank_score_v3_20260731` runtime; every completed
checkpoint historically received the frozen equal-300 K=4 evaluation. Starting with the controlled
Exp12-Exp15 follow-up, routine selection instead uses full 1,534-question BIRD-dev greedy@1
(`temperature=0`, `top_p=1`, `bird-set`), because its 1,534 trajectories are close in cost to the
1,200 trajectories from equal-300 K=4 while covering the complete dev distribution. K=4 is reserved
for at most the final selected method after full-dev comparison.

Exp16 and Exp17 are two preregistered dense full-response follow-ups requested after the full-dev
Exp12-Exp15 comparison showed that sparse tool-token credit produced only small, unstable gains.
They reuse the exact balanced mixed-60 SFT2 K=4 trajectories, task order, 60 one-question optimizer
steps, `1e-6` learning rate, one PPO iteration, zero KL, Rank coefficient `0.5`, beta `0.1`, and
action-mean trajectory scoring. Both initialize independently from the frozen SFT2
`checkpoint-1682`; neither resumes an RL checkpoint or performs online rollout. Unlike Exp14,
both set `trainable_part=all` and score Rank over all authored response tokens, so the `<think>`
carrier and tool JSON receive the same action-level coefficient.

Exp16 (`exp16_dense_uniform_full_response`) assigns raw action credit `+1` to every clean legal
turn in a correct trajectory and `-0.5` to every clean legal turn in an incorrect trajectory.
Tool/protocol/execution errors, adjacent exact repeats, and legal semantic no-state-change turns
all receive one non-stacking raw `-2` penalty. Every raw action value is divided by the number of
authored turns in that trajectory. Exp17 (`exp17_dense_strategic_full_response`) keeps the same
negative rules and adds, only on correct trajectories, `+0.5` to an observation turn whose newly
exposed harness evidence is later used and `+0.5` to a table-producing operator in the terminal
dependency backslice, before the same action-count normalization. Perception and table-producing
tool classes are disjoint, so the maximum legal positive action is `+1.5`, strictly below the
severe penalty magnitude. Rank uses the matching `dense_outcome` score/update scope: clean legal
turns on the positive side and every turn on the negative side.

These two fixed-pool views intentionally do not apply counterfactual Process screening. This is a
scoped experimental choice, not a change to the default online process-RL admission policy: the
research question is whether dense outcome-supervised gradients over the actor's full response can
train semantic interpretation of environment feedback rather than only a sparse set of executable
tool tokens. Their primary evaluation is the complete 1,534-question BIRD-dev greedy protocol;
fixed-prefix scores remain auxiliary and K=4 is deferred to at most the final selected method.

Current gate status (2026-07-30): deterministic replay coverage passes. A trajectory-level re-audit
found three confirmed denotation shortcuts, six valid observation-to-literal dependencies omitted
by the former edge extractor, and five annotation/reviewer conflicts. The six dependencies are now
represented by exact visible-cell locators. The current provenance code rebuilt all 703 packages;
after locator-only fields were normalized, eight packages changed semantically. At the user's
direction, a frozen Codex audit covered all eight changes plus twelve deterministic controls:
63/63 grounding edges had exact structural support and no missing edge was reported. This passes
the changed-extractor edge-precision gate under the recorded
`codex-primary-structured-audit` reviewer, not an external Flash/Pro dual review. The same audit
found one additional terminal shortcut, `bird_train_04696`, whose final projection hardcodes
`àbac-xinès` instead of deriving the selected pair from the top row. See
`docs/reports/rl/CODEX_GROUNDING_CHANGE_AUDIT_20260730_ZH.md`.

Path-independent counterfactual replay uses `process-counterfactual-suite-v2`. It keeps the actor's
operator structure fixed while rebinding only literal slots that the source harness proves were
copied from an unambiguous singleton observation. Question/external-knowledge literals remain
constant. Ambiguous repeated cells and selections from multi-row observations are not rebound,
because choosing one row may itself encode an unexecuted relation. Replay scores only the mandatory
terminal evidence table, requires at least one changed/non-vacuous gold denotation, and excludes a
correct trajectory from process optimization when any database distinguishes it. This preserves
an interactive policy's causal observation use while still exposing omitted predicates and
unexecuted relations such as an argmax performed only in model reasoning.

Process mode requires a content-hashed v2 counterfactual-suite manifest whose independent quality
gate is marked passed and bound to an audit SHA-256; stochastic database generation is never run
inside the optimizer. That release gate passed on 2026-07-30 for the frozen 23-task RL selection.
The suite contains two complete-schema BIRD database variants per task (46 databases total).
Known-correct replay passed for 102/105 source-correct sampled trajectories, with at least one
passing program for all 23 tasks; the three rejected source-correct trajectories were independently
auditable shortcuts (two hard-coded `USA` join keys and one hard-coded episode id). Four frozen
negative controls—omitted role/credit status, omitted expiration year, hard-coded observed argmax,
and `bird_train_04696`'s terminal constant output—were all rejected. The independent Codex audit
also passed 63/63 changed-extractor grounding edges with no reported missing edge. The trainer
dry-load bound all 23 `spider_train_*` task ids to 23 suites and 46 content-hashed databases.

The formal audit is
`data/results/process_gate_v2_20260730/process_rl_mandatory_gates.audit.json` (SHA-256
`6ba130933ec4ce20e58117b506d57cd80181f9b1c7c044765572ca8b04ba8e49`), and the promoted manifest is
`data/results/process_gate_v2_20260730/counterfactual_suite_v2.passed.json` (SHA-256
`5624c75c6ebbca556f1e0946069fdb95e923dc95d2d7692c8d929bedebd9568b`). Process experiment configs
marked `allowed_process_after_gates` may run only with that strict
`counterfactual-completeness` path; the historical `phase1_process_current` checkpoint remains
evaluation-only. See `docs/reports/rl/GROUNDING_SHORTCUT_REAUDIT_20260724.md` and
`docs/reports/rl/CODEX_GROUNDING_CHANGE_AUDIT_20260730_ZH.md`.

This remains the default `counterfactual-completeness` admission policy. The explicitly weaker
`denotation-nonempty` pilot instead treats fresh `bird-set` correctness on a retained task as the
episode-level success condition and uses replay only for local action shaping. It neither requires
nor claims a privileged operation sequence, dependency completeness, or counterfactual causal
proof. The two admission policies have distinct CLI values and must not share result directories.

The current `denotation-nonempty` simple-process pilot additionally excludes whole training tasks when the hidden
reference query returns zero rows, a NULL scalar, or a numeric scalar zero on the source database.
This is a task-admission policy rather than a negative reward: excluded tasks are never sampled by
the actor and contribute neither positive nor negative gradients. The filter runs read-only before
rollout, records only result shape/classification plus a reference-query hash, and never exposes
the query or result to the model. Launchers must set `EXCLUDE_EMPTY_REFERENCE_RESULTS=1`; the
trainer repeats the check and writes `reference_result_filter.json` into the isolated run output.

## Invariants

- Training tasks only; held-out BIRD/Spider development labels never select RL examples.
- The actor sees only prefix-visible state and the latest structured tool error.
- Rejected actions spend the shared action budget and remain audit-only, never SFT targets.
- Terminal scoring must complete before an episode is marked legal.
- All active terminal, replay, and reward scoring uses `bird-set`.
- A missing legacy `answer` field is never interpreted as an authored empty answer; current
  terminal correctness comes from the cited evidence relation.
- Under `counterfactual-completeness`, correct trajectories update only after their immutable
  counterfactual suite passes. Under `denotation-nonempty`, they update only after the hidden
  reference-result filter retains the task and fresh replay is `bird-set` correct.
- API transport retries are client events, not semantic actions.
- SFT, evaluation, and RL share harness semantics but use the renderer/parser belonging to their
  explicitly recorded tool scheme.
