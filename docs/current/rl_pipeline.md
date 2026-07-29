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
- `src/rl/frameworks/accelerate/`: the supported QLoRA transition-level training backend,
  including exact turn-token alignment, tokenwise fixed-reference KL, and resumable training state.

The historical Verl integration is archived under `archive/code/experimental_backends/verl/` and
is not a supported training entry point.

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

The active optimizer applies one scalar reward to the complete authored assistant turn and uses the
sum of its exact rollout-token log-probabilities as `log pi(action|state)`. It divides by the total
participating turn count `M`, not by episode or token count. Optional KL uses a frozen copy of the
SFT initialization adapter, computes a bounded k3 estimate per response token, and sums those
conditional KL terms within the turn. Result-only and process conditions share the same causal
environment and `bird-set` terminal scorer; reward mode is the controlled difference.

The default pilot optimizer uses learning rate `1e-6`, cosine scheduling, and 3% warmup. Each
checkpoint includes adapter, optimizer, scheduler, and Python/Torch/CUDA RNG state. Resume validates
the task artifact, reward condition, rollout settings, and optimizer settings before continuing.
The source-informed backend rationale and framework equivalence requirements are recorded in
`docs/decisions/process_rl_backend.md`.

`run_atomic_group_reinforce.sh` launches either condition. For a controlled ablation, pin the same
model/adapter, task-set artifact, group size, decoding settings, action/context budgets, learning
rate, update count, and KL coefficient; change only `REWARD_MODE` (and the process reward config
that is unused by the result-only control).

The backend also accepts `--tool-scheme atomic|action-block`. The action-block scheme currently
supports result-only RL. It deliberately rejects process mode because existing process credit is
atomic-action-local, while one action-block response may contain several successful, failed, and
blocked primitive calls. Assigning those local rewards to the whole authored block would violate
the credit boundary. Atomic process RL remains unchanged.

Process RL is the main research condition; result-only RL is only its matched baseline. Before the
process condition is launched, both deterministic completeness and independent edge-precision
audits must pass. Versioned reports and frozen configs live in `docs/reports/rl/` and
`src/rl/configs/` respectively.

Current gate status (2026-07-24): deterministic replay coverage passes. A trajectory-level re-audit
found three confirmed denotation shortcuts, six valid observation-to-literal dependencies omitted
by the former edge extractor, and five annotation/reviewer conflicts. The six dependencies are now
represented by exact visible-cell locators, but this changed grounding inference and therefore
requires a fresh independent edge-precision audit before process optimization is enabled.

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
inside the optimizer. The remaining release work is to re-audit the changed edge extractor and
build adequate BIRD suites for the three confirmed shortcut classes. See
`docs/reports/rl/GROUNDING_SHORTCUT_REAUDIT_20260724.md`.

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
