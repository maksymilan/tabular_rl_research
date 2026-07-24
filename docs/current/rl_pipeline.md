# Reinforcement-learning pipeline

## Scope

The active RL system trains a policy over typed table tools in the same closed-loop environment
used for rollout and evaluation. Gold SQL is hidden from the actor and is used only by the harness
to score terminal denotation on training tasks.

## Active modules

- `src/rl/tool_environment.py`: one causal model↔harness episode.
- `src/rl/task_loader.py`: model-visible task records plus hidden execution labels.
- `src/rl/terminal_reward.py`: the binary result-only control.
- `src/rl/target_support.py`: bounded hidden gold T/C/R support used only on the reward side.
- `src/rl/task_support.py`: deterministic model-visible literal support and canonical rewrites.
- `src/rl/process_credit.py`: replay-derived, step-local process credit.
- `src/rl/process_objective.py`: per-step policy objective.
- `src/rl/build_sft_task_set.py`: exact BIRD task cohort from the retained SFT index.
- `src/rl/audit_sft_process_rewards.py`: exact SFT-index reward coverage and one-factor ablations.
- `src/rl/audit_reward_sensitivity.py`: matched success/failure coefficient sensitivity scan.
- `src/rl/review_grounding_edges_external.py`: strict external grounding/dependency packages and
  reviews.
- `src/rl/select_grounding_recheck.py` and `src/rl/summarize_grounding_external_audit.py`:
  deterministic second-opinion selection and consensus aggregation.
- `src/rl/frameworks/accelerate/`: the supported single-GPU QLoRA training backend.

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
sum of its token log-probabilities as `log pi(action|state)`. It divides by the total participating
turn count `M`, not by episode count. Optional KL uses a frozen copy of the SFT initialization
adapter. Result-only and process conditions share the same causal environment and `bird-set`
terminal scorer; reward mode is the controlled difference.

`run_atomic_group_reinforce.sh` launches either condition. For a controlled ablation, pin the same
model/adapter, task-set artifact, group size, decoding settings, action/context budgets, learning
rate, update count, and KL coefficient; change only `REWARD_MODE` (and the process reward config
that is unused by the result-only control).

Before process credit is connected to an optimizer, both deterministic completeness and independent
edge-precision audits must pass. Versioned reports and frozen configs live in `docs/reports/rl/` and
`src/rl/configs/` respectively.

Current gate status (2026-07-24): deterministic replay coverage and independent grounding-edge
precision pass, but independent dependency completeness fails on confirmed denotation-shortcut
trajectories. Result-only RL may continue as the control; process-RL optimization remains disabled.
See `docs/reports/rl/ATOMIC_PROCESS_REWARD_PRECISION_SENSITIVITY_AUDIT_20260724.md`.

## Invariants

- Training tasks only; held-out BIRD/Spider development labels never select RL examples.
- The actor sees only prefix-visible state and the latest structured tool error.
- Rejected actions spend the shared action budget and remain audit-only, never SFT targets.
- Terminal scoring must complete before an episode is marked legal.
- All active terminal, replay, and reward scoring uses `bird-set`.
- API transport retries are client events, not semantic actions.
- SFT, evaluation, and RL share the same protocol renderer and harness semantics.
