# Reinforcement-learning pipeline

## Scope

The active RL system trains a policy over typed table tools in the same closed-loop environment
used for rollout and evaluation. Gold SQL is hidden from the actor and is used only by the harness
to score terminal denotation on training tasks.

## Active modules

- `src/rl/tool_environment.py`: one causal model↔harness episode.
- `src/rl/task_loader.py`: model-visible task records plus hidden execution labels.
- `src/rl/terminal_reward.py`: the binary result-only control.
- `src/rl/process_credit.py`: replay-derived, step-local process credit.
- `src/rl/process_objective.py`: per-step policy objective.
- `src/rl/frameworks/accelerate/`: the supported single-GPU QLoRA training backend.

The historical Verl integration is archived under `archive/code/experimental_backends/verl/` and
is not a supported training entry point.

## Result-only control

The deliberately coarse baseline uses:

```text
reward = 1  if the terminal answer has the correct executed denotation
         0  otherwise
```

It has no process shaping, error penalty, length reward, or KL term. Homogeneous groups have zero
relative advantage and skip the optimizer update. Because the episode-level advantage is applied to
all generated turns, recovered successful trajectories can positively train earlier erroneous
turns; this behavior is retained only as the control condition.

## Process credit

`process_credit.py` reconstructs harness-owned data, value, and grounding dependencies by replaying
the actor's own trajectory. Model-authored reasoning and plan text never determine factual credit.
Local protocol, argument, and execution errors remain assigned to the event that caused them;
recovery credit is assigned to the later legal action that uses the feedback.

Before process credit is connected to an optimizer, both deterministic completeness and independent
edge-precision audits must pass. Versioned reports and frozen configs live in `docs/reports/rl/` and
`src/rl/configs/` respectively.

## Invariants

- Training tasks only; held-out BIRD/Spider development labels never select RL examples.
- The actor sees only prefix-visible state and the latest structured tool error.
- Rejected actions spend the shared action budget and remain audit-only, never SFT targets.
- Terminal scoring must complete before an episode is marked legal.
- API transport retries are client events, not semantic actions.
- SFT, evaluation, and RL share the same protocol renderer and harness semantics.
