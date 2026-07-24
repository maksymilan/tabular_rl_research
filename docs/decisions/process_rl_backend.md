# Process-RL backend decision

## Decision

The research condition is turn-local process RL. Result-only RL is retained only as the matched
coarse baseline.

The active implementation continues to use transition-level assistant-turn samples over the causal
table harness. It does not directly adopt a stock multi-turn GRPO trainer, because the inspected
frameworks do not preserve the project's objective without changing reward semantics:

```text
L = -(1/M) sum_t r_t log pi(a_t | s_t) + beta/M sum_t KL_t
```

Here `a_t` is the complete authored `think + tool_call` turn and `r_t` belongs only to that turn.
There is no implicit return propagation, group normalization, learned critic, or terminal-reward
broadcast in the process condition.

## Open-source implementation findings

The following conclusions come from source inspection, not feature-list comparison.

- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) has a mature Ray/vLLM multi-turn executor,
  QLoRA, action-token masks, and tokenwise KL machinery. Its current
  `MultiTurnAgentExecutor`, however, sums all environment step rewards into `total_reward`; the
  experience maker then places that scalar on the final action token and computes returns. Direct
  adoption would turn local process credit into cumulative/terminal credit.
- [veRL](https://github.com/verl-project/verl) has the best future distributed extension point:
  its trainer consumes `token_level_rewards`, its agent loop retains a response mask over generated
  versus environment tokens, and advantage estimators are registered functions. Its default
  `AgentLoopOutput` and reward managers still place one scalar at the final response token, so an
  exact process implementation needs a custom dense reward manager and a no-propagation advantage
  estimator.
- [TRL](https://github.com/huggingface/trl) has a useful tool loop, QLoRA-friendly trainer stack,
  prefix-preserving chat-template checks, per-token KL, and rollout importance-sampling support.
  Its public custom reward contract returns one float per completion. Subclassing enough of
  `GRPOTrainer` to accept one reward per assistant turn would be more invasive than the active
  backend and would still require changing GRPO's reward normalization and loss reduction.
- [Agent Lightning](https://github.com/microsoft/agent-lightning) validates the transition-level
  data model used here: every LLM response can be represented as its own prompt/response/reward
  triplet, with only the new response active in the loss. Its current veRL bridge explicitly
  propagates the same final scalar reward to all triplets, so it is an architectural reference
  rather than a drop-in process trainer.
- [Multi-Turn-RL-Agent](https://github.com/SiliangZeng/Multi-Turn-RL-Agent) demonstrates that
  turn-aware advantages can improve tool-use training. Its released MT-GRPO implementation applies
  group-normalized trajectory/turn scalars across token masks; it is not the same as accepting an
  independently grounded reward vector `(r_1, ..., r_T)`.

## Reused mechanisms

The active backend deliberately reuses the mature mechanisms that do preserve the objective:

1. Exact rollout prompt and response token IDs are retained for every assistant turn. Training does
   not re-render or re-tokenize a completed trajectory.
2. Environment/observation tokens are context only. The log probability is the sum over all tokens
   in the current authored assistant turn.
3. Fixed-reference KL is computed with a bounded k3 estimate per token and then summed within the
   turn. Applying k3 to a sequence-summed log-ratio is prohibited because it is numerically unstable
   on long actions.
4. The optimizer uses a configurable scheduler and warmup and saves adapter, optimizer, scheduler,
   Python RNG, Torch RNG, and CUDA RNG state together. Resume rejects changes to the task artifact,
   reward condition, protocol-relevant rollout settings, or optimizer settings.
5. The default process pilot uses learning rate `1e-6`, cosine scheduling, and 3% warmup. The
   result-only baseline must use the identical initialization and optimizer configuration.

## Required equivalence gates

Before a different training framework can replace the active backend, it must pass frozen-trace
tests showing equality of:

- assistant-turn token spans;
- `log pi(a_t | s_t)` for every turn;
- the exact step reward attached to each turn;
- process-update exclusion masks;
- fixed-reference per-turn KL;
- total loss and gradients, up to declared floating-point tolerance.

Framework convenience is not allowed to silently introduce future-return propagation, per-token
reward normalization, group centering, or final-reward broadcast.

## Remaining reward-side gate

Single-database `bird-set` correctness cannot detect denotation shortcuts. The next completeness
guard should use counterfactual database replay: run both hidden gold SQL and the actor's executable
tool trajectory on schema-valid perturbed databases and reject process updates when a perturbation
distinguishes them. This follows the semantic principle of
[SQL distilled test-suite evaluation](https://github.com/taoyds/test-suite-sql-eval), while replaying
the actor's tool program instead of requiring it to emit SQL. It is path-independent: equivalent
tool plans pass; only differing denotations fail.

External reviewer labels remain audit evidence only. They are not reward targets and are not used as
an online allow/deny list.

The replay and manifest boundary is now implemented in `src/rl/trajectory_replay.py` and
`src/rl/counterfactual_suite.py`. Database generation remains a separate offline stage. Inspection
and pilot execution of the released TestSuiteEval generator showed that its semantic approach is
appropriate, but the original implementation cannot be imported unchanged for BIRD: it assumes
away NULLs and some schema forms and does not robustly quote all BIRD identifiers. The BIRD
generator must therefore be adapted and independently audited while preserving the same
query-conditioned, multi-database equivalence principle.
