# Process-RL backend decision

## Decision

The research condition is turn-local process RL. Result-only RL is retained only as the matched
coarse baseline.

The active implementation continues to use transition-level assistant-turn samples over the causal
table harness. Framework version changes after the original backend audit make TRL the current
optimization backend, but not the owner of table-tool semantics. A thin adapter retains one exact
rolling-state prefix per assistant action and maps the verified transition reward onto TRL's
tokenwise clipped policy loss:

```text
rho_ti = pi_theta(a_ti | s_t, a_t,<i) / pi_old(a_ti | s_t, a_t,<i)
L = mean_ti[-min(rho_ti A_t, clip(rho_ti, 1-epsilon, 1+epsilon) A_t)
             + beta KL_ti]
```

Here `a_t` is the complete authored `think + tool_call` turn. In the process condition
`A_t = r_t`, and that verified reward belongs only to that turn.
There is no implicit return propagation, learned critic, or terminal-reward broadcast in the
process condition. Result-only retains trajectory-group normalization as its deliberately coarse
control. Process rewards remain turn-local.

The former `frameworks/accelerate/group_reinforce.py` is frozen as an audit reference. Its direct
Transformers generation, hand-written log-probability objective, single-GPU serialization, missing
old-policy ratio/clipping, and rollout/training distribution mismatch make it unsuitable as the
scaling backend.

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
- [TRL](https://github.com/huggingface/trl) now has an environment factory, custom rollout
  function, dedicated vLLM server mode, QLoRA-friendly trainer stack, dropout control, per-token
  old/reference log probabilities, clipping, KL, and rollout importance correction. Its public
  reward contract still returns one scalar per completion, so the project adapter flattens completed
  causal episodes into exact `(prefix, assistant action, r_t)` transitions before invoking the
  framework loss. It does not use TRL's native tool parser because that would replace the frozen
  raw-JSON action carrier.
- [Agent Lightning](https://github.com/microsoft/agent-lightning) validates the transition-level
  data model used here: every LLM response can be represented as its own prompt/response/reward
  triplet, with only the new response active in the loss. Its current veRL bridge explicitly
  propagates the same final scalar reward to all triplets, so it is an architectural reference
  rather than a drop-in process trainer.
- [Multi-Turn-RL-Agent](https://github.com/SiliangZeng/Multi-Turn-RL-Agent) demonstrates that
  turn-aware advantages can improve tool-use training. Its released MT-GRPO implementation applies
  group-normalized trajectory/turn scalars across token masks; it is not the same as accepting an
  independently grounded reward vector `(r_1, ..., r_T)`.

## Framework-owned mechanisms

The active backend delegates the following mature mechanisms to TRL:

1. Exact rollout prompt and response token IDs are retained for every assistant turn. Training does
   not re-render or re-tokenize a completed trajectory.
2. Environment/observation tokens are context only; only the current authored assistant response is
   active in the loss.
3. vLLM sampled log probabilities are retained, actor old probabilities are recomputed at the same
   temperature, and TRL applies bounded importance correction.
4. PPO ratios, clip `0.2`, fixed-SFT-reference KL, entropy/clip/KL metrics, gradient clipping,
   optimizer scheduling, and checkpoint state are framework-owned.
5. Dropout is disabled and a zero-transition batch aborts before any optimizer or scheduler step.
6. The default uses learning rate `3e-7`, constant scheduling, `beta=0.001`,
   `temperature=1.0`, and `top_p=1.0`. Result-only and process controls must share these settings.

Framework readiness and reward admission are separate decisions. The TRL backend is the active
engineering implementation, but process-RL optimization remains disabled for production until the
deterministic completeness and independent grounding edge-precision gates pass. Until then,
result-only is the only admissible optimizer smoke/control; process rewards remain audit-only.

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
