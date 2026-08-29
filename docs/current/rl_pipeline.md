# Reinforcement-learning pipeline

## Scope

The active RL system trains a policy over typed table tools in the same closed-loop environment
used for rollout and evaluation. Gold SQL is hidden from the actor and is used only by the harness
to score terminal denotation on training tasks.

RL uses the same concise rolling student runtime prompt as SFT export and tool evaluation. It never
loads the external teacher's generation guidance or worked examples. Checkpoint metadata binds the
protocol version/hash, student prompt SHA-256, and public tool-schema SHA-256 so resume cannot
silently cross a prompt contract.

### Current SFT-to-RL handoff

The current handoff is the exact Qwen3-8B Atomic version26 SFT1 `checkpoint-560`; see
`training_mainline.md`. It is the matched 54.63% BIRD-dev behavior anchor and must retain the same
version26 prompt, `think-json-v1` carrier, recent-four legal history, tool contract, runtime, and
terminal `bird-set` scorer throughout RL. The Qwen2.5 SFT2 line and later
`checkpoint-relalg/atomic-v24-frozen-v1` projections are identity-separated diagnostics and are not
initialization sources for this mainline.

The next sequence is:

1. verify the frozen version26 SFT1 adapter and reproduction identities;
2. run the isolated version26 local evaluator and RL environment without protocol drift;
3. establish a matched binary terminal `bird-set` result-only baseline;
4. only after that baseline passes engineering and behavior gates, compare grounded process credit.

`src/rl/tool_environment_v26.py` is the current SFT-to-RL environment. The version26 evaluator,
configs, reports, and checkpoint identities remain frozen, but they are now the executable mainline
rather than merely historical controls. `src/rl/tool_environment.py`, the Qwen2.5 line, and later
checkpoint-relalg environments remain separate controls and may contribute engineering patterns
only when those patterns do not alter version26 semantics.

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
- `src/rl/tool_environment_v26.py`: isolated current Qwen3/version26 SFT-to-RL environment.
- `src/rl/frameworks/accelerate/`: the frozen single-GPU QLoRA REINFORCE baseline. It remains an
  audit/equivalence reference and must not receive new optimizer features.
- `src/rl/diagnostics/analyze_grpo_training.py`: the canonical read-only online-GRPO training
  audit. It binds manifest identity, rollout groups, reconstructed population-standardized
  advantages, checkpoint Trainer metrics, partial next-update progress, failures, timeouts,
  explicit rollout-policy global/micro/synchronization steps, sampled action-signature changes,
  and final precision into one JSON schema. LoRA parameter/effective-matrix movement remains in
  the complementary `compare_lora_updates.py`; experiment-specific replacement analyzers should
  not be added.

Algorithm names in experiment reports follow the active loss, not the trainer class name. The
TRL class is named `TransitionGRPOTrainer`, but only result-only experiments use K-way
group-standardized reward advantages. Process experiments pass exact per-turn harness rewards as
advantages into the same tokenwise PPO-clipped surrogate; they use neither a learned value model
nor group-relative normalization. Rank-only experiments set the policy-loss coefficient to zero.
Exact-prefix Action-DPO and streaming OPD + repair DPO use separate standalone trainers and are
neither PPO nor GRPO. Frozen per-experiment settings and the Rank definition are recorded in
`docs/reports/rl/RL_METHODS_AND_PAPER_PROVENANCE_20260805_ZH.md`.

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

The retained binary baseline remains unchanged. A separate TRUST-SQL-style engineering baseline
may select `result_reward_profile=execution-ladder`:

```text
reward = 1.0  verified correct terminal denotation
         0.2  legal terminal answer with an executable cited relation but wrong denotation
         0.0  no legal terminal answer (protocol/execution abort, max steps, or runtime exclusion)
```

This is an environment-faithful translation, not a claim that an atomic relational program is the
same action space as direct SQL. It uses `policy_reduction=trajectory_token_mean`: exact rolling
prefixes remain separate, but a turn is weighted by its authored response-token count divided by
the trajectory's total authored response-token count. Summing the separately rebuilt causal turns
therefore matches TRUST-SQL's default per-sample token mean. The historical `trajectory_mean` mode
instead gives every turn equal mass and is not paper-equivalent when turn lengths differ.
The frozen 60-task hardware-equivalent configuration uses K=8, full-response tokens, temperature
0.8, constant LR 8e-7, AdamW betas (0.9, 0.98), weight decay 0.1, asymmetric clipping (0.2, 0.28),
no KL, and one policy iteration. On two 24GB RTX 3090s it accumulates 30 one-question K8 rollout
microbatches per optimizer step, giving 240 trajectories per update; six updates expose all 60
tasks for three passes. This substitutes gradient accumulation for TRUST-SQL's approximately
32-question/256-trajectory physical global batch while keeping the student unchanged inside each
effective batch.

The author-code comparison is pinned to TRUST-SQL commit
`89df0661ad6b8e29ed8e61f7c950fbc2c1678b08`. Its sentence-level GRPO path reshapes rewards by
prompt group, computes a masked population standard deviation (`unbiased=False`) and divides by
`std + 1e-6`; it does not apply the separate global advantage whitening used by its PPO branch.
Its policy-loss reducer first takes the authored-token mean within each complete response and then
aggregates samples. Identical-reward groups therefore have exactly zero group-relative advantage.
The local `trajectory_token_mean`, population-std normalization, and exact homogeneous-group zero
handling match these mathematical choices while retaining the atomic Harness's exact rolling
prefixes. The two-GPU QLoRA run remains an engineering translation of the training mechanics, not
a reproduction claim for the paper's model, direct-SQL action space, distributed async system, or
schema-reward condition.

Every new TRL run manifest records the SHA-256 of its initial adapter and the executable RL
sources that define rollout, reward, exact-prefix batching, reduction, precision, protocol, and
tool execution. Long-running artifacts also retain an `implementation_lock.json` plus a complete
source snapshot; the snapshot hashes must match the lock before a result is treated as
reproducible. This is necessary because the shared worktree may continue evolving while a
multi-hour GPU process is still using already-imported code.
The lock is written atomically before rollout and contains only identity-bearing deterministic
fields: executable source hashes, initial adapter hash, protocol version/hash, and experiment
configuration hash. A same-directory resume must match it byte-for-byte; it may not silently
rewrite the run identity.

For the policy-only, zero-KL baseline, transitions whose standardized advantage is exactly zero
are mathematically absent from the policy gradient. The trainer therefore drops them before the
old-policy forward and backward passes. A fully homogeneous K group retains one zero-gradient
placeholder so it still occupies its declared gradient-accumulation slot. This optimization is
used for online and fixed-pool runs only when Rank and KL losses are both disabled; rollout,
reward accounting, update boundaries, and the resulting optimizer state are unchanged.
The canonical GRPO analyzer reports positive, negative, and zero advantage counts plus raw and
turn-weighted mean advantage separately for every reward tier. This makes the relative-credit
direction auditable: reward `0` must never receive positive advantage and reward `1` must never
receive negative advantage, while reward `0.2` may legitimately take either sign depending on the
other seven outcomes for that prompt. It also retains a compact summary for every K group with the
exact task id, trajectory ids, reward/eligibility/advantage vectors, turn counts, timeout recovery,
and rollout-policy global/micro/synchronization steps, so aggregate checks remain traceable to the
actual sampled group.

The QLoRA numerical-precision contract is explicit: the frozen 4-bit base continues to compute in
BF16, but every trainable adapter parameter is promoted to FP32 before optimizer construction and
must remain FP32 after Trainer wrapping. With `adamw_torch`, Adam first/second moments must also be
FP32. The launcher records both audits in the run manifest and writes `training_precision.json`
before the final adapter is saved; a mismatch fails the run instead of silently accepting rounded
updates. A result-only artifact whose trainable adapter or Adam moments were BF16 is an engineering
smoke only and must not be resumed as the corrected baseline; the corrected run starts from the
frozen SFT2 adapter in a new output directory.
The canonical training analyzer additionally reads every persisted `checkpoint-N` adapter and
`optimizer.pt` with safe tensor / weights-only loading. It records adapter, optimizer-state, and
Adam-moment dtype counts per update; final readiness requires every saved LoRA tensor and every
`exp_avg`/`exp_avg_sq` tensor to be FP32, in addition to the live callback and final precision
audit. Thus an in-memory assertion alone cannot hide a lower-precision checkpoint.

The trainer streams dequantized-base-plus-LoRA BF16 weights to vLLM at every optimizer boundary.
Because the training forward still uses a 4-bit base, it applies TRL's capped vLLM importance
correction to the sampled policy loss. New runs record the supported-token absolute sampling/train
log-ratio, applied ratio min/mean/max, and cap-exceeded fraction. This distinguishes genuine reward
optimization from a hidden QLoRA-versus-vLLM policy mismatch. The currently running source-locked
formal artifact predates these descriptive metrics; its correction remains executable-source
locked, and its online step/sync binding is audited separately without changing that run in place.

Every model-authored SQLite tool call now has a mandatory 10-second execution deadline in the
shared atomic executor. SQLite's progress handler interrupts an overrun, the Harness relation
registry/counters are restored, newly created temporary tables are removed, and the model receives
a recoverable `timeout_error` with code `tool_execution_timeout`. The same bound is recorded in RL
rollout settings and used by full-dev tool evaluation. On 2026-08-11 this replaced an unbounded
scale-60 control attempt that stopped before its first optimizer update after 128 rollouts while a
`language_corpus` operation retained very large deleted SQLite temporary files. That partial
artifact is audit-only and must not be resumed. The corrected timeout10 run starts independently
from the exact SFT2 checkpoint in a new output directory.
The canonical training audit distinguishes timeout events from terminal outcomes: it reports
timeout-bearing trajectories, event count, correct/legal recovery, unrecovered trajectories,
missing structured events, and state-preservation violations. A timeout is not automatically a
zero-reward episode; if the model uses the recoverable error and later reaches the verified result,
the ordinary result-only reward still applies. Every structured timeout must preserve the exact
Harness state hash and assert `state_preserved=true`; the final GRPO readiness audit requires this
timeout contract to pass on every optimizer update.

The first FP32 timeout-corrected scale-60 launch on 2026-08-11 exposed a separate provenance
failure before its second update: online rollout used the repository's then-current atomic
`version39` (`protocol_hash=be4953f2c78db88f`), while its queued full-dev evaluator was the frozen
`version36` (`protocol_hash=20a8d3b4356d883c`). Its first update did contain valid learning signal
(23/30 heterogeneous K=8 groups, 76.7% nonzero-advantage trajectories, gradient norm 0.0751, and a
0.1678% raw adapter update relative to SFT2), but it is retained only as a protocol-mismatch
engineering diagnostic and is not a GRPO accuracy baseline. The queued learning-rate sweep was
stopped with it.

The strict replacement is identity-bound to atomic `version36` for both training and evaluation.
Its predeclared K=8 smoke must fail closed unless the run manifest and every rollout carry the
exact version/hash, the group has at least two distinct rewards, normalized advantages and gradient
norm are nonzero, and both raw adapter parameters and effective LoRA matrices move. The frozen
example-1188 smoke passed these checks with 3/8 correct, mean reward 0.50, 100% nonzero advantages,
gradient norm 0.7194, and raw/effective relative update norms 0.1678%/0.1595%. A baseline is not considered established
until the complete matched-protocol BIRD-dev1534 greedy evaluation also reports accuracy, legal
termination, mean steps, failure types, paired gains/regressions, and exact McNemar p versus the
same SFT2 protocol.

Final baseline disposition is produced by
`src/rl/diagnostics/audit_grpo_baseline_readiness.py`, which combines rather than replaces the
canonical training, LoRA-movement, and evaluation analyses. It reports separate statuses for:
an engineering-valid fully evaluated pipeline, proven parameter plus deterministic-policy change,
an observed positive accuracy baseline, and a statistically supported positive baseline requiring
exact paired McNemar `p<0.05`. Parallel strict fields additionally require no net legal regression.
A finite loss or nonzero adapter norm alone cannot satisfy the first status; a valid but
accuracy-regressing run cannot satisfy the positive statuses, and a small nonsignificant gain is
reported as exploratory rather than statistically supported. The active formal queue has an
independent read-only watcher that will create this readiness artifact after the final adapter and
all 1,534 dev records exist. The evaluation contract is exact-version-and-hash bound: the
candidate and every comparator must report `version36` and `20a8d3b4356d883c`, in addition to
greedy temperature/top-p and `bird-set`; a same-version result with another executable protocol
hash cannot satisfy readiness. The candidate evaluation must also carry an immutable
`evaluation_identity.json` whose adapter path and SHA-256 exactly match the final adapter used by
the LoRA-movement audit; a renamed, stale, or unbound partial result cannot satisfy readiness.

The first strict-version36 formal attempt was stopped before checkpoint 1 after an author-code
comparison found that its `trajectory_mean` reduction averaged turn means rather than all authored
tokens in the trajectory. Its K=8 smoke proved protocol, gradient, and FP32 update plumbing, but the
formal partial artifact is objective-mismatch audit data only. The corrected run starts again from
the frozen SFT2 adapter with `trajectory_token_mean`; no partial optimizer state is reused. Its
independent example-1188 K=8 smoke also passed: 3/8 correct, mean reward 0.50, 100% nonzero
advantages, gradient norm 0.7372, all 392 trainable tensors and all 784 Adam moments FP32, and
raw/effective relative update norms 0.1678%/0.1602%.

The corrected token-mean formal run then exposed a floating-point edge case before its first
optimizer update. Eight identical `0.2` rewards produced residual advantages of about
`2.78e-11` because summing decimal floats made the computed variance nonzero. The 24-row partial
artifact is audit-only: it contains zero optimizer steps and shows that 55/176 causal transitions
belonged to the theoretically homogeneous group. Group normalization now detects identical
eligible rewards before mean/std arithmetic and returns strict zeros; the policy-only, zero-KL
path reduces that group to one zero-gradient placeholder. Local and remote tests pass, and a fresh
same-seed hardware run reproduced the first three K8 action/reward sequences exactly while moving
from the homogeneous third group to the fourth rollout in under five seconds. The replacement
formal run again starts from frozen SFT2 in a new output directory.

That replacement subsequently exposed the mixed-group form of the same issue before checkpoint 1:
for rewards `[0,0,0.2,0.2,0,1,0,0.2]`, the three `0.2` outcomes equal the mathematical group mean,
but built-in Python summation reconstructed advantages near `8.8e-17`. TRUST-SQL's float32 tensor
mean gives exact zeros for those entries. The 160-row/20-group partial artifact has zero optimizer
steps and is audit-only. Result-only normalization now uses `math.fsum` for the discrete reward
mean and variance in dependency-light diagnostics, while the actual training and production audit
paths now execute the author's float32 Torch `mean` and population `std(unbiased=False)` operations
directly. It matches the author-code coefficient rounding exactly for all 42 heterogeneous K=8
reward-count compositions. The three homogeneous compositions are deliberately returned as strict
zero, preserving GRPO's mathematical relative signal instead of reproducing Torch's own decimal
`0.2` reduction residual. The formal baseline must restart from frozen SFT2 in a new output
directory and may not resume this partial artifact.

The canonical analyzer also counts any nonzero reconstructed advantage with magnitude below
`1e-12`. No K=8 composition of the `{0, 0.2, 1}` execution ladder has a legitimate signal at that
scale, so `exact_zero_contract_passes` must hold for every optimizer update. The readiness audit
fails closed if even one such numerical pseudo-signal appears.

Each persisted K group must contain one and only one `example_index`, and every complete
30-prompt/240-trajectory optimizer update must contain 30 distinct question groups. The analyzer
reports duplicate task ids explicitly, and final readiness rejects an update that silently repeats
a question inside its effective batch even if its row count is otherwise correct.

Credit-direction auditing separately checks that every heterogeneous group's normalized
advantages remain ordered by reward, that its lowest/highest tiers have the expected signs, and
that its group sum is numerically centered. Exhaustive production-Torch FP32 enumeration of all 42
heterogeneous K=8 count compositions gives a worst absolute sum residual of
`1.0728836059570312e-6`; the analyzer therefore uses a documented `2e-6` bound. This is not the
exact-zero rule above: homogeneous groups and mathematically mean-tier coefficients must still be
strict zero, and any ordering/sign violation still fails independently.

The corrected max-30 example-4129 K=8 hardware smoke then completed one real optimizer update. Its
reward counts were `0:3, 0.2:2, 1:3`; the eight advantages were approximately
`[-0.9401, +1.2719, -0.9401, +1.2719, -0.4977, -0.4977, -0.9401, +1.2719]`, with zero direction
violations and zero tiny pseudo-signals. It produced gradient norm `0.2570`; all 392 trainable
tensors and 784 Adam moments were FP32. The raw adapter moved by `0.1678%` of its reference norm
and the effective LoRA update moved by `0.1607%`. Requiring a stochastic live group to contain an
exactly zero advantage was rejected: this group mean was `0.425`, so none of the three reward tiers
mathematically had zero advantage. Exact-zero behavior remains covered deterministically by the
complete 45-composition K=8 unit test, while live smoke requires heterogeneous rewards, correct
credit direction, no tiny pseudo-signal, positive finite gradient, and positive LoRA movement.

TRL's `completion_mask` is always the generated-token attention mask. A separate `tool_mask`
controls loss support. This distinction is mandatory even for experiments that currently train the
full response: reusing the loss mask as attention would score the updated policy under a different
reasoning context than rollout and old-policy log probabilities.

## Process credit

`process_credit.py` reconstructs harness-owned data, value, and grounding dependencies by replaying
the actor's own trajectory. Model-authored reasoning and plan text never determine factual credit.
Local protocol, argument, and execution errors remain assigned to the event that caused them;
recovery credit is assigned to the later legal action that uses the feedback.

Version52-version53 native bundles additionally record `native_bundle_rl_statistics`: bundle-size and call
status histograms, later references to produced handles/steps, preservation of structured error
feedback, and the shape of the next-turn correction. These are harness-derived descriptive
statistics only. Observational calls are explicitly labeled `observational_credit_unresolved`; the
statistics do not become reward, do not retroactively make an errored bundle a positive target, and
do not flatten several calls from one provider turn into independent policy turns.

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
