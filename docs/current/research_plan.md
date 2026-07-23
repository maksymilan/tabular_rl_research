# BIRD Two-Stage SFT and RL Plan

Status: current design and experiment SSOT, 2026-07-14.

This document records the decisions reached after the BIRD environment setup, the fixed 30-task
teacher pilot, the switch to single-step state-only SFT records, and the correction/recovery
terminology cleanup. It supersedes the BIRD-related planning assumptions in older Spider data plans;
those older documents remain historical records.

## 1. Research and data boundary

- BIRD **train only** is the primary source for large-database SFT/RL data construction.
- BIRD dev and Spider2 gold information must never enter trajectory generation, prompts, repair, or
  reward tuning.
- Spider2-Snow is the frozen external held-out benchmark. Spider1 remains useful for harness/protocol
  regression and historical comparison, but is not the main exploration-training source.
- BIRD gold SQL is harness-only. It may establish tool compatibility and score final denotations on
  train tasks, but neither the teacher nor the student may see gold SQL/results/answers.
- Accepted training trajectories must be legal, causally rendered, replayable, and terminally
  execution-correct.

## 2. Three phases and fixed terminology

There are only three sequential training phases:

1. **SFT-1: Teacher Demonstration SFT**
2. **SFT-2: On-policy Refinement SFT**
3. **RL: Policy Optimization**

The following names describe data types, not additional sequential stages.

| Symbol | Fixed name | Definition |
|---|---|---|
| `D_demo` | Teacher Demonstration | A legal external-teacher action generated from the causal current state. |
| `D_success` | On-policy Student Success | A legal action from the SFT-1 student's verifier-correct train rollout. |
| `D_correct` | Decision Correction | From state `S_t` before a diagnosed bad action, supervise a better action instead. |
| `D_recover` | Feedback Recovery | From the actual state after an error or informative observation, supervise the next recovery action. |

For a student transition

```text
S_t --a_bad--> observation/error --> S_(t+1)
```

- **Decision Correction** uses `S_t -> a_better`: it teaches how to avoid the bad decision.
- **Feedback Recovery** uses `(S_(t+1), feedback) -> a_recover`: it teaches how to continue after
  the consequence is already present.

A legal-but-wrong terminal trajectory without any model-visible intermediate warning can provide a
Decision Correction candidate through privileged offline diagnosis. It is not genuine Feedback
Recovery unless the training environment exposes terminal incorrectness and permits another action.
Offline diagnostic text, gold SQL, and future observations never enter the actor's state or SFT
record.

The full second-stage mixture is conceptually:

```text
D_SFT2 = D_demo_replay + D_success + D_correct + D_recover
```

The replay slice prevents tool-protocol forgetting. Mixture weights must be reported; recovery and
correction must not be silently drowned by routine describe/plan actions.

## 3. Single-step state-only SFT contract

The canonical source artifact remains a complete closed-loop episode:

```text
S_0 --a_0/o_0--> S_1 --a_1/o_1--> ... --a_T/final--> terminal
```

SFT export changes only the sample boundary. A T-action accepted episode yields T independent
standard ShareGPT records:

```text
(system, catalog, question, optional evidence, S_t, optional LAST TOOL ERROR) -> a_t
```

Each record contains exactly one human/gpt pair. It uses the same shared renderer as online eval/RL
and contains no previous assistant transcript, raw observation history, current action output, or
future step id. The harness-managed `EnvironmentState` is the sole resident task context.

This remains LLaMA-Factory-compatible ShareGPT; no custom trainer or loss is required. The format
change is record granularity, not tool-call syntax. The current exporter and index are
The completed SFT-1 exporter is archived at
`archive/experiments/sft/utilities/build_bird_sft1_steps.py`; it checks replay, state-before identity, future references,
current-output leakage, gold-SQL leakage, one-pair structure, and renderer equality.

Failed actions are retained in the canonical audit event log but are never SFT labels. The first
subsequent legal action may be exported with `feedback_recovery=true` and its error type. Origin and
transition type are orthogonal: a teacher episode may contain normal demonstration steps and teacher
feedback-recovery steps; the index must keep them separable. Before full SFT-1, compare or explicitly
choose `clean teacher steps only` versus `clean + teacher-recovery steps` rather than silently mixing
them.

## 4. BIRD environment and fixed pilot cohort

Current train-side support:

- 6,601 filtered BIRD train annotations across 69 databases;
- 69/69 local SQLite databases available, 32.4 GB total;
- 69/69 harness tool smoke passes, covering 532 tables;
- 5,915/6,601 tasks pass the adapter-owned SQL→tool-plan→gold-denotation compatibility gate
  under the five-second per-task limit.

The fixed pilot is `data/eval_inputs/bird_train_sft1_pilot30.jsonl`:

- seed 42;
- 30 distinct databases;
- 12 easy, 9 medium, 9 hard;
- difficulty is a selection-only SQL-structure proxy because official BIRD train has no difficulty
  field; labels are never model-visible.

The cohort must remain fixed for controlled generator/parser comparisons. A failed task must not be
silently replaced to make the success rate look better.

## 5. Pilot evidence so far

### 5.1 Early teacher/continuation pilot

The historical continuation run produced 4/30 replay-correct episodes and 35 one-step SFT records.
The 35-record LLaMA-Factory preprocessing smoke passed: one human/gpt pair, masked human labels,
decoded targets matching the source, and no record above the 4,096-token cutoff. This established
framework compatibility but not adequate data yield.

### 5.2 Strict same-episode recovery semantics

Active paths keep strict generated-action parsing. Recoverable protocol/argument/state-preserving
execution errors:

- consume the common action budget;
- leave factual state unchanged;
- appear as structured `LAST TOOL ERROR` in the next state-only input;
- are capped per error type;
- remain audit events and never become labels.

API transport retries are client mechanics, not semantic recovery. Whole-episode restarts are
separate attempts and must be reported as success@k rather than overwriting success@1.

### 5.3 `gpt-5.6-sol` pilot-only runner: `chatgpt_v1`

Result: **6/30 = 20%**, all six easy and all `recovered_success`; no clean success.

Important diagnosis:

- 24 terminal failures: 21 protocol, 2 execution, 1 state-changing nonrecoverable execution error;
- 85 total error events, 77 protocol errors;
- 74/77 protocol-error outputs contained 2--11 complete `<think>`/`<tool_call>` pairs;
- only three protocol-error outputs contained no complete pair;
- only six episodes reached `answer_from_context`, and all six were verifier-correct;
- accepted output: 6 episodes, 36 legal steps, including 10 feedback-recovery steps.

Therefore 20% is primarily the yield of a strict one-action parser applied to a teacher that often
emits an open-loop multi-action suffix. It is not a defensible estimate of BIRD semantic solvability
or final data acquisition cost.

### 5.4 Canonical generator: `bird_chatgpt_original_v1`

The canonical `src/sft/generate_teacher_rollouts.py` run on the same fixed cohort strengthened the
one-request/one-action prompt while keeping strict no-repair parsing.

Result: **16/30 = 53.33%** replay-verified success:

| Difficulty | Success | Rate | Accepted legal steps |
|---|---:|---:|---:|
| easy | 10/12 | 83.33% | 72 |
| medium | 5/9 | 55.56% | 48 |
| hard | 1/9 | 11.11% | 14 |
| total | 16/30 | 53.33% | 134 |

Of the 16 accepted episodes, five are clean and eleven recovered; 13 accepted actions are marked as
feedback recovery. The 14 terminal failures are nine wrong answers, four execution errors, and one
protocol error. Across all episodes there are 40 retained recoverable error events: 26 protocol and
14 execution. Multiple-complete-call turns fell from 77 in `chatgpt_v1` to 25, but remain a live
format bottleneck.

This 53.33% is the current closed-loop baseline, not a final large-scale yield estimate. Difficulty
skew is severe: hard tasks require targeted failure analysis before scale-up.

This run used the v2h protocol. The subsequent first-edge join-name repair produced the contract now
called `version1`; older v2h artifacts are historical controls and must not be mixed into final
version1 training data. In particular, 16/30 cannot serve as the sole control for a version1
teacher adapter because both the tool contract and teacher delivery would differ.

## 6. Immediate controlled experiment

Use the exact same 30 tasks throughout. First establish a fresh **version1 strict-parser baseline** with
the current join contract. Then run a second version1 arm with no changes to model, temperature, task
order, max steps, tool budget, verifier, or student/runtime parser, changing only the teacher
action-delivery mechanism.

Preferred order:

1. Native structured/tool calling with exactly one non-parallel tool action, if the configured API
   supports it reliably.
2. Otherwise, a **teacher-only first-complete-action adapter**: execute and store only the first
   complete causal think/tool-call pair, record how many extra raw calls were discarded, update the
   real environment, and request a fresh next action.

The strict parser remains the evaluation and student/runtime contract. A teacher adapter is a data
generation boundary, not permission to silently relax model evaluation. Unexecuted suffix calls
must never enter state, targets, or later prompts.

Report separately:

- raw single-action format compliance;
- adapted first-action parse/argument validity;
- clean terminal success;
- recovered terminal success;
- terminal success by difficulty;
- error-conditioned recovery rate;
- wrong-answer, execution, protocol, max-step, and API failure buckets;
- total accepted episodes and single-step targets;
- API/token/time cost.

Compare the two version1 arms directly. Keep the v2h 16/30 result as historical context, not as the sole
causal control. Do not extrapolate to the 5,915-task pool until this paired experiment is complete.

## 7. Next execution plan

### P0: close the 30-task pilot

1. Preserve the v2h canonical 16 successes as a historical result; do not mix them into final version1
   training data.
2. Run a strict version1 baseline on the same 30, then run the teacher-only structured/first-action arm
   from the same version1 code and configuration.
3. Export the selected version1 arm's successes to single-step SFT and run replay/no-leak/LLaMA-Factory
   preprocessing checks.
4. Manually audit the v2h four execution failures and a stratified sample of nine wrong answers,
   especially the eight failed hard tasks.
5. Verify that invalid plan ops, invalid handles/columns, and state-preserving execution failures are
   categorized as recoverable feedback where safe.
6. Keep raw teacher format compliance separate from adapted data-generation success.

### P1: estimate acquisition yield

After P0 freezes the teacher interface, run a larger fixed stratified pilot. Report episode-level and
step-level yield independently; the 4:3:3 ratio applies to selected questions, not the number of SFT
records. Preserve first-attempt outcomes and report any whole-episode restart as success@k.

Use database-diverse BIRD-train tasks only. Do not tune on BIRD dev or Spider2.

### P2: build and train SFT-1

Generate replay-verified teacher episodes, export single-step records, balance routine versus
feedback-conditioned steps, and train the tool-use bootstrap. Evaluate on an internal train-side
development split without contaminating held-out benchmarks.

### P3: build SFT-2 on the student's state distribution

Run SFT-1 on BIRD train:

- verifier-correct rollouts provide `D_success`;
- a privileged offline diagnoser may locate a suspect step but its output never enters training;
- a causal teacher actor receives only the selected state/prefix and produces `D_correct` or
  continues from real feedback to produce `D_recover`;
- all repaired suffixes are regenerated in the live environment and terminally verified;
- failed student/teacher actions remain context/audit events, never labels.

Train independent ablations from the same SFT-1 checkpoint:

| ID | Training data |
|---|---|
| A | SFT-1 Teacher Demonstration |
| B | A + On-policy Student Success |
| C | B + Decision Correction |
| D | B + Feedback Recovery |
| E | B + Correction + Recovery = full SFT-2 |

C, D, and E are branches, not a sequential C→D→E run.

### P4: RL comparison

Start both RL arms from the identical full SFT-2 checkpoint E:

- **Outcome-only RL**: reward is terminal denotation correctness only;
- **Process-shaped RL**: correctness remains dominant, with auditable exploration/feedback signals.

The main comparison must attribute gains to process reward rather than a different initialization,
task filter, rollout budget, or parser.

## 8. Acceptance gates before scale-up

- fixed task cohort and deterministic selection manifest;
- no BIRD dev/Spider2/gold information in model-visible prompts;
- exact shared online/SFT renderer;
- one action executed per environment transition;
- state-before contains no current/future output or id;
- every accepted episode replay- and denotation-correct;
- error actions excluded from SFT labels;
- clean, recovery, correction, and origin metadata kept separately;
- LLaMA-Factory preprocessing confirms prompt tokens are masked and decoded labels equal targets;
- episode-level success and step-record yield both reported by difficulty/tool/error type;
- raw protocol compliance never conflated with semantic task success.
