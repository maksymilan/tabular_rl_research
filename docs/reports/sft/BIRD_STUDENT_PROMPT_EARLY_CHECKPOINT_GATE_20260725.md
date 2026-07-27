# BIRD student-prompt length audit and early-checkpoint gate

Date: 2026-07-25

## Outcome

There is no transferable fixed system-prompt length for tool agents. The
relevant boundary is whether the static student contract completely specifies
the model-authored action language; dynamic database schema, resident state,
and rolling history should be measured separately.

For this 12-tool, nested-JSON protocol, the current version25 rolling student
prompt is too structurally lean at 5,096 characters / 1,050 Qwen tokens. It is
much shorter than the previous version24 prompt (14,250 characters / 3,170
tokens) and shorter than the historical SFT prompt (6,380 characters / 1,492
tokens), but length alone is not the defect. The important deletion was the
compact canonical argument shape for high-entropy tools.

The next student prompt should target roughly 1,400--1,800 Qwen tokens for the
static contract. It should restore formal, compact JSON shapes for
`join_tables`, `condition_filter`, `project`, `group_aggregate`,
`scalar_compute`, and `answer_from_context`, while continuing to omit
teacher-only boundary-case advice and case-by-case solution recipes.
Repeating the low-entropy `plan` shape was subsequently falsified by the
matched causal gate: it induced plan calls far beyond their frequency in the
training targets.

## Related implementations

All comparisons below use the public implementation, not a third-party
summary. Character and word counts are local measurements of the checked-in
static prompt text; they are not presented as tokenizer-independent limits.

| Work | Training pipeline | Static system-contract pattern |
| --- | --- | --- |
| [AgentTuning](https://github.com/THUDM/AgentTuning) | Filters 1,866 multi-turn agent interactions, mixes them with general ShareGPT data, and performs ordinary multi-turn SFT. | A 49-character generic assistant system message. Tool semantics live in the interaction data and environment-specific conversation rather than one large static contract. |
| [ToolBench](https://github.com/OpenBMB/ToolBench) | Converts every legal trajectory prefix into a next-assistant-action SFT target; its published full-SFT launcher uses two epochs and learning rate `5e-5`. | A concise ReAct carrier plus a dynamically inserted function dictionary. The actionable schemas are supplied at inference time, so the short fixed prose is not the whole protocol. |
| [Agent Lightning SQL](https://github.com/microsoft/agent-lightning/tree/main/examples/spider) | Online write-SQL, execute, check, and rewrite loop; the published SQL example uses GRPO, validates before training, and saves/evaluates periodically. | Role prompts are 519, 964, and 587 characters, but database schema is dynamically appended and capped separately at 2,048 characters. |
| [SkySQL in verl-tool](https://github.com/TIGER-AI-Lab/verl-tool/tree/main/examples/data_preprocess/skysql) | Multi-turn GRPO with five rollouts, a five-turn tool budget, observation masking, validation before training, and periodic saving/testing. | About 2,740 characters / 404 words of static SQL-agent instructions, followed by dynamic schema, external knowledge, and question placeholders. |
| [verl-agent multi-turn SFT](https://github.com/langfengQ/verl-agent/blob/main/verl/utils/dataset/multiturn_sft_dataset.py) | Applies the model chat template to complete conversations and masks non-assistant spans; length overflow is an explicit data error or declared truncation policy. | Prompt length is controlled at the rendered-conversation level rather than by copying an RL/teacher prompt into every student example. |
| [ReTool](https://github.com/ReTool-RL/ReTool) | Uses synthetic tool traces for cold-start SFT and then outcome-based RL. | Supports the same separation used here: learn the carrier and basic tool policy with causal traces, then optimize task outcomes; it does not justify hiding the public action grammar from the student. |

The common pattern is separation, not minimal prose: static role/behavior
instructions remain compact, exact callable schemas are provided dynamically
or formally, and full rendered context is bounded independently. That makes a
teacher prompt and a student prompt intentionally different without making
the student's action language ambiguous.

## Controlled pilot

### Data construction

The pilot uses 400 complete causal episodes from the existing replay-verified
teacher dataset:

- fixed difficulty quotas: 160 easy, 120 medium, and 120 hard episodes;
- 2,521 next-action targets and 400/400 terminal targets;
- 60 feedback-recovery targets;
- every selected episode passed the source token gate;
- target assistant actions, reasoning, observations, and history are byte
  unchanged; only the `system` field is replaced;
- 2,521/2,521 targets pass the current version25 strict parser.

The builder rejects partially retained episodes instead of making an
apparently complete pilot from token-gate fragments. It also records the
source manifest and the prompt, protocol, and schema hashes.

Exact Qwen chat-template token audit:

| Rendered full-prefix statistic | Tokens |
| --- | ---: |
| Minimum | 1,244 |
| Median | 2,699 |
| p90 | 3,679 |
| Maximum | 4,267 |
| Cutoff | 6,400 |

All 2,521 targets fit; no record is truncated. This separates the system
contract question from the earlier 6,400-token filtering question.

### Training

The diagnostic run uses Qwen2.5-7B-Instruct with QLoRA:

- one epoch over 2,521 action targets, 158 optimizer steps;
- LoRA rank 16, alpha 32, dropout 0.05, all linear layers;
- effective batch size 16;
- learning rate `5e-5`, cosine schedule, 3% warmup;
- checkpoint every 10 optimizer steps;
- GPU 0 for training and GPU 1 for the matched early-checkpoint evaluations.

The training loss fell from 1.117 at step 2 to 0.679 at step 20 and 0.603 at
step 70 without NaN or an unstable gradient spike. Loss alone did not predict
usable closed-loop behavior.

### Matched early gate

All rows use the same first 20 BIRD-dev tasks, greedy decoding, one sample,
maximum 30 actions, rolling history of four turns, resident state, and
`bird-set`. “Legal” means a grounded terminal answer was reached, not merely
that one call parsed.

| Model / checkpoint | Prompt | Correct | Legal | Mean actions |
| --- | --- | ---: | ---: | ---: |
| Base Qwen2.5-7B | version25 lean | 0/20 | 0/20 | 3.00 |
| Pilot step 10 | version25 lean | 0/20 | 2/20 | 4.80 |
| Pilot step 20 | version25 lean | 0/20 | 1/20 | 10.30 |
| Pilot step 40 | version25 lean | 0/20 | 2/20 | 9.55 |
| Pilot step 70 | version25 lean | 4/20 | 13/20 | 9.70 |
| Pilot final step 158 | version25 lean | 2/20 | 9/20 | 10.10 |
| Existing full 1k SFT checkpoint, first 20 | historical prompt | 6/20 | 9/20 | 10.30 |
| Same existing full 1k SFT checkpoint | version25 lean | 1/20 | 2/20 | 9.45 |

The last two rows are a prompt-migration ablation, not a retraining comparison:
the model, tasks, decoding, and metric are fixed. Changing only the prompt
reduced correctness from 6/20 to 1/20, legal termination from 9/20 to 2/20,
and increased argument-error events from 3 to 40. Therefore the lean prompt
is not a lossless removal of teacher-only material.

Early checkpoints show the expected learning order. The base model emitted no
complete carrier on all 60 attempted turns. By step 20, protocol errors were
mostly removed, but the policy repeatedly explored invalid argument
structures or exhausted the action budget. At step 40, 37 of 39 join attempts
failed. Step 70 is the first useful checkpoint: accuracy reached 4/20 and legal
termination 13/20 after 44% of the epoch. The training therefore has a real
effect, but it has not yet matched the existing full SFT checkpoint's 6/20
accuracy under its historical prompt.

Step 70 still recorded 31 local error events: 11 argument-validation, 10
execution, and 10 protocol errors. `join_tables` caused 12 errors in 34
attempts (35.3%), including eight malformed arguments and four invalid logical
namespaces; `group_aggregate` caused three errors in eight attempts. Nine
episodes ended legally with the wrong answer, so carrier learning and legal
termination are now ahead of relational correctness.

Completing the epoch did not improve this cohort. Step 158 retained only task
6, gained task 14, and regressed tasks 0, 8, and 19 relative to step 70. Legal
termination fell from 13/20 to 9/20, while join errors rose to 15/28 attempts
and aggregate/scalar errors remained. The 20-task difference is not a
population estimate, but it is a sufficient early-stopping signal: lower
training loss does not justify choosing the final checkpoint.

These failures are not fixed by making the prose longer in general. They point
to missing formal examples of logical joined-column references, quoted columns
with spaces, preserved output fields, terminal table evidence, and
aggregate/scalar reference shapes.

## Decision

This pilot is diagnostic and is not promoted as a new SFT source or production
student prompt.

The current direction—short student contract, causal complete trajectories,
and early closed-loop gates—is correct. The 1,050-token implementation crossed
the useful compression boundary because it replaced exact nested syntax with
signatures and prose. The next change should add only compact formal action
shapes, aiming for 1,400--1,800 Qwen tokens, and first pass a same-model,
same-task prompt-only gate before another training run.

## Artifacts

- Builder: `src/sft/build_complete_prompt_pilot.py`
- Training config:
  `archive/experiments/sft/configs/bird_student_prompt_pilot_qwen25_7b_qlora_6400.yaml`
- Local data manifest:
  `data/sft/student_prompt_pilot_20260725/bird_student_prompt_complete400_qwen25_6400.manifest.json`
- Remote output:
  `/home/dengyan/tabular_rl_outputs/checkpoints/qwen2.5-7b-bird-student-prompt-complete400-6400-qlora`
- Result directories:
  `data/results/diagnostics/qwen2.5_7b_v25_lean_prompt_*`
