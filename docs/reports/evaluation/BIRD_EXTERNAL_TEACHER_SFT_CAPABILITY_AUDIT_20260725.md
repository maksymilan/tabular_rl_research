# BIRD external-teacher SFT capability audit

Date: 2026-07-25

## Question

This audit separates three effects that were initially conflated after the
fixed-1K external-teacher SFT:

1. loss of the Coder model's underlying SQL capability;
2. the Coder-specific canonical tool-carrier failure;
3. multi-step tool-policy and recovery failures after the carrier was repaired.

All new denotation results in this report use `bird-set`.

## Evaluated artifacts

- source direct-SQL control:
  `data/results/qwen2.5_coder_7b_bird_direct_sql_base_greedy_rp105_dev1534_bird_ex`;
- repaired Coder direct-SQL:
  `data/results/qwen2.5_coder_7b_external_teacher_carrier_repair_pilot64_direct_sql_greedy_rp105_dev1534_bird_ex`;
- repaired Coder greedy tool-use first 100:
  `data/results/diagnostics/qwen2.5_coder_7b_external_teacher_fixed1000_carrier_repair_pilot64_merged_tool100_greedy1`;
- General 7B greedy tool-use paired first 50:
  `data/results/diagnostics/qwen2.5_7b_external_teacher_fixed1000_tool100_greedy1`.

The direct-SQL pair uses the full 1,534-task BIRD dev split, greedy decoding,
`repetition_penalty=1.05`, `max_tokens=1024`, the same schema and external
knowledge, and the same evaluator. The tool runs use version24, rolling legal
history with four turns, the full resident prompt, greedy decoding, 30
semantic actions, and `max_tokens=1024`.

The General run was deliberately stopped after the paired first-50 gate. This
was enough to test whether the residual failure was Coder-specific; it is not
reported as a 100-task result.

## Full-dev direct-SQL retention

| Model | Correct | Accuracy | Delta |
| --- | ---: | ---: | ---: |
| Source Qwen2.5-Coder-7B-Instruct | 748/1,534 | 48.76% | — |
| External-teacher SFT plus carrier repair | 742/1,534 | 48.37% | -0.39 pp |

The pair has 100 gains and 106 regressions. The exact paired McNemar p-value
is 0.728. There is no evidence of a large overall loss of underlying SQL
capability.

The aggregate hides a real difficulty shift:

| Difficulty | Source | SFT | Delta | Gains / regressions | Exact p |
| --- | ---: | ---: | ---: | ---: | ---: |
| Simple | 527/925 (56.97%) | 538/925 (58.16%) | +1.19 pp | 56 / 45 | 0.320 |
| Moderate | 171/464 (36.85%) | 169/464 (36.42%) | -0.43 pp | 36 / 38 | 0.908 |
| Challenging | 50/145 (34.48%) | 35/145 (24.14%) | -10.34 pp | 8 / 23 | 0.0107 |

Output control also degraded. Incomplete `<answer>...</answer>` carriers rose
from 9 to 135 tasks. Median output length changed only from 160 to 181
characters, but p90 grew from 274 to 2,550 characters. On challenging tasks,
42/145 SFT outputs have an incomplete answer carrier versus 1/145 for the
source model. This explains a substantial part of the challenging-task loss:
the SFT policy learned to spend much more of the fixed output budget on
reasoning before emitting the graded result.

## Repaired Coder tool-use first-100 gate

On the exact first 100 BIRD-dev tasks:

| Mode | Correct |
| --- | ---: |
| Source Coder direct SQL | 28/100 |
| SFT Coder direct SQL | 18/100 |
| Repaired SFT Coder tool use | 29/100 |

Tool use has 14 gains and 13 regressions against source direct SQL on the same
tasks. It has 17 gains and 6 regressions against the SFT model's direct SQL.
The repaired tool model therefore does not show the originally suspected
semantic collapse on this gate. The original zero-legal result was a carrier
failure, not BIRD incapability.

Reliability is nevertheless poor:

| Outcome | Count |
| --- | ---: |
| Correct legal answer | 29 |
| Legal wrong answer | 20 |
| Execution-error terminal | 11 |
| Max steps | 20 |
| Protocol-error terminal | 11 |
| Context overflow | 9 |
| Legal terminal, including wrong | 49 |

There are 86 recoverable process-error events: 45 protocol errors and 41
execution errors. Twenty tasks encounter at least one missing or incomplete
tool-call carrier. Inspection shows that these are predominantly long,
length-truncated reasoning turns rather than recurrence of random Unicode:
many outputs end after roughly 4,000-5,000 characters without reaching the
tool call.

No-progress repetition is the larger trajectory failure:

- 35/100 tasks repeat an identical action consecutively;
- 22/100 repeat an identical action at least five times;
- the maximum identical-action run is 29;
- mean trajectory length is 11.71 actions although the median is 7.

The tool-error content is also concentrated rather than diffuse:

- 43 missing/incomplete carrier events across 20 tasks;
- 18 SQL syntax errors across 7 tasks;
- 12 invalid joined-logical-column errors across 6 tasks;
- 10 missing-column errors across 6 tasks;
- 2 malformed-JSON events;
- 1 scalar-grounding misuse.

## Same-recipe General versus Coder gate

The exact first 50 tasks compare the two models trained on the same 4,119
records with the same QLoRA recipe:

| Model | Correct | Legal terminal | Mean actions | Repeat-action tasks |
| --- | ---: | ---: | ---: | ---: |
| General Qwen2.5-7B | 11/50 | 27/50 | 12.12 | 19/50 |
| Repaired Qwen2.5-Coder-7B | 14/50 | 27/50 | 11.72 | 19/50 |

Coder has seven paired gains and four regressions versus General. The
post-repair trajectory problem is therefore not a Coder-only semantic
failure. The Coder-only part was the frozen output-head carrier defect.

## SFT-1 is the stronger counterexample

SFT-1 rules out two overly broad explanations. It was also produced by an
external teacher, and its training configuration was also `1e-4` for two
epochs. Off-policy teacher data and that optimizer schedule therefore cannot,
by themselves, explain the current failure.

The material differences are:

| Property | SFT-1 | Current external-teacher SFT |
| --- | ---: | ---: |
| Base model | Qwen2.5-7B-Instruct | Qwen2.5-7B-Instruct and Qwen2.5-Coder-7B-Instruct |
| Action records | 4,319 | 4,119 |
| Contributing episodes | 651 | 703 |
| Retained terminal targets | 651/651 (100%) | 550/703 (78.2%) |
| Feedback Recovery targets | 387 (9.0%) | 84 (2.0%) |
| System-prompt characters | 6,380 | 14,250 |
| Mean target characters | 508 | 767 |
| Mean reasoning characters | 342 | 593 |
| Encoded length p50 | 2,967 tokens | approximately 4,576-4,909 tokens by source lane |
| Encoded length p90 | 4,063 tokens | approximately 5,861-6,085 tokens by source lane |
| Exact cutoff retention | 4,319/4,319 | 4,119/4,782 |
| Learning rate / epochs | `1e-4` / 2 | `1e-4` / 2 |
| Selected checkpoint | epoch 1 after an epoch-1/2 gate | epoch 2, then Coder carrier repair |

SFT-1's accepted trajectories were substantially more useful for recovery
learning. They contained 441 protocol-error events, six argument errors, and
23 execution errors that were all followed by eventual success. The exported
dataset retained 387 explicit recovery targets. The current stronger teacher
and cleaner interface produced only 84 such targets. Cleaner teacher
trajectories are not automatically better student supervision: they omit the
states that the weaker student most needs to recover from.

The current model-visible contract is also much heavier. The system prompt is
2.23 times as long, mean reasoning is 73% longer, and a typical encoded record
is roughly 1.6 times as long. This makes the 6,400-token cutoff active in a way
it was not for SFT-1 and gives the model many more state/protocol tokens to
condition on before predicting one action. The additional 800 teacher
trajectories were generated under the version10 provider contract and later
rendered for training/evaluation with the version24 full prompt. The actions
remain replay-valid, but SFT-1 had tighter generation/training/evaluation
contract alignment.

The metric-independent first-50 trajectory behavior confirms the difference:

| Model | Legal terminal | Max-step terminals | Exact-action repeat >=5 |
| --- | ---: | ---: | ---: |
| SFT-1 epoch 1 | 33/50 | 2/50 | 2/50 |
| Current General 7B | 27/50 | 10/50 | 10/50 |
| Current repaired Coder 7B | 27/50 | 9/50 | 11/50 |

On the larger exact first-100 pair, SFT-1 has 67/100 legal terminals,
3/100 max-step terminals, and 3/100 tasks with an exact-action repeat of at
least five. Current repaired Coder has 49/100, 20/100, and 22/100
respectively. The trajectory-control regression is therefore not an
impression caused by the original carrier failure.

Historical SFT-1 correctness was recorded under the older evaluation
contract, so its correctness count is not mixed with the current `bird-set`
numbers here. Legal termination and exact-action repetition do not depend on
that denotation metric.

The carrier failure still has a Coder-specific component: SFT-1 used the
General Qwen2.5-7B base, whose frozen output head already had a usable prior
for the two dedicated carrier tokens. The Coder base did not. The current
General-vs-Coder paired gate shows that, after repairing those two output
rows, the residual loop/termination problem is shared by both models.

## Why the older SFT-2 did not fail in the same way

The older SFT-2 and the current external-teacher SFT are not comparable
training recipes.

| Property | Older SFT-2 | Current external-teacher SFT |
| --- | ---: | ---: |
| Total action records | 11,874 | 4,119 |
| Primary student-success records | 10,968 | 0 |
| Contributing successful episodes | 1,677 student-success episodes, plus fallback/demo lanes | 703 teacher episodes |
| Terminal targets in the primary success lane | 1,677/1,677 | 550/703 |
| Recorded recovery targets | 261 in the primary student lane | 84 |
| Learning rate | `5e-5` | `1e-4` |
| Epochs | 1 | 2 |

The current dataset is off-policy: DeepSeek produced successful trajectories
from states that DeepSeek visited. A 7B student makes different early
decisions and reaches states absent from that successful-teacher
distribution. In a multi-step environment this exposure mismatch compounds.
This cannot be the sole explanation because SFT-1 was also external-teacher
data. It becomes important in combination with the current corpus's much
lower recovery density, missing terminal targets, and heavier context.
SFT-2 was additionally dominated by causal student-success trajectories and
contained recovery/fallback lanes, so its state distribution was closer to
the trained model.

The exact 6,400-token gate creates an additional current-data defect. Of 703
episodes contributing at least one record, 153 (21.8%) have no retained
`answer_from_context` target. Earlier legal actions from those episodes were
kept after the late record was rejected. The resulting dataset teaches many
episode prefixes without teaching their termination. This is consistent with
the observed max-step and repeated-read loops.

Finally, token-averaged SFT gives the two carrier boundaries and the compact
JSON action very little weight relative to hundreds of reasoning tokens. On
Coder, whose frozen output head did not already assign useful probability to
the two dedicated boundary tokens, this produced the exact carrier defect.
The selective two-row carrier repair fixes that representation error, but it
does not fix off-policy state coverage or termination supervision.

## Conclusion

The tool semantics are not the cause of a large full-dev capability collapse.
After carrier repair, full-dev direct SQL is effectively unchanged and the
first-100 tool gate is one task above the matched source direct-SQL control.
However, the current SFT recipe is not suitable as the next production
recipe:

1. the Coder carrier was incompatible with a transformer-only frozen-head
   QLoRA;
2. the current protocol and targets are materially longer than SFT-1;
3. 21.8% of contributing episodes lack terminal supervision;
4. recovery supervision fell from 9.0% of SFT-1 targets to 2.0%;
5. successful external-teacher trajectories do not cover enough of the
   current student's visited error states;
6. unrestricted reasoning consumes the action carrier's generation budget;
7. epoch 2 was used without the checkpoint-selection gate that selected
   SFT-1 epoch 1;
8. challenging direct-SQL capability regressed significantly.

## Required next SFT gate

Keep the version24 tool semantics fixed and change the data/training boundary:

1. generate causal on-policy student prefixes and request teacher corrections
   only at the actual student states;
2. require episode-complete admission through one grounded terminal target;
   never keep an orphan prefix merely because its late terminal record exceeds
   the token budget;
3. add real recovery targets from student errors and no-progress states;
4. use a model-aware carrier loss or carrier adapter so boundary and action
   tokens cannot be drowned by reasoning loss;
5. restore checkpoint selection starting at epoch 1; test a lower learning
   rate as an ablation rather than treating it as the established cause;
6. gate checkpoints on carrier completeness, legal termination, repeat rate,
   `bird-set`, and direct-SQL challenging-task retention rather than training
   loss alone.
