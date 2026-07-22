# BIRD Tool-Use SFT/RL Experiment Summary

Updated: 2026-07-22 17:10 CST

## 1. Research setup

The actor is Qwen2.5-7B-Instruct and interacts with BIRD databases through the typed table-tool
harness. New training trajectories are generated causally in a real model-harness loop. Gold SQL is
hidden from the actor and is used only by the harness for compatibility and final-denotation checks.
Rejected actions remain in audit logs but are never SFT targets.

The current model-visible protocol is `v2i-state-only-join-feedback-r2`. Evaluation results must be
separated by denotation metric:

| Metric | Semantics | Use |
|---|---|---|
| `bird-set` | BIRD reference EX set equality; ignores row order and duplicate multiplicity | Current held-out BIRD-dev reporting |
| `strict-multiset` | Normalized multiset equality | Historical tool evaluation, train replay, process-credit audits |

Results under the two metrics are not directly comparable.

## 2. Training data by stage

| Stage | Initialization | Data source | Raw generation | Accepted training data | Recovery/correction content |
|---|---|---|---|---|---|
| SFT-1 | Base Qwen2.5-7B-Instruct | External teacher in the real BIRD-train harness; verifier-correct, replay-correct and grounded trajectories only | Multiple teacher scale batches | 651 unique episodes; 4,319 single-action targets | 387 Feedback Recovery targets |
| SFT-2 | SFT-1 `checkpoint-270` | SFT-1 student K=4 rollout on 1,000 disjoint BIRD-train tasks, then Flash fallback only on pass@4 failures | 4,000 student episodes; 1,745 verifier-correct; 368 failed tasks sent to Flash | 11,874 targets total | 318 Feedback Recovery targets and 2 Decision Correction targets |
| Result-only RL control | SFT-2 `checkpoint-743` | Online K=4 rollout over 70 mixed-reward BIRD-train task groups | 70 groups / 280 episodes | 70 optimizer groups; only 28 had nonzero relative advantage | Terminal reward only; erroneous turns inside positive recovered episodes can receive positive group credit |
| Process-reward RL | SFT-2 `checkpoint-743` planned | Replay-derived grounded process credit | Not trained | Not available | Blocked pending deterministic-completeness and grounding-edge precision gates |

### 2.1 SFT-1 composition and training

| Item | Value |
|---|---:|
| Accepted episodes | 651 |
| Single-action targets | 4,319 |
| Easy / medium / hard targets | 1,808 / 1,481 / 1,030 |
| Feedback Recovery targets | 387 |
| Token cutoff retention | 4,319 / 4,319 |
| Encoded length p50 / p90 / max | 2,967 / 4,063 / 6,400 |
| Training | QLoRA, 2 epochs, 540 optimizer steps, 9:59:05 |
| Final epoch-2 train loss | 0.4546 |
| Selected model | Epoch 1, `checkpoint-270` |

Epoch 1 was selected because it scored 8/30 on the checkpoint gate versus 7/30 for epoch 2. This
small gate selects a checkpoint; it is not a scaling claim.

### 2.2 SFT-2 construction funnel

The 1,000 student tasks use a 400/300/300 easy/medium/hard proxy distribution and are disjoint from
the 651 SFT-1 teacher-success examples.

| Student rollout outcome | Count |
|---|---:|
| Total sampled episodes | 4,000 |
| Verifier-correct episodes | 1,745 |
| Legal episodes | 3,647 |
| Wrong answer | 1,902 |
| Execution error | 261 |
| Protocol error | 35 |
| Context overflow | 30 |
| Argument-validation error | 21 |
| Max steps | 4 |
| Nonrecoverable execution error | 2 |

Fresh replay and quality gates retained 1,677 student-success episodes and exported 10,968 targets.
The final deduplicated SFT-2 mixture is:

| SFT-2 lane | Targets | Share |
|---|---:|---:|
| Student success, normal | 10,707 | 90.17% |
| Student Feedback Recovery | 261 | 2.20% |
| SFT-1 demonstration replay | 650 | 5.47% |
| Flash teacher fallback | 254 | 2.14% |
| Decision Correction | 2 | 0.02% |
| **Total** | **11,874** | **100%** |

Across all lanes there are 318 Feedback Recovery targets: 261 from the SFT-1 student, 18 from
Flash fallback trajectories, and 39 from replayed SFT-1 demonstrations. The 261 student recoveries
occur in 231 ultimately successful episodes. Flash produced 38 raw verifier successes on 368
pass@4-failed tasks; quality gates retained 35 episodes, represented by 254 nonduplicate targets.
Only two examples establish a deterministic same-state student-bad-action to teacher-good-action
Decision Correction pair. Thus the corpus is success/recovery rich but direct-correction poor.

### 2.3 SFT-2 training

| Item | Value |
|---|---:|
| Initialization | SFT-1 `checkpoint-270` |
| Training records | 11,874 |
| Training | QLoRA, 1 epoch, 743 optimizer steps, 13:03:13 |
| Final train loss | 0.2791 |
| First / last ten-window mean loss | 0.2912 / 0.2728 |
| Selected model | `checkpoint-743` |

There was no held-out loss because the training run intentionally used `eval_strategy: no`.

### 2.4 Result-only RL training

| Item | Value |
|---|---:|
| Initialization | SFT-2 `checkpoint-743` |
| Tasks / samples per task | 70 / 4 |
| Difficulty | 22 easy / 20 medium / 28 hard |
| Episodes | 280 |
| Correct terminal episodes | 73 |
| Wrong valid terminals | 67 |
| No valid terminal | 140 |
| Correct samples per group | `{0:36, 1:13, 2:9, 3:6, 4:6}` |
| Effective nonzero-advantage updates | 28 / 70 |
| Runtime | 14.73 hours |
| Evaluated checkpoint | `checkpoint-70` |

The earlier short-budget RL run is invalid and excluded: 224/280 episodes ended in terminal
protocol errors. In the valid retry, 15 positive episodes still contain 17 erroneous turns. This is
an expected weakness of the coarse full-turn result-only control, not the intended process-credit
algorithm.

## 3. Current BIRD reference-EX results (`bird-set`)

All K=4 rows use temperature 0.7, top-p 0.95. Tool models use rolling-4/full/resident state, at most
30 tool steps, 1,024 output tokens, and BIRD external knowledge.

| Model | Evaluation status | Tasks | pass@1 | pass@2 | pass@4 |
|---|---|---:|---:|---:|---:|
| Base 7B Direct SQL, greedy | Complete | 1,534 | **650/1534 = 42.37%** | - | - |
| Base 7B Direct SQL, sampled K=4 | Complete | 1,534 | **624/1534 = 40.68%** | **713/1534 = 46.48%** | **785/1534 = 51.17%** |
| SFT-2 `checkpoint-743`, sampled K=4 | Stopped, resumable | 194/1,534 | **55/194 = 28.35%** | **71/194 = 36.60%** | **90/194 = 46.39%** |
| Result-only RL `checkpoint-70`, sampled K=4 | Active snapshot | 1,229/1,534 | **468/1229 = 38.08%** | **589/1229 = 47.93%** | **664/1229 = 54.03%** |

The partial rows cannot be compared to full-dev percentages because BIRD-dev is ordered and its
early prefix is substantially harder. The fair same-task comparisons at this snapshot are:

| Same-task comparison | Tasks | pass@1 delta | pass@2 delta | pass@4 delta |
|---|---:|---:|---:|---:|
| RL minus Direct SQL | 1,229 | **-2.60 pp** | **+1.87 pp** | **+3.17 pp** |
| SFT-2 minus Direct SQL | 194 | **+5.67 pp** | **+7.73 pp** | **+12.89 pp** |
| RL minus SFT-2 | 194 | **-2.58 pp** | **-2.58 pp** | **-4.64 pp** |

The RL snapshot suggests a possible pass@2/pass@4 diversity gain over Direct SQL, but no current
evidence of improvement over SFT-2: on the only 194 directly comparable BIRD-EX tasks, RL is lower
at all k. Both tool rows are fresh stochastic generations, so the 194-task difference includes
sampling variance. A final SFT-2/RL claim requires both complete 1,534-task BIRD-EX runs.

Direct-SQL reference-EX difficulty results are available now:

| Decode | Simple (925) | Moderate (464) | Challenging (145) |
|---|---:|---:|---:|
| Greedy | 51.14% | 31.47% | 21.38% |
| K=4 pass@1 | 49.73% | 28.45% | 22.07% |
| K=4 pass@2 | 55.68% | 33.84% | 28.28% |
| K=4 pass@4 | 59.68% | 39.66% | 33.79% |

## 4. Completed historical internal results (`strict-multiset`)

These results are useful for stage diagnosis but must not be placed in the same numeric comparison
as `bird-set` results.

| Model | Tasks | Decode | pass@1 / accuracy | pass@2 | pass@4 | Status |
|---|---:|---|---:|---:|---:|---|
| Base 7B Direct SQL | 1,534 | Greedy | 598/1534 = 38.98% | - | - | Complete |
| Base 7B Direct SQL | 1,534 | Sampled K=4 | 571/1534 = 37.22% | 648/1534 = 42.24% | 715/1534 = 46.61% | Complete |
| SFT-1 `checkpoint-270` | 1,534 | Greedy tool rollout | 563/1534 = 36.70% | - | - | Complete |
| SFT-1 `checkpoint-270` | 1,064 | Sampled K=4 | 379/1064 = 35.62% | 485/1064 = 45.58% | 580/1064 = 54.51% | Incomplete; tunnel loss |
| SFT-2 `checkpoint-743` | 1,534 | Sampled K=4 | 595/1534 = 38.79% | 719/1534 = 46.87% | 855/1534 = 55.74% | Complete |

Under this historical metric, SFT-2 minus sampled Direct SQL on all 1,534 tasks is +1.56, +4.63,
and +9.13 percentage points at k=1/2/4. The paired gains are +24/+71/+140 tasks, with exact McNemar
`p=0.259`, `0.00073`, and `5.37e-11`; only k=2 and k=4 are statistically supported. On the 1,064
tasks shared with the interrupted SFT-1 K=4 run, SFT-2 versus SFT-1 is 408/485/574 versus
379/485/580: SFT-2 improves pass@1, ties pass@2, and does not raise the observed pass@4 ceiling.

## 5. Conclusions for reporting

1. The causal tool-use pipeline is operational end to end: external-teacher SFT-1, on-policy
   student-first SFT-2, and a result-only RL control all train and execute on held-out BIRD-dev.
2. SFT-2 is dominated by verified student-success data. It contains meaningful same-episode
   recovery supervision, but only two strict Decision Correction targets; correction coverage is a
   current data bottleneck.
3. Under the completed historical strict metric, SFT-2 provides statistically supported K=2/K=4
   gains over sampled Direct SQL, but not a significant pass@1 gain.
4. Under the current BIRD reference metric, Direct SQL establishes the complete baseline at 42.37%
   greedy and 40.68/46.48/51.17% pass@1/2/4. SFT-2 is only 194 tasks complete and was paused to free
   GPU 1. Result-only RL is still running on GPU 0.
5. The available same-task BIRD-EX slice does not show an RL gain over SFT-2. Result-only RL is a
   baseline, while process-reward RL remains untrained pending reward-grounding quality gates.

## 6. Artifact index

| Artifact | Path |
|---|---|
| SFT-1 dataset | `data/sft/bird_sft1_grounded_v4d_all651_rolling4_resident_full.jsonl` |
| SFT-2 dataset | `data/sft/bird_sft2_onpolicy_full1000_flash_mixture.jsonl` |
| Direct SQL greedy BIRD-EX | `data/results/qwen2.5_7b_bird_direct_sql_base_greedy_dev1534_bird_ex/` |
| Direct SQL K=4 BIRD-EX | `data/results/qwen2.5_7b_bird_direct_sql_base_passk4_dev1534_bird_ex/` |
| SFT-2 partial BIRD-EX | `data/results/qwen2.5_7b_bird_sft2_onpolicy_full1000_epoch1_passk4_dev1534_bird_ex/` |
| RL partial BIRD-EX | `/home/dengyan/tabular_rl_outputs/results/qwen2.5_7b_bird_sft2_result_only_mixed70_retry1_checkpoint70_passk4_dev1534/` |
| Experiment chronology | `docs/archive/project_log.md` |
