# Historical project log (pre-cleanup AGENTS.md)

> Frozen chronology and superseded instructions retained for audit. The current source of truth is
> `/AGENTS.md` plus `docs/current/`. Do not execute an instruction from this log without checking
> whether it remains current.

Single source of truth both agents read (Codex loads this natively; `CLAUDE.md` imports it).
Repo-tracked + git-versioned. **Do not put secrets here** — the external-LLM key lives in `api.md`
(never commit/echo it). Update this file when a durable decision, milestone, or pitfall lands.

## Project

RL training environment for an LLM tool-use agent over relational tables. Research focus: **tool
design + dense process-reward design** for table reasoning. The model answers a question by calling
abstract tools (relational + perception + memory); the harness translates each call into composed
SQL over SQLite and verifies results. Current training-data generation uses a **real closed
model↔harness loop**: the model sees only the legal episode prefix and current environment feedback,
while gold SQL remains harness-only for train-side compatibility/final-denotation checks. The old
pipeline that compiled Spider/BIRD gold SQL into a complete tool trajectory and then enriched that
trajectory is retired because enrichment could see later trajectory information, causing future
leakage and poor-quality supervision. It is historical infrastructure, not the current data
mainline or a method claim. Regardless of source, no trajectory enters SFT unless it is replay- and
execution-verified.

## Current state (2026-07-14)

- **Gold-SQL trajectory compilation/enrichment retired (2026-07-22)**: do not generate new training
  data by compiling gold SQL into a complete tool-call trajectory and then enriching its plans,
  observations, thoughts, or perception steps. Because the enrichment stage has access to later
  actions/outputs in the completed trajectory, it can leak future information into earlier model
  turns; the resulting supervision was also empirically low quality. The current mainline must
  obtain actions and reasoning causally in a real model↔harness rollout from prefix-visible context
  and environment feedback. Gold SQL is hidden from the model/teacher and is restricted to
  harness-side compatibility and terminal-denotation verification. Existing SQL compilers,
  compiled artifacts, and related historical milestones may remain for audit/tests, but must not
  be presented as the current data-generation method, reused for trajectory enrichment, or claimed
  as the project's novelty.

- **BIRD SFT-2 reference-EX K=4 re-evaluation active (2026-07-22)**: a fresh full-dev run is
  regenerating all four samples per task from the frozen SFT-2 `checkpoint-743` under the aligned
  `bird-set` evaluation contract. Configuration matches the completed strict run: 1,534 tasks,
  temperature 0.7/top-p 0.95, rolling-4/full/resident context, max 30 steps, 1,024 output tokens,
  and BIRD external knowledge. It runs on otherwise-idle physical GPU 1 through
  `src/eval/run_bird_sft2_passk4_dev1534_table_rl.sh`; results incrementally persist to
  `data/results/qwen2.5_7b_bird_sft2_onpolicy_full1000_epoch1_passk4_dev1534_bird_ex/`. The first
  six tasks completed successfully after model readiness. Do not launch a duplicate; the runner
  monitors its SSH tunnel and endpoint and supports `--resume`. GPU 0 remains occupied by the
  independent RL-checkpoint held-out evaluation and was not touched.

- **BIRD evaluation denotation contract aligned to reference EX (2026-07-22)**: evaluation now
  exposes an explicit `--denotation-comparison {strict-multiset,bird-set}`. `bird-set` mirrors the
  released DPO-Text2SQL/BIRD scorer (`set(predicted_rows) == set(gold_rows)`), ignoring row order and
  duplicate multiplicity without numeric canonicalization; direct-SQL BIRD launchers also use the
  reference 20-second generated-query timeout. All BIRD-dev direct-SQL, SFT, and RL launchers pass
  `bird-set`, record it in manifests, and write to new `_bird_ex` result directories so prior strict
  artifacts cannot be resumed or mixed. Training-side replay, SFT quality/grounding gates, and RL
  rewards retain the existing normalized `strict-multiset` default. Re-scoring the frozen greedy
  direct-SQL outputs under the reference 20-second set contract gives **662/1534 = 43.16%**, versus
  the internal strict score **598/1534 = 38.98%**; report the metric name with every result. A fresh
  greedy run (`temperature=0`, `top_p=1`, one sample, evidence included when present) under this
  aligned contract completed **650/1534 = 42.37%**: simple 473/925 = 51.14%, moderate 146/464 =
  31.47%, and challenging 31/145 = 21.38%. Its outcomes are 650 correct, 514 wrong-result, and 370
  execution-error, with no API/incomplete-response failures. Canonical artifact:
  `data/results/qwen2.5_7b_bird_direct_sql_base_greedy_dev1534_bird_ex/`. Use 42.37% as the newly
  generated normal 7B direct-SQL baseline; retain 43.16% only as a frozen-output scorer audit. The
  matching fresh K=4 sampling run (`temperature=0.7`, `top_p=0.95`) is also complete: pass@1/2/4 is
  **624/713/785 of 1534 = 40.68/46.48/51.17%**. All 6,136 candidates are present; sample outcomes
  are 2,523 correct, 2,061 wrong-result, and 1,552 execution-error, with no API or incomplete
  responses. By released difficulty, pass@1/2/4 is simple 460/515/552 of 925, moderate 132/157/184
  of 464, and challenging 32/41/49 of 145. Canonical artifact:
  `data/results/qwen2.5_7b_bird_direct_sql_base_passk4_dev1534_bird_ex/`.

- **BIRD SFT-2 train/dev evaluation completed (2026-07-21)**: the frozen 11,874-record on-policy
  mixture continued SFT-1 checkpoint-270 for one QLoRA epoch / 743 optimizer steps in 13:03:13.
  Final train loss is 0.2791; logged first/last ten-window means are 0.2912/0.2728 and gradient norms
  remain stable. There is no held-out loss because the run intentionally used `eval_strategy: no`.
  Full BIRD-dev K=4 evaluation is complete on all 1,534 tasks with temperature 0.7/top-p 0.95,
  rolling-4/full/resident inputs, max 30 steps and 1,024 output tokens: pass@1/2/4 is **595/719/855
  = 38.79/46.87/55.74%**. By released difficulty, pass@1/2/4 is simple 46.92/54.81/63.68%,
  moderate 29.96/38.36/46.77%, and challenging 15.17/23.45/33.79%. Against direct SQL on the
  exact same tasks, paired SFT-2 gains are +24/+71/+140 at k=1/2/4; exact McNemar p-values are
  0.259/0.00073/5.37e-11. Thus pass@1 is only a point estimate, while k=2/4 gains are statistically
  supported. The SFT-1 K=4 job lost its SSH tunnel after 1,064 tasks and must not be reported as a
  full-dev result. On those 1,064 common tasks SFT-2 vs SFT-1 is 408/485/574 vs 379/485/580 at
  k=1/2/4: SFT-2 improves first-sample placement but does not improve the observed pass@4 ceiling.
  SFT-2 evaluation artifacts are remote at
  `/home/dengyan/tabular_rl_outputs/results/qwen2.5_7b_bird_sft2_onpolicy_full1000_epoch1_passk4_dev1534/`.

- **BIRD SFT-2 result-only RL mixed70 baseline completed but not yet evaluated (2026-07-22)**:
  both 70-group runs are weight-distance-confirmed to initialize from the SFT-2 adapter, not SFT-1.
  The first run is invalid as a baseline result: its short generation budget clips outputs before a
  complete tool call, producing 224/280 terminal protocol errors and only 13/70 effective updates.
  Keep it for audit only. The `retry1` run completes 70 groups / 280 episodes in 14.73 hours with
  73 correct samples, task pass@1/pass@4 20/34, reward-count buckets `{0:36,1:13,2:9,3:6,4:6}`,
  and **28/70 effective updates** (the six all-correct and 36 all-wrong groups have zero relative
  advantage). There are no optimizer/OOM failures. However, the trainer does not persist an args/
  config manifest, and no held-out BIRD-dev evaluation exists for any RL checkpoint, so no RL gain
  can yet be claimed. Also, full-turn result-only credit positively trains every generated turn in a
  positive-advantage recovered trajectory: 15 positive episodes contain 17 erroneous turns (12
  execution, 4 protocol, 1 argument-validation). This is acceptable only as the deliberately coarse
  result-only control; process-reward training must assign those errors negative/zero credit and
  reward the subsequent recovery separately. Candidate checkpoint:
  `/home/dengyan/tabular_rl_outputs/checkpoints/qwen25_bird_sft2_result_only_mixed70_baseline_retry1_20260721/checkpoint-70/`.
  Audit of the 280 retry1 episodes finds 140 genuinely scored terminal answers (73 correct, 67 wrong)
  and 140 without a valid terminal. The historical raw `legal` flag overcounts this as 225 because
  `ToolUseEnv` set it before terminal scoring and failed to roll it back when scoring raised; 85
  execution/protocol/context/max-step records were therefore sticky-legal. `env.py` now sets legal
  only after scoring returns, with a regression test; historical rewards are unaffected and old
  rollouts remain immutable, so their legal rate must be reconstructed from terminal outcomes.

- **BIRD result-only RL held-out evaluation active (2026-07-22)**: checkpoint-70 from the valid
  `retry1` run is being evaluated on all 1,534 BIRD-dev tasks at the exact SFT-2 K=4 protocol:
  temperature 0.7/top-p 0.95, rolling-4/full/resident, max 30 steps, 1,024 output tokens, and four
  complete samples per task. The remote-owned launcher is
  `src/eval/run_bird_rl_result_only_mixed70_checkpoint70_passk4_dev1534_table_rl.sh`; it uses GPU 0,
  runs a transport/audit smoke before full evaluation, and aborts after three failed endpoint health
  checks. The smoke produced 2 tasks / 8 complete samples with no API errors but zero legal terminals:
  all eight reached the semantic error cap through execution errors, mostly malformed expressions
  over spaced/parenthesized column names or invalid table-column choices. This is real policy behavior,
  not transport failure, and remains scored. Full results write incrementally to
  `/home/dengyan/tabular_rl_outputs/results/qwen2.5_7b_bird_sft2_result_only_mixed70_retry1_checkpoint70_passk4_dev1534/`.

- **BIRD grounded SFT-1 scale run completed (2026-07-19)**: scale1000d completed 1000/1000 tasks
  and yielded 295 raw verifier-correct episodes. Quality filtering retained 275 and deterministic
  V4d grounding retained 178. Combined with the two earlier grounded pools, the frozen training
  set has **651 unique replay-correct episodes / 4,319 rolling last-turn-only targets**
  (easy/medium/hard targets 1808/1481/1030; 387 feedback-recovery targets). Exact remote
  LLaMA-Factory/Qwen2.5 token audit at cutoff 6400 retained **4319/4319**, with encoded length
  p50/p90/max 2967/4063/6400 and no truncated final target. Qwen2.5-7B QLoRA completed 2 epochs /
  540 optimizer steps in **9:59:05**, final train loss **0.4546**, run id
  `bird_sft1_grounded_v4d_all651_6400_20260718_night`. On the fixed disjoint BIRD-dev stratified 30
  with matching rolling-4 resident/full inputs, epoch 1 (`checkpoint-270`) scores **8/30 = 26.7%**
  (simple/moderate/challenging 4/12, 2/9, 2/9; 24 legal terminal answers), while epoch 2 scores
  **7/30 = 23.3%** (3/12, 2/9, 2/9; 24 legal). Freeze checkpoint-270 as the SFT-1 student for
  on-policy SFT-2 construction; the 30-task difference is only a checkpoint-selection gate, not a
  scaling claim. Source episodes:
  `data/trajectories/bird_sft1_grounded_v4d_all651.jsonl`; SFT:
  `data/sft/bird_sft1_grounded_v4d_all651_rolling4_resident_full.jsonl`; result directory:
  `data/results/qwen2.5_7b_bird_sft1_grounded_v4d_all651_bird_dev_stratified30/`. The rolling-export
  no-leak check now inspects structured factual provenance fields instead of all textual `step_N`
  occurrences, so plan item ids are not mistaken for future evidence while real future/current
  provenance remains rejected.

- **BIRD SFT-2 easy student rollout pilot completed (2026-07-19)**: fixed selection
  `data/eval_inputs/bird_train_sft2_student_pilot1000.jsonl` contains 1000 BIRD-train-compatible
  tasks (400/300/300 proxy difficulty, all 69 DBs) with zero example-id overlap against the 651
  SFT-1 teacher-success episodes. `rollout_passk.py` now consumes common DatasetTask JSON/JSONL via
  adapter-provided `db_path` and `gold_sql`, carries `external_knowledge`, and supports the same
  rolling-legal-history/full/resident renderer and strict same-episode feedback contract as online
  eval. A 2-task x 2-sample integration smoke produced four legal, fully audited BIRD trajectories
  with no gold leakage (all wrong-answer, useful as correction candidates). The first-100 yield
  pilot completed under launchd label `com.tabularrl.bird-sft2-student-passk100`: K=4, temperature
  0.7/top-p 0.95, full turns, max 30 steps. Because the selector groups output by bucket, these 100
  are all **easy**, not a representative 40/30/30 subset. Results: pass@1/2/4 = **53/62/68%**;
  216/400 correct samples and 376/400 legal; sample outcomes are 216 correct, 160 wrong-answer,
  10 execution-error, 7 context-overflow, 5 argument-validation, 1 protocol, 1 max-steps. Per task,
  correct samples out of four are `{0:32, 1:6, 2:12, 3:14, 4:36}`, so 32% of easy tasks have
  mixed terminal rewards useful for outcome-only group RL. Result directory:
  `data/results/bird_sft2_student_epoch1_passk100_k4/`. The idle epoch-1 vLLM server was stopped
  after completion, freeing GPU 1. Next run a real medium/hard supplement before extrapolating yield
  or launching the remaining 900 tasks.

- **BIRD SFT-2 medium/hard supplement completed (2026-07-19)**: launchd job
  `com.tabularrl.bird-sft2-medium-hard60` exited cleanly after running 30 medium then 30 hard tasks,
  each at K=4 with the frozen 651-episode SFT-1 epoch-1 `checkpoint-270` and the same
  rolling-4/full/resident strict-recovery settings as the easy pilot. Medium pass@1/2/4 is
  **18/20/22 of 30 = 60.0/66.7/73.3%**, averaging 2.27 correct and 3.67 legal samples per task.
  Hard pass@1/2/4 is **6/11/17 of 30 = 20.0/36.7/56.7%**, averaging 1.13 correct and 3.67 legal
  samples per task. Result directories are
  `data/results/bird_sft2_student_epoch1_passk_medium30_k4/` and
  `data/results/bird_sft2_student_epoch1_passk_hard30_k4/`. Across easy100 + medium30 + hard30,
  aggregate task pass@1/2/4 is **77/93/107 of 160 = 48.1/58.1/66.9%**; aggregate sample yield is
  318/640 verifier-correct and 596/640 legal. The job cleaned up its vLLM service; physical GPU 0
  was subsequently occupied by another server user and must not be killed by this project.

- **BIRD SFT-2 on-policy construction pilot completed (2026-07-19)**: protocol SSOT is
  `draft/bird_sft2_onpolicy_data_protocol.md`; implementation is
  `prepare_bird_sft2_onpolicy.py`, `build_bird_sft2_corrections.py`, and
  `assemble_bird_sft2_mixture.py`. Student-first K=4 is mandatory and external teachers receive
  only pass@4-failed BIRD-train tasks. Fresh replay recovers all **318/318** correct student samples;
  quality gates accept 314 episodes / **1,953 targets**, including **45** tagged Feedback Recovery
  targets. Exact Qwen2.5/LLaMA-Factory cutoff-6400 audit retains **1953/1953**, max length 6311.
  The balanced Flash fallback pilot (2/2/2 easy/medium/hard, one attempt) yields only **1/6** verified,
  with three legal-wrong and two protocol-terminal outcomes; its one 10-step branch replays. A
  semantic first-divergence gate rejects mere `describe_table` ordering and admits one diagnosed
  correction (`Users` was an extraneous base table), whose bad student action is audit-only. The
  deduplicated candidate mixture has **2,613** records: 650 SFT-1 replay, 1,908 normal student
  success, 45 recovery, 9 teacher fallback, and 1 correction. This proves the pipeline but is not
  correction-rich enough to train. Expand the fixed Flash fallback to 20 failures; switch to Pro on
  the same ids if verified yield <30% or protocol-terminal rate >20%. Report:
  `src/sft/BIRD_SFT2_ONPOLICY_PILOT_REPORT.md`.

- **BIRD SFT-2 matched teacher20 gate completed (2026-07-19)**: deterministic selector
  `select_bird_sft2_teacher_fallback.py` froze 20 SFT-1 pass@4 failures at 8/6/6
  easy/medium/hard across 14 DBs and retained all prior pilot6 ids. Flash resumed only the 14 new
  tasks and finishes **3/20 = 15%** verified, 14 legal-wrong, 3 protocol-terminal. The preset gate
  triggered Pro on the exact same 20 ids; Pro is **2/20 = 10%**, 17 legal-wrong, 1 protocol-terminal.
  Pro improves formatting but not semantic yield; the raw provider union solves four tasks, with Pro
  adding one Flash miss. Quality/replay gates retain three unique teacher episodes / 24 targets
  (2 Flash + 1 Pro): one Flash success exceeds the 300-word think cap and one cross-provider success
  is a conflicting duplicate. Exact token audit keeps 24/24. Deterministic shared-state diagnosis
  admits **2** Decision Corrections and rejects the third teacher branch for lacking a provably bad
  student action; correction token audit keeps 2/2. Updated deduplicated candidate mixture
  `data/sft/bird_sft2_onpolicy_pilot160_teacher20_mixture.jsonl` has **2,627** records: 650 SFT-1
  replay, 1,908 normal student success, 45 Feedback Recovery, 22 teacher fallback, and 2 correction.
  Do not globally replace Flash with Pro or scale teacher fallback yet; use student K=4 → Flash →
  Pro-only-on-Flash-failures, and first improve semantic correction yield.

- **BIRD SFT-2 Flash-only residual set completed (2026-07-19)**: after choosing Flash for continued
  generation, `select_bird_sft2_teacher_fallback.py --all-failures` froze all **53** student pass@4
  failures (easy/medium/hard 32/8/13, 31 DBs), preserving the prior Flash20 records and requesting
  only the remaining 33. Flash finishes **6/53 = 11.32%** raw verified, with 42 legal-wrong and 5
  protocol-terminal outcomes and zero transport/context retries. Quality gates retain **5 episodes /
  35 targets** (easy 3, medium 1, hard 1; one feedback-recovery target); one raw success is rejected
  for >300-word reasoning. Deterministic shared-state pairing still admits only **2** corrections and
  rejects three branches without a provably bad student action. Final Flash-only candidate mixture
  `data/sft/bird_sft2_onpolicy_pilot160_flash_all53_mixture.jsonl` contains **2,638 unique records**:
  650 SFT-1 replay, 1,908 normal student success, 45 student Feedback Recovery, 33 non-duplicate
  Flash fallback, and 2 Decision Correction. Exact cutoff-6400 audit keeps Flash 35/35 (max 4634)
  and correction 2/2 (max 2197); strict/no-duplicate tests and 33 regression tests pass. Every
  pass@4 failure in this 160-task pilot has now received one Flash attempt; do not call Pro again for
  this pool.

- **BIRD SFT-2 student scale-out completed (2026-07-20)**: the 2,638-record Flash-only mixture above is
  a pipeline pilot, not the intended final SFT-2 volume. The remaining **840/1,000** frozen,
  SFT-1-disjoint student tasks are materialized as easy/medium/hard **300/270/270**, with zero overlap
  against the completed easy100 + medium30 + hard30 cohort. Launchd job
  `com.tabularrl.bird-sft2-remaining840` runs them sequentially at K=4 with checkpoint-270 on physical
  GPU 1, rolling-4/full/resident context, and the existing strict recovery contract. Result roots are
  `data/results/bird_sft2_student_epoch1_passk_{easy_remaining300,medium_remaining270,hard_remaining270}_k4/`.
  The model became ready at 20:11 Asia/Shanghai and the easy cohort began writing verified results;
  do not launch a second copy. After completion, run fresh replay and quality export before teacher
  fallback. Use Flash only for the new pass@4 failures, then rebuild the final deduplicated mixture.
  The original launch completed easy 300 but its SSH forwarding tunnel later dropped during medium:
  only **62/270** medium records have transport-complete K=4 samples; 179 attempted records contain
  at least one explicit `api_error` and 29 were never attempted. Those audit records remain unchanged
  and must not count as model failures or training data. Retry selector
  `select_transport_retry_tasks.py` materialized the exact **208** medium tasks requiring a fresh,
  independent transport attempt. Launchd job `com.tabularrl.bird-sft2-remaining478-retry1` is active
  on medium208 followed by hard270, writing medium to a separate `_transport_retry1` result directory.
  Its runner checks both the tunnel process and `/v1/models` every 15 seconds and aborts after three
  failed endpoint checks instead of silently producing a long API-error suffix. Retry1 completed
  medium208 plus hard270 and exited 0, releasing GPU 1. Combining the original easy100/medium30/
  hard30 pilot with transport-complete scale-out records yields exactly **1,000 unique tasks / 4,000
  sampled episodes**, with no duplicate task ids. Final pass@1/2/4 is **430/541/632 = 43.0/54.1/
  63.2%**; 1,745 episodes are verifier-correct and 3,647 are legal. Difficulty pass@1/2/4 is easy
  **206/245/275 of 400**, medium **146/184/210 of 300**, and hard **78/112/147 of 300**. Sample
  outcomes are 1,745 correct, 1,902 wrong-answer, 261 execution-error, 35 protocol-error, 30
  context-overflow, 21 argument-validation, 4 max-steps, and 2 nonrecoverable execution-error.
  Transport-audit attempts remain outside these totals. Next run fresh replay/quality export over
  correct episodes, then send only the 368 pass@4-failed tasks to the Flash fallback lane.

- **BIRD SFT-2 full1000 postprocess active (2026-07-20)**: transport-aware merger
  `merge_passk_transport_attempts.py` combines the seven pilot/scale shards in task-source order,
  requires exactly one complete K=4 attempt per task, and excludes incomplete transport attempts
  from all semantic statistics and fallback selection. Canonical artifact
  `data/results/bird_sft2_student_full1000_k4_canonical_all.jsonl` contains exactly 1,000 unique
  records (SHA256 `6246b6dbddf7db6feccb2374cb138f8fe685575d47097fe4c7b3f77f54d070ad`);
  its manifest retains all 179 excluded medium transport attempts for audit. Low-priority launchd job
  `com.tabularrl.bird-sft2-full1000-postprocess` is replaying all 1,745 correct episodes through a
  fresh harness, applying max-20-step/max-300-think-word/repetition gates, selecting all 368 true
  pass@4 failures for Flash, and then exporting rolling-4/full/resident single-action SFT records.
  Do not start Flash until this job exits 0 and its replay/quality manifest is reviewed.

- **BIRD SFT-2 full1000 student replay completed (2026-07-20)**: fresh double replay finishes
  1,745 verifier-correct samples with 1,744 replay-correct and one strictly rejected self/future
  reference (`bird_sft2_student_1661_sample_0`, `step_8` citing itself). Quality and action-dedup
  gates retain **1,677 episodes / 10,968 rolling action targets**, including **261 Feedback Recovery**
  targets. The other 67 rejects are 53 repeated-call, 9 think-limit, 3 exact-action duplicate, and
  2 step-limit. Rolling-4/full/resident export is active. Flash fallback has resumed from the 53
  preserved prior attempts and is calling only the 315 new ids; do not count the seeded records as
  new provider calls.

- **BIRD SFT-2 full1000 mixture frozen (2026-07-20)**: student rolling export contains 10,968
  targets from 1,677 replay/quality-accepted episodes. Flash completes all 368 fallback ids with
  38 raw verifier successes; the unchanged max-20-step/max-300-think-word/current-protocol gate
  retains 35 episodes / 256 targets (easy/medium/hard episodes 18/8/9), including 18 recovery
  targets, and structural/full-prompt audits pass. The deduplicated final mixture
  `data/sft/bird_sft2_onpolicy_full1000_flash_mixture.jsonl` has **11,874 records**: 10,968 student
  success (10,707 normal + 261 recovery), 254 nonduplicate Flash fallback, 650 SFT-1 replay, and
  2 previously verified Decision Corrections. Exact remote Qwen2.5/LLaMA-Factory cutoff-6400 audit
  retains **11,874/11,874**, length p50/p90/max 2855/3816/6400, with no truncated final target.
  A broader correction replay was deliberately stopped after an unbounded `image_and_language`
  student-failure query exceeded 20 minutes; it is noncritical for the deadline baseline and must be
  rerun later with a VM deadline. One-epoch continuation config and launcher are
  `qwen2.5_7b_qlora_bird_sft2_onpolicy_full1000_6400_table_rl.yaml` and
  `run_qwen25_7b_bird_sft2_onpolicy_full1000_6400_table_rl.sh`; remote sync/launch was blocked by
  Codex approval-usage exhaustion, not by a training or data failure.

- **BIRD SFT-2-to-result-only-RL deadline path prepared (2026-07-20)**: controlled launchd watcher
  `com.tabularrl.bird-sft2-full1000-flash-after-replay` waits for the full1000 replay manifest,
  requires exactly 368 fallback tasks, and only then launches one Flash attempt per pass@4 failure
  with the established rolling-4/full/2048/strict-recovery contract. It stops rather than calling the
  API if replay fails or counts differ. Result-only BIRD adaptation is now explicit: `task_data.py`
  accepts DatasetTask JSON/JSONL or historical `{examples:...}` selections, uses adapter db paths and
  hidden gold SQL, and preserves external knowledge. `ToolUseEnv` now supports the same rolling-4/
  full/resident legal-history context as SFT-1/SFT-2 while keeping invalid actions audit-only and
  returning their structured errors through `LAST TOOL ERROR`. RL tests **29/29** and harness tests
  **136/136** pass. The SFT1-policy
  K=4 pool contains 202 mixed-reward tasks (pass@1 fail, pass@4 success); frozen candidate and
  70-group pilot files are `data/rl/bird_sft2_result_only_mixed_candidates_from_sft1_passk4.json`
  and `data/rl/bird_sft2_result_only_mixed70_pilot.json`, with pilot difficulty 22/20/28 easy/medium/
  hard. Historical accelerate timing is 6.48 hours for 70 full-turn groups; a 1,000-group pass would
  be roughly 92 hours and is not a valid Wednesday-afternoon commitment. Deadline target is one-epoch
  SFT-2 followed by the 70-group terminal-{0,1} baseline and held-out BIRD-dev evaluation. The
  accelerate trainer now preserves adapter `db_path`/external knowledge, stops generation only
  after the model emits the real `</tool_call>` tag, and writes complete episode audits to
  `rollouts.jsonl`. A remote BIRD smoke on checkpoint-270 produced rewards `[1,0,1,1]`, loss
  `0.002787`, and `updated=true` with no parser repair. The frozen 70-group pilot's 40 databases
  (11 GB on disk) are synced to `table_rl` and pass a no-difference rsync verification.
  The 368 full1000 Flash-fallback ids include all 53 tasks already audited in the earlier Flash-only
  residual run. The watcher now validates that exact 53-id subset, seeds the prior success/failure/all
  records into the new resumable outputs, and calls Flash only for the remaining 315 ids; do not
  overwrite the first attempts or pay for duplicate calls.

- **BIRD SFT-1 model selection dev-100 completed (2026-07-19)**: the recent dev-30/SFT-2 model was
  the new 651-episode model's epoch-1 `checkpoint-270`, not the older small model. On the shared
  dev-30, old grounded-R2 206-episode epoch 1 and new 651-episode epoch 1 both score 8/30; they share
  six successes and each uniquely solves two, while the new model has 24 versus 23 legal terminals.
  This is a tie, not evidence that scaling is worse. Deterministic selection
  `data/eval_inputs/bird_dev_stratified100_disjoint_seed20260719.indices.json` contains a fresh
  40/30/30 simple/moderate/challenging cohort with zero overlap against the historical dev-30.
  Launchd job `com.tabularrl.bird-sft1-model-selection-dev100` evaluated old `checkpoint-82` and
  new `checkpoint-270` sequentially on GPU 1 with identical rolling-4/full/resident strict settings
  and released the GPU. Old scores **30/100**, 77 legal, simple/moderate/challenging 15/40, 8/30,
  7/30. New scores **32/100**, 79 legal, difficulty 19/40, 7/30, 6/30, and uses fewer average steps
  (8.05 vs 9.37). Paired outcomes are both-correct 21, old-only 9, new-only 11, neither 59;
  exact McNemar `p=0.824`, so the two-point lead is not significant. Combining the disjoint 100
  with historical dev-30 gives old/new **38/130 vs 40/130**. Freeze new checkpoint-270 as the
  current point-estimate-best SFT-1 student (slightly higher correct/legal and already used by the
  SFT-2 pilot), but retain old checkpoint-82 as a statistically tied strong control. Results:
  `data/results/qwen2.5_7b_bird_grounded_r2_206_epoch1_bird_dev_disjoint100/` and
  `data/results/qwen2.5_7b_bird_sft1_grounded_651_epoch1_bird_dev_disjoint100/`.

- **Best SFT-1 full BIRD-dev evaluation completed (2026-07-19)**: do not compare the new tool
  model's deliberately hard-weighted 40/30/30 sample rate directly to the base direct-SQL full-dev
  38.98%. On the exact same disjoint dev-100 ids, base Qwen2.5-7B direct SQL scores 36/100 and the
  new 651-episode tool SFT-1 scores 32/100; on historical dev-30 they score 9/30 and 8/30. Combined
  same-task totals are **45/130 = 34.6% direct SQL vs 40/130 = 30.8% tools**, a 3.8-point gap;
  paired outcomes are both 23, direct-only 22, tool-only 17, neither 68, exact McNemar `p=0.522`.
  The apparent 8-point comparison was therefore an unmatched-cohort artifact, though a smaller
  interface gap remains plausible. Launchd job `com.tabularrl.bird-sft1-best-full-dev` completed
  normally on all 1,534 BIRD-dev tasks (workers=4, rolling-4/full/resident, strict parser): the new
  checkpoint-270 scores **563/1534 = 36.70%**, versus direct SQL **598/1534 = 38.98%**, a **2.28
  point** gap. Paired outcomes are both-correct 349, direct-only 249, tool-only 214, neither 722;
  exact McNemar `p=0.114`, so the full-dev point estimate is lower but the paired difference is not
  significant at 0.05. Results are at
  `data/results/qwen2.5_7b_bird_sft1_grounded_651_epoch1_bird_dev_full1534/`. The launchd job exited
  0 and cleaned up the project vLLM service.

- **BIRD direct-SQL pass@k baseline completed (2026-07-20)**: GPU 0 ran base
  Qwen2.5-7B-Instruct independently of the GPU-1 SFT-2 rollout under launchd label
  `com.tabularrl.bird-direct-sql-passk4-base`. `text2sql_passk.py` now accepts adapter-exported
  DatasetTask JSON/JSONL and uses `task_db_path`, `task_gold_sql`, BIRD external knowledge, the same
  full-schema prompt as the established direct-SQL evaluator, and a 5-second VM deadline per
  generated SQL; Spider remains the default when no task file is supplied. The baseline samples
  exactly K=4 candidates at temperature 0.7/top-p 0.95 and reports pass@1/2/4 on all 1,534 BIRD-dev
  tasks. A separate smoke-10 required four returned completions per task and passed with 1/10 at all
  three k values; do not extrapolate this single-database prefix. Full artifacts are written to
  `data/results/qwen2.5_7b_bird_direct_sql_base_passk4_dev1534/`. The launch script monitors both its
  SSH tunnel and model endpoint and aborts after three failed health checks, so transport failures
  cannot silently become baseline failures. The first full attempt stopped after 86 persisted tasks
  because shared `protocol.normalize_rows` called `round()` on a `NaN` result cell. The scorer now
  canonicalizes `NaN`, `+Inf`, and `-Inf` as distinct stable values before finite-number rounding;
  evaluator tests (13) and the full harness suite (136) pass. The launch script uses `--resume`, and
  the second run successfully rescored the missing q84 case and continued without changing the 86
  prior records. The resumed job completed all **1,534/1,534** tasks and exited 0, releasing GPU 0.
  Final sampled scores are pass@1 **571/1534 = 37.22%**, pass@2 **648/1534 = 42.24%**, and pass@4
  **715/1534 = 46.61%**. All 6,136 candidates are present with zero API/incomplete-response tasks;
  candidate outcomes are 2,246 correct, 2,301 wrong-result, 1,588 execution-error, and one no-SQL.
  K=4 gains 9.39 points over this run's sampled pass@1 and 7.63 points over the separate temperature-0
  greedy baseline (38.98%); sampled pass@1 itself is not the same decoding condition as greedy.

- **BIRD SFT-1 tool pass@k evaluation active (2026-07-20)**: launchd job
  `com.tabularrl.bird-sft1-passk4-dev1534` uses physical GPU 0 to evaluate frozen checkpoint-270 on
  the same 1,534 BIRD-dev tasks at K=4, temperature 0.7/top-p 0.95, rolling-4/full/resident context,
  max 30 tool steps, and strict same-episode recovery. GPU 1 continues the independent SFT-2 train
  rollout. `rollout_passk.py` now requires explicit `--allow-eval-tasks` for dev-named DatasetTask
  inputs and adds `dataset_purpose:evaluation` plus `sft_export_eligible:false` only to those eval
  manifests; training manifests remain hash-compatible. Based on measured train rollouts weighted by
  the official dev difficulty distribution, expected wall time is about 9.6 compute hours plus
  overhead, reported operationally as **10--12 hours**. Results write incrementally to
  `data/results/qwen2.5_7b_bird_sft1_grounded_651_epoch1_passk4_dev1534/`; the first four tasks landed
  successfully after model readiness, so the full run is live rather than merely queued.

- **BIRD SFT-1 scale protocol freeze (2026-07-18)**: while scaling the external-teacher corpus,
  keep the model-visible `v2i/R2` action contract intact, including `join_tables` prefixes and
  `P__column` materialized join-column names. Reward grounding, weights, filtering, and other
  harness-private RL calculations may evolve under separately versioned reward implementations,
  but must not silently alter stored tool actions. The next scale phase uses DeepSeek v4 Flash as
  the primary teacher with independent whole-episode budgets easy/medium/hard = 1/2/3, stops after
  the first verifier-correct attempt, preserves every attempt for success@k, and reserves Pro only
  for a later verified hard-quota shortfall. Any future structured/dotted join-reference experiment
  is a separately versioned ablation; migrate and replay old data or regenerate from zero rather
  than overwriting this protocol.
- **BIRD SFT-1 Flash scale1000c launched (2026-07-18)**: deterministic seed-20260718 selection
  `data/eval_inputs/bird_train_sft1_scale1000c.jsonl` contains 1000 previously unused compatible
  BIRD-train tasks (400/300/300 easy/medium/hard proxy, 69 DBs), excluding context100, pilot30,
  scale500, and scale500b. Difficulty-specific inputs live under
  `data/eval_inputs/bird_train_sft1_scale1000c_buckets/`. The exact external-rollout protocol hash
  is `ad8b58d8b28b4ea9`; Flash uses rolling legal history=4/full/2048 tokens, strict no-repair, and
  easy `(attempts=1,max_steps=30,error_cap=3)`, medium `(2,30,3)`, hard `(3,40,4)`. The three
  resumable generators are owned by launchd label `com.tabularrl.bird-scale1000c-flash` via
  `src/sft/run_bird_scale1000c_flash.sh`; logs are `logs/sft_scale1000c/{easy,medium,hard}.log` and
  outputs are `data/trajectories/bird_scale1000c_flash_{easy,medium,hard}_success*`. Do not launch
  Pro rescue until this batch finishes, replay/quality gates run, and the verified hard quota
  shortfall is measured.
- **BIRD SFT-1 Flash scale1000c generation complete (2026-07-18)**: all 1000 tasks finished in
  1636 persisted whole-episode attempts. Verifier-correct episodes are 381: easy 139/400, medium
  119/300, hard 123/300. All three success files pass the structural and full-prompt gates with no
  structural issues. The standard quality contract (<=12 legal steps, <=300 think words, no
  repeated identical calls, current action protocol, rolling history=4) accepts 356 episodes / 2377
  action targets: easy 139, medium 115, hard 102, including 270 feedback-recovery targets. It rejects
  25 successes (12 step limit, 11 think limit, 2 repeated call). Candidate pool:
  `data/trajectories/bird_scale1000c_flash_verified_candidate_pool.jsonl`, SHA256
  `dc447834dd1d4c6a5b5d600fd39357c407347ed89f5e90f7b414074abaa7ba2f`. This remains candidate-only
  until the deterministic grounding-completeness gate runs; Pro hard rescue must use the resulting
  post-grounding shortfall, not the raw 123 hard successes. The completed launchd job was removed.
- **BIRD scale1000c grounding gate + scale1000d launch (2026-07-19)**: V4d replay/grounding over
  the 356 quality candidates gives 355/356 replay-correct and 239/356 deterministic-complete, but
  the complete count includes replay-failed `bird_train_01054`; the strict conjunction admits
  **238 episodes / 1608 targets** (easy 104, medium 76, hard 58; 175 recovery targets, 57 DBs).
  Rejections record one replay failure, 107 unsupported-final-value hits, and 18 unsupported-action-
  literal hits (issue counts can overlap). Eligible artifact:
  `data/trajectories/bird_scale1000c_flash_grounded_eligible.jsonl`, SHA256
  `b64b6c95d4efa55454e181516bb5bcbc38135272a3c3e1695407ff3057351ef1`; train-side eval fraction is
  deliberately zero because downstream evaluation remains BIRD-dev. Combined with the previous V4
  pool, current deterministic/replay eligible supply is 473 (easy 247, medium 142, hard 84).
  Only 35 never-used compatible hard tasks remain. Therefore scale1000d is 400/300/300, with 735
  fresh tasks (400 easy, 300 medium, 35 hard) plus 265 independent hard rescues selected only from
  historical tasks lacking any grounded-eligible success; all 473 eligible ids are excluded from
  rescue. Selection is `data/eval_inputs/bird_train_sft1_scale1000d.jsonl` (seed 20260719, 68 DBs).
  Flash generation is active under launchd label `com.tabularrl.bird-scale1000d-flash` via the
  corrected `src/sft/run_bird_scale1000d_flash.sh`; it uses the same 1/2/3 attempt and 30/30/40 step
  budgets. This runner explicitly terminates its caffeinate child on exit, fixing scale1000c's
  completed-job keepalive loop.
- **Scale1000c grounding rejects preserved (2026-07-19)**: no quality or grounding reject was
  deleted. `data/trajectories/bird_scale1000c_grounding_gate/` contains canonical trajectory JSONL
  for 238 eligible and 118 rejected-any episodes, plus separate (overlapping) replay-incorrect (1),
  unsupported-final-value (107), unsupported-action-literal (18), and multiple-issue (8) files.
  `gate_index.jsonl` records per-id reasons and unsupported values/literals; `manifest.json` pins all
  source/output hashes. This archive is the source for future harness-semantic recovery (for example
  empty-set cardinality grounding) and SFT-2 correction extraction. Never regenerate or discard it
  merely because the current conservative gate excludes those episodes.
- **Grounded-473 exploration/reward-topology audit (2026-07-19)**: the combined 235 old + 238
  scale1000c replay/deterministic-grounding eligible episodes do satisfy basic closed-environment
  exploration and dense multi-step credit, but not a strong multi-branch-credit claim. All 473 use
  `describe_table`; 370 (78.2%) describe multiple tables, 188 (39.7%) use `inspect_column`, 365
  (77.2%) use `read_subtable`, 188 (39.7%) join, and 223 (47.1%) recover after real tool feedback.
  Excluding terminal submission, 193 (40.8%) contain off-final-slice actions and 153 (32.3%) contain
  off-slice perception. Every episode rewards >1 step (positive-step p50/p90 5/7; max-step positive
  share p50/p90 0.258/0.404). Observation credit is substantial: positive `describe_table` 503/509,
  `read_subtable` 526/568, `inspect_column` 95/220. Corrected topology reconstruction (the persisted
  rollout steps omit runtime reference sidecars, so the audit must rebuild them from tool arguments
  and handle producers exactly as `execute_tool` does) finds 369/473 merges, 393/473 forks, and
  395/473 non-chain action graphs. Of these, 337 have data+grounding merges, 26 have >=2 produced-
  data step parents, one has a data+value merge, and five use multi-handle final grounding. All 196
  Join calls have at least two data references: joint `(produced-step parents, source parents)`
  counts are `(0,2)=42`, `(1,1)=101`, `(2,0)=20`, `(1,2)=19`, `(0,3)=6`, `(1,3)=5`, `(2,1)=2`,
  `(1,4)=1`. The sole SetOp has `(0,2)`, i.e. two base-source edges. Thus runtime parameter-based
  A-output-handle -> B edges are present and general non-chain propagation is exercised, while
  independent produced-data-branch merges and multi-handle finals remain uncommon. Audit artifact:
  `src/rl/audit_exploration_reward_topology.py` and
  `data/rl/bird_grounded473_exploration_reward_topology.json`. Next repair should add per-table/
  per-column evidence-unit edges for base-table joins and multi-handle final dependencies, then test
  on student SFT-2/RL rollouts containing genuine branch/reuse structures; do not force inefficient
  branching into the teacher prompt merely to improve this metric.
- **Grounded-473 tool-frequency audit (2026-07-19)**: 473 eligible episodes contain 3100 legal
  action targets. Calls/episode coverage are: condition_filter 701/434, read_subtable 568/365,
  describe_table 509/473, answer_from_context 473/473, group_aggregate 221/214, inspect_column
  220/188, join_tables 196/188, plan 74/36, extreme_value_select 71/70, project 66/66, and set_op
  **1/1**. Join coverage is adequate; SetOp protocol learning is not. This is primarily source
  scarcity: only 9/5915 compatible BIRD-train gold SQL tasks contain a set operation, all `UNION`;
  scale1000d includes six of those nine, but random scaling cannot supply INTERSECT/EXCEPT balance.
  After scale1000d grounding, build a separately tagged/ablated BIRD-database SetOp curriculum,
  rather than forcing a uniform main distribution: target roughly 20--30 verified episodes per
  union/intersect/except and cover source-source, source-handle, and handle-handle operands. Keep the
  curriculum around 1--3% of the final SFT mixture and retain natural-distribution metrics separately.

- **BIRD single-step SFT learning gate passed (2026-07-17)**: the completed Qwen2.5-7B adapter
  trained from 40 verified episodes / 262 last-turn-only targets was tested independently of the
  active 1,303-target run. On 117 teacher-forced states from 20 unseen train episodes, it produced
  116/117 strict-valid actions, matched the teacher tool 91/117, and exactly matched tool+arguments
  45/117; the base control produced 0/117 strict-valid actions. On a fixed, disjoint BIRD-dev 30
  sample (12/9/9 simple/moderate/challenging, 11 DBs), exact historical prompt and pre-R2 full
  observations, the adapter reached **6/30 execution-correct and 24/30 legal terminal answers**
  versus base **0/30 correct and 0/30 legal**. Successes were 3/12 simple, 1/9 moderate, and 2/9
  challenging; 18 legal-but-wrong answers show the remaining bottleneck is semantic planning and
  arguments, not protocol acquisition. Report: `src/sft/BIRD_SINGLE_STEP_SFT_GATE_REPORT.md`.

- **BIRD dev-30 difficulty/scoring audit (2026-07-18)**: epoch-1's apparent challenging > simple
  rate is not meaningful: both buckets have exactly three strict successes (3/9 vs 3/12), Fisher
  exact two-sided `p=1.0`, with very wide intervals. Labels are copied verbatim from official BIRD,
  but they do not equal current tool difficulty: challenging success 513 is a single-table
  filter/order task whose visible evidence gives the mapping directly, while the sampled simple
  bucket includes a three-join aggregate and tool-awkward date arithmetic. Of nine simple failures,
  eight are genuine wrong/incomplete outcomes. Case 1462 is a contract-boundary false negative for
  human semantics: its reason states all four correct category/amount pairs, but it cites an
  unprojected seven-column evidence table with empty structured answer. Strict scoring remains
  unchanged; counting 1462 diagnostically gives simple 4/12 = challenging 3/9 = 33.3%. Report:
  `src/sft/BIRD_DEV30_DIFFICULTY_SCORING_AUDIT.md`.

- **v0 SFT done** (Qwen2.5-3B LoRA): exec-acc **63.8%** on full Spider dev vs text-to-SQL parity line
  **64–66%** vs base+few-shot **4%**. Conclusion: SFT = behavior cloning of interface + canonical
  plans (NOT a contribution by itself); model is feedback-blind (e.g. filters `country='French'`,
  gets 0 rows, ignores it). The RL/process-reward delta is the actual research story.
- **v1 data done**: 6773 train + 998 dev execution-verified+legal trajectories, **268 with
  `add_to_memory`** (scalar subqueries), think-filled by the external LLM (99.7%). Compile coverage
  99.9%, execution-verified **98.4%** on 2000q. System prompt updated with `add_to_memory` +
  `value_ref`/`in_table` predicate forms.
- **7B baselines done** on all 1,034 Spider dev examples with Qwen2.5-7B-Instruct:
  zero-shot direct SQL **716/1034 = 69.25%**; strict two-shot tool use **40/1034 = 3.87%**.
  The tool run reached only 78 legal final answers. Its dominant failure is formatting:
  950 final `protocol_error`s, and 954 trajectories omitted a complete
  `<tool_call>...</tool_call>` block at least once (usually the closing tag). Do not reinterpret
  this score as pure table-reasoning failure or silently relax the parser when comparing to SFT.
  Full model inputs/outputs and per-case success/failure JSON live under
  `data/results/qwen2.5_7b_baselines/` (gitignored).
- **7B / v1 SFT done** (2026-06-13 13:03 Asia/Shanghai): Qwen2.5-7B QLoRA completed 2 epochs /
  830 optimizer steps in 13:01:20. Final train loss is **0.2495**, validation loss **0.2271**.
  Adapter output is `checkpoints/qwen2.5-7b-spider-v1-qlora`. The viable 24G configuration is
  `cutoff_len=8192`, LoRA rank/alpha 16/32, `paged_adamw_8bit`, bf16, gradient checkpointing,
  and effective batch 16. See `src/sft/EXPERIMENTS.md`.
- **7B / v1 zero-shot tool evaluation done** on all 1,034 Spider dev examples: **707/1034 =
  68.38%**, with 985 legal final answers (95.26%). Failures: 276 wrong answers, 42 execution
  errors, 8 protocol errors, and 1 max-steps case. On the 998 examples covered by verified v1 dev
  trajectories, accuracy is **705/998 = 70.64%**. The 36 examples outside v1 coverage score only
  **2/36 = 5.56%**. The 29-example `add_to_memory` subset scores **9/29 = 31.03%**, versus
  **696/969 = 71.83%** on the covered non-memory subset; this confirms the known scalar-memory
  semantics/provenance weakness. Full model I/O and separate success/failure artifacts live under
  `data/results/qwen2.5_7b_sft_v1/tool_zero_shot_dev1034/` (gitignored).
- **V2a memory + provenance DONE & verified (2026-06-14)**: the scalar-memory/provenance repair the
  old "Next" called for is implemented and replaces the v1 memory shape. `add_to_memory` is now
  harness-grounded — the model emits ONLY `{type:"derived_value", source_step_id}`; it never authors
  value/key/provenance. The harness extracts the scalar from the cited step, builds a deterministic
  `derivation`+`key`+`content`, assigns `memory_id = mem_<source_step_id>` and
  `authority:"harness_grounded"`. Predicates reference the `memory_id` via `value_ref`. Every step
  carries harness-authored `references`/`produces`; `emitter.backward_slice` reverse-derives the
  answer's dependency set (through memory too). Strict scalar-source validation rejects
  NULL/0-row/multi-row/multi-col → manifest `memory_reject_*` buckets (v2a dropped 4). New SHARED
  module `src/harness/memory_semantics.py` is called by BOTH emitter and rollout. Observation
  envelope is now `{step_id, status, output}` so the model copies a `step_id` as `source_step_id`.
  Verified: Spider dev replay **998/998**, strict per-tool schema 30,534 calls / **0** fail,
  backward-slice invariant **7767/7767**, unit 98+6+4, exec-verified **98.4%**, **v1 files byte-identical
  (untouched)**. Data (gitignored): `data/trajectories/spider_{train,dev}_v2{,_think}.jsonl` (6769+998,
  `schema_version:"v2a"`), SFT `data/sft/spider_v2_*` (6767+998, `protocol_hash:eedbb946aa0f2cb7`).
  Full write-up: `draft/v2a_memory_report.md`. Scope: V2a = scalar `derived_value` ONLY;
  `evidence_pointer` (V2b) and `plan`/`hypothesis` (V2c) are deferred, separately-ablated mixtures.
- **V2b memory removal + unified references DONE & verified (2026-06-23)**: `add_to_memory` is GONE as
  a tool/concept (it was a redundant wrapper — `memory_id == mem_<source_step_id>`, and the aggregate
  step already parked its scalar). A predicate's `value_ref` now cites the producing `step_id`
  DIRECTLY; the harness grounds the scalar from history with strict validation at resolve time
  (online: an illegal `value_ref` → `execution_error`, never a silent value). New SHARED
  `src/harness/provenance.py::build_references` builds typed `references` edges (`type=data|value`,
  structured `target`) for BOTH emitter and rollout; `backward_slice(reference_type=…)` is
  parameterized (default data+value). `memory_semantics.py` → `scalar_grounding.py` (just
  `extract_scalar` + neutral `ground_scalar_reference`; no memory_id/key/content/derivation).
  `supporting_memory_ids` deleted; `refine_memory` placeholder removed. PROTOCOL_VERSION v2a→**v2b**
  (hash `95c58d18ea4cca28`), `schema_version` v2-ctx→**v3**. Verified: run_all unit **98/98** + compile
  1998/2000, emit 296/300 verified, **online replay 296/296 = 100%** (incl. 5 scalar-subquery value
  edges), SFT build 296/296 with **0 memory residue**. SSOT + impl log: `draft/provenance_redesign.md`
  §5. Deferred (in SSOT §2): C (perception grounding edges + reward), F (subtable consolidation —
  empirically triggered by the first perception-SFT failure modes), D-3/D-4 (regenerate perception
  data + SFT).
- **V2d plan/context scaffolding (2026-07-05)**: `plan(ops)` is now a model-visible,
  harness-managed task-control tool for creating/updating/deleting subgoals. Plan state is not
  factual evidence: it cannot support `value_ref` or final answers and is excluded from data/value
  provenance slices. New data must not let the model author `result`, `conclusion`, `notes`,
  scalar/list/boolean values, or final answer values inside plan items. Model-visible plan item
  state is only `goal`, `status`, and `evidence` (plus item id): the model supplies `evidence` as a
  prior `step_id` string, and the harness expands it into `{step_id, tool, output}` from actual tool
  history in the environment snapshot. `evidence_step_id` is tolerated only as a legacy alias; do
  not emit it in new prompts/data. PROTOCOL_VERSION is **v2d-plan-evidence**. New shared
  `src/harness/environment_state.py` maintains resident context with plan items plus per-table/handle
  schema, inspected column domains, reads, and produced handles.
  Online rollout/RL env and offline SFT rendering can include this `state` snapshot in observation
  envelopes. The raw SQL→trajectory emitter now produces the verified relational backbone only; it
  does **not** mechanically inject `describe_table` / `inspect_column` / `read_subtable`. External-
  model enrichment owns natural plan wording/updates and perception-step insertion, while harness
  checks remain the trust boundary. `src/sft/enrich_plan.py` is the standalone plan-enrichment pass:
  it calls the external model for initial `plan` + updates, splices plan steps into a verified
  trajectory, then replays through the harness so `EnvironmentState` and terminal correctness are
  checked. Its `--dry-run-template` mode is only for local smoke tests, not final training data.
- **Spider scale/context audit (2026-07-08)**: current `data/spider_data` is classic Spider 1.x, not
  Spider 2.0-Lite. It contains 166 DBs / 873 tables; table rows have median 12, p90 100, p95 2240,
  p99 25575, max 510437 (`wta_1.rankings`). Classic Spider dev examples whose DB has any table
  over 500 rows are only `world_1`, `flight_2`, and `wta_1` (**262/1034** examples). Latest 7B
  external-rollout-v3 scores on that large-table subset: epoch2 original **142/262 = 54.20%**,
  epoch4 original **148/262 = 56.49%**; retrying only context/execution-failure cases with larger
  limits raises the approximate merged scores to **59.16%** and **61.83%** respectively. Context
  overflow is dominated by `read_subtable` row payloads duplicated between historical tool
  observations and `CURRENT ENVIRONMENT STATE`; plan state is small (~1% of overflow payload).
- **V2g state-only model context switch (2026-07-13)**: model-visible context is now rebuilt from
  the resident `EnvironmentState` before each assistant turn instead of accumulating a full
  assistant/tool-observation transcript. `PROTOCOL_VERSION` is **v2g-state-only**. SFT records render
  as `human,gpt,human,gpt,...`: the first human turn is the catalog+question, later human turns are
  `CURRENT ENVIRONMENT STATE` plus optional `LAST TOOL ERROR`; there are no `observation` role turns
  and no `observation_tag` in new dataset registries. The emitter and external-rollout generator
  write `environment_state_before` and post-step `environment_state`; SFT must use the before-state
  for the current assistant step, or the previous step's after-state only as legacy fallback. Online
  eval, pass@k filtering, external rollout generation, and accelerate RL all call the shared
  `model_context_messages()` renderer, so old tool outputs may remain in debug artifacts but must not
  enter `model_input`. `EnvironmentState` also stores scalar `values` and deduplicates repeated
  `read_subtable`/inline row reads. Clean-switch gates used here: SFT unit tests, harness `run_all`,
  Python compile check, sample SFT render audit, and grep for `with_environment_state`,
  `observation_tag`, and `from:"observation"` returning no active-path matches.
- **Spider 2.0-Lite local adapter prepared (2026-07-08)**: official `xlang-ai/Spider2` is downloaded
  under `data/spider2/Spider2` (gitignored), and official `local_sqlite.zip` is unpacked into
  `data/spider2/Spider2/spider2-lite/resource/databases/spider2-localdb`. Lite has 547 examples:
  BigQuery 205, Snowflake 207, SQLite/local 135. The local zip contains 30 SQLite DB files and
  maps all 135 local examples. New adapter `src/harness/spider2_adapter.py` can summarize Lite,
  execute local gold SQL, run current compiler+tool round-trip smoke, and export normalized local
  records to `data/eval_inputs/spider2_lite_local.jsonl`. Dataset adapters should convert examples
  to the common `DatasetTask` IR (`src/harness/dataset_ir.py`) first; SQL-to-tool compilation is now
  used only as an environment-support/coverage check, not as the default SFT construction path.
  Current smoke: all 30 local SQLite DBs pass basic `describe_table`/`read_subtable` tool access;
  official local gold SQL is available for 24 local examples and executes **24/24** directly; current
  SQL compiler/tool round trip covers only **2/24**, mainly failing on CTE/derived CTE handles and
  `USING`/complex joins. Direct SQL baseline for the released-gold local subset lives at
  `src/eval/text2sql_spider2_lite.py`.
- **BIRD training environment prepared (2026-07-14)**: BIRD is the primary source for large-database
  tool-exploration and train-only SFT/RL trajectory construction; Spider2-Snow remains the external
  held-out benchmark. Official `birdsql/bird23-train-filtered` supplies **6,601** train annotations
  across **69** databases under `data/bird/train_filtered/`; the official BIRD archive is unpacked
  under `data/bird/train/train_databases/` with **69/69** SQLite DBs (**32.4 GB**). New
  `src/harness/bird_adapter.py` normalizes the records to `DatasetTask`, runs harness tool smoke,
  executes train gold SQL only for train-side checks, and exports
  `data/eval_inputs/bird_train_filtered.jsonl`. `src/eval/rollout.py --tasks-json
  data/eval_inputs/bird_train_filtered.jsonl` now consumes adapter-provided SQLite paths without
  changing legacy Spider behavior. Verification at setup: tool smoke **69/69** DBs, 532 tables, no
  errors; first 100 train gold SQL executions **100/100**. The compiler/tool compatibility gate is
  deliberately adapter-owned: `bird_train_tool_compatible.jsonl` contains **5,915/6,601** train
  tasks whose SQL→current-tool-plan→execution result matched gold under a **5 s** hard per-task
  limit, spanning all 69 DBs. Its summary classifies the excluded 686 tasks (231 execution errors,
  326 timeouts, 52 mismatches, 70 compile errors, 7 adapter errors). Use this compatible file, not
  the unfiltered export, for SQL-compiled tool trajectories; the core compiler/harness remains
  dataset-neutral. Do not use BIRD dev or Spider2 gold information to construct training trajectories.
- **BIRD SFT-1 teacher pilot (2026-07-14)**: fixed **30** BIRD-train-compatible candidates are saved
  at `data/eval_inputs/bird_train_sft1_pilot30.jsonl` (seed 42, **12/9/9** easy/medium/hard, **30
  distinct DBs**). Official BIRD train has no difficulty field; the manifest records the deterministic
  selection-only SQL-structure proxy (easy score <=2, medium 3--5, hard >=6); labels never enter
  teacher context. Protocol is now **v2h-state-only-evidence** (`b791b3fdb8ef3963`): optional BIRD
  `external_knowledge` is rendered by the shared state-only renderer, and every online/step-SFT turn
  is one user message containing catalog+question+evidence+current state+optional last error. The
  initial raw 30 first-turn audit is preserved under `bird_sft1_teacher_pilot30_{all,failures}.jsonl`.
  A runner gate initially rejected 15 tool-call-only first outputs despite the current shared parser
  accepting them; the canonical continuation reuses those exact saved first outputs (never reissues
  them), executes them in the harness, and only then calls the teacher for later turns. Canonical
  continuation results: **4/30 = 13.33%** replay-correct episodes (easy **2/12**, medium **1/9**,
  hard **1/9**), 21 protocol failures and 5 wrong answers. Success source:
  `data/trajectories/bird_sft1_teacher_pilot30_continued_success.jsonl`; all/failures/validation
  siblings retain every first-pass outcome. The one-step export at
  `data/sft/bird_sft1_teacher_pilot30_continued_step_train.jsonl` has **35** records (easy 16,
  medium 6, hard 13), indexed by `...continued_step_index.jsonl`; each record is one human/gpt pair
  rendered from state-before, with current output/future ids/gold SQL excluded. Replay and no-leak
  audit pass. LLaMA-Factory preprocessing smoke on table_rl (`sft` env, Qwen2.5-7B tokenizer) passes
  all 35: labels mask human tokens, decoded targets match, token min/p50/p90/max **1671/2620/3519/4096**,
  none exceed cutoff 4096. Generator/export scripts are `src/sft/bird_sft1_teacher.py`,
  `src/sft/build_bird_sft1_steps.py`, and `src/sft/select_bird_sft1_pilot.py`; do not treat this
  tiny pilot as a final training mixture or perform hidden semantic retries.
- **Same-episode error recovery (2026-07-14)**: all active online paths (`src/eval/rollout.py`,
  `src/eval/rollout_passk.py`, `src/sft/rollout_external_data.py`, and the BIRD teacher) use the
  strict no-repair parser for generated actions. `protocol_error`, `argument_validation_error`, and
  state-preserving `execution_error` consume the shared action budget but retain the episode and
  appear as structured `LAST TOOL ERROR` on the next state-only input. Each error type is capped
  independently (default 3); mutated-state execution failures are terminal. Error actions stay only
  in `turns`/`error_events` with state hashes and are never SFT targets. The first following legal
  step is tagged `feedback_recovery`; verifier-correct episodes are `clean_success` or
  `recovered_success`. API transport retries remain client-side and separate from semantic recovery;
  full restarts are distinct attempts for success@k reporting. Existing BIRD pilot artifacts remain
  immutable historical records and were not regenerated under this policy.
- **BIRD strict-recovery rerun (2026-07-14)**: independent full attempt `recovery_v7` ran all
  **30/30** fixed BIRD-train SFT-1 candidates with the external teacher, strict no-repair parsing,
  `max_steps=30`, and three recoverable errors per class. It produced **0** legal actions / verified
  successes and therefore **0 SFT targets**: terminal failures are 28 `protocol_error` and 2
  `argument_validation_error`. The audit retains **128/128** state-preserving error events, so the
  no-success result is protocol-format compliance, not silent parameter repair or a restart. Full
  all/failures/summary artifacts are `data/trajectories/bird_sft1_teacher_pilot30_recovery_v7_*`
  (gitignored). Earlier partial `v2`--`v6` attempt artifacts are preserved only as interrupted
  infrastructure/parser audits and must not be aggregated with `v7`.
- **BIRD ChatGPT strict-recovery test (2026-07-14)**: a separate full 30-task run with the
  api.md-configured `gpt-5.6-sol` model (`chatgpt_v1`) yielded **6/30 replay-verified
  `recovered_success`** episodes, all easy; no clean successes. It retained 85 error events:
  **84/84** recoverable (`protocol_error`/`argument_validation_error`/`execution_error`) events
  preserved state, while one state-changing `nonrecoverable_execution_error` was terminal. The six
  accepted trajectories contain 10 `feedback_recovery` steps; 47 legal recovery steps occurred
  across all complete/failing records. There were no API transport retries. Complete all/success/
  failures/summary artifacts are `data/trajectories/bird_sft1_teacher_pilot30_chatgpt_v1_*`
  (gitignored); do not merge its 24 failure records into SFT targets. This **20% is not a semantic
  BIRD acquisition estimate**: 21/24 terminal failures were protocol failures, 77/85 error events
  were protocol errors, and 74/77 of those contained **2--11 complete** think/tool-call pairs rather
  than malformed JSON. Only six episodes reached `answer_from_context`, and all six were correct;
  the dominant issue was the teacher emitting an open-loop multi-action suffix per request.
- **Canonical BIRD interactive generator (2026-07-14)**: subsequent BIRD external-model rollouts
  must use the existing `src/sft/rollout_external_data.py`, not the earlier pilot-only
  `bird_sft1_teacher.py`. The canonical generator now accepts common `DatasetTask` JSONL via
  `--examples-file`, uses adapter-provided `db_path`/`gold_sql` plus `external_knowledge`, and
  retains the same strict state-only recovery semantics. Its prompt now states `ONE REQUEST = ONE
  ACTION` and requires an immediate stop after the single `</tool_call>`; strict parser feedback
  reports the actual number of complete call blocks. On the same fixed BIRD 30 with `gpt-5.6-sol`,
  canonical run `bird_chatgpt_original_v1` achieved **16/30 = 53.33%** replay-verified success
  (5 clean, 11 recovered), with **40/40** recoverable errors state-preserving and 13 recovery steps
  in accepted trajectories. Multiple-tool protocol-error turns dropped from 77 in the earlier pilot
  to 25 (still a live failure bucket). Artifacts are
  `data/trajectories/bird_chatgpt_original_v1_{success,all,failures}.jsonl` plus its manifest
  (gitignored). API transport retries: 2. Error actions remain excluded from SFT targets.
- **Join name-management repair (2026-07-14)**: the environment had a model-visible contract bug:
  `join_tables` documentation permitted a declared first-edge prefix such as `G__id`, but the
  executor only accepted raw `id` before it materialized prefixes, yielding misleading internal SQL
  errors like `L.G__id`. `Harness.join_tables` now accepts only the documented declared
  `P1__column` form on that first left edge; it still rejects `L.`/`R.` and table-qualified names,
  so no arbitrary model parameter is silently repaired. `format_tool_error` now gives the exact
  model-facing identifier rule plus schemas. Verified on the earlier BIRD Pengo failure: the
  `G__id/GP__id/GPL__id/RS__region_id` chain now returns the gold 4 rows. Protocol is
  **v2i-state-only-join-feedback**; prior v2h trajectory artifacts remain historical and must not
  be mixed without recording the protocol change. Unit coverage includes documented first-edge
  prefix acceptance and internal SQL alias rejection.
- **Canonical execution contract (2026-07-14)**: `tool_design/canonical_execution_contract.md` is
  the concise SSOT for new episodes: exactly one strict think/tool-call action, state-only next-turn
  context, harness-owned execution/provenance, and audit-only error events. `parse_assistant` plus
  `aggregate` and old two-table join fields are replay compatibility only; `parse_assistant_strict`
  rejects them for new SFT/eval/RL actions. `src/rl/env.py` now uses the same strict parser and
  recovery contract as live evaluation: every error spends an action, carries a before/after state
  hash, is capped per error type, is sent back as `LAST TOOL ERROR`, and the first later legal action
  is marked `feedback_recovery`. It uses adapter-provided `db_path`/`gold_sql`, so the same episode
  implementation runs Spider and BIRD. Keep v2i `prefixes`/`P__column` intact until a fully atomic
  v2j dotted-source-name migration regenerates the compiler, state, data, and tests.
- **DS Flash v4 strict BIRD smoke (2026-07-14)**: fixed 30-task BIRD run
  `bird_ds_flash_v4_v2i_smoke30` completed all **30 unique** tasks (12/9/9 easy/medium/hard) under
  strict v2i. It yielded **0 legal / 0 verified trajectories**: DeepSeek returned a valid
  `<tool_call>` in `message.content` but placed its reasoning only in the provider-side
  `reasoning_content`, so every turn correctly failed the required non-empty `<think>` check.
  The run retained 90 state-preserving protocol-error events (three per unique task); raw audit has
  33 records because a resume race duplicated three attempts, which remain preserved rather than
  overwritten. The manifest now reports both raw attempt records and unique examples by aggregating
  persisted `all.jsonl`, fixing misleading resume-only counts. Do not wrap `reasoning_content` in
  synthetic `<think>` tags: that would violate strict no-repair semantics. Artifacts are gitignored
  under `data/trajectories/bird_ds_flash_v4_v2i_smoke30_*`.
  The generator prompt now additionally says that `<think>` must appear in visible
  `message.content` and that provider reasoning/metadata does not count. A fresh two-task probe
  still produced content-only tool calls and failed 2/2, so this is a provider output-channel
  constraint rather than an omitted instruction. Do not use raw DS Flash v4 content directly for
  strict SFT generation; use the audited provider adapter and still reject missing carrier fields.
- **DS Flash v4 transport-adapter rerun (2026-07-14)**: `src/sft/provider_adapter.py` now treats
  DS's legitimate split carrier as transport, not parser repair: only non-empty
  `reasoning_content` + an otherwise complete, think-free visible tool-call is rendered as canonical
  `<think>reasoning</think> + <tool_call>`. Raw content/reasoning and the adapter decision remain in
  `all.jsonl`; malformed/empty-carrier shapes still fail strict parsing. Fixed BIRD-30 rerun
  `bird_ds_flash_v4_adapter_smoke30` produced **7/30 = 23.33%** verified successes (5 easy, 2
  medium; 5 clean + 2 recovered), replayed **7/7**. Failures: 7 wrong answers, 16 protocol errors;
  all attempts had 101 adapted actions and 77 unusable DS carrier turns. Success trajectory metadata
  records adapter applied/unusable action counts. This validates the adapter but also shows DS does
  not consistently populate `reasoning_content`; do not pretend a missing reasoning field is valid
  by synthesizing a think target.
- **DS Flash explicit-contract diagnosis and token-limit control (2026-07-14)**: the fixed BIRD-30
  `bird_ds_flash_v4_explicit_contract_clean30` reached **9/30** verified successes, but 10 episodes
  still terminated after the per-class protocol-error limit. Their 30 terminal error turns are 11
  visible-text prefixes before a tool block, 10 unclosed `</tool_call>` blocks, and 9 missing
  `reasoning_content`; these are output-carrier failures, not tool-argument repairs. The canonical
  generator now records the API `finish_reason` on each turn and a precise adapter rejection reason;
  it does not auto-complete an XML tag or synthesize a missing reasoning field. A same-10-task
  counterfactual with a DS-only replacement prompt and `--max-tokens 2048` yielded 0/10 and all 30
  carrier errors were visible-text prefixes with `finish_reason=stop`; replacing the shared
  visible-`<think>` wording made Flash worse and was reverted. A proper same-prompt, same-10,
  2048-only control then improved from **0/10** to **2/10 recovered_success**, with six further
  legal-but-wrong terminal answers and only two protocol terminals. Its 13 rejected carrier turns
  had 11 `stop` and 2 `length` finish reasons, so truncation is not the sole error source but the
  larger budget materially improves usable action yield. `rollout_external_data.py` now defaults DS
  Flash/Pro to **2048** when `--max-tokens` is omitted; an explicit CLI value still wins, and other
  providers retain 1024. Keep the prior appended split-carrier contract plus the new audit fields.
  The next candidate is a separately probed native structured/tool-call transport, never a parser
  relaxation.
- **DS Flash 2048 BIRD-30 re-smoke (2026-07-14)**: the fixed 30-task pilot was rerun with the
  original appended split-carrier prompt, strict parser/recovery, and the new provider-default
  2048 budget. The persisted manifest has exactly **30 unique** attempts (the runner's default
  `--limit` is 20, so full fixed-pilot runs must pass `--limit 30` explicitly): **13/30 = 43.33%**
  execution-correct, up from the earlier 1024 run's 9/30. Successes are easy 7/12, medium 4/9,
  hard 2/9; 6 are clean and 7 are `recovered_success`. Terminal failures are 10 wrong_answer and
  7 protocol_error; hence among failures, semantic wrong answers are **58.8%** and protocol
  terminals **41.2%**. The 269 model turns include 239 adapter-carried actions and 30 rejected
  carrier turns: visible prefix 15, incomplete/suffixed tool call 11, visible think tag 4; all 30
  report `finish_reason=stop`, with one additional state-preserving execution error. This is an
  operational improvement, not a deterministic causal estimate because hosted-model outputs can
  vary between calls. Artifacts: `bird_ds_flash_v4_tokens2048_smoke30_*` (gitignored).
- **AIHubMix reasoning-continuation transport gate (2026-07-14)**: a two-turn no-BIRD-data probe
  against `deepseek-v4-flash` at the configured AIHubMix endpoint returned only `content`,
  `reasoning_content`, `refusal`, and `role`; it did **not** return `reasoning_details`. Thus the
  AIHubMix documented `reasoning_details` pass-through/interleaved-thinking route is not verified
  for this DS Flash transport and must not be synthesized or enabled in the canonical rollout.
  The reproducible probe is `src/sft/probe_reasoning_continuity.py`, with its gitignored report at
  `data/results/aihubmix_ds_flash_reasoning_continuity_probe.json`. Any future comparison of a
  bounded client-side legal-history context must be an isolated experimental protocol, not a claim
  of provider-managed memory and not a replacement for state-only SFT/RL without matched student
  training and evaluation.
- **Rolling legal-history context probe (2026-07-14)**: `rollout_external_data.py` now has an
  opt-in `--context-mode rolling-legal-history --history-turns N` experiment (default remains
  state-only). It sends the initial user task, prior **harness-successful** assistant
  think/tool-call messages and their tool-result envelopes, then the current full environment
  state. Rejected assistant text is excluded; its `LAST TOOL ERROR` remains in current state.
  Success trajectories from this mode are explicitly `sft_export_eligible:false` until a matched
  student input format exists. A fixed first-5 BIRD probe with all prior legal turns (`N=0`) got
  **5/5** execution-correct clean successes versus 3/5 in the earlier state-only run on those task
  ids; mean legal-think length fell 537→333 chars, total turns 25→22, and protocol events 2→0.
  Input tokens increased 109,436→152,933 (+40%). This is a promising but non-deterministic,
  underpowered signal, not a protocol replacement. Artifacts:
  `bird_ds_flash_v4_rolling_history_probe5_*`; unit coverage is
  `src/sft/tests/test_rolling_legal_history.py`. The next valid comparison is a 30-task bounded
  (e.g. 4 recent legal turns) test with matched state-only evaluation and a context-cost report.
- **DS Flash rolling legal-history BIRD-30 smoke (2026-07-14)**: the requested full fixed-pilot
  run used the same strict parser/recovery, 2048 budget, and all prior legal turns (`N=0`). It has
  exactly **30 unique** single attempts and yields **14/30 = 46.67%** execution-correct: easy
  7/12, medium 4/9, hard 3/9; 7 clean and 7 `recovered_success`. Against the separate
  state-only 2048 run (13/30), protocol terminal failures dropped 7→1 and audit error events
  dropped 31→19 (18 protocol, 1 state-preserving execution), but terminal semantic wrong answers
  rose 10→15. It therefore improves format/continuation behavior rather than demonstrating a
  causal table-reasoning gain. It used 248 vs 270 requests and 1,054,693 vs 951,737 prompt tokens
  (+10.8%; 612,736 cached), while reasoning tokens fell 57,407→24,029. Strictly parsed error
  actions are audit-only; 18 first legal post-error actions carry `feedback_recovery`. The
  trajectory manifest declares `sft_export_eligible:false`; do not mix it into state-only SFT.
  Artifacts: `bird_ds_flash_v4_rolling_history_smoke30_*` (gitignored).
- **Paired BIRD context-100 checkpoint (2026-07-14, intentionally paused)**: a deterministic
  `40/30/30` easy/medium/hard train-compatible selection (seed `314159`, 69 DBs) is saved as
  `data/eval_inputs/bird_train_context_compare100.jsonl` with its manifest. The selector now
  accepts `--total` (positive multiples of 10, exact 4:3:3); its default 30-task behavior is
  unchanged. The strict DS Flash 2048 `state-only` baseline was deliberately interrupted for a
  host shutdown after **81/100** independently persisted records: **31** execution-correct and
  **50** failures. Its `all`, `success`, and `failures` files are
  `bird_ds_flash_v4_context100_state_only_*` (gitignored); the final manifest does not yet exist
  because the run was interrupted. Resume exactly this baseline with the original command plus
  `--resume` and the same `--limit 100 --workers 1 --max-steps 30 --max-errors-per-type 3
  --api-timeout 300 --api-retries 3 --context-mode state-only`. Do not start the paired
  `rolling-legal-history --history-turns 4` run until that baseline reaches 100 unique task ids.
- **Paired BIRD context-100 result (2026-07-15)**: the paused state-only baseline was resumed
  without duplicate attempts, then the identical 100 train tasks were run once under strict
  `rolling-legal-history --history-turns 4`. Both artifacts have 100 unique records and every
  accepted episode replayed from the initial database (**35/35** state-only, **47/47** rolling).
  Rolling reached **47/100 = 47%** versus state-only **35/100 = 35%**: easy 19/40 vs 15/40,
  medium 17/30 vs 13/30, hard 11/30 vs 7/30; clean/recovered are 28/19 vs 22/13. Pairwise, 23
  tasks pass in both, 24 switch state-only-fail→rolling-pass, 12 switch the other way, and 41 fail
  in both (exact McNemar two-sided 0.065; directional evidence only because this is one stochastic
  hosted-model draw per mode, not a deterministic causal estimate). Legal terminal answers rose
  73→89 and protocol terminals fell 27→9; total audit events fell 143→91 (protocol 124→80,
  execution 17→9), but semantic wrong-answer terminals rose 38→42 and rolling has 2 execution
  terminals. Rolling used 840 vs 895 requests; prompt tokens were effectively equal (3.141M vs
  3.143M), while completion/reasoning tokens fell 246,127→138,219 and 201,396→93,855. Thus a
  bounded legal history looks promising for continuation/format robustness, but does not itself
  cure semantic table reasoning. Artifacts are `bird_ds_flash_v4_context100_{state_only,rolling4}_*`
  (gitignored). Rolling success files remain `sft_export_eligible:false`: do not train on them until
  the renderer/exporter/loss-mask/runtime protocol is made consistent and separately tested.
- **Rolling SFT/inference pipeline smoke (2026-07-15)**: bounded rolling history is now a shared
  protocol renderer in `src/sft/protocol.py` (`ROLLING_CONTEXT_VERSION=v1-bounded-legal-history`),
  with a rolling-specific system suffix that says the bounded legal transcript is continuity context
  while state/error feedback remains authoritative. `rollout_external_data.py` and
  `src/eval/rollout.py` accept `--context-mode rolling-legal-history --history-turns N`; only
  successfully executed assistant/tool pairs enter history. New
  `src/sft/build_rolling_sft_data.py` replays every accepted source episode, reconstructs the
  bounded student online prefix for each legal target, rejects current/future-step and gold-SQL leaks, and
  emits ShareGPT prefix records whose final assistant action is the only intended label. The
  LLaMA-Factory 0.9.5 environment on NewGNN supports `mask_history: true`, which masks historical
  assistant turns while retaining their chat roles. The 47 verified BIRD rolling-4 successes
  replayed 47/47 during export and produced **334** target records (27 feedback-recovery targets);
  Qwen3.5-9B preliminary content-token filtering at 4096 retained **301** and dropped 33
  overlength prefixes. This filter does not include chat-template overhead, so it is not the final
  no-truncation gate; the scale-up gate must audit LLaMA-Factory's actual encoded token/label
  lengths and preserve every final target. Two-step 4-bit LoRA smoke on an idle RTX 3090 completed with
  loss 0.3759→0.2532 (train loss 0.3145) and saved
  `/home/dengyan/tabular_rl_outputs/checkpoints/qwen3.5-9b-bird-rolling4-smoke-qlora`. A temporary
  vLLM LoRA service plus local BIRD harness ran two true rolling closed-loop inference episodes;
  both terminated protocol_error after three errors (expected after two updates), but each retry
  received structured `LAST TOOL ERROR`, confirming adapter→OpenAI API→local rolling runner→harness
  transport. This is an infrastructure smoke, **not** an accuracy result. Artifacts:
  `data/sft/bird_ds_flash_v4_rolling4_smoke_train*`, config
  `src/sft/configs/qwen3.5_9b_qlora_bird_rolling4_smoke_table_rl.yaml`, and gitignored inference
  audit `data/results/qwen35_bird_rolling4_smoke_inference2/`. Do not scale until a quality filter
  and 4k overflow policy are finalized; rolling data must remain separate from state-only mixtures.
- **Rolling safe-compact prompt ablation (2026-07-15)**: the old generic compact prompt omitted
  v2i-critical `value_ref`/`in_table`, `P__column` join naming, and recovery rules, so it must not
  be used for rolling training. `protocol.py` now provides the opt-in
  `ROLLING_COMPACT_PROMPT_VERSION=v1-safe-compact`, retaining the strict one-action carrier,
  state/history/error authority, plan boundary, value references, and documented join identifiers
  while reducing the rolling system prompt from **6,363 to 2,211 characters (65.3%)**. The default
  state-only and full rolling prompts are unchanged. Generator/eval/export each record
  `--rolling-prompt-variant full|compact`; pair source rollout and student export/evaluation with
  the same variant for scale. Existing rolling-4 successes replayed **47/47** and export unchanged
  **334** targets (27 recovery) under compact; prefix characters p50/p90/mean fell
  **12,633/16,075/12,559 → 8,481/11,923/8,407**. Compact smoke files are gitignored under
  `data/sft/bird_ds_flash_v4_rolling4_compact_smoke_*`; Qwen2.5-7B config is
  `src/sft/configs/qwen2.5_7b_qlora_bird_rolling4_compact_4k_table_rl.yaml` and requires
  LLaMA-Factory `mask_history: true`. This is ready for a 7B preprocessing/training smoke, but
  exact template-token overflow filtering remains a required gate before scaling.
- **Qwen2.5-7B compact rolling train40 + held-out smoke (2026-07-15)**: deterministic seed
  `20260715` splits the 47 replay-verified rolling successes into **40** train episodes
  (easy/medium/hard **16/15/9**) and 7 disjoint holdouts (**3/2/2**) via
  `src/sft/select_rolling_sft_split.py`. The train export has **291** step targets including 24
  feedback-recovery targets. The new exact, installed-LLaMA-Factory audit
  `src/sft/audit_rolling_lf_tokens.py` encodes the real Qwen2.5 `qwen` template with
  `mask_history:true`: all **291/291** final targets are intact at cutoff 4096 (encoded p50/p90/max
  **2196/3107/4096**); no target was silently truncated. On table_rl GPU0, 4-bit rank-16 QLoRA
  completed 4 epochs / 76 steps in **56:28** with final train loss **0.6166**; final adapter is
  `/home/dengyan/tabular_rl_outputs/checkpoints/qwen2.5-7b-bird-rolling4-compact-train40-4k-qlora`.
  A temporary GPU1 vLLM plus local BIRD harness evaluated the matching compact rolling protocol on
  the seven holdouts. First-attempt, unique-task result is **3/7 = 42.86%** execution-correct,
  **7/7 legal final answers**, no protocol/API terminal failures, and four semantic `wrong_answer`s
  (one error-feedback recovery occurred before a final wrong answer). Main audit is gitignored at
  `data/results/qwen2.5_7b_bird_rolling4_compact_train40_holdout7_v2/`; it also retains later
  resume duplicates as audit-only records, so report the first-attempt unique-task statistic rather
  than raw-line accuracy. The earlier non-escalated `...holdout7/` directory contains sandbox
  `api_error`s only and is invalid for model scoring. Temporary vLLM/tunnel processes were stopped;
  both table_rl GPUs were verified released. This is a tiny behavior/pipeline smoke, not a reliable
  generalization estimate or a scale-up decision.
- **BIRD rolling-4 review gate (2026-07-15)**: the experiment console's Data Construction view
  now exposes `bird_ds_flash_v4_context100_rolling4_success_review.jsonl` (47 execution-verified
  normalized candidates) and `..._all_review.jsonl` (all 100 raw rollout attempts). Raw rollouts
  render their actual turns, structured error events, and the first legal feedback-recovery action.
  `src/sft/audit_verified_rollouts.py` reports **47/47** verified episodes, **334** legal targets,
  **27** recovery targets, and no structural or gold-SQL-leak violations. It also records **30**
  excluded semantic-error events (26 protocol, 2 argument-validation, 2 execution). The historical
  teacher uses the **full** rolling prompt, whereas the intended scale configuration uses
  **compact**, so the prompt-variant gate fails. Human spot checks found correct but inefficient
  wandering, excessive plan updates, and occasional stale/malformed recovery reasoning. Do not
  start unrestricted large-scale generation: first run a compact-prompt, stratified quality pilot;
  audit its exact template tokens and use a review filter before scaling.
- **BIRD compact rolling-4 paired pilot (2026-07-15)**: the requested same-task, stratified
  compact-prompt pilot completed. Canonical **first-attempt** results are **10/100** verified and
  **35/100** legal (easy/medium/hard **5/40, 5/30, 0/30**) versus full-prompt **47/100** and
  **89/100** legal. Pairwise, 9 pass in both, 38 full-pass→compact-fail, 1
  full-fail→compact-pass, and 52 fail in both. Compact reduced prompt tokens by **27.9%**
  (3.141M→2.263M) but is a severe quality regression, especially on hard tasks. The 10 successes
  pass structural/provenance audit (64 legal targets, 6 feedback-recovery targets; compact variant
  gate passes). Four same-attempt duplicate raw records from local executor overlap are retained
  audit-only; `*_first_attempt_{all,success}.jsonl` is canonical. Review files are
  `bird_ds_flash_v4_context100_rolling4_compactpilot100_{all,success}_review.jsonl`; the comparison
  is `..._comparison.json`. **Do not scale the compact prompt.**
- **Full-prompt 8k preparation (2026-07-15)**: compact reduced the rolling system prompt from
  **6,363** to **2,211** characters (**65.3%**) and full exported-prefix p50 from **12,633** to
  **8,481** characters (**32.9%**), but the paired quality collapse means keep the full system
  contract. The teacher-matched full prompt is unchanged (historical/current system SHA-256 both
  `9db347...28058`). `data/sft/bird_ds_flash_v4_rolling4_full_train40.jsonl` replays **40/40**
  source episodes into **291** last-turn-only targets (24 recovery) and
  `qwen2.5_7b_qlora_bird_rolling4_full_8k_table_rl.yaml` raises only `cutoff_len` to **8192**.
  The real table_rl SFT-environment audit using the Qwen template passed **291/291** intact targets:
  encoded length min/p50/p90/max **1656/3216/4127/6390**, with **0** final-target truncations.
  The retained remote data and local audit manifest are
  `bird_ds_flash_v4_rolling4_full_train40_qwen25_8k.jsonl` and its `.token_audit.json`; training is
  ready to launch but has not been started.
- **BIRD scale-500 teacher construction (2026-07-15, in progress)**: the fixed train-only candidate
  set is `data/eval_inputs/bird_train_sft1_scale500.jsonl` (seed `20260715`), with **500** unique
  compatible tasks across all 69 databases and the proxy split **200/150/150** easy/medium/hard.
  The selection excludes the historical context-100 train pool (zero overlap), though that pool is
  no longer an evaluation set. Generation uses the proven strict DS Flash 2048, full rolling-history
  (`N=4`) protocol in the local tmux session `bird_scale500_flash`, with resumable all/success/failure
  artifacts named `bird_ds_flash_v4_scale500_rolling4_full_*`. Future model evaluation is on
  `data/eval_inputs/bird_dev_20240627.jsonl`; never use BIRD dev gold to construct trajectories.
- **Scale-500 first completion + parallel gate (2026-07-15)**: first batch completed all **500**
  single attempts with **183/500 = 36.6%** verifier-correct (easy **103/200**, medium **57/150**,
  hard **23/150**); failures are 153 wrong answers, 6 recoverable execution errors, 156 protocol
  errors, and 2 state-changing execution errors. The full-prompt rolling success audit passes
  structural and variant gates: 1,172 legal targets, 76 recovery targets, no repeated identical
  calls. Quality selection via `src/sft/select_verified_rollouts.py` rejects 2 step-limit and 3
  think-limit episodes then deterministically selects **40** clean episodes (**16/12/12**) into
  `...scale500_rolling4_full_train40.jsonl`. The rendered SFT set has **262** records / 13 recovery
  targets; table_rl's exact 8k audit retains **262/262** (max 8192) with no target truncation.
  The subsequent Qwen2.5-7B full-prompt 8k QLoRA attempt on table_rl GPU0 did **not** complete:
  it reached 1/68 optimizer steps, then the second batch failed in causal-LM cross entropy with a
  4.64 GiB allocation request while only 4.21 GiB was free. The output directory
  `qwen2.5-7b-bird-scale500-train40-rolling4-full-8k-qlora` is empty and contains no usable adapter;
  both GPUs were later verified released. This was an 8k/logits memory peak, not a laptop-sleep or
  tmux interruption. Do not report inference results for this failed attempt. A memory-safe retry
  uses the separate config
  `qwen2.5_7b_qlora_bird_scale500_train40_rolling4_full_6400_table_rl.yaml`: the real Qwen template
  audit at cutoff 6400 retains **262/262** final targets (0 dropped; length min/p50/p90/max
  1723/3310/4387/6400, target max 628), and the audited JSONL is byte-identical to the registered
  training file. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is explicit in the dedicated
  launcher. Remote run id `bird_scale500_train40_full_6400_20260716_202420` completed **4 epochs /
  68 optimizer steps** on 2026-07-16 21:43 Asia/Shanghai in **4730.6 s**. Final average train loss
  is **0.55586**; late logged losses are roughly 0.38--0.45. The final adapter and epoch checkpoints
  are under `qwen2.5-7b-bird-scale500-train40-rolling4-full-6400-qlora` (top-level adapter plus
  checkpoints 17/34/51/68). This gate intentionally used `val_size:0` / no eval, so training
  completion is confirmed but model quality is not yet established. The first next gate is a
  matched rolling/full closed-loop evaluation on the disjoint seven-task BIRD holdout, followed by
  BIRD Dev only if legality and execution are sound. In parallel, the
  disjoint next candidate set
  `bird_train_sft1_scale500b.jsonl` (same 200/150/150 split, seed 20260716) is generating in local
  tmux `bird_scale500b_flash`; its artifacts use the `bird_ds_flash_v4_scale500b_rolling4_full_*`
  prefix. The rolling SFT exporter now treats future `step_n` mentions in historical assistant
  reasoning as plans, not state leaks; it still rejects future/current references in harness-authored
  user state/tool feedback. Regression coverage: `test_build_rolling_sft_data`.
- **Process-reward v1 implemented and audited (2026-07-15; RL no-go yet)**:
  `src/rl/process_reward.py` implements harness-replayed `B/E/S/F`, configurable penalties,
  linear positive/negative normalization, penalty capping, and exact reward conservation;
  `process_objective.py` implements the step-weighted policy objective and requires explicit KL
  values from a frozen SFT-2 reference when beta is positive. The fixed-root search term is limited
  to comparable filter/top-k operations and has direct telescope/empty/expansion tests. Pilot
  config is `src/rl/configs/process_reward_v1.json`; report CLI is
  `src/rl/build_process_reward_report.py`. On all **183** Scale-500 verified successes, harness replay
  is **183/183**, reward conservation passes, and correct totals remain positive. However only
  **79/183 = 43.2%** have strict final grounding (50 declared evidence + 29 denotation-matched
  handles); 103 explicit answers are ungrounded and one denotation is deliberately too large to
  materialize. There are 69 terminal-fallback trajectories and **35 ungrounded non-fallback** cases
  where feedback credit suppresses fallback; per-trajectory maximum positive share has p50/p90=1.
  Therefore `process_reward_ready:false`: do not launch process-shaped RL until final evidence/
  perception grounding reaches the gate (default 90%) and ungrounded non-fallback is zero. Full
  report: `data/rl/bird_scale500_success183_process_reward_v1/`; 9 reward/objective unit tests pass.
- **Process-reward v2 harness-owned grounding (2026-07-16; structural gate passed, precision gate
  pending)**: reward dependency no longer trusts or requires model-authored
  `answer_from_context.evidence`. The harness scans backward to the nearest prior table-bearing legal
  action for the final table, and shared `provenance.build_grounding_references` derives schema,
  inspected-domain, and row-value edges from legal action parameters plus harness-owned outputs.
  `BackSlice` now traverses data/value/grounding edges; used perception steps receive `B`, and first
  used schema/domain/row facts receive `E`. On the same 183 verified Scale-500 successes, replay is
  183/183, structural grounding rises **79→182 (99.45%)**, fallback drops **69→1**, positive/zero
  steps change **350/822→881/291**, and maximum per-episode positive share p50/p90 drops
  **1.0/1.0→0.271/0.422**. Perception slices: describe **196/200**, inspect **33/88**, read
  **184/213**; unused truncated observations remain zero. `bird_train_01915` now distributes reward
  over describe/filter/read/filter/read as 0.1856/0.1754/0.1856/0.2679/0.1856 instead of terminal
  +1. Future error events also preserve structured attempted tool/arguments. Structural gate passes,
  but `process_reward_ready:false` until a stratified manual inferred-edge precision audit approves
  it; nearest-last-table and common-literal matching can still over-connect. Current report:
  `src/rl/PROCESS_REWARD_V2_REPORT.md`; artifacts:
  `data/rl/bird_scale500_success183_process_reward_v2/`; 12 reward tests + 136 harness tests pass.
- **Scale500b positive+negative reward audit (2026-07-16; mechanism valid, config/data no-go)**:
  the second disjoint teacher batch completed **500/500** unique attempts with 139 verified
  successes, 127 wrong answers, 228 protocol terminals, one execution terminal, and five API
  transport failures. New `src/rl/external_failure_adapter.py` preserves legal turns and audit-only
  error events while excluding the five nonsemantic API failures, yielding 356 replayable failures.
  On 139 successes, replay/grounding is **139/139**, no fallback, reward mean 0.8878, and max positive
  share p50/p90/max is **0.277/0.422/0.600**; perception B/E is describe 148/152, inspect 22/53,
  read 131/141. All 356 failures match failure expectation and conserve reward; 889 steps are
  negative, with applied negative mass split terminal/tool-error/ignored-feedback
  **176.81/74.63/33.36**. Current `lambda_terminal_failure=1.0 >= Pmax=0.8` saturates every failed
  episode at **-0.8**, so extra errors change allocation but not episode severity. A noncommitted
  `lambda_fail=.4/tool=.15/ignored=.1` counterfactual moves wrong-answer mean to -0.4819 while
  error-heavy protocol failures remain capped. Data is also severely difficulty-skewed: easy
  **96/200**, medium **43/150**, hard **0/150** successes; 145/150 hard tasks end in protocol error.
  Do not launch process RL from this mixture. Tune terminal penalty below the cap, repair/regenerate
  hard protocol failures, and complete grounding-edge precision audit first. Full report:
  `src/rl/SCALE500B_REWARD_AUDIT.md`; artifacts under
  `data/rl/bird_scale500b_process_reward_v2/`. Tests: 14 RL adapter/reward tests pass.
- **External grounding audit (2026-07-16; process RL still no-go)**: all 322 replay-correct BIRD
  teacher successes were packaged without model think text and reviewed by DS Flash; every
  Flash non-pass, `common_literal` risk, plus a deterministic 30-case pass control was independently
  rechecked by DS Pro. The corrected v2 package includes trusted data/value provenance context:
  Flash gives 275 pass / 35 fail / 12 ambiguous with 1,065 valid / 20 invalid / 3 ambiguous local
  grounding edges; Pro completes 84/85 selected cases (50/27/7, one unresolved). They agree on 20
  trajectory fails; the pass control is 24 pass, 2 fail, 3 ambiguous, 1 unresolved. A later audit-
  display fix retains target-matching rows beyond the first-five preview; targeted v3 re-review
  reduces five double-invalid edges to two and turns one of five consensus failures into a model
  disagreement, leaving at least **19/322 = 5.90%** confirmed trajectory failures. Two deterministic
  harness bugs remain: row grounding flattens all columns and creates cross-column numeric
  collisions (`Season` value 7 -> `Match_Winner=7`), and scalar equality can connect an unrelated
  ID to a numerically equal final count. Nearest-last-table also misses multi-handle answers. Do not
  approve process reward until row edges are column-aware, answer grounding is source-structured,
  multi-source final dependencies are handled, and audit is rerun. Report:
  `src/rl/GROUNDING_EXTERNAL_AUDIT.md`; artifacts: `data/rl/grounding_external_audit_v{1,2,3}/`.
  Current tests: 18 RL + 136 harness pass.
- **Process-reward v3 distributed responsibility (2026-07-16; allocation fixed, semantic gate still
  pending)**: `process_reward.py` now separates a bounded terminal-outcome budget from event-local
  penalties. Success fallback credit is normalized over legal/state-changing/terminal environment
  features instead of defaulting to the last action. Failure outcome budget (`lambda_fail=0.30`)
  is normalized over the harness-inferred attempted dependency chain with state-change and terminal
  bonuses; tool errors (`0.08`), repeats (`0.06`), no-state legal calls (`0.03`), and ignored
  feedback (`0.05`) remain on their exact event steps. The historical v2 config explicitly retains
  terminal-only behavior; current config is `configs/process_reward_v3_distributed.json`. On 139
  Scale500b successes, reward min/mean is **0.79/0.9649** and max positive share p50/p90/max remains
  **0.277/0.422/0.600**. On 356 semantic failures, total reward min/p50/p90/max is
  **-0.80/-0.64/-0.30/-0.30** instead of uniform -0.8; all **356/356** have multi-step negative
  allocation, negative-step count p50/p90/max is **3/8/19**, and maximum single-step negative share
  is at most **0.556**. Only one six-error episode reaches the cap. Runtime reward uses only
  deterministic harness facts; external models are offline QA only. This fixes concentration and
  severity calibration but does not clear the known grounding false-edge or hard-data balance gates.
  Report: `src/rl/PROCESS_REWARD_V3_DISTRIBUTED_REPORT.md`; artifacts:
  `data/rl/bird_scale500b_process_reward_v3_distributed/`. Tests: 20 RL + 136 harness pass.
- **Current-config DS V4 Pro hard rescue probe (2026-07-17)**: a deterministic seed-20260717 sample
  selected 30 of the Scale500b hard tasks that had all terminated as Flash `protocol_error`; the
  selector and manifest are `src/sft/select_hard_rescue_probe.py` and
  `data/eval_inputs/bird_scale500b_hard_protocol_pro30*`. Only the teacher changed to
  `deepseek-v4-pro`; rolling-history-4, full prompt, 2048 tokens, one attempt, max 30 steps, strict
  no-repair parser, and protocol hash `d18c629372c8a7ab` were matched. Pro achieved **20/30 =
  66.67%** execution-correct and **30/30 legal terminal**: 20 success, 10 wrong answer, zero terminal
  protocol errors, versus paired Flash 0 success / 30 protocol terminals. Error events fell
  **90 protocol → 5 protocol + 2 execution**. All accepted episodes replay **20/20**; 16 are clean
  and 4 recovered. Existing SFT gates retain **18/20 episodes / 149 step targets** (two think-length
  rejects). The run used 278 requests, one transport retry, 1.120M total tokens, and 433.8 s with
  four workers. This is a conditioned rescue result, not an all-hard accuracy estimate, but it
  confirms Flash carrier reliability was the dominant failure in this bucket. Report:
  `src/sft/BIRD_HARD_PRO_CURRENT_CONFIG_REPORT.md`; trajectory prefix:
  `data/trajectories/bird_ds_v4pro_scale500b_hard_protocol30_rolling4_full_*`.
- **Scale500 train40 adapter holdout gate failed (2026-07-17)**: the completed 40-episode / 262-target
  adapter was evaluated in the matched rolling/full tool loop on the disjoint seven-task BIRD-train
  holdout. An 8k two-worker run reached 4/7 once but was not stable across reruns. The controlled
  12k single-worker run is the decision artifact:
  `data/results/qwen2.5_7b_bird_scale500_train40_full6400_holdout7_12k_serial/`; it scored **2/7**
  execution-correct and **5/7** legal, with 3 wrong answers, 1 protocol error, and 1 context overflow
  (input at least 12,161 tokens plus the 128-token minimum output budget). Do not run this adapter on
  BIRD Dev or treat the tiny run as a quality result. Its vLLM and SSH tunnel were stopped and both
  GPUs released.
- **Rolling resident-observation R2 + strict read bound (2026-07-17)**: bounded rolling history still
  carries the prior four legal assistant actions, but historical tool messages now contain only
  structured result summaries; full rows/schema/values appear once in CURRENT ENVIRONMENT STATE.
  `read_subtable.limit` is strictly **1..20**; larger/non-integer requests are argument errors, never
  clamped. Protocol is `v2i-state-only-join-feedback-r2`, rolling context
  `v2-bounded-legal-history-resident-observations`, hash `a0c53715fcadc3eb`; the full system prompt
  remains the full variant (SHA-256 `3a0a7f...b8d60`). On the same train40 records, max rendered
  characters fell 67,572→41,442; after rejecting old over-limit reads, the 316-episode candidate
  pool's max is 24,992. Its exact Qwen 8k audit retains **1970/1970** targets (max 6,985 tokens).
- **Process-reward V4 grounding and deterministic SFT gate (2026-07-17)**: row grounding is now
  column-aware and follows harness schema/FK lineage; missing BIRD FK target columns resolve only to
  a unique referenced-table primary key. Final scalar equality no longer creates direct observation
  edges, and multi-value answers may collect several read handles. `read_subtable` column metadata
  is harness-private so historical state replay remains unchanged. On all 316 current quality
  candidates, replay is 316/316, structural slices 315/316, final-value completeness 244/316,
  action-literal completeness 305/316, and both deterministic gates **235/316 = 74.37%**. Reward
  min/mean/max is 0.79/0.9662/1.0; max positive share p50/p90/max is 0.272/0.422/0.625; conservation
  passes. Process RL remains **no-go** because 81 episodes are incomplete and revised-edge precision
  has not been re-audited. Report: `src/rl/PROCESS_REWARD_V4_GROUNDING_REPORT.md`; artifacts:
  `data/rl/bird_candidate316_process_reward_v4d_grounding/`; tests: 60 related unit tests + 136
  harness tests pass.
- **Grounded BIRD SFT scale run completed (2026-07-18)**: the conservative pre-final-FK split keeps
  228 deterministic-complete episodes (a valid subset of final V4's 235): **206 train / 22 eval**
  episodes, **1303 / 130** single-step targets, zero episode overlap. Exact Qwen/LLaMA-Factory 6400
  audits retain 1303/1303 and 130/130 complete targets. Remote run
  `bird_grounded_r2_6400_20260717_main` completed normally on table_rl GPU0 with Qwen2.5-7B QLoRA, full prompt,
  rolling history 4, effective batch 16, and 2 epochs / **164 optimizer steps**; epoch-level eval uses
  the separate 130-target set. It passed step 1 in 67.8 s at about 17.1/24.6 GiB without OOM. Log:
  `/home/dengyan/tabular_rl_outputs/logs/bird_grounded_r2_6400_20260717_main.log`; output:
  `/home/dengyan/tabular_rl_outputs/checkpoints/qwen2.5-7b-bird-grounded-r2-train1303-eval130-6400-qlora`.
  All **164/164** optimizer steps and both epoch-level evaluations completed without OOM. Final
  train loss is **0.52388**, final held-out eval loss is **0.46080**, train runtime is 10,811.6 s
  (about 3 h), and final evaluation finished at 2026-07-18 00:09 Asia/Shanghai. The final adapter,
  `checkpoint-82`, `checkpoint-164`, `train_results.json`, `eval_results.json`, and plots all exist;
  both GPUs were idle after completion. The next gate is the fixed BIRD-dev closed-loop comparison.
- **Grounded BIRD R2 checkpoint gate (2026-07-18)**: on the fixed 30-task BIRD-dev gate,
  `checkpoint-82` (epoch 1) scores **8/30 = 26.7%** with 23 legal terminals, while the final epoch-2
  adapter scores **7/30 = 23.3%** with 24 legal terminals; the old 40-episode adapter scored 6/30.
  Difficulty results for epoch 1 are 3/12 simple, 2/9 moderate, and 3/9 challenging. Final-adapter
  teacher-forced metrics on the disjoint 130-target eval are 126/130 strict-valid, 100/130 teacher
  tool match, and 48/130 exact tool+arguments. Validation loss improves 0.4753→0.4608 from epoch 1
  to 2, but closed-loop accuracy does not, so behavior selects **checkpoint-82** for now. This is a
  small 30-task gate; compare both checkpoints on a fixed dev-100 before a scaling claim or RL.
  Report: `src/sft/BIRD_GROUNDED_R2_EVAL_REPORT.md`.
- **Qwen2.5-7B BIRD direct-SQL matched baseline (2026-07-15)**: `src/eval/text2sql.py` now accepts
  adapter-exported DatasetTask JSON/JSONL through `--tasks-json`, using its `db_path`, `gold_sql`,
  and optional `external_knowledge`; its legacy Spider-dev default is unchanged. A temporary base
  (no LoRA) `Qwen2.5-7B-Instruct` vLLM run on the same seven disjoint rolling-SFT holdouts achieved
  **2/7 = 28.57%** execution accuracy, with 2 semantic wrong results and 3 SQLite execution errors
  (wrong table/column/alias use). It is a one-shot full-schema SQL task with no tool or execution
  feedback. The matching compact rolling tool adapter reached 3/7, but both scores are far too
  small for a generalization claim. Artifact:
  `data/results/qwen2.5_7b_bird_direct_sql_base_holdout7/` (gitignored). The temporary vLLM/tunnel
  was stopped and both table_rl GPUs were verified released.
- **Qwen2.5-7B full BIRD Dev direct-SQL baseline (2026-07-15)**: the official public
  `dev_20240627` archive is local under `data/bird/dev_20240627` (**1,534** examples, 11 SQLite
  databases; simple/moderate/challenging **925/464/145**) and is evaluation-only. The separate
  `src/harness/bird_dev_adapter.py` exports `data/eval_inputs/bird_dev_20240627.jsonl`; all 11
  databases exist and an initial 50/50 gold-SQL smoke passed. The strict held-out BIRD `test` gold
  is not public and requires official submission, so this released Dev split is the locally
  reproducible benchmark. Base (no LoRA) `Qwen2.5-7B-Instruct`, full schema + official evidence,
  one-shot SQL, no tools or execution feedback achieved **598/1534 = 38.98% EX**: simple
  **433/925 = 46.81%**, moderate **137/464 = 29.53%**, challenging **28/145 = 19.31%**. Final
  artifact is `data/results/qwen2.5_7b_bird_dev_direct_sql_base_v4/` (gitignored). Its manifest
  records a 5-second SQLite VM limit for generated SQL and 30-second whole-task watchdog; failures
  are 583 wrong results, 349 execution errors, and 4 explicit `task_timeout`s. Earlier remote
  v1--v3 runs are interrupted diagnostic artifacts only: unbounded generated SQL could stall on
  large `codebase_community` queries. `text2sql.py` now has adapter task windows, a generated-SQL
  VM deadline, and an opt-in isolated task watchdog so such failures are explicit rather than
  blocking the full benchmark. The temporary vLLM was stopped and table_rl GPU1 was verified free.
- **Experiment dashboard checkpoint (2026-07-15)**: the local dashboard now supports an
  evaluation's own `expected_total` and either a result directory or an `all.jsonl` artifact, rather
  than assuming every run is Spider dev-1034. It registers the completed BIRD Dev direct-SQL
  baseline (**598/1534**) and the paired BIRD Context-100 rolling-4 construction result
  (**47/100**, versus state-only **35/100**). Backend tests cover both records and frontend
  production build passes. The rolling source success JSONL is an audit/source artifact and remains
  `sft_export_eligible:false`; only the separately rendered matched-format split is eligible for
  student training.
- **Flash-2048 successful-trajectory quality audit (2026-07-14)**: the 13 verified episodes yield
  109 legal steps. They are semantically grounded rather than pure boilerplate: a simple
  action-argument mention proxy hits 103/109 thinks, and the hard Pengo/Gujarat traces correctly
  carry concrete handles, keys, and values across joins and recoveries. However, they are not yet
  polished SFT demonstrations: think length is p50 87 / p90 217 / max 557 words, 64/109 recap
  state, and there are at least 9 redundant exact repeat calls (five reads of `join_002` in one
  easy episode, three reads of `join_005` in a hard episode, plus repeats elsewhere). Exact text
  is not duplicated because the model paraphrases the recaps. SFT-1 admission must therefore add
  quality gates beyond execution correctness: remove repeated identical reads/calls, suppress
  no-information plan or recap steps, cap trajectory contribution, and retain recovery labels only
  for the first legal post-error action. Keep full raw audits unchanged.
- **DS Flash v4 explicit-format BIRD-30 test (2026-07-14)**: a user-authorized clean rerun with an
  explicit, provider-native `reasoning_content`/visible-`content` example is
  `bird_ds_flash_v4_explicit_contract_clean30`. It has 30 unique single attempts: **9/30** verified,
  versus 7/30 before. Crucially, action-level adapter compliance rises **101/178 = 56.7% → 233/275 =
  84.7%**, protocol events fall **56 → 42** (31.5%→15.3% of actions), and missing-reasoning events
  fall **34 → 10**. Residual protocol events are 17 unterminated tool calls, 15 visible-prose-before-
  tool-call cases, and 10 missing-reasoning shapes. Terminal outcomes are 10 protocol_error and 11
  wrong_answer; the larger semantic-error share reflects reaching more actual tool episodes, not a
  parser relaxation. This supports keeping the explicit DS transport contract for Flash, but it is
  not yet sufficient for large-scale high-yield SFT without further carrier/reliability work.
- **DS V4 Pro format probe (2026-07-14)**: before any full batch, the first two fixed BIRD tasks
  were checked under the same adapter. Both completed correctly in 5 actions with **0 errors**;
  all **10/10** turns had non-empty `reasoning_content` and visible content containing exactly one
  complete tool-call block (including harmless surrounding newlines), so the adapter applied 10/10.
  This is materially more format-stable than Flash. The accidentally started full-30 Pro process
  was terminated after the user requested the small-batch inspection; only the probe artifacts
  `bird_ds_v4pro_format_probe2_*` are retained so far.
- **DS V4 Pro clean BIRD-30 smoke (2026-07-14)**: isolated run
  `bird_ds_v4pro_adapter_clean30` has exactly **30 unique single-attempt** records (no resume
  duplicates): **12/30 = 40.00%** verified and replayed 12/12. Successes are easy 9, medium 2,
  hard 1; 6 clean and 6 recovered. Terminal failures: 14 protocol_error and 4 wrong_answer;
  audit has 58 protocol and 2 argument-validation events. The DS adapter carried 207 actions and
  rejected 58 unusable carrier turns, preserving strict parsing. This is the canonical Pro smoke;
  ignore the stale partial `bird_ds_v4pro_adapter_smoke30_*` artifacts from the interrupted run.
- **BIRD two-stage SFT/RL plan frozen (2026-07-14)**: the detailed SSOT is
  `draft/bird_two_stage_sft_rl_plan.md`. There are three training phases, not six sequential stages:
  (1) `SFT-1 = Teacher Demonstration`, (2) `SFT-2 = On-policy Refinement`, and (3) RL. Dataset terms
  are orthogonal and fixed: `Teacher Demonstration`, `On-policy Student Success`, `Decision
  Correction` (replace the action from state-before), and `Feedback Recovery` (continue from the
  real post-error/observation state). Full SFT-2 mixes clean replay + student success + correction +
  recovery; `Outcome-only RL` and `Process-shaped RL` must branch from the same full SFT-2
  checkpoint. Canonical source remains a full verified episode, while every SFT target is a standard
  one-human/one-gpt **single-step** record `(question, catalog, state_before, optional last error) ->
  one action`; failed actions remain audit events and are never labels. Because the 16/30 canonical
  run is a v2h historical artifact and join feedback is now v2i, the immediate next gate is a paired
  same-30 v2i comparison: first rerun the strict baseline, then change only the teacher delivery to
  a single-action adapter/structured-output mode. Retain the strict student/runtime parser and
  separately report raw format compliance, adapted action validity, clean/recovered terminal
  success, and per-difficulty yield before any scale-up or final SFT export.
- **7B / V2a and V2-ctx evaluations done (2026-06-15)**: V2a scores **66.83%** and V2-ctx scores
  **62.77%** on the full 1,034-example Spider dev set, versus v1 **68.38%** and direct SQL
  **69.25%**. V2-ctx preserves the large-database context invariant but exposes a planning weakness:
  current trajectories teach perception as a fixed ritual rather than evidence that can change the
  next action. In all 7,767 V2-ctx train+dev trajectories, every one of the 6,361
  `read_subtable` calls is penultimate and immediately followed by `answer_from_context`.
- **Qwen3.5-9B pilot SFT (2026-06-25)**: base model is now local on NewGNN at
  `/home/dengyan/models/Qwen3.5-9B`; Claude's baseline scripts live in `/home/dengyan/run_qwen35_*.sh`.
  Baselines already run on Spider dev: direct SQL no-thinking **751/1034 = 72.6%**, direct SQL
  thinking **756/1034 = 73.1%**, tool 2-shot no-thinking **422/1034 = 40.8%**, tool 2-shot thinking
  **401/1034 = 38.8%**. Qwen3.5 has a 248k vocab, so cross-entropy logits OOM on long records even
  when activation memory fits: the 178-record v8 pilot at 8192 cutoff OOMed after 3/46 steps. The
  working pilot uses Qwen3.5-tokenized <=4096 records (`175/178` kept; dropped indices 140,176,177)
  with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, output under
  `/data/dengyan/tabular_rl_outputs/checkpoints/qwen3.5-9b-spider-v8-pilot-ready-4k-qlora`. It
  completed 44 steps / 2 epochs in **39:21**, final train loss **0.5535**, peak GPU memory about
  **23.5/24.6 GiB** on one RTX 3090. A matched 4-epoch rerun on the same 175 records completed
  88 steps in **1:17:28**, final train loss **0.4179** and last logged loss **0.2633**, output under
  `/data/dengyan/tabular_rl_outputs/checkpoints/qwen3.5-9b-spider-v8-pilot-ready-4k-epoch4-qlora`.
  Full Spider dev tool zero-shot evals (no thinking, vLLM `--enforce-eager`, max len 4096) are done:
  2 epochs **377/1034 = 36.5%** and 4 epochs **440/1034 = 42.6%**. The 4-epoch run fixes some
  simple/mid-complexity tool chains (e.g. the youngest-singer songs case), but complex schemas with
  joins/set difference still produce many `steps=0` API/protocol failures; use the saved failure
  artifacts for the next correction-data loop. Local helper/configs:
  `src/sft/make_qwen35_4k_subset.py` and `src/sft/configs/qwen3.5_9b_qlora_v8_pilot_ready_4k.yaml`.
- **RL pilot environment prepared (2026-06-27)**: first-stage RL work should start from `src/rl/`.
  `env.py` wraps the same closed-loop model↔harness protocol used by eval; the trainer only supplies
  assistant text, while the environment owns parsing, tool execution, observation messages, and
  terminal scoring. `reward.py` is an auditable pilot reward (correctness dominates; legal answers,
  valid tool calls, evidence reads get small bonuses; repeated calls, tool/protocol/API failures and
  max-steps get penalties). Before any PPO/GRPO run, use `build_reward_report.py` on rollout/pass@k
  artifacts and manually audit a sample; use `select_pilot_tasks.py` to choose mixed-success or
  legal-but-wrong tasks. Do not train RL directly on API/protocol failure-heavy buckets.
- **Result-only RL baseline (2026-07-11)**: reusable task/environment/reward logic lives directly
  under `src/rl/`; backend glue is isolated under `src/rl/frameworks/<backend>/`. The current
  hardware-compatible implementation is `frameworks/accelerate/group_reinforce.py`: single-GPU
  4-bit LoRA group-REINFORCE with exactly `reward = 1` for an execution-correct final denotation and
  `0` otherwise; it has no process shaping, error penalty, KL term, or length reward. The 881
  external-rollout trajectories are only the SFT adapter initialization, never the RL task filter.
  The full baseline samples the complete **7,000-question Spider train split** (four episodes per
  question) and automatically evaluates the final adapter on all 1,034 Spider dev questions.
- **SUPERSEDED — Next data iteration (SFT v10 two-lane plan, 2026-06-28)**: this historical plan's
  gold-SQL canonical/enrichment lane is retired by the 2026-07-22 decision above. Do not add a reflection tool,
  `invalidate` state, memory, or any model-visible sidecar. Use two data lanes: (A) clean canonical
  data from gold-SQL verified trajectories with observation/think enrichment; (B) recovery data only
  from current SFT rollout failures on **training-set examples** (never Spider dev / held-out eval),
  where the external LLM rewrites a failed attempt into a validated recovery trajectory under
  harness feedback. Recovery repair defaults to **10 attempts**.
  Keep the mixture mostly clean canonical (about 1600-1700) plus a smaller recovery slice
  (about 300-400). Fixed plan and commands: `draft/sft_v10_two_lane_data_plan.md`. Candidate
  selector: `src/sft/select_recovery_candidates.py`; train rollout input builder:
  `src/sft/build_rollout_examples.py`. Current protocol index: `tool_design/current_trajectory_protocol.md`.
- **Claude implementation handoff**: read `draft/trajectory_data_generation_v2.md` before changing
  trajectory generation or starting another SFT/RL run. It records the audited blockers, canonical
  `add_to_memory(key, source_step_id)` ownership model, deterministic semantic derivations, online
  harness-authored provenance, diversity sources, implementation order, and acceptance gates.
  BIRD Mini-Dev is downloaded under `data/bird_mini_dev/` and should initially remain evaluation-only.
- **PARTIALLY SUPERSEDED — Memory decision (2026-06-14)**: the memory authority design remains
  historical context, but any construction from a remaining gold Plan is retired by the 2026-07-22
  data decision. Memory remains broader than scalar values, but is typed by
  authority. `derived_value` and non-scalar `evidence_pointer` are harness-grounded from tool
  history; large lists/tables stay in `data_view`. `plan` and `hypothesis` are model-authored
  working state and cannot serve as factual evidence or `value_ref`. Plan SFT can be constructed
  programmatically from the abstract remaining gold Plan, then diversified with verified rollouts;
  an external LLM is optional and never authoritative. See `final_tool_design.md` §1.1 and the v2
  handoff document §3.
- **Long-context OOM result**: the 9,113-token longest record OOMs at cutoff 10,240. At cutoff
  8,192, standard AdamW OOMs only after its first optimizer-state allocation, so a one-step smoke
  is misleading. Rank 16 + paged 8-bit AdamW passed two worst-case updates at 23,646/24,576 MiB.
  The full run uses `spider_tools_v1_8k`: 6,763 records after dropping 8 records over 7,900 raw
  content tokens; longest kept is 7,701. Remote SFT env now has `bitsandbytes==0.46.1`.
- **v1 SFT data built**: `build_sft_data.py` now accepts `--input-pattern`, `--output-dir`,
  `--output-prefix`, and `--dataset-name`. The reproducible v1 command uses the think-filled
  trajectories and `--max-est-tokens 8900`, producing 6771 train + 998 dev records under
  `data/sft/spider_v1_*`. It deliberately drops `spider_train_3698` and `spider_train_3697`;
  Qwen2.5-7B tokenizer audit found them above/too close to the 10240 training cutoff. Final raw
  content-token maximum is 9113 train / 8166 dev (6 train records exceed 8192), so keep
  `cutoff_len: 10240`. The files are synced to `~/tabular_rl_project/data/sft/` on NewGNN.

## Memory v2 design (shared decision, 2026-06-14)

**Status: `derived_value` (scalar) is IMPLEMENTED & verified as V2a — see the Current-state V2a entry
and `draft/v2a_memory_report.md`.** `evidence_pointer`/`plan`/`hypothesis` remain design-only (V2b/V2c).
The implemented field names for the scalar case are `key`/`content`/`derivation` (not the design's
`definition`/`description`); the ownership rules below hold unchanged.

Canonical details live in `tool_design/final_tool_design.md` §1.1 and
`draft/trajectory_data_generation_v2.md` §3. Claude and Codex must follow these rules when changing
the compiler, emitter, harness, protocol, trajectory schema, SFT construction, or RL rewards.

- Memory is a typed task-level workspace, not an untyped scalar dictionary.
- `derived_value`: a scalar or small structured result extracted by the harness from a cited tool
  output. It has `harness_grounded` authority and may support the final answer. Only this type may
  be consumed through `condition_filter.value_ref`.
- `evidence_pointer`: a semantic pointer to a non-scalar intermediate result such as a filtered
  row set, grouped table, join result, or ranked subset. The actual rows remain in `data_view`;
  memory stores the table handle, structured operation definition, source step ids, and a compact
  description. Do not copy large row lists or tables into memory.
- `plan`: model-authored future goals/subgoals and their statuses. It is control state rather than
  evidence, cannot support the final answer, and cannot be used as `value_ref`.
- `hypothesis`: a model-authored tentative claim with `unverified` status. It should be paired with
  `refine_memory`: later tool evidence may cause the harness to mark it `confirmed`, `rejected`, or
  `revised`. The model may propose an update, but only the harness may grant `confirmed` status.

For grounded non-scalar memory, separate the fields by ownership:

- `definition`: structured operation semantics copied from the executed tool call, owned by the
  harness. Example: input table + `condition_filter` + exact conditions + output table.
- `description`: deterministic readable rendering of `definition`, owned by the harness. Templates
  may accurately describe what operation produced the result, but must not invent task-level
  interpretations.
- `purpose`: optional model-authored explanation of why the result may be useful. It is not factual
  authority.
- `source_step_ids` and the `data_view` handle: harness-authored provenance and data authority.

Thus code can reliably render descriptions such as "rows from customers where risk_score > 0.8"
or "employee counts grouped by department". Claims such as "these are the customers most worth
contacting" belong in `purpose` or `hypothesis`, not in the grounded description.

**Superseded data-construction note (2026-07-22):** do not build initial plan memory from an
abstract remaining gold Plan, and do not enrich any complete gold-compiled trajectory. New model
turns must be produced causally from the visible episode prefix in a real model↔harness loop.
Harness-derived memory may still be grounded deterministically in tool outputs that have already
occurred in that same episode; execution and provenance checks remain the acceptance gate.

## Architecture (`src/harness/`)

- `executor.py` — relational core. Each table-producing tool registers a named SQL view; reading/
  scalar tools run a SELECT. Tools: condition_filter, project, join_tables (prefixes columns
  internally via left_prefix/right_prefix), group_aggregate, aggregate, extreme_value_select
  (merged old order_limit; table-producing), set_op, derive_column, window, preview
  (inlines table content for model perception), rows, gold.
- `plan.py` — Plan IR: `Step(id, tool, args)`. `run_plan` threads step ids → view names; a `values`
  map threads scalars (V2b: aggregate parks its scalar under its step id; a predicate's `value_ref`
  points at the producing step directly); `resolve_cond` resolves `value_ref` and `in_table` in condition trees.
- `compiler.py` — `Compiler(schema).compile(sql)`: sqlglot AST → Plan, walking FROM/JOIN → WHERE →
  GROUP → HAVING → ORDER/LIMIT → SELECT → DISTINCT. Scalar subquery → aggregate → predicate
  `value_ref` = that aggregate step's id (no add_to_memory). IN/NOT-IN subquery → membership via
  in_table. Unsupported → CompileError.
- `scalar_grounding.py` (V2b, SHARED by emitter + rollout) — `extract_scalar(history, source_step_id)`
  / `ground_scalar_reference`: strict scalar extraction (scalar tool / 1×1 table, non-NULL else
  `ScalarGroundingError`). The ONLY place a `value_ref`'s scalar is produced; no memory_id/key/content.
- `provenance.py` (V2b, SHARED by emitter + rollout) — `build_references(tool, args, resolve_step)`
  builds typed `references` edges (`type=data|value`, structured `target`) over the model-facing arg
  shape; `backward_slice(traj, reference_type=("data","value"))` is parameterized (grounding excluded
  by default). `col_lineage` + `grounding` edges land in V2c (阶段 C).
- `verify.py` — `round_trip`: compile → run → compare to gold SQL on the real DB.
- `emitter.py` — verified Plan → training trajectory. Each step gets harness-authored typed
  `references` (via `provenance.build_references`) + `produces`; a predicate's `value_ref` cites the
  producing step directly (no add_to_memory step). `provenance.backward_slice(traj)` reverse-derives
  the answer's dependency set; `validate()` = legality + reference-integrity gate.
  Trajectories carry `schema_version`.
- `run_all.py` (tests + compile coverage), `run_spider.py [N]` (execution-verified on real DBs),
  `gen_trajectories.py [train|dev]` (batch emit → `data/trajectories/spider_*.jsonl`, gitignored).

## SFT pipeline (`src/sft/`, `src/eval/`)

- `protocol.py` — SINGLE source of truth for the model↔harness protocol (system prompt + tool specs
  + `<think>`/`<tool_call>` rendering + parse). V2a: observation envelope `{step_id, status, output}`
  (`tool_output_message(step_id, output)`), `validate_arguments` strict per-tool schema, `PROTOCOL_VERSION`
  + `protocol_hash()`. Imported by both build_sft_data and rollout so SFT and eval can never drift.
- `build_sft_data.py` — trajectories → LLaMA-Factory sharegpt jsonl (loss on assistant turns only);
  manifest records `protocol_version`/`protocol_hash`.
- `fill_think.py` — replaces templated `think` with grounded reasoning from the external LLM (api.md).
  `splice_think.py` — reuse prior-version `think` for a new schema at ZERO API cost (memory steps get
  the template; renamed memory keys are substituted).
- `rollout.py` — closed-loop eval (live model ↔ harness) + `--replay`. V2b: online step_ids +
  `tool_history` + harness-derived typed `references` (`provenance.build_references`) + online
  `value_ref` grounding via `scalar_grounding.extract_scalar` (illegal ref → execution_error) —
  model-claimed provenance is never trusted. Doubles as the RL env.
- Configs: `src/sft/configs/qwen2.5_{3b_lora,7b_qlora}_sft.yaml`.

## Experiment dashboard

- `experiment_dashboard/` is the local React experiment console. It records the dataset, training
  configuration, loss history, elapsed time, evaluation summaries, and per-case JSON for each run.
- Its overview is split into independent **Spider** and **BIRD** dataset dashboards. Every chart,
  baseline comparison, and aggregate is scoped to the selected dataset; each dashboard declares
  its evaluation and construction-data scope.
- `experiment_dashboard/data/experiments.json` is the editable experiment registry;
  `data/trainer_states/*.json` are local snapshots of remote LLaMA-Factory trainer states.
- The registry includes the complete direct-SQL and two-shot tool baselines as first-class
  experiments. The backend dynamically enumerates every existing artifact for each experiment
  (including `all.jsonl`, manifests, summaries, trainer state, and smoke runs); do not restore a
  hard-coded frontend source list.
- JSON records use a structure-aware viewer (conversation, tool trajectory, direct-SQL comparison,
  or generic collapsible tree) with raw JSON as an alternate view.
- The dependency-free Python API scans repository JSONL/manifests, refreshes trainer state through
  `ssh table_rl`, serves the built React app, and proxies OpenAI-compatible requests to
  `VLLM_BASE_URL` (default `http://127.0.0.1:18001/v1`; tunnel port 18001→dell vLLM 8000).
- After every remote evaluation finishes, sync the result directory back into local
  `data/results/...` and make sure the dashboard registry points to it. The dashboard refresh button
  now syncs registered evaluation directories from table_rl as well as trainer state; use it or an
  explicit `scp -r` before expecting the frontend to show the latest run.
- The Playground system prompt is loaded directly from `src/sft/protocol.py`; do not duplicate or
  hand-maintain a second prompt in React. Its service controller may start only registry-backed
  LoRA adapters on an actually idle table_rl GPU. It tracks its own PID/metadata under remote
  `logs/dashboard_vllm.*` and must never stop an unowned process.
- Run the API with `.venv/bin/python experiment_dashboard/backend/server.py`; run the frontend with
  `cd experiment_dashboard/frontend && npm run dev`, or `npm run build` and use the API server alone.

## How to run (local, Mac)

```
.venv/bin/python src/harness/run_all.py          # unit tests + Spider compile coverage
.venv/bin/python src/harness/run_spider.py 2000  # execution-verified on real DBs
.venv/bin/python src/harness/gen_trajectories.py train   # (and dev)
.venv/bin/python src/sft/fill_think.py --split train --n 99999 --workers 16 --out <path>
.venv/bin/python src/sft/build_sft_data.py both \
  --input-pattern 'data/trajectories/spider_{split}_think.jsonl' \
  --output-prefix spider_v1 --dataset-name spider_tools_v1 --max-est-tokens 8900
```
Uses the project venv `.venv` (sqlglot 30.9). Spider DBs in `data/spider_data/` (gitignored).

## GPU server (`ssh table_rl`)

- **dell PowerEdge T640**, host 10.214.243.15 port 222, user dengyan, key auth.
  **2× RTX 3090 24G, driver 560.35.03 (=CUDA 12.6 max)**, 48 cores, 503G RAM.
  Disk `/dev/sda2` 4.9T (3.2T free), everything under `/home/dengyan`.
  Project at `~/tabular_rl_project`; Qwen3.5-9B at `~/models/Qwen3.5-9B`;
  checkpoints at `~/tabular_rl_outputs/checkpoints/`.
- Internet via reverse tunnel from the Mac: `nohup bash src/sft/tunnel_table_rl.sh > /tmp/tunnel_table_rl.log 2>&1
  & disown` (auto-reconnect loop; -R 28472→Mac clash 7897 gives server egress; -L 18001→dell vLLM 8000
  lets the Mac reach the model). Server side: `export http(s)_proxy=http://127.0.0.1:28472`.
- conda envs: `sft` (torch2.6.0+cu124, llamafactory0.9.5) and `vllm-qwen35` (torch2.10.0+cu126, vllm0.19.1).
- **NewGNN** (8× RTX 3090, driver 550, port 16014) kept for reference; tunnel ports 28471/18000; project
  same path. `/home` was 100% full as of 2026-06-25 — use `/data/dengyan/tabular_rl_outputs/` on NewGNN.

## PITFALLS — hard-won, do not re-step

**Ops / server:**
- **Version pins (table_rl / dell, driver 560 = cu126):** sft env uses torch2.6.0+cu124 (from
  NewGNN freeze, cu124 wheels still install fine on cu126 driver); vllm env uses torch2.10.0+cu126 +
  vllm0.19.1 + transformers5.12.0. Install with `pip install -r req.txt --extra-index-url
  https://download.pytorch.org/whl/cu124` (sft) or `cu126` (vllm). On dell, upgrade pip first
  (`conda run -n ENV pip install --upgrade pip`) — old conda pip ≤26 truncates the available-version
  list and fails to find 1.x packages like accelerate==1.11.0 or aiohappyeyeballs==2.6.2.
- **NewGNN version pins (legacy, driver 550 = cu124):** `vllm==0.8.5.post1` + `torch==2.6.0
  torchaudio==2.6.0 torchvision==0.21.0` + `transformers==4.51.3`. Still relevant if using NewGNN.
- **dell missing `libcuda.so` (triton/bitsandbytes JIT link fails):** dell's driver ships only
  `/usr/lib/x86_64-linux-gnu/libcuda.so.1` (no unversioned 64-bit `libcuda.so` symlink), so QLoRA
  training dies at "Quantizing model to 4 bit" — triton JIT-compiles `cuda_utils` and `gcc -lcuda`
  fails with `collect2: ld returned 1`. Fix (no sudo): `ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1
  ~/cuda_link/libcuda.so` then `export LIBRARY_PATH=$HOME/cuda_link:$LIBRARY_PATH` in every training
  launcher. (vLLM inference doesn't hit this; NewGNN already has the symlink so it never showed there.)
- vLLM/training must run with **`HF_HUB_OFFLINE=1`** (processes have no proxy env; model is cached).
- HF downloads: official huggingface.co via the clash tunnel + **`HF_HUB_DISABLE_XET=1`** (the Xet
  client ignores proxy env); do NOT use hf-mirror (hub ≥1.x rejects its HEAD). Command is
  `hf download` (`huggingface-cli` is deprecated/no-op in hub 1.x).
- **`pkill -f` self-kill:** a pattern matching the running command's own argv (or a literal in a
  comment on the same line) kills the ssh session (exit 255). Use a bracket: `pkill -f "vllm serv[e]"`
  and keep explanatory text in the chat, never on the command line.
- **`sed -i` anchoring:** anchor to line start (`^output_dir:`) — `output_dir: .*` also hit
  `overwrite_output_dir:`.
- **GPU contention:** public box — `nvidia-smi` before every job, set `CUDA_VISIBLE_DEVICES` to a
  truly-idle card (cards get grabbed between checks; in-process GPU 0 = the physical card you pinned).
  OOM from long samples / fragmentation → tune `cutoff_len` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`;
  7B on 24G needs QLoRA (4-bit).
- **Idle GPU detection must be strict:** do not select a card by memory alone. Treat a GPU as idle
  only when memory is low (dashboard default ≤512 MiB), utilization is near zero (default ≤5%), and
  `nvidia-smi --query-compute-apps` shows no compute process for that GPU UUID. Re-check after a
  short sleep immediately before launching vLLM/training to reduce races with other users.
- **vLLM cleanup is mandatory:** after every inference/evaluation test, stop the vLLM server and
  verify with `nvidia-smi` that its GPU memory is released. Do not leave an idle vLLM process
  reserving GPUs. Before killing anything, confirm the PID belongs to user `dengyan` and its command
  is the vLLM instance started for this project.
- Tunnel dies when the Mac sleeps; `tunnel.sh` reconnects and `caffeinate` keeps the Mac awake.

**Code / compiler:**
- sqlglot 30 uses the `from_` arg key (not `from`).
- Column qualification for joins is **internalized in `join_tables`** (no separate rename steps);
  the model sees one `join_tables` call, prefixing is harness-internal.
- Scalar/list threading lives in `plan.resolve_cond`: `value_ref` (V2b: the producing `step_id`,
  grounded to its scalar at execution) and `in_table` (IN-subquery set). Trajectory **display keeps
  value_ref(step_id)/in_table**; execution resolves them — keep this split when editing emitter/run_plan.
- **Scalar trust boundary (V2b):** a predicate's `value_ref` cites only a `step_id`; the harness owns
  the value via `scalar_grounding.extract_scalar` (strict 1×1 non-NULL; online illegal ref →
  execution_error). NEVER let the model write a literal threshold value (the v1 bug that broke replay
  and is forgeable in RL). `references`/`produces` are harness-authored sidecars, never `tool_call.arguments`.
- **Verification gate is sound:** unsupported SQL → CompileError (counted, never mis-compiled);
  wrong decompositions → round-trip mismatch → dropped. New datasets lower coverage, never corrupt.
- **Ceiling:** execution-verified ~98.4% is the hard ceiling — 100% is impossible because
  `ORDER BY agg LIMIT k` with ties makes the gold SQL itself nondeterministic. ZERO real
  decomposition bugs remain; remaining failures are ties + 2 exotic 4-way self-joins.

## Conventions

- Two memory layers: this `AGENTS.md` (shared, repo, durable) + Claude's private auto-memory under
  `~/.claude/projects/<proj>/memory/` (Claude-only working notes). Put anything Codex should know
  HERE.
- Design discussion notes live in `draft/` (process_reward_density, subtable_vs_memory,
  table_text_handling, experiment_design, sft_plan, tool_design_summary, trajectory_data_generation_v2,
  v2a_memory_report).
- User preference: narrate every executed command + one-line why (the user checks and learns ops).
- Commit on the user's request; branch off master only if asked (repo convention is direct-to-master).
