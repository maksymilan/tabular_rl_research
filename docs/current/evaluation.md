# Evaluation contract

The current Qwen3 training/evaluation identity is the frozen Atomic version26 SFT1 contract. Its
trusted anchor is checkpoint-560 at 838/1534 = 54.63% BIRD-dev greedy `bird-set`, measured by the
isolated evaluator under `reproductions/trust_sql/qwen3_8b_atomic_sft1/`. It parses one inline
`<think>...</think>` plus one JSON action, uses recent-four legal history and the frozen full
resident-state prompt, and enables the Qwen3 thinking chat template. DeepSeek's provider split
carrier is not this local evaluator. The later checkpoint-relalg/Atomic-v24-frozen projections and
checkpoints are diagnostic-only and must not be reported as the matched mainline. See
`training_mainline.md`.

Tool-agent, RL-checkpoint, and direct-SQL comparisons must use the same task ids, sampling condition,
database snapshot, external knowledge, generated-query deadline, and denotation metric. Every result
must name its metric; strict multiset and BIRD reference set equality are not interchangeable.

For released BIRD EX reporting, `bird-set` matches the reference scorer by comparing sets of rows,
ignoring row order and duplicate multiplicity. All current/new evaluation, SFT replay, grounding
gates, and RL reward audits use this same `bird-set` contract. `strict-multiset` remains available
only through low-level compatibility code for immutable historical artifacts; active launchers
accept only `bird-set`.
Tuple position remains significant in BIRD EX, so column order is not interchangeable. Active tool
evaluation therefore uses terminal-answer contract `exact-cited-table-v1`: the exact rows and column
order of the table cited by `answer_from_context` are graded. Historical tool artifacts without
that manifest field were produced by a compatibility scorer that could accept a same-width column
permutation; they must be replayed under the exact contract before being compared with released
BIRD EX numbers. Retired trajectories that explicitly authored an `answer` field retain a
replay-only compatibility path, but current model-visible terminals do not expose authored values.
Use `src/eval/rescore_tool_artifact_bird_ex.py` to audit an atomic historical artifact without
regenerating model outputs; write its results to a separate directory rather than modifying the
source artifact.
Arctic-Text2SQL-R1-7B's headline BIRD-dev result uses greedy decoding (`n=1`,
`temperature=0`) with this `bird-set` contract. The existing direct-SQL greedy launcher uses those
decoding and denotation settings; set the generated-query timeout to 10 seconds when matching the
pinned upstream evaluator rather than the launcher's more permissive 20-second default. Arctic's
optional eight-sample majority voting is a different candidate aggregation method and must not be
reported as pass@k. Direct-SQL multi-candidate evaluation exposes the upstream soft-denotation
medoid as `--candidate-aggregation arctic-majority`; use `n=8`, `temperature=0.8`, `top_p=1`,
a 10-second generated-query timeout, and `bird-set` for that optional Arctic setting.

`src/eval/rollout.py` is the retained-atomic single-sample evaluator and
`src/eval/rollout_passk.py` is its multi-sample evaluator. Their dynamic episode scheduling and
vLLM batching remain the basis of the isolated version26 reproduction boundary. Do not mix them
with checkpoint-relalg result roots or provider parsers. `src/eval/text2sql.py` and
`src/eval/text2sql_passk.py` are direct-SQL controls. Result directories and manifests must not mix
different denotation contracts.

The active BIRD-train cohort for all new baseline experiments is the frozen, representative,
disjoint 300-task file `data/eval_inputs/bird_train_baseline300_v1.jsonl` (SHA-256
`87b37b307ea2710acdff7d03bdc474a6ba2cdb8ed51b005cff0fe8fa54c67c03`). It covers all 69 train
databases and was selected against the public database, question-length, and external-knowledge
distribution of all 6,601 normalized train tasks. The former
`bird_train_tool_interface_validation200_version4.jsonl` cohort is deprecated for new experiments
and retained only to reproduce immutable fixed-200 reports. Results across the old 200 and new 300
are not paired comparisons. The full selection, audit, lifecycle, and mandatory future-change
procedure is in `baseline_datasets.md`; every evaluation manifest must record the exact cohort path
and hash.

The completed version51 / `native-tool-bundle` fixed-200 diagnostic uses the same frozen cohort as
version50 and historical version24. It scored 147/200 correct and 200/200 legal versus version50 at
133/200 and 183/200, with 20 paired gains, six regressions (`p=0.00936`), and 17 legal gains with no
regression. Against version24's 145/200 it had 15 gains and 13 regressions (`p=0.8506`), while using
1.92x tokens and producing 54 versus 29 process errors. Treat it as the forward provider-tool-call
diagnostic baseline, not an accuracy or training-data promotion; multi-call turns remain intact.

`src/tool_modules/sql_common/runner.py --interface search-values-execute-sql-v2` evaluates the active separate
two-tool diagnostic; `search-values-execute-sql-v1` remains selectable only for frozen reproduction.
It uses the same causal provider loop and hidden
`bird-set` scorer, but the model sees only bounded database-level `search_values` and
`execute_sql(sql, mode)`. A final SQL must first succeed in inspect mode. Frozen v1 Gate16 scored
7/16 versus fresh atomic version39 at 10/16. V2 preserves the tools while adding
prompt/feedback/context/no-progress optimizations. Its disjoint Holdout Gate15 scored 6/15 versus
fresh atomic version39 at 10/15; both arms were 15/15 legal, while v2 used 38.3% of atomic's tokens
and 76.5% of its actions. V2 passed engineering stability but failed the preregistered accuracy
expansion threshold. Keep both diagnostic-only and do not infer SFT/RL admission from replayed
successes.
`src/tool_modules/direct_sql_search/audit.py` independently audits both versioned interfaces.

`src/tool_modules/sql_common/runner.py --interface execute-sql-submit-sql-v6` evaluates the distinct
`iterative-sql` scheme. The model sees only `execute_sql(sql)` and `submit_sql(sql)`: it must first
execute the exact final query successfully, then submit that same query. Safety, prior-inspection,
timeout, syntax, and execution failures are structured state-preserving feedback and do not end the
episode. A successfully executed submission is terminal and is hidden-scored; wrong-answer or
verifier information is never returned to the model. The runner records this as diagnostic-only,
uses the official DeepSeek endpoint guard, and keeps the historical v4, v3, and
`execute_sql_submit_sql_v2` interfaces only for explicit reproduction. On the 15-task
development gate, v3 scored 7/15 and the failure-derived v4 scored 12/15, both with 15/15 legal
termination; all recorded outcomes passed fresh replay and structural/no-hidden-input-key audit.
Because v4 was optimized on this same set, it requires a new independent holdout before expansion
and remains ineligible for training. V5 keeps execution semantics fixed while incorporating an
externally reviewed prompt/context/audit cleanup. On the active baseline300 frozen Prefix20, which
has zero overlap with the old fixed-200, v5 initially scored 14/20 versus v4 12/20. The completed
Prefix50 scored 36/50 versus 34/50, with four gains, two regressions, exact paired `p=0.6875`,
49/50 versus 50/50 legal, 10 versus seven process errors, nearly identical actions, and 12.1% more
tokens. Stop expansion: v5 remains diagnostic-only and is not a reliable replacement for v4. See
`docs/reports/evaluation/BIRD_ITERATIVE_SQL_V5_PREFIX50_EXPANSION_20260805_ZH.md`.
V6 keeps that complete execution/context/feedback contract and adds only an explicit result-table
rule: scalar = 1x1, each mapped field = one ordered column, and concatenation is legal only when
the task explicitly requests one formatted/combined string. Its disjoint frozen tasks 51–70 Gate20
scored 12/20 versus v5 12/20, with one gain, one regression, 20/20 legal in both arms, and +2.8%
tokens. Because that slice had no direct multi-field-name target or explicit-concatenation control,
the target rule remains unvalidated and v6 is not promoted. See
`docs/reports/evaluation/BIRD_ITERATIVE_SQL_V6_DISJOINT_GATE20_20260805_ZH.md`.
The subsequent zero-overlap Target Gate20 scored 15/20 for v6 versus 10/20 for v5, with five gains,
zero regressions, 20/20 legal in both arms, 99 versus 111 actions, one versus six process errors,
and 339,600 versus 371,036 tokens. Multi-field targets improved from 1/6 to 5/6 and all 10 valid
single-field/ordinary controls were retained. All numeric preregistration gates and audits passed.
The intended `field1+field2` combined-string controls were semantically invalid—the reference also
required separate columns—so this authorizes a new representative paired gate, not SFT/RL or an
anti-overseparation claim. See
`docs/reports/evaluation/BIRD_ITERATIVE_SQL_V6_TARGET_GATE20_RESULT_20260805_ZH.md`.
The subsequent user-authorized v6-only baseline300 run reused the audited tasks 51–70 and requested
the missing 280 episodes. It scored **215/300 (71.67%)** with **298/300 legal**; all 300 records
passed fresh replay and structure audit. Because tasks 1–70 had already been consumed during v5/v6
development or diagnosis, the primary generalization read is the untouched tasks 71–300:
**168/230 (73.04%)** with **228/230 legal**. Both clean 115-task halves scored 84/115. The full run
used 1,698 actions, produced 72 process errors, and consumed 7,000,561 tokens. This is a single-arm
absolute-capability result: no v5 arm was run on the clean 230, so it cannot establish an overall
v6-over-v5 gain or a comparison with another scheme. Keep v6 diagnostic-only. See
`docs/reports/evaluation/BIRD_ITERATIVE_SQL_V6_FLASH_FULL300_20260805_ZH.md`.

Direct-SQL input construction is separately versioned in `src/eval/direct_sql_prompt.py`.
`canonical-json-v1` preserves the historical JSON full-schema control. The diagnostic
`sql-astra-appendix-v1` profile renders DDL with BIRD column descriptions, deterministic live
representative values, PK/FK constraints, and the disclosed SQL-ASTRA step-by-step instructions.
Select it explicitly with `--prompt-profile`; record `--schema-value-count` and any explicit
`--schema-metadata-json`. Prompt profiles do not change decoding, SQL execution, candidate
aggregation, or denotation scoring, and their artifacts require separate result directories.

Denotation metrics are registered in `src/eval/denotation.py`; all active evaluation entry points
select one through `--denotation-comparison` and record the selected name in their manifest.
Candidate generation and aggregation are independent: greedy/sampling settings belong to runners,
pass@k aggregation belongs to `src/eval/passk.py`, and Arctic's execution-result selection belongs
to `src/eval/candidate_selection.py`. Adding a denotation metric therefore does not require changing
decoding or aggregation code.

Cross-model comparisons must also audit model-level `generation_config.json` defaults. vLLM applies
defaults such as `repetition_penalty` and `top_k` when the request omits them, even when temperature
and top-p are explicit. `src/eval/text2sql_passk.py` therefore exposes
`--repetition-penalty` and records it in manifests and result records; the table_rl launcher accepts
the matching `REPETITION_PENALTY` environment variable. Pin the value when compared checkpoints
ship different defaults, or the run is a model-plus-decoding comparison rather than a controlled
checkpoint comparison.

The Arctic-compatible direct-SQL settings are:

```bash
# Headline greedy EX
.venv/bin/python src/eval/text2sql_passk.py ... \
  --n-samples 1 --pass-k 1 --temperature 0 --top-p 1 \
  --execution-timeout-seconds 10 --denotation-comparison bird-set

# Optional eight-candidate execution self-consistency
.venv/bin/python src/eval/text2sql_passk.py ... \
  --n-samples 8 --pass-k 1,2,4,8 --temperature 0.8 --top-p 1 \
  --execution-timeout-seconds 10 --denotation-comparison bird-set \
  --candidate-aggregation arctic-majority
```

Transport failures must be reported separately from semantic policy failures. A run may resume only
when its manifest, task selection, protocol, model, decoding parameters, and denotation comparison
match exactly.

For the current controlled RL candidates, the routine checkpoint-selection evaluation is the full
1,534-question BIRD-dev greedy run (`n=1`, `temperature=0`, `top_p=1`). The former equal-difficulty
300-question K=4 screen produces about 1,200 trajectories but covers only 300 questions; a full-dev
greedy run produces 1,534 trajectories at comparable order of compute and is the more representative
primary comparison. Fixed-prefix scoring remains a secondary behavioral diagnostic. K=4 sampling is
deferred to at most the final selected method and must never replace the full-dev greedy result.

`src/tool_modules/relational_program/evaluator.py` is the diagnostic evaluator for the separate
`relational-program-v6` scheme. It uses the same hidden `bird-set` terminal scorer and causal
model↔harness loop. Its exclusive model prompt exposes only `observe`, `relational_program`, and
`answer_from_context`; the harness derives a DAG from exact program-local parameter references and
maps node operations to internal atomic execution. Version 5 uses disjoint typed objects for source
tables, resident tables/steps, and current-program nodes; only current-program node objects create
DAG edges. Scalar operands and predicate `value_from` use the same typed value-source objects and
lower to unchanged harness-grounded producing-step references. A current-node join with
`base_role` uses that declared namespace for the base node's bare columns. Its outputs are isolated by scheme and remain
`diagnostic_only_pending_protocol_scale_gate`; they cannot be admitted to SFT or RL from evaluation
success alone.

`src/tool_modules/action_block/evaluator.py` defaults to the separate `action-block-v34`
diagnostic. A work
turn is an ordered 1..8 sequence of atomic operations, not a declarative program. The model
declares no DAG, dependency fields, result root, exports, or handles. Same-block backward
references connect already-determined consecutive operations; the harness returns one complete
ordered result/error/blocked entry per submitted operation. The only simplified public primitive
is a one-edge `join(left,right,left_on,right_on,how?)`; the harness deterministically lowers it to
one frozen `join_tables` edge without guessing schema, columns, predicates, or intent. The private
executor name never appears in v34 prompts or feedback. The DeepSeek diagnostic allows up to
40 model actions, records primitive counts separately, uses `history_turns=4` and `bird-set`, and
always writes `sft_export_eligible=false`. Frozen v32/v33 results and v34 results require separate
directories and protocol hashes.
