# Evaluation contract

Tool-agent, RL-checkpoint, and direct-SQL comparisons must use the same task ids, sampling condition,
database snapshot, external knowledge, generated-query deadline, and denotation metric. Every result
must name its metric; strict multiset and BIRD reference set equality are not interchangeable.

For released BIRD EX reporting, `bird-set` matches the reference scorer by comparing sets of rows,
ignoring row order and duplicate multiplicity. Training replay, grounding gates, and reward audits
retain normalized strict-multiset comparison unless an experiment explicitly declares otherwise.
Arctic-Text2SQL-R1-7B's headline BIRD-dev result uses greedy decoding (`n=1`,
`temperature=0`) with this `bird-set` contract. The existing direct-SQL greedy launcher uses those
decoding and denotation settings; set the generated-query timeout to 10 seconds when matching the
pinned upstream evaluator rather than the launcher's more permissive 20-second default. Arctic's
optional eight-sample majority voting is a different candidate aggregation method and must not be
reported as pass@k. Direct-SQL multi-candidate evaluation exposes the upstream soft-denotation
medoid as `--candidate-aggregation arctic-majority`; use `n=8`, `temperature=0.8`, `top_p=1`,
a 10-second generated-query timeout, and `bird-set` for that optional Arctic setting.

`src/eval/rollout.py` is the single-sample tool-agent evaluator and
`src/eval/rollout_passk.py` is the multi-sample evaluator. `src/eval/text2sql.py` and
`src/eval/text2sql_passk.py` are direct-SQL controls. Result directories and manifests must not mix
different denotation contracts.

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
