# Evaluation contract

Tool-agent, RL-checkpoint, and direct-SQL comparisons must use the same task ids, sampling condition,
database snapshot, external knowledge, generated-query deadline, and denotation metric. Every result
must name its metric; strict multiset and BIRD reference set equality are not interchangeable.

For released BIRD EX reporting, `bird-set` matches the reference scorer by comparing sets of rows,
ignoring row order and duplicate multiplicity. Training replay, grounding gates, and reward audits
retain normalized strict-multiset comparison unless an experiment explicitly declares otherwise.

`src/eval/rollout.py` is the single-sample tool-agent evaluator and
`src/eval/rollout_passk.py` is the multi-sample evaluator. `src/eval/text2sql.py` and
`src/eval/text2sql_passk.py` are direct-SQL controls. Result directories and manifests must not mix
different denotation contracts.

Transport failures must be reported separately from semantic policy failures. A run may resume only
when its manifest, task selection, protocol, model, decoding parameters, and denotation comparison
match exactly.
