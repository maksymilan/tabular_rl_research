# Qwen2.5-Coder checkpoint-320 tool use versus base direct SQL — first-100 diagnostic

Date: 2026-07-26

## Outcome

On the exact first 100 BIRD-dev tasks, the version26 raw-JSON Coder
`checkpoint-320` tool agent and the untrained Qwen2.5-Coder-7B-Instruct direct-SQL
control have essentially the same `bird-set` accuracy:

| Mode | Correct | Accuracy |
| --- | ---: | ---: |
| SFT Coder `checkpoint-320`, closed-loop tools | 26/100 | 26% |
| Untrained Coder, direct SQL | 27/100 | 27% |
| Tool minus direct SQL | -1 | -1 pp |

The paired table is:

| Outcome | Tasks |
| --- | ---: |
| Both correct | 13 |
| Tool only | 13 |
| Direct SQL only | 14 |
| Both wrong | 60 |

The exact two-sided McNemar p-value is 1.0. A paired normal interval for the
accuracy difference is approximately `[-11.2, +9.2]` percentage points. This
100-task diagnostic therefore provides no evidence of an aggregate gain or
loss from the tool route. It does show substantial complementarity: 27 tasks
are solved by exactly one of the two modes.

This result must not be compared directly with the untrained Coder's full-dev
`748/1534 = 48.76%` result. The tool run covers only the first 100 tasks, of
which 89 are from `california_schools` and 11 from `financial`. The matched
direct-SQL score on those same tasks is 27%, so the low absolute number is
largely a cohort property. A full-dev tool run is still required for a
population-level BIRD-dev comparison.

## Evaluation contract

Both modes use:

- the same first 100 tasks from `data/eval_inputs/bird_dev_20240627.jsonl`;
- greedy decoding, one sample, `temperature=0`, `top_p=1`;
- `max_tokens=1024`;
- BIRD external knowledge;
- `bird-set` denotation.

The direct-SQL control uses the untouched
`/home/dengyan/models/Qwen2.5-Coder-7B-Instruct`, complete upfront schema, no
execution feedback, a 20-second generated-query deadline, thinking disabled,
and a vLLM server started with `--generation-config vllm`. No request-level
repetition penalty is supplied, so this matched control uses vLLM's default
`1.0`.

The tool run uses the same base plus:

`/home/dengyan/tabular_rl_outputs/checkpoints/qwen2.5-coder-7b-bird-student-raw-json-6400-qlora/checkpoint-320`

It uses protocol version26, strict `<think>...</think>` plus one raw JSON
action, rolling legal history with four turns, resident state, at most 30
semantic actions, and `--generation-config vllm`.

As a sensitivity check, the previously completed base direct-SQL artifact with
an explicit repetition penalty of 1.05 scores 28/100 on this cohort. Against
that control the tool delta is -2 pp, with 13 tool-only and 15 SQL-only tasks
and exact McNemar p=0.851. The conclusion is unchanged.

## Tool reliability

The tool agent reaches a grounded legal terminal on 52/100 tasks:

| Terminal outcome | Tasks |
| --- | ---: |
| Correct legal answer | 26 |
| Legal wrong answer | 26 |
| Protocol-error terminal | 23 |
| Max steps | 16 |
| Execution-error terminal | 7 |
| Context overflow | 2 |

Mean trajectory length is 10.58 actions. The 100 trajectories contain 107
recoverable process-error events:

| Error event | Count |
| --- | ---: |
| Protocol | 77 |
| Execution | 28 |
| Argument validation | 2 |

There are zero API transport retries, 13 bounded context retries, and two
terminal context overflows.

All 77 protocol events have the same strict-parser message: the response is
not exactly one complete think block followed by one JSON action. Direct
inspection separates them into:

- 38 unclosed think blocks, usually length-truncated after long reasoning;
- 39 repeated/nested literal think tags, often caused by recovery reasoning
  that discusses the `<think>` marker itself.

There is no recurrence of the earlier random-Unicode substitution of
`<tool_call>` boundaries. Version26 no longer uses those tags. The remaining
carrier problem is long or self-referential reasoning around the still-strict
think block.

No-progress behavior also remains material:

- 26/100 tasks repeat an identical parsed action consecutively;
- 18/100 repeat one action at least five times;
- the maximum identical-action run is 28.

The most frequent parsed calls are `read_subtable` (305),
`inspect_column` (209), `condition_filter` (162), and `describe_table` (101).
This is consistent with the observed long perception/re-read loops.

## Interpretation

The old Coder-specific output-head failure is no longer the primary blocker.
The current checkpoint can produce legal raw-JSON tool trajectories and
matches base direct SQL in aggregate on this diagnostic. Its residual
weaknesses are:

1. only 52% legal terminal completion;
2. long-think truncation and literal-tag recovery loops;
3. repeated no-progress perception actions;
4. relational execution errors;
5. 26 legal but semantically wrong terminal answers.

The paired wins show that the tool interface is not simply a degraded SQL
translator. It recovers 13 tasks missed by direct SQL, while losing 14 tasks
that direct SQL solves. Improving trajectory control could therefore turn the
existing complementary tool wins into an aggregate gain.

`checkpoint-320` was frozen for this comparison. Training was separately
resumed from its exact optimizer, scheduler, and RNG state toward step 560.
Any later checkpoint requires the same paired gate; these results must not be
attributed to checkpoint 560.

## Artifacts

Tool:

`data/results/diagnostics/qwen25_coder7b_raw_json_cp320_tool_first100_greedy1_bird_set`

- manifest SHA-256:
  `708d0626d9b864948dc3ccc8c91721b4b15e97912ffeaf91b5984f73f52412ce`
- all records SHA-256:
  `f9586d6572702dffa328bd3f42763993e693d653bfe0c3ba29179ee2f473dac0`
- summary SHA-256:
  `ad35f9c26d4e7647e2f9b86b5c301a6f6b131ad502a4961ff7eca851248de2d0`

Matched untrained direct SQL:

`data/results/diagnostics/qwen25_coder7b_base_direct_sql_first100_greedy_rp100_bird_set`

- manifest SHA-256:
  `bd1a86a5759c4eaa2603edf86ac1d543aaedb846f8b85f8481ead746ab96819d`
- all records SHA-256:
  `ca4a5cb5362f8f1d783cfe2ef862c0a346ec3d216402d14b7f62257819f2d06d`
- summary SHA-256:
  `18cd4c95b3d11819f44587227a085b77163c9ff6a02bcc6143cf69e50133d216`

Historical full-dev base control:

`data/results/qwen2.5_coder_7b_bird_direct_sql_base_greedy_rp105_dev1534_bird_ex`
