# OmniSQL / Qwen SFT-1+SFT-2 tool-vs-direct fixed-200 evaluation

Date: 2026-07-24

## Scope

This isolated ablation uses historical protocol `v2i-state-only-join-feedback-r2`, not the current
experimental protocol. Both models received the same deterministic 1,024-action mixture
(512 SFT-1 + 512 non-replay SFT-2 actions) and the same two-epoch rank-16 QLoRA schedule.

The primary comparison in this report uses the exact same difficulty-stratified 200 BIRD-dev
questions and the same `strict-multiset` denotation comparison:

- direct SQL: the post-SFT checkpoint generates SQL, which is executed on SQLite;
- tool use: the post-SFT checkpoint acts in the closed-loop harness, and only the exact
  harness-owned table cited by `answer_from_context` is graded.

Model-authored answer values, SQL text, reasoning, and evidence-column permutations are not tool
scoring fallbacks.

## Result

| Post-SFT model | Direct SQL | Tool, strict carrier | Tool, raw-JSON carrier diagnostic | Tool vs direct | Relative retention |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-Coder-7B | 60/200 (30.0%) | 0/200 (0.0%) | 19/200 (9.5%) | -20.5 pp | 31.7% |
| OmniSQL-7B | 100/200 (50.0%) | 0/200 (0.0%) | 17/200 (8.5%) | -41.5 pp | 17.0% |

The paired reductions under the carrier-compatible tool diagnostic are large:

- Qwen: both correct 11, direct only 49, tool only 8, both wrong 132;
  exact McNemar p = `2.72e-08`.
- OmniSQL: both correct 13, direct only 87, tool only 4, both wrong 96;
  exact McNemar p = `2.26e-21`.

Difficulty breakdown:

| Model | Metric | Simple (80) | Moderate (60) | Challenging (60) |
|---|---|---:|---:|---:|
| Qwen | Direct SQL | 38 | 13 | 9 |
| Qwen | Tool diagnostic | 10 | 4 | 5 |
| OmniSQL | Direct SQL | 48 | 29 | 23 |
| OmniSQL | Tool diagnostic | 12 | 4 | 1 |

The stronger direct-SQL prior of OmniSQL did not transfer proportionally to the historical tool
interface. Under the carrier diagnostic, its absolute tool score is slightly below Qwen's.

## Why the strict format failed

This is not a parser defect and not malformed source data.

- All 1,024 source targets match the canonical
  `<think>...</think><tool_call>{...}</tool_call>` shape.
- The exact LLaMA-Factory preprocessing audit retained the complete final target for 1,024/1,024
  records for both tokenizers. Neither masking nor the 6,400-token cutoff removed the wrapper.
- `<tool_call>` and `</tool_call>` are vocabulary tokens `151657` and `151658`. They are present in
  the training labels. They are not in `all_special_ids`, and decoding with
  `skip_special_tokens=True` preserves them.
- The adapters update only `q/k/v/o/up/down/gate_proj`; embeddings and `lm_head` remain frozen.
  Wrapper tokens account for about 1.7% of the 120,914 target tokens under an unweighted mean loss.

At the first tool boundary of the same saved request, the adapter greatly increased the desired
opening token's probability but did not make it greedy top-1:

| Model | Base rank of `<tool_call>` | SFT rank | Greedy SFT token |
|---|---:|---:|---|
| Qwen | 68,547 | 82 | `<|im_start|>` |
| OmniSQL | 49,205 | 83 | `<|im_start|>` |

Offline vLLM token inspection confirmed that the models did not generate token `151657`; vLLM did
not remove a correct wrapper. It did remove the incorrectly generated `<|im_start|>` special token
from returned text, exposing a mostly correct raw tool JSON with occasional surrounding garbage.

The root cause is therefore an optimization/configuration failure at two low-frequency boundary
tokens: the LoRA learned much of the action JSON but did not calibrate the strict carrier. The
parser correctly surfaced that failure.

### Why the earlier Qwen2.5-7B SFT did not show this failure

The earlier successful tool checkpoint is not the same base-model/data regime:

- it used `Qwen2.5-7B-Instruct`, while this ablation uses
  `Qwen2.5-Coder-7B-Instruct` and OmniSQL-7B;
- its selected SFT-1 checkpoint had already seen 4,319 action targets for one epoch, then SFT-2
  continued from that adapter on 11,874 additional targets for one epoch;
- this ablation starts from each new base and mixes 1,024 targets for two epochs: 2,048 target
  exposures versus about 16,193 in the earlier staged checkpoint, a 7.9x difference;
- the earlier continuation reduced the learning rate from `1e-4` in SFT-1 to `5e-5` in SFT-2;
  this ablation uses `1e-4` from the base throughout.

The base prior is the most direct difference. On the exact same saved prompt and think prefix,
before any adapter:

| Base/checkpoint | Rank of `<tool_call>` | Log probability |
|---|---:|---:|
| Qwen2.5-7B-Instruct base | 1 | log p = -1.305 |
| Earlier Qwen2.5-7B SFT-1 checkpoint-270 | 1 | log p = -0.0024 |
| Qwen2.5-Coder-7B-Instruct base | 68,547 | log p = -19.469 |
| Current Qwen-Coder 1K adapter | 82 | log p = -8.175 |
| OmniSQL-7B base | 49,205 | log p = -14.793 |
| Current OmniSQL 1K adapter | 83 | log p = -8.167 |

The old and new adapters have the same projection-only LoRA target set and no saved output head.
The earlier run succeeded despite the frozen `lm_head` because the Instruct base already had the
correct carrier as its top-1 continuation and the staged corpus reinforced it. The current bases
lack that prior, and the much smaller mixed run cannot move the wrapper token from rank ~50k-70k
to rank 1.

## Carrier-compatible diagnostic

The diagnostic parser first tries the strict parser, then accepts exactly one complete
`{"tool": ..., "arguments": ...}` object plus exactly one non-empty think block. It retains the
current tool whitelist and strict argument validation. It does not repair JSON, infer tools,
normalize arguments, or accept terminal shorthand.

This changes only action transport. Every accepted action still executes through the harness, and
terminal scoring remains tool-output-only. It is not a protocol-compliance score and must always be
reported beside the strict 0/200 result.

Across the diagnostic runs:

| Model | Legal termination | Successful tool solve | Wrong terminal table | Other terminal failure |
|---|---:|---:|---:|---:|
| Qwen | 164/200 | 19 | 145 | 36 |
| OmniSQL | 135/200 | 17 | 118 | 65 |

Qwen executed 1,247 non-terminal tool calls; OmniSQL executed 1,188. Their mean actions were 7.87
and 8.07 respectively. OmniSQL had more argument-validation failures (43 task-level, 201 turn-level)
than Qwen (12 task-level, 62 turn-level).

Thus fixing only the wrapper is necessary but insufficient: after the carrier is bypassed, most
legal terminal trajectories still cite an incorrect tool-derived table.

## Implication

The 1K-action QLoRA does not yet preserve the models' direct-SQL capability when that capability
must be expressed through this tool interface. Qwen retains about one third of its paired direct
accuracy; OmniSQL retains about one sixth.

The next experiment should separate two gates:

1. carrier gate: make strict wrapper emission reliable, using a targeted boundary-token objective
   and/or an output-head-capable adapter, then require a small strict-format smoke test to pass;
2. semantic tool gate: only after carrier reliability, improve multi-step arguments, execution
   recovery, and final evidence-table construction.

A raw-JSON protocol would avoid this particular special-token bottleneck, but adopting it would be
a protocol decision, not an evaluation repair.

## Remote artifacts

- Direct same-cohort rescore:
  `/home/dengyan/tabular_rl_outputs/ablation/omnisql_sft12_1k/results/direct_sql/fixed200_strict_multiset_rescore.json`
- Final comparison:
  `/home/dengyan/tabular_rl_outputs/ablation/omnisql_sft12_1k/results/TOOL_VS_DIRECT_FIXED200_SUMMARY.json`
- Qwen tool diagnostic:
  `/home/dengyan/tabular_rl_outputs/ablation/omnisql_sft12_1k/results/tool/qwen25-coder_sft_fixed200_tool_output_only_strict_multiset_carrier_compatible`
- OmniSQL tool diagnostic:
  `/home/dengyan/tabular_rl_outputs/ablation/omnisql_sft12_1k/results/tool/omnisql_sft_fixed200_tool_output_only_strict_multiset_carrier_compatible`
