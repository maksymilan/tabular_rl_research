# Qwen2.5-Coder-7B exact carrier repair

Date: 2026-07-25

## Outcome

The post-SFT random-Unicode replacement of `<tool_call>` and `</tool_call>` is
fixed without weakening the protocol parser or changing the trained relational
policy. The source checkpoint's LoRA tensors remain byte-for-byte identical;
the repair trains only the two affected output-head rows.

This is a carrier repair, not a new full SFT run and not a claim about full
BIRD-dev accuracy.

## Root-cause isolation

Both the epoch-1 `checkpoint-258` and epoch-2 `checkpoint-516` produced zero
legal samples in the frozen five-task, four-sample diagnostic. Reasoning and
tool JSON were generally present, but the two exact carrier boundaries were
replaced by Thai, Cyrillic, or other Unicode tokens.

A direct Transformers+PEFT probe reproduced the same failure from
`checkpoint-516`; disabling the adapter in the same process made the base
model emit bare JSON. This rules out the vLLM parser and LoRA server as the
cause.

The relevant model facts are:

- `<tool_call>` and `</tool_call>` are dedicated token IDs `151657` and
  `151658`;
- Qwen2.5-Coder-7B has `tie_word_embeddings=false`;
- the source QLoRA targets only `q_proj`, `k_proj`, `v_proj`, `o_proj`,
  `gate_proj`, `up_proj`, and `down_proj`;
- neither `lm_head` nor selective output-token rows were trainable.

Token-averaged SFT loss therefore hid a two-token exact-carrier failure while
reasoning and JSON content improved normally.

## Repair method

Implementation:

- `src/sft/carrier_repair.py`;
- `src/sft/train_carrier_token_repair.py`;
- `src/sft/merge_carrier_repaired_model.py`;
- `src/eval/probe_lora_carrier.py`.

The repair:

1. loads the frozen source LoRA over the original Coder base;
2. obtains hidden states with the source adapter active and all source weights
   frozen;
3. optimizes only the two corresponding `lm_head` rows;
4. uses exact boundary positions as positive examples and eight deterministic
   non-boundary assistant positions per record as negative contexts;
5. saves the original LoRA tensors plus the two selective output rows;
6. verifies every source LoRA tensor is exactly unchanged.

The pilot selected 64 complete records, round-robin across all available tool
types, from the unchanged 4,119-record SFT dataset. Records longer than 5,200
tokens were excluded from this repair pilot rather than truncated. The 64
records had lengths 3,382–5,190 tokens and covered all 12 public tools.

Training parameters:

- trainable parameters: `2 * 3584 = 7,168`;
- optimizer updates: 8;
- learning rate: `1e-3` with cosine decay;
- effective batch: 8;
- source LoRA exact-tensor preservation: passed;
- negative-context boundary top-1 rate during training: 0.

Repair adapter:

`/home/dengyan/tabular_rl_outputs/checkpoints/qwen2.5-coder-7b-bird-external-teacher-fixed1000-carrier-repair-pilot64`

## Deployment

vLLM 0.19.1 correctly rejects PEFT's
`lm_head.token_adapter.trainable_tokens_delta` as an unsupported dynamic LoRA
weight. The adapter is therefore merged into a standard BF16 model rather than
handled with an inference-time parser workaround.

Merged model:

`/home/dengyan/models/Qwen2.5-Coder-7B-Instruct-external-teacher-carrier-repair-pilot64`

The merge gate verified:

- both carrier rows are exactly preserved after BF16 casting;
- no PEFT tuner layer remains;
- vLLM loads the model in standalone mode;
- generation uses `--generation-config vllm`.

## Validation

### Raw generation

The repaired PEFT adapter produced canonical tool-call boundaries on 5/5 direct
Transformers probes.

### Paired frozen diagnostic

The same first five BIRD-dev tasks and four samples per task used by the failed
source-checkpoint diagnostic were rerun:

| Metric | Source Coder checkpoint | Repaired merged model |
| --- | ---: | ---: |
| Legal samples | 0/20 | 14/20 |
| Correct samples | 0/20 | 8/20 |
| Task pass@4 | 0/5 | 3/5 |
| Complete canonical carrier turns | 0 usable trajectories | 146/147 |

The one incomplete repaired turn was length/context truncated; no random
Unicode boundary substitution remained.

Artifact:

`data/results/diagnostics/qwen2.5_coder_7b_external_teacher_fixed1000_carrier_repair_pilot64_merged_tool5_passk4`

### Twenty-task diversity gate

A separate greedy run covered the first 20 BIRD-dev tasks:

- complete canonical carrier turns: 216/218 (99.08%);
- random-Unicode carrier substitutions: 0;
- legal termination: 12/20;
- `bird-set` correct: 7/20.

The four recorded protocol events were two length-truncated generations and
two correctly delimited calls with malformed JSON. They are not recurrences of
the repaired boundary-token failure. Remaining non-legal outcomes are
execution-error recovery, max-step, context-overflow, and ordinary policy
errors.

Artifact:

`data/results/diagnostics/qwen2.5_coder_7b_external_teacher_fixed1000_carrier_repair_pilot64_merged_tool20_greedy1`

## Decision

The exact carrier regression is repaired. Future Coder evaluation should use
the merged model above, not the unsupported selective-token PEFT adapter and
not a permissive JSON extractor. A larger accuracy evaluation can now measure
the trained tool policy without the carrier defect, but its relational
accuracy is a separate question.
