# Selectable tool schemes

## Contract

The repository exposes two complete and independently selectable model action schemes:

| Scheme id | Model turn | Top-level actions | Student carrier |
|---|---|---|---|
| `atomic` | exactly one primitive tool | the original planning, perception, relational, and terminal tools | `<think>` plus one tagged `<tool_call>` |
| `action-block` | one ordered block of 1..N primitive tools, or terminal | `action_block`, `answer_from_context` | `<think>` followed directly by one raw JSON action |

The ids are defined in `src/sft/tool_schemes.py`. A model sees exactly one scheme. Do not combine
both top-level schemas in one prompt and do not infer a scheme from trajectory shape.

Both schemes share the harness-owned primitive relational semantics and resident state. They do
not share:

- model-visible system prompts;
- top-level argument schemas;
- assistant carriers or strict parsers;
- trajectory or manifest identities;
- SFT targets;
- model adapters/checkpoints or result directories.

## Runtime adapters

`src/sft/tool_schemes.py` builds a complete `ToolScheme` object containing the prompt, protocol
version/hash, carrier, top-level actions, primitive actions, and batch bound. It also provides
strict render/parse round trips for local student models.

The original DeepSeek action-block evaluator retains provider-native reasoning plus raw visible
JSON. Local/SFT action-block models use the semantically equivalent inline carrier:

```text
<think>one non-empty reason</think>
{"tool":"action_block","arguments":{"calls":[...]}}
```

The provider and student carriers are adapters around the same structured action. They are
separately hashed and recorded.

## Evaluation and causal rollout

Preferred unified evaluation entry:

```bash
PYTHONPATH=src/harness:src/sft:src/eval:src/rl \
  .venv/bin/python src/eval/run_tool_scheme.py \
  --tool-scheme atomic -- <atomic rollout.py arguments>

PYTHONPATH=src/harness:src/sft:src/eval:src/rl \
  .venv/bin/python src/eval/run_tool_scheme.py \
  --tool-scheme action-block -- <action-block evaluator arguments>
```

The action-block launcher selects the safe low-friction interface unless an explicit historical
ablation flag is supplied. DeepSeek v4 uses the default provider-native carrier. A local or
student-model OpenAI-compatible endpoint uses:

```bash
--assistant-carrier inline-think-raw-json --model <student-model-name>
```

The evaluator then parses the model's complete inline turn directly; it does not require a
provider-native reasoning field.

The matching causal generation entry is
`src/sft/generate_tool_scheme_rollouts.py`. It dispatches to the original external-teacher loop for
`atomic` and the real model↔harness action-block loop for `action-block`.

Every new episode and manifest records:

- `tool_scheme`;
- `tool_scheme_registry_version`;
- protocol version/hash;
- assistant carrier;
- scheme-specific budgets.

Result directories must remain isolated by scheme.

## SFT

Atomic bounded-history SFT continues through `src/sft/build_rolling_sft_data.py`.

Action-block bounded-history SFT is exported by
`src/sft/build_action_block_sft_data.py`. It:

- requires the source episode to be explicitly marked `sft_export_eligible=true`;
- requires `bird-set`, correct legal termination, and no process errors;
- freshly replays every primitive action and terminal denotation;
- converts provider-native legal history to the inline student carrier;
- creates one last-turn-only record per causal decision prefix;
- records source and target protocol identities;
- never makes failed or blocked block calls into SFT targets.

Current action-block evaluation artifacts remain ineligible because the fixed-200 promotion gate
was not passed. The exporter is an implementation path for a future promoted protocol, not an
authorization to train on v10/v11 diagnostics.

Datasets and adapters from different schemes must not be mixed. `assert_record_tool_scheme`
provides the mandatory pre-export guard.

## RL

`src/rl/tool_environment.py::create_tool_use_env` constructs either:

- `ToolUseEnv` for `atomic`; or
- `ActionBlockToolUseEnv` for `action-block`.

`group_reinforce.py --tool-scheme ...` can therefore run matched result-only training for both
schemes. Checkpoint metadata, rollout logs, and metrics include the scheme and scheme-specific
budgets, so resume cannot silently switch protocols.

Atomic process credit remains available only for `atomic`. Action-block process optimization is
rejected until a block-to-primitive credit objective is independently designed and audited.

## Paired experiment rule

For parallel comparison, pin task ids, base model, initialization checkpoint, decoding,
denotation metric, and provider/runtime limits. Use separate:

- SFT datasets and dataset registry names;
- adapters and checkpoint directories;
- rollout/result directories;
- protocol hashes and scheme ids.

Compare terminal `bird-set`, legal rate, transport failures, semantic process errors, model turns,
primitive actions, and tokens. For action-block additionally report block size, root errors,
blocked descendants, and information-barrier violations.
