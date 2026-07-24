# BIRD version24 fixed-200 SFT admission audit

Date: 2026-07-24

## Final decision

The frozen 200-task tool-usability cohort contributes the successful causal trajectories to the
fixed-1K SFT dataset. Admission uses the evaluation run's declared `bird-set` denotation metric,
fresh replay, structural/no-leak checks, and an exact LLaMA-Factory full-prefix token gate.

| Gate | Result |
| --- | ---: |
| Terminal `bird-set` success | 145/200 (72.5%) |
| Structural/no-leak audit | 145/145 |
| Fresh database replay | 145/145 |
| Legal action records before token filtering | 969 |
| Records retained at Qwen cutoff 6400 | 831/969 |
| Final target truncations | 0 |
| Current source/state truncations | 0 |
| Records omitted for an incomplete oldest rolling-history pair | 138 |

All 145 successful episodes contribute at least one supervised action. Thirty-three episodes lose
one or more late records, so 112 episodes remain complete at record level.

The earlier exploratory `think <= 300 words` rule is **not** an admission gate. It was a
demonstration-quality heuristic, not a causal or tokenizer-safety requirement, and would discard
valid model-authored reasoning. No reasoning was shortened, rewritten, or removed. The only final
record-level omission rule is exact tokenizer evidence that the complete causal prefix does not fit
the configured cutoff.

## Inputs and metric

Frozen sources:

- `data/trajectories/tool_usability_20260724/version24_fixed200_first50_bird_set.all.jsonl`
- `data/trajectories/tool_usability_20260724/version24_fixed200_remaining150_bird_set.all.jsonl`

The normalized source trajectories predate the per-trajectory `denotation_comparison` field, so
replay receives the source manifests' explicit `bird-set` value. This is deliberate: BIRD reference
EX is a set comparison. Six of these 145 successes would fail the separate
`strict-multiset` training audit because of duplicate multiplicity; the two metrics must not be
mixed or reported as interchangeable.

The replay helper was fixed during this audit so new trajectories persist their comparison mode
and replay reuses it. No frozen evaluation row was edited.

## Causal rendering and token gate

The rolling exporter reconstructs each target from the legal episode prefix only:

- current catalog, question, optional external knowledge, and resident state;
- legal rolling assistant/history turns available before the target;
- latest recoverable environment feedback when present;
- exactly the model-authored next action as target.

Gold SQL, future actions, the current tool result, and rejected calls are not SFT targets. Every
rendered record was freshly executed before export.

Both Qwen2.5-7B-Instruct and Qwen2.5-Coder-7B-Instruct were audited with the installed
LLaMA-Factory `qwen` template, `mask_history=true`, and `cutoff_len=6400`. Their retained record sets
are byte-identical. For this fixed-200 component:

- first 50: 222 rendered, 204 retained;
- remaining 150: 747 rendered, 627 retained;
- maximum untruncated prefix length: 11,202 tokens;
- maximum target length: 1,792 tokens.

The 138 omitted records all preserve their final target under truncation, but lose part of the
oldest task/system pair. They are excluded because training on a partial source would no longer
match the model-visible protocol. Earlier records from the same causal episodes remain valid and
are retained; no episode or trajectory is spliced or rewritten.

## Relationship to the fixed-1K training set

These 831 records are combined with verified records from the disjoint difficulty-stratified
additional-800 cohort. The final selection and training run are documented in
`docs/reports/sft/BIRD_EXTERNAL_TEACHER_FIXED1000_DUAL_7B_SFT_20260724.md`.
