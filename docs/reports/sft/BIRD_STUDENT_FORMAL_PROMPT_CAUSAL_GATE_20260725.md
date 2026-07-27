# BIRD student formal-prompt causal gate (2026-07-25)

## Question

Can the student prompt restore exact nested action syntax without copying the external teacher's
case-heavy generation guidance or changing tool semantics?

All comparisons use Qwen2.5-7B-Instruct, one-epoch QLoRA, the same deterministic 400 complete
causal episodes (2,521 last-turn targets), the same optimizer/configuration, greedy closed-loop
rollout over the same first ten BIRD dev tasks, rolling legal history, and `bird-set`. Only the
student system prompt changes.

## Shared implementation boundary

`src/sft/public_tool_contract.py` is the shared public action contract. It supplies concise
tool-local semantics to the student renderer and nested enums/key sets to strict protocol
validation. Teacher-only cases remain separate. The canonical student prompt is unchanged and the
formal prompt remains an explicit candidate rather than a promoted default.

The exact remote LLaMA-Factory token audit for the high-entropy candidate retained 2,521/2,521
records under the full-prefix 6,400-token policy. No current source or target was truncated;
original sequence length was min/p50/p90/max 1,737/3,192/4,172/4,760 tokens.

## Full formal grammar: rejected

The first formal candidate expanded nested syntax for every tool, including the low-entropy
`plan.ops` shape. Its from-base checkpoint-10 result was 0/10 correct, 2/10 legal, and 17.7 mean
actions. The ten episodes contained 154 `plan` attempts despite only 23 plan targets among all
2,521 training records. The matched lean checkpoint-10 was also 0/10 correct and 2/10 legal but
used only 5.2 mean actions.

This is a direct structural prompt failure: a repeated formal action shape became a copyable action
template and distorted tool-selection frequency. More training cannot be used to explain it away.

## High-entropy formal grammar

The revised candidate expands only the nested argument forms that are easy to confuse:
conditions, projections, grounded scalar operands, multi-relation joins, grouped aggregations, and
terminal table evidence. It removes duplicate formal shapes for `plan`, `extreme_value_select`,
and `set_op`; their callable signatures and semantics remain in the ordinary tool contract.
The grammar explicitly states that expansion does not imply tool priority.

The runtime prompt is 7,112 characters / 1,543 Qwen tokens, compared with 5,096 characters /
1,050 tokens for the canonical lean prompt. It is below the predeclared 1,400--1,800 token target
and contains no teacher cookbook, task-specific values, or gold-derived solution path.

## Early checkpoint result

| Prompt / checkpoint | Correct | Legal | Mean actions | Plan attempts | Join errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| Lean step 10 | 0/10 | 2/10 | 5.2 | — | — |
| Full formal step 10 | 0/10 | 2/10 | 17.7 | 154 | 11/11 |
| High-entropy formal step 10 | 0/10 | 1/10 | 6.6 | 10 | 13/13 |
| Lean step 20 | 0/10 | 1/10 | 12.1 | 0 | 12/12 |
| High-entropy formal step 20 | 0/10 | 0/10 | 18.7 | 0 | 10/10 |

The high-entropy restriction repairs the action-frequency distortion: plan attempts fall from 154
to 10 and the action count returns close to the matched lean checkpoint. It does not yet improve
tool competence. All 13 join calls used a bare left key where the protocol requires the logical
`relation.column` name, despite the correct rule being present in the prompt. The failure is no
longer a copied placeholder or a plan loop; at step 10 the model has not learned the join
interface from the 186 join targets in the mixture.

Because the structural regression is gone and ten update steps are too early to test a
low-frequency target, training is resumed from the exact optimizer state to checkpoint 20. That
is the last permitted small gate for this candidate: continuation requires an actual increase in
legal join use, not merely lower loss.

Checkpoint 20 fails that gate. No join succeeds, no episode terminates legally, and five episodes
exhaust the 30-action budget. The failure moves from the removed formal plan template to repeated
low-information actions: 134 `describe_table` attempts, including four episodes with 27--30
schema descriptions. Ten join calls still use a bare left key or, in one case, a dotted right key.
Ten condition calls contain malformed carrier JSON. There are zero transport or context retries,
so this is a semantic policy failure rather than an API failure.

The training data is not teaching the opposite join convention: its 186 join targets contain 216
edges, with 216/216 dotted left references and 216/216 bare right references. The failed candidate
therefore shows that adding a compact textual grammar to a low-frequency, mixed action dataset is
not sufficient, and can make early policy dynamics worse even when the grammar is correct.

## Decision

Both formal candidates are rejected. Do not continue either to checkpoint 40, do not promote the
formal prompt to runtime, and do not use its pilot as an approved SFT source. The canonical lean
prompt remains the default by compatibility, not because the 400-episode pilot validated it.

The next experiment should hold the 1,050-token canonical prompt fixed and change the supervision
distribution rather than add more prompt prose: construct a small causal curriculum whose
selection gate explicitly covers join, aggregate, scalar, terminal-shape, and feedback-recovery
targets. Early checkpoints must be evaluated on both a capability slice and a no-regression
control slice. This tests whether exact action semantics can be learned from data, which is the
intended student boundary, without making tool-selection policy depend on a longer system prompt.

Completed diagnostic configs are archived under `archive/experiments/sft/configs/`; none remains
beside the active SFT entry points.
