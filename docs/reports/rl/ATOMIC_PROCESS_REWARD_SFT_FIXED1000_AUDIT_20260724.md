# Atomic process reward audit on the fixed-1000 SFT data

Date: 2026-07-24

## Decision

The formal atomic process-reward equation is now implemented end to end in the active RL code.
Every active scorer and RL environment is fixed to `bird-set`; the controlled context is bounded
rolling legal history with four turns and resident observations.

On the exact retained SFT dataset, deterministic reward computation passes completely:

- 703/703 contributing source episodes replay to the correct `bird-set` denotation;
- 4,119/4,119 retained SFT turns resolve to their source action;
- 703/703 episodes have a harness-owned final dependency slice and nonzero process signal;
- 703/703 satisfy target SQL/result support and reward-conservation checks;
- zero replay errors occurred.

This establishes reward coverage and algebraic correctness. It does **not** establish policy
improvement. A frozen SFT checkpoint, online result-only/process runs, and held-out BIRD EX
evaluation are still required for that claim.

## Fixed experimental contract

| Variable | Value |
| --- | --- |
| Denotation metric | `bird-set` only |
| Context | `rolling-legal-history` |
| History | 4 successful assistant/observation pairs |
| Observation style | resident |
| SFT targets | 4,119 |
| Contributing episodes | 703 |
| Source actions, including rejected actions | 4,890 |
| Legal source actions | 4,767 |
| Rejected actions | 123/4,890 = 2.52% |

The source SFT manifest previously reported 704 contributing episodes because a `defaultdict`
lookup inserted an episode whose every turn had been rejected by the token gate. The episode was
`bird_train_02705`. The manifest builder and regression test are fixed; the corrected values are
704 source episodes and 703 contributing episodes. Dataset and index hashes are unchanged.

## Formula coverage

The active implementation computes

```text
r_it = C_i c+_it
       + (1-C_i) eta DeltaPhi_it
       + lambda_A A_it
       - P_i c-_it
```

from harness replay rather than model reasoning. Coverage is:

| Formal item | Implementation status |
| --- | --- |
| `C_i` | Terminal executed denotation under `bird-set` |
| `B_it` | Harness provenance backward slice from the cited terminal relation |
| `E_it` | First acquisition of harness-grounded evidence used by the final slice |
| `DeltaPhi_it` | Hidden training-side gold T/C/R support; bounded rows use raw BIRD cell equality |
| `S+_it`, `S-_it` | Fixed-root log search reduction and unverified-empty penalty |
| `F_it`, ignored feedback | Previous real error/empty result plus changed legal action, changed state, and verified progress |
| Tool/protocol error | Negative credit on the rejected assistant turn |
| Repeat/no-op/guess/empty | Separate action-local penalties |
| Terminal failure | Negative credit on the last semantic assistant turn |
| `Pmax` and `c-` | Episode penalty cap followed by proportional negative allocation |
| `G_i=0` mask | Correct zero-grounding episodes excluded; no terminal-credit fallback |
| `M` normalization | Loss divided by participating assistant turns, not episodes |
| Full action likelihood | Sum of token log probabilities for complete `think + tool_call` |
| KL | Optional sampled action-level forward-KL against a frozen SFT adapter |

`plan` remains control state and receives no factual positive reward. This is intentional, not a
coverage hole. It can still receive action-local penalties. No public tool or tool argument was
added for reward computation.

## Per-tool coverage on retained SFT turns

| Tool | Turns | Positive | Negative | Zero | Main observed signals |
| --- | ---: | ---: | ---: | ---: | --- |
| `answer_from_context` | 550 | 550 | 0 | 0 | answer format, feedback recovery |
| `condition_filter` | 1,066 | 935 | 37 | 94 | B, E, S+, R potential, local penalties |
| `describe_table` | 754 | 743 | 4 | 7 | B, E, T/C potential |
| `extreme_value_select` | 100 | 90 | 0 | 10 | B, E, S+, R potential |
| `group_aggregate` | 330 | 316 | 0 | 14 | B, E, R potential |
| `inspect_column` | 269 | 122 | 11 | 136 | B, E, feedback response |
| `join_tables` | 314 | 311 | 0 | 3 | B, R potential, feedback response |
| `plan` | 38 | 0 | 1 | 37 | control-only; ignored-feedback penalty |
| `project` | 239 | 239 | 0 | 0 | B, E, R potential |
| `read_subtable` | 409 | 195 | 4 | 210 | B, E, actually exposed target rows |
| `scalar_compute` | 45 | 43 | 0 | 2 | B, E, grounded scalar/result rows |
| `set_op` | 5 | 5 | 0 | 0 | B |
| **Total** | **4,119** | **3,549** | **57** | **513** | |

The complete source histories contain 123 rejected actions: 80 calls without a parsed tool, plus
28 attempted `condition_filter`, 11 `join_tables`, 2 `project`, and 2 `scalar_compute` calls.
Rejected actions are RL transitions but remain excluded from SFT targets.

## Grounding and unsupported guesses

Deterministic feature-computation completeness is 703/703. A separate stricter diagnostic finds
615/703 episodes with no unsupported action constant. The remaining 88 episodes contain 95 events,
primarily entity IDs or codes copied into a later filter without a row/domain observation. Of the
retained SFT turns, 82 receive the unsupported-guess penalty.

The literal-support checker was corrected before this audit. It now recognizes only deterministic
model-visible rewrites such as:

- `1970/1/1` to `1970-01-01`;
- named year/month start and end dates;
- ordered SQL `LIKE` fragments;
- SQL apostrophe escaping;
- small written cardinalities such as “one” and “both”.

Those cases are no longer false penalties. Hidden IDs such as an unobserved `movie_id=559` remain
penalized. SQL expressions are parsed so their real source columns and predicate literals are
audited instead of treating an entire `CASE` expression as a column name.

## SFT reward and one-factor ablations

The full reward gives total retained reward mass 642.8375. Step signs are 3,549 positive, 57
negative, and 513 zero. Because positive credit is renormalized within each episode, ablation
effects should be read primarily through changed steps, sign loss, and reward L1—not raw total
reward mass.

| Removed item | Changed retained turns | Lost positive turns | Retained reward L1 |
| --- | ---: | ---: | ---: |
| Back-slice `B` | 2,980 | 398 | 158.2574 |
| New used evidence `E` | 2,993 | 0 | 152.9985 |
| Search reduction `S+` | 2,741 | 2 | 85.8163 |
| Target potential in correct trajectories | 2,993 | 9 | 65.8449 |
| Answer-format reward | 550 | 544 | 11.0000 |
| Feedback response `F` | 338 | 0 | 7.7914 |
| Unsupported-guess penalty | 82 | 0 | 6.5600 |
| Ignored-feedback penalty | 61 | 0 | 3.0500 |
| Empty-result penalty | 31 | 0 | 3.1000 |
| Repeat penalty | 7 | 0 | 0.4200 |
| Legal no-op penalty | 8 | 0 | 0.2400 |
| Tool-error penalty | 0 | 0 | 0.0000 |
| All local penalties | 173 | 0 | 13.3700 |

Tool-error removal changes zero retained SFT turns by construction, but changes 123 source actions
with reward L1 9.84. Across all source actions, removing all local penalties changes 342 actions
with reward L1 27.93.

Failure progress is not identifiable from a positive-only SFT dataset. As a supplementary check,
all 55 semantic failures from the fixed-200 DeepSeek v4 Flash run were replayed under `bird-set`
(52 wrong answers, one argument-validation terminal, one execution terminal, and one max-step
terminal):

- 55/55 total rewards are negative;
- mean reward is -0.3181, range `[-0.775, -0.23]`;
- removing bounded failure progress changes 87 turns by L1 2.0169, while all 55 episodes remain
  negative;
- removing answer-format reward changes 52 turns by L1 1.04, while all remain negative;
- removing local penalties changes 72 turns by L1 4.7147, while all remain negative.

Thus `eta DeltaPhi` and `lambda_A A` provide the intended limited signals without overpowering
terminal failure.

## Implemented engineering changes

- Separated terminal `{0,1}` control from replay-derived process credit.
- Added a bounded hidden T/C/R target-support module.
- Made perception observations, structured data/value references, and cited terminal relations
  participate in backward slicing without trusting model prose.
- Added exact action-local penalties and terminal-boundary failure allocation.
- Added pure process-objective and sampled-KL helpers plus the online process training path.
- Loaded the SFT adapter a second time as a frozen KL reference when `beta > 0`.
- Added exact SFT cohort construction and retained-turn/source-action audit artifacts.
- Fixed the SFT manifest all-dropped-episode accounting bug.
- Fixed active metric drift: new eval, SFT replay, RL environment, and process reward accept only
  `bird-set`; strict multiset remains low-level historical compatibility code only.
- Fixed RL context drift: process and result-only use rolling4 resident observations, matching SFT.
- Excluded API/transport and generation-OOM events from semantic optimization.

## Remaining experiment boundary

The implementation is ready for an online GPU smoke, but a policy-effect claim is not yet
available. At the status check, both remote GPUs were still occupied by the general and coder 7B
SFT jobs at step 105/516, so no frozen SFT initialization exists and no RL job was launched.

After SFT completes, run a small controlled smoke first, then matched result-only/process and
one-factor reward ablations with identical task IDs, `K`, decoding, action/context budgets,
optimizer, KL, and checkpoint evaluation. A fresh independent grounding-edge precision review is
also still required by the project gate; deterministic coverage alone is not a precision estimate.

## Artifacts

- Audit summary:
  `data/results/rl_process_reward_sft_fixed1000_20260724/summary.json`
- Retained SFT step rewards:
  `data/results/rl_process_reward_sft_fixed1000_20260724/scored_sft_steps.jsonl`
- All source action rewards:
  `data/results/rl_process_reward_sft_fixed1000_20260724/scored_source_actions.jsonl`
- Episode diagnostics:
  `data/results/rl_process_reward_sft_fixed1000_20260724/episode_diagnostics.jsonl`
- Active reward config:
  `src/rl/configs/atomic_process_reward.json`
- Controlled launcher:
  `src/rl/frameworks/accelerate/run_atomic_group_reinforce.sh`

Local verification: harness 81/81, SFT 90/90, RL 48/48, evaluation 27/27, launcher shell syntax,
and Python compilation all pass.
