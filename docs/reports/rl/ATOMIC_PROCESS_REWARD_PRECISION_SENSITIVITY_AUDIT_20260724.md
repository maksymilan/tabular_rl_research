# Atomic process-reward precision and sensitivity audit

Date: 2026-07-24

## Decision

The corrected reward implementation has complete deterministic coverage on the fixed SFT cohort,
and the independent **grounding-edge precision** gate passes. The independent **dependency
completeness** gate does not pass, so `process_reward_ready` remains **false**.

Result-only RL can continue as the deliberately coarse control. Process credit must not be connected
to policy optimization yet: at least 14 replay-correct, denotation-correct trajectories contain a
two-reviewer-confirmed invalid or missing dependency. Thirteen omit or contradict a required
question operation/entity mapping; one has a final relation both reviewers accept but lacks a
required schema observation.

This report supersedes the reward counts in
`ATOMIC_PROCESS_REWARD_SFT_FIXED1000_AUDIT_20260724.md` and the implementation conclusions in
`GROUNDING_EXTERNAL_AUDIT.md`. Those reports remain unchanged as historical audit records.

## Fixed contract

| Variable | Value |
| --- | --- |
| Task cohort | Exact 703 source episodes contributing to the 4,119-turn fixed SFT index |
| Success trajectories | 703 |
| Supplementary failure trajectories | 55 semantic failures from the fixed-200 run |
| Terminal metric | `bird-set` |
| Context | rolling four legal turns with resident observations |
| Reward scenarios | 39 one-factor coefficient settings |
| External first review | DeepSeek v4 Flash |
| External targeted recheck | DeepSeek v4 Pro |
| Gold SQL visibility | Hidden from actor and external reviewers |

All model-facing tools and arguments are unchanged. The fixes are confined to harness provenance,
reward-side feature computation, and independent audit tooling.

## Reward implementation corrections

The audit found and fixed four implementation errors.

1. `B` had included schema observations, so `describe_table` received duplicate `B+E` credit.
   Schema edges now belong only to `E`; row/domain observations remain factual data dependencies.
2. Visible rows emitted by relation-producing tools were not universally eligible as value
   evidence. Every harness-rendered row preview now participates, including computed aggregate and
   scalar aliases.
3. The unsupported-literal parser flattened malformed dictionary-valued conditions into fake
   literals. Dictionary `value_ref` shapes are no longer treated as scalar values.
4. A row/domain observation could receive evidence credit merely for repeating a literal already
   supplied by the question or external knowledge. Such redundant references are now removed before
   slicing and reward allocation.

Deterministic task-side literal support covers exact values plus bounded canonical transforms:
attached IDs, date normalization, SQL escaping/LIKE fragments, written small cardinalities,
schema-declared `_K` scaling, and conventional initialisms.

The external-audit path was also corrected:

- edge IDs include a target hash, so parallel edges between the same steps cannot collapse;
- compact packages include task external knowledge and retain rows matching the audited value;
- subset reviews cannot overwrite their complete package input;
- reviewer prompts explicitly distinguish task literals, visible literal copies, `value_ref`, and
  missing semantic operations;
- prompt version/hash and raw request attempts are retained;
- consensus aggregation checks trajectory uniqueness and exact current edge coverage.

## Deterministic coverage and reward distribution

The corrected 703-episode replay has:

- 703/703 correct `bird-set` denotations;
- 703/703 process updates;
- 703/703 structured and deterministic grounding records;
- 0 replay errors and 0 unresolved retained SFT turns;
- 4,119 retained turns: 3,497 positive, 62 negative, and 560 zero;
- retained reward mass 641.4722, mean 0.155735, median 0.155380, and p90 0.300000;
- positive-credit maximum-share mean 0.3350 and p90 0.4822;
- mean normalized positive-credit entropy 0.9704.

Per-tool retained coverage is:

| Tool | Turns | Positive | Negative | Zero | `B` hits | `E` hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `answer_from_context` | 550 | 550 | 0 | 0 | 0 | 0 |
| `condition_filter` | 1,066 | 999 | 34 | 33 | 999 | 163 |
| `describe_table` | 754 | 743 | 4 | 7 | 0 | 739 |
| `extreme_value_select` | 100 | 97 | 0 | 3 | 96 | 58 |
| `group_aggregate` | 330 | 325 | 0 | 5 | 325 | 182 |
| `inspect_column` | 269 | 7 | 17 | 245 | 7 | 7 |
| `join_tables` | 314 | 312 | 0 | 2 | 311 | 0 |
| `plan` | 38 | 0 | 1 | 37 | 0 | 0 |
| `project` | 239 | 239 | 0 | 0 | 239 | 209 |
| `read_subtable` | 409 | 177 | 6 | 226 | 176 | 176 |
| `scalar_compute` | 45 | 43 | 0 | 2 | 43 | 41 |
| `set_op` | 5 | 5 | 0 | 0 | 5 | 0 |

The sharp reduction in rewarded `inspect_column` turns is intentional: most inspected values were
already present in the task. This removes redundant exploration credit without changing tool
legality or answer execution.

## Coefficient sensitivity

The 39 scenarios use the same 758 episodes and 5,396 actions. Every scenario keeps all 703 success
totals positive and all 55 failure totals negative.

The default configuration remains the recommended setting:

- success total reward: min/mean/max = 0.48 / 0.9902 / 1.02;
- failure total reward: min/mean/max = -0.775 / -0.3093 / -0.23;
- step signs: 4,091 positive, 330 negative, 975 zero;
- failure positive-offset ratio: mean 0.1673, max 0.2333;
- no successful episode hits the penalty cap; two failures do.

The principal ablation conclusions are:

- `w_search_reduction=4x` raises positive max-share p90 to 0.5764 and makes
  `condition_filter` 44.93% of absolute reward mass;
- `w_new_evidence=4x` lowers normalized entropy to 0.9080 and makes `describe_table` dominant;
- `eta_failure_progress=0.2` raises the weakest failure to -0.08 and the maximum positive-offset
  ratio to 0.7333;
- `lambda_answer_format=0.08` moves failure mean to -0.2526 and is too large for a format-only term;
- doubling local penalties caps six successful episodes;
- `penalty_cap=0.4` clips six successes and twelve failures.

Keep the current coefficients. The useful conservative ranges are `eta` 0.025–0.05,
answer-format 0.01–0.02, terminal failure 0.30–0.45, local penalties 1x, and cap 0.6–0.8.

## Independent grounding and dependency audit

Flash reviewed all 703 packages. The targeted set sent to both Flash and Pro contains every initial
fail/ambiguous case, all 25 `common_literal` risks, and 15 deterministic pass controls, for 82
trajectories total. Prompt ambiguities were repaired with small probes before final aggregation:
question-supplied literals require no domain observation, while an exact scalar copied from a
visible output is a valid grounding edge even without `value_ref`.

The scorer correction removed 333 redundant edges from 153 trajectories and added none. Existing
strict reviews were projected only onto this subset of still-current edge IDs; no new edge inherited
an old label.

Final effective component results are:

| Reviewer scope | Valid edges | Invalid edges | Decided precision | Errors |
| --- | ---: | ---: | ---: | ---: |
| Flash, all 703 | 1,803 | 1 | 99.9446% | 0 |
| Pro, targeted 82 | 240 | 1 | 99.5851% | 0 |
| Two-reviewer consensus | 239 valid/valid | 0 invalid/invalid | no confirmed false edge | 0 |

At trajectory level within the 82-case dual review:

- 59 are strong pass;
- 14 are fail/fail;
- 3 are ambiguous/disputed;
- 6 are direct pass/fail disagreements.

Both reviewers mark 13 final dependencies invalid. Both report missing dependencies on nine
trajectories. Among the 15 pass controls, 12 are pass/pass and three are disputed; none is
fail/fail. The control disagreement means the audit should not be interpreted as a calibrated
whole-cohort semantic failure rate.

The confirmed failures fall into stable categories:

- omitted negation/filter/definition constraints (`bird_train_00541`, `02918`, `05527`, `00525`,
  `01039`);
- missing entity-ID or extremum mappings (`05873`, `00739`, `04109`, `03877`, `03663`, `06577`,
  `05875`);
- wrong output entity/grain (`00921`);
- missing schema dependency with an otherwise accepted final relation (`01792`).

These are denotation shortcuts: the current database happens to produce the reference answer, but
the executed relation path does not establish the full question. External labels remain audit
evidence only and are not injected into rewards or SFT targets.

## Gate and next action

The independent edge-precision gate passes after the implementation fixes. Dependency completeness
does not. Therefore:

- result-only RL may be used as the matched coarse control;
- process-RL optimization remains disabled;
- the next research task is a path-independent, harness-verifiable completeness gate for omitted
  question constraints/entity mappings, not a list of reviewer-labeled trajectory IDs and not
  alignment to one privileged gold SQL path.

## Artifacts

- Corrected reward audit:
  `data/results/rl_process_reward_sft_fixed1000_precision_v4_20260724/`
- Corrected coefficient scan:
  `data/results/rl_reward_precision_sensitivity_20260724/sensitivity_reward_v4.json`
- Current 703 audit packages:
  `data/results/rl_reward_precision_sensitivity_20260724/grounding_external/packages703_reward_v4.jsonl`
- Final consensus summary:
  `data/results/rl_reward_precision_sensitivity_20260724/grounding_external/final_reward_v4_consensus_summary.json`
- Confirmed cases:
  `data/results/rl_reward_precision_sensitivity_20260724/grounding_external/final_reward_v4_consensus_fail_cases.jsonl`
- Raw Flash/Pro reviews, prompt-version probes, selection reasons, and request attempts:
  `data/results/rl_reward_precision_sensitivity_20260724/grounding_external/`

Verification: harness 81/81, SFT 90/90, RL 61/61, evaluation 27/27, targeted audit tests 52/52,
Python compilation, and `git diff --check` pass.
