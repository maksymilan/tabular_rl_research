# BIRD version20 carrier/schema fixed-200 validation

Date: 2026-07-23

## Conclusion

Version20 materially improves interface reliability but does **not** pass the requested tool-usability
gate:

- **138/200 = 69.0% normalized strict-multiset**
- **199/200 = 99.5% legal termination**
- **24 process errors over 200 episodes**
- required gate: **150/200 = 75.0%**

Relative to the protocol-matched-cohort version19 canonical rerun, version20 gains 6 correct tasks,
9 legal terminations, and removes every observed provider/carrier protocol error. The paired
accuracy change is positive but not statistically decisive (11 gains, 5 regressions; exact
two-sided McNemar/binomial p = 0.210). Version20 is therefore the stronger interface contract, but
the current actor-plus-tool system is not yet validated for SFT construction under the 75% rule.

## Controlled setup

All runs use the same frozen 200-task file:

`data/eval_inputs/bird_train_tool_interface_validation200_version4.jsonl`

Version20 fixed-200 settings:

- model: `deepseek-v4-flash`
- decoding: temperature 0, thinking enabled, reasoning effort `high`
- context: rolling legal history, 4 successful assistant/tool pairs
- plan policy: optional
- maximum semantic actions: 30
- initial maximum completion: 2048 tokens
- denotation: normalized `strict-multiset`
- provider carrier: JSON Output
- policy prompt: canonical, without the global relational-invariants suffix
- strict parser: no response or argument repair
- protocol hash: `8252f8c6e50af961`

Artifact:

`data/trajectories/tool_usability_20260723/version20_final_fixed200_json.*`

The version19 P0/P1 comparison runs used the same task selection, model, decoding, rolling context,
plan policy, action budget, and denotation metric, but used the tool-call provider carrier. Thus the
full comparison measures the complete version20 carrier-plus-local-schema package; it is not a
single-variable carrier causal estimate.

## What changed in version20

1. DeepSeek-facing examples no longer contain copy-prone literal API field labels. They show only
   the final action shape.
2. The carrier contract requires every tool parameter to remain inside `arguments`, and asks the
   model to reserve completion budget for the final action.
3. Short local argument invariants sit beside the affected tools: join key direction and namespace,
   producing-step scalar references, aggregation-level `where`, and closed project/read argument
   sets.
4. A response ending with `finish_reason=length` is not immediately charged as a semantic protocol
   action. The client retries the same turn with a bounded doubled completion budget, records the
   truncated response as `provider_retry_events`, and still requires the fresh response to pass the
   strict parser. At most 8192 completion tokens are requested. Exhausted retries remain visible
   failures.

No model action is synthesized, repaired, or silently admitted. Gold SQL remains hidden from the
actor and is used only by the verifier/audit.

## Small gate

Before the full run, the same first-30 cohort was checked against version19:

| Metric | version19 P0 | version20 pre-final fixed30 |
| --- | ---: | ---: |
| Strict-multiset correct | 20/30 | 20/30 |
| Legal termination | 29/30 | 30/30 |
| Process errors | 8 | 2 |
| Tasks with process errors | 4 | 2 |

The pathological `bird_train_00593` previously produced three empty visible responses and five
total process errors under version19. After bounded completion retry it completed legally with no
tool error; it still returned only the 11-day row while gold contains 11- and 18-day rows. That
remaining mismatch is a model grain/multiplicity decision, not a carrier failure.

After adding the final “all parameters inside arguments” sentence, a targeted
`bird_train_03275` run under the exact full-run protocol hash completed 11 steps, with zero process
errors, and was correct.

## Fixed-200 results

| Metric | version19 P0 canonical | version19 P1 global relational prompt | version20 canonical |
| --- | ---: | ---: | ---: |
| Strict-multiset correct | 132/200 (66.0%) | 137/200 (68.5%) | **138/200 (69.0%)** |
| Legal termination | 190/200 (95.0%) | 188/200 (94.0%) | **199/200 (99.5%)** |
| Mean semantic actions | 7.825 | 7.935 | **7.460** |
| Process errors | 91 | 116 | **24** |
| Tasks affected by process errors | 56 | 69 | **21** |
| Protocol/carrier errors | 71 | 76 | **0** |
| Argument-validation errors | 9 | 20 | 10 |
| Execution errors | 11 | 20 | 14 |
| API completion-length retries | not separated | not separated | 2 |
| Total API requests | 1580 | 1597 | **1510** |
| Total tokens | 7,959,288 | 8,426,301 | 7,961,828 |

The version20 system prompt is 16,145 characters versus 15,019 for version19 P0 (+7.5%), but total
token use is nearly unchanged (+2,540, +0.03%) because shorter/cleaner trajectories reduce API
requests. Relative to P1, version20 uses 5.5% fewer total tokens.

Version20 versus version19 P0:

- both correct: 127
- both wrong: 57
- version20-only correct: 11
- version19-only correct: 5
- legal gains: 9
- legal regressions: 0

The legal gain is one-sided in the observed cohort (exact two-sided p = 0.0039). Accuracy is not:
11 versus 5 discordant wins gives p = 0.210.

Version20 versus P1 has 8 gains and 7 regressions (exact two-sided p = 1.0). The global relational
suffix therefore remains an ablation rather than part of the canonical prompt.

## Remaining error boundary

Of the 62 version20 failures:

- 1 reaches `max_steps`;
- 61 terminate with a legal but wrong answer;
- only 9 failed episodes contain any process error;
- **53 failed episodes contain no protocol, validation, or execution error**.

Even the unrealistic assumption that fixing every recorded process error would turn all nine
affected failures into successes yields only 147/200. Reaching 150 therefore cannot be justified as
an error-handler-only task.

The 24 remaining process errors are:

| Error | Count | Main forms |
| --- | ---: | --- |
| Argument validation | 10 | join left key not exact (3), join `type` at call level (2), empty aggregate `where` (3), scalar step-id shape (1), aggregate passthrough shape (1) |
| Execution | 14 | optional plan missing `op` (4), table result used as scalar threshold (4), join handle prefixed onto an already logical column (2), scalar dotted-column lookup (2), ambiguous aggregate column (1), malformed predicate value (1) |

Plan remains unchanged by decision. The most defensible next tool changes are deterministic,
semantics-preserving name/shape resolution: safe unique-bare or redundant-handle-prefix resolution
for join left keys, consistent unique-suffix resolution for named scalar columns, and a single
join-type default without removing per-edge overrides. Empty aggregate `where` and scalar
passthrough coercions require a separate decision because they relax the strict public grammar.

The error-free failures remain dominated by policy choices already seen in earlier audits:
operator ordering, answer entity/slot selection, row grain and multiplicity, and choosing an
aggregation or relation that is legal but semantically wrong. These need targeted prompt/SFT/RL
evidence rather than a harness that guesses gold intent.

## Decision

- Adopt version20 over version19 as the current carrier/interface baseline.
- Keep the global relational-invariants suffix non-canonical.
- Do not construct new SFT data yet: **138/200 fails the 150/200 gate**.
- Preserve optional planning; the prior forced-plan diagnostic showed no benefit.
- Continue with a separately versioned, small-first tool-resolution pilot. Any later fixed-200 run
  must keep the same cohort, rolling history, plan policy, model, decoding, and strict-multiset
  verifier.
