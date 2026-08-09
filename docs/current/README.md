# Current documentation

These files describe the active implementation. Historical plans, compiler experiments, completed
experiment reports, and superseded interfaces live under `docs/archive/` or `archive/` and are not
sources of truth.

- `overview.md`: research objective and non-negotiable method boundary.
- `architecture.md`: active code layout and ownership.
- `checkpoint_relalg_v1_zh.md`: `checkpoint-relalg-v1` 的前向协议、三种 mode、短提示词、
  关系工件/检查点状态，以及诊断准入边界。完整设计规格属于 Harness 与实现侧输入，
  不是逐轮发送给模型的 prompt。
- `provider_api.md`: official DeepSeek endpoint and credential contract, AimixHub/AIHubMix
  deprecation, Chat Completions versus FIM boundary, and migration verification.
- `deepseek_native_tool_calls_zh.md`: official DeepSeek native function-call adapter,
  thinking-mode history contract, completed version50/version51 results, and the diagnostic
  version52 compact prompt, version53 reviewed prompt, and frozen version54 no-plan successor.
- `tool_schemes.md`: the seven exclusive model action schemes and their train/eval boundaries.
- `direct_sql_search_tool_scheme_zh.md`: two-tool out-of-band value search + direct-SQL feedback
  diagnostic, frozen v1 Gate16, optimized v2 prompt/feedback/context protocol, completed paired
  Holdout Gate15, replay audit, and non-promotion boundary.
- `iterative_sql_tool_scheme_zh.md`: execute-SQL/submit-SQL diagnostic with prior-execution
  grounding, recoverable final-call errors, strict read-only safety, the completed v4 development
  Gate15, v5 prompt review, independent Prefix20, rejected Prefix50 expansion gate, and the
  v6 relational-output-shape rule, target Gate20, and completed v6-only baseline300 full run.
- `../reports/evaluation/BIRD_ITERATIVE_SQL_V4_OPTIMIZATION_GATE15_20260805_ZH.md`: matched Flash
  Gate15 baseline/optimization results, paired comparisons, fresh replay audit, remaining failure
  taxonomy, and the independent-holdout requirement.
- `../reports/evaluation/BIRD_ITERATIVE_SQL_PROMPT_REVIEW_V5_20260805_ZH.md`: itemized disposition
  of the external prompt review, v5 implementation boundary, rejected query-id schema change, and
  independent-holdout requirement.
- `../reports/evaluation/BIRD_ITERATIVE_SQL_V5_INDEPENDENT_PREFIX20_GATE_20260805_ZH.md`: paired
  frozen v4/v5 Flash result on the active baseline300 prefix, replay audit, efficiency statistics,
  gains, remaining contract-conflict failures, and diagnostic-only decision.
- `../reports/evaluation/BIRD_ITERATIVE_SQL_V5_PREFIX50_EXPANSION_20260805_ZH.md`: completed
  baseline300 Prefix50 expansion showing a non-significant accuracy gain but failed legal/error/
  token stability, multi-field representation regressions, and the stop decision.
- `../reports/evaluation/BIRD_ITERATIVE_SQL_V6_RELATIONAL_OUTPUT_SHAPE_RULE_20260805_ZH.md`: minimal
  prompt-only rule making scalar/table and multi-field column representation explicit, with its
  original disjoint-test requirement and target-coverage boundary.
- `../reports/evaluation/BIRD_ITERATIVE_SQL_V6_DISJOINT_GATE20_20260805_ZH.md`: completed frozen
  tasks 51–70 v5/v6 Flash comparison; tied accuracy/legal behavior, slightly higher v6 token cost,
  replay/structure audits, and the finding that the slice did not directly cover the target rule.
- `../reports/evaluation/BIRD_ITERATIVE_SQL_V6_TARGET_GATE20_PLAN_20260805_ZH.md`: gold-invariant,
  zero-overlap output-shape Target Gate20 cohort and preregistered category/reliability/cost gates.
- `../reports/evaluation/BIRD_ITERATIVE_SQL_V6_TARGET_GATE20_RESULT_20260805_ZH.md`: v6 15/20 versus
  v5 10/20 with five gains and no regressions, target-category recovery, improved errors/actions/
  tokens, complete audits, and the invalid combined-control caveat.
- `../reports/evaluation/BIRD_ITERATIVE_SQL_V6_FLASH_FULL300_20260805_ZH.md`: v6-only baseline300
  result (215/300, 298/300 legal), untouched tasks 71–300 generalization read (168/230), complete
  replay/structure audit, error/token long-tail analysis, and the no-paired-v5 interpretation
  boundary.
- `tool_protocol.md`: the forward `checkpoint-relalg-v1` action/context protocol plus the retained
  atomic protocol index, including isolated version40-version43 diagnostics,
  version44-version45 search diagnostics, and the
  rejected version46-version49 context-management ablations, including the dependency-aware
  active/archive renderer.
- `atomic_tool_interface_zh.md`: standalone Chinese reference for the atomic version39 public
  tools, strict argument shapes, error contract, and full-context/F profile differences.
- `atomic_version40_prompt_diagnostic_zh.md`: isolated no-plan/`inspect_rows` external-teacher
  prompt diagnostic, prompt-size audit, and failed Gate50 result.
- `atomic_version41_output_corrections_zh.md`: prompt-only output-contract consolidation and
  Gate50-derived error-correction examples over the frozen version40 behavior.
- `atomic_version42_terminal_columns_zh.md`: diagnostic terminal carrier requiring explicit
  grounded evidence columns with deterministic harness projection.
- `atomic_version43_unique_bare_terminal_columns_zh.md`: minimal terminal-column resolver
  diagnostic allowing only one unambiguous dotted-column suffix match.
- `atomic_version44_checkpoint_zh.md`: frozen no-plan/`inspect_rows`/`search_values` checkpoint,
  inspect-only BIRD column semantics, and external-teacher interface smoke results.
- `atomic_version45_bounded_search_zh.md`: version44-compatible bounded SQL candidate recall,
  exact-first behavior, latency benchmark, and completed non-promoted external-teacher Gate50.
- `../reports/evaluation/BIRD_ATOMIC_CONTEXT_HANDLE_CARD_ABLATION_GATE16_20260731_ZH.md`:
  paired handle-card, archived-row/latest-full-observation, interpret-before-act, and
  dependency-aware active/archive context gate.
- `../reports/evaluation/BIRD_ATOMIC_VERSION49_CONTEXT_STABILITY_K3_GATE8_20260731_ZH.md`:
  three-run paired stability follow-up showing unchanged accuracy but substantially higher
  actions, reads, tokens, and max-step failures for version49.
- `../reports/evaluation/BIRD_VERSION24_BOUNDED_SEARCH_SINGLE_VARIABLE_GATE50_20260730_ZH.md`:
  historical version24 reproduction and strict search_values-only controlled Gate50.
- `../reports/evaluation/BIRD_ATOMIC_VERSION24_FLASH20_PRO200_CAPABILITY_CEILING_20260803_ZH.md`:
  current Flash20 reproduction and frozen Flash200/Pro200 capability-ceiling comparison, including
  paired accuracy, reliability, tool-use, token, prefix-recovery, and no-leak/replay audits.
- `../reports/evaluation/BIRD_ATOMIC_VERSION50_NATIVE_TOOL_CALLS_FIXED200_20260805_ZH.md`:
  completed version39-semantics/native-function-call Flash fixed-200 diagnostic: 133/200 versus
  historical version24 145/200, with replay/no-leak audits and rejection decision.
- `../reports/evaluation/BIRD_VERSION51_NATIVE_TOOL_BUNDLE_GATE32_20260805_ZH.md`:
  passed carrier-failure/control Gate32: 22/32 versus version50 16/32, 32/32 legal, six paired
  gains with no regression, and full replay/structure/provider-history/no-leak audits.
- `../reports/evaluation/BIRD_VERSION51_NATIVE_TOOL_BUNDLE_FIXED200_20260805_ZH.md`:
  completed fixed-200 behavior gate: 147/200 and 200/200 legal versus version50 133/200 and
  183/200, with significant paired recovery, but statistically tied with version24's 145/200;
  replay and full structural/native-bundle audits pass while SFT/RL admission remains closed.
- `../reports/evaluation/BIRD_VERSION53_NATIVE_BUNDLE_PROMPT_REVIEW_STATIC_AUDIT_20260806_ZH.md`:
  itemized external-review disposition, role-separated v53 implementation, static prompt/schema
  size audit, unchanged execution boundary, and no-live-result/non-admission decision.
- `../reports/evaluation/BIRD_VERSION54_NO_PLAN_TEACHER1500_PREFIX200_PLAN_20260806_ZH.md`:
  frozen v54-only teacher1500 Prefix200 absolute acceptance gate, privacy boundary, and conditional
  remaining-1300 expansion rule; no external request or result exists before authorization.
- `../reports/evaluation/CHECKPOINT_RELALG_V1_CAUSAL_SMOKE_20260809_ZH.md`: actual official
  DeepSeek checkpoint-relalg request, authenticated model preflight, explicit HTTP 402 insufficient-
  balance transport block, passing failure-artifact structure/fresh-replay audits, and the resulting
  no-semantic-result/non-admission boundary.
- `tool_design_and_motivation_zh.md`: complete Chinese report connecting the current atomic tool
  surface, ownership/grounding boundaries, prompt and context decisions, version history,
  external-teacher evidence, limitations, and next optimization priorities.
- `relation_derivation.md`: fact-only semantics attached to every derived table handle.
- `execution_contract.md`: harness semantics and error handling.
- `data_generation.md`: causal SFT-1/SFT-2 data construction, including the pre-rollout hidden-gold
  nonempty-task gate and the frozen teacher1500 v2 source cohort.
- `../reports/sft/BIRD_ATOMIC_TEACHER1500_NONEMPTY_TASK_GATE_20260806_ZH.md`: full 6,601-task
  read-only result audit, 6,599/5,915 filtered pools, privacy boundary, and identity-preserving
  teacher1500 v2 certification.
- `sft_pipeline.md`: accepted data and export path.
- `rl_pipeline.md`: result-only control and process-credit mainline.
- `evaluation.md`: evaluation contracts and baseline comparability.
- `baseline_datasets.md`: active BIRD-train baseline300, deprecated fixed-200 cohort, distribution
  audit, immutable versioning rules, and the mandatory baseline-data change log.

When documentation conflicts, the scheme registry in `src/tool_modules/registry.py`, the selected
scheme's executable protocol, harness behavior, and explicit evaluation manifests take precedence;
update the affected current document in the same change.

The forward implementation line is now `checkpoint-relalg-v1` under scheme
`checkpoint-relalg`, with an explicit `mode=direct|atomic|hybrid`. Every provider assistant turn
authors exactly one official DeepSeek native tool call. All modes share the same relation-artifact,
environment-state, checkpoint/restore, and exact-artifact terminal contract. This is the required
starting point for new tool and experiment development, but it remains `diagnostic-only` until
fresh replay, structure, provider-history, no-leak, behavior, and explicit SFT/RL admission gates
pass. “Forward mainline” identifies where new development starts; it does not imply a training or
accuracy promotion.

The original `atomic` line remains available for already-running RL work, frozen controls, and
exact reproduction; version54 / `native-tool-bundle` remains a diagnostic control/reproduction
line and does not gain RL admission. Do not delete them, silently migrate their
checkpoints, mix their trajectories with `checkpoint-relalg`, or start a new protocol experiment
from them. Version51-version54 remain ineligible for the current SFT/RL exporters. The first actual
official checkpoint-relalg request was transport-blocked by insufficient balance before any tool
call; it supplies no live semantic pilot result or SFT/RL admission evidence.
