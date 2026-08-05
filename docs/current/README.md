# Current documentation

These files describe the active implementation. Historical plans, compiler experiments, completed
experiment reports, and superseded interfaces live under `docs/archive/` or `archive/` and are not
sources of truth.

- `overview.md`: research objective and non-negotiable method boundary.
- `architecture.md`: active code layout and ownership.
- `provider_api.md`: official DeepSeek endpoint and credential contract, AimixHub/AIHubMix
  deprecation, Chat Completions versus FIM boundary, and migration verification.
- `tool_schemes.md`: the four exclusive model action schemes and their train/eval boundaries.
- `direct_sql_search_tool_scheme_zh.md`: two-tool out-of-band value search + direct-SQL feedback
  diagnostic, frozen v1 Gate16, optimized v2 prompt/feedback/context protocol, completed paired
  Holdout Gate15, replay audit, and non-promotion boundary.
- `tool_protocol.md`: model-visible action and context protocol, including the version39 default,
  isolated version40-version43 diagnostics, version44-version45 search diagnostics, and the
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
- `tool_design_and_motivation_zh.md`: complete Chinese report connecting the current atomic tool
  surface, ownership/grounding boundaries, prompt and context decisions, version history,
  external-teacher evidence, limitations, and next optimization priorities.
- `relation_derivation.md`: fact-only semantics attached to every derived table handle.
- `execution_contract.md`: harness semantics and error handling.
- `data_generation.md`: causal SFT-1/SFT-2 data construction.
- `sft_pipeline.md`: accepted data and export path.
- `rl_pipeline.md`: result-only control and process-credit mainline.
- `evaluation.md`: evaluation contracts and baseline comparability.

When documentation conflicts, the scheme registry in `src/sft/tool_schemes.py`, the selected
scheme's executable protocol, harness behavior, and explicit evaluation manifests take precedence;
update the affected current document in the same change.
