# Current documentation

These files describe the active implementation. Historical plans, compiler experiments, completed
experiment reports, and superseded interfaces live under `docs/archive/` or `archive/` and are not
sources of truth.

- `overview.md`: research objective and non-negotiable method boundary.
- `architecture.md`: active code layout and ownership.
- `tool_schemes.md`: the three exclusive model action schemes and their train/eval boundaries.
- `tool_protocol.md`: model-visible action and context protocol, including the version39 default
  and isolated version40/version41 no-plan/`inspect_rows` diagnostics.
- `atomic_tool_interface_zh.md`: standalone Chinese reference for the atomic version39 public
  tools, strict argument shapes, error contract, and full-context/F profile differences.
- `atomic_version40_prompt_diagnostic_zh.md`: isolated no-plan/`inspect_rows` external-teacher
  prompt diagnostic, prompt-size audit, and failed Gate50 result.
- `atomic_version41_output_corrections_zh.md`: prompt-only output-contract consolidation and
  Gate50-derived error-correction examples over the frozen version40 behavior.
- `atomic_version42_terminal_columns_zh.md`: diagnostic terminal carrier requiring explicit
  grounded evidence columns with deterministic harness projection.
- `relation_derivation.md`: fact-only semantics attached to every derived table handle.
- `execution_contract.md`: harness semantics and error handling.
- `data_generation.md`: causal SFT-1/SFT-2 data construction.
- `sft_pipeline.md`: accepted data and export path.
- `rl_pipeline.md`: result-only control and process-credit mainline.
- `evaluation.md`: evaluation contracts and baseline comparability.

When documentation conflicts, the scheme registry in `src/sft/tool_schemes.py`, the selected
scheme's executable protocol, harness behavior, and explicit evaluation manifests take precedence;
update the affected current document in the same change.
