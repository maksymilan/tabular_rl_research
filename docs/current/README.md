# Current documentation

These files describe the active implementation. Historical plans, compiler experiments, completed
experiment reports, and superseded interfaces live under `docs/archive/` or `archive/` and are not
sources of truth.

- `overview.md`: research objective and non-negotiable method boundary.
- `architecture.md`: active code layout and ownership.
- `tool_protocol.md`: model-visible action and context protocol.
- `relation_derivation.md`: fact-only semantics attached to every derived table handle.
- `execution_contract.md`: harness semantics and error handling.
- `data_generation.md`: causal SFT-1/SFT-2 data construction.
- `sft_pipeline.md`: accepted data and export path.
- `rl_pipeline.md`: result-only control and process-credit mainline.
- `evaluation.md`: evaluation contracts and baseline comparability.
- `research_plan.md`: current staged BIRD research plan.

When documentation conflicts, executable protocol code in `src/sft/protocol.py`, harness behavior,
and explicit evaluation manifests take precedence; update the affected current document in the same
change.
