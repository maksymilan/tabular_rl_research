# Code and experiment archive

Archived code is retained for reproducibility and audit, not imported by the active system.

- `code/gold_sql_compiler/`: SQL→Plan→complete trajectory compiler, verifier, emitter, samples,
  and tests.
- `code/trajectory_enrichment/`: post-hoc plan/thought/observation enrichment utilities and tests.
- `code/experimental_backends/verl/`: historical unsupported Verl integration.
- `experiments/`: completed SFT/evaluation launchers, configs, and launchd definitions.
- `code/legacy_tool_modules/`: retired action-block, checkpoint-relalg, direct/iterative SQL,
  native-tool-bundle, relational-program, and shared SQL runners.
- `code/legacy_compatibility/`: retired thin aliases and branch-specific runners moved out of
  `src/` so they cannot be imported by the active mainline.
- `code/legacy_migration.md`: the completed source-to-archive map and replay boundary.

Do not repair archived code merely to satisfy active tests. If a historical experiment must be
reproduced, restore its frozen environment and paths explicitly.
