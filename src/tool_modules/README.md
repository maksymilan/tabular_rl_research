# Active tool registry

The active source tree contains only the Atomic version26 registry used by the frozen SFT and RL
mainline. The registry is fixed runtime infrastructure, not an experiment-scenario or data
script package. It is intentionally limited to `atomic`; it does not expose builders, constants,
or parser aliases for retired tool schemes.

Historical scheme implementations and their thin import aliases are preserved under
`archive/code/legacy_tool_modules/` and `archive/code/legacy_compatibility/` for audit only. They
are not on the active import path and must not be used to create new data, evaluations, or RL
runs. Current scenario adapters live under `src/rl/scenarios`; historical root supervisors were
moved to `archive/experiments` so no duplicate script module remains at the repository root.
