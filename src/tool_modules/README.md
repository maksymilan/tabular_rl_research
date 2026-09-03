# Active tool registry

The active source tree contains only the Atomic version26 registry used by the frozen SFT and RL
mainline. The registry is intentionally limited to `atomic`; it does not expose builders,
constants, or parser aliases for retired tool schemes.

Historical scheme implementations and their thin import aliases are preserved under
`archive/code/legacy_tool_modules/` and `archive/code/legacy_compatibility/` for audit only. They
are not on the active import path and must not be used to create new data, evaluations, or RL
runs.
