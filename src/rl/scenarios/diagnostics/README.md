# Diagnostic scenarios

One-off audits, cohort screens, replay comparisons, and training reports are
organized here by experiment scenario. They may depend on a frozen historical
artifact, but must import reusable I/O, ranking, reward, and environment
functions from stable modules under `rl.diagnostics`, `rl.runtime`, or
`rl.objectives`. New current experiments should add one clearly named scenario
module instead of extending the shared runtime.
