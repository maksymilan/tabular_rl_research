# Project overview

This repository studies reinforcement learning for an LLM agent that solves relational-table tasks
through typed tools. The actor does not emit SQL. It chooses planning, perception, relational, and
terminal actions; the harness translates legal relational actions into composed SQLite operations,
maintains resident state, and returns structured observations or errors.

## Current method boundary

All new supervision is causal. Student and external-teacher trajectories are generated inside a real
model↔harness loop from the visible episode prefix. A completed gold-derived trajectory is never
shown to a model for plan, thought, observation, or perception enrichment.

Gold SQL is hidden harness metadata. On training tasks it may be used for database compatibility and
terminal-denotation scoring; it is not an actor action, prompt field, trajectory template, or process
credit target.

The current production-training path is not the union of every tool experiment. It is the frozen
Qwen3-8B Atomic version26 SFT1 contract whose checkpoint-560 reached 54.63% on matched BIRD-dev
greedy evaluation. SFT expansion, local evaluation, and RL must keep this protocol, prompt,
`think-json-v1` carrier, history renderer, Harness, and scorer fixed. Official DeepSeek requests are
an external-teacher/data-generation transport; their provider envelope is not the local Qwen
inference contract. The later checkpoint-relalg/Atomic-v24-frozen trajectory union, projections,
reasoning repairs, checkpoint/restore, Direct/Hybrid, action-block, SQL, and prompt-trigger
experiments are retained diagnostic branches with zero current SFT/RL admission. Exact identities,
entry points, and the expansion boundary are in `training_mainline.md`.

## Research focus

The central comparison is between coarse terminal result learning and tool-local process credit.
Typed tool execution makes legality, state change, provenance, error timing, and recovery observable
at action granularity. Process credit must be inferred from harness-owned state and dependencies,
not from model-authored explanations or alignment to a single gold path.
