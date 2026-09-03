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
greedy evaluation as a small-sample RL-feasibility model, followed by the A100 two-trainer-card
FSDP RL route from the fixed cumulative-SFT checkpoint-6380. The current final RL scheme is
four-level result reward with SAAM asymmetric-error credit and span-balanced full-response loss;
its formal 700-record gate is still running. The live A100 manifest starts from cumulative SFT
checkpoint-6380, which must not be silently conflated with checkpoint-560; see
`decision_register.md`. All stages keep protocol, prompt, `think-json-v1` carrier, history
renderer, Harness, and scorer fixed. Official DeepSeek requests are an external-teacher/data-
generation transport; their provider envelope is not the local Qwen inference contract. Later
checkpoint-relalg/Atomic-v24-frozen, projection/reasoning repairs, checkpoint/restore, Direct/
Hybrid, action-block, SQL, and prompt-trigger experiments are retained diagnostic artifacts with
zero current SFT/RL admission.

## Research focus

The central comparison is between coarse terminal result learning and tool-local process credit.
Typed tool execution makes legality, state change, provenance, error timing, and recovery observable
at action granularity. Process credit must be inferred from harness-owned state and dependencies,
not from model-authored explanations or alignment to a single gold path.

The current server split is `a100` for the RL main experiment (two FSDP trainer cards plus one
online-vLLM card), with `table_rl` and `NewGNN` reserved for evaluation, SFT, and other behavior
work. See `server_resources.md` and `decision_register.md` for the live status and unresolved
checkpoint/promotion decisions.
