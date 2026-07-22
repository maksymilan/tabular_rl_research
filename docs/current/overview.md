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

## Research focus

The central comparison is between coarse terminal result learning and tool-local process credit.
Typed tool execution makes legality, state change, provenance, error timing, and recovery observable
at action granularity. Process credit must be inferred from harness-owned state and dependencies,
not from model-authored explanations or alignment to a single gold path.
