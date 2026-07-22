# Tabular RL Research

Reinforcement-learning environment for an LLM agent that reasons over relational tables through
typed tools. The model plans, inspects, transforms, and answers; the harness executes relational
actions over SQLite and supplies structured state and feedback.

The active method uses causal model↔harness rollouts. Gold SQL is hidden verification metadata and
is not compiled into trajectories for post-hoc enrichment.

## Start here

- [Project overview](docs/current/overview.md)
- [Architecture](docs/current/architecture.md)
- [Tool protocol](docs/current/tool_protocol.md)
- [Causal data generation](docs/current/data_generation.md)
- [SFT pipeline](docs/current/sft_pipeline.md)
- [RL pipeline](docs/current/rl_pipeline.md)
- [Evaluation contract](docs/current/evaluation.md)

Historical code and completed experiments are retained under [`archive/`](archive/README.md);
superseded documents and the full project chronology are under
[`docs/archive/`](docs/archive/README.md). Active code must not import archived modules.
