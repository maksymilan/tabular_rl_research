# Tabular RL Research

这是一个让 LLM 通过 typed tools 操作 relational tables 的 RL 环境。当前唯一训练/评测工具
是 Atomic version26；当前 RL 最终方案是 four-level result reward + SAAM asymmetric-error
credit + reason/tool 加权梯度，在 `a100` 上由两张 A100 做 FSDP trainer、第三张 A100 提供 online vLLM。`table_rl`
和 `NewGNN` 只承担评测、SFT 和其他行为诊断。

## 从这里开始

- [当前最终契约](docs/current/final_project_contract.md)
- [决策登记表（含待确认项）](docs/current/decision_register.md)
- [服务器资源与职责](docs/current/server_resources.md)
- [训练主线](docs/current/training_mainline.md)
- [工具协议](docs/current/tool_protocol.md)
- [因果数据生成](docs/current/data_generation.md)
- [SFT 管线](docs/current/sft_pipeline.md)
- [RL 管线](docs/current/rl_pipeline.md)
- [评测契约](docs/current/evaluation.md)
- [统一评测衔接](docs/current/evaluation_handoff.md)

模型只看真实 causal model↔Harness 前缀；gold SQL 仅作 Harness 隐藏验证，不能用于编译或
事后丰富轨迹。旧代码和已完成实验保留在 [`archive/`](archive/README.md) 与
[`docs/archive/`](docs/archive/README.md) 供审计；旧代码迁移清单见
[`archive/code/legacy_migration.md`](archive/code/legacy_migration.md)，不是新实验入口。
