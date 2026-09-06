# 当前文档入口

当前项目已收敛到一套工具和一条 RL 路线。先读
[`final_project_contract.md`](final_project_contract.md)，再按任务读取下列文档。历史方案、
失败诊断和旧结果不再列为当前分支，统一保留在 `docs/archive/`、`docs/reports/` 和
`archive/` 供审计。

旧代码的归档迁移清单见 `archive/code/legacy_migration.md`。`src/` 不保留旧路线的
compatibility stub；需要重放历史结果时必须显式使用 archive 中的完整源文件和环境。

## 必读

- [`final_project_contract.md`](final_project_contract.md)：工具、SFT 锚点、RL 候选、状态和停止线。
- [`decision_register.md`](decision_register.md)：已确认事项与需要用户拍板的未决项。
- [`server_resources.md`](server_resources.md)：`a100`、`table_rl`、`NewGNN` 的硬件和职责。
- [`rl_pipeline.md`](rl_pipeline.md)：four-level reward、SAAM credit、A100 单卡 replicated RL 运行契约。
- [`kl_ablation_protocol.md`](kl_ablation_protocol.md)：后续 KL 增益验证的匹配和审计约束。

## 实现参考

- [`architecture.md`](architecture.md)：当前代码分层和唯一入口。
- [`workflows.md`](workflows.md)：四类操作的统一工作流参数和计划入口。
- [`tool_protocol.md`](tool_protocol.md)：Atomic version26 public action/context contract。
- [`sft_pipeline.md`](sft_pipeline.md)：因果 SFT 数据和冻结 SFT1 训练身份。
- [`data_generation.md`](data_generation.md)：teacher↔Harness 数据生成、replay 和 no-leak 门禁。
- [`evaluation.md`](evaluation.md)：BIRD-dev matched evaluator 和结果准入。
- [`baseline_datasets.md`](baseline_datasets.md)：当前数据集 identity 与固定 cohort 规则。
- [`provider_api.md`](provider_api.md)：官方 DeepSeek endpoint 和 credential 规则。
- [`relation_derivation.md`](relation_derivation.md)：Harness 派生关系事实和 lineage 契约。

## 当前默认入口

```text
工具：Atomic version26
RL 配置：src/rl/configs/experiments/qwen3_8b_atomic_v26_saam_screened500_single_gpu.yaml
RL launcher：src/rl/scenarios/rl_main/run_qwen3_8b_atomic_v26_saam_fourlevel_spanbalanced_700_single_gpu_a100.sh
环境：src/rl/runtime/tool_environment_v26.py
评测：reproductions/trust_sql/qwen3_8b_atomic_sft1/
```

版本、prompt、carrier、Harness、scorer、checkpoint、cohort 和服务器角色必须在每次
run manifest 中明确记录。没有完整 artifact、fresh replay、结构/no-leak 审计和 matched
评测，不能把候选结果写成最终结论。
