# 实验记录

按时间倒序；“诊断/中止”不代表正式方案已提升。

## 2026-09-08

- **K=8 rollout 静态语义审计**：覆盖 a100/table_rl 既有语料；见 `docs/reports/rl/ROLLOUT_CORPUS_SEMANTIC_AUDIT_20260908_ZH.md`。仅为诊断证据，未改变 reward、cohort 或正式评测。
- **span alpha=0.25 对照**：table_rl，`checkpoint-6380`，新版 60 题，K=8、4 updates、4-bit frozen base + FP32 LoRA/AdamW；训练后 matched BIRD-dev1534。与 alpha=0.5 独立比较，不能替代正式 arm。见 `docs/reports/rl/SAAM_ALPHA025_TABLE_RL_20260908.md`。
- **A100 主线**：update-40 checkpoint 可审计；step49 因 OOM 退出。旧 700 题/14 题每 update，仅作历史诊断。

## 2026-09-07

- **SAAM 新版 60 题训练**：从 `checkpoint-6380` 启动，four-level + span-balanced，训练完成；NewGNN matched 评测仍是衔接证据，尚未形成正式全量准入结论。
- **SMC 审计修复**：原始 cohort 完成 120 题组/960 候选的完整字段、连接和选择一致性检查；新版 cohort 仍需完整概率审计。
- **SMDP/IQL 离线诊断**：已实现 transition、Q/V 与可识别性审计入口；`diagnostic_only`，不更新 actor。

## 当前准入状态

正式 RL 仍待约 500 题 cohort 冻结、immutable manifest、fresh replay/audit 和独立 matched BIRD-dev1534 证据。历史报告和旧实验只能通过 `docs/reports/`、`docs/archive/` 审计，不能自动恢复为当前入口。
