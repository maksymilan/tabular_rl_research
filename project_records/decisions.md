# 决策记录

按时间倒序；详细当前门禁见 [`docs/current/decision_register.md`](../docs/current/decision_register.md)。

## 2026-09-08

- 继续以 Atomic v26、`checkpoint-6380`、four-level reward、SAAM asymmetric-error、reason/tool 各 0.5 为唯一正式 RL 入口；约 500 题 cohort 冻结和 matched BIRD-dev 评测前，不宣称 accuracy promotion。
- 单轮 rollout 预算统一为 `max_new_tokens=4096`；历史运行保留原 manifest 值。
- 先做 a100/table_rl K=8 rollout 的静态语义审计；distance/format 仅作诊断，不能进入 reward、actor 或自动筛题。
- 批准 table_rl 的 `span_balance_alpha=0.25` 独立对照；不改变正式 alpha=0.5，使用相同 checkpoint/cohort/runtime 并完成同机 BIRD-dev1534 评测。

## 2026-09-07

- SMC 概率审计修复后要求每个 4-update run 落盘 120 个题组；新版 cohort 在完整审计前不作 SMC 优化结论。
- Harness-conditioned SMDP/IQL 先做离线可识别性审计；未同时观察到精确 state 的 action 和 outcome 变化前，不训练 actor、不替换 SAAM/GRPO。

## 2026-09-06

- A100 采用一张卡 replicated BF16 trainer 加另一张卡 online vLLM，只使用 GPU 0–3；3090 仅作为隔离的降级入口。
- 新 RL cohort 目标约 500 题，每题 K=8、正确轨迹数 2–6、每 update 30 题，从 `checkpoint-6380` 启动。

## 2026-08-29

- 采用 `saam-asymmetric-error` credit 和 four-level result reward；timeout 按 policy error 处理。历史 binary-only、tool-only、PCGrad 等方案冻结为诊断。
