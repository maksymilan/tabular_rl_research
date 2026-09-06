# 当前决策登记表

更新时间：2026-09-06。这里把已确认事项和仍需用户拍板的事项分开，避免文档再出现多套
“当前方案”。用户回复后，只修改本表和 `final_project_contract.md`，再同步对应 manifest。

## 已确认

| 事项 | 当前答案 | 证据/落点 |
|---|---|---|
| 工具版本 | Atomic version26 | `docs/current/final_project_contract.md` |
| A100 训练方式 | 单卡 replicated trainer + 独立单卡 online vLLM | update-40 run manifest；world size=1 |
| 服务器职责 | A100 优先做 RL 主实验；A100 不可用时，table_rl/NewGNN 可按降级拓扑做 RL；三机均可评测 | 用户于 2026-09-06 确认；`server_resources.md` |
| A100 serving | 另用一张 A100 跑 online vLLM | 当前 live gate 进程和 formal launcher |
| RL credit | `saam-asymmetric-error` | `src/rl/frameworks/trl/state_action_ambiguity.py` |
| RL reward | four-level result reward | `src/rl/runtime/terminal_reward.py` |
| RL 梯度更新 | reason/tool 各 0.5 的加权 full-response 梯度 | `span_balance_alpha=0.5` |
| timeout | 视为 policy error，timeout action 使用局部负向惩罚 | SAAM asymmetric-error |
| RL 起点 | cumulative SFT `checkpoint-6380` | A100 formal/live launcher |
| RL 数据与批次 | 目标约500题；每题K=8且正确数2–6；30题/update；从checkpoint-6380启动 | 2026-09-06跨服务器筛选盘点 |
| GPU 选择 | A100 仅允许 GPU 0–3；动态选择一张 trainer 与一张不同的 vLLM 卡 | 用户确认；launcher 已加入 allowlist |
| 旧代码 | 全部移入 `archive/`，暂不删除；`src/` 不保留 compatibility stub | `archive/code/legacy_migration.md` |
| KL 验证 | 必须通过独立匹配对照验证是否有增益；当前正式 arm 仍为 `kl_beta=0` | `docs/current/rl_pipeline.md` |
| A100 资源约束 | 单卡拓扑；trainer/vLLM 必须不同卡；只允许 GPU 0–3 | 用户于 2026-09-06 确认 |
| 单卡资源入口 | 作为当前主线；默认 BF16 冻结基座、FP32 AdamW、独立 output root | `qwen3_8b_atomic_v26_saam_screened500_single_gpu.yaml` |
| 3090 降级入口 | 两张独立 3090：单卡 replicated trainer + 单卡 online vLLM；从 1 row / 8,192 tokens 起步 | 用户于 2026-09-06 确认；需独立 launcher、manifest/output root |
| Qwen2.5-7B SFT 对照 | 在 NewGNN 用四卡训练 `Qwen/Qwen2.5-7B-Instruct`，复用 Qwen3 SFT1 的同一 4,471-record model-visible 数据；只作为模型族对照，不改变 Qwen3/RL 主线 | 用户于 2026-09-04 明确要求；run `qwen25_7b_atomic_v26_sft1_same4471_4gpu_full_20260904_203529` 已以 exit 0 完成，checkpoint-560 已保存 |

## 待用户确认

1. **KL 对照系数/调度**：是否采用固定 beta（例如 `0.01`）或预注册的 beta sweep，待
   对照启动前登记；不得在看到结果后再挑选系数。

2. **新 cohort 最终规模**：当前跨服务器去重后确认 430 道满足 2–6 的题目；补充筛选完成后再冻结约500题 manifest。

## 未决前的执行规则

- 新 RL 只能从 version26 + 当前 four-level/SAAM + reason/tool 加权 config 启动；约500题
  cohort manifest 和 matched eval 完成前，不宣称 accuracy promotion。
- A100 update-40 运行已同步：checkpoint-40 可审计，step49 因 OOM 退出；它使用旧700题/14题每update配置，只保留作历史诊断。
- 不启动新的外部 DeepSeek teacher batch。
- Qwen2.5-7B 四卡 SFT 对照使用 Qwen2.5 原生 `qwen` chat template；数据内容 SHA-256
  固定为 `c19742ec55d99980075e51a52e79285e4322f89a1d1991dea592748c279e9a14`。该运行不是
  Qwen3 SFT anchor，也不得直接作为当前 RL 起点；正式 run 已完成，最终 train loss 为
  `0.5573`，global step 为 `560`。
- 评测优先放在 table_rl/NewGNN；A100 可用时承担主 RL，A100 不可用时允许在两张空闲 3090 上
  运行降级 RL 及其 rollout serving。
- KL 是否有增益已确定要通过实验回答；在对照完成前，正式 arm 继续使用 `kl_beta=0`，且
  任何 KL 对照必须与主 run 使用同一 checkpoint、cohort、decode、runtime、GPU 拓扑和预算，
  并使用独立 output root。

## 归档迁移说明

旧 scheme 的实现目标目录是 `archive/code/`，旧 launcher/config/adapter/trajectory 目标目录
是 `archive/experiments/` 或对应历史结果目录。当前 `src/tool_modules/registry.py` 已收敛为
Atomic-only；旧 scheme 包、薄别名和仅支持旧方案的 runner 已迁入 archive，不能被默认 config
或 launcher 导入。
