# 当前决策登记表

更新时间：2026-09-03。这里把已确认事项和仍需用户拍板的事项分开，避免文档再出现多套
“当前方案”。用户回复后，只修改本表和 `final_project_contract.md`，再同步对应 manifest。

## 已确认

| 事项 | 当前答案 | 证据/落点 |
|---|---|---|
| 工具版本 | Atomic version26 | `docs/current/final_project_contract.md` |
| A100 训练方式 | 确实使用双卡训练 | A100 live process 的 FSDP world size=2 |
| 服务器职责 | A100 做 RL 主实验；table_rl、NewGNN 做评测等其他行为 | 用户确认；`server_resources.md` |
| A100 serving | 另用一张 A100 跑 online vLLM | 当前 live gate 进程和 formal launcher |
| RL credit | `saam-asymmetric-error` | `src/rl/frameworks/trl/state_action_ambiguity.py` |
| RL reward | four-level result reward | `src/rl/terminal_reward.py` |
| RL 梯度更新 | reason/tool 各 0.5 的加权 full-response 梯度 | `span_balance_alpha=0.5` |
| timeout | 视为 policy error，timeout action 使用局部负向惩罚 | SAAM asymmetric-error |
| RL 起点 | cumulative SFT `checkpoint-6380` | A100 formal/live launcher |
| RL 预算 | 700 条；14 prompts/update；K=8；200 updates | 当前最终预算 |
| GPU 选择 | launcher 动态选择空闲的两张 trainer 卡和一张 vLLM 卡 | 实际 id 写入 manifest |
| 旧代码 | 全部移入 `archive/`，暂不删除；`src/` 不保留 compatibility stub | `archive/code/legacy_migration.md` |
| KL 验证 | 必须通过独立匹配对照验证是否有增益；当前正式 arm 仍为 `kl_beta=0` | `docs/current/rl_pipeline.md` |

## 待用户确认

1. **KL 对照系数/调度**：是否采用固定 beta（例如 `0.01`）或预注册的 beta sweep，待
   对照启动前登记；不得在看到结果后再挑选系数。

## 未决前的执行规则

- 新 RL 只能从 version26 + 当前 four-level/SAAM + reason/tool 加权 config 启动；正式 700
  条结果和 matched eval 完成前，不宣称 accuracy promotion。
- 不启动新的外部 DeepSeek teacher batch。
- 评测继续放在 table_rl/NewGNN；A100 只承担主 RL 及其必要的 rollout serving。
- KL 是否有增益已确定要通过实验回答；在对照完成前，正式 arm 继续使用 `kl_beta=0`，且
  任何 KL 对照必须与主 run 使用同一 checkpoint、cohort、decode、runtime、GPU 拓扑和预算，
  并使用独立 output root。

## 归档迁移说明

旧 scheme 的实现目标目录是 `archive/code/`，旧 launcher/config/adapter/trajectory 目标目录
是 `archive/experiments/` 或对应历史结果目录。当前 `src/tool_modules/registry.py` 已收敛为
Atomic-only；旧 scheme 包、薄别名和仅支持旧方案的 runner 已迁入 archive，不能被默认 config
或 launcher 导入。
