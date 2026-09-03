# Atomic version26 RL 运行契约

更新时间：2026-09-03

当前只保留一条 RL 方案：**result-only four-level reward + SAAM asymmetric-error credit +
reason/tool 加权 full-response policy loss**。它是最终执行方案；700 条正式运行和 matched
promotion 完成前，不把它写成已验证的 accuracy 提升。

## 固定身份

- environment：`src/rl/tool_environment_v26.py`
- protocol runtime：Atomic version26，commit `4cd47c957fc6ae791e76a10594c8cd22f4d3b6de`
- protocol hash：`4da19387399bd3a5`
- student prompt SHA：`848598074e653648d52b58dfa1a54dd777beabae16cb91d83c4ba1a54b5d7316`
- model：Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`
- initial adapter：cumulative SFT `checkpoint-6380`（`checkpoint-560` 仅为小样本 RL 可行性验证）
- action carrier：`think-json-v1`
- causal history：recent 4 legal turns；max 30 agent steps；max 2,048 new tokens

## Reward

Reward 只读取 Harness 的 terminal denotation 和结构化 error bit，不读取 reasoning、gold
SQL 或 gold trajectory：

| terminal | 无 Harness error | 有 Harness error |
|---|---:|---:|
| correct | +1.5 | +1.0 |
| incorrect | -0.5 | -1.0 |

timeout 属于有结构化 Harness error 的 policy action；即使后续恢复并答对，timeout action
也要按局部负向惩罚计入 SAAM credit，而不是 zero credit。

实现：`src/rl/terminal_reward.py` 的 `four-level` profile。timeout 按当前 transition
contract 处理，不把模型文字解释当作额外 reward 事实。

## SAAM asymmetric-error credit

实现：`src/rl/frameworks/trl/state_action_ambiguity.py` 和 TRL transition trainer。

1. 对每个完整 `(state, full action)` 建立跨结果组的匹配。
2. 同一动作同时出现在正确和错误轨迹时，正确组保留其正优势，错误组该动作优势置零，避免
   结果标签把共享前缀误归因给 actor。
3. 仅局部错误动作使用 `-max(abs(advantage), lambda_error)`，当前
   `lambda_error=1.0`；timeout action 同样使用 `-max(abs(advantage), lambda_error)`。
4. 其余动作沿 vanilla trajectory advantage；所有 mask/count 必须写入 transition manifest。

policy reduction 为 `trajectory_token_mean`；完整 response 使用 reason/tool 各 0.5 的
`span_balance_alpha=0.5` 加权梯度。不启用 tool-only、PCGrad 或 rank loss。当前正式 arm 的
KL 为 `kl_beta=0`；后续必须用独立匹配 KL arm 验证是否带来增益。该 arm 只能改变
`kl_beta`，固定 `checkpoint-6380`、700 条 cohort、decode/runtime、GPU 拓扑、optimizer
updates 和评测口径，并写入独立 output root。系数/调度在启动前预注册，结果出来后不得回挑。

## 正式预算

| 参数 | 固定值 |
|---|---:|
| cohort | 700 records = 600 mixed + 100 manual homogeneous |
| prompts/update | 14 |
| rollouts/prompt | K=8 |
| optimizer updates | 200（四轮覆盖） |
| optimizer | AdamW，LR `4e-7`，weight decay `0.1`，clip `0.2` |
| PPO iterations | 1 |
| agent limits | max steps 30，history 4，temperature 0.8，top-p 1 |
| actor | Qwen3-8B full trainable |

配置：`src/rl/configs/experiments/qwen3_8b_atomic_v26_saam_fourlevel_spanbalanced_700.yaml`。

## A100 运行拓扑

- 两张 A100 做 FSDP full-shard BF16 actor trainer（`world_size=2`）；这是用户确认的双卡
  训练。
- 第三张 A100 起 online vLLM；trainer GPU 与 vLLM GPU 必须不同。
- 正式动态 transition packing 上限：`max_rows=8`、`token_budget=32768`。实际 GPU id、
  master/vLLM port 和 runtime path 由 launcher 写入 manifest，不在文档中永久绑定。
- canonical launcher：
  `src/rl/experiments/run_qwen3_8b_atomic_v26_saam_fourlevel_spanbalanced_700_a100.sh`

launcher 必须 fail-closed：preflight cohort/config/adapter，检查目标 GPU/端口，训练完成后
校验 run manifest、implementation lock、precision audit、checkpoint 和 replay/audit。

## 准入和评测

1. 先完成 14-record live gate，确认 FSDP、vLLM、权重同步和 transition loss 无 OOM/NCCL
   错误。
2. 再完成 700-record/200-update formal run；输出必须完整且可 fresh replay。
3. 训练后在 `table_rl` 或 `NewGNN` 做同协议 matched candidate-vs-baseline greedy eval，
   题号覆盖、API error、runtime/prompt/checkpoint identity 全部通过才可比较。
4. 只有 matched gate 通过后才能决定是否 promotion；candidate-only 结果不能当作提升。

评测衔接见 `evaluation_handoff.md`。KL 对照要求和待登记的系数/调度见
`decision_register.md`；在对照完成前不对 KL 的收益或损失下结论。

## 明确不采用

binary-only、execution-ladder、tool-only/fixed-span reward、PCGrad、rank loss、checkpoint-
relalg、Atomic v24/v39/v51/v54、Direct/Hybrid/iterative-SQL 和 projection/rewrite/delete
均为历史诊断；保留用于审计，但不得进入当前 RL config、cohort 或 output root。
