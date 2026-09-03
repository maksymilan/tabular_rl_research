# 当前项目最终契约

更新时间：2026-09-03（Asia/Shanghai）

这是项目当前唯一的收敛说明。`docs/current/` 中其他文档只补充这里定义的契约；历史分支、
失败诊断和旧 checkpoint 仍可审计，但不再是新实验入口。

## 1. 已确定的工具版本

训练、评测和 RL 全部固定为 **Atomic version26**：

| 项目 | 固定值 |
|---|---|
| runtime | `4cd47c957fc6ae791e76a10594c8cd22f4d3b6de` 导出树 |
| carrier | `think-json-v1`：非空 `<think>...</think>` + 一个 raw JSON action |
| prompt | rolling-full，SHA-256 `848598074e653648d52b58dfa1a54dd777beabae16cb91d83c4ba1a54b5d7316` |
| protocol | `version26`，hash `4da19387399bd3a5` |
| context | causal legal history 最近 4 轮 + 当前完整 resident state |
| limits | 每 episode 最多 30 semantic steps；每轮最多 2,048 new tokens |
| model | Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218` |
| terminal scorer | `bird-set`，结果必须引用 Harness 产生的 exact artifact |

Gold SQL、gold result 和任何完整 gold path 只在 Harness 内部用于执行兼容和 denotation
验证；不能进入模型 prompt、teacher request、trajectory 或 reward 解释。

## 2. 已确定的训练路线

```text
version26 causal SFT
  → 同 runtime/prompt/carrier 的 matched BIRD-dev greedy
  → A100 双卡 FSDP online RL（另用一张 A100 起 vLLM）
  → 在 table_rl/NewGNN 做 matched evaluation 与行为审计
```

冻结 SFT1 的小样本 RL 可行性验证模型是 `checkpoint-560`，BIRD-dev greedy 为 838/1534 =
54.63%；当前正式 RL 固定从 cumulative SFT `checkpoint-6380` 作为初始 adapter。两者
用途不同，报告中不得混称。

## 3. 唯一 RL 方案（当前最终方案）

当前新 RL 只允许使用：

- result-only `four-level` reward：
  - correct + no Harness error：`+1.5`
  - correct + Harness error：`+1.0`
  - incorrect + no Harness error：`-0.5`
  - incorrect + Harness error：`-1.0`
- `saam-asymmetric-error` credit：共享 `(state, full action)` 在正确轨迹保留正优势、在
  错误轨迹置零；局部错误 action 使用 `-max(|A|, 1.0)`；timeout 视为 policy error，
  timeout action 同样使用局部负向惩罚。
- `trajectory_token_mean` policy reduction；reason span 和 tool span 各占 0.5 的加权梯度
  （`span_balance_alpha=0.5`）；不使用 tool-only reward、PCGrad 或 reasoning 作为事实。
- 当前运行 `kl_beta=0`；后续必须做独立匹配对照验证 KL 是否带来增益，再决定是否启用。
  对照固定同一 `checkpoint-6380`、700 条 cohort、decode/runtime、GPU 拓扑和更新预算，
  只改变 `kl_beta`，使用独立 output root；具体系数/调度在对照登记前保持未定。
- Qwen3-8B 全参数 actor，AdamW，learning rate `4e-7`，weight decay `0.1`，clip `0.2`，
  200 optimizer updates，ppo iterations `1`。
- 700 条固定 cohort（600 mixed + 100 manual homogeneous），每 update 14 prompts，每个
  prompt K=8，四轮覆盖；max agent steps 30，history 4，temperature 0.8，top-p 1。
- A100 FSDP full-shard、BF16 base storage：两张卡为 trainer ranks（world size 2），第三
  张卡为 online vLLM；正式动态打包上限为 8 rows / 32,768 transition tokens。

配置和入口：

- `src/rl/configs/experiments/qwen3_8b_atomic_v26_saam_fourlevel_spanbalanced_700.yaml`
- `src/rl/experiments/run_qwen3_8b_atomic_v26_saam_fourlevel_spanbalanced_700_a100.sh`

截至本文更新时间，A100 只完成一 update、14 records 的 live gate；700-record formal run
和 matched candidate-vs-baseline eval 尚未完成。因此该方案虽已确定为唯一执行方案，但尚未
取得最终 accuracy promotion。

## 4. 服务器分工

| 服务器 | 硬件 | 允许的主要工作 |
|---|---|---|
| `a100` | 8 × A100 PCIe 40GB | RL 主实验；双卡 trainer + 独立 vLLM 卡 |
| `table_rl` | 2 × RTX 3090 24GB | matched evaluation、候选评测、行为诊断 |
| `NewGNN` | 8 × RTX 3090 24GB | SFT、评测、数据准备、行为诊断 |

GPU id、端口和具体输出目录由每次 launcher 动态记录；不要把当前快照当成永久资源分配。
详见 `server_resources.md`。

## 5. 当前状态和停止线

- 外部 DeepSeek 生成暂停；恢复前不发新 teacher batch。
- 新数据必须真实 model↔Harness 因果生成、fresh replay、结构和 no-leak 全部通过。
- checkpoint-relalg、Atomic v24/v39/v51/v54、Direct/Hybrid/iterative-SQL、projection/
  rewrite/delete、binary-only、execution-ladder、tool-only/fixed-span/PCGrad 均冻结为历史
  诊断，不得进入当前 SFT/RL 或默认配置。
- 旧代码按 `archive/code/legacy_migration.md` 迁入 `archive/`，先不删除 checkpoint、trajectory
  和 report；保留原始 identity 与复现说明。`src/` 不保留旧路线 compatibility stub，历史
  replay 必须显式恢复 archive 路径。

待用户确认的事项集中在 `decision_register.md`，确认后只更新该登记表和本契约，不再恢复一
套平行路线。
