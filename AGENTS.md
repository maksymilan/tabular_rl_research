# AGENTS.md — 当前共享记忆

这是本仓库给 Codex/Claude 使用的**当前约束**，不是历史实验日志。唯一的项目入口是
`docs/current/final_project_contract.md`；历史方案和结果只在 `docs/archive/`、
`docs/reports/`、`archive/` 中用于审计，不得自动恢复为当前主线。

## 项目目标

本项目研究 relational-table 上的 LLM tool-use agent。模型只能通过 typed planning、
perception、relational、terminal 工具操作；Harness 执行 SQLite、维护 resident state、
返回结构化 observation/error，并验证 terminal denotation。研究重点是工具边界和基于
Harness 事实的 dense process credit；模型自写 reasoning 和任何 gold path 都不是 reward
权威。

## 当前唯一工具契约

- 训练、评测和 RL 统一使用 **Atomic version26**，runtime 从 commit
  `4cd47c957fc6ae791e76a10594c8cd22f4d3b6de` 导出。
- carrier 是 `think-json-v1`：一个非空 `<think>...</think>` 后紧跟一个严格 JSON action。
- 使用 Qwen3 thinking chat template，`enable_thinking=true`；rolling legal history 最近
  4 轮，完整 Harness resident state，最多 30 个 semantic steps，单轮最多 2,048 new
  tokens。
- protocol hash `4da19387399bd3a5`，student prompt SHA-256
  `848598074e653648d52b58dfa1a54dd777beabae16cb91d83c4ba1a54b5d7316`。任何运行都必须在
  manifest 中记录这些身份，不能混用其他 protocol、prompt、carrier、Harness 或 scorer。
- gold SQL 只留在 Harness 做兼容和 terminal denotation 检查，不进入模型或 teacher request。

## 当前唯一 RL 方案（最终方案，结果仍待准入门禁）

- 基座：Qwen3-8B；RL 固定从 cumulative SFT `checkpoint-6380` 启动。`checkpoint-560` 只
  用作小样本 SFT 后 RL 可行性验证模型，不是当前 RL 起点或最终 baseline。
- reward：result-only `four-level`：正确且无错误 `+1.5`，正确但有错误 `+1.0`，错误且
  无错误 `-0.5`，错误且有错误 `-1.0`；timeout 视为 policy error，timeout action 使用
  负向惩罚，不作零 credit。
- credit：`saam-asymmetric-error`；共享 `(state, full action)` 在正确轨迹上保留正优势、
  在错误轨迹上置零；局部错误动作使用 `-max(|A|, lambda_error)`，`lambda_error=1.0`；
  timeout action 同样走局部负向惩罚。policy reduction 为 `trajectory_token_mean`，reason
  和 tool span 各 0.5 的加权梯度（`span_balance_alpha=0.5`），不启用 PCGrad、tool-only
  或固定 span reward。
- 规模：目标约 500 道题；每题 K=8 且正确轨迹数 2–6，每 update 30 题，200 optimizer updates；
  AdamW、LR `4e-7`、weight decay
  `0.1`、clip `0.2`。
- 实现：A100 上 replicated BF16 actor，**一张卡承担 trainer（world size=1）**，
  另一张 A100 运行 online vLLM；两张卡只能使用 GPU 0–3。动态 micro-batch 的参数为最多 8 rows / 32,768
  transition tokens（以 launcher 和 run manifest 为准）。
- 若 A100 不可用，允许在同一台 3090 服务器上使用两张独立 3090 承担 trainer/vLLM；从
  1 row / 8,192 tokens 起步，显式记录 4-bit base 或 8-bit AdamW 等降级参数，结果与 A100
  输出隔离。
- KL：当前正式 arm 保持 `kl_beta=0`；必须用独立匹配对照验证 KL 是否带来增益，再决定
  是否启用。系数/调度在对照启动前预注册，不能看结果后回挑。
- 当前状态：A100 update-40 运行已同步，checkpoint-40 可审计，随后 step49 因 OOM 退出；新约500题
  cohort 尚未冻结，因此不能宣称 accuracy 提升或“已验证最终方案”。

除非 `decision_register.md` 明确更新，上述是唯一允许的新 RL 入口。旧的 binary-only、
execution-ladder、version24/39/51/54、checkpoint-relalg、Direct/Hybrid/iterative-SQL、
projection/rewrite/delete、tool-only/fixed-span/PCGrad 均为冻结诊断，不得进入当前 SFT/RL。

## 数据和 teacher 约束

- 新 SFT 数据必须来自真实 model↔Harness 因果循环；每个 turn 只能看合法 episode prefix、
  当前 state 和最新 feedback。
- 只收录正确、fresh replay 通过、执行可复现、结构/no-leak 门通过的 episode；不得把 gold
  SQL 编译成轨迹、补写 reasoning、或用后续轨迹丰富早期 turn。
- 官方 DeepSeek 只允许使用 `https://api.deepseek.com` 的现有 `/chat/completions` transport；
  不得使用 AimixHub/AIHubMix 或其他代理。外部生成当前暂停，只有用户明确恢复后才发送新请求。
- API key 只能放在被忽略的 `api.md`，不得写入文档、配置、命令、日志或聊天。

## 三台服务器职责

| SSH alias | 硬件 | 当前职责 |
|---|---|---|
| `a100` | 8 × A100 PCIe 40GB | RL 首选；单卡 replicated trainer + 单卡 vLLM |
| `table_rl` | 2 × RTX 3090 24GB | matched evaluation、行为诊断；A100 不可用时可做降级 RL |
| `NewGNN` | 8 × RTX 3090 24GB | SFT、评测、数据/行为诊断；A100 不可用时可做降级 RL |

GPU 编号、端口和路径由 launcher 动态选择空闲资源后写入 run manifest；不能根据一次
`nvidia-smi` 快照假设空闲，也不能停止其他用户进程。详见
`docs/current/server_resources.md`。

## 当前代码入口

- Harness/environment：`src/rl/runtime/tool_environment_v26.py`
- RL trainer/rollout：`src/rl/frameworks/trl/`
- RL config：`src/rl/configs/experiments/qwen3_8b_atomic_v26_saam_screened500_single_gpu.yaml`
- A100 launcher：`src/rl/scenarios/rl_main/run_qwen3_8b_atomic_v26_saam_fourlevel_spanbalanced_700_single_gpu_a100.sh`
- SFT：`src/sft/prepare_qwen3_atomic_sft1_newgnn.sh`、
  `src/sft/train_qwen3_8b_atomic_sft1_newgnn.sh`
- matched evaluator：`reproductions/trust_sql/qwen3_8b_atomic_sft1/`

启动任何实验前先做 preflight，确认 protocol/prompt/model/checkpoint/cohort 哈希和输出目录；
训练必须生成 immutable manifest、implementation lock、precision audit 和 fresh replay/audit
证据。变更代码后运行与改动相关的单元测试和 `git diff --check`。

## 变更和安全纪律

- 当前文档以 `docs/current/` 为准；任何状态变化同步更新
  `final_project_contract.md`、`decision_register.md` 和对应 manifest/report。
- 旧工具 scheme 代码和实验入口按用户决定迁入 `archive/`，先不删除；迁移时必须保留原始
  identity 和可复现说明。`src/` 不保留 compatibility stub；历史 replay 必须显式恢复
  archive 路径，不得用旧 artifact 初始化 version26。
- 根目录不得出现 `*.remote.current.py`、`*.remote.py` 或 `tmp_*`。远程代码快照必须放在
  `archive/code/remote_snapshots/<runtime-id>/src/...` 并附来源与哈希；一次性诊断脚本放在
  `archive/` 的日期/用途目录，不能作为当前入口。
- 保留用户已有的 dirty worktree 修改；只编辑本次请求明确涉及的文件。不要使用 destructive
  git/reset/recursive delete。
- 不输出秘密，不在 shell 命令中展开 key；外部服务失败必须显式失败，禁止静默 fallback。
