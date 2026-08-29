# 训练主线：Qwen3-8B Atomic version26 SFT1 → 同协议评测 → RL

日期：2026-08-26  
状态：恢复已验证的 Atomic version26 SFT1 主线；后续 checkpoint-relalg/
Atomic-v24-frozen 大数据、reasoning 压缩/重写和相关训练全部冻结为诊断产物

本文是当前训练、评测和 RL 的唯一主线入口。相冲突的新实验、历史报告或脚本不自动获得
主线资格。

## 一句话主线

```text
同一 Atomic version26 因果正确轨迹
  → 同一 version26 think-json-v1 Qwen3 SFT
  → 同一 version26 BIRD-dev greedy 评测
  → 从同一 SFT checkpoint 做 binary result-only RL
  → 通过 matched gate 后再比较 grounded process credit
```

工具、prompt、carrier、history、Harness、scorer 和模型身份在全链路不更换。

## 可信基线

冻结结果见
`../reports/sft/QWEN3_8B_ATOMIC_V26_SFT1_BASELINE_20260808_ZH.md`：

| 运行 | BIRD-dev greedy `bird-set` | Legal termination |
|---|---:|---:|
| Qwen3-8B base | 270/1534 = 17.60% | 586/1534 = 38.20% |
| version26 SFT1 `checkpoint-560` | **838/1534 = 54.63%** | **1317/1534 = 85.85%** |

这是同底模、同题、同工具、同提示、同解码和同 scorer 的全量配对结果，不是小样本推断。

## 冻结身份

| 项目 | 主线值 |
|---|---|
| model | Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218` |
| runtime | atomic version26，从 commit `4cd47c957fc6ae791e76a10594c8cd22f4d3b6de` 隔离导出 |
| carrier | `think-json-v1`：一个非空 `<think>...</think>` + 一个 JSON action |
| prompt SHA-256 | `848598074e653648d52b58dfa1a54dd777beabae16cb91d83c4ba1a54b5d7316` |
| history | rolling legal history，recent 4，full resident-state prompt |
| limits | max 30 semantic steps；local generation max tokens 2048 |
| scorer | `bird-set` |
| local thinking | Qwen3 chat template `enable_thinking=true`，不使用 vLLM reasoning parser |
| SFT checkpoint | `qwen3-8b-bird-atomic-v26-sft1-6400-qlora/checkpoint-560` |

Runtime 复现、评测身份和产物门在
`reproductions/trust_sql/qwen3_8b_atomic_sft1/README.md`。

## 恢复的数据构造方法

1. 教师必须在真实 model↔Harness 循环中作答；每轮只看当时合法前缀和最新工具反馈。
2. Gold SQL 不进入教师请求，仅由 Harness 做终止 denotation 检查。
3. 只接收 terminal `bird-set` 正确、因果历史完整、fresh replay 通过、执行可复现、
   no-leak 通过的 episode。
4. 不编译 gold SQL 为工具轨迹，不程序化修理 reasoning，不重写教师当时真实输出。
5. 学生投影只处理 Qwen3 历史拼接和 token 完整性；当前 target 不改写。
6. 整个数据集必须绑定 source/index/prompt/runtime/tokenizer/template 哈希。

原 SFT1 路径是 fixed-1000 BIRD-train 教师数据。教师共得到 704/1000 正确 episode；
经历史学生投影和完整前缀门后，Qwen3 SFT1 冻结源为：

- 678 个完整 causal episode；
- 4,471 个 next-action target；
- source SHA-256 `39a8a298bd67f1a931ff582331cab6574185c1b8d17f4f23ac5de40cc4031310`；
- index SHA-256 `3c4f3ad4b01de0f9fbebfe838d79ca5648f7252fccee9f318bc33a709a219cac`；
- 所有 target 在 Qwen3/LLaMA-Factory 前缀下完整，最长 5,738 tokens < 6,400。

准备入口：`src/sft/prepare_qwen3_atomic_sft1_newgnn.sh`。

## 数据扩展规则

后续扩展不从 2026-08-21--26 的新轨迹池挑数据，而是按上节原方法增加新的
version26-compatible 因果 episode。

- 保持 Atomic version26 公共 action schema 和学生 prompt。
- 保持同一 rolling-history 政策和 6,400-token 完整前缀门。
- 只加入正确且 replay/no-leak 通过的 episode，不为达到数量降低门槛。
- 采用新的 dataset identity，不覆盖原 678-episode 数据。
- 先按 episode 分层评估长度、工具覆盖、题库覆盖和与原 SFT1 的重叠，再决定混合权重。
- 外部 DeepSeek 生成目前暂停；只有用户明确恢复后才能发起。

## 冻结 SFT 配置

准备脚本：`src/sft/prepare_qwen3_atomic_sft1_newgnn.sh`  
训练脚本：`src/sft/train_qwen3_8b_atomic_sft1_newgnn.sh`  
配置：`src/sft/configs/bird_external_teacher_qwen3_8b_sft1_qlora_6400.yaml`

| 参数 | 值 |
|---|---|
| quantization | 4-bit QLoRA |
| LoRA | all-linear, rank 16, alpha 32, dropout 0.05 |
| devices | 2 × RTX 3090 |
| micro batch | 1/device |
| gradient accumulation | 8 |
| global batch | 16 |
| epochs | 2 |
| optimizer steps | 560 |
| seed/data_seed | 42/42 |
| cutoff | 6,400 |

历史正式训练耗时 4:32:43，最终 train loss 0.53854。扩数后必须从同一 base 新建训练，
不从后续诊断 checkpoint 继续。是否从原 checkpoint-560 做增量 SFT 必须作为单独对照，
不能替代 fresh-from-base 主线。

## 同协议评测

正式 evaluator 使用
`reproductions/trust_sql/qwen3_8b_atomic_sft1/launch_detached_remote_newgnn.sh`，而不是
`checkpoint_relalg/runner.py`。

- BIRD-dev 1,534 题；
- greedy n=1, temperature=0, top_p=1；
- 单卡 vLLM，4 路 episode 动态并发；
- 同一 episode 内部严格因果串行；
- `max_steps=30`, `max_tokens>=2048`；
- 完成后必须 1534/1534 唯一覆盖，API/断线污染为 0。

增量数据的任何新 SFT 必须同时评测 base、原 checkpoint-560 和新 checkpoint，不能只报
新模型绝对分数。

## RL 接续

RL 从冻结 version26 SFT1 `checkpoint-560` 及其已有环境开始：

- environment：`src/rl/tool_environment_v26.py`；
- 现有 result-only 工程入口：
  `src/rl/experiments/run_qwen3_8b_atomic_v26_vanilla_grpo_table_rl.sh`；
- 准入顺序：先 binary terminal `bird-set` result-only，再 grounded process credit；
- result-only/process 两臂必须共享初始 checkpoint、题目、decode、runtime、训练预算和 matched eval；
- process credit 只来自 Harness replay/state/provenance/dependency，不把 reasoning 或 gold path 当作事实权威。

历史 version26 RL 产物只是工程证据和对照，不直接宣称新结果。扩数后的 SFT 必须重新做
matched result-only gate。

## 明确废弃/冻结的后续路线

以下产物保留供审计，但对当前 SFT/RL 准入为 0：

- `checkpoint-relalg/atomic-v24-frozen-v1` 2026-08-21--26 生成的大批长 reasoning 轨迹；
- Qwen3 projection-v1/v2/v3 数据和所有相关 checkpoint；
- projection-v4 extractive suffix 压缩；
- 本地 Qwen3-8B reasoning rewrite/delete 试验及输出；
- checkpoint/restore、semantic-v2--v6、Direct/Hybrid、action-block、iterative-SQL 等创新支线。

2026-08-26 已停止 `qwen3_atomic_v24_projection_v3_full_table_rl_20260825_1204`，不得 resume；
不得将其 adapter 用作 version26 主线初始化。

## 当前可执行入口

| 阶段 | 入口 | 状态 |
|---|---|---|
| SFT1 data preparation | `src/sft/prepare_qwen3_atomic_sft1_newgnn.sh` | 冻结可复现 |
| SFT1 training | `src/sft/train_qwen3_8b_atomic_sft1_newgnn.sh` | 冻结可复现 |
| version26 matched eval | `reproductions/trust_sql/qwen3_8b_atomic_sft1/` | 54.63% 基线已完成 |
| expanded causal generation | 同 version26-compatible teacher↔Harness contract | 暂停外部 API，待用户恢复 |
| result-only RL | `src/rl/tool_environment_v26.py` + version26 vanilla GRPO | 下一主线工程阶段 |
| grounded process RL | 同 version26 environment | 等待 result-only matched gate |

## 变更纪律

- 任何“当前”状态变化同时更新本文、`AGENTS.md` 和对应 manifest/report。
- 任何新工具或 prompt 实验默认是 diagnostic，不自动改变训练主线。
- 开始 SFT/eval/RL 前必须先校验 protocol、prompt、carrier、history、Harness 和 checkpoint 哈希一致。
