# SFT pipeline

## 当前目标

当前唯一可进入后续评测与 RL 的 SFT 主线，是已验证的 Qwen3-8B Atomic version26 SFT1
方法。行为锚点为 `checkpoint-560`：BIRD-dev greedy `bird-set` 838/1534，正确率
54.63%。完整身份与回退决定见 `training_mainline.md`。

## 数据准入

训练轨迹必须来自真实的外部教师或学生模型与 Harness 的逐轮因果交互：每一轮只看到当时的
状态和最新工具反馈，gold SQL 对模型不可见。只有终局答案通过 denotation 验证，并同时通过
fresh replay、执行一致性、协议、质量与 no-leak 检查的完整轨迹，才能进入 SFT。

以下操作不属于当前数据方法：

- 根据 gold SQL 编译或补全工具轨迹；
- 改写、压缩或删除已有 reasoning 来制造新的监督目标；
- 混合不同工具 schema、prompt、carrier、history renderer 或 Harness 版本；
- 将失败诊断、截断记录、无法精确 replay 的记录改标签后纳入训练。

## 冻结 SFT1 数据

历史 fixed-1000 教师生成中有 704 条正确轨迹。经过 Qwen3 rolling-prefix 投影后，冻结 SFT1
数据包含 678 个完整因果 episode、4,471 个 next-action target；所有 target 都在 6,400-token
截断线内，最后 answer target 不被改写或丢弃。

| 身份 | 冻结值 |
|---|---|
| SFT source SHA-256 | `39a8a298bd67f1a931ff582331cab6574185c1b8d17f4f23ac5de40cc4031310` |
| SFT index SHA-256 | `3c4f3ad4b01de0f9fbebfe838d79ca5648f7252fccee9f318bc33a709a219cac` |
| rolling-full prompt SHA-256 | `848598074e653648d52b58dfa1a54dd777beabae16cb91d83c4ba1a54b5d7316` |
| carrier | `think-json-v1` |
| history | recent four legal turns |
| cutoff | 6,400 tokens |

数据准备入口是 `src/sft/prepare_qwen3_atomic_sft1_newgnn.sh`。任何扩容数据都必须生成新的
dataset identity，但应保持 version26 的工具、学生 prompt、carrier、history 投影和准入规则
不变。

## 冻结训练方法

训练入口是 `src/sft/train_qwen3_8b_atomic_sft1_newgnn.sh`，配置为
`src/sft/configs/bird_external_teacher_qwen3_8b_sft1_qlora_6400.yaml`：

- Qwen3-8B base revision `b968826d9c46dd6066d109eabc6255188de91218`；
- 4-bit all-linear QLoRA，rank 16、alpha 32、dropout 0.05；
- 两张 RTX 3090，per-device batch 1，gradient accumulation 8，global batch 16；
- 两个 epoch、560 optimizer steps，`seed=42`、`data_seed=42`。

冻结 adapter 位于
`/home/dengyan/tabular_rl_outputs/checkpoints/qwen3-8b-bird-atomic-v26-sft1-6400-qlora/checkpoint-560`。
扩容训练必须从同一 base revision 新启，不能从后来失败的 projection checkpoint 或压缩试验
续训。

## 扩容顺序

1. 用相同 version26-compatible 教师↔Harness contract 生成更多真实因果 episode；
2. 按终局正确、fresh replay、执行与 no-leak gate 过滤；
3. 用同一 rolling-prefix 投影构造监督目标，不改写模型文本；
4. 从同一 Qwen3-8B base 重新训练，并与冻结 `checkpoint-560` 做 matched evaluation；
5. SFT 行为通过后，在同一 version26 环境中先做 binary result-only RL，再比较 grounded
   process credit。

外部 DeepSeek 教师生成目前暂停，只有用户明确恢复后才启动。

## 明确排除

2026-08-21--26 的 `checkpoint-relalg/atomic-v24-frozen-v1` 大批长 reasoning 数据、
projection-v1/v2/v3、projection-v4 压缩、Qwen3 本地 rewrite/delete 输出及对应 checkpoint 均为
诊断产物，当前 SFT/RL 准入数为 0。保留它们用于审计，但不得与 version26 数据合并、重新
命名或作为续训起点。

## 可执行入口

- 数据准备：`src/sft/prepare_qwen3_atomic_sft1_newgnn.sh`
- SFT：`src/sft/train_qwen3_8b_atomic_sft1_newgnn.sh`
- 复现说明：`reproductions/trust_sql/qwen3_8b_atomic_sft1/README.md`
- 匹配评测：`reproductions/trust_sql/qwen3_8b_atomic_sft1/launch_detached_remote_newgnn.sh`
- RL 环境：`src/rl/tool_environment_v26.py`
