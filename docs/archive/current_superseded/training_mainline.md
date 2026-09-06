# 训练主线：Atomic version26 SFT → A100 单卡 replicated RL

更新时间：2026-09-03

本文只描述当前执行主线；旧版本和诊断分支不在这里展开。

## 主线顺序

```text
version26 causal SFT
  → 同协议 BIRD-dev greedy matched evaluation
  → A100 单卡 replicated trainer + 一卡 online vLLM RL
  → table_rl/NewGNN matched evaluation 与 replay/行为审计
```

所有阶段固定 Atomic version26 的 runtime、prompt、carrier、history、Harness、scorer 和
模型身份。不得在阶段之间切换工具版本或把旧实验 artifact 混入结果目录。

## 工具和模型身份

| 项目 | 值 |
|---|---|
| model | Qwen3-8B，revision `b968826d9c46dd6066d109eabc6255188de91218` |
| runtime | commit `4cd47c957fc6ae791e76a10594c8cd22f4d3b6de` 导出 |
| protocol | version26，hash `4da19387399bd3a5` |
| carrier | `think-json-v1` |
| student prompt | SHA-256 `848598074e653648d52b58dfa1a54dd777beabae16cb91d83c4ba1a54b5d7316` |
| history/limits | recent-4 legal history；max 30 steps；max 2,048 new tokens |
| scorer | `bird-set` |

## SFT 锚点

冻结 SFT1 的 `checkpoint-560` 只用于小样本 SFT 后 RL 可行性验证，BIRD-dev greedy 为
838/1534 = 54.63%；正式 RL 起点固定为 cumulative SFT `checkpoint-6380`。其数据来自真实
teacher↔Harness 因果循环：678 个完整 episode、4,471 个 next-action targets；gold SQL 未
进入模型请求。二者用途不同，不能在报告中混称。

当前 A100 RL live manifest 使用 cumulative SFT `checkpoint-6380`，这正是正式 RL 起点。

SFT 入口：

- 数据准备：`src/sft/prepare_qwen3_atomic_sft1_newgnn.sh`
- 训练：`src/sft/train_qwen3_8b_atomic_sft1_newgnn.sh`
- 配置：`src/sft/configs/bird_external_teacher_qwen3_8b_sft1_qlora_6400.yaml`

## RL 接续

当前唯一执行方案是 `saam-asymmetric-error` + four-level result reward + reason/tool 各
0.5 的加权 full response，完整参数见 [`rl_pipeline.md`](rl_pipeline.md)。A100 正式资源是一张卡做 replicated trainer（world size=1），另一张卡起 vLLM，且只允许 GPU 0–3；`table_rl` 和 `NewGNN` 不运行主
RL optimizer。

A100 update-40 历史运行已完成 checkpoint-40，随后 step49 因 OOM 退出。约500题 formal run、
matched candidate-vs-baseline eval 和 promotion gate 尚未完成，任何报告都必须标注这一状态。

## 数据准入

1. 新数据必须由真实 model↔Harness loop 产生；每轮只看当时合法前缀和最新反馈。
2. 通过 terminal denotation、fresh replay、结构、执行复现和 no-leak 门后才可入库。
3. 不编译 gold SQL 为完整轨迹，不重写 reasoning，不把后续观察泄漏到早期 turn。
4. 外部 DeepSeek 生成暂停；恢复时只能使用官方 `https://api.deepseek.com` Chat Completions。

## 旧路线处理

checkpoint-relalg、Atomic v24/v39/v51/v54、Direct/Hybrid/iterative-SQL、projection/rewrite/
delete、binary-only 和 execution-ladder 仅保留审计，不是当前入口。旧代码按迁移清单统一移入
`archive/`，暂不删除 checkpoint、trajectory 或 report。
