# Qwen3 Atomic-v24 projection-v3 筛选与重训记录

日期：2026-08-25  
状态：数据、质量审计、16K token 审计与最长样本 smoke 均通过；fresh epoch 1 运行中

## 为什么冻结 projection-v2

projection-v2 修正了 Qwen3 与 DeepSeek carrier 冲突，但训练目标审计发现：8K 视图的 3,869 个
episode 中 989 个没有 `answer` 监督；target token 中约 92% 是 reasoning；另有少量目标讨论
benchmark/gold memory、provider carrier，出现明显重复推理或相邻完全重复 action。v2 正式训练因此在
持久化 step 20、epoch 0.0282、loss 1.5608 时停止，plus-3 watcher 同时停止。原日志和目录保留，
冻结标记禁止 resume，也禁止把该未完成 adapter 当作新初始化。

## Projection-v3 选择

脚本：`src/sft/build_checkpoint_relalg_qwen3_projection_v3.py`。

选择只删除当前监督 record，不改写模型可见文本，不删除同 episode 的其他合法 turn，也不因
schema/strict mismatch 删除 `bird-set` 值正确数据。结果：

| 项目 | 数量 |
|---|---:|
| source canonical | 27,636 |
| 8K full-prefix core | 22,614 |
| 8K--16K supplement | 4,373 |
| combined records | 26,987 |
| contributing episodes | 3,877 |
| episodes with answer target | 3,769 |
| quality exclusions | 89 |
| strict exact records | 13,204 |
| schema exact/value correct records | 843 |
| schema relaxed/value correct records | 12,940 |

89 个删除目标的分类为：benchmark/gold-memory 73、provider-carrier discussion 4、重复 8-gram
比例至少 0.20 的 reasoning 10、相邻完全相同 successful action 2。

训练视图 SHA-256：

```text
589cebc607ecb5b0180b68d04c7ef2956ce7ad76d07af239e88efe0d0eb26a22
```

数据目录：

```text
/home/dengyan/tabular_rl_outputs/data/
  qwen3_8b_checkpoint_relalg_atomic_v24_sft3_20260825_projection_v3/
```

## 审计

- carrier/quality audit：26,987/26,987 通过；record ID 与 model input 均唯一；provider carrier、
  benchmark-memory、高重复与相邻重复 action 命中均为 0；
- exact LLaMA-Factory/Qwen3 16K audit：26,987/26,987 full-prefix 完整，target 截断 0，历史 prefix
  截断 0；总长度 p50 4,981、p90 9,601、max 16,381；
- 本地 checkpoint-relalg/SFT 相关回归：292 passed + 34 subtests。

## Smoke 与正式训练

smoke 使用实际入选记录中精确长度最长的 32 条，而不是数据前 32 条。双卡完成 2/2 optimizer
steps，exit 0；第一步 loss 1.726，每卡峰值约 16.8 GiB。

正式 run：

```text
qwen3_atomic_v24_projection_v3_full_table_rl_20260825_1204
```

从 `/home/dengyan/models/Qwen3-8B-TrustSQL-baseline` 全新初始化，不加载 adapter；table_rl GPU
0、1，QLoRA rank 16，micro batch 1，gradient accumulation 16，effective global batch 32，学习率
`1e-4`，epoch 1 共 844 optimizer steps。

启动后已完成 step 2 的首个持久化日志点：loss 2.259、grad norm 3.017、learning rate
`3.846e-06`，双卡持续 100% 计算；前两步平均约 147 秒/step，对应当前粗略 ETA 约 34.5 小时。

当前仍使用标准 last-assistant-only token CE。仓库和已安装 LLaMA-Factory 都没有经审计的
reasoning/action 分段 loss 实现，因此本轮没有热修改服务器 site-packages；index 中的 artifact tier 与
建议权重只作为后续可复现实验输入。
