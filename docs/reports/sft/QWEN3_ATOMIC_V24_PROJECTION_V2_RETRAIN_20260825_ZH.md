# Qwen3 Atomic-v24 projection-v2 重训记录

日期：2026-08-25  
状态：projection-v2 数据与训练均已冻结为诊断；epoch 1 在 step 20 停止，禁止 resume

## 为什么必须重新训练

旧 projection-v1 的 22,694 条样本把 assistant target 投影为本地 Qwen3 的
`<think>...</think> + JSON`，但每一条 system prompt 仍保留 DeepSeek provider 专用的
“reasoning 只能放在 `reasoning_content`”条款。模型同时收到相互矛盾的输出合同。这个冲突已经进入
旧 adapter 的梯度，不能靠更换评测 parser、修改 manifest 或继续旧 checkpoint 消除。

因此旧数据与旧 checkpoint 原样冻结为诊断产物；旧追加训练于 2026-08-25 以 exit 143 安全停止。
此次重训从未加载旧 adapter。

## 修正后的数据

- scheme/profile：`checkpoint-relalg` / `atomic-v24-frozen-v1`
- 本地 carrier：`checkpoint-relalg-qwen3-inline-think-json-v1`
- Qwen3 student prompt SHA-256：
  `92d4e226d0f5bfa82687c2034f04b1343e28601b27f41c853688110a66b2caea`
- 原生候选：28,707
- projection-v2 输出：27,636
- 完整 Qwen3 tokenizer prefix 审计后准入：22,661
- contributing episodes：3,869/3,869
- training-view SHA-256：
  `ea74fdb6f4cc3f446f86ea6934c7615d6df6b7f86450c738d0356dbf778db09c`
- 当前监督 target carrier：22,661/22,661 通过
- 历史 JSON-only assistant actions：55,855
- provider-only carrier 泄漏：0
- 历史非 canonical envelope：19，仅保留为因果错误/恢复前缀，不作为当前监督 target

训练数据目录：

```text
/home/dengyan/tabular_rl_outputs/data/
  qwen3_8b_checkpoint_relalg_atomic_v24_sft2_20260825_projection_v2/
```

## 训练计划与当前状态

初始化模型：

```text
/home/dengyan/models/Qwen3-8B-TrustSQL-baseline
```

epoch 1 run id：

```text
qwen3_atomic_v24_projection_v2_full_table_rl_20260825_1043
```

训练使用 `table_rl` GPU 0、1，两卡 QLoRA；micro batch 1、gradient accumulation 16、
effective global batch 32、学习率 `1e-4`，原计划 709 optimizer steps。两步 smoke 已完成。正式
训练在数据质量复核后由用户授权停止；最后持久化记录为 step 20、epoch 0.0282、loss 1.5608，GPU 和
plus-3 watcher 均已释放。独立 freeze marker 明确 `resume_allowed=false` 与
`source_adapter_usable=false`。

该 epoch 1 未达到成功条件，守护脚本已终止，不会再启动后续 3 epoch。后继主线是独立的
projection-v3 fresh-from-base 训练，详见
`QWEN3_ATOMIC_V24_PROJECTION_V3_FILTERED_RETRAIN_20260825_ZH.md`。

日志与 manifest：

```text
/home/dengyan/tabular_rl_outputs/logs/
  qwen3_atomic_v24_projection_v2_full_table_rl_20260825_1043.log
  qwen3_atomic_v24_projection_v2_full_table_rl_20260825_1043.launch_manifest.json
  qwen3_atomic_v24_projection_v2_epoch1_plus3_table_rl_20260825.wait.log
```

## 后续门

训练 loss 只判断优化过程是否健康。checkpoint 的模型质量必须使用同一
`atomic-v24-frozen-v1`、同一 Qwen3 inline carrier、本地 vLLM 动态批处理和 `bird-set` scorer 做闭环评测。
在 package-owned 本地 evaluator 完成并通过 identity/fresh-replay audit 前，不得用 DeepSeek runner
替代，也不得把旧 version26 或 projection-v1 结果报告成此次模型的性能。
