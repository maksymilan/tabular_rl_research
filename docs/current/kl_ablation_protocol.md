# KL 增益验证协议

当前正式 RL arm 固定 `kl_beta=0`。用户已确认需要用实验回答 KL 是否带来增益；在对照
完成前，不把“无 KL”解释为结论。

## 预注册约束

- 起点：同一 cumulative SFT `checkpoint-6380`，加载同一 frozen PEFT reference。
- 数据：同一 700 条最终 cohort、同一题目顺序和 K=8 rollout 结构。
- 运行：同一 Atomic version26 runtime/prompt/carrier、Qwen3 revision、decode 参数、
  max steps/token budget、optimizer、200 updates、FSDP world size=2 和 online vLLM 拓扑。
- 唯一自变量：`optimizer.kl_beta`；KL arm 使用独立 output root 和独立 manifest。
- 结果：在 `table_rl` 或 `NewGNN` 用同一 matched greedy evaluator 比较 BIRD `bird-set`、
  legal termination、Harness error、token/step 成本和训练稳定性；必须通过 fresh replay、
  identity、结构和 no-leak 审计。

## 尚未登记的选择

KL 系数或 sweep 范围尚未确定。启动前必须在 `decision_register.md` 登记固定 beta 或预注册
sweep，并保持选择盲于结果。不得把 KL arm 混入正式 `kl_beta=0` 结果目录，也不得改变
SAAM、four-level reward 或 reason/tool 0.5/0.5 梯度定义。
