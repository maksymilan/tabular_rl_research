# A100 FSDP 运行后端

更新时间：2026-09-03

FSDP 不是另一套算法，而是当前 Atomic version26 RL 的正式资源实现：两张 A100 共同训练
一个 full-shard BF16 actor，第三张 A100 运行现有 server-mode vLLM。它不改变 Harness、
causal prefix、SAAM credit、four-level reward、reason/tool 加权梯度或 transition token loss。

## 资源和固定参数

- trainer：launcher 动态选择两张空闲卡，FSDP `world_size=2`；
- serving：动态选择一张与 trainer 不重叠的卡；
- base storage：BF16，FSDP full-shard；LoRA/actor 参数和 optimizer 细节由 run manifest 记录；
- dynamic packing：最多 8 rows、32,768 transition tokens；
- 700 records、14 prompts/update、K=8、200 optimizer updates；
- NCCL 默认禁用 IB，实际 P2P 设置和端口写入 manifest，不在文档中硬编码为永久事实。

## 入口和门禁

正式入口：
`src/rl/experiments/run_qwen3_8b_atomic_v26_saam_fourlevel_spanbalanced_700_a100.sh`

launcher 的 `preflight` 必须先验证 cohort manifest、manual audit、experiment config、
adapter、protocol runtime 和脚本语法；`run` 还要检查三张目标卡、vLLM port、完整 training
artifact、FSDP precision audit 和 fresh replay。任一项失败即停止，不复用已有 output root。

当前 A100 live gate 只完成 14 records/1 update；正式 700-record run 完成前，本页描述的是
执行后端和最终方案，不是已验证的 accuracy 结果。

## 历史与当前边界

旧的单卡 trainer、Scheme 2 smoke、execution-ladder、tool-only、fixed-span 和 PCGrad 产物
保留用于工程审计，但不再作为独立主线。需要复现历史 run 时必须使用原 manifest、原
runtime 和隔离目录。
