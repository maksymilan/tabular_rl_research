# Qwen3-8B Atomic v26：非对称 SAAM 与梯度冲突记录实验准备

日期：2026-08-29

## 目的

本轮只验证更新后的 SAAM credit 规则，不改变 v26 SFT1 的模型、提示词、工具协议、
rollout carrier、Harness、采样和优化器身份，也不把 BIRD-dev 1534 评测题用于训练。
训练数据沿用 Gate60 的 BIRD-train 任务身份与 K=8 轨迹；BIRD-dev 1534 题保持纯评测。

## 训练目标

首先按普通 binary result-only GRPO 在每个问题的 K=8 轨迹内得到轨迹优势 (A_i)，并使用
掩码前的 transition/trajectory 数做 `trajectory_token_mean` 归一化。随后对每个已解析工具
动作应用确定性的非对称规则：

- 同一题、同一 lineage-aware 确定性状态、同一完整工具参数在正负轨迹都出现时，正确轨迹保留
  正向优势，错误轨迹置零；
- `argument_validation_error`、`execution_error`、`no_progress_repeat`、`protocol_error`
  都是局部明确错误，优势覆盖为
  \(A^{\mathrm{err}}=-\max(|A|,\lambda_{\mathrm{error}})\)，即使该轨迹最终答对也惩罚该步；
- 基础设施 timeout 不提供负向语义信号，置零；
- 其他动作保留 vanilla GRPO 优势；掩码/覆盖后不重新归一化。

首个配置固定 `lambda_error=1.0`。`0.5/2.0` 作为后续离线敏感性或独立消融，不在同一
训练进程中混用。

## 梯度记录

开启 `record_gradient_conflicts: true` 后，每个 optimizer update（而不是每个 microbatch）
在梯度裁剪前记录三组梯度：

1. 实际混合更新梯度；
2. 仅保留正优势 transition 的反向梯度；
3. 仅保留负优势 transition 的反向梯度。

记录内容包括完整参数向量的 norm、正负梯度 cosine、正/负与混合梯度 cosine、冲突质量
`max(0,-2 dot(g+,g-))`、逐参数层 norm/nonzero，以及固定大小的可复现梯度 sketch。默认不
落盘完整 FP32 向量，避免 QLoRA 长实验产生数十至数百 GB 文件；如确需逐参数重算，可用
`gradient_conflict_save_vectors: true` 单独启动。输出位于：

```
<output-dir>/gradient_conflicts/parameter_manifest.json
<output-dir>/gradient_conflicts/summary.jsonl
<output-dir>/gradient_conflicts/step_XXXXXX_sketch.pt
```

正/负反向只是在同一个已构造 batch 上重新计算，不会改变 optimizer 梯度：记录完成后会恢复
实际混合梯度，再由 Trainer 执行正常的 clip 和 optimizer step。为避免分布式 collective
死锁，所有 rank 都执行两次 counterfactual backward，但只由主 rank 写文件。该诊断要求
`rank_loss=0`、`KL=0`、`gradient_accumulation_steps=1`。

## 已准备的实现

- `src/rl/frameworks/trl/state_action_ambiguity.py`：lineage-aware 身份、非对称 SAAM、
  明确错误覆盖与审计计数；旧 `trajectory`/`saam-strict` 语义保留。
- `src/rl/frameworks/trl/gradient_conflict.py`：梯度统计、sketch 和可选完整向量记录。
- `src/rl/frameworks/trl/transition_grpo.py`：非对称 credit 分支和梯度记录钩子。
- `src/rl/frameworks/trl/run_transition_grpo.py`：CLI、manifest、实现源码锁。
- `src/rl/config/experiment_config.py`：新 credit、`error_penalty`、梯度记录配置校验。
- `src/rl/configs/experiments/qwen3_8b_atomic_v26_saam_asymmetric_gate60.yaml`：本轮
  Gate60 配置，默认只保存梯度统计和 sketch。

本地和远端 runtime 均已通过 Python 编译检查、SAAM 单元测试（10/10）和配置测试（5/5），
并完成 1 题、K=8、1 update 的真实双卡 smoke：成功写出 checkpoint、梯度参数清单、梯度
sketch 和 summary；首次 smoke 暴露的 cosine 浮点越界也已在记录器中裁剪到 [-1,1]。正式
Gate60 已在 `table_rl` 双卡启动（训练卡 GPU0、vLLM 卡 GPU1），输出目录和日志由 run3
manifest 固定绑定；在线 rollout 仍按现有协议需要两张卡。

## 训练后必须检查

除 held-out BIRD-dev 1534 的同题评测外，保存并比较：梯度正负 cosine/冲突质量、SAAM
错误类型计数、共享正确保留与共享错误抑制数、工具错误率、合法终止率、平均动作数、工具
调用分布 JS divergence，以及 wrong→right/right→wrong 配对变化。只有梯度记录、轨迹结构
审计、训练 manifest 和评测结果同时齐全，才判断 SAAM 是否改善了震荡。
