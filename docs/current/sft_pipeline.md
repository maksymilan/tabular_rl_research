# version26 SFT 管线

更新时间：2026-09-05

SFT 只服务于 Atomic version26 主线，使用与评测/RL 完全相同的 prompt、carrier、Harness 和
action schema。新的工具 scheme 或历史 projection 数据不得混入。

## 冻结 SFT1 锚点

- base：Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`
- source：fixed-1000 cohort 的真实 DeepSeek teacher↔Harness causal loop
- admitted data：678 个完整 causal episodes、4,471 个 next-action targets
- gold SQL：只在 Harness 做终止 denotation，不在 teacher/student 输入中
- projection：只做官方 Qwen3 chat-template 历史拼接和 token 完整性，不改 target 内容
- training：4-bit all-linear QLoRA，rank 16、alpha 32、dropout 0.05；global batch 16；
  两轮、560 optimizer steps、seed/data_seed 42、cutoff 6,400
- anchor：`checkpoint-560`，BIRD-dev greedy 838/1534 = 54.63%

入口：

- data prep：`src/sft/prepare_qwen3_atomic_sft1_newgnn.sh`
- training：`src/sft/train_qwen3_8b_atomic_sft1_newgnn.sh`
- config：`src/sft/configs/bird_external_teacher_qwen3_8b_sft1_qlora_6400.yaml`
- matched evaluator：`reproductions/trust_sql/qwen3_8b_atomic_sft1/`

## Qwen2.5-7B 四卡模型族对照（已完成）

用户于 2026-09-04 明确要求在 NewGNN 启动 `Qwen/Qwen2.5-7B-Instruct` 四卡 SFT，并复用
上述 Qwen3-8B SFT1 的同一 4,471-record model-visible training view。该对照保持
version26 prompt/carrier/target 内容、QLoRA 超参数、global batch 16、两轮和 560 steps；
模型及 tokenizer/chat template 必然改为 Qwen2.5 revision
`a09a35458c702b33eeacc393d103063234e8bc28` 与 `qwen`。

- 数据 SHA-256：`c19742ec55d99980075e51a52e79285e4322f89a1d1991dea592748c279e9a14`
- Qwen2.5 token gate：4,471/4,471 full-prefix 保留，零 target 截断，最大 5,742/6,400 tokens
- 四卡 smoke：2/2 steps 通过，无 OOM/NCCL error
- formal run：`qwen25_7b_atomic_v26_sft1_same4471_4gpu_full_20260904_203529`，exit 0，560/560 steps，最终 train loss `0.5573`
- final adapter SHA-256：`4ea1db0b17338e1c14c72af2254ddf9c3ca9241e57105080416f789a2156ac87`
- launcher：`src/sft/train_qwen25_7b_atomic_sft1_4gpu_newgnn.sh`
- config：`src/sft/configs/bird_external_teacher_qwen25_7b_atomic_v26_sft1_same4471_4gpu_qlora_6400.yaml`

这是显式模型族对照，不替换 Qwen3 SFT anchor，不进入当前 RL 起点；完成结果与 matched eval
必须单独报告。

## 新数据规则

1. 只生成 version26-compatible causal episodes；每轮 provider 只看到合法 prefix、当前
   resident state 和最新 Harness feedback。
2. 轨迹须通过 terminal correctness、fresh replay、执行可复现、结构和 no-leak gate。
3. 禁止把 gold SQL 编译成完整工具轨迹，禁止程序化补写 reasoning/observation/plan。
4. 每个 dataset 使用新的 source/index/prompt/runtime/tokenizer identity，不覆盖 SFT1。
5. 外部 DeepSeek 生成当前暂停；恢复时仅使用官方 `https://api.deepseek.com` Chat Completions。

## 当前 RL 起点

正式 RL 固定使用 cumulative SFT `checkpoint-6380`。`checkpoint-560` 只用于小样本 SFT 后
RL 可行性验证，不是正式 RL 起点；其 54.63% 结果也不能直接作为 checkpoint-6380 RL 的
matched baseline。

## 历史数据

旧 SFT2、checkpoint-relalg/Atomic-v24 projection、reasoning rewrite/delete 和 version51--54
数据保留在 archive/reports 供审计，当前 admission 为 0。
