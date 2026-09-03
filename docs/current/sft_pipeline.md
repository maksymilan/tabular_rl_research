# version26 SFT 管线

更新时间：2026-09-03

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
