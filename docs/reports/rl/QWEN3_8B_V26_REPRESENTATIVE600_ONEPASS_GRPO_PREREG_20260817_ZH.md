# Qwen3-8B version26 representative600 单遍 vanilla GRPO 预注册

日期：2026-08-17

## 研究问题

在不使用 process reward、rank loss、自定义 credit、reward-conditioned cohort 或
checkpoint 事后选择的前提下，最基础的 binary result-only GRPO 能否提升冻结的
Qwen3-8B Atomic version26 SFT1？

此前 mixed180 两遍训练的正式 matched BIRD-dev1534 结果为 `+10/1534`
（`+0.6519pp`，McNemar `p=0.4876`），且 legal 净减少 5 题，因此没有通过晋级门。
本实验只改变训练覆盖：从 reward-conditioned 的 180 题两遍，改为早已冻结、仅由公开
字段分层选择的 representative600 一遍。它不是 Arm B，也不使用自研 credit。

## 冻结训练合同

- 初始策略：Qwen3-8B Atomic version26 SFT1 checkpoint-560。
- 任务：frozen representative600，600 个唯一题、69 个数据库；不用 gold SQL、执行结果
  或 rollout reward 做选择或排序。
- 目标：binary terminal result reward；正确且可训练为 1，否则为 0。
- 算法：vanilla GRPO，group size 8，PPO iteration 1，clip 0.2，KL beta 0。
- 优化：30 prompts/update，20 updates，600 题恰好一遍，共 4,800 条 fresh on-policy
  trajectories；LR `8e-7`，constant scheduler，AdamW，LoRA/Adam moments FP32。
- rollout：temperature 0.8、top-p 1、thinking on、max new tokens 2048、context 16384、
  max agent steps 30、history 4、SQLite timeout 10 秒。
- checkpoint：每步保存，用于恢复；唯一允许正式评测的策略是 final，且必须与
  checkpoint-20 字节一致。禁止在 checkpoint 1–19 中挑最好者。

## step-5 同步安全门

checkpoint-5 保存后、optimizer step 6 开始前，同步执行不可变审计。只有以下全部通过才
继续；失败即保留证据并停止，不改阈值、不换任务、不调参：

- 前 5 步严格为 150 个唯一任务、每题 K8，共 1,200 条；policy/sync identity 完整。
- mixed group 至少 30/150，且每步至少 1 个 mixed group。
- semantic-eligible 至少 97%；generation-length exclusion 至多 3%；legal 至少 90%。
- generation OOM、context overflow、tokenization warning 为 0；timeout 必须被排除且有
  state-preserving 结构化证据；无未知 exclusion。
- 5 步 loss/gradient 有限，`0 < grad_norm <= 1`，LR 恒为 `8e-7`，每步同步 252 个
  QLoRA 层。
- importance `log_ratio_abs_mean <= 0.05`，applied ratio mean 在 `[0.95,1.05]`，
  cap fraction 至多 `0.001`。
- checkpoint-5 的 trainable LoRA 与 Adam moments 为 FP32；raw/effective LoRA 均发生
  有限、非零移动。

该门只判断训练信号与工程健康，不声明泛化性能。receipt 固定前 1,200 条轨迹和规范化
身份；断点恢复必须保留 checkpoint-5 并完整重算 receipt，不能只信旧的 `passes=true`。

## 完成与正式评测合同

训练完成审计要求 20×30×8=4,800 条、600 题各 K8 一次、全程 binary result-only、
任务字段绑定、无 OOM/context/tokenization 污染、实现锁与 runtime/model/protocol 身份一致、
FP32 精度通过，且 final 与 checkpoint-20 完全一致。

完成后只运行一次 fresh matched BIRD-dev1534：同一 GPU、同一 version26 runtime、相同
decode/并发/timeout，顺序 SFT1→final。晋级必须同时满足：

- accuracy 净增至少 16 题（至少约 `+1pp`）；
- exact two-sided McNemar `p < 0.05`；
- legal 净变化不小于 0。

未通过只能表述为“该冻结规模下未检测到可靠提升”；不得据结果延长本次训练、重复 dev
直到显著、选择中间 checkpoint，或把它改称自研策略结果。
