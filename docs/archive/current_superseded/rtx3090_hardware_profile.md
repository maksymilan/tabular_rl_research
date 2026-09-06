# RTX 3090 服务器说明（次要资源）

更新时间：2026-09-03

RTX 3090 只承担 SFT、matched evaluation 和行为/协议诊断；当前 RL 主实验固定放在
`a100`。完整三机分工见 [`server_resources.md`](server_resources.md)。

## `table_rl`

- 2 × RTX 3090，24GB/卡。
- 用于 version26 candidate-vs-baseline matched eval、候选评测和其他行为诊断。
- 不作为当前 RL optimizer 或 online RL 主机。

## `NewGNN`

- 8 × RTX 3090，24GB/卡。
- 用于冻结 SFT、数据准备、评测和行为诊断。
- 不作为当前 RL optimizer 或 online RL 主机。

## 使用规则

- 每次评测记录 alias、实际 GPU id、端口、checkpoint、runtime/prompt/cohort identity。
- 不根据一次 `nvidia-smi` 快照抢占 GPU；其他用户任务不得停止。
- 3090 上的 diagnostic training 不自动获得 version26 SFT/RL admission。
