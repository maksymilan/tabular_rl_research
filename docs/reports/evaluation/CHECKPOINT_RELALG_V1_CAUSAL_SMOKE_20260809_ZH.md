# checkpoint-relalg-v1 官方教师 API 因果 Smoke（2026-08-09）

## 结论

本次真实外部调用验证了官方 DeepSeek 身份检查、结果落盘、结构审计与 fresh replay 路径，
但没有形成可评价的模型工具轨迹。官方 `https://api.deepseek.com/chat/completions` 在 Direct
模式首轮返回 `HTTP 402 / Insufficient Balance`。因此本次状态是 **transport-blocked**，不是
Direct 语义失败；Atomic 与 Hybrid 未继续发送相同的必败请求，也没有改用第三方 provider、
代理或 fallback。

当前不能从本次尝试报告三种 mode 的准确率、合法终止率或相对表现，SFT/RL 准入保持关闭。
官方账户恢复可用余额后，应在冻结代码、任务选择与运行参数下重新运行 Direct、Atomic、
Hybrid 各自独立的因果 smoke。

## 冻结边界

- protocol：`checkpoint-relalg-v1`
- scheme：`checkpoint-relalg`
- attempted mode：`direct`
- model：`deepseek-v4-flash`
- admission：`diagnostic-only`
- backend / dialect：`sqlite` / `sqlite`
- provider 身份预检：已认证，目标模型存在
- student prompt SHA-256：
  `87d47b6f4e461a2a53d0d216f5f1013a79222b87502a2f5013506d901a28d13a`
- teacher prompt SHA-256：
  `adc69b25324931f0cc812420d18907a04c2ca7080078dc588a119440e6de5e39`
- Direct tool schema SHA-256：
  `1262589f58c2b2c4f7dbd5a22052b57693c43b0560cb04c0b87c24f1860af792`

任务只按既定位置由 runner 选择；本报告不读取或披露 question、external knowledge、gold SQL、
gold result、模型 reasoning 或数据库值。

## 实际结果

| 项目 | 结果 |
| --- | --- |
| 官方 `/models` 身份检查 | 通过 |
| Chat Completions 请求 | `HTTP 402 / Insufficient Balance` |
| 形成 authored native tool call | 否 |
| 形成语义 Harness step | 否 |
| tool execution error | 0 |
| 结构审计 | 1/1 记录通过，0 issue |
| fresh replay | 1/1 记录通过，0 issue |
| 可用于行为评价的 episode | 0 |

结果目录为 ignored 的
`data/results/checkpoint_relalg_v1_causal_smoke_20260809/direct/`。失败记录只证明官方传输请求
被明确拒绝，以及失败 artifact 可以被确定性审计；它不证明任何工具 mode 的能力。

## 后续恢复条件

1. 只恢复官方 DeepSeek 账户余额，不更换 endpoint、provider 或 credential 来源。
2. 源码或协议若有变化，重新运行 `generate-hard-db-rollouts` preflight 并冻结新哈希。
3. 三个 mode 使用独立结果目录，报告 provider/transport failure 与 semantic failure 两套口径。
4. 每条成功 episode 必须通过结构、provider-history、no-leak、state-mutation 与 fresh replay
   审计；隐藏 scorer 只回写布尔/计数，不回流 gold 内容。
5. smoke 仍是 diagnostic-only，不能进入现有 atomic SFT/RL 数据或 checkpoint。
