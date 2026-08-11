# Checkpoint-RelAlg 模型自主阶段 v2 对照 Gate8（2026-08-11）

## 结论

`model-choice-commit-v2` 已经解决 v1 的零触发问题，但没有通过行为收益门：模型在 8 题中的
5 题使用 checkpoint，共 7 次 commit，全部被 Harness 接受；官方 `bird-set` 正确率为 4/8，
而 fresh `checkpoint-disabled-v1` 对照为 6/8。配对结果是双方都对 4、仅 v2 对 0、仅对照对 2、
双方都错 2。这个 Gate8 证明“模型可以自己选择阶段边界”，但不支持“当前 checkpoint 策略提高
准确率”。两臂仍是 diagnostic-only，不能进入 SFT/RL。

## 冻结设计

- 数据：teacher1500 v2 nonempty，positions `1,14,19,22,42,44,47,54`。
- 两臂：Atomic `semantic-v2`、A/Text-JSON、官方 `deepseek-v4-flash`。
- candidate：`model-choice-commit-v2`，Harness eligibility=`none`，最多 8 次 checkpoint，
  restore=0。模型在开局自行判断 single-stage 或 multi-stage；只有它判断一个可复用语义阶段已经
  稳定且仍有不同后续阶段时才 commit。
- control：`checkpoint-disabled-v1`，checkpoint=0，restore=0。
- 每题 provider token 上限 425,000；16 条 episode 名义总上限 6.8M。
- 同题两臂相邻，4 条动态 lane 并行，防止长轨迹阻塞整批。

结果目录：

`data/results/checkpoint_relalg_v1_flash_text_json_atomic_model_choice_v2_vs_disabled_gate8_20260811/`

## 汇总

| 指标 | model-choice v2 | disabled |
|---|---:|---:|
| episodes | 8 | 8 |
| bird-set correct | 4 | 6 |
| strict artifact | 2 | 3 |
| schema match | 4 | 3 |
| legal termination | 8 | 7 |
| model turns | 95 | 89 |
| provider attempts | 104 | 89 |
| total tokens | 1,927,945 | 1,862,756 |
| prompt tokens | 1,807,034 | 1,798,322 |
| completion tokens | 120,911 | 64,434 |
| tool/process errors | 7 | 5 |
| checkpoint episodes | 5 | 0 |
| successful commits | 7 | 0 |
| restores | 0 | 0 |
| token-cap stops | 0 | 1 |

v2 比对照多 65,189 tokens（+3.50%）、6 turns 和 15 provider attempts。它没有超过预注册的
1.15 倍成本边界，errors 也恰好等于 `control + 2`；但正确率落后 2 题，超过允许落后 1 题的
行为门。checkpoint exposure 门（coverage >=5/8、commits >=6）通过，准确率门失败。

在 v2 实际 commit 的五题中，positions 22、42 双方都正确；44 只有对照正确；47 双方都错；
54 的 v2 合法答错，而对照因 token cap 未形成合法答案。没有出现 v2-only correct。因此当前
证据不能把 position 54 的合法终止改善等价为语义正确率收益。

## 审计

- 16/16 record structure、manifest binding、cohort identity、fresh replay 通过。
- 16/16 response model identity 为 `deepseek-v4-flash`；endpoint 为官方
  `https://api.deepseek.com/chat/completions`。
- 15/16 overall audit 通过。唯一失败是 disabled position 54 返回后累计 464,081 tokens，超过
  冻结的 425,000 单题上限并以 `max_provider_tokens` 停止；其 record structure 与 fresh replay
  仍通过。
- 全批实际使用 3,790,701 tokens、193 provider attempts，低于 6.8M 名义总上限。

## 解释与下一步

v1 的实际问题是提示过弱，20 题 223 turns 中零 commit，无法比较 checkpoint 效果。v2 用
“开局自主判断任务是否多阶段；若是，在自行认定的稳定阶段边界必须 commit”解决了 manipulation
failure，同时没有恢复 producer-count 或 Harness 决策门。这个改变成功制造了干预，但当前干预
仍可能在错误的语义边界压缩上下文，且 commit 后额外决策没有带来 paired gain。

下一步不应继续单纯增强 commit 强度，也不应据此删除 checkpoint。应保持模型选择和 Harness
eligibility=`none`，对这 8 对做 content-free 阶段位置审计，并设计一个更窄的 v3：只澄清“稳定
阶段必须已经经过 observation 验证、后续目标不能依赖尚未保留的细粒度证据”，不规定 producer
数量、turn 数或 Harness eligibility。任何扩大必须 fresh paired，并以 v2 作为 exposure control。

## 审批基础设施说明

本批启动前出现一次桌面审批通道断线。根因不是 DeepSeek 权限或项目预授权失效，而是启动命令
被错误标成 `require_escalated`。该评测只写 workspace，且当前 sandbox 已启用 network access，
本来无需提权。仓库 `AGENTS.md` 已固定规则：少于 200 条、官方 DeepSeek、仅写配置 writable
roots 的预授权批次必须直接在 `workspace-write` 沙箱运行，不得仅因联网而请求提权。修复后剩余
请求全部无审批弹窗完成。
