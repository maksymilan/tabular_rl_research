# Qwen3-8B v26：reasoning/tool carrier 与 outcome-gradient 投影诊断

日期：2026-08-29

## 实验边界

本轮只使用 NewGNN 的 GPU7，table_rl 保留给独立的 credit 训练；不启动第二张卡。
输入是 Gate60 的真实 K=8 rollout，固定普通 binary group-GRPO advantage；不读取 Gold SQL，
不重新 rollout，也不改变 v26 SFT1、Harness、prompt 或生产 trainer。

`positive_aligned` 表示正优势样本的已采样 response 平均 log-prob 增加，
`negative_aligned` 表示负优势样本的已采样 response 平均 log-prob 减少。
每个 carrier 都从新鲜 adapter 做一个 `lr=4e-6` 的 SGD-style update。

## Carrier 一步对照

| group | update carrier | full 正/负平均 Δlogp | tool 正/负平均 Δlogp | span 正/负平均 Δlogp |
|---|---|---:|---:|---:|
| q0（30/4） | full | −88.1e−6 / −46.7e−6 | −109.1e−6 / +152.9e−6 | −129.1e−6 / +44.5e−6 |
| q0（30/4） | tool | −280.4e−6 / −531.2e−6 | −97.3e−6 / −309.9e−6 | −279.8e−6 / −773.0e−6 |
| q0（30/4） | span | −6.5e−6 / −116.5e−6 | **+33.0e−6 / −62.6e−6** | **+33.2e−6 / −208.2e−6** |
| q124（30/26） | full | +8.9e−6 / +22.4e−6 | −16.9e−6 / +18.8e−6 | −18.1e−6 / +16.1e−6 |
| q124（30/26） | tool | +9.7e−6 / −198.8e−6 | **+33.8e−6 / +37.6e−6** | +24.6e−6 / −120.1e−6 |
| q124（30/26） | span | **+21.4e−6 / −203.1e−6** | −60.3e−6 / +92.2e−6 | −33.4e−6 / −70.0e−6 |

q0 的 span 对 tool/span 两侧较好，但 q124 的同一 span 权重把 tool 正向推成负、负向推成正。
tool-only 在 q124 提高了正向工具 log-prob，却同时提高了错误分支工具 log-prob；它不是安全的
最终方案。两个 group 的方向不一致，不能据此宣称 carrier 改动具有稳定收益。

## Outcome-gradient PCGrad

在 q124 上仅对正优势与负优势的 full-carrier 梯度做正锚定 PCGrad：

```text
g_neg' = g_neg - min(0, <g_neg,g_pos>/||g_pos||²) g_pos
g = g_pos + g_neg'
```

原始正/负梯度余弦为 `−0.21938`，投影后为 `9e−9`；负梯度范数只从 0.2816 降到 0.2748。
但一步 sampled log-prob 结果为：

| update | full 正/负 | tool 正/负 | span 正/负 |
|---|---:|---:|---:|
| vanilla full | −46.9e−6 / −202.8e−6 | +134.7e−6 / +133.1e−6 | +46.6e−6 / −35.9e−6 |
| PCGrad full | −137.1e−6 / +68.6e−6 | −22.9e−6 / −25.9e−6 | −132.5e−6 / +28.4e−6 |

因此“把正负梯度余弦变成非负”没有转化为正确的动作方向，反而破坏了 vanilla 原本较好的
tool/span 方向。当前不应把 PCGrad 直接接入正式 RL。

## 双余弦门控复核（q0）

为检验一个可在线判断的保护机制，进一步计算了 full 与 tool 两套正/负梯度；仅当两者余弦
都小于 `−0.1` 时才启用投影，否则沿用 vanilla。q0 满足门控条件（full `−0.156`、tool
`−0.366`），但投影后的结果为：full 正/负 `+99.8e−6 / +448.5e−6`，tool 正/负
`−49.4e−6 / +13.1e−6`，span 正/负 `+58.5e−6 / +284.8e−6`。相比 vanilla，正样本的
full/span 方向短暂改善，但负样本方向全部被翻错，工具正向也恶化；因此双余弦门控仍不能
作为可用的 loss-conflict 优化规则。q124 的 tool 余弦仅约 `−0.014`，该门控会回退 vanilla，
也没有新增收益。

## 对“reasoning/tool 冲突”的判断

q0 的直接 span 梯度余弦为：正优势 `0.104`、负优势 `0.021`，接近正交而非相互抵消。
此前四个混合 probe 的 full 正/负余弦为 `−0.156, −0.219, −0.108, −0.137`，而 tool-only
余弦为 `−0.366, −0.014, +0.198, −0.063`。这说明稳定出现的是**正/负 outcome 更新之间**的
冲突；目前没有证据说明 reasoning 梯度与 tool 梯度本身存在一个可普遍投影的负夹角。

## 当前决策

1. 不采用 tool-only、固定 span 或 PCGrad 作为正式 RL 更新规则。
2. 将本轮结论限定为机制诊断：full-response 的工具信号确实可能被稀释，但 carrier 权重和
   outcome-gradient 投影都表现出明显的 group 依赖，不能只凭冻结轨迹的一步 Δlogp 选择算法。
3. 若继续做梯度维度实验，应先在多个独立 update 上记录 `g_pos/g_neg` 及 full/tool/span
   方向，再考虑带门控的投影或仅在冲突且工具方向未被破坏时启用；在此之前不扩展正式 RL。

产物：

- `artifacts/rl/gate60_carrier_one_step_newgnn_q0_lr4e6.json`
- `artifacts/rl/gate60_carrier_one_step_newgnn_step1_q124_lr4e6.json`
- `artifacts/rl/gate60_pcgrad_one_step_newgnn_step1_q124_lr4e6.json`
- `artifacts/rl/gate60_gated_pcgrad_one_step_newgnn_q0_lr4e6.json`
- `artifacts/rl/gate60_reason_tool_gradient_geometry_newgnn_q0.json`
