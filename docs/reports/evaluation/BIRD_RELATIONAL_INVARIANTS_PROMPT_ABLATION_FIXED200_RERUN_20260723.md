# BIRD relational-invariants prompt 固定 200 题重跑

日期：2026-07-23

## 决策

**暂不把 relational-invariants prompt 设为 canonical prompt，也不据此开始新 SFT 数据构造。**

在冻结 200 题、严格 paired A/B 下：

- P0 canonical：**132/200 = 66.0%**；
- P1 relational invariants：**137/200 = 68.5%**；
- P1 净增 5 题，但只有 13 gains / 8 regressions；
- 双侧 exact McNemar `p=0.3833`，差异不显著；
- P1 legal terminal 从 190 降到 188，recoverable error events 从 91 增到 116；
- P1 没有达到预先要求的 **150/200 = 75%**。

因此，P1 提供了有用的关系推理方向，但不能证明它稳定提高了工具可用性。固定 30 题的
24/30 是一个偏乐观的小样本信号；本次 200 题结果是当前主结论。

## 控制条件

两组都从头运行完整冻结 cohort，没有拼接或复用 fixed-30 轨迹：

`data/eval_inputs/bird_train_tool_interface_validation200_version4.jsonl`

共同配置：

- 模型：`deepseek-v4-flash`；
- 200 tasks，one semantic attempt/task；
- temperature=0，max tokens=2048；
- rolling legal history=4，full rolling prompt；
- optional plan；
- thinking enabled，reasoning effort=high；
- `tool-call` provider carrier；
- max steps=30，max recoverable errors/type=3；
- table output rows=0；
- strict no-repair parser；
- strict-multiset denotation。

唯一实验变量：

- P0：`--policy-prompt-variant canonical`；
- P1：`--policy-prompt-variant relational-invariants`。

Artifacts：

- P0：
  `data/trajectories/tool_usability_20260723/prompt_ablation_tool_call_p0_fixed200_rerun.*`
- P1：
  `data/trajectories/tool_usability_20260723/prompt_ablation_tool_call_p1_fixed200_rerun.*`

两组 manifest 均为 200 unique examples、0 duplicate attempts、0 final API failures。

本次沿用 fixed-30 预检后可工作的 `tool-call` carrier，因为当前 API 路由下 JSON Output
会频繁把 action JSON 放入 `reasoning_content` 并返回空 visible `content`。P0/P1 内部
carrier 完全一致，因此 prompt A/B 有效；但不能把本次 provider-level 指标直接混同于旧
JSON Output runs。

## 主结果

| 指标 | P0 canonical | P1 relational invariants | 差值 |
| --- | ---: | ---: | ---: |
| Strict-multiset correct | 132/200 | **137/200** | **+5** |
| Accuracy | 66.0% | **68.5%** | **+2.5 pp** |
| Legal terminal | **190/200** | 188/200 | -2 |
| Wrong answer | 58 | **51** | -7 |
| Final protocol error | 9 | 9 | 0 |
| Final argument validation | 1 | 0 | -1 |
| Final execution error | 0 | 2 | +2 |
| Final max steps | 0 | 1 | +1 |
| Mean actions/task | **7.825** | 7.935 | +0.110 |
| Total model turns | **1,565** | 1,587 | +22 |
| Recoverable error events | **91** | 116 | +25 |
| Optional plan calls | 14 | 5 | -9 |

逐题配对：

| Pair outcome | Count |
| --- | ---: |
| Both correct | 124 |
| P1 gain | 13 |
| P1 regression | 8 |
| Both wrong | 55 |

McNemar discordant count 是 21，13/8 的双侧 exact `p=0.3833`。不能据此宣称 P1
建立了稳定的统计提升。

新的 P0 为 132/200，超过旧的 125/200 原始设计目标，也略高于历史 version11 的
130/200；但历史 runs 的协议版本、provider route 和 carrier 不完全相同。对 P1 的干净
因果比较只能使用本次同步运行的 P0=132，而不是跨运行选择较低的 125 作为 baseline。

## 难度分层

| Difficulty | P0 correct | P1 correct | Gains | Regressions | Net |
| --- | ---: | ---: | ---: | ---: | ---: |
| Easy, n=80 | 59 | 60 | 3 | 2 | +1 |
| Medium, n=60 | 38 | 38 | 4 | 4 | 0 |
| Hard, n=60 | 35 | 39 | 6 | 2 | +4 |

P1 的描述性增益主要出现在 hard subset，但 hard 的 discordant 样本仍只有 8 个，不能把
这一分层结果解释成稳定的 difficulty interaction。

## Fixed-30 是否复现

从本次全新 200-run 中重新取冻结输入的前 30 题：

- P0：20/30；
- P1：22/30；
- 20 both correct、2 P1 gains、0 regressions、8 both wrong。

先前独立 fixed-30 是 21/30 对 24/30。方向相同，但具体分数与 gain tasks 不完全复现，
说明 temperature=0 并没有消除 provider/backend 路由和生成轨迹的方差。24/30 不应继续
被当作 prompt 的稳定效果估计。

## 13 个 gains

| ID | P0 问题 | P1 修正 | 与 P1 规则的关系 |
| --- | --- | --- | --- |
| 01535 | 输出 business id 外又增加 city/state/stars 等列。 | 只输出 `business_id`。 | 明确符合 exact output slots。 |
| 01692 | 对 nominee people 做 distinct count，错误折叠重复 nominee rows。 | 按 gold 保留每条 nominee row 计数。 | 明确符合 preserve observed grain。 |
| 02512 | 把 first/last name 拼成一个 full_name 列。 | 保留 first_name、last_name 两列。 | 明确符合不合并独立 output slots。 |
| 02513 | count inspection rows=211。 | count distinct businesses/license=203。 | 符合问题实体 grain。 |
| 02682 | 直接取 Product 当前 StandardCost。 | join ProductCostHistory 后取历史平均。 | 符合“所需 population/columns 依赖另一关系时先 join”。 |
| 02918 | P0 因 carrier/protocol errors 终止。 | P1 合法返回 0。 | 主要是生成/carrier 方差，不能归因于关系规则。 |
| 03373 | 在 desert name 外添加常量 count 列。 | 只返回 gold 的 desert names。 | 符合 exact output，但问题文字本身同时问数量，含 benchmark slot 歧义。 |
| 03724 | top-5 输出 title 与辅助 rental_count。 | 只输出 title。 | 明确符合 helper ranking column 不进入答案。 |
| 04189 | 使用 left join，保留没有 review 的 app rows。 | 使用默认 inner join。 | 明确符合默认 inner join 规则。 |
| 04598 | 保留重复 alias。 | `distinct=true` 匹配 gold。 | 符合输出实体集合，但不是新增 suffix 的单向直接结果。 |
| 04906 | P0 因 carrier/protocol errors 终止。 | P1 完成 broccoli/tomato ratio。 | 主要是生成/carrier 方差。 |
| 05316 | 输出 restaurant label 与辅助 review。 | 只输出 label。 | 明确符合 exact output slots。 |
| 05760 | 用 FBI code 近似 weapons violation。 | 使用 IUCR primary description 的正确关系。 | 符合 relation choice / required relation。 |

其中约 9 题与 P1 的 exact entity、join population 或 grain 规则有直接一致的轨迹证据；
2 题主要是 P0 carrier termination；其余含 benchmark slot 或普通生成方差。不能把全部
13 gains 都视为 prompt 的可重复语义收益。

## 8 个 regressions

| ID | P1 回归 | 判断 |
| --- | --- | --- |
| 01002 | 在 weapon 外额外输出 count。 | 违反 P1 自己的 exact output 规则，普通生成失误。 |
| 01297 | 错用 `value_ref`，随后两次使用不存在的 derived-handle join namespace。 | 工具引用/namespace 失误。 |
| 01425 | 把数值 `grad_100` 当成字符串类别 `"yes"`，再 count rows。 | 列语义与 aggregate 选择错误。 |
| 01730 | 把 organization 误解为 school，选择 `enrolled` 而非 `enlist`。 | relation/entity 理解错误。 |
| 02254 | 三次让 scalar_compute 引用 read-only step，而不是 table-producing step。 | 工具 reference-chain 失误。 |
| 03042 | 已形成合理的 goalie aggregation，但重复读取后因 carrier errors 终止。 | carrier/trajectory instability。 |
| 03216 | 对 criteria ID 求和后除以 3，而不是 count criteria。 | aggregate operator 语义错误。 |
| 03262 | 未做 distinct，保留重复 station/item rows。 | 很可能是 preserve-grain 规则的过度应用；这是 P1 的真实风险。 |

P1 并未导致统一一种回归模式。只有 03262 明显可能由“不要折叠重复观测”过度强化；
其余主要是未遵守 exact-output、错误 relation/operator、工具引用错误或 carrier 终止。
这仍然意味着 P1 不能被视为无副作用的 policy patch。

## Process errors 与 carrier

| Recoverable event | P0 | P1 |
| --- | ---: | ---: |
| Protocol/carrier | 71 | 76 |
| Argument validation | 9 | 20 |
| Execution | 11 | 20 |
| Total | **91** | **116** |

Protocol/carrier 的主要原因：

| 原因 | P0 | P1 |
| --- | ---: | ---: |
| Visible text/field labels before tool_call | 55 | 56 |
| Empty visible content | 10 | 6 |
| Incomplete tool_call | 5 | 8 |
| Invalid tool-call JSON | 1 | 5 |
| Other | 0 | 1 |

因此，`tool-call` 只是避开了当前 JSON Output 路由的空 content 灾难，并没有成为一个
高稳定 carrier。两组都有 9 个 final protocol failures；约四分之三的 recoverable errors
仍是 provider carrier formatting，而不是 relational tool execution。

P1 的 execution errors 增加主要来自：

- `scalar_compute`：3 → 8；
- `join_tables`：1 → 5；
- `condition_filter`：两组均为 5。

Argument-validation events 也从 9 增到 20，常见错误仍是：

- join `right` 错误使用 qualified column；
- `group_aggregate.passthrough` 不是 list；
- 把 `where`、`conditions` 或 `result_name` 放到错误层级；
- read limit 越界；
- scalar operand shape 错误。

P1 没有修改工具 schema，却增加了这些错误，说明增加关系决策文字并未让工具调用本身
更稳，甚至可能因额外策略负担和更频繁的复杂关系操作提高 schema drift。

## 成本

| 指标 | P0 | P1 | 差值 |
| --- | ---: | ---: | ---: |
| Prompt tokens | 7,582,431 | 8,044,585 | +6.10% |
| Completion tokens | 376,857 | 381,716 | +1.29% |
| Total tokens | 7,959,288 | 8,426,301 | +5.87% |
| API request attempts | 1,580 | 1,597 | +17 |
| API transport retries | 15 | 10 | -5 |

静态 system prompt 增加 807 字符（15,019 → 15,826，+5.37%）；本次实测总 token 增长
5.87%，同时 actions 和 recoverable errors 也增加。Fixed-30 中“prompt 变长但轨迹变短”
的结果没有在 200 题复现。

## 最终结论

1. P1 有真实但有限的方向性价值，尤其能提醒模型处理 exact output slots、join-before-
   rank 和 row grain；它在本次 200 题净增 5。
2. 该增益不显著、伴随 8 regressions、更低 legal rate、更多 schema/execution errors 和
   5.87% token 开销。它不应替换 canonical prompt，也不应被计为 tool/RL 改进。
3. 当前 version19 canonical 工具在这次 provider 条件下为 132/200，已高于原 125/200
   baseline 目标；P1 为 137/200。但两者都低于用户要求的 150/200，尚不能用“75% 工具
   可用性”作为开始构造新 SFT 数据的依据。
4. 下一步优先级不是继续扩写全局语义提示，而是：
   - 先做 carrier-only prompt/transport 小样本消融，减少复制 field labels 导致的
     protocol failures；
   - 针对反复出现的 join right-column、named scalar source 和 aggregate argument shape，
     用更短、更局部的 schema guidance 验证是否减少 process errors；
   - 将 relational invariants 保留为独立 ablation 或用于失败样本的训练信号，而不是在
     Base/SFT/RL 中未经验证地全局硬编码。
