# BIRD hard60：atomic version39 按需列语义实验

日期：2026-07-29

## 结论

在同一份新筛选 hard60、同一 `atomic version39`、同一 DeepSeek v4 Flash
temperature=0 单轨迹配置下，按需加入 BIRD 列语义没有带来准确率提升：

| 指标 | version39 lazy catalog 基线 | version39 按需 BIRD 语义 | 差值 |
|---|---:|---:|---:|
| `bird-set` 正确率 | 42/60 = 70.0% | 40/60 = 66.7% | -2 题 / -3.33pp |
| 合法终止 | 59/60 | 59/60 | 0 |
| process errors | 20 | 20 | 0 |
| 平均动作数 | 9.77 | 10.30 | +0.53 |
| 总 token | 4,152,091 | 4,538,508 | +386,417 / +9.31% |

配对结果为 1 个 gain、3 个 regression、39 个稳定正确、17 个稳定错误。四个
discordant 样本上的 exact two-sided sign test 为 `p=0.625`，没有显著证据说明该方案改变了
准确率。当前结果既不支持“描述一定有害”，也不支持把它提升为默认 atomic 协议；明确可见
的代价是更多 perception 调用和 token。

## 实验问题与唯一处理差异

两组都从普通 lazy catalog 开始，保留完整的 version39 atomic 工具、参数、执行器、
canonical resident state、recent-4 legal history、JSON Output carrier 和 `bird-set` scorer。
工具 schema hash 在两组完全相同：

`4d9e6cae7652ba958f316177367f18c8b68a48c4cc7e447e1b11eb3d13487125`

实验组仅增加一个模型可见、不可训练的 diagnostic profile：

- `describe_table` 在 BIRD CSV 提供该信息时，为原始列附加短
  `semantic_name`，来源字段是 `column_name`。
- `inspect_column` 在原有实时值域统计之外，为当前列附加长
  `column_description`。
- 不返回 CSV 的 `data_format`、`value_description` 或任何元数据样例值。
- 原始 table/column name 仍是唯一可执行标识。
- overlay 发生在工具执行和 canonical state 更新之后，因此不改变 harness 状态、
  provenance、SQL 执行或 verifier。
- 实验组 prompt 只增加与上述返回字段对齐的说明；这是使模型正确解释新字段所必需的
  treatment 组成部分。

基线 profile 是 `catalog-v1`；实验 profile 是
`catalog-bird-semantics-on-demand-v1`。两组 prompt hash 不同是预期处理差异，不是配置
污染。

## 数据集与解码

数据集：
`data/eval_inputs/bird_train_schema_context_hard60_20260729.jsonl`

SHA-256：
`e93b913f9628984a05924dc043ad52fcc03d9bc7e870e403635a163f157b6f94`

该集合是从冻结 200 题的未使用后段独立筛出的 60 个 `difficulty_proxy=hard`、
复杂度分数至少为 6 的任务，覆盖 34 个数据库。筛选不读取模型 correct、legal、输出或
轨迹；它是困难诊断集，不应当被解释为 BIRD 全集的无偏准确率估计。

两组共同设置：

- 模型：`deepseek-v4-flash`
- 每题 1 条 trajectory；不是 pass@k 或投票
- temperature：0
- 最大 30 个 atomic action
- workers：8
- completion 上限：2048 token
- API timeout：300 秒；transport retries 上限 10
- context：rolling legal history，`history_turns=4`
- plan：optional
- denotation：`bird-set`
- 所有结果均为 `diagnostic_only_pending_protocol_scale_gate`

两组各有 60 个唯一任务、0 个重复记录。基线的唯一非正常终止和实验组的唯一非正常终止
都是 `bird_train_06299` 达到 30 步；没有任务因 API/transport/carrier 失败退出。

## 语义信息实际覆盖

实验组不是“只改 prompt、没有返回数据”：

| observation | 调用数 | 有新增字段的调用 | 新增字段数 | 至少覆盖的任务 |
|---|---:|---:|---:|---:|
| `describe_table.semantic_name` | 77 | 72 | 894 | 59/60 |
| `inspect_column.column_description` | 52 | 48 | 48 | 31/60 |

覆盖组合为：29 题只有 describe 短名，30 题同时获得短名和 inspect 长描述，1 题只有
inspect 长描述。因此 60/60 至少获得过一种真实的 BIRD 语义 annotation。

语义组相比基线明显更愿意做 perception：

- `inspect_column`：35 → 52，增加 17 次（+48.6%）。
- `describe_table`：71 → 77，增加 6 次。
- attempted parsed actions：576 → 606（+5.21%）。
- process error 总数保持 20，但 argument validation error 从 11 增到 15。

所以额外 9.31% token 并非只有字段文本长度造成，也来自更多模型动作。

## 四个配对变化

### Gain：`bird_train_00600`

问题要求比较患高血压的男性与女性患者数量。

- 基线算出的计数正确，但终端表为两行 `gender, count`，与 benchmark 的一行
  `male_count, female_count` 形状不一致，因此判错。
- 实验组 inspect 了 condition 描述和 patient gender，之后使用
  `group_aggregate(output_layout="columns")` 产生正确的一行两列结果。

这是一个 output-shape recovery。新增描述与额外检查和恢复共同出现，但一次配对不能证明
长描述本身是因果原因。

### Regression：`bird_train_01535`

问题要求列出每天营业的 tires businesses。

- 基线最终只输出 benchmark 要求的 `business_id`。
- 实验组在得到正确的 28 个 ID 后又 join 回 `Business`，附带 city/state/stars 等列，
  形成 exact-output-shape regression。
- 该题没有调用 `inspect_column`，只获得 describe 短名。

这不是长 `column_description` 造成的直接错误；更像短名/prompt 扰动后多做了一次不必要
的实体信息扩展。

### Regression：`bird_train_01787`

问题询问 UCLA 学生中 air force department 男生的百分比。

- 基线按 benchmark 计算 `1 / 7 = 14.2857%`。
- 实验组把分母解释成全部 UCLA 学生，计算 `1 / 236 = 0.4237%`。
- `inspect_column(enlist.organ)` 返回的描述“the organization that the student enlisted in”
  是正确的，但没有解决问题中百分比作用域的组合语义。

这里的失败核心仍是 denominator scope reasoning，不是 schema 缺失。

### Regression：`bird_train_05161`

问题询问含 cheese 的 recipes 中 calories > 200 的百分比。

- 基线保留 ingredient join multiplicity，以 105 行为分母，得到 benchmark 的 80%。
- 实验组先 distinct recipe ID，得到 90 个 recipe，再计算 77.78%。
- 从自然语言“recipes”看去重有一定合理性，但与冻结 gold denotation 不一致。

这更接近 benchmark/gold 口径歧义，而不是纯粹的数据库语义能力下降。它也说明更丰富的
语义可能让模型采用“自然语言上更合理、但不等于 gold SQL”的粒度。

## 解释与决策

本实验回答了两个不同问题：

1. `column_description` 可以安全地放在 `inspect_column` 返回中；实现上不需要污染
   canonical state 或改变工具 schema，且 provenance 可以审计到 CSV hash。
2. “可以返回”不等于“当前模型会因此涨点”。在 hard60 上，模型更多地调用了 inspect，
   但净结果是 -2 题，主要错误仍然是输出形状、百分比分母和行粒度，而非不知道列含义。

因此当前 profile 保持 diagnostic-only，不提升为默认 atomic，不作为 SFT 数据源。若继续
定位效果，应拆成两个独立 treatment：

- 只在 `describe_table` 返回短 `semantic_name`；
- 只在 `inspect_column` 返回长 `column_description`。

这样才能区分短名、长描述和“提示模型更多检查”三者的作用。也应继续逐题审查 gold 粒度，
否则 `bird-set` 的变化会把更自然的 distinct 语义计作退化。

## 可复现产物

- 基线 manifest：
  `data/trajectories/bird_train_atomic39_hard60_context_ablation_20260729/catalog_v1/gate60.manifest.json`
- 实验 manifest：
  `data/trajectories/bird_train_atomic39_hard60_context_ablation_20260729/catalog_bird_semantics_on_demand_v1/gate60.manifest.json`
- 完整 paired machine-readable audit：
  `data/trajectories/bird_train_atomic39_hard60_context_ablation_20260729/paired_analysis.json`
- 分析脚本：
  `src/eval/analyze_atomic_context_pair.py`

实现提交基于隔离分支 `codex/atomic39-hard60`，不会与主工作区正在迭代的 atomic 代码互相
覆盖。
