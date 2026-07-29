# BIRD hard60：atomic version39 inspect-only 列语义实验

日期：2026-07-29

## 结论

按照“只在 `inspect_column` 返回列语义，`describe_table` 完全不返回”重新实现并运行后，
inspect-only 组得到 **41/60 = 68.33% `bird-set`**：

| 指标 | 原始 lazy 基线 | describe 短名 + inspect 长描述 | inspect-only 短名 + 长描述 |
|---|---:|---:|---:|
| 正确率 | 42/60 = 70.00% | 40/60 = 66.67% | 41/60 = 68.33% |
| 合法终止 | 59/60 | 59/60 | 59/60 |
| process errors | 20 | 20 | **11** |
| 平均动作数 | 9.77 | 10.30 | 10.28 |
| 总 token | 4,152,091 | 4,538,508 | 4,492,315 |

inspect-only 相对原始基线是 1 gain、2 regressions，净 -1 题、-1.67pp，exact two-sided
sign test `p=1.0`。相对组合语义组是 2 gains、1 regression，净 +1 题、+1.67pp，
同样 `p=1.0`。当前样本不支持准确率提升；较明确的积极信号是 process errors 从 20
降到 11，但它同时增加了 31 个动作和 8.19% token。

因此该方案可以作为更干净的 progressive-disclosure 设计保留，但仍应是
diagnostic-only，不能根据这 60 题提升为默认 atomic 或进入 SFT。

## 严格的处理定义

两组都使用同一份 `atomic version39`、相同公开工具 schema 和执行语义。inspect-only
profile 为：

- `describe_table` 返回普通 atomic 原始 schema，只有 raw `name/type/pk/foreign_keys`。
- `describe_table` 不返回 `semantic_name`，也不返回 `column_description`。
- `inspect_column` 保留 harness 实时产生的 distinct/null/frequent-values 信息。
- 只有 `inspect_column` 可附加：
  - BIRD CSV `column_name` → 模型可见 `semantic_name`；
  - BIRD CSV `column_description` → 模型可见 `column_description`。
- 不暴露 CSV 的 `data_format`、`value_description` 或元数据样例值。
- raw table/column name 始终是唯一可执行标识。
- overlay 在工具执行和 canonical state 更新之后发生，不改变 resident state、
  provenance、SQL 执行和 verifier。

落盘审计检查了全部 71 个成功 `describe_table` observation：**0 个 annotation
violation**。真实 BIRD 检查中，`describe_table(gender_age)` 仍只有 raw schema，而
`inspect_column(gender_age.device_id)` 返回：

```json
{
  "semantic_name": "device id",
  "column_description": "unique number of devices"
}
```

原始 observation 和模型可见 observation 分别记录 SHA-256，原始对象没有被修改。

## 数据、模型和解码

- 数据：
  `data/eval_inputs/bird_train_schema_context_hard60_20260729.jsonl`
- 数据 SHA-256：
  `e93b913f9628984a05924dc043ad52fcc03d9bc7e870e403635a163f157b6f94`
- 模型：DeepSeek v4 Flash
- temperature：0
- 每题一条 trajectory；不是投票或 pass@k
- 最大 30 个 atomic actions
- recent-4 rolling legal history
- optional plan
- JSON Output carrier
- `bird-set` denotation
- 60 个唯一任务、0 个重复
- 没有 API/transport/carrier 退出

三组唯一非正常终止都是 `bird_train_06299` 达到 30 步。

## 信息覆盖与行为成本

inspect-only 共发生 63 次成功 `inspect_column`：

| 可见信息 | 调用覆盖 | 任务覆盖 |
|---|---:|---:|
| `semantic_name` | 34/63 | 25/60 |
| `column_description` | 58/63 | 39/60 |
| 两者同时存在 | 33/63 | — |
| 两者都不存在 | 4/63 | — |

共 39 题实际获得至少一个新字段，21 题没有调用或没有获得 inspect annotation：

- 有新字段的 39 题：原始基线 24 对，inspect-only 23 对。
- 没有新字段的 21 题：18 对 → 18 对，没有 correctness 变化。

所有三个配对 correctness 变化都发生在实际看到 inspect 语义的任务中，说明变化局部上
与 treatment 覆盖一致；但净效果仍为 -1。

相对原始基线：

- `inspect_column`：35 → 63，增加 80%。
- `read_subtable`：99 → 107。
- `describe_table`：71 → 71，调用数相同且输出不含语义。
- process errors：20 → 11。
- attempted parsed actions：576 → 607。
- 总 token：+340,224，增加 8.19%。

所以额外成本主要来自 prompt 诱导的更多定点检查和读取，而不是把全表 description 塞进
`describe_table`。

## 相对原始基线的三个配对变化

### Gain：`bird_train_02088`

问题询问 area code 787 中 P.O. Box Only 和 Post Office 城市数量之差。

- 基线先按 raw row count 得到 benchmark 所需的 `-64`，但随后认为“number of cities”
  应当 count-distinct city，改成 `-61` 并判错。
- inspect-only 在 `inspect_column(zip_data.type)` 看到
  `column_description="the type of the postal point"`，直接按两类 row count 得到 `-64`。

这个 gain 与 benchmark 一致，但 count-distinct 从自然语言角度也有合理性，因此它仍含有
gold 粒度歧义，不能作为描述提升语义能力的干净证据。

### Regression：`bird_train_01787`

问题询问 UCLA 学生中 air force department 男生的百分比。

- 基线按 benchmark 计算 `1/7 = 14.2857%`。
- inspect-only 在看到 `organ → organization` 以及
  “the organization that the student enlisted in”后，仍将分母解释为全部 UCLA 学生，
  得到 `1/236 = 0.4237%`。

描述正确，但没有解决百分比 denominator scope 的组合语义，和上一组合语义组发生了相同
回归。

### Regression：`bird_train_05161`

问题询问含 cheese 的 recipes 中 calories > 200 的比例。

- 基线保留 Quantity join multiplicity，以 105 行为分母，得到 gold 的 80%。
- inspect-only 看到 Ingredient.category 的长描述后，将 recipe ID distinct 成 90 个，
  得到 77.78%。

按自然语言“recipes”去重有合理性，但与 gold SQL 的 join-multiplicity 口径不一致。该题
再次说明语义信息可能推动更自然的行粒度，却不一定提高 BIRD execution accuracy。

## 与组合语义组的关系

将短语义名从 `describe_table` 移到 `inspect_column` 后：

- 恢复了 `bird_train_01535`，避免最终多输出 Business 详情列，但用了 27 步和 3 个
  process errors。
- 新恢复 `bird_train_02088`。
- 丢失组合组曾恢复的 `bird_train_00600` 输出形状题。
- 总分 40 → 41，process errors 20 → 11，token 下降约 1.02%。

方向上 inspect-only 比“全表 describe 短名”更干净，也减少了错误，但 2 gains / 1
regression 仍不显著，不能视为准确率 promotion。

## 决策

1. 工程设计上，列的短名和长描述可以只放在 `inspect_column`，且无需污染 canonical
   state；这是比 describe 全表返回语义更符合 lazy catalog 的边界。
2. 现有 hard60 没有显示准确率增益：41/60 低于原始基线 42/60。
3. inspect-only 显著增加 inspect 倾向和 token，因此后续若保留，应把“是否需要语义”
   视为有成本的 perception 决策。
4. 三个变化都涉及输出粒度或百分比作用域，其中两题存在明显自然语言/gold 口径张力。
   下一步比继续增加描述更重要的是独立审计 benchmark grain，并训练 population/grain
   决策。

## 产物

- inspect-only manifest：
  `data/trajectories/bird_train_atomic39_hard60_context_ablation_20260729/catalog_bird_semantics_inspect_only_v1/gate60.manifest.json`
- 相对原始基线的配对分析：
  `data/trajectories/bird_train_atomic39_hard60_context_ablation_20260729/paired_analysis_inspect_only_vs_baseline.json`
- 相对组合语义组的配对分析：
  `data/trajectories/bird_train_atomic39_hard60_context_ablation_20260729/paired_analysis_inspect_only_vs_combined.json`
- 实现提交：`bb44bd4`
- 隔离分支：`codex/atomic39-hard60`

