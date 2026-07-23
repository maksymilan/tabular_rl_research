# BIRD 条件聚合与宽输出工具 pilot

日期：2026-07-23

## 目的

在不改变 resident plan 行为和默认策略的前提下，针对 version11 失败审计中“模型已得到
正确集合或计数，但现有工具使 denominator、grain 或一行/多行形状漂移”的题，验证两个
最小关系语义扩展：

1. 每个 `group_aggregate.aggregations[]` 可带独立 `where`；
2. 类别比较可由同一 `group_aggregate` 直接输出一行多列。

Gold SQL 只用于本地 verifier 和轨迹审计，没有进入模型或 teacher prompt。所有语义对照
均使用 DeepSeek v4 Flash、temperature=0、max tokens=2048、max steps=30、
rolling legal history=4、optional plan 和 tool-call carrier。

## version14 基线小样本

冻结 8 个 version11 K-SHAPE/近邻题：

`01152, 02408, 04848, 01692, 00040, 00600, 03724, 04038`

结果目录：

`data/trajectories/tool_usability_20260723/known_shape8_version14_tool_call.*`

strict-multiset 为 **2/8**，正确题是 `02408` 和 `00600`。

主要观察：

- `00600` 能算出 male=180、female=193，但要靠两条过滤/聚合分支再 cross join，轨迹长；
- `04038` 得到 F=553、M=572，却输出成两行 `gender,count`；
- `01692` 在分支拆解后把 nominee denominator 从 joined 59 rows 漂移到 69；
- `03724` 保留了排名辅助列。

## version15：聚合项级 where

`group_aggregate` 增加：

```json
{
  "op": "count_distinct",
  "column": "patient",
  "as": "male_count",
  "where": {"column": "gender", "op": "=", "value": "M"}
}
```

执行器将多个指标编译成同一输入表上的 `CASE WHEN` 聚合。在线执行和 replay 都会解析
谓词内的 `value_ref`/`in_table`，canonical arguments 保留原引用；provenance 同时记录
data、value、schema 和 literal-grounding edges。

首次 8 题调用先受本地网络沙箱拦截，随后同一输出前缀又追加了 8 条有效 API 记录，导致
该 manifest 混有 16 条记录，不能作为正式汇总。只审计后 8 条有效记录时仍为 **2/8**：
`02408`、`03724` 正确。模型在 `00600/04038` 上仍自然选择普通 `group_by gender`，说明
“有条件聚合能力”本身不足以覆盖已有分组路径后的输出转置。

## version16/17：为什么不保留独立 pivot

中间实验曾公开一个独立 `pivot`：

```json
{
  "table": "group_003",
  "key_column": "gender",
  "value_column": "patient_count",
  "key_values": ["M", "F"]
}
```

它能把两行 key/value 变成一行两列，但会增加一个与 `project` 竞争的工具选择，并让模型
在正确 group 后再多走一步。version16 的半截轨迹还显示：

- `00600` 已主动采用两项条件 `count_distinct`，生成正确的一行两列，之后才遭遇
  `SSL UNEXPECTED_EOF`；
- `04038` 取得正确 F=553、M=572 后仍调用 `project`；
- 三题均因同一批 API SSL 传输中断，没有形成可计分终局。

因此独立 `pivot` 只保留 replay compatibility，不进入当前公共工具集。

## version18：合并到 group_aggregate

当前调用把宽输出作为聚合布局，而不是第二个工具：

```json
{
  "tool": "group_aggregate",
  "arguments": {
    "table": "hypertension_patients",
    "group_by": ["gender"],
    "aggregations": [
      {"op": "count_distinct", "column": "patient", "as": "patient_count"}
    ],
    "output_layout": "columns",
    "category_values": ["M", "F"]
  }
}
```

约束：

- 默认 `output_layout="rows"` 保持普通 GROUP BY 行布局；
- `columns` 模式要求一个 group key、一个 aggregation、无 passthrough；
- `category_values` 同时规定类别和输出列顺序；
- `output_columns` 仅是可选重命名；
- `project` 明确保持行方向，不能承担转置。

### 有效配对结果

`00600`：

`data/trajectories/tool_usability_20260723/known_shape_00600_version18_merged_retry.*`

- strict-multiset：**1/1**；
- predicted / gold：`[[180,193]]`；
- 模型先产生普通两行分组，随后用同一个 `group_aggregate` 加
  `output_layout="columns"` 重算为一行两列；
- 没有独立 reshape action；
- 轨迹虽恢复了一个 join 参数错误和一个 provider carrier 错误，但最终语义正确。

`04038`：

`data/trajectories/tool_usability_20260723/known_shape_04038_version18_merged.*`

- legal terminal，但 strict-multiset 错误；
- 模型和工具得到正确计数 F=553、M=572，输出为
  `[["F",553],["M",572]]`；
- gold 是 `[[553,572]]`；
- 模型 reasoning 明确认为问题“provide the number for each gender”自然对应
  `gender,count` 两行。问题和 external knowledge 没有规定一行两列，因此这更接近
  benchmark output-shape 欠规定，而不是工具仍无法表达。

另一次 `00600` 探针三次均为 visible-content 前缀 carrier error，没有合法动作，不计入
工具准确率：

`data/trajectories/tool_usability_20260723/known_shape_00600_version18_merged.*`

## 当前结论

1. 条件聚合补上了真实 SQL 语义缺口：多个指标可以固定在同一输入 population/grain，
   不再依赖易漂移的 filter branches。
2. 类别宽输出应与 `group_aggregate` 合并。独立 `pivot` 增加工具竞争和轨迹长度；合并后
   `00600` 已证明模型能自然调用并得到正确 denotation。
3. `04038` 不能继续作为纯工具失败样本。它的两行输出符合问题直觉，而 gold 隐含一行
   两列；强迫模型只会拟合 benchmark 未明示的布局。
4. 当前证据仍是小样本，不足以扩展到固定 200，更不能开始 SFT 构造。下一阶段应挑选
   一组 question/gold 明确要求并列输出槽的条件计数/比例/差值题，先验证 merged
   aggregate 的稳定采用率，再决定是否跑 200。

## 回归

- Harness：63/63
- SFT：77/77
- RL：34/34
- Eval：25/25

这些检查覆盖条件 `where` SQL、wide-layout SQL、strict arguments、online
`value_ref` execution、replay compatibility 和 provenance grounding。Resident plan 的实现、
可见性和 optional 默认策略均未改变。
