# BIRD SQL 对齐 join、稳定分页与 terminal-columns Gate20

日期：2026-07-30  
指标：`bird-set`  
模型：DeepSeek v4 Flash  
性质：工程协议 checkpoint + terminal-columns 控制变量诊断

## 结论

1. `read_subtable.limit` 的上限在旧 prompt 中已经写成 `1..20`，但旧工具没有
   `offset`，因此模型在需要读取 22 行时只能发出非法的 `limit=22`。新基础协议同时在
   prompt、静态参数校验和 executor 三层强制 `1 <= limit <= 20`，并加入稳定分页。
2. 旧 `join_tables(base, joins[])` 的左右列规则不对称：左列必须是已经引入的完整逻辑
   列，右列必须是新表的裸列。新基础协议改为一个调用只表达一条边的
   `join(left, right, on, how?, left_alias?, right_alias?)`。左右输入和左右 join key
   对称解析，多跳关系通过多次 `join` 构造；历史 `join_tables` 只保留在 replay 路径。
3. 基础协议在冻结 Gate20 上为 **14/20 = 70%**，20/20 合法终止，只有两个可恢复
   错误。已知分页失败 `bird_train_02179` 在两次正式运行中均答对；terminal-columns
   运行还实际执行了 `offset=20` 的第二页读取。
4. 显式 terminal columns 变体为 **13/20 = 65%**，相对基础版 **0 gain / 1
   regression**。它没有修复任何基础版失败，并使 `bird_train_06489` 从正确变错误。
   因而保留为独立诊断分支，不并入基础协议。
5. Gate20 暴露了 derived-input alias 的一个窄缺口：SQL 风格的
   `g.filter_001.order_id` 应当表示别名 `g` 限定完整原逻辑列
   `filter_001.order_id`。该问题已在 Gate20 后修复并加入 deterministic test；下表的
   教师结果仍对应修复前提交，避免事后改写实验身份。

## 协议

### 稳定分页

```text
read_subtable(
  table,
  limit=20,
  columns=None,
  order_by=None,
  offset=0
)
```

- `limit` 必须是整数 `1..20`，绝不能超过 20。
- `offset` 必须是非负整数。
- `offset > 0` 时必须显式给出非空 `order_by`。
- executor 在模型给出的排序项之后追加其余列作为稳定 tie-breaker。
- 返回 `has_more` 和 `next_offset`；下一页复用相同的 `columns/order_by`。
- 工具只观察行，不创建或修改关系。

### SQL 风格单边 join

```text
join(
  left,
  right,
  on=[{"left": "...", "right": "..."}],
  how="inner",
  left_alias=None,
  right_alias=None
)
```

- 每次只构造一条 join 边；复合等值键仍可在 `on[]` 中一次声明。
- `on.left` 只在左输入解析，`on.right` 只在右输入解析。
- 两侧都可使用完整逻辑列、输入/别名限定列，或输入内唯一的裸列。
- `cross` 要求 `on=[]`；支持 `inner|left|cross`。
- derived input 的 alias 限定完整原逻辑列。例如输入列是
  `filter_001.order_id`，别名为 `g` 时使用 `g.filter_001.order_id`。
- 不允许 executor 根据问题或 gold 猜 join 边。

### terminal-columns 变体

```json
{
  "tool": "answer_from_context",
  "arguments": {
    "evidence": {
      "table": "join_003",
      "columns": ["orders.status"]
    }
  }
}
```

- `columns` 必须是有序、非空、无重复的精确现有逻辑列。
- harness 仅执行这一确定性投影，不读取问题、reason 或 gold。
- 它可以删除 helper 列、改变声明的列顺序，但不能改变行和值。

## 冻结 Gate20

选择文件：
`data/eval_inputs/bird_train_sql_aligned_terminal_gate20_20260730.jsonl`

SHA-256：
`9e2d59d4fe7314466a903d6eeafb4649a1384926d235300465b08b166ab52d51`

组成：

- 原 version41/version42 output-shape Gate16；
- 四个接口失败目标：`bird_train_00074`、`bird_train_01599`、
  `bird_train_02179`、`bird_train_02438`。

共同运行条件：

- `deepseek-v4-flash`
- 单次 greedy、`attempts_per_example=1`
- `rolling-legal-history`、`history_turns=4`
- `max_steps=30`
- `max_tokens=8192`
- `tool-call` carrier
- `bird-set`
- 20 workers

服务端当前把 JSON Output 下的 `{"reason":...}` 放进 native reasoning 字段，却返回空
visible content；`finish_reason=stop`。对应 infrastructure-only 运行全部/大部分在第一个
语义动作前失败，已排除。单题 probe 证明 `tool-call` carrier 能稳定形成动作，因此正式
A/B 两侧共同改用该 carrier。

## 结果

| 指标 | 基础协议 | terminal-columns | 变化 |
|---|---:|---:|---:|
| 正确 | 14/20 (70%) | 13/20 (65%) | -1 题 / -5pp |
| 合法终止 | 20/20 | 20/20 | 0 |
| 可恢复错误 | 2 | 5 | +3 |
| 模型动作 | 134 | 148 | +14 (+10.4%) |
| API 请求 | 141 | 155 | +14 |
| transport retries | 7 | 7 | 0 |
| prompt tokens | 801,304 | 900,581 | +99,277 |
| completion tokens | 47,611 | 54,958 | +7,347 |
| total tokens | 848,915 | 955,539 | +106,624 (+12.6%) |

配对结果：

- gain：0
- regression：1（`bird_train_06489`）
- 两边同为正确：13
- 两边同为错误：6
- 只有一个 discordant pair，双侧 exact McNemar `p=1.0`

这只是有针对性的 20 题诊断，不足以估计总体 BIRD 正确率；但足以否定
“terminal-columns 在当前 prompt 下是单调改进”。

## 关键轨迹

### `bird_train_02179`：分页目标恢复

旧运行在结果有 22 行时尝试 `limit=22`，随后重复失败。新基础协议答对；显式列版本
进一步实际执行：

```json
{"table":"project_003","columns":["rootbeerbrand.BrandName"],
 "order_by":["rootbeerbrand.BrandName"],"limit":20,"offset":20}
```

该动作合法，且最终 22 行结果答对。这验证了分页不是仅写在 prompt 中，而是模型、
状态、executor 和 verifier 的完整路径已经打通。

### `bird_train_02438`：不是只靠终止投影能解决

两版都构造了订单、历史状态和状态字典的关系，最后均保留了
`order_id + status_value`；gold 只要求 status。terminal-columns 版本也显式选择了这
两列，说明 harness 忠实执行了模型的错误输出意图，不能把这个错误归因于解析器。

这题同时暴露了 derived alias 缺口：

```text
g.filter_001.order_id
```

修复前解析器错误地把最后一个点前的 `g.filter_001` 当作 qualifier。修复后只剥离首个
alias `g`，再精确匹配完整原逻辑列 `filter_001.order_id`。

### `bird_train_06489`：terminal prompt 引发语义回归

基础版依据 external knowledge 返回 `OBJ_SAMPLE_ID`，答对。terminal-columns 版额外
join `OBJ_CLASSES` 并返回对象类别文本 `paper`，答错。确定性投影本身没有出错；回归
来自新增终止列契约改变了模型对“object”输出槽的理解。

### terminal exact-column 错误

显式列版本出现两次 exact-column rejection：

- `Name`，实际列为 `Product.Name`
- `title/type/price`，实际列为 `titles.title/titles.type/titles.price`

模型均能从反馈恢复到合法终止，但最终语义仍错。是否允许 unique-bare terminal column
可作为下一项独立变量，不能混入本 Gate20 的“精确列”结论。

## 提交与产物

评测提交：

- 基础：`c5d813c`
- terminal-columns：`9e4c4ae`

Gate20 后 alias 修复：

- 基础 checkpoint：`a479fd1`
- terminal-columns checkpoint：`0e673e8`

结果：

- `data/trajectories/version24_sql_aligned_gate20_20260730/base_tool_call/verified.all.jsonl`
  - SHA-256 `1af15433e7e5c2a54e1d69f7cbe9861f4c1ca8ae0a2f31a359ff1eef3ac19406`
- `data/trajectories/version24_sql_aligned_gate20_20260730/terminal_columns/verified.all.jsonl`
  - SHA-256 `bdf15f7e393e896c4140f9cbdf5bf95c178930031f73aaedbb870f81212fb20c`

本地 deterministic tests：

- active harness：88/88
- SFT：85/85
- eval：38/38（terminal 分支为 39/39）
- RL：35/35

## 决策

- 将 `a479fd1` 作为“稳定分页 + SQL 对齐对称 join”的工程 checkpoint。
- 保留 `0e673e8` 作为显式 terminal columns 的可控实验分支。
- 不把 terminal-columns 变体合入基础 checkpoint：Gate20 没有 gain，出现一个语义
  regression，并增加 10.4% 动作和 12.6% tokens。
- 历史 `join_tables` 继续只用于 replay，不能重新暴露给新 SFT/RL 轨迹。
