# Atomic version43 唯一裸终止列 Gate16 与 first-50

## 结论

version43 的最小接口修正本身有效，但**没有形成总体准确率提升**：

- 冻结 output-shape Gate16：**13/16 = 81.25%**，通过全部预注册门槛；
- 原冻结 first-50：**39/50 = 78%**，未达到预注册的 42/50；
- 相对同题 version24 baseline 的 42/50：1 个恢复、4 个回退，净 -3，
  exact two-sided paired binomial `p=0.375`；
- 相对 version40 的 37/50：4 个恢复、2 个回退，净 +2，
  paired `p=0.6875`；
- 50/50 合法终止，12 次 unique-bare column resolution 全部确定性成功，
  `terminal_projection_error=0`；
- 但过程错误为 9，超过预注册上限 5。

因此 version43 **停止于 first-50，不运行其余 150，不准入 SFT/RL，也不能宣称已达到
80% 总体正确率**。Gate16 的 81.25% 是针对既有 output-shape 错误和匹配控制的局部结果，
不能外推到总体任务分布。

## 变更

version43 只改变 version42 的 terminal column resolver：

1. 完整逻辑列名优先；
2. 仅对不含 `.` 的裸列名尝试 dotted-column suffix；
3. 候选恰好一个才解析；
4. 候选为零或多个仍返回结构化错误；
5. 错误限定名不做 suffix 修复。

Harness 不读取 question、external knowledge、reason 或 gold，不改变行、值或声明顺序。
所有非终止工具、prompt 规则、reasoning/history、join 和 provider carrier 均不变。

实现提交：`854c7f8 feat: resolve unique bare terminal columns`。

## Gate16

### 预注册门槛

| 门槛 | 要求 | 实际 | 结果 |
|---|---:|---:|---|
| 目标准确 | ≥4/8 | 5/8 | pass |
| 控制保持 | ≥7/8 | 8/8 | pass |
| 语义终止 | 16/16 | 16/16 | pass |
| terminal projection errors | 0 | 0 | pass |
| 总过程错误 | ≤2 | 1 | pass |

总体：**pass，按预注册进入原冻结 first-50。**

### 配对

| 指标 | version41 | version42 | version43 |
|---|---:|---:|---:|
| 正确 | 10/16 | 12/16 | **13/16** |
| 合法终止 | 16/16 | 16/16 | **16/16** |
| 过程错误 | 0 | 4 | **1** |
| 合法动作 | 87 | 82 | **85** |
| total tokens | 430,725 | 400,125 | **409,988** |

version43 相对 version42 多恢复 `bird_train_06336`，没有回退。该题 terminal declaration
从错误的 `["wid","word"]` 变为正确的 `["word","wid"]`；这不是 resolver 自动重排，
而是本次模型自行声明了正确顺序。

version42 的两次终止列名错误消失：

- `n_name`/`nation.n_name`
- `playerID`/`players.playerID`

本轮模型大多直接使用了完整列名；unique-bare 实际用于 `Name` 和 `OBJ_SAMPLE_ID` 两列，
均成功。唯一过程错误属于普通 join logical-column ownership，不是 terminal projection。

## first-50

### 预注册门槛

| 门槛 | 要求 | 实际 | 结果 |
|---|---:|---:|---|
| 正确 | ≥42/50 | 39/50 | **fail** |
| 语义终止 | 50/50 | 50/50 | pass |
| 对 version24 回退 | ≤3 | 4 | **fail** |
| terminal projection errors | 0 | 0 | pass |
| 总过程错误 | ≤5 | 9 | **fail** |

总体：**fail，停止扩量。**

### 与 baseline 配对

| 指标 | version24 | version40 | version43 |
|---|---:|---:|---:|
| 正确 | **42/50** | 37/50 | 39/50 |
| 正确率 | **84%** | 74% | 78% |
| 合法终止 | 50/50 | 50/50 | 50/50 |
| 过程错误 | **3** | 9 | 9 |
| 有过程错误的题 | 3 | 7 | 6 |
| 合法动作 | 294 | 298 | **263** |
| 平均动作 | 5.88 | 5.96 | **5.26** |
| API requests | 295 | 309 | **265** |
| prompt tokens | 1,499,140 | 1,266,284 | **1,094,970** |
| completion tokens | 58,500 | 98,432 | 65,321 |
| reasoning tokens | **47,746** | 86,555 | 53,793 |
| total tokens | 1,557,640 | 1,364,716 | **1,160,291** |

version43 对 version24：

- 两者都对：38
- 两者都错：7
- version43 独赢：1（`00593`）
- version24 独赢：4（`06489`、`04189`、`02901`、`01730`）

version43 对 version40：

- 两者都对：35
- 两者都错：9
- version43 独赢：4（`03390`、`06336`、`02408`、`00593`）
- version40 独赢：2（`04189`、`01730`）

version43 相比 version24 将动作减少 10.5%、总 token 减少 25.5%，但准确率下降 6 个
百分点。相比 version40，动作减少 11.7%、token 减少 15.0%，准确率回升 4 个百分点。
它是更高效的 version40 系列接口，但不是 version24 的准确率替代品。

### resolver 行为

- 50 个终止调用全部完成确定性 projection；
- `terminal_projection_error=0`；
- 6 题共 12 列使用 `unique-bare`；
- 没有 ambiguous bare column 被错误接受；
- 9 个过程错误均来自非终止工具：
  - `condition_filter`：5
  - `inspect_rows`：2
  - `group_aggregate`：1
  - `scalar_compute`：1
  - 其中 7 个 argument validation，2 个 adjacent no-progress。

因此 unique-bare resolver 解决了它自己的易用性问题；first-50 失败不能归因于该 lowering
执行不稳定。

## 11 个错误的结构

### 最终输出槽/表示错误：5

- `06489`：external knowledge 明确将 object 映射到 `OBJ_SAMPLE_ID`，模型仍连接
  `OBJ_CLASSES` 并提交 `OBJ_CLASS="paper"`。
- `01152`：模型严格服从 “player name refers to playerID”，而 gold 要
  first/middle/last；这是 external knowledge 与 gold 的直接冲突。
- `02925`：模型找到正确 ProductID=873，却把问题中的 product 解释为 `Product.Name`；
  题面没有显式声明 ID/name 输出槽。
- `05316`：reasoning 先说只需 label，terminal declaration 却主动提交
  `["label","review"]`。
- `01730`：问题只要求 organization，模型主动加入 student name。

显式 terminal columns 让这些错误变得可审计，却不能替模型决定语义输出槽。再增加同义
prompt 说明不太可能稳定解决；version41 已经验证过这一点。

### population、约束、顺序和并列语义错误：6

- `04189`：使用 left join，保留无 review 的 app 行，而 gold 是 inner join。
- `06026`：把多个区域表的利润相加，gold 只取 `south_superstore` 的 distinct Profit。
- `06492`：按“每张图片少于 15 个 object samples”分组计数，而 gold 实际是
  `OBJ_SAMPLE_ID < 15` 后计数；external knowledge 与 gold 的表达也不一致。
- `05440`：先在 Paper 上取异常最大 Year=800190，再 left join 得到 NULL homepage；
  gold 是先 inner join Journal，再排序取一行。模型观察到异常却选择合理化它。
- `03688`：保留所有最长影片及其全部 inventory ties；gold 的 `ORDER BY ... LIMIT 1`
  只取一个 joined row。
- `02901`：reasoning 明确列出 `Gender='M'`，实际 `condition_filter` 却遗漏 Gender，
  导致 top-10 population 错误。工具完整支持该谓词，历史 reasoning 也保留，但模型没有
  将自己的约束落实到 action arguments。

这些错误不是缺列、工具不可表达或 terminal projection 问题。继续扩充工具签名会增加
选择和 prompt 负担，却无法验证自然语言约束是否完整落入关系程序。

## 对“当前工具是否适合外部模型”的判断

从协议可用性看，**适合**：

- 50/50 合法终止；
- terminal projection 50/50 成功；
- unique-bare 12/12 成功；
- action/token 明显低于 version24；
- 工具足以表达上述 11 题中至少 9 题的 gold 关系程序。

从单次 greedy 教师正确率看，**尚不适合作为 80% 目标的唯一方案**：

- first-50 只有 78%；
- 同题 version24 为 84%；
- 过程错误仍有 9；
- 主要瓶颈已从 action carrier/列名转为 semantic policy 和任务规范冲突。

## 选择性 K=2 恢复诊断

first-50 失败后，又对其 11 个失败题各运行一次全新、互不看见首轮轨迹的 causal attempt。
模型仍看不到 gold；Harness 只在 terminal 后用 `bird-set` 判定是否接受。

### 预注册结果

| 门槛 | 要求 | 实际 | 结果 |
|---|---:|---:|---|
| fresh recoveries | ≥3/11 | 3/11 | pass |
| 合并 verifier-selected pass@2 | ≥42/50 | 42/50 | pass |
| fresh 语义终止 | 11/11 | 10/11 | **fail** |
| terminal projection errors | 0 | 0 | pass |
| fresh 过程错误 | ≤3 | 7 | **fail** |

恢复题为：

- `04189`：第二次改用 inner join；
- `05316`：第二次只提交 label；
- `02901`：第二次把 Gender 条件落实到关系 action。

这三题证明同一模型/工具下存在明显 trajectory variance。但整体 gate 失败：`05440` 因三次
argument validation error 未形成 terminal；其余还出现四次参数错误。7 个错误全部是
argument validation，其中 join 5 次、scalar 1 次、condition filter 1 次。

### 成本

| 指标 | version24 单次 | version43 单次 | version43 选择性 pass@2 |
|---|---:|---:|---:|
| 正确 | 42/50 | 39/50 | 42/50 |
| 模型动作 | 294 | 263 | 346 |
| API requests | 295 | 265 | 353 |
| total tokens | 1,557,640 | 1,160,291 | 1,588,917 |
| 过程错误 | 3 | 9 | 16 |

version43 选择性 K=2 只恢复到 version24 单次的相同 42/50，却多用 2.0% token、17.7%
模型动作和 19.7% API requests，过程错误从 3 增至 16。它证明采样能恢复部分 semantic
policy variance，但当前 version43 K=2 **不是优于 version24 的教师生成方案**，不扩量。

K=2 审计：

- records：
  `data/trajectories/tool_usability_20260729/version43_first50_failures11_fresh_second_attempt_bird_set.all.jsonl`
  （SHA-256
  `02340a4c3867fa9abcae0c956c53f585dcc632038241d99dbe2d9c4b47302aeb`）
- manifest：
  `data/trajectories/tool_usability_20260729/version43_first50_failures11_fresh_second_attempt_bird_set.manifest.json`
  （SHA-256
  `e6f63c60257f6359fed708ebaf30050db427454f488aabb698accafc749697e7`）

## 下一步

1. 停止 version43 全 200；保留实现和轨迹作为 diagnostic，不推广。
2. 不再增加通用 prompt prose、plan 或新的终止字段。当前 terminal columns 已经等价于
   一个显式最终 select，再增加 model-authored slot 字段只会重复同一个决定。
3. 单独治理题目规范：
   - external knowledge 与 gold 冲突；
   - 隐含 ID/name 输出槽；
   - `MAX` 是 ties 还是 ordered single row；
   - “count images” 是 distinct/group count 还是 row count。
   这些题应从工具可用性 gate 中分层报告，不能把 benchmark 歧义当成工具缺陷。
4. 选择性 K=2 已恢复到 42/50，但合法率、过程错误和成本门槛失败；不扩量。
5. 下一轮如继续追求 80%，应先在 version24 的 55 个 full-200 failures 中分离：
   - 规范冲突/隐含输出槽；
   - 工具可表达但 semantic policy 错误；
   - 真正缺少工具表达能力。
   只对后两类做冻结目标/控制实验。当前结果不支持再写一个更长 prompt 或再加一个
   model-authored checklist 工具。

## 审计产物

Gate16：

- records：
  `data/trajectories/tool_usability_20260729/version43_unique_bare_terminal_columns_gate16_bird_set.all.jsonl`
  （SHA-256
  `f57a91b20c9d0768bec7f1feb88e77b430ada11c161acdbcac4950a95b3dab9b`）
- manifest：
  `data/trajectories/tool_usability_20260729/version43_unique_bare_terminal_columns_gate16_bird_set.manifest.json`
  （SHA-256
  `0a4c2c05d9568a3ca23d7cf654a937d4ed29868496a6f69f0d6b8e635d73f15c`）

first-50：

- records：
  `data/trajectories/tool_usability_20260729/version43_unique_bare_terminal_columns_fixed200_first50_bird_set.all.jsonl`
  （SHA-256
  `aaf5ba6557867365a51d7fc48269e1fdbb137550775432dc250a0d342998043f`）
- manifest：
  `data/trajectories/tool_usability_20260729/version43_unique_bare_terminal_columns_fixed200_first50_bird_set.manifest.json`
  （SHA-256
  `32f4d4a54b2655aa3df131e0b12087a109614c2a9627dc75ab68581556ea571f`）

本轮没有启动其余 150。
