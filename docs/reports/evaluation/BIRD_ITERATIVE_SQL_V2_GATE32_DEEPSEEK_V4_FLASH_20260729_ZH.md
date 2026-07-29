# DeepSeek v4 Flash 交互式 SQL Gate32 审计

## 结论

当前仓库存在一个真正的交互式 SQL 评测环境：
`src/eval/iterative_sql.py`。模型每轮只能执行一个动作：

- `execute_sql(sql)`：执行只读 SQL，观察列名、最多 20 行结果或真实 SQLite 错误；
- `submit_sql(sql)`：提交此前已经成功执行过的同一条 SQL，进行隐藏的 `bird-set` 判分。

在与 atomic version38 完全相同的冻结 32 题上，DeepSeek v4 Flash 得到
**10/32 = 31.25%**。32/32 都合法终止，API、transport 和 carrier 终止故障均为 0；
独立 fresh replay 也逐题复现为 10/32。

这一结果不支持“仅将工具换成可交互 SQL 就能恢复这些失败题”：

- atomic version37：16/32；
- atomic version38：18/32；
- iterative SQL v2：10/32。

相对 version38，iterative SQL 只有 1 个配对新增成功、9 个配对回归，净变化 -8，
exact two-sided paired binomial `p=0.021484375`。在原 16 个 version37 失败目标中，
iterative SQL 只恢复 1 个；在 16 个原正确控制中只保留 9 个。

交互式 SQL 的主要价值是更短、更便宜和更通用的数据库探索接口，而不是在本批任务上
提高教师正确率。它与 version38 的正确集合取并集也只有 19/32，只比 version38
单独运行增加 1 题。

本实验仅用于接口诊断，不准入 SFT。

## 环境语义

模型初始只看到：

- 表名；
- 行数；
- 外键关系；
- 问题和可选 external knowledge。

模型看不到完整 schema，必须用 `PRAGMA table_info(...)` 或 SQLite catalog 查询探索。
成功执行的查询进入滚动 SQL workspace；最近四个合法 assistant/observation 对进入上下文。
错误通过 `LAST SQL ERROR` 返回。SQLite 连接启用 `PRAGMA query_only=ON`，只允许单条
`SELECT`、`WITH`、`PRAGMA` 或 `EXPLAIN QUERY PLAN`。

终止前，`submit_sql` 的 SQL 必须与此前某条成功 `execute_sql` 完全一致。终止时环境重新
执行该 SQL，并在模型不可见的边界使用 gold SQL 结果进行 `bird-set` 比较。

## 载体修复

首次 v1 运行得到 0/32，但该结果是工程无效结果，不能视作模型能力：

- DeepSeek provider adapter 已按当前项目合同产生 `<think>...</think>` 后直接跟 raw JSON；
- iterative SQL 的旧 parser 仍要求已取消的 `<tool_call>...</tool_call>`；
- 因此 32 题均在 3 次 protocol error 后终止，没有执行任何 SQL。

v2 只修正这个接口自身的 action carrier：

- 改用共享的 `think-json-v1` strict parser；
- prompt 不再出现 `<tool_call>`；
- legacy tag 明确拒绝；
- manifest 记录 `iterative-sql-v2`、`execute_sql_submit_sql_v2` 和
  `assistant_carrier=think-json-v1`。

该修改不影响 atomic、action-block 或 relational-program。对应 6 个协议单元测试全部通过。

## 冻结设置

- 任务：
  `data/eval_inputs/bird_train_version38_semantic_discipline_gate32_20260729.jsonl`
- task SHA256：
  `0993665a2df7bc6447ee1b558119fa41a674251262b90c4130f5f22bc9a79cb8`
- 模型：`deepseek-v4-flash`
- 协议：`iterative-sql-v2`
- carrier：`think-json-v1`
- 判分：`bird-set`
- 最大模型动作：30
- 同类错误上限：3
- SQL 执行超时：20 秒
- 结果预览：最多 20 行
- 合法历史：recent 4
- workers：8
- API retries：10
- 每题只保留一次语义尝试；错误答案没有重试
- gold SQL 和 gold rows 对模型不可见

## 汇总

| 指标 | iterative SQL v2 |
|---|---:|
| 正确 | 10/32 |
| 合法终止 | 32/32 |
| 平均模型动作 | 7.47 |
| 模型动作总数 | 239 |
| `execute_sql` 尝试 | 192 |
| 成功 SQL 执行 | 189 |
| `submit_sql` 尝试 | 47 |
| 中间错误 | 18 |
| 有错误但最终合法终止的 episode | 14/14 |
| 总 tokens | 538,304 |
| API request attempts | 239 |
| API/transport/carrier retries | 0 |

18 个中间错误由两类组成：

- 15 个 `argument_validation_error`：模型直接提交了尚未原样执行验证的 SQL；
- 3 个 `execution_error`：真实的缺失列错误。

错误反馈的恢复机制是有效的：18 个错误后都有下一步合法 recovery，14 个受影响 episode
最终全部合法提交。但其中只有 3 题最终答对、11 题仍然语义错误。另有 11 个错误答案全程
没有任何过程错误。因此主要瓶颈不是 SQL 是否可执行或接口是否允许恢复，而是模型选择了
错误的总体、粒度、关系、公式或输出形态。

模型确实使用了环境探索：192 次 `execute_sql` 中包含 75 次 `PRAGMA` schema 查询。
但“观察过 schema 和样例结果”并不自动推出正确任务语义。

## 与 atomic 的配对结果

### iterative SQL v2 对 atomic version37

- 两者都对：9；
- iterative SQL 新增成功：1（3969）；
- iterative SQL 回归：7（1152、1692、6165、2682、600、2088、2512）；
- 两者都错：15；
- 净变化：-6；
- exact two-sided paired binomial `p=0.0703125`。

### iterative SQL v2 对 atomic version38

- 两者都对：9；
- iterative SQL 新增成功：1（3042）；
- iterative SQL 回归：9（4189、2408、6454、6165、2513、2682、600、3011、2512）；
- 两者都错：13；
- 净变化：-8；
- exact two-sided paired binomial `p=0.021484375`。

iterative SQL 与 version38 的并集为 19/32。唯一新增题 3042 是输出表示差异：
SQL 版本保留 `firstName,lastName` 两个源字段，而 version38 将 full name 拼成单列。
这不是交互式 SQL 普遍恢复复杂推理的证据。

## 22 个错误答案的主要类型

### 1. 最终输出字段或形态不精确（8）

- 600：返回 `gender,count` 两行，而非问题要求的男女比较输出；
- 1152：附带 player ID、出生日期，且姓名字段不完整；
- 2408：除作者名外又输出 author ID；
- 2512：正确找到员工，但继续携带 inspection count；
- 3011：正确找到守门员，但继续携带 player ID 和 team count；
- 3724：正确得到 top-5 标题，但继续携带 rental count；
- 6454：返回 rail/mail 两行计数，没有只输出获胜 ship mode；
- 6489：已定位正确 object sample，仍输出类别、坐标和尺寸等额外列。

### 2. 总体、粒度、过滤或公式错误（11）

- 593：观察到两条符合条件的 11/18 天疗程后，无依据只保留一个 encounter；
- 1050：外部公式只要求 `Unit Price - Unit Cost`，模型又乘以 order quantity；
- 2088：把普通行计数改成 distinct city；
- 2438：题目没有 latest/current，却只取每个订单的最新 status；
- 2513：计 inspection rows，而非通过的 businesses；
- 2682：从 Product 当前记录取均值，没有选择正确的历史成本关系；
- 4189：使用 left join，保留没有 translated review 的 App；
- 4869：无依据增加 current employee 和未结束 shift 限制；
- 4906：外部映射要求 object sample count，模型改成 distinct image count；
- 5161：外部映射要求普通 recipe row count，模型改成 distinct recipe；
- 5440：接受异常最大年份并使用 left join，返回没有 homepage 的记录。

### 3. 任务终止语义或关系未完成（2）

- 5544：问题要求女性代表人数，最终却返回男女 representative 明细；
- 6165：长达 24 步后仍使用错误/不完整的 paper-author 关系，并携带额外 paper ID。

### 4. 数值表示不一致（1）

- 1692：总体计算得到 39/59，但 SQL 将百分比提前 round 到两位，最终值为 66.1；
  `bird-set` 比较下与未舍入目标值不相等。

## 推理质量判断

交互式 SQL 提高了可观察性和局部可纠错性，但没有提高这批题的整体推理质量：

1. schema/列名错误可以被真实 SQLite 反馈修复；
2. 候选 SQL 可以在提交前看到真实样例结果；
3. 但错误结果通常同样“可执行且看起来合理”，环境没有事实依据告诉模型自然语言总体、
   输出槽或 benchmark 口径是否解释错；
4. raw SQL 的自由度反而让模型更容易附带 helper columns、提前 round、增加自认为合理的
   current/distinct/latest 条件；
5. 20 行预览能验证局部可行性，不能验证完整结果是否与问题语义等价。

因此，这个实验区分了两种能力：

- **数据库交互能力**：模型能探索 schema、修复列错误并合法提交，表现良好；
- **任务语义与精确 denotation 能力**：仍明显低于当前 atomic version38。

## 实验边界和下一步

这不是纯粹只改变 action space 的因果 A/B。version38 使用了新增的 external-teacher
语义纪律 guidance；iterative SQL v2 使用的是更短的 SQL 专用 prompt。当前结果比较的是
“接口 + prompt package”，不能把全部 -8 净差异都归因于 SQL 本身。

若继续验证，合理的下一步是：

1. 保持 SQL 工具、执行和反馈完全不变；
2. 只给 iterative SQL 加入与 version38 同义但更短的任务语义优先级和 exact-output
   约束；
3. 先在相同 16 目标 + 16 控制上做 paired gate；
4. 预注册至少恢复多个当前 SQL 输出形态错误、且不得损失更多控制；
5. 小 gate 没有信号则停止，不直接扩到 200 题。

不能把本次 10 条成功轨迹直接当成 SFT 数据。它们属于新的 SQL action protocol，
需要单独的 replay、质量、防泄漏和训练协议准入决策。

## 产物

- 正式 v2 结果：
  `data/trajectories/bird_train_semantic_gate32_iterative_sql_v2_ds_v4_flash_20260729`
- 无效 v1 carrier 诊断：
  `data/trajectories/bird_train_semantic_gate32_iterative_sql_v1_ds_v4_flash_20260729`

fresh replay 对 32 条最终 SQL 全量重执行，10 条通过、22 条不通过，与记录逐题一致，
无执行失败。报告只依据题目、external knowledge、模型自己生成的 SQL、模型可见执行反馈
和隐藏 verifier 的通过/失败信号；未查看或暴露 gold SQL 或 gold answer rows。
