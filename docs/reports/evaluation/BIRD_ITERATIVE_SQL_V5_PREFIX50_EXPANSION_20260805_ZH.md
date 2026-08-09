# BIRD iterative-SQL v5 Prefix50 扩展门

日期：2026-08-05  
状态：已完成；准确率小幅正向但不显著，可靠性与成本门失败，不继续扩量

## 结论

在 active `bird_train_baseline300_v1` 冻结 Prefix50 上，DeepSeek v4 Flash 的
`iterative-sql-v5` 为 **36/50**，冻结 v4 为 **34/50**。配对结果为 4 gains、2 regressions、
32 both correct、12 both wrong，exact two-sided `p=0.6875`。v5 的 +4 percentage points
不显著。

扩展同时推翻了 Prefix20 的效率信号：v5 legal 为 **49/50**，低于 v4 的 50/50；process
errors 为 10 versus 7；actions 基本持平（263 versus 265）；tokens 增至 984,082 versus
877,576（+12.1%）。因此 v5 不满足继续扩量或训练准入条件。它可以保留为冻结诊断版本，
但不能描述为对 v4 的可靠替代。

## 构造与授权边界

- cohort：`data/eval_inputs/bird_train_baseline300_v1.jsonl`；
- SHA-256：`87b37b307ea2710acdff7d03bdc474a6ba2cdb8ed51b005cff0fe8fa54c67c03`；
- Prefix20 使用前一轮冻结 K=1 结果；本轮只新增 indices 20–49 的 30 条未见任务；
- 每个 task/protocol 只有一次语义轨迹，没有重采样前 20，也没有选择性重跑慢任务；
- 新增 30 条使用与 Prefix20 相同的 Flash、official endpoint、lazy catalog、recent-4、
  30 steps、2,048 tokens、20 preview rows、20 秒 SQLite deadline 与 `bird-set`；
- 外部 API 只接收用户授权的 question、external knowledge、catalog/schema 和逐步只读反馈；
  gold SQL、gold rows 与隐藏验证器数据未进入模型输入。

## 新增 30 题与累计 50 题

| 指标 | 新增 v4 | 新增 v5 | Prefix50 v4 | Prefix50 v5 |
|---|---:|---:|---:|---:|
| correct | 22/30 | 22/30 | 34/50 | **36/50** |
| legal | 30/30 | 29/30 | **50/50** | 49/50 |
| actions | **151** | 160 | 265 | **263** |
| mean actions | 5.03 | 5.33 | 5.30 | **5.26** |
| process errors | **4** | 6 | **7** | 10 |
| total tokens | **494,079** | 634,829 | **877,576** | 984,082 |
| token change | — | +28.5% | — | +12.1% |

累计明细：

| 指标 | v4 | v5 | 变化 |
|---|---:|---:|---:|
| successful `execute_sql` | 208 | 204 | -4 |
| submit attempts | 51 | 49 | -2 |
| prompt tokens | **794,487** | 834,160 | +5.0% |
| completion tokens | **83,089** | 149,922 | +80.4% |
| reasoning tokens | **70,893** | 137,863 | +94.5% |
| API requests | **270** | 276 | +6 |
| completion retries | **5** | 13 | +8 |
| carrier / transport retries | 0 / 0 | 0 / 0 | 0 |
| fresh replay correct | 34 | **36** | +2 |
| structural pass | 50/50 | 50/50 | 0 |

v5 的 204 个成功 query audit 中，`literal_only_select` 全部为 false。硬编码预览值的问题没有
出现，但该局部成功不足以抵消表示回退、非法终止和成本增加。

## 四个 gains

四个 v5-only gains 全部属于 exact output-slot 修复：

- `bird_train_00218`：去掉 movie title/runtime，只返回 genre；
- `bird_train_02757`：top-5 + helper fields 改为 top-1 role name；
- `bird_train_02991`：去掉聚合 count helper，只返回 object id；
- `bird_train_03875`：去掉 method/solution helper ids，只返回 solution path。

这说明 v5 的“最终只保留请求槽位”模块有稳定的正向作用。

## 两个 regressions

两个回退具有完全相同的模式：

- `bird_train_02546` 的 external knowledge 明写 full name refers to `first_name, last_name`；
- `bird_train_04251` 明写 full name refers to `firstName, middleName, lastName`。

冻结 v4 分别返回两列和三列并通过 scorer；v5 将字段拼接成一个字符串。v5 prompt 虽保留了
“mapping to several fields remains several columns”的一般规则，但压缩时删除了 v4 针对
`full name refers to ...` 的具体示例。扩展结果说明抽象规则不足以稳定覆盖该高频表示模式，
这两个回退抵消了新增 30 题上的两个 output-slot gains。

## Legal regression 与成本异常

唯一 legal regression 是 `bird_train_01492`。其问题要求“每年”的 elite-user increment，
external knowledge 同时把范围限定为 2005–2014，却给出包含 `year_id=2015` 的单标量公式；
reference 又使用 2005 作为分母。三个契约彼此不一致。

- v4：7 actions、1 error、4 completion retries、54,225 tokens，最终合法但错误；
- v5：8 actions、3 consecutive argument-validation errors、12 completion retries、139,858
  tokens，达到错误上限后非法终止。

三个 v5 errors 都来自 reasoning completion 达到长度上限后没有可见 JSON action。该任务贡献
了 v5 相对 v4 的 85,633 token 增量。即使排除这一个冲突任务，剩余 Prefix49 的 v5 tokens
仍比 v4 高 2.5%，所以总体成本回退不完全是一个 outlier，但 +12.1% 的主要来源确实是它。

这不是网络 transport failure 或 malformed native tool call：两臂 API carrier/transport
retries 都为 0。问题是面对不可同时满足的语义契约时，v5 reasoning 没有及时形成下一动作；
三个相关 turn 都以 `finish_reason=length` 结束，耗尽 bounded completion retries 后形成
empty-visible-JSON carrier rejection，并最终达到同类错误上限。

## Process errors

Prefix50 累计错误类型：

- v4：2 argument-validation、3 SQLite execution、2 protocol errors；
- v5：3 argument-validation、6 SQLite execution、1 protocol error。

除 `bird_train_01492` 外，v5 的错误都最终恢复为合法提交。v5 没有出现重复成功调用、危险
SQL、hidden verifier feedback 或 literal-only final。

## 决策

1. 停止 v5 在该 cohort 上继续扩到 100/200；准确率增益不显著，legal、errors 和 tokens 均
   未通过行为稳定性门。
2. v5 保持 `diagnostic_only`，不得进入 SFT/RL，也不得用 36/50 宣称替代 v4。
3. Prefix50 已被消费，不再据此修改 v5 后原集合复测。
4. 若设计 v6，最小候选应只恢复具体的 multi-field full-name representation 边界，并增加
   对不可满足 contract 的 bounded decision 行为；必须在与 Prefix50 不重叠的新 slice 上测试。
5. 对 question/external/reference 冲突任务做单独数据质量标记；不能把 verifier failure 直接
   变成教模型忽略 external knowledge 的 SFT 负例。

## 产物

- Prefix20 v4/v5：
  `data/trajectories/iterative_sql_20260805/baseline300_prefix20_v4_flash_r1/` 与
  `baseline300_prefix20_v5_flash_r1/`；
- 新增 30 v4/v5：
  `data/trajectories/iterative_sql_20260805/baseline300_20_50_v4_flash_r1/` 与
  `baseline300_20_50_v5_flash_r1/`；
- machine-readable Prefix50 summary：
  `data/trajectories/iterative_sql_20260805/paired_baseline300_prefix50_v4_v5_summary.json`；
- 四个 arm/batch 的 fresh replay/structural audits 全部通过。
