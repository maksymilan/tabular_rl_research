# BIRD action-block v21 固定 200 题评估

日期：2026-07-26  
模型：DeepSeek v4 Flash  
指标：`bird-set`  
协议：`action-block-v21`  
结论：**未通过 150/200 SFT 数据构造门槛，不构造 SFT 数据。**

## 冻结配置

本次运行使用与 v21 fixed-40 完全相同的可见协议和运行参数：

- 固定数据：
  `data/eval_inputs/bird_train_tool_interface_validation200_version4.jsonl`
- 固定前 200 题；
- 每个 action block 最多 5 个原子调用；
- rolling history：4；
- 最大原子动作：30；
- 最大模型轮次：30；
- DeepSeek thinking enabled；
- reasoning effort：high；
- temperature：0；
- provider JSON Output；
- `bird-set` denotation；
- gold SQL 对模型不可见；
- system prompt SHA256：
  `be6f8d2c898c7cb973d885ece19a1f29dc0c30a496aa06e43e231d92c7238b02`；
- protocol hash：`188c8a1d98adb16c`。

运行前本地回归全部通过：

- harness：91/91；
- SFT：122/122；
- RL：87/87，9 skip；
- atomic eval：28/28；
- action-block：42/42。

未生成、导出或装配任何 SFT 数据。

## 最终结果

| 指标 | action-block v21 | atomic version24 | 差值 |
|---|---:|---:|---:|
| 正确 | **136/200（68.0%）** | 145/200（72.5%） | -9 |
| 合法终止 | **194/200（97.0%）** | 197/200（98.5%） | -3 |
| 过程错误 | **63** | 29 | +34 |
| 模型轮次 | **1,092** | 1,490 | -398（-26.7%） |
| API 请求 | **1,149** | 1,509 | -360（-23.9%） |
| 总 token | **4,982,816** | 8,420,861 | -3,438,045（-40.8%） |

action-block 明显降低模型往返与总上下文成本，但正确率、合法率和过程错误均不如 atomic
version24。它没有通过 75% 工具可用性门槛。

## 配对结果

同一 200 题逐题配对：

- 两者都正确：126；
- 两者都错误：45；
- 仅 action-block v21 正确：10；
- 仅 atomic version24 正确：19；
- 净变化：-9；
- 双侧 exact paired `p=0.1360`。

总体差异未达到 0.05 显著性，但没有任何证据支持 v21 优于 atomic；方向和点估计均为退化。

v21-only：

`02512`、`02868`、`03042`、`03262`、`03636`、`03724`、`03969`、`03977`、
`05316`、`05440`。

atomic-only：

`00454`、`00833`、`01050`、`01148`、`01290`、`01297`、`01425`、`01599`、
`01926`、`02734`、`02901`、`02918`、`03373`、`04244`、`04290`、`04906`、
`05053`、`05760`、`06489`。

128/200 题触发了至少一次 action-block 接口归一化。该子集为 5 gains / 15
regressions，`p=0.0414`；未触发归一化的 72 题为 5 gains / 4 regressions。这个分组是按
v21 运行后的行为定义，困难题更容易触发归一化，因此不能据此推断归一化本身造成退化；
但它说明当前接口摩擦集中在模型已经难以处理的题上。

## action block 的实际使用

- action blocks：1,079；
- 原子动作：1,856；
- 提交调用：1,871；
- 平均每 block 提交调用：1.734；
- block size 分布：
  - 1 调用：576；
  - 2 调用：313；
  - 3 调用：115；
  - 4 调用：54；
  - 5 调用：22。

action-block 将 1,856 个原子动作压缩到 1,079 个反馈边界，因此显著减少模型轮次；但
模型执行的原子动作比 atomic 的 1,490 个更多。当前效率收益来自批量执行和更短上下文，
不是更短的关系推导。

## 接口归一化

共发生 367 次确定性解析：

| 规则 | 次数 |
|---|---:|
| prior-block local-id → resident handle | 186 |
| 去除 resident handle 的 `$` | 82 |
| 唯一列后缀解析 | 69 |
| resident handle → producing step | 18 |
| 已验证单单元 predicate `value_ref` | 9 |
| resident `handle.column` → producing step | 2 |
| 已验证跨结果 `column_value` → `value_ref` | 1 |

v21 新增的跨结果单单元规则在 `05097` 被实际触发，任务正确合法。未再次观察到
`column_value` 被静默改写成当前表自比较。

## 错误与恢复

总过程错误 63 次：

- argument validation：24；
- execution：23；
- protocol：16；
- blocked descendants：25；
- 46/200 题至少发生一次过程错误；
- 10/200 题至少发生一次依赖阻塞；
- 发生过程错误的题中有 26 题最终正确。

这说明 bounded recovery 能够工作，但接口摩擦仍明显高于 atomic。

高频摩擦包括：

- `read_subtable.limit` 超出 1..20：5 次；
- malformed action-block call shape/size：7 次；
- `group_aggregate` columns layout 参数不完整：3 次；
- `join_tables.on.right` 非新表裸列：3 次；
- `condition_filter` 将 `and` 放在错误层级：3 次；
- 非标量结果被当作 `value_ref`；
- terminal 引用了 step id 或 `read_subtable` 调用，而不是 resident derived handle。

这些问题大多可以通过未来的 action-block-only adapter 继续降低，但本次冻结协议没有在
运行中修改。

## 64 道失败的边界

最终失败构成：

- 58 道合法 `wrong_answer`；
- 2 道 provider-carrier failure；
- 1 道 API transport failure；
- 1 道 argument-validation error-budget exhaustion；
- 1 道 execution error-budget exhaustion；
- 1 道 max-atomic-actions。

58 道错误答案中：

- **42 道完全没有过程错误**；
- 16 道经历错误后合法终止，但答案仍错。

因此 72.4% 的错误答案不是接口拒绝导致，而是模型稳定执行了错误的关系语义。继续增加
接口归一化最多改善合法率、错误数和成本，不能将当前 136/200 推到 150/200。

### 六道非合法/传输失败

| 题目 | 类型 | 说明 |
|---|---|---|
| `00593` | API error | 第 14 轮远端断开连接；单列为 transport failure。 |
| `05668` | argument validation | 反复把 join `on.right` 写成带命名空间列，耗尽错误预算；4 错误、6 阻塞。 |
| `01926` | execution | 反复把多行 person 结果当标量引用，耗尽错误预算。 |
| `00582` | provider carrier | 三次请求后 visible content 仍为空。 |
| `01461` | provider carrier | 三次请求后 visible content 仍为空。 |
| `06299` | max actions | 23 轮、30 个原子动作后仍未终止。 |

provider 侧共记录 2 次 transport retry、34 次 completion retry、22 次 carrier retry；
最终仍有 1 个 API failure 和 2 个 carrier failure。provider 没有返回可审计的 system
fingerprint。

## 是否适合构造 SFT 数据

本次结果不批准构造 action-block v21 SFT 数据：

1. 136/200 明确低于预先要求的 150/200；
2. 相比 atomic version24 净退化 9 题；
3. 合法终止仅 97%；
4. 过程错误为 atomic 的两倍以上；
5. 42 道错误答案是无错误的完整合法轨迹，若只按合法性筛选会把错误语义带入教师数据。

action-block 的执行效率和恢复机制有价值，但当前 DeepSeek v4 Flash 教师没有表现出足够
稳定的语义质量。建议保持 v21 为 diagnostic-only，不进入 SFT 数据构造；是否继续做
action-block-only 的机械接口 v22，应由下一步研究决策单独批准。

## 产物

- 完整轨迹：
  `data/trajectories/batch_plan_20260726/action_block_v21_fixed200_r1.all.jsonl`
- manifest：
  `data/trajectories/batch_plan_20260726/action_block_v21_fixed200_r1.all.manifest.json`
- 64 道失败审计：
  `data/trajectories/batch_plan_20260726/action_block_v21_fixed200_r1.audit.jsonl`
- 对 atomic version24 的 paired comparison：
  `data/trajectories/batch_plan_20260726/action_block_v21_vs_version24_fixed200.paired.json`

