# BIRD action-block v18 优化与 20 题轨迹审计

日期：2026-07-26  
模型：DeepSeek v4 Flash，thinking enabled，reasoning effort high，temperature 0  
指标：`bird-set`  
固定样本：`bird_train_tool_interface_validation200_version4.jsonl` 的前 20 题

## 结论

`action-block-v18` 在本组 20 题上取得 **17/20 = 85%**，是本轮 action-block
实验中的最高单次结果。它相对 v14 为 2 个净恢复、0 个回退；相对最初 v4 为 2
个恢复、1 个回退；相对原子工具 version24 为 1 个恢复、0 个回退。

但这还不能证明稳定的模型能力提升：

- 20 题配对差异均不显著；
- v18 新增的两个确定性归一化规则在这次完整运行中没有实际命中，两个主要恢复
  更像是模型采样路径改善，而不是归一化的直接因果收益；
- v18 相比 v14 增加了 15 个模型轮次和 35.6% token；
- 最难的 `00593` 仍未解决，并出现 1 个 provider carrier 终止失败。

因此，v18 可以作为下一轮扩大评测的候选，但不能据这 20 题宣称 action-block
已经稳定超过原子工具，也不能作为 SFT 数据协议晋级。

## 本轮保留的优化

所有变化只作用于 action-block，原子工具协议、参数、执行环境和提示均未改变。

1. 终端调用是 action block 内唯一的 sink，可以引用同一 block 的结果；终端次序由
   harness 调度。
2. resident 一行表可直接作为 `scalar_compute.value_ref`，harness 自动解析其 producing
   step；模型不维护 step id。
3. `"$resident_handle"` 在表引用位置自动去除无歧义的 `$`。
4. `"$resident_handle.column"` 在 scalar operand 中自动归一化为
   `value_ref + column`，随后仍执行严格的 1×1、列名和非 NULL 校验。
5. 形如 `[predicate, {"op":"and"}, predicate]` 的同一布尔操作中缀列表自动转成
   标准 predicate tree；混合 `and/or` 不自动转换，避免猜测优先级。
6. `date_diff_days`、`subtract`、`percent` 的 operand 顺序在 action-block prompt 中显式
   给出。

v16/v17 尝试加入“不得强制截断多行”和逐行日期表达式提示。它们在 `00593` 上分别
造成 16/17 个 block 的读取循环或错误，未使模型保留两条正确时长，因此已回退，
不属于 v18。

## 固定 20 题结果

| 方案 | 正确 | 合法终止 | 模型轮次 | 原子动作 | 过程错误 | 总 token |
|---|---:|---:|---:|---:|---:|---:|
| 原子 version24 | 16/20 | 20/20 | 127 | 127 | 1 | 690,165 |
| action-block v4 | 16/20 | 20/20 | 95 | — | 9 | 450,896 |
| action-block v12 | 13/20 | 17/20 | 115 | — | 13 | 527,502 |
| action-block v13 | 15/20 | 19/20 | 96 | 163 | 7 | 379,436 |
| action-block v14 | 15/20 | 19/20 | 85 | 151 | 5 | 330,616 |
| **action-block v18** | **17/20** | **19/20** | **100** | **158** | **5** | **448,405** |

v18 的 97 个已解析 action block 中，block 宽度分布为：

- 1 call：60；
- 2 calls：22；
- 3 calls：8；
- 4 calls：7；
- 5 calls：0。

平均宽度为 1.61。19 个终端 block 中有 6 个把终端与同 block 的证据生产调用合并。
这说明模型确实使用了并行/链式 block，但大多数反馈边界仍选择单调用；当前收益不是
来自把每题压成一两个大 batch。

## 逐题配对

| 对照 | 新版独有正确 | 对照独有正确 | 净变化 | 精确双侧 p |
|---|---|---|---:|---:|
| v14 | `04189`, `06489` | 无 | +2 | 0.5 |
| v13 | `05440`, `06489` | 无 | +2 | 0.5 |
| v12 | `00541`, `04189`, `05440`, `06489` | 无 | +4 | 0.125 |
| v4 | `05440`, `06489` | `00593` | +1 | 1.0 |
| 原子 version24 | `05440` | 无 | +1 | 1.0 |

样本太小，任何一组都不能排除随机波动。

## 推理质量审计

### 明确改善

`04189` 正确地：

- 探索 `playstore` 与 `user_reviews` schema；
- 验证 `Category=SPORTS`、`Type=Free`；
- 先固定 free sports app population，再按 `App` join；
- 保留一对多 review 行，并只输出 `App, Translated_Review`。

其推理明确说明“每个 App 可能有多个 review，因此保留 join 的全部行”，最终
denotation 与 gold 一致。v14 在同题把列名 `App` 当成了字面量；v18 没有这个混淆。

`06489` 正确识别 `IMG_OBJ` schema 后，以 `IMG_ID=5, X=634, Y=468` 过滤，并返回
`OBJ_SAMPLE_ID=18`。v14 错把问题理解成对象类别名称 `paper`。v18 的改进来自对问题
输出槽和 schema 的更好对齐，不是接口归一化命中。

### 剩余失败

`00593`：

- v14 因 `"$filter_004.STOP"` 等 resident scalar 写法产生接口错误；
- v15 定向实验已经做到 13/13 工具成功、合法终止，证明接口摩擦可以消除；
- 但模型只取第一条用药记录，输出 11，漏掉第二条 18；
- v16/v17 的通用基数提示没有修复，反而诱发长读取循环；
- v18 完整运行执行 20 个原子动作后出现 provider visible-content 为空，最终非法终止。

根因是工具可表达性与策略共同作用：`scalar_compute` 是 1×1 工具，而题目需要对两行
逐行算日期差。仅靠 prompt 告诉模型使用通用 SQL 表达式没有稳定生效。后续更合理的
设计是给 `project` 增加 typed row-wise expression，例如
`{"op":"date_diff_days","columns":["START","STOP"],"as":"duration_days"}`，而不是继续
增加针对该题的提示。

`01152` 已找到正确的最年轻获奖者 Kyrie Irving，但输出只含 first/last，漏掉
`middleName=Andrew`；这是最终输出槽选择错误。轨迹还先把 `order_by` 写成对象列表，
产生一次可恢复执行错误，随后改为字符串格式。结果行正确，姓名形状不完整。

`06026` 无过程错误，但模型把 South 与 West 的利润相加得到 48.392；gold 只要求
`south_superstore` 中该产品的 distinct Profit 33.8744。这是数据库范围/问题语义
理解错误，接口自动修正不应替模型猜测使用哪个区域表。

## 接口、推理与能力上限

v18 相对 v4 把过程错误从 9 降到 5，总 token 基本持平，准确率单次增加 1 题；相对
原子 version24，模型轮次减少 21.3%，token 减少 35.0%，单次准确率增加 1 题。
这支持 action block 对“减少接口摩擦和模型往返”有价值。

但它没有证明通用推理能力显著上升。v18 有 60/97 个单调用 block，且相对 v14
准确率提高的同时 token 增加 35.6%。平均 provider reasoning 长度约 1,011 字符/有效
轮次，仍明显超过 prompt 所要求的 120 words，说明模型没有稳定遵循短推理约束。
观察到的上限主要来自：

1. 多行逐行计算等工具可表达性缺口；
2. 输出槽和数据范围的语义理解；
3. provider reasoning 过长导致的 carrier 风险；
4. 当工具已给出充分事实后，模型仍可能重复读取而不收敛。

## 下一步门槛

1. 保留 v18 为当前 action-block 候选，不采纳 v16/v17。
2. 不再针对三条失败追加题目特定 prompt。
3. 若继续改工具，优先做 action-block 专属的 typed row-wise `project` expression，
   以及无歧义的 `order_by` 对象到字符串归一化；原子工具保持不变。
4. 在固定 200 题上验证前，先用包含多行日期/数值计算的独立小门控证明 typed
   expression 的普适收益。
5. 固定 200 题必须分别报告 `bird-set` 正确率、合法终止、过程错误、carrier/transport
   失败、token，以及与 version24 的逐题配对；20 题结果不能作为 SFT 晋级依据。

## 产物

- v18 完整轨迹：
  `data/trajectories/batch_plan_20260726/action_block_v18_interface_final_pilot20_r1.all.jsonl`
- v18 manifest：
  `data/trajectories/batch_plan_20260726/action_block_v18_interface_final_pilot20_r1.all.manifest.json`
- v18 失败审计：
  `data/trajectories/batch_plan_20260726/action_block_v18_interface_final_pilot20_r1.audit.jsonl`
- 配对结果：
  `action_block_v18_vs_{v14,v13,v12,v4,version24}_pilot20.paired.json`

最终回归：

- harness：91/91；
- SFT：122/122；
- RL：87/87（9 skipped）；
- atomic eval：28/28；
- action-block eval：37/37。
