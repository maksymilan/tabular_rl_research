# BIRD action-block v21 固定 40 题语义审计

日期：2026-07-26  
模型：DeepSeek v4 Flash  
指标：`bird-set`  
数据：`bird_train_tool_interface_validation200_version4.jsonl` 的固定前 40 题  
协议：每个 `action_block` 最多 5 个原子调用，rolling history=4，thinking enabled，
reasoning effort=high，temperature=0，gold SQL 对模型不可见。

## 结论

扩大到 40 题后，action-block 暴露了新的语义错误，但没有证据表明增加通用 prompt
约束能够稳定修复它们。v19 的“终答前逐项核对”提示在定向样本上偶尔修复了遗漏条件，
但完整 40 题为 31/40，较 v18 的组合 40 题少 2 题，且没有 paired gain，因此已撤回。

本轮真正发现并修复的是一个 action-block 独有的接口语义漏洞：跨结果
`column_value:"$call.column"` 曾被静默解析成当前表内的同名列，可能把谓词改成
`status_id = status_id`。v21 现在只在来源被 harness 验证为严格 1 行×1 列时，将其转换
为 grounded `value_ref`；其他情况返回可恢复错误，不再静默改义。该修改不影响 atomic
工具方案。

v21 的最终单次固定 40 题结果是 **31/40 = 77.5%**，合法终止 **40/40**。它没有在该
小样本上提高准确率，但消除了已经观察到的危险静默改义，并保留了 action-block 的轮次
和 token 优势。因此 v21 是工程语义优化，不是准确率 promotion，仍不能作为 SFT 数据源。

## 主要结果

| 方案 | 正确 | 合法终止 | 过程错误 | 阻塞调用 | 模型轮次 | 总 token |
|---|---:|---:|---:|---:|---:|---:|
| action-block v18，两个 20 题运行合并 | 33/40 | 38/40 | 10 | 13 | 183 | 803,302 |
| action-block v19，新增终答核对 prompt | 31/40 | 40/40 | 2 | 0 | 168 | 677,238 |
| action-block v20，撤回 prompt、保留接口归一化 | 31/40 | 40/40 | 9 | 0 | 190 | 793,222 |
| **action-block v21，修复跨结果静默改义** | **31/40** | **40/40** | **7** | **1** | **181** | **734,815** |
| atomic version24，同一固定 40 题 | 33/40 | 40/40 | 2 | 不适用 | 240 | 1,301,937 |

v21 对 atomic version24：

- 正确率少 2 题，paired 为 1 gain / 3 regressions，双侧精确检验 `p=0.625`；
- 合法终止相同；
- 模型轮次减少 59，下降 24.6%；
- 总 token 减少 567,122，下降 43.6%；
- 过程错误多 5 个，说明接口仍有可继续压缩的摩擦，但这些错误均未造成非法终止。

v21 对 v18 组合运行：

- 正确率少 2 题，差异不显著；
- 合法终止增加 2 题；
- 过程错误从 10 降到 7，阻塞调用从 13 降到 1；
- 总 token 下降 8.5%。

v18 是两个不同时段的 20 题运行合并；v20/v21 是完整 40 题重跑。DeepSeek v4 Flash
没有提供可审计的 provider fingerprint，同一 prompt、temperature=0 的多次运行仍出现
答案波动。因此 1–2 题差异不能仅凭单次运行归因于代码或 prompt。

## 扩展的后 20 题暴露了什么

v18 在新增的索引 20–39 上得到 16/20，四道失败如下。

| 题目 | 原因 | 处理 |
|---|---|---|
| `02901` | 思考中明确提到 `Gender='M'`，但实际过滤只执行了婚姻和职位条件。 | 尝试过通用终答核对 prompt；定向运行偶尔恢复，完整运行不稳定，故不保留。 |
| `02925` | 模型返回可读产品名，gold 要求 `ProductID=873`；题目只说 “product”。 | 判为答案实体表示歧义，不添加“product 一律输出 ID”的补丁式规则。 |
| `03688` | v18 对最大值的三种等价引用均被接口拒绝，产生 3 个错误和 9 个阻塞；修复后返回最长影片的全部 5 个库存副本，而 gold 任意取第一行。 | 等价标量引用已由 harness 归一化；接口错误降为 0。剩余 tie/`LIMIT 1` 属于 benchmark 歧义。 |
| `06492` | 外部知识明确说每张图 `COUNT(OBJ_SAMPLE_ID)<15`，模型按图分组后统计；gold 却执行 `WHERE OBJ_SAMPLE_ID<15` 后数行。 | 判为 external-knowledge/gold 冲突，不改 prompt。 |

## v19 prompt ablation 为什么被拒绝

v19 只增加一条通用要求：终答前确认题目和外部知识中的过滤、排序、计数粒度和输出槽
都出现在已执行关系中。它在 `02901` 的一次定向运行中有效，但完整 40 题结果为：

- 31/40，较 v18 的 33/40 少 2；
- paired 没有新增正确题；
- `06454`、`06489` 从 v18 正确变错。

这说明“模型在思考中再次声明约束”不等于“关系操作必然落实约束”。继续增加类似 prompt
会增加 SFT 协议负担，却没有稳定能力收益。v20/v21 的可见 system prompt 已逐字恢复为
v18 prompt；接口修复留在 harness。

## v20 发现的静默改义与 v21 修复

`02437` 要求找所有取消订单。v20 的模型调用等价于：

```json
{
  "table": "order_history",
  "conditions": {
    "column": "status_id",
    "op": "=",
    "column_value": "$cancel_status.status_id"
  }
}
```

`column_value` 的原语义是“与同一输入表的另一列比较”。旧 adapter 将局部列引用解析为
裸 `status_id`，实际执行成 `order_history.status_id = order_history.status_id`，返回全部
7,548 个订单。这是接口层不能接受的静默语义变化。

v21 的处理规则：

1. harness 查验被引用调用已经成功；
2. 查验其输出严格为 1 行×1 列；
3. 查验引用列名与唯一输出列一致；
4. 满足以上事实时转换为 producing-step `value_ref`；
5. 否则拒绝该调用并提供可恢复错误。

新增回归测试同时覆盖安全转换和多行来源拒绝。`02437` 的 v21 定向门和完整 40 题运行
均正确、合法、0 错误。该规则只存在于 action-block adapter，atomic 协议不变。

## v21 最终 9 道失败逐题分类

### 明确的模型语义/关系执行错误

| 题目 | 轨迹问题 |
|---|---|
| `01152` | 正确找到 Kyrie Irving，但漏掉 gold 要求的 middle name `Andrew`。 |
| `02901` | 思考要求 male，执行过滤再次漏掉 `Gender='M'`；一次错误恢复后仍沿用缺条件的人群。 |
| `05440` | 先在 `Paper` 全表取最大年份，选到没有 Journal 的异常记录；正确关系应先 inner join Journal，再在可回答人口上排序。终答还漏掉 homepage。 |
| `06454` | 已得到 RAIL 和 MAIL 两个计数，但终答返回两个计数，没有执行 argmax 并输出单个 ship mode。 |
| `06489` | 将 “object” 解释为类别名 `paper`，而 gold 要 `OBJ_SAMPLE_ID=18`。 |

`02901` 和 `05440` 已被当前全局规则覆盖：prompt 已要求保留显式过滤条件，并要求在相关
表形成目标人口后再排序/聚合。模型仍未稳定执行，说明这不是缺少一句规则，而是当前模型
的关系推理上限。

### benchmark/gold 歧义或冲突

| 题目 | 冲突 |
|---|---|
| `02925` | 题目说 product，模型输出人类可读名称，gold 输出 ProductID。 |
| `03688` | 最长影片有 5 个库存副本；模型全部返回，gold 的无 tie-break `LIMIT 1` 任取一行。 |
| `06026` | 同名同 Product ID 同时存在 West 和 South；题目未指定区域。模型按观察到的 West 记录聚合，gold 固定查询 South。 |
| `06492` | 外部知识要求按图像的对象样本数分组，gold 实际按 `OBJ_SAMPLE_ID<15` 过滤行。 |

对这些题增加全局 prompt 会把 benchmark 的偶然选择编码成协议，容易伤害其他题，因此
不做修改。

## 推理质量观察

action-block 确实减少了环境往返，但没有显示出稳定的语义能力上升：

- 模型经常在 think 中写出正确约束，却在同一个或后续 block 中漏掉；
- 模型可以利用同 block 数据依赖，接口修复后也能从错误中恢复；
- 主要剩余错误发生在答案实体、人口范围、tie、输出粒度，而不是 DAG 调度；
- 40 题上的准确率与 atomic 基线接近但没有超过，效率优势明显。

因此当前证据支持的结论是：action-block 提升了执行效率和接口容错上限，但尚未提升
DeepSeek v4 Flash 的关系语义能力上限。下一步若继续验证，应使用冻结更大 cohort 和
重复运行/配对统计，而不是继续堆通用 prompt。

## 产物

- v18 后 20 题：
  `data/trajectories/batch_plan_20260726/action_block_v18_expansion_next20_r1.all.jsonl`
- v19 prompt ablation：
  `data/trajectories/batch_plan_20260726/action_block_v19_optimized_fixed40_r1.all.jsonl`
- v20 40 题：
  `data/trajectories/batch_plan_20260726/action_block_v20_interface_fixed40_r2.all.jsonl`
- v21 定向门：
  `data/trajectories/batch_plan_20260726/action_block_v21_target_gate_02437_03688.all.jsonl`
- v21 最终 40 题：
  `data/trajectories/batch_plan_20260726/action_block_v21_fixed40_r1.all.jsonl`
- v21 失败审计：
  `data/trajectories/batch_plan_20260726/action_block_v21_fixed40_r1.audit.jsonl`
- v21 对 v18：
  `data/trajectories/batch_plan_20260726/action_block_v21_vs_v18_fixed40.paired.json`
- v21 对 atomic version24：
  `data/trajectories/batch_plan_20260726/action_block_v21_vs_version24_fixed40.paired.json`

第一次 v20 运行因本地 sandbox 禁止网络，40/40 均为首请求 `api_error`，没有任何模型
动作或工具执行，文件
`action_block_v20_interface_fixed40_r1.all.jsonl` 仅保留作传输故障审计，不参与能力统计。

