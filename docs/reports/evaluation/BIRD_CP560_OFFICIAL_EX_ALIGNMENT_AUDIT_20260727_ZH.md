# checkpoint-560 BIRD official EX 对齐审计

日期：2026-07-27

## 结论

当前 `bird-set` 的底层比较器已经实现 BIRD reference EX 的核心语义：

```python
set(predicted_rows) == set(gold_rows)
```

它忽略行顺序和重复行数，但不忽略列顺序，也不做数值或字符串归一化。

审计发现工具终止评分在精确比较失败后仍保留了一个历史兼容分支：对同宽证据表尝试
列排列。该分支与模型可见的“精确行、列和列顺序”契约冲突，也不属于 BIRD EX。当前
评分已删除这一主动容错，并以 `terminal_answer_contract=exact-cited-table-v1` 记录在
新评测的 manifest 和记录中。历史显式 `answer` 字段只保留为退休协议的 replay
兼容路径；当前 `answer_from_context` 不允许模型撰写答案值。

## checkpoint-560 重评分

重评分不调用模型。它按原 action 顺序重新执行每条历史正确轨迹，然后只用终止调用
引用的证据表按 BIRD set equality 比较 gold SQL 结果。

### Greedy

| 指标 | 历史 scorer | 严格 official EX | 变化 |
| --- | ---: | ---: | ---: |
| Pass@1 | 734/1534 = 47.85% | **734/1534 = 47.85%** | 0 |

734 条历史正确轨迹全部通过；没有 replay error，也没有依赖列排列的样本。

### Sampling K=4

| 指标 | 历史 scorer | 严格 official EX | 变化 |
| --- | ---: | ---: | ---: |
| Pass@1 | 779/1534 = 50.78% | **778/1534 = 50.72%** | -1 |
| Pass@2 | 924/1534 = 60.23% | **924/1534 = 60.23%** | 0 |
| Pass@4 | 1035/1534 = 67.47% | **1033/1534 = 67.34%** | -2 |

共有九个正确 sample 只通过历史列排列分支；因为同一题的其他 sample 仍可能正确，
最终 Pass@4 只撤销两题：

- example 37：预测
  `[Street, City, Zip, State]`，gold 为 `[Street, City, State, Zip]`；
- example 81：预测
  `[City, School, grade]`，gold 为 `[City, grade, School]`。

example 58 的 sample 0 也因列顺序撤销，使 Pass@1 减一，但后续 sample 正确，所以
Pass@2/4 保持正确。

## 与 Direct SQL 对照的更新

Direct SQL 本来就直接比较预测 SQL 的原始 `fetchall()` tuple，没有列排列兼容。
其完成的 K=4 control 仍为 727/820/904。严格对齐后的工具净领先改为：

| 指标 | 工具 checkpoint-560 | Direct SQL base | 工具净领先 |
| --- | ---: | ---: | ---: |
| sampled Pass@1 | 778/1534 = 50.72% | 727/1534 = 47.39% | +51 / +3.32 pp |
| Pass@2 | 924/1534 = 60.23% | 820/1534 = 53.46% | +104 / +6.78 pp |
| Pass@4 | 1033/1534 = 67.34% | 904/1534 = 58.93% | +129 / +8.41 pp |

Pass@k 仍是使用隐藏 gold 判定“前 k 条中是否至少一条正确”的 oracle coverage，
不能替代多数投票、PRM 或其他无需 gold 的候选选择结果。

## 超时边界

- 当前 Direct SQL control 使用 20 秒生成 SQL 执行超时。
- 完成的 Direct SQL greedy artifact 没有 `interrupted`/timeout 边界样本，因此本次
  20 秒与 BIRD/论文脚本常见的 10 秒或 30 秒差异没有改变分数。
- K=4 工具 artifact 对每个工具/终止 SQLite 操作使用 20 秒超时。多轮工具 agent
  没有一条可直接提交给官方 scorer 的单次预测 SQL deadline，因此论文对比必须同时
  报告其多轮工具预算，不能把超时条件隐藏在 EX 名称中。

## 实现与产物

- 活跃终止评分：`src/eval/rollout.py`
- atomic K 采样 manifest：`src/eval/rollout_passk.py`
- action-block manifest：`src/eval/evaluate_batch_plan.py`
- 历史 artifact 重评分：`src/eval/rescore_tool_artifact_bird_ex.py`
- greedy audit：
  `data/results/qwen25_coder7b_raw_json_cp560_tool_dev1534_greedy1_bird_official_ex_rescore/`
- K=4 audit：
  `data/results/qwen25_coder7b_raw_json_cp560_tool_dev1534_passk4_bird_official_ex_rescore/`

回归验证：

- `src/eval/test_eval.py`：30/30；
- `src/eval/test_batch_plan.py`：44/44；
- greedy replay：1534/1534 records audited，0 replay errors；
- K=4 replay：1534/1534 records audited，0 replay errors。

Reference evaluator:

- <https://github.com/bird-bench/mini_dev/blob/b3d4bcbbae9a96934ad812551eb400c7a3b23c12/evaluation/evaluation_ex.py>
- <https://github.com/RUCKBReasoning/OmniSQL/blob/a66d732010c89fac4353488d1c59e0b06f92b742/train_and_evaluate/evaluate_bird.py>
