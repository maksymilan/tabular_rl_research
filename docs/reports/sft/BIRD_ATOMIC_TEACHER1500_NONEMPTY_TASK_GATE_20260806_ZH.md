# BIRD 训练任务非空结果门槛与 Atomic Teacher1500 v2

日期：2026-08-06

## 结论

已把“结果非空”从 trajectory 导出规则提升为 rollout 前的任务级硬门槛。环境私下执行每条
任务的 gold SQL；只有成功返回至少一行的任务才能进入训练任务池。SQL、结果行、结果值和
逐题空/非空标签都不会进入教师上下文或外部 API。

实测没有发现真实零行的 BIRD-train gold denotation：

| 集合 | 输入 | 非空且成功 | 空结果 | 执行错误 | 输出 |
|---|---:|---:|---:|---:|---:|
| normalized train reference | 6,601 | 6,599 | 0 | 2 | 6,599 |
| atomic tool-compatible | 5,915 | 5,915 | 0 | 0 | 5,915 |
| 原 teacher1500 v1 | 1,500 | 1,500 | 0 | 0 | 1,500 |

两条 reference 执行错误按 fail-closed 排除，未被误标为“空结果”。`COUNT(*)=0` 等返回一行
且单元格值为零的结果仍是非空任务。

## 实现边界

- SQLite 以 URI `mode=ro` 打开，并再次设置 `PRAGMA query_only=ON`。
- 使用 SQLGlot 验证 gold SQL 是单个 query。
- 只调用 `fetchone()` 判断是否存在第一行；不持久化 row/value。
- 每条查询有独立执行上限；错误只在 private audit 中记录脱敏 exception class。
- 输入记录中的 `gold_exec_results=[]` 是占位字段，不参与判定。
- private audit 只含 task id、db id、状态和脱敏错误类型，声明
  `teacher_visible=false`；无 gold SQL、gold rows 或结果值。

## 冻结产物

- `data/eval_inputs/bird_train_filtered_nonempty_v1.jsonl`：6,599 条；
  SHA-256 `0e13b0e5ce40f5837672748de86a3ee1468b9ce3c2c144546fb29aa7adf153a6`。
- `data/eval_inputs/bird_train_tool_compatible_nonempty_v1.jsonl`：5,915 条；
  SHA-256 `c8ea3c6419ac57f05021eca470c7519843456e8ac02b04cca9af39e779ec7545`。
- `data/eval_inputs/bird_train_nonempty_v1.private_status.jsonl`：6,601 条 private 状态；
  manifest 为 `data/eval_inputs/bird_train_nonempty_v1.manifest.json`。
- `data/eval_inputs/bird_train_atomic_teacher1500_v2_nonempty.jsonl`：1,500 条；
  SHA-256 `9c4995daf81f94053931fbdaa7bd11beef9157909bacde03be6cb61904952264`。

teacher1500 v2 没有重新抽样。因为 v1 的 1,500 条全部通过认证，v2 保留相同 task ids、
相同顺序以及相同任务文件 hash，只新增对过滤 manifest 的哈希绑定。这样过滤规则不会与
训练分布变化混杂。v2 相对 6,599 条 reference 的数据库 TV 为 `0.00680649`，问题形状 TV
为 `0.00061029`，隐藏 SQL tool proxy TV 为 `0.00017301`；全部 acceptance gates 通过。

## 与 trajectory 空结果规则的关系

任务级门槛解决“gold 答案本身为空，不能成为训练任务”。现有
`causal-empty-result-target-filter-v1` 继续解决 episode 内部的问题：中间空表只保留为因果
反馈、不作为正 target；终局证据表为空则整条 trajectory 排除。两道门槛必须同时通过，
不能互相替代。
