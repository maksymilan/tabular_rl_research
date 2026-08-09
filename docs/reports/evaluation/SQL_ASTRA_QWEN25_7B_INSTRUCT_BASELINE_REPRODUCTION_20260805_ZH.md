# SQL-ASTRA Qwen2.5-7B-Instruct BIRD baseline 复现实验（2026-08-05）

## 结论

在论文公开信息能够支持的最接近单轮设置下，SQL-ASTRA 表 1 中
Qwen2.5-7B-Instruct 的 BIRD `47.5` 分数**未能复现**。

- 论文报告：`729/1534 = 47.52%`（表中四舍五入为 `47.5`）。
- 本次主要 greedy 复现：`637/1534 = 41.53%`。
- 差距：`-92` 题，`-5.99` 个百分点。
- 附录中出现的 `temperature=0.6` 和 `temperature=1.0` 也不能解释差距：它们分别得到
  `39.96%` 和 `41.46%`。

因此，论文当前披露的 prompt、解码和评测信息不足以严格重建 `47.5`。论文分数可以作为
published reference，但不能把它描述成已经由本仓库复现的 baseline。

## 复现边界

论文附录 F 披露了以下输入要素，本次均已实现：

- Qwen2.5-7B-Instruct；
- SQLite DDL；
- BIRD column descriptions；
- 每列两个真实数据库值；
- primary/foreign keys；
- BIRD external knowledge；
- `You are a helpful SQL assistant.` system role；
- step-by-step instruction 和 fenced SQL 输出格式。

附录 F 随后进入最多三次 `run_sql_remote` 的 agent case study，但论文把表 1 的 `47.5`
描述为单轮 greedy baseline。本次主要实验因此停在公开的单轮 fenced-SQL carrier，不引入执行反馈，
避免把 SQL-ASTRA agent 的额外能力混入基础模型 baseline。

仍未公开或存在歧义的变量包括：作者实际使用的完整 baseline prompt、真实值检索/排序实现、
chat template 与推理框架版本、最终 SQL 提取器、BIRD evaluator 版本，以及表 1 的 greedy 描述与
附录 C 中 `temperature=0.6/1.0, top_p=0.95` 之间的对应关系。

## 固定实验设置

除解码参数外，三个 arm 完全一致：

| 项目 | 固定值 |
|---|---|
| 数据集 | BIRD dev 全量 1534 题 |
| 模型 | `/home/dengyan/models/Qwen2.5-7B-Instruct` |
| prompt profile | `sql-astra-disclosed-single-turn-v1` |
| 每列真实值 | 2 |
| 每题采样 | 1 |
| max tokens | 2048 |
| thinking | disabled |
| execution feedback | disabled |
| predicted SQL timeout | 10 秒 |
| denotation scorer | `bird-set` |
| repetition penalty | 1.0 |

受控解码矩阵：

| Arm | Temperature | Top-p | 来源 |
|---|---:|---:|---|
| greedy | 0.0 | 1.0 | 论文表 1 的 greedy 描述 |
| validation-0.6 | 0.6 | 0.95 | 附录 C validation 设置之一 |
| validation-1.0 | 1.0 | 0.95 | 附录 C validation 设置之一 |

## 全量结果

| 设置 | Correct / 1534 | Accuracy | 相对论文正确题差 | 相对论文差值 |
|---|---:|---:|---:|---:|
| SQL-ASTRA 论文 | 729 | 47.52% | 0 | 0.00pp |
| 本次 greedy | 637 | 41.53% | -92 | -5.99pp |
| 本次 T=0.6 | 613 | 39.96% | -116 | -7.56pp |
| 本次 T=1.0 | 636 | 41.46% | -93 | -6.06pp |
| 先前 Appendix-v1 近似 prompt | 679 | 44.26% | -50 | -3.26pp |

先前 `sql-astra-appendix-v1` 使用本仓库加强过的 system instruction 和 `<answer>` carrier，
数值上更接近论文，但它不等于附录公开的原始 role/carrier，所以只能称为 prompt-adapted
approximation，不能称为严格复现。

## Paired 结果

所有 paired 统计都基于同一 1534 题，`p` 为 discordant pairs 的双侧 exact binomial test。

| A vs B | Both correct | A only（gain） | B only（regression） | Both wrong | Exact p |
|---|---:|---:|---:|---:|---:|
| 本次 greedy vs 先前 Appendix-v1 | 484 | 153 | 195 | 702 | 0.0278 |
| T=0.6 vs 本次 greedy | 485 | 128 | 152 | 769 | 0.1692 |
| T=1.0 vs 本次 greedy | 490 | 146 | 147 | 751 | 1.0000 |
| T=1.0 vs T=0.6 | 474 | 162 | 139 | 759 | 0.2047 |

关键观察：`T=1.0` 与 greedy 几乎只是替换了正确题集合（146 gains、147 regressions），总正确数
只差一题。附录温度歧义并不能恢复论文的 92 题差距。

## 产物完整性

| 设置 | 唯一题数 | Fenced outputs | No SQL | Transport/context/incomplete | 非 SELECT/WITH 起始 | Execution error | Wrong result |
|---|---:|---:|---:|---:|---:|---:|---:|
| greedy | 1534 | 1534 | 0 | 0 | 0 | 332 | 565 |
| T=0.6 | 1534 | 1534 | 0 | 0 | 0 | 360 | 561 |
| T=1.0 | 1534 | 1534 | 0 | 0 | 0 | 328 | 570 |

三个 arm 都覆盖题号 0–1533，无重复、无漏题。`execution_error` 是模型生成 SQL 的列名、语法或
执行问题，属于模型结果，不是基础设施丢失。全部 GPU 进程在矩阵完成后已退出。

实验 smoke 初次发现两项基础设施问题：隔离任务文件仍带本机绝对数据库路径，以及旧 SQL parser
会把 fenced SQL 前解释中的 `Select ...` 一并抽取。前者通过服务器 BIRD 镜像修正；后者改为优先
抽取 fenced block，并增加真实输出形态的回归测试。本地和服务器隔离 runtime 均通过 `39/39`
测试后才重跑正式矩阵。无效产物未删除，分别以 `.failed_local_db_path` 和
`.failed_fenced_sql_parser` 后缀保留供审计。

## Baseline 使用建议

1. Related Work 表中保留论文公开的 `47.5`，明确标注为 published score。
2. 本地方法主比较使用同一 harness 下可复现的 baseline：
   - 最忠于当前公开单轮 carrier 的 baseline：`41.53%`；
   - 更强但经过本仓库 prompt adaptation 的保守 baseline：`44.26%`。
3. 如果论文主张超过相关工作，建议同时报告两条本地 baseline，并把 `47.5` 单独列为作者报告值；
   不应直接把 `41.53% -> 方法分数` 与论文 `47.5` 混成一个严格同协议提升。
4. 若必须声称复现 `47.5`，下一步需要作者的 baseline 代码/完整 prompt、value selection、模型服务
   版本和逐题输出；继续猜测 temperature 的价值已经被本实验基本排除。

## 代码与远端产物

- Prompt：`src/eval/direct_sql_prompt.py`
- SQL 提取器：`src/eval/text2sql.py`
- 回归测试：`src/eval/test_eval.py`
- 单 arm launcher：
  `archive/experiments/evaluation/launchers/run_qwen25_7b_sql_astra_disclosed_arm_table_rl.sh`
- 矩阵 launcher：
  `archive/experiments/evaluation/launchers/run_qwen25_7b_sql_astra_disclosed_matrix_table_rl.sh`
- 隔离 runtime：
  `/home/dengyan/tabular_rl_outputs/sql_astra_reproduction_matrix_20260805`
- 正式结果根目录：
  `/home/dengyan/tabular_rl_outputs/evaluations/sql_astra_reproduction_matrix_20260805`
- 日志根目录：
  `/home/dengyan/tabular_rl_outputs/logs/sql_astra_reproduction_matrix_20260805`

