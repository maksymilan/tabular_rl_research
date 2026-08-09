# Qwen3-8B atomic version26 SFT1 baseline 结果

日期：2026-08-08

## 结论

在完全相同的 BIRD-dev 1,534 题、Qwen3-8B revision、atomic version26 工具协议和 greedy 解码合同下，外部教师 SFT1 的 checkpoint-560 QLoRA 将执行准确率从 **270/1534（17.60%）** 提升到 **838/1534（54.63%）**。

逐题配对共有 600 个提升、32 个回退，净增 568 题，即 **+37.03 个百分点**；双侧 exact McNemar 检验为 `p=8.50977e-137`。这不是小样本波动。

54.63% 在绝对数值上同时超过：

- 本地已完成的作者 SQL-tool 协议 Qwen3-8B raw 复现：709/1534（46.22%），多 129 题、+8.41 个百分点；
- 论文报告的 Qwen3-8B raw baseline 47.9%：在 1,534 题上至少对应 735 题，本结果多 103 题、+6.71 个百分点。

因此，这次训练已经建立了当前原子工具协议上的可信 Qwen3-8B SFT 起点。不过它不能被写成“精确复现 TRUST-SQL baseline”：论文的 SQL exploration/proposal 工具、上下文 transcript 和这里的 resident-state 原子关系工具并不相同。

## 冻结实验合同

训练：

- 底模：`Qwen/Qwen3-8B`，revision `b968826d9c46dd6066d109eabc6255188de91218`；
- 数据：外部教师生成并经 fresh replay、执行验证和 no-leak gate 接纳的 678 个完整 episode，展开为 4,471 个 assistant target；
- Qwen3 历史投影：4,471/4,471 条 Hugging Face 与 LLaMA-Factory 前缀 token 完全一致，final target 无改写，最长序列 5,738 tokens，小于 cutoff 6,400；
- QLoRA：4-bit、all-linear、rank 16、alpha 32、dropout 0.05；可训练 43,646,976 / 8,234,382,336 参数（0.5301%）；
- 两张 3090，per-device batch 1、gradient accumulation 8、global batch 16、2 epochs、560 optimizer steps、seed/data_seed 42；
- 训练成功结束于 global step 560，耗时 16,363.93 秒（4:32:43），最终 train loss 0.53854；
- 被评测权重固定为 `checkpoint-560`，`adapter_model.safetensors` SHA-256 为 `3ecbbe3dbb65bb26d0308b09d20496c0023b3090ecc44a36c98bb51024efbab5`。

评测：

- BIRD-dev 2024-06-27，1,534 个唯一题号；
- base 和 adapter 使用同一 atomic version26 `think-json-v1` 合同、同一 system prompt、同一 Qwen3 thinking chat template；
- greedy `n=1`、`temperature=0`、`top_p=1`、`max_steps=30`、`max_tokens=2048`、`bird-set` denotation；
- 两臂均使用单卡 vLLM、相同 4 路 episode 并发配置；
- 全量结果中未发现 `api_error`、`ChatAPIError`、连接拒绝或断线污染；先前 tunnel 失败产物没有 resume 或并入正式结果。

## 全量结果

| 运行 | EX（Wilson 95% CI） | Legal termination（Wilson 95% CI） | Process errors | Mean steps |
|---|---:|---:|---:|---:|
| Qwen3-8B base | 270/1534，17.60%（15.78%–19.59%） | 586/1534，38.20%（35.80%–40.66%） | 3,983 | 5.346 |
| SFT1 QLoRA checkpoint-560 | 838/1534，54.63%（52.13%–57.11%） | 1317/1534，85.85%（84.02%–87.51%） | 1,179 | 7.864 |

逐题正确性配对：

| Base | Adapter | 题数 |
|---|---|---:|
| 正确 | 正确 | 238 |
| 错误 | 正确 | 600 |
| 正确 | 错误 | 32 |
| 错误 | 错误 | 664 |

合法终止由 586/1534 提升到 1317/1534，逐题为 781 个提升、50 个回退，净增 731 题（+47.65 个百分点），双侧 exact McNemar `p=1.04093e-169`。过程错误减少 2,804 个。SFT 后平均步数上升，不代表效率变差：未适配的 base 经常因连续协议或参数错误提前终止，因而以更短但无效的轨迹结束。

SFT1 失败类型仍包括 479 个 wrong answer、131 个 protocol error、47 个 execution error、24 个 argument validation error、14 个 max-steps 和 1 个 context overflow。54.63% 是一个可用起点，不是问题已经解决。

## 可复现性与边界

- 两臂的 1,534 个 `db_id`、question 和 gold SQL 哈希逐题一致；原始/派生评测数据及 11 个 SQLite 数据库均由 gate 锁定。
- base 和 adapter 的 runtime gate 完全相同；adapter gate 固定 checkpoint 名、global step 和三项 adapter 文件哈希。
- 最终归档重新计算了五个底模 shard、训练数据、训练日志、checkpoint、环境和两臂结果哈希，`status=ok`、0 errors、0 incomplete；归档 payload SHA-256 为 `3ee333384f6aedfa92ef1507cfcc9239eb485e1b62b68be314c912eeb8c503b8`。
- 这是单个训练 seed。题级配对统计证明这一次训练的增益，不覆盖训练 seed 方差。
- 这是冻结的历史 atomic version26 checkpoint-560 控制实验，不是当前 version54 `native-tool-bundle` 的开发起点，也没有复现 TRUST-SQL 的 RL 阶段。

## 产物

- Base 正式结果：`data/results/qwen3_8b_base_atomic_v26_bird_dev1534_greedy1_bird_set/`
- Adapter 正式结果：`data/results/qwen3_8b_sft1_qlora_atomic_v26_bird_dev1534_greedy1_bird_set/`
- 最终归档：`data/results/qwen3_8b_atomic_v26_sft1_artifact_manifest.json`
- 严格配对分析器：`reproductions/trust_sql/qwen3_8b_atomic_sft1/analyze_paired_bird_dev.py`
- 运行与边界说明：`reproductions/trust_sql/qwen3_8b_atomic_sft1/README.md`

正式 `all.jsonl` SHA-256：

- base：`5434723ec50800dbf112dc81e94c0f342b466bd8150576c09924ec44ec5d3c16`；
- adapter：`1f4035e0f7fc6c767f7e1be5c1b6c480abddf7a2296a27cd92190926796e1bcb`。
