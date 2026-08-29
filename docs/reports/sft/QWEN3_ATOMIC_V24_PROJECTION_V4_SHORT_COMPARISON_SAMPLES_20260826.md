# Projection-v4-short 压缩前后实物样本索引

日期：2026-08-26  
完整样本：`QWEN3_ATOMIC_V24_PROJECTION_V4_SHORT_COMPARISON_SAMPLES_20260826.jsonl`  
完整样本 SHA-256：`93ba60ca3c16e0e4f5d9c6aa67f39e10edfff430103c3ab16fb4a83e2c6d656c`

## 文件结构

JSONL 共 5 行，每行都是一个完整样本，未使用省略号。每行包含：

- `sample_role`、`record_id`、`tool_name`；
- 压缩前后的真实 tokenizer token 数；
- `source_character_span` 和完整 projection metadata；
- `source_target_full`：压缩前完整 `<think> + JSON action`；
- `projected_target_full`：压缩后完整 `<think> + JSON action`；
- `source_canonical_row_full`：压缩前完整 canonical SFT row，包括 system 和 causal conversations；
- `projected_canonical_row_full`：压缩后完整 canonical SFT row。

因此既可以只比较本轮完整 target，也可以比较整个训练样本，确认 system、history 和 JSON action 没变。

## 五个样本

| 角色 | record_id | tool | 压缩前 reasoning tokens | 压缩后 reasoning tokens | 说明 |
|---|---|---|---:|---:|---|
| extreme_long | `bird_train_02170_turn_02` | `filter_rows` | 11,838 | 246 | 全数据最长 reasoning，验证极端长尾 |
| answer_long | `spider_train_06013_turn_05` | `answer` | 7,715 | 249 | 超长终端决策 |
| join_mid | `bird_train_01267_turn_09` | `join` | 1,798 | 217 | 中长关系连接决策 |
| scalar_mid | `bird_train_03603_turn_12` | `scalar_compute` | 1,201 | 256 | 中长标量计算决策 |
| unchanged_control | `spider_train_00724_turn_05` | `answer` | 150 | 150 | 合法短 reasoning 原样保留 |

选择只依据 tool 和 reasoning 长度，不读取 gold SQL、gold rows 或后续反馈。

## 阅读建议

先比较同一行的 `source_target_full` 与 `projected_target_full`：JSON action 应完全相同；压缩后的
reasoning 应当能在压缩前 reasoning 的 `source_character_span` 处逐字找到。随后比较两个
`*_canonical_row_full`：除最后 assistant target 的 `<think>` 内容和新增审计 metadata 外，system、
此前 conversations 与 action 均应相同。

该文件特意包含两个极端长样本，因此不把约 84,000 字符的全文再次复制到 Markdown；JSONL 是完整、
可哈希和可程序校验的原始对比材料。
