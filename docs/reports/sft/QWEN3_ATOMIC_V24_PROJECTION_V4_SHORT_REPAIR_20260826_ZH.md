# Qwen3 Atomic-v24 projection-v4 短 reasoning 修复

日期：2026-08-26  
状态：全量数据与工程审计完成，但人工语义审查否决；禁止用于训练

## 结论

本轮证明无需重新调用 DeepSeek，也无需丢弃已经付费生成的 causal 轨迹。projection-v4-short 从
projection-v3 的当前监督 target 中抽取一段与当前 action 直接相关的连续原文，保留 system、完整因果
history 和 action JSON 不变。最终保留 26,979/26,987 条记录（99.970%）和全部 3,877 个 episode，
reasoning token 总量由 17,973,269 降到 4,891,487（-72.785%）。

但后续人工审查确认：连续 suffix 虽然满足动作锚点和字节级审计，却经常从原推理中段起步，缺少前文
语义主语或问题背景，不能作为面向 7B 学生的连贯决策说明。因此 projection-v4-short 是失败的
extractive 压缩诊断，训练准入为 0；不得将工程审计通过误写成语义质量通过。

该数据原本试图修复监督 target 过长、重复和解释性文字占比过高的问题。它不改 Harness 交互、不改工具
选择、不改动作参数，也不把 gold SQL、gold rows、未来 tool feedback 或额外模型生成的总结引入训练。

## 问题证据

使用相同 Qwen3 tokenizer 对目标文本复核：

| 数据 | records | reasoning mean | p50 | p90 | p95 | p99 | max | reasoning share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 历史 Atomic-v26 SFT1 | 4,471 | 148.8 | 103 | 296 | 405 | 774.2 | 1,597 | 79.33% |
| projection-v3 | 26,987 | 666.8 | 275 | 1,738 | 2,662.7 | 5,203 | 11,838 | 94.45% |
| projection-v4-short r2 | 26,979 | 181.3 | 232 | 254 | 255 | 256 | 505 | 82.22% |

projection-v3 中 reasoning 至少 512/1,024/2,048/4,096/8,192 tokens 的记录分别为
9,188/5,100/2,139/549/26。修复后大于 512 的记录为 0；1,114 条略高于 soft limit 256，但全部不超过
hard limit 512，原因是保留完整句子和动作锚点，不做机械 token 截断。

此前 checkpoint-250 BIRD-dev 截断快照中的 451 条 truncated episode 全部属于循环：448 条语义或
完全重复动作循环、2 条 alias 枚举循环、1 条零结果循环；没有发现“正常推理进行到一半被截断”的样本。
因此本次优先减少监督中的冗长推理和重复，而不是提高推理 max tokens。

## 修复合同

实现：

- `src/sft/build_checkpoint_relalg_qwen3_projection_v4_short.py`
- `src/sft/audit_checkpoint_relalg_qwen3_projection_v4_short.py`

冻结约束：

1. 输入只允许 projection-v3 canonical/index/manifest，三者 SHA-256 均写入新 manifest；
2. 只允许修改当前 supervised target 的 `<think>` 内容；
3. 新 reasoning 必须是原 reasoning 的一个精确连续 substring，并记录源字符区间与两侧 SHA-256；
4. system、先前 causal messages 和当前 action JSON 必须逐字节不变；
5. 缩短后的 reasoning 必须包含当前 action 的工具、表、列、字面量或规范同义词锚点；
6. soft limit 256、hard limit 512、minimum 4，重复 8-gram ratio 必须小于 0.12，禁止重复完整句；
7. 不调用外部模型，不访问 gold 或未来反馈；不能安全抽取时排除记录，禁止猜测性改写。

第一轮 r1 因 minimum=8 且对较短重复文本直接拒绝，排除了 88 条。修正为 minimum=4，并让短重复文本
也走同一 grounded suffix 提取后，最终 r2 只排除 8 条。这 8 条都无法在 512-token 窗内找到同时满足
动作锚点和非重复条件的连续原文，其中只有 1 条是 answer target。

## 最终数据身份

服务器目录：

```text
/home/dengyan/tabular_rl_outputs/data/
  qwen3_8b_checkpoint_relalg_atomic_v24_sft4_projection_v4_short_r2_20260826/
```

训练视图：

```text
checkpoint_relalg_atomic_v24_union_qwen3_projection_v4_short_r2.jsonl
```

关键 SHA-256：

- source projection-v3 canonical：`63a3f2f46efa312698e336c5d4b64c806f21cc81fe52668c1159666dc789711f`
- source projection-v3 index：`d6d1c9234607ed2adbc57643244df7c13d79fbe8f7e4c03c7cd68d7ba38d3c6c`
- training view：`20ce21d8e7fb14c70b7ec1efd42dbbd0d9c3cd8e9672bbb5286febe3827048f5`
- canonical：`fd8e4d88cc51152c5035d3e54640acef07bc00db96f6b2f9b12d579e6d06f192`
- index：`023182a50152b5134fb4db5fb76e031cec3d5ce348d009b5eecadc15b8c3908d`
- excluded：`c43ccd6d7539b2c25b6bdc18420f42f32e75fed90c1fe7da762bfb2707a05ae3`
- audit：`08a671c05c2c9256d8187ec1efc2b3bbc84fa85bcb9ba88cb87ede38d4024311`
- reasoning stats：`b5a2ba0fdfb321f02a6e9edb1943c32c37cf5f7d69056e9788987e45034ac2b3`

最终选择：

- 26,979 records，14,061 shortened，12,918 unchanged，8 excluded；
- 3,877 selected episodes，3,768 episodes 含 answer target；
- reasoning mean/p50/p90/p95/p99/max = 181.3/232/254/255/256/505；
- action token mean 35.20，和 projection-v3 的 35.21 基本相同；
- target token mean 220.50，reasoning aggregate share 82.22%；
- malformed targets 0。

## 验证

独立全量 audit 通过 26,979/26,979 条，重新读取 source 和 output 后验证：

- source/output 文件 hash；
- selected/excluded 完整且互斥分区；
- record、episode、turn 和 tool identity；
- system/history/action 不变；
- substring span 与 reasoning hash；
- tokenizer token limits、重复率、完整句重复与 action anchor；
- manifest 统计与真实文件重新聚合一致。

本地回归为 21 passed，覆盖正向构建、长且不 grounded 的排除、重复句压缩、空候选 tokenizer、
projection-v3/v2、Qwen3 carrier 与 SFT audit 兼容。

人工抽查覆盖 filter、shape、join、aggregate、scalar、rank、set、inspect、read 和 answer。抽取后的文字仍
解释当前关系、计算或终端 artifact，未出现半句话、动作脱节或新增事实。它仍保留少量自然语言冗余，
但已经消除训练目标中的超长尾；进一步把 soft limit 压到 128/192 会成为新的数据策略实验，不能在未做
matched SFT/eval 前把本版静默改掉。

## 使用边界与下一步

projection-v3 原始数据及既有训练/checkpoint 保持不变，仅作为可追溯 source 和诊断产物。
projection-v4-short 也只保留作失败诊断，不启动 SFT。下一修复必须保留 `<think>`，但把它重建为短而
完整的决策说明：当前可见状态、尚缺的信息或下一目标、选择当前工具的原因；JSON action 仍逐字保留。
不能继续使用无上下文的 extractive suffix，也不能完全删除 think。
