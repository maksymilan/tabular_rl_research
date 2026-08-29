# Qwen3 reasoning 完整 episode 手工清洗输入

这三条均来自原始 causal rollout `all.jsonl`，按完整 episode 导出；包括失败 action、每轮完整 model input、原 reasoning、固定 action 和 Harness result。

- `spider_train_00658.md`：5 turns，短轨迹。
- `bird_train_00081.md`：7 turns，包含一次 invalid_arguments 及下一轮修复。
- `bird_train_00686.md`：24 turns，长计算轨迹。
- `original_full_episodes.jsonl`：原始 all.jsonl 三条记录，未经修改。
- `manual_cleaning_template.jsonl`：逐 turn 结构化模板，`manual_cleaned_reasoning` 为空。

不得修改 action/result，不得引入 gold 或未来反馈；同一 episode 的术语、artifact 名和阶段计划应保持一致。
