# RL 实验场景模块

实验场景模块描述一个具体的数据准备、训练配置或审计场景。它们组合 `rl.shared`、
`rl.data_selection`、`rl.frameworks.trl` 与 runtime 固定模块。场景可以定义自己的 CLI 参数
和输出格式，但不得复制共享 I/O、哈希、cohort 谓词、trainer 或 GPU 编排逻辑。

当前子包：

- `data/`：cohort 和 artifact 准备场景；
- `audits/`：process-reward、grounding 和行为审计场景；
- `counterfactual/`：反事实测试场景。

历史入口保留原始文件名并置于这些子包内，便于审计，同时保持 `src/rl` 根目录整洁。
