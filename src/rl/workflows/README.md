# 实验工作流层

`config.py` 和共享 runner 属于固定基础设施。新的实验场景应在这里（或对应 YAML）增加一个
小型场景定义，提供 cohort 路径、规模、机制名称和资源参数，然后调用已有的 RL launcher、
机制组件、数据选择策略或评测规划器，不复制这些实现。

支持四类场景：`rl_verify`、`rl_full`、`evaluate` 和 `sft_frozen`。`cli.py` 规划器保持无副作用，
因此可在申请 GPU 前审阅场景参数。
