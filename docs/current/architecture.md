# 当前代码架构

更新时间：2026-09-06

当前实现只有一条训练/评测协议：Atomic version26。其他 scheme 和版本仍在仓库中，但只
作历史 replay/审计，不从当前文档或默认脚本启动。

## 数据流

```text
student/teacher prompt
        ↓
version26 parser + typed action
        ↓
Harness / tool_environment_v26
        ↓
resident state + relation artifacts + structured errors
        ↓
terminal denotation / replay audit
        ↓
reward + SAAM transition credit
        ↓
replicated actor update（A100 单卡 trainer）
```

online rollout vLLM 在第二张 A100 上运行；评测在 `table_rl` 或 `NewGNN` 的独立 GPU 上
运行，避免把评测资源和主 RL optimizer 混在一起。

## 代码所有权

| 层 | 当前入口 | 责任 |
|---|---|---|
| Harness/environment | `src/rl/runtime/tool_environment_v26.py` | 合法调用、状态、SQLite、denotation、错误 |
| Runtime adapters | `src/rl/runtime/` | task loading、replay、terminal reward、rollout scoring 等固定运行时能力 |
| Atomic protocol | 外部 immutable version26 runtime 的 `src/sft/protocol.py`、`src/sft/prompt_contract.py`（工作树源码可能为历史版本） | action schema、carrier、prompt/history |
| Relation facts | `src/harness/relation_derivation/` | derived relation 的事实 provenance |
| Rollout | `src/rl/frameworks/trl/rollout.py` | causal model↔Harness episode |
| Reward | `src/rl/runtime/terminal_reward.py` | four-level terminal reward |
| Objectives | `src/rl/objectives/` | process objective 与 credit 计算 |
| Credit | `src/rl/frameworks/trl/state_action_ambiguity.py` | SAAM asymmetric-error masks |
| Trainer | `src/rl/frameworks/trl/run_transition_grpo.py` 的 replicated trainer | transition loss、单卡更新、checkpoint |
| Evaluation | `reproductions/trust_sql/qwen3_8b_atomic_sft1/` | version26 matched BIRD-dev greedy |
| Experiment scenarios | `src/rl/scenarios/` | 数据准备、审计和反事实实验的场景入口；复用固定模块 |

## 身份边界

run manifest 必须记录 protocol/prompt/carrier/history、base model、initial adapter、cohort、
GPU/port、source tree 和 runtime hash。不同身份不得共享 rollout、adapter、result 或
checkpoint 目录。Gold SQL 永远是 Harness-only；reward 和 process credit 只能来自
Harness replay/state/provenance/dependency。

## 默认入口

- RL config：`src/rl/configs/experiments/qwen3_8b_atomic_v26_saam_screened500_single_gpu.yaml`
- A100 launcher：`src/rl/scenarios/rl_main/run_qwen3_8b_atomic_v26_saam_fourlevel_spanbalanced_700_single_gpu_a100.sh`
- SFT launcher：`src/sft/train_qwen3_8b_atomic_sft1_newgnn.sh`
- matched handoff：`src/rl/scenarios/evaluation/run_v26_matched_eval_handoff.sh`

## 历史代码策略

旧版本和 scheme 迁入 `archive/`，暂不删除、不重命名 identity、不并入 version26；迁移期间
不保留旧 scheme 的 compatibility stub。当前收敛通过 active docs、默认 config、
launcher 和 manifest identity 实现。

## 文件归属和临时快照

项目根目录只保留项目元数据和入口文档。可执行的当前代码必须位于明确的 `src/`、
`scripts/` 或 `reproductions/` 模块目录，并使用模块内的稳定名称；`*.remote.current.py`
和根目录 `tmp_*` 都不是合法的当前入口。远程运行时的对比副本必须放在
`archive/code/remote_snapshots/<runtime-id>/src/...`，保留原始相对路径、来源、采集时间和
SHA-256；一次性诊断脚本放在带日期和用途的 archive 目录。布局测试会阻止这类副本重新
出现在 active tree 中。
