# 工具方案登记

更新时间：2026-09-03

当前只允许一个新实验 scheme：**Atomic version26**。本页不再维护多分支 registry；历史
scheme 的设计和结果保留在 archive/reports，只有在明确指定原始 identity 时才可 replay。

## 当前方案

| 项目 | 当前值 |
|---|---|
| scheme | `atomic` |
| registry | `tool-scheme-registry-v13-atomic-only` |
| protocol | `version26`，hash `4da19387399bd3a5` |
| carrier | `think-json-v1` |
| model-visible tools | version26 typed perception/relational/terminal surface |
| state | Harness-owned resident relations、artifacts、provenance、structured errors |
| terminal | exact Harness result artifact，模型不提交答案值 |
| training/RL | 唯一准入方向；当前 four-level + SAAM candidate 尚待 formal gate |

实现和身份详见 [`tool_protocol.md`](tool_protocol.md)。

## 旧方案处理

checkpoint-relalg、native-tool-bundle、Direct/Hybrid、iterative-SQL、relational-program、
action-block 以及 Atomic v24/v39/v40–v54 都是历史诊断或 replay 兼容。它们不再出现在
默认 prompt、config、launcher、SFT exporter 或 RL output root 中，也不得与 version26
trajectory、adapter、manifest 混用。

这些旧实现按 [`archive/code/legacy_migration.md`](../../archive/code/legacy_migration.md) 迁入
`archive/`，先不删除；`src/` 不保留 compatibility stub，历史 replay 必须显式恢复归档路径。

如需复现旧结果，必须同时提供原 scheme、protocol/prompt/carrier、runtime hash、cohort 和
独立 output directory；复现完成后仍不会改变当前 scheme 登记。
