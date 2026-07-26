# Atomic / action-block 双工具方案解耦说明（2026-07-26）

## 结论

原子工具方案与 action-block 方案现在作为两个并列、互斥选择的模型协议存在：

- `atomic`：保留原来一轮一个工具的 prompt、parser、轨迹和训练行为；
- `action-block`：一轮输出一个 action block，内部执行多个原子工具，使用独立 prompt、
  parser、轨迹、SFT exporter 和 RL 环境。

模型不会同时看到两套顶层工具。两套方案只共享底层 harness 的关系算子、resident
state、执行语义和 `bird-set` scorer，因此可以控制底层能力一致，又不会把两种 action
carrier 混入同一个模型分布。

## 已完成的隔离边界

`src/sft/tool_schemes.py` 是唯一 scheme registry，固定两个稳定 id：

- `atomic`
- `action-block`

每个 scheme 都独立记录：

- system prompt；
- protocol version/hash；
- assistant carrier；
- 顶层工具集合；
- 可用原子工具集合；
- action-block 的 batch 上限。

所有新评测、教师轨迹、SFT manifest、RL rollout 和 checkpoint metadata 都显式记录
`tool_scheme`。数据导出前会检查 scheme，不能把 action-block 轨迹送进 atomic exporter，
也不能反向混用。

## 模型直接使用

atomic 学生模型继续输出：

```text
<think>...</think>
{"tool":"describe_table","arguments":{...}}
```

action-block 学生模型输出：

```text
<think>...</think>
{"tool":"action_block","arguments":{"calls":[...]}}
```

两套学生协议共用严格的 `think-json-v1` renderer/parser，但 JSON 内允许的顶层工具
和参数 schema 不同。它与 DeepSeek 的 provider-native reasoning + raw JSON 只在
传输层不同，结构化 action 完全一致。传输载体与工具 schema 分别记录，避免训练和
推理漂移。

本地或学生模型可以直接通过 action-block evaluator 运行：

```bash
... --assistant-carrier think-json-v1 --model <student-model-name>
```

此路径直接解析模型生成的完整 inline action，不依赖 DeepSeek 的独立 reasoning 字段。

## 训练边界

atomic SFT 和 process RL 保持不变。

action-block 已具备：

- 因果闭环 rollout；
- last-turn-only SFT 转换；
- provider history 到 student carrier 的无损转换；
- 每条 episode 的 fresh replay；
- result-only RL 环境。

但是当前 action-block v10/v11 数据仍然不能用于 SFT，因为它没有通过 fixed-200
准确率门。exporter 强制要求 `sft_export_eligible=true`，并拒绝包含 root error 或
blocked call 的轨迹。

action-block 暂不允许直接使用 atomic process credit。一个 block 中可能同时包含成功、
错误和 blocked primitive actions，把一个 scalar reward 赋给整个模型输出会污染局部
credit。现阶段只开放 matched result-only RL；后续需要独立设计 block-to-primitive
credit 后再开 process RL。

## 统一入口

评测：

```bash
.venv/bin/python src/eval/run_tool_scheme.py --tool-scheme atomic -- ...
.venv/bin/python src/eval/run_tool_scheme.py --tool-scheme action-block -- ...
```

因果轨迹生成：

```bash
.venv/bin/python src/sft/generate_tool_scheme_rollouts.py --tool-scheme atomic -- ...
.venv/bin/python src/sft/generate_tool_scheme_rollouts.py --tool-scheme action-block -- ...
```

RL：

```bash
... group_reinforce.py --tool-scheme atomic ...
... group_reinforce.py --tool-scheme action-block --reward-mode result-only ...
```

并行实验必须使用不同的数据集名、adapter/checkpoint 目录和结果目录；模型、任务集合、
解码参数和 `bird-set` metric 保持配对一致。

## 本次验证

本次只完成协议、数据和训练入口的工程解耦，没有启动新的训练或远程 200 题评测，
因此不产生新的准确率结论。代码回归结果为：

- active harness：81/81；
- SFT tests：99/99；
- RL tests：80/80，另有 9 项因本地可选训练依赖未安装而跳过；
- atomic eval tests：28/28；
- action-block tests：29/29。

同时验证了两套统一 launcher 的命令分派、action-block inline student 完整 episode、
两种 carrier 的严格 render/parse round trip、SFT fresh replay、scheme 数据防混用，
以及 checkpoint 从 atomic 切换到 action-block 时的 metadata 拒绝。旧
provider-native action-block prompt 哈希保持不变。

## Action carrier 修正

双方案解耦后的第一次提交错误地沿用了 atomic 的历史 tagged carrier 描述。当前实现已
修正为与 coder 训练一致的唯一 active carrier：

```text
<think>...</think>
{"tool":"...","arguments":{...}}
```

atomic 与 action-block 共用 `think-json-v1` 的 serializer/parser。它们的差异只存在于
JSON 中允许的工具集合、参数 schema 和执行协议。`<tool_call>` 不再是新评测、SFT 或
RL action 的一部分；旧 tagged action 只可由显式命名的离线迁移/replay 路径读取。
registry 因此递增为 `tool-scheme-registry-v2`。RL checkpoint 现在同时绑定 carrier、
protocol version/hash；旧 tagged-carrier checkpoint 不会被静默当作新 atomic
checkpoint 继续训练。
