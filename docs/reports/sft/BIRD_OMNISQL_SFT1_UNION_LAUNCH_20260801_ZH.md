# OmniSQL-7B 联合 SFT1：数据血缘与启动记录

日期：2026-08-01
状态：正式 4-epoch QLoRA 已在 NewGNN GPU 5 运行

## 1. 当前 RL 的 SFT2 起点

当前 Exp1–Exp11 使用同一个初始化：

- base：`/home/dengyan/models/Qwen2.5-Coder-7B-Instruct`
- adapter：`/home/dengyan/tabular_rl_outputs/checkpoints/qwen2.5-coder-7b-bird-sft2-batch2-cp560-repeated-only-6400-single-gpu-qlora/checkpoint-1682`
- adapter SHA-256：`d880e2d7cc3203fdb0d11a7c188d8f607fd297b174eff23f741b6fe73cc3ce6e`
- SFT2 dataset：`bird_batch2_cp560_repeated_only_qwen25coder_6400_complete`
- 协议：atomic `version26`，`history_turns=4`，student runtime prompt，
  `think-json-v1`
- student prompt SHA-256：
  `848598074e653648d52b58dfa1a54dd777beabae16cb91d83c4ba1a54b5d7316`

SFT2 从 SFT1 `checkpoint-560` 继续训练两个 epoch；单卡 checkpoint-841 是
epoch 1，checkpoint-1682 是 epoch 2。

## 2. OmniSQL 联合 SFT1 数据

联合的是实际进入原训练的两个完整-episode 数据集，而不是更小的抽样：

| 来源 | 完整 episode | action target |
|---|---:|---:|
| 原 SFT1 | 678 | 4,471 |
| 原 SFT2 batch2 | 2,275 | 13,454 |
| 联合 | **2,953** | **17,925** |

两批之间的 episode ID、底层 `bird_train_*` task 和
`(model_input_sha256, target_sha256)` 均无重叠。两批使用相同的 version26 student
prompt/carrier，可以直接联合。

使用 OmniSQL 自己的 tokenizer、Qwen template 和 cutoff 6,400 重做了精确审计：

- 17,925/17,925 target 完整；
- 0 个 current source 截断；
- 0 个最早 history pair 不完整；
- 最长样本 6,366 tokens；
- 2,953/2,953 episode 完整保留；
- 每个 episode 恰有一个 `answer_from_context`；
- 所有 target 均通过非空 `<think>` + strict raw JSON carrier 检查；
- canonical union SHA-256：
  `b02743aa08df521242d58effb0e6fee30a43aa5f2482aa3e6c74bd2d9638d353`。

SFT1 与 SFT2 的 audit metadata schema 不完全相同。canonical union 保留全部 metadata；
训练另用只含 `system + conversations` 的 schema-stable projection，模型可见内容和 loss
target 不变。training view SHA-256：
`d1a4c62d490af4a3a52cf53b91f717f8de6e9e914075bef6e14142b5227f4a18`。

NewGNN 数据目录：
`/home/dengyan/tabular_rl_outputs/data/omnisql_sft1_union_sft1_sft2_20260801`

## 3. 模型与训练配置

- 模型：`seeklhy/OmniSQL-7B@af4eed67f561bbeea555c017dae4b38b93bac2eb`
- NewGNN 路径：`/home/dengyan/models/OmniSQL-7B-af4eed67`
- `model.safetensors` 大小：15,231,272,152 bytes
- `model.safetensors` SHA-256：
  `dbdd444b3233a3decb600547c8d3cda0e0118c7cfe8085b68a05c519b0e80b01`
- 隔离环境：`/home/dengyan/miniconda3/envs/omnisql-sft`
- 环境：torch 2.6.0+cu124、Transformers 5.6.0、PEFT 0.18.1、
  Datasets 4.0.0、LLaMA-Factory 0.9.5
- QLoRA：4-bit、rank 16、alpha 32、dropout 0.05、all linear
- cutoff：6,400；mask history；不 packing
- 单卡 batch 1，gradient accumulation 16，有效 batch 16
- learning rate `5e-5`，cosine，warmup ratio 0.03
- 4 epochs，1,121 optimizer steps/epoch，共 4,484 steps
- `save_strategy=epoch`，预计 checkpoint-1121/2242/3363/4484 全部保留
- 每 2 steps 写 loss；训练结束生成 `loss_by_epoch.json` 和
  `loss_by_epoch.csv`

正式输出：
`/home/dengyan/tabular_rl_outputs/checkpoints/omnisql-7b-bird-sft1-union-sft1-sft2-6400-4epoch-qlora`

## 4. 启动与首条运行证据

- run id：`bird_omnisql_sft1_union_4epoch_20260801_1822`
- launcher PID：`4177474`
- trainer PID：`4177630`
- physical GPU：5
- 首条正式记录：step 2，loss `0.9187032580`，epoch
  `0.0017852162`，elapsed `00:01:48`
- 该时刻 GPU memory 22,072 MiB，utilization 100%
- Trainer 首次 ETA：约 2 天 19 小时 38 分

训练日志：
`/home/dengyan/tabular_rl_outputs/logs/bird_omnisql_sft1_union_4epoch_20260801_1822.log`

实时结构化 loss：
`/home/dengyan/tabular_rl_outputs/checkpoints/omnisql-7b-bird-sft1-union-sft1-sft2-6400-4epoch-qlora/trainer_log.jsonl`

启动 manifest：
`/home/dengyan/tabular_rl_outputs/logs/bird_omnisql_sft1_union_4epoch_20260801_1822.launch_manifest.json`
