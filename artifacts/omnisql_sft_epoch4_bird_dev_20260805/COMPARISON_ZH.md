# OmniSQL SFT 前后与 SFT2 的 BIRD-dev 对比

## Greedy 主结果

| 模型/系统 | 训练与输出方式 | 评测接口 | 正确题 | Accuracy | Legal/Valid | 平均步骤 | 相对 OmniSQL SFT 后 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| OmniSQL 原模型（SFT 前） | OmniSQL 官方 Text-to-SQL 权重；直接生成 SQL | 官方完整 DDL、列描述、检索值、BIRD evidence；Direct SQL greedy | 983/1534 | **64.08%** | 未记录 | 单次 SQL | **+194 题 / +12.65 pp** |
| OmniSQL SFT union epoch 4（SFT 后） | SFT1+SFT2 联合轨迹训练；原子工具调用 | atomic version36，temperature=0，n=1，bird-set | 789/1534 | **51.43%** | 1264/1534 = **82.40%** | 9.45 | 基准行 |
| Qwen2.5-Coder-7B SFT2 | checkpoint-1682 工具模型 | atomic version36，temperature=0，n=1，bird-set | 762/1534 | **49.67%** | 1194/1534 = **77.84%** | 8.37 | **-27 题 / -1.76 pp** |

注意：OmniSQL SFT 前的 64.08% 是 Direct SQL 完整系统结果，并非与工具调用版本严格相同的单变量接口对照。它提供完整 schema/描述/检索值并直接生成 SQL；后两行则需要多轮调用原子工具，因此 64.08% 应视为需要追回的系统级能力参考。

## 严格逐题配对

| 配对 | 两者都正确 | OmniSQL SFT 后独有正确 | 对照独有正确 | 两者都错误 | OmniSQL SFT 后净变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 对比 OmniSQL 原模型 Direct SQL | 701 | 88 | 282 | 463 | **-194 题 / -12.65 pp** |
| 对比 SFT2 checkpoint-1682 version36 工具模型 | 640 | 149 | 122 | 623 | **+27 题 / +1.76 pp** |

## 工具模型的失败构成

| 终止类型 | OmniSQL SFT 后 | SFT2 | OmniSQL SFT 后变化 |
| --- | ---: | ---: | ---: |
| correct | 789 | 762 | +27 |
| wrong_answer | 475 | 432 | +43 |
| max_steps | 151 | 100 | +51 |
| protocol_error | 60 | 178 | -118 |
| context_overflow | 31 | 13 | +18 |
| argument_validation_error | 23 | 43 | -20 |
| execution_error | 5 | 6 | -1 |

OmniSQL SFT 后明显减少了协议错误和参数错误，但更多轨迹转化成了长轨迹、上下文溢出或语义错误，所以最终只比 SFT2 高 1.76 pp。

## Sampling / Pass@k（辅助对比）

| 模型 | 协议说明 | Pass@1 | Pass@2 | Pass@4 |
| --- | --- | ---: | ---: | ---: |
| OmniSQL SFT 后 | version36，temperature=0.7，top-p=0.95，n=4 | **817/1534 = 53.26%** | **946/1534 = 61.67%** | **1033/1534 = 67.34%** |
| Qwen2.5-Coder-7B checkpoint-560（历史） | 历史 atomic artifact，使用严格 official EX 重评分 | 778/1534 = 50.72% | 924/1534 = 60.23% | 1033/1534 = 67.34% |
| 表面差值 | 非严格同协议，只作探索空间参考 | +39 / +2.54 pp | +22 / +1.43 pp | 0 / 0.00 pp |

SFT2 checkpoint-1682 当前 version36 没有对应的全量 K=4 产物；表中的 checkpoint-560 是单独的历史模型结果。
