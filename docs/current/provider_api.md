# DeepSeek provider contract

新数据生成、评测、审计和恢复请求只使用官方 OpenAI-compatible DeepSeek 服务：

```text
BASE_URL=https://api.deepseek.com
POST /chat/completions
```

运行时凭据只放在仓库根目录被忽略的 `api.md`，其中 `BASE_URL` 必须是上面的官方地址。
密钥不得进入文档、配置、命令、日志或模型上下文。官方服务不可用时显式失败，不切换第三方
代理或旧地址。

当前主线是 Atomic version26 的 Chat Completions 闭环：每轮只产生一个模型可见工具动作，
Harness 执行工具并返回因果反馈；隐藏 gold SQL 只用于 Harness 终止判分。外部教师使用同一
学生运行时契约加上生成指导，不能看到 gold SQL、gold 结果或未来轨迹。

DeepSeek FIM `/beta/completions` 不属于当前工具调用链，除非另行建立独立、命名明确的代码
补全功能，不得替代 Chat Completions。

2026-09-09 用户明确授权的 `teacher-process-credit-v1` 是独立的事后轨迹审计例外：
可读取完整的既有model-visible action/observation序列及终局correct标志，用于离散过程
分类，不产生SFT action/target，不反填因果轨迹。Gold SQL/result仍禁止进入请求；教师
不能输出连续reward，只输出规则规定的类别。首轮上限16次请求，外部数据生成仍暂停。

每次官方请求记录模型身份、prompt/tool-schema hash、请求控制、token 上限和结果目录；凭据
本身永不记录。历史 provider 元数据仅用于审计，不授权恢复旧 scheme。
