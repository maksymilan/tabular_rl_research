# 迁移到 table_rl(dell)进度 — SSOT

> 目的:把 Qwen3.5 训练/评测环境 + 指标数据 + 本地远程连接逻辑从 NewGNN 迁到新算力。
> 本会话 Claude 工具反复故障(no content / elided / 串扰),若中断,按本文件接续。

## 目标机器(已确认 2026-06-26)

- **dell-PowerEdge-T640**,`ssh table_rl`(10.214.243.15:222),user dengyan
- **2× RTX 3090 24G,driver 560.35.03(=CUDA 12.6 max)**,48 核,503G 内存
- 磁盘 `/dev/sda2` 4.9T(3.2T 可用),HOME=/home/dengyan
- ⚠️ 最初探查曾出现 `amax`(2×4090D,driver550)= **工具故障期的一次性误连,忽略**。以 dell 为准。
- dell 与 NewGNN(也是 3090)同档 → QLoRA 24G 配置可照搬;driver 560>550,原生支持 cu126。

## 网络(无外网,经 Mac 隧道)

- 脚本:`src/sft/tunnel_table_rl.sh`。`-R 28472`→Mac clash 7897(给 dell egress);`-L 18001`→dell vLLM 8000。
- 启动:`nohup bash src/sft/tunnel_table_rl.sh > /tmp/tunnel_table_rl.log 2>&1 & disown`
- dell 上用代理:`export https_proxy=http://127.0.0.1:28472 http_proxy=http://127.0.0.1:28472`
- conda 代理:dell `~/.condarc` 写 proxy_servers http/https = http://127.0.0.1:28472
- ✅ 已验证:dell 经隧道 `curl https://huggingface.co` → HTTP 200

## 环境复刻(NewGNN 的真实栈,AGENTS.md 旧 pin 已过时)

NewGNN 两个 env 的 `pip freeze` 已存 `/tmp/sft_req.txt`、`/tmp/vllm_req.txt`(Mac)。清理:
`bitsandbytes @ file://` → `bitsandbytes==0.46.1`;删含 `@ file:` 的 packaging 行。

- **sft**(训练):python3.11,torch2.6.0(cu124),transformers5.6.0,llamafactory0.9.5,
  peft0.18.1,accelerate1.11.0,trl0.24.0,datasets4.0.0,bitsandbytes0.46.1,sentencepiece
  → `pip install -r sft_req.txt --extra-index-url https://download.pytorch.org/whl/cu124`
- **vllm-qwen35**(推理):python3.11,torch2.10.0+cu126,vllm0.19.1,transformers5.12.0
  → `pip install -r vllm_req.txt --extra-index-url https://download.pytorch.org/whl/cu126`
- 安装脚本:`/tmp/install_envs.sh`(在 dell tmux 里跑:`tmux new -d -s envsetup "bash ~/install_envs.sh > ~/install_envs.log 2>&1"`)

## 模型传输

- Qwen3.5-9B 在 NewGNN `/home/dengyan/models/Qwen3.5-9B`(~18GB)。
- NewGNN ↔ dell **不可直连**(不同内网段,ping/22/16014 全不通)→ 经 Mac 两跳。
- 实测:NewGNN→Mac **23MB/s**,Mac→dell **91MB/s**。tar 流式管道(不落 Mac 盘)~13-16min:
  ```
  ssh table_rl 'mkdir -p ~/models'
  ssh NewGNN 'cd /home/dengyan/models && tar cf - Qwen3.5-9B' | ssh table_rl 'cd ~/models && tar xf -'
  ```

## 进度清单

- [x] 阶段1 网络隧道(dell HF 200)
- [x] 阶段2 装 miniconda + 两个 env(dell 上,tmux)
  - sft env ✅ torch2.6.0+cu124 CUDA True;vllm-qwen35 ✅ vllm0.19.1/torch2.10.0+cu126 CUDA True
  - 坑1:conda 新版需 `conda tos accept --override-channels --channel .../pkgs/{main,r}`
  - 坑2:`conda run` 默认 pip 太旧→必须先 `conda run -n ENV pip install --upgrade pip`
  - 坑3:`conda run` 不继承 shell proxy→pip 联网失败,要 `conda run -n ENV env https_proxy=... pip ...`
  - 坑4:vllm_req.txt 完整 freeze 装时 ResolutionImpossible(starlette 冲突)→改装关键 3 包
    `vllm==0.19.1 torch==2.10.0 transformers==5.12.0` 让 pip 自解析,已用 smoke 验证可用
- [x] 阶段3 传模型 + 传代码(rsync)+ 传 checkpoints
  - 代码 rsync ✅ (325MB → ~/tabular_rl_project);ckpt e2(722M)+ e4(1.3G)✅
  - Qwen3.5-9B(19G)✅ 字节级验证一致 → ~/models/Qwen3.5-9B
  - 坑:tar 管道中断后别重拉全量;逐文件比字节大小,只补缺失/截断的(本次只缺 shard-2 + 4 小文件)
  - SFT 数据:spider_v8_pilot_ready_qwen35_4k(175 条)→ ~/tabular_rl_outputs/sft/(做训练 smoke 用)
- [x] 阶段4 切本地逻辑:dashboard SSH_HOST→table_rl, VLLM_BASE_URL→18001, AGENTS.md GPU 段已更新
- [x] 阶段5 smoke 验证 ✅ 全过
  - vLLM:base+v8-e4 LoRA 加载成功(含修复的 shard-2),经隧道发真实工具推理 → adapter 输出
    协议正确的 `<think>`+`<tool_call>{"tool":"describe_table"...}`,用完已关 vllm 显存释放
  - 训练:llamafactory QLoRA 3-step,loss 1.046→1.003→0.876,trainable 0.4578%,无 OOM,干净结束
  - 坑(重要):dell 缺 64 位 `libcuda.so` symlink → triton/bitsandbytes JIT 链接失败。
    修法:`ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 ~/cuda_link/libcuda.so` +
    训练脚本 `export LIBRARY_PATH=$HOME/cuda_link:$LIBRARY_PATH`(真实训练也必需,已记入 AGENTS PITFALLS)

## 指标数据

- NewGNN 的 results/checkpoints 大部分已同步到本地 `data/results/`、`data/trainer_states/`。
- 训练用的 adapter checkpoint 需传到 dell(继续训/评)。Qwen3.5 e4_fixed eval 最终:**541/1034=52.3%,legal97%,api_error8**(修复后)。
