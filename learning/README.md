# MiniMind-3 源码研读：7 天路线与学习入口

目标：能从公式推导张量计算，沿真实源码追踪训练与生成，再把这些知识迁移到 vLLM、LLM Serving 与 AI Infra。7 天是第一轮密集研读，不能等同于已经掌握生产级推理引擎。

## 1. 本次源码基线

- 学习目录：`E:\Deskbook\MiniMind`。
- 上游：[jingyaogong/minimind](https://github.com/jingyaogong/minimind)。当前目录直接克隆，无额外套一层目录。
- 分支：`master`；浅克隆，仅包含当前历史切片。
- 固定提交：`233d00fa068ec2b8c0d4f07b703934d6391a5a73`，提交日期 2026-10-27。
- 讲义编写日期：2026-11-01。所有源码行号对应上述提交。
- 本轮主线：Dense MiniMind-3；MoE、LoRA、DPO、GRPO 等列为后续拓展。

先完成第一节：[01：RMSNorm](01_RMSNorm.md)。题目与答案位于每节末尾，先独立回答再展开答案。

已编写课次：[01：RMSNorm](01_RMSNorm.md)、[02：Token Embedding](02_Embedding.md)。第二节起直接列出涉及的源码，并在必要位置添加教学注释。

## 2. 如何使用提供的网站

你给的 [深度掌握路线](https://www.minimind.wiki/docs/guide/mastery) 是 joyehuang 的学习笔记站，页脚也说明基于 MiniMind 的笔记与实验；项目上游 README 才是当前项目的一手说明。这里结合两者，但以本地固定版本的代码行为为判据。

该站给出基础、数据、训练、进阶的路径，强调对照实验与迭代学习；其 [系统学习页](https://www.minimind.wiki/docs/guide/systematic) 从归一化等组件进入架构组装。本课程保留组件实验思路，并把 KV Cache、张量布局、显存核算提前，以对接你的 Infra 目标。你要求的讲解顺序采用：数学 → 源码与形状 → 最小实验。

对照时注意这些版本差异：

| 笔记站示例 | 本地源码事实 | 本课程处理 |
|---|---|---|
| `scripts/train_tokenizer.py` | 实际为 `trainer/train_tokenizer.py` | 按实际路径阅读 |
| `pretrain_hq.jsonl`、`sft_mini_512.jsonl` | 训练默认是 `pretrain_t2t_mini.jsonl`、`sft_t2t_mini.jsonl` | 不直接复制旧命令 |
| 示例 hidden size 512 | 当前模型与训练脚本默认 768 | 讲清默认值与实验缩小值的区别 |
| 将 Packing 作为数据学习点 | 当前 `PretrainDataset` 对单条 text 截断、补齐；未在该类跨样本 packing | 以类实现为准 |
| 把固定 PPL 阈值作为达标点 | 指标受数据、分词器与评测方式影响 | 以可复现验证与留出集结果验收 |

上游 README 当前也不建议随意重训 tokenizer。先理解并使用仓库自带分词器，重训列为拓展实验。

## 3. 仓库地图：先知道去哪里找

```text
MiniMind/
├── model/
│   ├── model_minimind.py    # Config、RMSNorm、RoPE、Attention、FFN、MoE、Block、Model、LM、generate
│   ├── model_lora.py        # 后续：低秩适配
│   ├── tokenizer.json
│   └── tokenizer_config.json
├── dataset/
│   ├── lm_dataset.py        # PretrainDataset、SFTDataset 等
│   └── dataset.md           # 数据说明；完整训练数据未随本次克隆下载
├── trainer/
│   ├── train_tokenizer.py
│   ├── train_pretrain.py    # 预训练入口
│   ├── train_full_sft.py    # 全参数 SFT 入口
│   ├── trainer_utils.py     # 初始化、学习率、checkpoint、分布式等
│   ├── rollout_engine.py    # 进阶：生成后端
│   └── train_*.py           # LoRA、DPO、PPO、GRPO、蒸馏、Agent 等拓展
├── scripts/
│   ├── convert_model.py     # 向 Qwen3/Qwen3-MoE 等格式映射
│   ├── serve_openai_api.py  # API 服务入口
│   ├── chat_api.py
│   └── web_demo.py
├── eval_llm.py              # 模型加载与推理体验
├── requirements.txt
└── learning/                # 本轮新增的讲义、实验与学习日志
```

当前重点源码定位（无需一次读完整文件）：

| 位置 | 阅读对象 | 需要回答的问题 |
|---|---|---|
| `model/model_minimind.py:10` | MiniMindConfig | D、Hq、Hkv、d、V 分别取什么值？ |
| 同文件 `:50` | RMSNorm | 在哪个轴归一化？ |
| 同文件 `:62`、`:80` | RoPE | 位置如何进入 Q/K？ |
| 同文件 `:86`、`:91` | repeat_kv、Attention | GQA 的共享发生在哪里？ |
| 同文件 `:136`、`:183` | FFN、Block | 门控与残差如何衔接？ |
| 同文件 `:201`、`:239` | Model、ForCausalLM | token 如何变成 logits？ |
| 同文件 `:250` 起 | forward 内 loss | 哪一处进行 next-token shift？ |
| 同文件 `generate` | 自回归生成 | prefill 后每次实际输入多少 token？ |
| `dataset/lm_dataset.py:40`、`:61` | 两类数据集 | 哪些 label 被设为 -100？ |

基础符号：B=batch，T=本次输入长度，S=含历史的 KV 长度，D=hidden size，Hq=查询头数，Hkv=KV 头数，d=head dim，V=词表大小。默认 D=768、Hq=8、Hkv=4、d=96、V=6400；注意可显式覆盖 head_dim，不能在所有配置中假定 D=Hq×d。

全局导航仅记一行：token IDs → embedding → 多个 Block → final norm → LM head → logits；训练计算 loss，推理选择下个 token。本节不展开这些模块。

## 4. 7 天计划（约 32 小时，可按进度顺延）

一天可以含多节，但每节只处理一个核心模块。每节顺序固定为：数学与手算 → 输入输出合同 → 逐段源码与形状 → 实验 → 替代实现与性能 → 2～3 题自测。答不清就留在本节，不为了赶天数跳过。

| 天 | 时长 | 分节顺序 | 必须提交的实验/学习产物 | vLLM / Infra 连接点 |
|---|---:|---|---|---|
| Day 1 | 4h | 仓库导航；第 1 节 RMSNorm；第 2 节 token embedding | 运行本节脚本；手绘 [B,T]→[B,T,D]；解释 last-dim reduction | RMSNorm 融合、词表权重占用 |
| Day 2 | 5h | 第 3 节因果 Attention；第 4 节 GQA | 手算 3-token attention；验证未来 token 不影响过去输出；计算 MHA/GQA cache 差异 | Attention backend、访存与中间矩阵 |
| Day 3 | 4h | 第 5 节 RoPE | 验证旋转保持范数；比较同一相对位移的内积；逐步核对位置 offset | 旋转位置编码算子与 cache 位置 |
| Day 4 | 5h | 第 6 节 KV Cache | 固定输入，比较 full forward 与逐 token cache 的 logits；打印逐层 KV；计算 cache 字节数 | prefill/decode、分页 KV、prefix cache |
| Day 5 | 5h | 第 7 节 SwiGLU FFN；第 8 节 Block；第 9 节完整 Forward | 用 tiny config 跑 [B,T]→[B,T,V]；hooks 记录中间形状；核对权重共享 | kernel 融合、模型执行器、只算末 token logits |
| Day 6 | 5h | 第 10 节 Loss；第 11 节 Pretrain 训练循环 | 手算交叉熵；确认 shift 仅一次；合成数据过拟合；保存恢复并对比 | 训练态/推理态内存、权重导出 |
| Day 7 | 4h | 第 12 节 SFT；第 13 节 Serving 边界 | 打印对话模板和有效 labels；验证仅目标回答监督；绘制 API→tokenizer→模型→采样的数据流 | Qwen3 格式映射、调度、TTFT/TPOT |

Day 2～7 的实验是学习任务，尚未执行。当前已编写第 1、2 节完整讲义与对应实验，运行情况见验证记录；后续逐节补齐，避免一次讲完源码。

### 各模块需要弄清的输入、输出和代价

这张表是预习索引，详细推导在相应课次展开。

| 模块 | 输入 → 输出 | 为什么设计 / 可比较的实现 | 推理显存与性能关注点 |
|---|---|---|---|
| RMSNorm | 浮点 [...,D]→同形状 | 控制特征尺度；对比 LayerNorm、融合 RMSNorm | FP32 临时张量、带宽、launch |
| Embedding | 整数 [B,T]→[B,T,D] | 查表获得可学表示；对比 one-hot 乘法 | V×D 权重、输出头共享 |
| Attention | [B,T,D]→[B,T,D]，可附 KV | 混合可见上下文；手写与 SDPA | 显式 scores 为 [B,Hq,T,S] |
| GQA | Q 多头，K/V 较少头→查询头输出 | 共享 K/V；对比 MHA/MQA | Hkv 决定 cache 大小；repeat 临时张量 |
| RoPE | Q/K 与位置→旋转后 Q/K | 注入相对位置关系；对比绝对位置/ALiBi | cos/sin buffer、位置索引与融合 |
| KV Cache | 新 token 与旧 KV→新输出与追加 KV | 复用历史投影；对比重算/预分配/分页 | cache 随 S 增长；torch.cat 复制 |
| SwiGLU | [B,T,D]→[B,T,D] | 门控非线性；对比 GELU FFN | 中间宽度 I、矩阵乘法开销 |
| Block | 隐藏态、位置、旧 KV→隐藏态、新 KV | 预归一化与残差；对比 Post-Norm | 激活与算子融合 |
| Forward | IDs 与可选 labels/cache→logits/loss/cache | 组装统一计算；对比仅末位置投影 | [B,T,V] logits 可以很大 |
| Loss | logits 与 labels→标量 | next-token 负对数似然；比较 label smoothing | 标准 serving 通常不计算训练 loss |
| Pretrain | text→IDs/labels→权重更新 | 学习续写分布；比较 packing 数据管线 | 训练还需梯度、优化器与反传激活 |
| SFT | conversations→模板/监督位置→权重更新 | 学习指令响应；对比全序列监督/LoRA | 全参 SFT 未缩小部署参数或 KV |
| Serving | 请求→token 流/响应 | 管理多请求；对比单请求 generate | 并发、调度、cache 生命周期 |

## 5. 学习环境和运行方式

本次独立创建 `.venv`（Python 3.12），避免改动原 Anaconda 环境。第 1 节只需要 CPU PyTorch 和模型定义的导入依赖，不需要下载权重或训练集。`learning/requirements-cpu.txt` 是教学最小依赖，不是整个项目依赖。

在 PowerShell、项目根目录执行：

```powershell
uv venv .venv --python 3.12
uv pip install --python .venv\Scripts\python.exe torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv\Scripts\python.exe transformers==4.57.6 numpy==1.26.4
.\.venv\Scripts\python.exe learning\experiments\01_rmsnorm.py
```

环境已创建时直接执行最后一行即可。使用显式解释器路径，无须激活环境。后续进入 Dataset 和训练脚本时再补充对应依赖；本环境的 CPU torch 不用于 CUDA 训练性能评估。

## 6. 验收与下一步

- 每节必须：能独立讲公式；写出张量轴；运行实验并解释断言；说出一种替代实现及其代价。
- 第 7 天必须：不看讲义解释 Attention→RoPE→Cache→Forward→Loss→Pretrain→SFT 的数据流，区分哪些属于模型、哪些属于推理引擎。
- 不以“看完所有文件”作为学会的标准。完整训练、真实基准评测与 GPU 服务压测是后续阶段。
- 每节只做 2～3 道检查题，答案随节提供；额外复习用 [面试风格题库](INTERVIEW_BANK.md)。题库为概念型模拟题，不虚构公司或真实面试出处。
- 用 [学习日志](PROGRESS.md) 记录自己的解释和实验结果。下一次把第 1 节三题的回答发来，再进入 embedding；如果基础已掌握，可以直接按计划进入 Attention 单节。

参考：[上游固定版本](https://github.com/jingyaogong/minimind/tree/233d00fa068ec2b8c0d4f07b703934d6391a5a73)、[vLLM RMSNorm 实现接口](https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/layernorm/)、[vLLM Paged Attention 设计](https://docs.vllm.ai/en/latest/design/paged_attention/)。vLLM 链接为 2026-11-01 查阅的 latest 文档；本次未克隆、部署或固定 vLLM 版本，迁移阶段须另行固定。
