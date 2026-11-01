# 分模块面试风格题库与参考答案

本题库提前给出复习目标，随相应课次使用，不要求今天学完。题目按 LLM 算法和推理 Infra 面试常见考察方式设计，未声称来自某公司的真实面试。每节仍先完成该节的 2～3 道检查题；本文件用于阶段复盘。

## 1. RMSNorm

使用 [第 1 节末尾的三道题及答案](01_RMSNorm.md#7-本节检查题先回答再展开)，覆盖手算、广播、数值精度和算子融合。

## 2. Embedding

**Q1：`nn.Embedding(V,D)` 与 one-hot 乘权重是否等价？哪种更适合实际执行？**

答：对合法 token ID，其结果等价于 one-hot 行向量乘 [V,D] 表；查表不需要显式构造 [B,T,V] 的稀疏 one-hot，避免大量无用存储与计算。输入是整数 [B,T]，输出是浮点 [B,T,D]。

**Q2：共享 embedding 和 LM head 能省什么？是否免去了输出投影计算？**

答：共享同一 [V,D] 参数，本仓库通过参数对象赋值实现。相对两份独立权重少存一份 V×D 参数；输出投影仍要计算，生成 logits 的计算量没有因此消失。

## 3. Causal Attention

**Q1：为什么用 `QKᵀ/√d`？mask 加在哪里？**

答：在分量近似独立、均值零、方差一的简化假设下，点积方差约为 d；缩放有助于控制 logits 尺度和 softmax 饱和。因果 mask 在 softmax 前加到 scores，未来位置置为 -Inf 后概率为零。QK norm 改变分布，但本仓库仍保留该缩放。

**Q2：`flash_attn=True` 是否保证执行 FlashAttention CUDA 内核？**

答：不保证。该字段只允许符合条件时进入 PyTorch SDPA；最终 backend 还受设备、dtype、尺寸和 PyTorch 调度影响。当前源码单 token decode、已有 cache 等情况下会进入手写分支。要用 profiler 验证真正执行的内核。

## 4. GQA

**Q1：默认 8 个 Q 头、4 个 KV 头，每个 KV 头对应多少 Q 头？相对 8-KV-head MHA，cache 少多少？**

答：每组 2 个 Q 头。保持层数、长度、batch、每头维度、dtype 不变，持久 KV 元素数减半。输出仍保留 8 个查询头的计算结果，不能直接说总计算量也减半。

**Q2：`repeat_kv` 后不是又变成 8 头了吗，为什么仍能省 cache？**

答：源码在 repeat 前建立 `past_kv=(xk,xv)`，持久保存的是 4 头；计算用的 K/V 再扩展为 8 头。`expand` 可以是视图，但后续 reshape 可能发生物化，不能假定整条链完全零复制；原生 GQA kernel 可避免显式扩展。

## 5. RoPE

**Q1：为什么旋转 Q/K 能引入相对位置？**

答：忽略额外幅度缩放，对正交旋转矩阵有 `(R_m q)ᵀ(R_n k)=qᵀR_(n-m)k`，内积依赖位置差；不同二维子空间采用不同角频率。当前实现采用前后半维配对，需要按 rotate_half 的代码核对，不能机械套交错偶奇维的排列。

**Q2：decode 新 token 的位置为何不能总从 0 开始？**

答：缓存 K 已按历史绝对位置旋转。新 Q/K 必须使用连续位置，才能让相对角度对应真实 token 距离。本仓库从历史 KV 序列长度计算 start_pos。

## 6. KV Cache

**Q1：推导 cache 字节数，并计算默认 Dense 在 B=1、S=4096、FP16 时的 cache 大小。**

答：`2×L×B×S×Hkv×d×bytes`，2 表示 K 和 V。代入 L=8、Hkv=4、d=96、bytes=2，得 50331648 字节，即 48 MiB。仅为持久 cache，不含权重、激活、重复 KV、分页元数据等。

**Q2：KV Cache 会把 decode Attention 变成 O(1) 吗？分页又改变了什么？**

答：不会。它避免重算历史 token 的投影和各层状态，但当前 Q 仍需读取历史 K/V；单步注意力仍随上下文长度增长。分页改变存储管理与寻址，改善分配、复用与碎片等问题，不消除读取历史上下文的数学工作量。MiniMind 的 torch.cat 还会复制旧 cache，不能当作生产级分页实现。

## 7. SwiGLU FFN

**Q1：本仓库 FFN 的公式、形状与主要参数量是什么？**

答：`down(SiLU(gate(x)) * up(x))`。两路 [B,T,D]→[B,T,I]，逐元素乘后投影回 [B,T,D]。无 bias 时三个投影约 3DI 参数。默认 I=ceil(768π/64)×64=2432。

**Q2：FFN 会在不同 token 间交换信息吗？**

答：该 Dense FFN 对每个 token 独立应用同一组参数，不混合 T 轴；上下文交互来自 Attention。FFN 仍可能占大量参数和矩阵乘工作量。

## 8. Block 与残差

**Q1：为什么 `post_attention_layernorm` 这个名字不代表 Post-Norm？**

答：看执行顺序：先 `x+Attention(norm(x))`，再 `h+MLP(norm(h))`。两个 norm 都位于各自子层之前，这是 Pre-Norm 结构；变量名不能替代调用顺序。

**Q2：残差与归一化分别解决什么问题？**

答：残差提供直接的信息与梯度通路；归一化调整分支输入尺度。它们协同改善优化条件，但不等于保证训练必然稳定。

## 9. 完整 Forward

**Q1：输入 [B,T]，最终 logits 的形状是什么？训练和常规 decode 分别需要哪些位置？**

答：默认完整输出 [B,T,V]。训练通常需要所有可监督位置；常规生成选择下一个 token 时通常只需最后位置 logits。当前 forward 支持 logits_to_keep，但不要在需要全序列 loss 时随意截取。

**Q2：为什么 `logits_to_keep=0` 在本实现代表保留所有位置？**

答：`slice(-0,None)` 等于 `slice(0,None)`，所以完整保留；正整数 k 才是最后 k 个位置。它只裁剪 LM head 输入位置，不跳过之前各层对输入 token 的计算。

## 10. Loss

**Q1：next-token 对齐在哪里做？为什么不能 Dataset 和 forward 各 shift 一次？**

答：当前 Dataset 返回与输入位置对齐的 labels，forward 用 `logits[..., :-1, :]` 配 `labels[..., 1:]`。再额外 shift 会错位。例：输入 [A,B,C]，监督配对是 A 的 logits→B、B 的 logits→C。

**Q2：-100 表示 attention mask 吗？全是 -100 有什么问题？**

答：这里是交叉熵的 ignore_index，只排除相应 target 的损失，并不自动禁止模型读取该输入 token。若 shift 后没有任何有效目标，mean reduction 可能得到 NaN，应在数据/批次层检查有效监督数。

## 11. Pretrain

**Q1：本仓库预训练优化的是什么？padding 是否参与 loss？**

答：自回归 next-token 交叉熵，MoE 模式另加辅助损失。PretrainDataset 将 pad 位置 labels 设为 -100；shift 后对应目标不参与交叉熵。当前 Dense 辅助损失为零。

**Q2：梯度累积为什么除以 accumulation_steps？是否任何情况下都等于大 batch？**

答：用于近似平均多个 microbatch 的梯度。但有效监督 token 数不同、随机性、归约方式与末尾不足一个累积窗口等都会影响等价性；当前脚本尾部仍除以固定累积数，须专门检查尾窗权重，不能笼统宣称完全等价。

## 12. SFT

**Q1：SFT 和 Pretrain 的核心差异是否是换了交叉熵？**

答：当前二者复用相同 LM forward 和交叉熵，主要差异在样本、模板、有效监督位置、起始权重和训练超参。SFTDataset 识别 assistant 片段来生成 labels，其他位置为 -100。

**Q2：user labels 为 -100，模型还能使用 user 内容吗？**

答：能。user token 仍是模型输入，assistant 的因果 Attention 可以读取它；mask 的是直接预测 user target 的损失，梯度也可以通过 assistant 计算路径传回前面的表示。

## 13. Serving 与 vLLM 迁移

**Q1：把模型包成 HTTP API 是否就完成了高吞吐 Serving？**

答：API 是接口层，还要考虑请求调度、批处理、KV 生命周期、取消/结束回收、并发隔离和资源管理。先理解单请求 generate，再读引擎调度。

**Q2：如何判断优化真的有效？**

答：保持模型、硬件、精度、输入/输出长度和负载一致，分别观测 TTFT（首 token 延迟）、TPOT（输出 token 间耗时）、总吞吐、峰值显存和输出正确性；同时固定版本、预热并处理异步计时。不能只比较单次 wall time 或只看 tokens/s。

vLLM 迁移概念参考：[RMSNorm 接口](https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/layernorm/)、[分页注意力设计](https://docs.vllm.ai/en/latest/design/paged_attention/)。实际接入需核对导出的 architecture 与所用 vLLM 版本，不能仅凭自定义模型类名保证兼容。
