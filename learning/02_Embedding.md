# 第 2 节：Token Embedding——从整数 ID 到隐藏向量

建议 60～90 分钟。上一节：[RMSNorm](01_RMSNorm.md)。本节只讲输入 embedding，以及与它直接相关的输出权重共享；Attention、Tokenizer 的 BPE 训练和语言模型 loss 的推导留到各自课次。

源码基线仍为 `233d00fa068ec2b8c0d4f07b703934d6391a5a73`。涉及的真实代码全部在下面列出，注释为教学补充。实验直接使用本地 MiniMind 类，无权重和数据下载。

## 1. 数学：Embedding 是一张可学习的表

分词器把文本变成整数 token ID。token 不一定对应完整汉字或单词。ID 仅标识词表条目，ID 100 和 101 相邻并不意味着语义更相似。

设词表大小为 V，隐藏维度为 D，定义参数矩阵：

$$E\in\mathbb{R}^{V\times D}.$$

对于整数 ID 矩阵 I∈{0,…,V−1}^(B×T)，输出为：

$$X_{b,t,:}=E_{I_{b,t},:},\qquad X\in\mathbb{R}^{B\times T\times D}.$$

意思是：每个 ID 选取 E 的一行。取出的 D 维浮点向量才是后续模型计算的输入。D 个分量是共同学习的表示坐标，没有预设的“第 0 维表示性别、第 1 维表示颜色”等固定语义。

### 一个能手算的例子

```text
E = [[0.0, 0.1],    # token 0
     [1.0, 1.1],    # token 1
     [2.0, 2.1],    # token 2
     [3.0, 3.1]]    # token 3

input_ids = [[1, 3, 1]]             # [B=1,T=3]
X = [[[1.0,1.1], [3.0,3.1], [1.0,1.1]]]  # [1,3,D=2]
```

相同 ID 在不同位置查到相同向量。**输入查表本身没有上下文信息**；进入后续模型层后，同一 token 在不同句子、不同位置可以得到不同的上下文隐藏态。Dropout 处于训练态且 p>0 时，也可能让同一查表向量出现不同的随机掩码，但那不代表查表读取了上下文。

### 与 one-hot 的数学关系

构造 one-hot 张量 O∈R^(B×T×V)，每个位置只有对应 ID 的分量是 1，则：

$$X=OE.$$

这是数学等价式。实际执行通常直接索引 E 的行，避免构造巨大的 [B,T,V] one-hot 和执行大量无用乘加。

### 参数如何学到：梯度按 ID 汇总

如果 G_(b,t) 是损失传回第 (b,t) 个向量的梯度，单看 embedding 查表路径：

$$\frac{\partial L}{\partial E_{v,:}}=\sum_{(b,t):I_{b,t}=v}G_{b,t,:}.$$

重复 ID 的梯度累加到同一行。没有被查询的行，在这条查表路径上的梯度是零。**这不意味着完整模型训练时其他行一定不变**：本仓库默认和 LM head 共享参数，输出路径也会贡献梯度；优化器的 weight decay 等也与“当前查表梯度是否为零”是不同问题。

## 2. 输入与输出合同

| 对象 | 形状/类型 | 含义 |
|---|---|---|
| `input_ids` | [B,T]，通常 `torch.long` | 合法范围 0≤ID<V；和权重位于兼容设备 |
| `embed_tokens.weight` | [V,D]，浮点 Parameter | 可训练的词表向量 |
| `embed_tokens(input_ids)` | [B,T,D]，与表的 dtype 一致 | 每个位置查出的向量 |
| `dropout(...)` | [B,T,D] | 形状不变；是否随机取决于训练状态和 p |

tokenizer 输出整数，embedding 输出浮点；不要把 token ID 当成连续值送入普通线性层。整数 ID 本身不求梯度，学习的是被引用的浮点参数。

## 3. 真实源码，直接列出并注释

文件：[`model/model_minimind.py`](../model/model_minimind.py)。以下是相关行摘录，保持其执行语义；省略的是其他模块，不是让你把片段直接替换进上游文件。

### 3.1 初始化与输入查表（第 206～207、219 行）

```python
# MiniMindModel.__init__ 内：
self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
# 默认 V=6400，D=768；weight.shape == [6400, 768]
# 没有指定 padding_idx，因此没有自动冻结某一行。

self.dropout = nn.Dropout(config.dropout)
# 当前 config.dropout 默认 0.0；默认情况下不会随机丢弃特征。

# MiniMindModel.forward 内：
hidden_states = self.dropout(self.embed_tokens(input_ids))
# input_ids: [B,T]，整数
# 查表结果: [B,T,D]，浮点
# Dropout 后: [B,T,D]
```

这就是本节输入端的全部核心代码。`nn.Embedding(V,D)` 默认使用稠密梯度存储，不能因为“只查几行”就假设梯度张量或优化器状态也只占几行。

### 3.2 与输入表直接相关的输出权重共享（第 246～248、254 行）

```python
# MiniMindForCausalLM.__init__ 内：
self.lm_head = nn.Linear(self.config.hidden_size,
                         self.config.vocab_size, bias=False)
# nn.Linear 的权重形状是 [out_features, in_features]，即 [V,D]。

if self.config.tie_word_embeddings:
    self.model.embed_tokens.weight = self.lm_head.weight
    # 引用同一个 Parameter，不是复制两个数值相同的独立矩阵。

self.post_init()
# Transformers 的初始化收尾；本节实验会在它之后填入教学权重。

# MiniMindForCausalLM.forward 内：
logits = self.lm_head(hidden_states[:, slice_indices, :])
# 此处 hidden_states 已经过模型层；不是原始查表结果。
# 若保留全部 T 个位置：[B,T,D] -> [B,T,V]。
```

这里只解释共享为什么可行，不展开完整 Forward。设某个位置经过模型后的隐藏态为 h∈R^D，输出打分为：

$$z_v=h^TE_{v,:},\qquad z=hE^T.$$

同一份 E 用在两端：输入时按 ID 取一行，输出时与所有候选行点积。输出是未归一化 logits，不是概率，也不是把 embedding “求逆”。

核对是否真正共享，应检查对象身份：

```python
assert model.model.embed_tokens.weight is model.lm_head.weight
```

只用 `torch.equal` 比较数值不够：两份独立参数也可能恰好数值相同。

## 4. 为什么这样设计，替代实现与显存

| 方案 | 好处 | 代价/注意点 |
|---|---|---|
| 当前查表 | 直接得到可学习的 D 维表示，避免巨大 one-hot | 存储 V×D 参数；访问模式影响带宽利用 |
| one-hot 再乘 E | 便于理解和验证数学等价 | 显式 one-hot 通常浪费空间与计算 |
| 输入/输出不共享 | 两端能学习独立参数 | 相对共享版本多 V×D 参数 |
| 低精度或量化表 | 可减少权重存储/带宽 | 支持程度、质量变化、反量化成本需验证 |
| 词表维度切分 | 多卡分担词表权重 | 引入分片索引、通信和可能的对齐填充 |

默认 V=6400、D=768，输入表有 4,915,200 个参数：

- FP32：19,660,800 字节，即 18.75 MiB。
- FP16/BF16：9,830,400 字节，即 9.375 MiB。
- 对比两端各存一份独立 [V,D]，共享节省一份同样大小的参数存储。

以上只计算参数。训练时还有梯度和优化器状态，不能直接把参数字节数当作总显存。

查表工作量与输出写入量大致按 BTD 增长；显式 dense one-hot 矩阵乘法约需 BTVD 量级工作。输出 LM head 本身仍需与词表所有候选打分，**权重共享不免除输出投影计算**。

Embedding 不创建历史 K/V，不减少 KV Cache。prefill 会查询输入序列的多个位置；常规逐 token decode 查询本轮新输入位置。服务中真实瓶颈还取决于 Attention、FFN、输出头、batch 与长度等，不能由“查表快”推导出整个模型快。

## 5. 最小可运行实验：真实模块 + 可预测权重

脚本：[experiments/02_embedding.py](experiments/02_embedding.py)。构造 V=6、D=4 的 tiny MiniMind，直接调用真实输入表与 LM head；它不执行完整模型 Forward。初始化一个 Block 是为了保持真实模型的参数共享行为，本节不展开 Block。

在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe learning\experiments\02_embedding.py
```

完整实验代码如下（与脚本一致）：

```python
"""Token embedding: real MiniMind modules, deterministic toy weights, CPU only."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch.nn import functional as F
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


def main():
    torch.manual_seed(42)
    config = MiniMindConfig(
        vocab_size=6, hidden_size=4, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1,
        intermediate_size=8, max_position_embeddings=16,
        dropout=0.0, tie_word_embeddings=True,
    )
    model = MiniMindForCausalLM(config).eval()
    embedding = model.model.embed_tokens
    # Each token gets a visibly different row; these are teaching weights.
    table = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 10
    with torch.no_grad():
        embedding.weight.copy_(table)

    ids = torch.tensor([[1, 3, 1], [0, 2, 5]], dtype=torch.long)
    vectors = embedding(ids)  # [B,T] -> [B,T,D], no attention involved.
    indexed = embedding.weight[ids]
    one_hot = F.one_hot(ids, num_classes=6).to(table.dtype)
    torch.testing.assert_close(vectors, indexed)
    torch.testing.assert_close(vectors, one_hot @ embedding.weight)
    torch.testing.assert_close(vectors[0, 0], vectors[0, 2])
    assert vectors.shape == (2, 3, 4)
    assert embedding.padding_idx is None  # Current repo does not freeze a pad row.
    print('IDs shape:', tuple(ids.shape))
    print('embedding shape:', tuple(vectors.shape))
    print('token 1 vector:', vectors[0, 0].detach().tolist())
    print('lookup == indexing == one_hot @ weight: PASS')

    # Embedding-only path: repeated IDs accumulate row gradients.
    model.zero_grad(set_to_none=True)
    embedding(torch.tensor([[1, 3, 1]])).sum().backward()
    grad = embedding.weight.grad
    torch.testing.assert_close(grad[1], torch.full((4,), 2.0))
    torch.testing.assert_close(grad[3], torch.ones(4))
    torch.testing.assert_close(grad[[0, 2, 4, 5]], torch.zeros(4, 4))
    print('embedding-only gradient rows:', grad.abs().sum(-1).nonzero().flatten().tolist())

    # Same Parameter object: input table [V,D] is also output classifier weight.
    assert embedding.weight is model.lm_head.weight
    hidden = torch.ones(1, 1, 4)  # A probe hidden state; not a complete LM forward.
    logits = model.lm_head(hidden)
    torch.testing.assert_close(logits, hidden @ embedding.weight.T)
    print('shared Parameter:', embedding.weight is model.lm_head.weight)
    print('LM head probe shape:', tuple(logits.shape))

    # Output softmax considers every class, so shared rows can receive gradients
    # even when their token IDs were never present at the embedding input.
    model.zero_grad(set_to_none=True)
    F.cross_entropy(model.lm_head(hidden).reshape(1, 6), torch.tensor([0])).backward()
    active_rows = embedding.weight.grad.abs().sum(-1).nonzero().flatten().tolist()
    assert active_rows == list(range(6))
    print('LM-head CE gradient rows:', active_rows)
    print('ALL CHECKS PASSED')


if __name__ == '__main__':
    main()
```

观察点：查表等于索引、等于 one-hot 乘法；ID=1 重复两次，其纯查表梯度是每维 2；只经过 LM head 的交叉熵探针也能把梯度传给共享输入表的其他行。该探针用于隔离输出路径，并非完整语言模型训练。

## 6. 对应 vLLM：从单张表到词表分片

在 [vLLM 的 vocab_parallel_embedding 模块](https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/vocab_parallel_embedding/) 中找 `VocabParallelEmbedding` 和 `ParallelLMHead`。前者提供词表分片查表，后者与输出权重相关，支持权重绑定；实际输出 logits 的执行接口与这里的普通 `nn.Linear` 有区别。

把词表分给多个 rank 后，简单理解是：每个 rank 只负责一段 ID，查询本地行，屏蔽不属于自己的结果，再按配置合并各分片贡献。这将本节的“按 ID 查哪一行”扩展成“哪张卡持有这一行”。这是概念映射，本次没有部署或固定 vLLM 版本。

## 7. 本节三道检查题与答案

均为面试风格模拟题。先作答再展开；不要求今天推导 Attention。

### 题 1：查表和形状

E 的形状为 [6400,768]，输入 IDs 为 [2,5]。输出形状、参数量各是多少？同一 ID 在两个位置查到的原始向量是否相同？

<details><summary>参考答案</summary>

输出 [2,5,768]；参数量 6400×768=4,915,200，与当前 B、T 无关。相同 ID 查到同一行，因此原始向量相同。后续 Dropout 或上下文计算后的表示可以不同。

</details>

### 题 2：梯度追踪

只执行 `embedding([[1,3,1]]).sum().backward()`，没有其他损失、共享输出路径或优化器更新。第 1 行、第 3 行、其他行的梯度各是多少？如果完整模型默认共享 LM head，还能说未出现在输入中的行梯度一定为零吗？

<details><summary>参考答案</summary>

第 1 行每个分量为 2，第 3 行每个分量为 1，其余行是零。完整模型不能做此断言：输出头通常对所有候选计算 logits，标准 softmax 交叉熵会向共享权重的许多其他行传播梯度。区分“纯输入查表路径”与“所有训练路径”。

</details>

### 题 3：权重共享和 Infra

为什么 `nn.Embedding(V,D)` 能和 `nn.Linear(D,V,bias=False)` 共享权重？如何检查真实共享？共享在 FP16 默认配置下省多少参数存储，能否减少 KV Cache 或免去输出投影？

<details><summary>参考答案</summary>

两者参数形状都是 [V,D]。输入查表取行，输出线性变换使用该矩阵的转置，符合两个操作各自的语义。用 `embedding.weight is lm_head.weight` 检查同一 Parameter。相对独立两份权重节省 9.375 MiB 参数存储；不减少 KV 元素数，也不免去 LM head 打分工作。

</details>

下一节：Causal Attention，先从单头 Q/K/V 与三 token 手算开始，再对照源码的 mask 和张量乘法。GQA 另起一节。
