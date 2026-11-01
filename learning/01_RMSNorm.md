# 第 1 节：RMSNorm——每个 token 的特征向量如何归一化

建议 60～90 分钟。今天只研究 RMSNorm。读完应能手算输出、解释 `mean(-1)`、说明 FP32 中间计算的意义，并指出它在模型中的使用位置。

源码：[model/model_minimind.py 第 50～60 行](../model/model_minimind.py)。固定版本见 [学习入口](README.md)。以下行号均针对该版本。

## 1. 先明确输入、输出与问题

模型内部传递浮点隐藏向量。假设一个 token 的向量为 x=[3,4]，另一个为 [300,400]。它们方向相同，幅度却相差百倍。归一化减少后续计算对整体幅度的敏感性，保留各维之间的相对关系，再用可学习参数恢复有用的逐维缩放。

输入通常是 `x: [B,T,D]`，输出还是 `[B,T,D]`；参数 `weight: [D]`。每个 token 独立处理，不跨 batch、不跨 token 混合。因此这层本身不会让过去位置读取未来 token。

本仓库还把它用于 `[B,T,H,d]` 的 Q/K 张量，此时参数是 `[d]`，每个头的向量单独归一化。归一化轴永远由实际传入张量的最后一维决定。

## 2. 数学原理：均方根，不是标准差

对 x∈R^D，定义：

$$
m_2=\frac{1}{D}\sum_{j=1}^{D}x_j^2,\qquad
r=\sqrt{m_2+\epsilon},\qquad
y_i=\gamma_i\frac{x_i}{r}.
$$

- `m₂` 是均方；它没有减均值，不是方差。
- ε 防止全零或极小输入时除以零，且位于根号内部。
- γ 是可学习的逐维缩放参数，初值为 1；当前实现没有 β 偏置。
- γ 长度为 D，所有 token 共享同一组 γ。

**手算**：x=[3,4]，γ=[1,1]，先忽略很小的 ε。

$$m_2=(9+16)/2=12.5,\quad r\approx3.535534,$$
$$y\approx[0.848528,1.131371].$$

输出均值约 0.98995，并非 0；输出平方的均值约为 1。更精确地，在 γ 全为 1 时，输出均方为 `m₂/(m₂+ε)`，所以并不严格等于 1。γ 学习后也不保证均方为 1。

### 为什么缩放不太改变结果

对正数 a，忽略 ε 时 RMSNorm(ax)=RMSNorm(x)。保留 ε 后：

$$\frac{ax_i}{\sqrt{a^2m_2+\epsilon}}=\frac{x_i}{\sqrt{m_2+\epsilon/a^2}}.$$

所以这是受 ε 影响的近似尺度不变性。a<0 会额外改变符号；不要不加条件地声称所有缩放都完全不变。加常数偏移也不会一般保持结果。

### 与 LayerNorm 的对比

LayerNorm 的核心是 `(x-mean(x))/sqrt(var(x)+ε)`，再做仿射变换。对 [3,4]，忽略 ε 且权重为 1、偏置为 0，LayerNorm 得到 [-1,1]，RMSNorm 得到上面的两个正数。

RMSNorm 省去中心化，计算更简洁；LayerNorm 显式消除均值。二者对同一输入的行为不同，不能在已有权重中随意互换而期待输出保持一致。替换通常需要训练适配和质量验证。

### 为什么能反向传播（可选深入）

令 `r=sqrt(mean(x²)+ε)`，则：

$$\frac{\partial y_i}{\partial x_j}=\gamma_i\left(\frac{\delta_{ij}}r-\frac{x_ix_j}{D r^3}\right).$$

因此某一维输入影响同一向量的所有输出维度，因为它参与了公共分母。这个耦合只发生在最后一维内部，没有跨 token。归一化有助于控制分支输入尺度，但不保证整个网络绝不会梯度爆炸；优化器、残差、初始化、学习率等仍然重要。

## 3. 对照真实源码与张量形状

当前实现只有以下主体（第 50～60 行）：

```python
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return (self.weight * self.norm(x.float())).type_as(x)
```

对 `x: [2,3,768]`，逐步展开：

| 代码 | 形状 | dtype / 含义 |
|---|---|---|
| `x.float()` | [2,3,768] | 转成 FP32 |
| `x.pow(2)` | [2,3,768] | 逐元素平方 |
| `.mean(-1, keepdim=True)` | [2,3,1] | 每个 token 的均方 |
| `+ eps`、`rsqrt` | [2,3,1] | 分母的倒数 |
| `x * ...` | [2,3,768] | 广播回每个特征 |
| `weight * ...` | [2,3,768] | [768] 广播，逐维学习缩放 |
| `.type_as(x)` | [2,3,768] | 恢复原输入 dtype |

`keepdim=True` 保留最后的长度 1 轴，确保广播语义正确。删除后得到 [2,3]，一般无法和 [2,3,768] 相乘；某些尺寸恰巧匹配时还可能产生错误但不报错的广播。

为什么 `.float()` 出现在平方之前：FP16 中较大的数平方可能溢出。转 FP32 后再做平方和 reduction，可扩大数值范围并改善精度；最后转回输入类型有利于后续低精度计算。它并不能修复输入中已经出现的 Inf/NaN。

**两个 eps 不要混淆**：单独实例化 `RMSNorm(dim)` 时默认 ε=1e-5；模型实际使用 `config.rms_norm_eps`，当前配置默认 1e-6。本实验显式指定 1e-6。

### 它在当前模型中的调用点

只定位，不在本节展开 Attention 或完整 Forward：

| 调用点 | 输入最后一维 | 参数数目 | 作用位置 |
|---|---:|---:|---|
| `input_layernorm`（第 187 行） | hidden_size=768 | 768 | Attention 分支之前 |
| `post_attention_layernorm`（第 188 行） | 768 | 768 | FFN 分支之前；名称不表示 Post-Norm 架构 |
| `q_norm`（第 104 行） | head_dim=96 | 96 | Q 投影 reshape 后、RoPE 前 |
| `k_norm`（第 105 行） | 96 | 96 | K 投影 reshape 后、RoPE 前 |
| `MiniMindModel.norm`（第 209 行） | 768 | 768 | 最后一层 Block 之后 |

`q_norm` 的 [96] 权重共享于所有查询头；不是为 8 个头各建一份 [96]。K 使用独立的 `k_norm`，不和 Q 共享参数。

模型每个 Block 的 norm 参数量是 `2D+2d=1728`，8 层加最后的 D 为 14592。以 FP32 参数存储计为 58368 字节（57 KiB），不含梯度、优化器状态和临时激活；这个计数只包含本节所述归一化参数。

## 4. 最小可运行实验

脚本：[experiments/01_rmsnorm.py](experiments/01_rmsnorm.py)。它直接导入仓库真实 RMSNorm，不复制实现或加载权重。主实验在 CPU、固定 seed、FP32 输入运行，另有 FP16 数值演示。

根目录运行：

```powershell
.\.venv\Scripts\python.exe learning\experiments\01_rmsnorm.py
```

按顺序观察：

1. [3,4] 的输出与手算是否一致，且输出均值不为零。
2. [B,T,D] 与 [B,T,H,d] 的形状均不改变；将其他 token 修改后，目标 token 输出不变。
3. 与独立写出的数学表达式是否吻合；正向放大 10 倍的结果是否近似一致。
4. FP16 输入 [300,400] 直接平方出现 Inf，而实际模块经 FP32 计算得到有限输出。
5. x 与 γ 都获得有限梯度。这里用平方和作为教学探针损失，不是语言模型训练 loss。

关键结果应包括：

```text
manual output: [0.848528..., 1.131370...]
3D shape: (2, 3, 4)
4D shape: (2, 3, 2, 4)
raw fp16 square has inf: True
ALL CHECKS PASSED
```

完整实测输出和依赖版本见 [验证记录](VALIDATION.md)。实验通过只说明这些输入上的数值、轴与梯度符合预期，不代表测出了 GPU 性能。

动手修改：把输入改成 [0,0]；把 γ 改成 [2,0.5]；把 `eps` 改为 1。每次运行前先预测输出变化，再解释为何原有“尺度近似不变”的误差会变化。不要直接改上游模型文件，先在实验脚本中构造新实例。

## 5. 为什么这样设计，以及替代实现

| 方案 | 特点 | 需要验证什么 |
|---|---|---|
| 当前手写 RMSNorm | 公式透明、易调试、FP32 中间计算 | 临时张量、算子 launch、编译是否融合 |
| PyTorch `nn.RMSNorm` | 同类语义的标准接口 | eps、权重、dtype、数值容差和后端 |
| LayerNorm | 减均值再按标准差归一化 | 换层后模型质量，不能视为等价替换 |
| 融合 residual add + RMSNorm | 合并残差相加和归一化 | 返回的 residual、原地修改、精度与内存语义 |

推理代价：计算量 O(BTD)，参数 O(D)，层本身不维护随上下文增长的 KV Cache。Q/K norm 会改变后续 K 的数值，但不改变缓存元素数量。

本实现的平方、均方、乘法与 FP32 转换可能产生中间激活。在 eager 模式下通常涉及多次读写和 launch；最终输出 dtype 低并不表示中间全是低精度。是否提速取决于设备、尺寸、编译和内核实现，不能仅凭少一次减法断言某个固定加速倍数。

## 6. 与 vLLM 的对应关系

vLLM 的 [layernorm 模块](https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/layernorm/) 提供 RMSNorm；查阅时实现支持带 residual 的路径，并调用 RMSNorm 或融合 add+RMSNorm 算子。对应的数学对象与本节相同，但接口、原地更新与执行后端需要另行核对。

迁移阅读练习：在 vLLM 侧找到 RMSNorm 后，追踪三个问题：归约用什么精度？残差在哪里更新？读写了几份全尺寸张量？今天只建立这个映射，不要求安装 vLLM 或写 CUDA kernel。

## 7. 本节检查题（先回答，再展开）

以下为 LLM 算法 / Infra 面试风格模拟题；没有可核验的公司出处，不标注为某厂真题。

### 题 1：手算与设计

输入 [3,4]、γ=[1,1]，忽略 ε。RMSNorm 输出多少？它与 LayerNorm 的差别是什么？为什么不能说输出“均值为 0、方差为 1”？

<details>
<summary>参考答案与评分点（3 分）</summary>

RMS=sqrt(12.5)，输出约 [0.848528,1.131371]（1 分）。LayerNorm 减去均值 3.5 并除以标准差 0.5，得到 [-1,1]（1 分）。RMSNorm 没有中心化，γ 还会逐维缩放，因此不能保证均值为零或方差为一；γ=1、忽略 ε 时保证的是均方为一（1 分）。

</details>

### 题 2：形状与真实源码

Q 形状为 [2,5,8,96]，传入 `RMSNorm(96)`。均方张量和输出是什么形状？总共多少可学习参数？不同 token 或 head 是否共同参与分母计算？

<details>
<summary>参考答案与评分点（3 分）</summary>

均方为 [2,5,8,1]、输出为 [2,5,8,96]（1 分）。参数只有 96 个，所有 token/head 共享这组 γ（1 分）。每个 batch、token、head 独立计算自己的 96 维均方，不跨这些轴归约；共享参数不等于共享分母（1 分）。

</details>

### 题 3：数值与性能追问

为什么当前实现先 `.float()` 再平方、最后 `.type_as(x)`？它会减少 KV Cache 吗？如果把残差相加与 RMSNorm 融合，主要可能省下什么？

<details>
<summary>参考答案与评分点（3 分）</summary>

FP32 中间计算扩大平方/归约的数值范围，之后恢复输入 dtype；不能修复已存在的 Inf（1 分）。该层不减少 Hkv、S 或 d，所以不会减少 cache 元素数量（1 分）。融合可能减少 launch 与中间张量的全局内存读写，收益需基准验证；须保持残差返回值和精度语义（1 分）。

</details>

达到 7/9 分且运行实验后，把你的三个答案写进学习日志；不足则针对出错点重跑。下一节：[Token Embedding 的查表与形状](02_Embedding.md)，之后进入你重点关注的 Attention。
