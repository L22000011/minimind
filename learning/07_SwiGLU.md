# 07_SwiGLU

```python
# 本次仅编写，未运行；以下公式/形状/性能说明为教学注释。
# 输入/输出:[B,T,D]；中间 gate/up:[B,T,I]。
# y=Wd(SiLU(Wg x) * Wu x)，SiLU(z)=z*sigmoid(z)。
# 默认 I=ceil(D*pi/64)*64，D=768 时 I=2432。
# 三个无 bias 投影参数=3DI；逐 token 变换，不混合序列 T 轴。
# 替代：GELU/ReLU 两层 FFN；比较时应控制参数量而非只控制 I。
# 性能：大矩阵乘、门控中间激活 O(BTI)；融合 gate/up/activation 可能降低访存。
# vLLM 映射：门控激活与并行线性层；不负责持久 KV cache。
```

```python
# 真实源码：model/model_minimind.py:136，FeedForward
# 以下保持当前源码原样；与上方注释及下方教学例区分。
class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig, intermediate_size: int = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

```powershell
# 仅记录将来手动执行命令，本次没有执行。
.\.venv\Scripts\python.exe learning\code_examples\07_SwiGLU.py
```

```python
# 独立教学示例；辅助模块见code_examples/_shared.py。
# 本次仅编写，未运行。
from _shared import *
cfg = config()
ffn = FeedForward(cfg).eval()
x = torch.randn(2,3,32)
y = ffn(x)
manual = ffn.down_proj(F.silu(ffn.gate_proj(x))*ffn.up_proj(x))
torch.testing.assert_close(y, manual)
changed=x.clone(); changed[:,2]+=100
torch.testing.assert_close(ffn(changed)[:,:2], y[:,:2])
assert sum(p.numel() for p in ffn.parameters()) == 3*32*64
print(y.shape)
```

```python
# Q1: FFN 会混合 token 吗？
# Q2: 为什么有三份矩阵？
# Q3: 扩大 I 的代价？
```

<details>
<summary>参考答案：先独立回答再展开</summary>

```python
# A1: Dense FFN 独立处理各 token，跨 token 交互由 Attention 提供。
# A2: 两份升维分别生成门与内容，逐元素相乘，再由第三份降维。
# A3: 参数、矩阵乘工作量和中间激活都随 I 增长。
```

</details>
