# 04_GQA

```python
# 本次仅编写，未运行；以下公式/形状/性能说明为教学注释。
# 输入 KV:[B,S,Hkv,d] -> 计算用 KV:[B,S,Hq,d]；n_rep=Hq/Hkv 必须为整数。
# MHA:Hkv=Hq；MQA:Hkv=1；GQA:1<Hkv<Hq。
# 默认 Hq=8,Hkv=4,d=96，每个 KV 头服务两个 Q 头。
# 源码在 repeat_kv 前保存 cache，因此持久缓存仍只有 Hkv 个头。
# 持久 KV 字节=2*L*B*S*Hkv*d*bytes；固定其他项，默认相对 MHA 减半。
# expand 是视图，但 expand 后 reshape 可能物化，不能宣称整条操作零复制。
# 替代：支持 GQA 的原生 kernel，避免显式 repeat；输出仍有 Hq 个查询头。
# vLLM 映射：分别传递 query heads 与 KV heads，核对张量并行分片规则。
```

```python
# 真实源码：model/model_minimind.py:86，repeat_kv
# 以下保持当前源码原样；与上方注释及下方教学例区分。
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1: return x
    return (x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim))
```

```python
# 真实源码：model/model_minimind.py:92，Attention.__init__
# 以下保持当前源码原样；与上方注释及下方教学例区分。
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.num_key_value_heads = config.num_attention_heads if config.num_key_value_heads is None else config.num_key_value_heads
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.head_dim
        self.is_causal = True
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn
```

```powershell
# 仅记录将来手动执行命令，本次没有执行。
.\.venv\Scripts\python.exe learning\code_examples\04_GQA.py
```

```python
# 独立教学示例；辅助模块见code_examples/_shared.py。
# 本次仅编写，未运行。
from _shared import *
x = torch.arange(2.).reshape(1, 1, 2, 1)
y = repeat_kv(x, 2)
print(y.flatten())  # [0,0,1,1]；相邻两组查询共享一个 KV 头。
torch.testing.assert_close(y.flatten(), torch.tensor([0.,0.,1.,1.]))
cfg = config()
layer = Attention(cfg).eval()
with torch.no_grad():
    out, cache = layer(torch.randn(1, 3, 32), positions(cfg, 0, 3), use_cache=True)
print('output:', out.shape, 'persistent K:', cache[0].shape)
assert cache[0].shape == (1, 3, 2, 8)
for hkv in [8, 4, 1]:
    print(hkv, 'KV MiB:', 2*8*4096*hkv*96*2/2**20)
```

```python
# Q1: 默认 cache 节省比例？
# Q2: GQA 总 FLOPs 也减半吗？
# Q3: repeat 后为什么还省 cache？
```

<details>
<summary>参考答案：先独立回答再展开</summary>

```python
# A1: 固定其他维度，Hkv 从 8 到 4，持久 KV 元素数减半。
# A2: 不能如此推断；Q 头数不变，Attention 的查询头计算仍保留。
# A3: 持久 cache 在重复前建立；临时重复开销与持久存储是两回事。
```

</details>
