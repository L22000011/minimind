# 05_RoPE

```python
# 本次仅编写，未运行；以下公式/形状/性能说明为教学注释。
# 输入 Q:[B,T,Hq,d]，K:[B,T,Hkv,d]，cos/sin:[T,d]；输出同形状。
# 公式 R_m q 与 R_n k 内积=q^T R_(n-m) k；未额外缩放时旋转保持范数。
# inv_freq_i=theta^(-2i/d)；不同维度对有不同角频率，d 必须为偶数。
# 当前 rotate_half 按前后半维配对，不是交错偶奇配对。
# decode 位置从历史 KV 长度 start_pos 开始；新 Q/K 不能从位置 0 重新旋转。
# YaRN 分支改变频率并可能改变幅度；不能把普通 RoPE 的所有结论无条件搬过去。
# 替代：绝对位置 embedding、ALiBi；不能直接换层后保证权重等价。
# 内存：预计算 cos/sin buffer 为 O(max_position*d)，不是每层历史 KV。
# vLLM 映射：rotary embedding；核对 pairing 布局、theta、position 与 scaling。
```

```python
# 真实源码：model/model_minimind.py:62，precompute_freqs_cis
# 以下保持当前源码原样；与上方注释及下方教学例区分。
def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), rope_base: float = 1e6, rope_scaling: dict = None):
    freqs, attn_factor = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)), 1.0
    if rope_scaling is not None: # YaRN: f'(i) = f(i)((1-γ) + γ/s), where γ∈[0,1] is linear ramp
        orig_max, factor, beta_fast, beta_slow, attn_factor = (
            rope_scaling.get("original_max_position_embeddings", 2048), rope_scaling.get("factor", 16),
            rope_scaling.get("beta_fast", 32.0), rope_scaling.get("beta_slow", 1.0), rope_scaling.get("attention_factor", 1.0)
        )
        if end / orig_max > 1.0:
            inv_dim = lambda b: (dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_base))
            low, high = max(math.floor(inv_dim(beta_fast)), 0), min(math.ceil(inv_dim(beta_slow)), dim // 2 - 1)
            ramp = torch.clamp((torch.arange(dim // 2, device=freqs.device).float() - low) / max(high - low, 0.001), 0, 1)
            freqs = freqs * (1 - ramp + ramp / factor)
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1) * attn_factor
    return freqs_cos, freqs_sin
```

```python
# 真实源码：model/model_minimind.py:80，apply_rotary_pos_emb
# 以下保持当前源码原样；与上方注释及下方教学例区分。
def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    def rotate_half(x): return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed
```

```powershell
# 仅记录将来手动执行命令，本次没有执行。
.\.venv\Scripts\python.exe learning\code_examples\05_RoPE.py
```

```python
# 独立教学示例；辅助模块见code_examples/_shared.py。
# 本次仅编写，未运行。
from _shared import *
cfg = config()
q = torch.randn(1, 4, 4, 8)
k = torch.randn(1, 4, 2, 8)
qr, kr = apply_rotary_pos_emb(q, k, *positions(cfg, 0, 4))
torch.testing.assert_close(q.norm(dim=-1), qr.norm(dim=-1))

# 保持内容向量相同，只平移两个绝对位置，验证相对位移。
a, b = torch.randn(1,1,1,8), torch.randn(1,1,1,8)
def dot_at(m, n):
    qa, _ = apply_rotary_pos_emb(a, a, *positions(cfg,m,1))
    _, kb = apply_rotary_pos_emb(b, b, *positions(cfg,n,1))
    return (qa*kb).sum()
torch.testing.assert_close(dot_at(2,5), dot_at(9,12))
print('norm and relative-position identities defined for verification')
```

```python
# Q1: 为什么通常不旋转 V？
# Q2: 前后半配对能直接改成交错配对吗？
# Q3: cache offset 错了有什么影响？
```

<details>
<summary>参考答案：先独立回答再展开</summary>

```python
# A1: RoPE 通过 Q/K 内积调制位置关系；本实现把 V 保留为聚合内容。
# A2: 要同步处理权重/布局；不能只改 rotate_half 就期待同一预训练模型等价。
# A3: 新 Q/K 与历史 K 的相对角度错误，造成注意力与无缓存结果不一致。
```

</details>
