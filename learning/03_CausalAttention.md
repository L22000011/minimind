# 03_CausalAttention

```python
# 本次仅编写，未运行；以下公式/形状/性能说明为教学注释。
# 输入 x:[B,T,D]，输出 y:[B,T,D]，可选 cache 在后续课次展开。
# 数学：Q=XWq，K=XWk，V=XWv；A=softmax(QK^T/sqrt(d)+M)，Y=AV。
# 本仓库额外对 Q/K 做 RMSNorm，并对 Q/K 做 RoPE，不对 V 做 RoPE。
# Q:[B,Hq,T,d]，K/V:[B,Hq,S,d]，scores:[B,Hq,T,S]。
# 因果 mask 在 softmax 前加，未来位置为 -inf，不能先 softmax 再直接清零。
# cached chunk 时只遮挡 scores 最后 T 列的上三角；历史 S-T 列均可见。
# 替代：SDPA/融合 Attention；flash_attn 配置不保证真实使用 FlashAttention 内核。
# 手写 scores 空间 O(B*Hq*T*S)，全序列 Attention 的 T*S=T²。
# vLLM 映射：Attention backend；本课程不把 MiniMind 的手写实现当作分页内核。
```

```python
# 真实源码：model/model_minimind.py:91，Attention
# 以下保持当前源码原样；与上方注释及下方教学例区分。
class Attention(nn.Module):
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

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xq, xk = self.q_norm(xq), self.k_norm(xk)
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None
        xq, xk, xv = (xq.transpose(1, 2), repeat_kv(xk, self.n_rep).transpose(1, 2), repeat_kv(xv, self.n_rep).transpose(1, 2))
        if self.flash and (seq_len > 1) and (not self.is_causal or past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            output = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, is_causal=self.is_causal)
        else:
            scores = (xq @ xk.transpose(-2, -1)).float() / math.sqrt(self.head_dim)
            if self.is_causal: scores[:, :, :, -seq_len:] += torch.full((seq_len, seq_len), float("-inf"), device=scores.device).triu(1)
            if attention_mask is not None: scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            output = self.attn_dropout(F.softmax(scores, dim=-1).type_as(xq)) @ xv
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv
```

```powershell
# 仅记录将来手动执行命令，本次没有执行。
.\.venv\Scripts\python.exe learning\code_examples\03_CausalAttention.py
```

```python
# 独立教学示例；辅助模块见code_examples/_shared.py。
# 本次仅编写，未运行。
from _shared import *
import math

cfg = config(num_key_value_heads=4)  # 先隔离 GQA，令 Q 与 KV 头数相同。
attn = Attention(cfg).eval()
x = torch.randn(1, 3, cfg.hidden_size)
with torch.no_grad():
    y, _ = attn(x, positions(cfg, 0, 3))
    changed = x.clone()
    changed[:, 2] += 100  # 只修改未来 token。
    y2, _ = attn(changed, positions(cfg, 0, 3))
torch.testing.assert_close(y[:, :2], y2[:, :2])

# 单头三 token 手算版本，独立展示 mask 和 softmax 的轴。
q = torch.eye(3).view(1, 1, 3, 3)
k = q.clone()
v = torch.arange(6.).reshape(1, 1, 3, 2)
scores = q @ k.transpose(-2, -1) / math.sqrt(3)
mask = torch.ones(3, 3, dtype=torch.bool).triu(1)
prob = scores.masked_fill(mask, -torch.inf).softmax(-1)
print('attention probabilities:', prob)
print('output:', prob @ v)
assert prob[0, 0, 0, 1:].sum() == 0
```

```python
# Q1: 为什么除以 sqrt(d)？
# Q2: softmax 沿哪个轴？
# Q3: 单 token decode 能直接 is_causal=True 吗？
```

<details>
<summary>参考答案：先独立回答再展开</summary>

```python
# A1: 独立零均值单位方差的简化假设下，点积方差约 d，缩放控制 logits 尺度。
# A2: 沿最后的 key 轴 S；每个 query 对所有可见 key 的概率和为 1。
# A3: 带历史的非方阵 mask 必须对齐真实位置；不能把无历史三角规则机械套用。本仓库走手写路径。
```

</details>
