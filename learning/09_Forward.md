# 09_Forward

```python
# 本次仅编写，未运行；以下公式/形状/性能说明为教学注释。
# IDs:[B,T] -> embedding:[B,T,D] -> L Blocks -> final norm -> logits:[B,T,V]。
# 返回 MoeCausalLMOutputWithPast，含 logits/loss/aux_loss/past_key_values/hidden_states。
# logits_to_keep=0：slice(-0,None)=slice(0,None)，保留全部位置。
# logits_to_keep=1 只减少 LM head 位置数，不跳过前面的模型层计算。
# 计算全序列 labels loss 时不要随意裁剪 logits 造成错位或空张量。
# 替代：分离 prefill/decode 执行接口；vLLM 在模型执行器侧组织输入与输出位置。
# 性能：全量 logits O(BTV)，可能明显大于 O(BTD) 隐藏态；cache 另计。
```

```python
# 真实源码：model/model_minimind.py:201，MiniMindModel
# 以下保持当前源码原样；与上方注释及下方教学例区分。
class MiniMindModel(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniMindBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.head_dim, end=config.max_position_embeddings, rope_base=config.rope_theta, rope_scaling=config.rope_scaling)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, **kwargs):
        batch_size, seq_length = input_ids.shape
        if hasattr(past_key_values, 'layers'): past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        hidden_states = self.dropout(self.embed_tokens(input_ids))
        # Recompute RoPE buffers lost during meta-device init (transformers>=5.x)
        if self.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(dim=self.config.head_dim, end=self.config.max_position_embeddings, rope_base=self.config.rope_theta, rope_scaling=self.config.rope_scaling)
            self.freqs_cos, self.freqs_sin = freqs_cos.to(hidden_states.device), freqs_sin.to(hidden_states.device)
        position_embeddings = (self.freqs_cos[start_pos:start_pos + seq_length], self.freqs_sin[start_pos:start_pos + seq_length])
        presents = []
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)
        hidden_states = self.norm(hidden_states)
        aux_loss = sum([l.mlp.aux_loss for l in self.layers if isinstance(l.mlp, MOEFeedForward)], hidden_states.new_zeros(1).squeeze())
        return hidden_states, presents, aux_loss
```

```python
# 真实源码：model/model_minimind.py:250，MiniMindForCausalLM.forward
# 以下保持当前源码原样；与上方注释及下方教学例区分。
    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False, logits_to_keep=0, labels=None, **kwargs):
        hidden_states, past_key_values, aux_loss = self.model(input_ids, attention_mask, past_key_values, use_cache, **kwargs)
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            x, y = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
            loss = F.cross_entropy(x.view(-1, x.size(-1)), y.view(-1), ignore_index=-100)
        return MoeCausalLMOutputWithPast(loss=loss, aux_loss=aux_loss, logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)
```

```powershell
# 仅记录将来手动执行命令，本次没有执行。
.\.venv\Scripts\python.exe learning\code_examples\09_Forward.py
```

```python
# 独立教学示例；辅助模块见code_examples/_shared.py。
# 本次仅编写，未运行。
from _shared import *
model=tiny_model().eval()
ids=torch.tensor([[1,2,3],[4,5,6]])
handles=[]
def hook(name):
    def report(module, inputs, output):
        value=output[0] if isinstance(output,tuple) else output
        print(name,tuple(value.shape))
    return report
for name,layer in [('embedding',model.model.embed_tokens),
                   ('block0',model.model.layers[0]),('head',model.lm_head)]:
    handles.append(layer.register_forward_hook(hook(name)))
with torch.no_grad():
    full=model(ids)
    last=model(ids,logits_to_keep=1)
torch.testing.assert_close(full.logits[:,-1:],last.logits)
for handle in handles: handle.remove()
```

```python
# Q1: hidden_states 与 logits 区别？
# Q2: 为什么 keep=0 保留全部？
# Q3: 仅最后 logits 是否等于仅算最后 token？
```

<details>
<summary>参考答案：先独立回答再展开</summary>

```python
# A1: 前者最后一维 D 表示隐藏特征；后者最后一维 V 表示词表候选分数。
# A2: Python 中 -0=0，切片从位置 0 到末尾。
# A3: 不是；输入中全部 token 仍经过模型层，仅输出投影裁剪。
```

</details>
