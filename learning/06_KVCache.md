# 06_KVCache

```python
# 本次仅编写，未运行；以下公式/形状/性能说明为教学注释。
# 输入：本轮 IDs:[B,T] 与每层历史 K/V:[B,S_old,Hkv,d]。
# 输出：logits:[B,T,V] 与追加 KV:[B,S_old+T,Hkv,d]。
# prefill 通常处理整段 prompt；普通 decode 每轮只输入新 token。
# cache 复用历史投影与层状态，但每个新 Q 仍读取历史 KV，单步不是 O(1)。
# 当前用 torch.cat 追加，反复分配/复制，不是生产级分页 KV。
# 默认 L=8,Hkv=4,d=96，B=1,S=4096,FP16，持久 cache=48 MiB。
# 替代：预分配连续 buffer、paged cache；分页主要解决分配/寻址/复用问题。
# vLLM 映射：KV blocks、block table、cache manager；不等同于 FlashAttention。
```

```python
# 真实源码：model/model_minimind.py:111，Attention.forward
# 以下保持当前源码原样；与上方注释及下方教学例区分。
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

```python
# 真实源码：model/model_minimind.py:214，MiniMindModel.forward
# 以下保持当前源码原样；与上方注释及下方教学例区分。
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
# 真实源码：model/model_minimind.py:261，MiniMindForCausalLM.generate
# 以下保持当前源码原样；与上方注释及下方教学例区分。
    @torch.inference_mode()
    def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192, temperature=0.85, top_p=0.85, top_k=50, eos_token_id=2, streamer=None, use_cache=True, num_return_sequences=1, do_sample=True, repetition_penalty=1.0, **kwargs):
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
        past_key_values = kwargs.pop("past_key_values", None)
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        if streamer: streamer.put(input_ids.cpu())
        for _ in range(max_new_tokens):
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0
            outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache, **kwargs)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1) if attention_mask is not None else None
            logits = outputs.logits[:, -1, :] / temperature
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):
                    seen = torch.unique(input_ids[i]); score = logits[i, seen]; logits[i, seen] = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)
            if top_k > 0: 
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
            if eos_token_id is not None: next_token = torch.where(finished.unsqueeze(-1), next_token.new_full((next_token.shape[0], 1), eos_token_id), next_token)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values if use_cache else None
            if streamer: streamer.put(next_token.cpu())
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all(): break
        if streamer: streamer.end()
        if kwargs.get("return_kv"): return {'generated_ids': input_ids, 'past_kv': past_key_values}
        return input_ids
```

```powershell
# 仅记录将来手动执行命令，本次没有执行。
.\.venv\Scripts\python.exe learning\code_examples\06_KVCache.py
```

```python
# 独立教学示例；辅助模块见code_examples/_shared.py。
# 本次仅编写，未运行。
from _shared import *
model = tiny_model().eval()
ids = torch.tensor([[1,7,9,4,2]])
with torch.inference_mode():
    full = model(ids).logits
    past, parts = None, []
    for t in range(ids.size(1)):
        result = model(ids[:,t:t+1], past_key_values=past, use_cache=True)
        past = result.past_key_values
        parts.append(result.logits)
        print(t, [tuple(k.shape) for k,v in past])
    torch.testing.assert_close(full, torch.cat(parts,dim=1), atol=1e-5, rtol=1e-4)
    # cached chunk T>1：也必须正确处理历史部分与本轮三角遮挡。
    pre = model(ids[:,:2], use_cache=True)
    chunk = model(ids[:,2:], past_key_values=pre.past_key_values, use_cache=True)
    torch.testing.assert_close(full[:,2:], chunk.logits, atol=1e-5, rtol=1e-4)
print('cache bytes:', sum(z.numel()*z.element_size() for kv in past for z in kv))
```

```python
# Q1: 为什么只缓存 K/V 而不累计历史 Q？
# Q2: 什么时候比较 cache 等价？
# Q3: 权重更新后旧 cache 可以复用吗？
```

<details>
<summary>参考答案：先独立回答再展开</summary>

```python
# A1: 新输出只需要新 Q 查询历史 K/V，历史 Q 对未来查询没有直接用途。
# A2: eval、dropout 关闭、相同位置与mask、相同权重，比较 logits 并使用合理浮点容差。
# A3: 通常不能，旧 KV 来源于旧参数；跨请求复用还必须核对相同前缀和模型状态。
```

</details>
