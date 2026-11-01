# 10_Loss

```python
# 本次仅编写，未运行；以下公式/形状/性能说明为教学注释。
# 输入 logits:[B,T,V]，labels:[B,T]；输出有效 next-token 位置的平均交叉熵标量。
# L=-mean(log softmax(logits[b,t])[labels[b,t+1]])，忽略 label=-100。
# 当前 Dataset 不提前shift，forward 内只shift一次；首token无前驱，末logit无本序列目标。
# label mask 不等于 attention mask；忽略 user loss 仍可读取 user 上下文。
# 全部目标为 -100 时 mean loss 可能 NaN，调用侧需检查有效监督数量。
# 替代：label smoothing、token/sequence 不同归约；会改变优化目标。
# vLLM 常规生成不计算训练 CE；训练需要保留反传图和可能很大的 logits。
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
.\.venv\Scripts\python.exe learning\code_examples\10_Loss.py
```

```python
# 独立教学示例；辅助模块见code_examples/_shared.py。
# 本次仅编写，未运行。
from _shared import *
model=tiny_model().eval()
ids=torch.tensor([[1,7,9,2]])
labels=ids.clone(); labels[:,0]=-100
out=model(ids,labels=labels)
logp=out.logits[:,:-1].log_softmax(-1)
targets=labels[:,1:]
valid=targets.ne(-100)
safe=targets.masked_fill(~valid,0)  # gather 不能直接索引 -100。
nll=-logp.gather(-1,safe.unsqueeze(-1)).squeeze(-1)
manual=(nll*valid).sum()/valid.sum()
torch.testing.assert_close(out.loss,manual)
print(out.loss)
# input [A,B,C] 对应 logits[A]->B、logits[B]->C；不要在 Dataset 再 shift。
```

```python
# Q1: next-token shift 在哪里？
# Q2: -100 会禁止 Attention 读取吗？
# Q3: PPL 如何得到？
```

<details>
<summary>参考答案：先独立回答再展开</summary>

```python
# A1: MiniMindForCausalLM.forward 内：logits去掉末位置，labels去掉首位置。
# A2: 不会，它只是 CE 的 ignore_index。
# A3: 相同评测分词与自然对数 NLL 定义下 exp(平均NLL)，不能跨不同tokenizer直接公平比较。
```

</details>
