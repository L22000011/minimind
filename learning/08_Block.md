# 08_Block

```python
# 本次仅编写，未运行；以下公式/形状/性能说明为教学注释。
# 输入隐藏态:[B,T,D]，位置与可选cache；输出同形隐藏态、新cache。
# h=x+Attention(Norm(x))；y=h+MLP(Norm(h))。
# 两个 norm 均位于分支之前，是 Pre-Norm；不能根据 post_attention_layernorm 名字判断架构。
# 残差提供直接信息/梯度通路，归一化控制分支尺度，但均不保证训练必然稳定。
# 替代：Post-Norm、不同 residual scaling；涉及训练和权重行为变化。
# vLLM 映射：residual add+RMSNorm 融合，注意原地修改和残差返回语义。
```

```python
# 真实源码：model/model_minimind.py:183，MiniMindBlock
# 以下保持当前源码原样；与上方注释及下方教学例区分。
class MiniMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniMindConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value
```

```powershell
# 仅记录将来手动执行命令，本次没有执行。
.\.venv\Scripts\python.exe learning\code_examples\08_Block.py
```

```python
# 独立教学示例；辅助模块见code_examples/_shared.py。
# 本次仅编写，未运行。
from _shared import *
cfg = config()
block = MiniMindBlock(0,cfg).eval()
x = torch.randn(1,3,32)
pos = positions(cfg,0,3)
with torch.no_grad():
    result,_ = block(x.clone(),pos)
    attn,_ = block.self_attn(block.input_layernorm(x),pos)
    h = x+attn
    manual = h+block.mlp(block.post_attention_layernorm(h))
torch.testing.assert_close(result,manual)
print(result.shape)
```

```python
# Q1: Pre-Norm 的判据？
# Q2: 输出尺寸为什么不变？
# Q3: 融合时应核对什么？
```

<details>
<summary>参考答案：先独立回答再展开</summary>

```python
# A1: 看 norm 在每个子层分支之前，而不是看变量名。
# A2: Attention 和 MLP 均投影回 D，才能进行逐元素残差相加。
# A3: 输出数值、residual 保存/更新位置、别名和原地操作、dtype。
```

</details>
