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
