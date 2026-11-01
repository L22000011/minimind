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
