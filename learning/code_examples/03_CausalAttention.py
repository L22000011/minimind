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
