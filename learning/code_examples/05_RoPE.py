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
