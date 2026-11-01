# 本次仅编写，未运行。
from _shared import *
x = torch.arange(2.).reshape(1, 1, 2, 1)
y = repeat_kv(x, 2)
print(y.flatten())  # [0,0,1,1]；相邻两组查询共享一个 KV 头。
torch.testing.assert_close(y.flatten(), torch.tensor([0.,0.,1.,1.]))
cfg = config()
layer = Attention(cfg).eval()
with torch.no_grad():
    out, cache = layer(torch.randn(1, 3, 32), positions(cfg, 0, 3), use_cache=True)
print('output:', out.shape, 'persistent K:', cache[0].shape)
assert cache[0].shape == (1, 3, 2, 8)
for hkv in [8, 4, 1]:
    print(hkv, 'KV MiB:', 2*8*4096*hkv*96*2/2**20)
