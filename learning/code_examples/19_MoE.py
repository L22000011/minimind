# 本次仅编写，未运行。
from _shared import *
cfg=config(use_moe=True,num_experts=4,num_experts_per_tok=1)
moe=MOEFeedForward(cfg).train()
x=torch.randn(2,3,32)
y=moe(x)
loss=y.square().mean()+moe.aux_loss
loss.backward()
print(y.shape,moe.aux_loss.detach())
print('router grad norm:',moe.gate.weight.grad.norm())
