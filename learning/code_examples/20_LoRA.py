# 本次仅编写，未运行。
from _shared import *
from model.model_lora import LoRA
base=torch.nn.Linear(8,8,bias=False)
adapter=LoRA(8,8,rank=2)
base.requires_grad_(False)
x=torch.randn(3,8)
torch.testing.assert_close(base(x)+adapter(x),base(x))
optimizer=torch.optim.AdamW(adapter.parameters(),lr=1e-3)
loss=(base(x)+adapter(x)).square().mean()
loss.backward(); optimizer.step()
merged=base.weight.detach()+adapter.B.weight.detach()@adapter.A.weight.detach()
torch.testing.assert_close(base(x)+adapter(x),F.linear(x,merged),atol=1e-5,rtol=1e-4)
print('adapter parameters:',sum(p.numel() for p in adapter.parameters()))
