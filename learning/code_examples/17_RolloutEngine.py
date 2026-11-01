# 本次仅编写，未运行。
from _shared import *
model=tiny_model().eval()
ids=torch.tensor([[1,7,9,2,4]])
n_keep=2
with torch.no_grad():
    logits=model(ids,logits_to_keep=n_keep+1).logits[:,:-1]
    logps=logits.log_softmax(-1).gather(-1,ids[:,-n_keep:,None]).squeeze(-1)
assert logps.shape==(1,2)
print('last response token logps:',logps)
# 如需真正采样，参照真实RolloutEngine；本探针没有调用远端或模拟返回伪造old logp。
