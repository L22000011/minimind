# 本次仅编写，未运行。
from _shared import *
rewards=torch.tensor([[1.,2.,3.],[4.,4.,4.]])
adv=((rewards-rewards.mean(1,keepdim=True))/(rewards.std(1,keepdim=True,unbiased=False)+1e-4)).flatten()
old=torch.full((6,3),-1.)
ref=torch.full((6,3),-1.1)
mask=torch.tensor([[1,1,0]]*6,dtype=torch.float32)
for mode in ['grpo','cispo']:
    new=torch.full((6,3),-0.9,requires_grad=True)
    ratio=(new-old).exp(); diff=ref-new
    kl=diff.exp()-diff-1
    if mode=='cispo':
        objective=ratio.clamp(max=1.2).detach()*adv[:,None]*new
    else:
        objective=torch.minimum(ratio*adv[:,None],ratio.clamp(.8,1.2)*adv[:,None])
    loss=(((-objective+0.04*kl)*mask).sum(1)/mask.sum(1).clamp(min=1)).mean()
    loss.backward()
    print(mode,loss.detach(),new.grad)
