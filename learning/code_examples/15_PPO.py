# 本次仅编写，未运行。
from _shared import *
rewards=torch.tensor([[0.,0.,1.]])
values=torch.tensor([[0.1,0.2,0.3]])
gamma,lam=0.99,0.95
adv=torch.zeros_like(values); carry=torch.zeros(1)
for t in reversed(range(3)):
    nxt=values[:,t+1] if t<2 else torch.zeros(1)
    delta=rewards[:,t]+gamma*nxt-values[:,t]
    carry=delta+gamma*lam*carry
    adv[:,t]=carry
returns=adv+values
old=torch.tensor([[-1.,-1.,-1.]])
new=torch.tensor([[-0.8,-1.1,-0.9]],requires_grad=True)
ratio=(new-old).exp()
policy_loss=-torch.minimum(ratio*adv,ratio.clamp(0.8,1.2)*adv).mean()
vnew=values.clone().requires_grad_()
vclip=vnew.clamp(values-0.2,values+0.2)
value_loss=0.5*torch.maximum((vnew-returns).square(),(vclip-returns).square()).mean()
(policy_loss+0.5*value_loss).backward()
print(adv,policy_loss.detach(),value_loss.detach())
