# 本次仅编写，未运行。
from _shared import *
model=tiny_model().eval()
ids=torch.tensor([[1,7,9,2]])
labels=ids.clone(); labels[:,0]=-100
out=model(ids,labels=labels)
logp=out.logits[:,:-1].log_softmax(-1)
targets=labels[:,1:]
valid=targets.ne(-100)
safe=targets.masked_fill(~valid,0)  # gather 不能直接索引 -100。
nll=-logp.gather(-1,safe.unsqueeze(-1)).squeeze(-1)
manual=(nll*valid).sum()/valid.sum()
torch.testing.assert_close(out.loss,manual)
print(out.loss)
# input [A,B,C] 对应 logits[A]->B、logits[B]->C；不要在 Dataset 再 shift。
