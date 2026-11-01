# 本次仅编写，未运行。
from _shared import *
temperature=2.0
student=torch.randn(4,8,requires_grad=True)
teacher=torch.randn(4,8)
mask=torch.tensor([True,True,False,True])
student_logp=F.log_softmax(student[mask]/temperature,-1)
teacher_p=F.softmax(teacher[mask]/temperature,-1).detach()
kd=F.kl_div(student_logp,teacher_p,reduction='batchmean')*temperature**2
ce=F.cross_entropy(student[mask],torch.tensor([1,2,3]))
loss=.5*ce+.5*kd
loss.backward()
print('CE/KD:',ce.detach(),kd.detach())
