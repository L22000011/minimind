# 本次仅编写，未运行。
from _shared import *
# 数学隔离示例；实际数据和训练链路见下方完整仓库源码。
policy=torch.tensor([[-1.,-1.],[-2.,-2.]],requires_grad=True)
reference=torch.tensor([[-1.5,-1.5],[-1.5,-1.5]])
mask=torch.ones_like(policy)
beta=0.1
p=(policy*mask).sum(-1); r=(reference*mask).sum(-1)
margin=(p[0]-p[1])-(r[0]-r[1])
loss=-F.logsigmoid(beta*margin)
loss.backward()
print(loss.detach(),policy.grad)
# 梯度下降推动chosen的logp上升、rejected下降；这不是完整概率模型参数化。
