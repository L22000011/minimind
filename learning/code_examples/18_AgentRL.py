# 本次仅编写，未运行。
from _shared import *
# 轨迹位置教学，不执行工具，也不请求外部服务。
# prompt [1,7]，assistant动作[8,9]，tool观察[20,21]，assistant动作[10,2]。
ids=torch.tensor([[1,7,8,9,20,21,10,2]])
action_mask=torch.tensor([[0,0,1,1,0,0,1,1]],dtype=torch.bool)
model=tiny_model().eval()
logits=model(ids).logits[:,:-1]
token_logp=logits.log_softmax(-1).gather(-1,ids[:,1:,None]).squeeze(-1)
mask=action_mask[:,1:]
advantage=torch.tensor([1.])
loss=-(token_logp*mask*advantage[:,None]).sum()/mask.sum()
loss.backward()
print('policy-target positions:',mask)
# 这是固定轨迹REINFORCE探针；完整Agent GRPO/CISPO见真实文件。
