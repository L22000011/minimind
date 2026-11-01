# 本次仅编写，未运行。
from _shared import *
# 教学合成数据过拟合，不代表训练出通用语言能力，也不替代真实数据验证。
model=tiny_model().train()
optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
ids=torch.tensor([[1,7,8,9,2],[1,7,8,9,2]])
labels=ids.clone()
for step in range(20):
    result=model(ids,labels=labels)
    loss=result.loss+result.aux_loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    print(step,loss.detach().item())
# 如需手动保存，使用独立学习目录；本次不执行本脚本或写任何checkpoint。
# torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict()}, 'learning/toy_checkpoint.pt')
