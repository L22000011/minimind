# 19_MoE

```python
# 本次仅编写，未运行；以下公式/形状/性能说明为教学注释。
# x:[B,T,D] -> flatten:[N,D] -> router:[N,E] -> topk:[N,K] -> expert加权散射 -> [B,T,D]。
# y_n=sum_k p_(n,k)*Expert_(idx(n,k))(x_n)，当前没有shared expert。
# top1+norm_topk_prob时权重前向为1，使用straight-through项给router传梯度。
# aux_loss根据专家负载与平均路由概率构造，防止长期集中少数专家；不保证完美均衡。
# active参数少不代表全部权重无需驻留；专家不活跃时仍有总参数存储。
# 替代：Dense、不同top-k/shared experts；多卡expert parallel需考虑all-to-all。
# vLLM映射：fused MoE/expert dispatch；本地Python专家for循环不是高效部署内核。
```

```python
# 真实源码：model/model_minimind.py:148，MOEFeedForward
# 以下保持当前源码原样；与上方注释及下方教学例区分。
class MOEFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([FeedForward(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.num_experts)])
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)
        scores = F.softmax(self.gate(x_flat), dim=-1)
        topk_weight, topk_idx = torch.topk(scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False)
        if self.config.norm_topk_prob:
            if self.config.num_experts_per_tok > 1: 
                topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
            else:
                top1 = torch.topk(F.softmax(self.gate(x_flat.detach()), dim=-1), k=1, dim=-1, sorted=False)[0]
                topk_weight = top1 - top1.detach() + 1.0 # k=1: 1.0 forward on purpose, gradient via straight-through
        y = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (topk_idx == i)
            if mask.any():
                token_idx = mask.any(dim=-1).nonzero().flatten()
                weight = topk_weight[mask].view(-1, 1)
                y.index_add_(0, token_idx, (expert(x_flat[token_idx]) * weight).to(y.dtype))
            elif self.training:
                y[0, 0] += 0 * sum(p.sum() for p in expert.parameters())
        if self.training and self.config.router_aux_loss_coef > 0:
            load = F.one_hot(topk_idx, self.config.num_experts).float().mean(0)
            self.aux_loss = (load * scores.mean(0)).sum() * self.config.num_experts * self.config.router_aux_loss_coef
        else:
            self.aux_loss = scores.new_zeros(1).squeeze()
        return y.view(batch_size, seq_len, hidden_dim)
```

```powershell
# 仅记录将来手动执行命令，本次没有执行。
.\.venv\Scripts\python.exe learning\code_examples\19_MoE.py
```

```python
# 独立教学示例；辅助模块见code_examples/_shared.py。
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
```

```python
# Q1: 激活参数少是否等于显存同样减少？
# Q2: top1权重归一化为1后router怎么学？
# Q3: MoE推理瓶颈？
```

<details>
<summary>参考答案：先独立回答再展开</summary>

```python
# A1: 不是，通常仍需存储全部专家权重，另有路由/通信开销。
# A2: 当前用straight-through表达式与辅助loss传递梯度，直接常量1会丢失该路径。
# A3: 专家负载不均、dispatch/gather、通信、碎片化小矩阵乘，不能只看FLOPs。
```

</details>
