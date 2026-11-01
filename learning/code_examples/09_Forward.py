# 本次仅编写，未运行。
from _shared import *
model=tiny_model().eval()
ids=torch.tensor([[1,2,3],[4,5,6]])
handles=[]
def hook(name):
    def report(module, inputs, output):
        value=output[0] if isinstance(output,tuple) else output
        print(name,tuple(value.shape))
    return report
for name,layer in [('embedding',model.model.embed_tokens),
                   ('block0',model.model.layers[0]),('head',model.lm_head)]:
    handles.append(layer.register_forward_hook(hook(name)))
with torch.no_grad():
    full=model(ids)
    last=model(ids,logits_to_keep=1)
torch.testing.assert_close(full.logits[:,-1:],last.logits)
for handle in handles: handle.remove()
