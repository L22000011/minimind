# 本次仅编写，未运行。
from _shared import *
model = tiny_model().eval()
ids = torch.tensor([[1,7,9,4,2]])
with torch.inference_mode():
    full = model(ids).logits
    past, parts = None, []
    for t in range(ids.size(1)):
        result = model(ids[:,t:t+1], past_key_values=past, use_cache=True)
        past = result.past_key_values
        parts.append(result.logits)
        print(t, [tuple(k.shape) for k,v in past])
    torch.testing.assert_close(full, torch.cat(parts,dim=1), atol=1e-5, rtol=1e-4)
    # cached chunk T>1：也必须正确处理历史部分与本轮三角遮挡。
    pre = model(ids[:,:2], use_cache=True)
    chunk = model(ids[:,2:], past_key_values=pre.past_key_values, use_cache=True)
    torch.testing.assert_close(full[:,2:], chunk.logits, atol=1e-5, rtol=1e-4)
print('cache bytes:', sum(z.numel()*z.element_size() for kv in past for z in kv))
