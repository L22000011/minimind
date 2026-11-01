# 本次仅编写，未运行。
from _shared import *
model=tiny_model().eval()
prompt=torch.tensor([[1,7,9]])
# 随机权重只用于调用链学习，不能期待自然语言质量。
output=model.generate(input_ids=prompt,max_new_tokens=4,do_sample=False,
                      top_k=0,top_p=1.0,temperature=1.0,
                      eos_token_id=None,use_cache=True)
print('prompt:',prompt)
print('generated IDs:',output)
# 这里只是单请求生成，无HTTP、调度器、分页KV或GPU性能测量。
