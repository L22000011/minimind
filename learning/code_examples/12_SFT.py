# 本次仅编写，未运行。
from _shared import *
# 最小监督mask探针；ID仅演示，不冒充真实chat模板。
model=tiny_model().train()
ids=torch.tensor([[1,11,12,20,21,2]])
# 假设位置3、4、5为assistant内容与结束符；其余输入仍作为上下文。
labels=torch.full_like(ids,-100)
labels[:,3:]=ids[:,3:]
result=model(ids,labels=labels)
assert labels[:,1:].ne(-100).sum()==3
result.loss.backward()
print('supervised targets:',labels[:,1:])

# 阅读真实模板不需要重训分词器；需要 transformers 及仓库 model/tokenizer 文件。
from transformers import AutoTokenizer
from pathlib import Path
root=Path(__file__).resolve().parents[2]
tokenizer=AutoTokenizer.from_pretrained(root/'model',local_files_only=True)
messages=[{'role':'user','content':'你好'}, {'role':'assistant','content':'你好，有什么可以帮你？'}]
print(tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=False))
