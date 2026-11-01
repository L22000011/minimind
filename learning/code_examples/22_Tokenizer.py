# 本次仅编写，未运行。
from pathlib import Path
from transformers import AutoTokenizer
root=Path(__file__).resolve().parents[2]
tokenizer=AutoTokenizer.from_pretrained(root/'model',local_files_only=True)
text='你好，MiniMind！'
ids=tokenizer.encode(text,add_special_tokens=False)
print('IDs:',ids)
print('pieces:',tokenizer.convert_ids_to_tokens(ids))
print('decoded:',tokenizer.decode(ids))
messages=[{'role':'user','content':'解释一下KV Cache'}]
print(tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True))
# 本示例仅使用现有词表；不训练BPE，也不保存分词器文件。
