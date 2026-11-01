# 13_Serving

```python
# 本次仅编写，未运行；以下公式/形状/性能说明为教学注释。
# 输入 messages/生成参数 -> tokenizer -> IDs -> generate -> token流/HTTP响应。
# 此课是模型到服务边界；完整 HTTP 源码附在下方，仅阅读，不启动服务。
# TTFT=请求至首token耗时；TPOT=输出token间耗时，常需单独计入排队与prefill。
# 当前单请求generate不是continuous batching调度器，HTTP封装不自动提供高并发。
# 替代：vLLM等执行引擎；先核对导出architecture、tokenizer、dtype和具体版本支持。
# 当前convert_model.py提供Qwen3/Qwen3-MoE映射；不要把原生自定义类名直接当作兼容证明。
# 并发显存包含权重、每请求KV、临时激活；取消和结束时须回收请求资源。
# 示例只展示本地greedy生成；代码不发网络请求、不启动服务。
```

```python
# 真实源码：model/model_minimind.py:261，MiniMindForCausalLM.generate
# 以下保持当前源码原样；与上方注释及下方教学例区分。
    @torch.inference_mode()
    def generate(self, inputs=None, attention_mask=None, max_new_tokens=8192, temperature=0.85, top_p=0.85, top_k=50, eos_token_id=2, streamer=None, use_cache=True, num_return_sequences=1, do_sample=True, repetition_penalty=1.0, **kwargs):
        input_ids = kwargs.pop("input_ids", inputs).repeat(num_return_sequences, 1)
        attention_mask = attention_mask.repeat(num_return_sequences, 1) if attention_mask is not None else None
        past_key_values = kwargs.pop("past_key_values", None)
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        if streamer: streamer.put(input_ids.cpu())
        for _ in range(max_new_tokens):
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0
            outputs = self.forward(input_ids[:, past_len:], attention_mask, past_key_values, use_cache=use_cache, **kwargs)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones(attention_mask.shape[0], 1)], -1) if attention_mask is not None else None
            logits = outputs.logits[:, -1, :] / temperature
            if repetition_penalty != 1.0:
                for i in range(input_ids.shape[0]):
                    seen = torch.unique(input_ids[i]); score = logits[i, seen]; logits[i, seen] = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)
            if top_k > 0: 
                logits[logits < torch.topk(logits, top_k)[0][..., -1, None]] = -float('inf')
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                mask = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
                mask[..., 1:], mask[..., 0] = mask[..., :-1].clone(), 0
                logits[mask.scatter(1, sorted_indices, mask)] = -float('inf')
            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1) if do_sample else torch.argmax(logits, dim=-1, keepdim=True)
            if eos_token_id is not None: next_token = torch.where(finished.unsqueeze(-1), next_token.new_full((next_token.shape[0], 1), eos_token_id), next_token)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values if use_cache else None
            if streamer: streamer.put(next_token.cpu())
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if finished.all(): break
        if streamer: streamer.end()
        if kwargs.get("return_kv"): return {'generated_ids': input_ids, 'past_kv': past_key_values}
        return input_ids
```

```python
# 真实源码：scripts/serve_openai_api.py:1，完整文件
# 以下保持当前源码原样；与上方注释及下方教学例区分。
import argparse
import json
import re
import os
import sys

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import torch
import warnings
import uvicorn

from threading import Thread
from queue import Queue
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import apply_lora, load_lora

warnings.filterwarnings('ignore')

app = FastAPI()


def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.load_from)
    if os.path.basename(os.path.normpath(args.load_from)) == 'model':
        moe_suffix = '_moe' if args.use_moe else ''
        ckp = f'../{args.save_dir}/{args.weight}_{args.hidden_size}{moe_suffix}.pth'
        model = MiniMindForCausalLM(MiniMindConfig(
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            max_seq_len=args.max_seq_len,
            use_moe=bool(args.use_moe),
            inference_rope_scaling=args.inference_rope_scaling
        ))
        model.load_state_dict(torch.load(ckp, map_location=device), strict=True)
        if args.lora_weight != 'None':
            apply_lora(model, rank=args.lora_rank)
            load_lora(model, f'../{args.save_dir}/{args.lora_weight}_{args.hidden_size}{moe_suffix}.pth')
    else:
        model = AutoModelForCausalLM.from_pretrained(args.load_from, trust_remote_code=True)
    print(f'MiniMind模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M(illion)')
    return model.half().eval().to(device), tokenizer


class ChatRequest(BaseModel):
    model: str
    messages: list
    temperature: float = 0.7
    top_p: float = 0.92
    max_tokens: int = 8192
    stream: bool = True
    tools: list = Field(default_factory=list)
    open_thinking: bool = False
    chat_template_kwargs: dict = None
    
    def get_open_thinking(self) -> bool:
        """兼容多种方式开启 thinking"""
        if self.open_thinking:
            return True
        if self.chat_template_kwargs:
            return self.chat_template_kwargs.get('open_thinking', False) or \
                   self.chat_template_kwargs.get('enable_thinking', False)
        return False


class CustomStreamer(TextStreamer):
    def __init__(self, tokenizer, queue):
        super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
        self.queue = queue
        self.tokenizer = tokenizer

    def on_finalized_text(self, text: str, stream_end: bool = False):
        self.queue.put(text)
        if stream_end:
            self.queue.put(None)


def parse_response(text):
    reasoning_content = None
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if think_match:
        reasoning_content = think_match.group(1).strip()
        text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
    elif '</think>' in text:
        parts = text.split('</think>', 1)
        reasoning_content = parts[0].strip()
        text = parts[1].strip() if len(parts) > 1 else ''
    tool_calls = []
    for i, m in enumerate(re.findall(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)):
        try:
            call = json.loads(m.strip())
            tool_calls.append({"id": f"call_{int(time.time())}_{i}", "type": "function", "function": {"name": call.get("name", ""), "arguments": json.dumps(call.get("arguments", {}), ensure_ascii=False)}})
        except Exception:
            pass
    if tool_calls:
        text = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL)
    return text.strip(), reasoning_content, tool_calls or None


def generate_stream_response(messages, temperature, top_p, max_tokens, tools=None, open_thinking=False):
    try:
        new_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=tools or None, open_thinking=open_thinking)
        inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)

        queue = Queue()
        streamer = CustomStreamer(tokenizer, queue)

        def _generate():
            try:
                model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    attention_mask=inputs.attention_mask,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    streamer=streamer
                )
            except Exception as e:
                queue.put({"error": str(e)})
                queue.put(None)

        Thread(target=_generate).start()

        pending = ""
        in_tool_call = False
        tool_count = 0
        thinking_ended = not bool(open_thinking)
        strip_newlines = False

        while True:
            text = queue.get()
            if text is None:
                break
            if isinstance(text, dict):
                yield json.dumps(text, ensure_ascii=False)
                continue
            pending += text

            while pending:
                if strip_newlines:
                    pending = pending.lstrip('\n')
                    if not pending:
                        break
                    strip_newlines = False
                if in_tool_call:
                    end = pending.find('</tool_call>')
                    if end < 0:
                        break
                    end += len('</tool_call>')
                    block, pending = pending[:end], pending[end:]
                    _, _, tool_calls = parse_response(block)
                    if tool_calls:
                        call = tool_calls[0]
                        call['index'] = tool_count
                        call['id'] = f"call_{int(time.time())}_{tool_count}"
                        tool_count += 1
                        yield json.dumps({"choices": [{"delta": {"tool_calls": [call]}}]}, ensure_ascii=False)
                    else:
                        yield json.dumps({"choices": [{"delta": {"content": block}}]}, ensure_ascii=False)
                    in_tool_call = False
                    continue

                marker = '<tool_call>' if thinking_ended else '</think>'
                field = 'content' if thinking_ended else 'reasoning_content'
                pos = pending.find(marker)
                if pos < 0:
                    keep = next((n for n in range(len(marker) - 1, 0, -1)
                                 if pending.endswith(marker[:n])), 0)
                    end = len(pending) - keep
                    if end:
                        yield json.dumps({"choices": [{"delta": {field: pending[:end]}}]}, ensure_ascii=False)
                        pending = pending[end:]
                    break
                if pos:
                    yield json.dumps({"choices": [{"delta": {field: pending[:pos]}}]}, ensure_ascii=False)
                pending = pending[pos:]
                if thinking_ended:
                    in_tool_call = True
                else:
                    pending = pending[len(marker):]
                    thinking_ended = True
                    strip_newlines = True

        if pending:
            field = 'content' if thinking_ended else 'reasoning_content'
            yield json.dumps({"choices": [{"delta": {field: pending}}]}, ensure_ascii=False)
        yield json.dumps({"choices": [{"delta": {}, "finish_reason": "tool_calls" if tool_count else "stop"}]}, ensure_ascii=False)

    except Exception as e:
        yield json.dumps({"error": str(e)})


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    try:
        if request.stream:
            return StreamingResponse(
                (f"data: {chunk}\n\n" for chunk in generate_stream_response(
                    messages=request.messages,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    max_tokens=request.max_tokens,
                    tools=request.tools,
                    open_thinking=request.get_open_thinking()
                )),
                media_type="text/event-stream"
            )
        else:
            new_prompt = tokenizer.apply_chat_template(
                request.messages,
                tokenize=False,
                add_generation_prompt=True,
                tools=request.tools or None,
                open_thinking=request.get_open_thinking()
            )
            inputs = tokenizer(new_prompt, return_tensors="pt", truncation=True).to(device)
            with torch.no_grad():
                generated_ids = model.generate(
                    inputs["input_ids"],
                    max_new_tokens=request.max_tokens,
                    do_sample=True,
                    attention_mask=inputs["attention_mask"],
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    top_p=request.top_p,
                    temperature=request.temperature
                )
                answer = tokenizer.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            content, reasoning_content, tool_calls = parse_response(answer)
            message = {"role": "assistant", "content": content}
            if reasoning_content:
                message["reasoning_content"] = reasoning_content
            if tool_calls:
                message["tool_calls"] = tool_calls
            return {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": "minimind",
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls" if tool_calls else "stop"
                    }
                ]
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Server for MiniMind")
    parser.add_argument('--load_from', default='../model', type=str, help="模型加载路径（model=原生torch权重，其他路径=transformers格式）")
    parser.add_argument('--save_dir', default='out', type=str, help="模型权重目录")
    parser.add_argument('--weight', default='full_sft', type=str, help="权重名称前缀（pretrain, full_sft, dpo, reason, ppo_actor, grpo, spo）")
    parser.add_argument('--lora_weight', default='None', type=str, help="LoRA权重名称（None表示不使用，可选：lora_identity, lora_medical）")
    parser.add_argument('--lora_rank', default=16, type=int, help="LoRA rank，需与训练时一致（默认16）")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=8192, type=int, help="最大序列长度")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--inference_rope_scaling', default=False, action='store_true', help="启用RoPE位置编码外推（4倍，仅解决位置编码问题）")
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help="运行设备")
    args = parser.parse_args()
    device = args.device
    model, tokenizer = init_model(args)
    uvicorn.run(app, host="0.0.0.0", port=8998)
```

```python
# 真实源码：scripts/convert_model.py:1，完整文件
# 以下保持当前源码原样；与上方注释及下方教学例区分。
import os
import sys
import json

__package__ = "scripts"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import transformers
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.model_lora import apply_lora, merge_lora

warnings.filterwarnings('ignore', category=UserWarning)

def convert_torch2transformers_minimind(torch_path, transformers_path, dtype=torch.float16):
    MiniMindConfig.register_for_auto_class()
    MiniMindForCausalLM.register_for_auto_class("AutoModelForCausalLM")
    lm_model = MiniMindForCausalLM(lm_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(torch_path, map_location=device)
    lm_model.load_state_dict(state_dict, strict=False)
    lm_model = lm_model.to(dtype)  # 转换模型权重精度
    model_params = sum(p.numel() for p in lm_model.parameters() if p.requires_grad)
    print(f'模型参数: {model_params / 1e6} 百万 = {model_params / 1e9} B (Billion)')
    lm_model.save_pretrained(transformers_path, safe_serialization=False)
    tokenizer = AutoTokenizer.from_pretrained('../model/')
    tokenizer.save_pretrained(transformers_path)
    # ======= transformers-5.0的兼容低版本写法 =======
    if int(transformers.__version__.split('.')[0]) >= 5:
        tokenizer_config_path, config_path = os.path.join(transformers_path, "tokenizer_config.json"), os.path.join(transformers_path, "config.json")
        json.dump({**json.load(open(tokenizer_config_path, 'r', encoding='utf-8')), "tokenizer_class": "PreTrainedTokenizerFast", "extra_special_tokens": {}}, open(tokenizer_config_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
        config = json.load(open(config_path, 'r', encoding='utf-8'))
        config['rope_theta'] = lm_config.rope_theta; config['rope_scaling'] = None; del config['rope_parameters']
        json.dump(config, open(config_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f"模型已保存为 Transformers-MiniMind 格式: {transformers_path}")


# QwenForCausalLM/LlamaForCausalLM结构兼容生态
def convert_torch2transformers(torch_path, transformers_path, dtype=torch.float16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(torch_path, map_location=device)
    common_config = {
        "vocab_size": lm_config.vocab_size,
        "hidden_size": lm_config.hidden_size,
        "intermediate_size": lm_config.intermediate_size,
        "num_hidden_layers": lm_config.num_hidden_layers,
        "num_attention_heads": lm_config.num_attention_heads,
        "num_key_value_heads": lm_config.num_key_value_heads,
        "head_dim": lm_config.hidden_size // lm_config.num_attention_heads,
        "max_position_embeddings": lm_config.max_position_embeddings,
        "rms_norm_eps": lm_config.rms_norm_eps,
        "rope_theta": lm_config.rope_theta,
        "tie_word_embeddings": lm_config.tie_word_embeddings
    }
    if not lm_config.use_moe:
        qwen_config = Qwen3Config(
            **common_config, 
            use_sliding_window=False, 
            sliding_window=None
        )
        qwen_model = Qwen3ForCausalLM(qwen_config)
    else:
        qwen_config = Qwen3MoeConfig(
            **common_config,
            num_experts=lm_config.num_experts,
            num_experts_per_tok=lm_config.num_experts_per_tok,
            moe_intermediate_size=lm_config.moe_intermediate_size,
            norm_topk_prob=lm_config.norm_topk_prob
        )
        qwen_model = Qwen3MoeForCausalLM(qwen_config)
        # ======= transformers-5.0的兼容低版本写法 =======
        if int(transformers.__version__.split('.')[0]) >= 5:
            new_sd = {k: v for k, v in state_dict.items() if 'experts.' not in k or 'gate.weight' in k}
            for l in range(lm_config.num_hidden_layers):
                p = f'model.layers.{l}.mlp.experts'
                new_sd[f'{p}.gate_up_proj'] = torch.cat([torch.stack([state_dict[f'{p}.{e}.gate_proj.weight'] for e in range(lm_config.num_experts)]), torch.stack([state_dict[f'{p}.{e}.up_proj.weight'] for e in range(lm_config.num_experts)])], dim=1)
                new_sd[f'{p}.down_proj'] = torch.stack([state_dict[f'{p}.{e}.down_proj.weight'] for e in range(lm_config.num_experts)])
            state_dict = new_sd

    qwen_model.load_state_dict(state_dict, strict=True)
    qwen_model = qwen_model.to(dtype)  # 转换模型权重精度
    qwen_model.save_pretrained(transformers_path)
    model_params = sum(p.numel() for p in qwen_model.parameters() if p.requires_grad)
    print(f'模型参数: {model_params / 1e6} 百万 = {model_params / 1e9} B (Billion)')
    tokenizer = AutoTokenizer.from_pretrained('../model/')
    tokenizer.save_pretrained(transformers_path)

    # ======= transformers-5.0的兼容低版本写法 =======
    if int(transformers.__version__.split('.')[0]) >= 5:
        tokenizer_config_path, config_path = os.path.join(transformers_path, "tokenizer_config.json"), os.path.join(transformers_path, "config.json")
        json.dump({**json.load(open(tokenizer_config_path, 'r', encoding='utf-8')), "tokenizer_class": "PreTrainedTokenizerFast", "extra_special_tokens": {}}, open(tokenizer_config_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
        config = json.load(open(config_path, 'r', encoding='utf-8'))
        config['rope_theta'] = lm_config.rope_theta; config['rope_scaling'] = None; del config['rope_parameters']
        json.dump(config, open(config_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f"模型已保存为 Transformers 格式: {transformers_path}")


def convert_transformers2torch(transformers_path, torch_path):
    model = AutoModelForCausalLM.from_pretrained(transformers_path, trust_remote_code=True)
    torch.save({k: v.cpu().half() for k, v in model.state_dict().items()}, torch_path)
    print(f"模型已保存为 PyTorch 格式: {torch_path}")


def convert_merge_base_lora(base_torch_path, lora_path, merged_torch_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lm_model = MiniMindForCausalLM(lm_config).to(device)
    state_dict = torch.load(base_torch_path, map_location=device)
    lm_model.load_state_dict(state_dict, strict=False)
    apply_lora(lm_model)
    merge_lora(lm_model, lora_path, merged_torch_path)
    print(f"LoRA 已合并并保存为基模结构 PyTorch 格式: {merged_torch_path}")


def convert_jinja_to_json(jinja_path):
    with open(jinja_path, 'r') as f: template = f.read()
    escaped = json.dumps(template)
    print(f'"chat_template": {escaped}')


def convert_json_to_jinja(json_file_path, output_path):
    with open(json_file_path, 'r') as f: config = json.load(f)
    template = config['chat_template']
    with open(output_path, 'w') as f: f.write(template)
    print(f"模板已保存为 jinja 文件: {output_path}")


if __name__ == '__main__':
    lm_config = MiniMindConfig(hidden_size=768, num_hidden_layers=8, max_seq_len=8192, use_moe=False)

    # convert torch to transformers
    torch_path = f"../out/full_sft_{lm_config.hidden_size}{'_moe' if lm_config.use_moe else ''}.pth"
    transformers_path = '../minimind-3'
    convert_torch2transformers(torch_path, transformers_path)

    # # merge lora
    # base_torch_path = f"../out/full_sft_{lm_config.hidden_size}{'_moe' if lm_config.use_moe else ''}.pth"
    # lora_path = f"../out/lora_identity_{lm_config.hidden_size}{'_moe' if lm_config.use_moe else ''}.pth"
    # merged_torch_path = f"../out/merge_identity_{lm_config.hidden_size}{'_moe' if lm_config.use_moe else ''}.pth"
    # convert_merge_base_lora(base_torch_path, lora_path, merged_torch_path)

    # convert_transformers2torch(transformers_path, torch_path)
    # convert_json_to_jinja('../model/tokenizer_config.json', '../model/chat_template.jinja')
    # convert_jinja_to_json('../model/chat_template.jinja')
```

```powershell
# 仅记录将来手动执行命令，本次没有执行。
.\.venv\Scripts\python.exe learning\code_examples\13_Serving.py
```

```python
# 独立教学示例；辅助模块见code_examples/_shared.py。
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
```

```python
# Q1: API服务等于高吞吐引擎吗？
# Q2: 首token慢一定是decode慢吗？
# Q3: 如何迁移vLLM？
```

<details>
<summary>参考答案：先独立回答再展开</summary>

```python
# A1: 不等于，还需要调度、批处理、cache管理、取消与资源回收。
# A2: 不一定，TTFT包含排队、预处理与prefill，须分阶段观察。
# A3: 固定版本并核对导出架构/权重/模板，验证logits或生成一致性后再测吞吐与显存。
```

</details>
