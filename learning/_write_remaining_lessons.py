"""Document authoring only: read source and write Markdown/Python text, no model execution."""
from pathlib import Path
import ast
import textwrap

ROOT = Path(__file__).resolve().parents[1]
LEARN = ROOT / 'learning'
CODE = LEARN / 'code_examples'
CODE.mkdir(exist_ok=True)

SHARED = '''# 仅供你之后手动执行示例；本次没有运行。
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from torch.nn import functional as F
from model.model_minimind import (
    MiniMindConfig, MiniMindForCausalLM, Attention, FeedForward,
    MiniMindBlock, MOEFeedForward, precompute_freqs_cis,
    apply_rotary_pos_emb, repeat_kv,
)

def config(**overrides):
    values = dict(hidden_size=32, num_hidden_layers=2, vocab_size=64,
                  num_attention_heads=4, num_key_value_heads=2,
                  intermediate_size=64, max_position_embeddings=128,
                  dropout=0.0, flash_attn=False)
    values.update(overrides)
    return MiniMindConfig(**values)

def tiny_model(**overrides):
    torch.manual_seed(42)
    return MiniMindForCausalLM(config(**overrides))

def positions(cfg, start, length):
    cos, sin = precompute_freqs_cis(cfg.head_dim, cfg.max_position_embeddings,
                                   cfg.rope_theta, cfg.rope_scaling)
    return cos[start:start+length], sin[start:start+length]
'''
(CODE / '_shared.py').write_text(SHARED, encoding='utf-8')

# name, annotated contract, exact source selections, standalone teaching code, questions/answers
LESSONS = []
def add(name, notes, sources, example, qa):
    LESSONS.append((name, notes, sources, example, qa))

M = 'model/model_minimind.py'
D = 'dataset/lm_dataset.py'

add('03_CausalAttention', '''
# 输入 x:[B,T,D]，输出 y:[B,T,D]，可选 cache 在后续课次展开。
# 数学：Q=XWq，K=XWk，V=XWv；A=softmax(QK^T/sqrt(d)+M)，Y=AV。
# 本仓库额外对 Q/K 做 RMSNorm，并对 Q/K 做 RoPE，不对 V 做 RoPE。
# Q:[B,Hq,T,d]，K/V:[B,Hq,S,d]，scores:[B,Hq,T,S]。
# 因果 mask 在 softmax 前加，未来位置为 -inf，不能先 softmax 再直接清零。
# cached chunk 时只遮挡 scores 最后 T 列的上三角；历史 S-T 列均可见。
# 替代：SDPA/融合 Attention；flash_attn 配置不保证真实使用 FlashAttention 内核。
# 手写 scores 空间 O(B*Hq*T*S)，全序列 Attention 的 T*S=T²。
# vLLM 映射：Attention backend；本课程不把 MiniMind 的手写实现当作分页内核。
''', [(M, 'Attention')], '''
from _shared import *
import math

cfg = config(num_key_value_heads=4)  # 先隔离 GQA，令 Q 与 KV 头数相同。
attn = Attention(cfg).eval()
x = torch.randn(1, 3, cfg.hidden_size)
with torch.no_grad():
    y, _ = attn(x, positions(cfg, 0, 3))
    changed = x.clone()
    changed[:, 2] += 100  # 只修改未来 token。
    y2, _ = attn(changed, positions(cfg, 0, 3))
torch.testing.assert_close(y[:, :2], y2[:, :2])

# 单头三 token 手算版本，独立展示 mask 和 softmax 的轴。
q = torch.eye(3).view(1, 1, 3, 3)
k = q.clone()
v = torch.arange(6.).reshape(1, 1, 3, 2)
scores = q @ k.transpose(-2, -1) / math.sqrt(3)
mask = torch.ones(3, 3, dtype=torch.bool).triu(1)
prob = scores.masked_fill(mask, -torch.inf).softmax(-1)
print('attention probabilities:', prob)
print('output:', prob @ v)
assert prob[0, 0, 0, 1:].sum() == 0
''', [('为什么除以 sqrt(d)？', '独立零均值单位方差的简化假设下，点积方差约 d，缩放控制 logits 尺度。'), ('softmax 沿哪个轴？', '沿最后的 key 轴 S；每个 query 对所有可见 key 的概率和为 1。'), ('单 token decode 能直接 is_causal=True 吗？', '带历史的非方阵 mask 必须对齐真实位置；不能把无历史三角规则机械套用。本仓库走手写路径。')])

add('04_GQA', '''
# 输入 KV:[B,S,Hkv,d] -> 计算用 KV:[B,S,Hq,d]；n_rep=Hq/Hkv 必须为整数。
# MHA:Hkv=Hq；MQA:Hkv=1；GQA:1<Hkv<Hq。
# 默认 Hq=8,Hkv=4,d=96，每个 KV 头服务两个 Q 头。
# 源码在 repeat_kv 前保存 cache，因此持久缓存仍只有 Hkv 个头。
# 持久 KV 字节=2*L*B*S*Hkv*d*bytes；固定其他项，默认相对 MHA 减半。
# expand 是视图，但 expand 后 reshape 可能物化，不能宣称整条操作零复制。
# 替代：支持 GQA 的原生 kernel，避免显式 repeat；输出仍有 Hq 个查询头。
# vLLM 映射：分别传递 query heads 与 KV heads，核对张量并行分片规则。
''', [(M, 'repeat_kv'), (M, 'Attention.__init__')], '''
from _shared import *
x = torch.arange(2.).reshape(1, 1, 2, 1)
y = repeat_kv(x, 2)
print(y.flatten())  # [0,0,1,1]；相邻两组查询共享一个 KV 头。
torch.testing.assert_close(y.flatten(), torch.tensor([0.,0.,1.,1.]))
cfg = config()
layer = Attention(cfg).eval()
with torch.no_grad():
    out, cache = layer(torch.randn(1, 3, 32), positions(cfg, 0, 3), use_cache=True)
print('output:', out.shape, 'persistent K:', cache[0].shape)
assert cache[0].shape == (1, 3, 2, 8)
for hkv in [8, 4, 1]:
    print(hkv, 'KV MiB:', 2*8*4096*hkv*96*2/2**20)
''', [('默认 cache 节省比例？', '固定其他维度，Hkv 从 8 到 4，持久 KV 元素数减半。'), ('GQA 总 FLOPs 也减半吗？', '不能如此推断；Q 头数不变，Attention 的查询头计算仍保留。'), ('repeat 后为什么还省 cache？', '持久 cache 在重复前建立；临时重复开销与持久存储是两回事。')])

add('05_RoPE', '''
# 输入 Q:[B,T,Hq,d]，K:[B,T,Hkv,d]，cos/sin:[T,d]；输出同形状。
# 公式 R_m q 与 R_n k 内积=q^T R_(n-m) k；未额外缩放时旋转保持范数。
# inv_freq_i=theta^(-2i/d)；不同维度对有不同角频率，d 必须为偶数。
# 当前 rotate_half 按前后半维配对，不是交错偶奇配对。
# decode 位置从历史 KV 长度 start_pos 开始；新 Q/K 不能从位置 0 重新旋转。
# YaRN 分支改变频率并可能改变幅度；不能把普通 RoPE 的所有结论无条件搬过去。
# 替代：绝对位置 embedding、ALiBi；不能直接换层后保证权重等价。
# 内存：预计算 cos/sin buffer 为 O(max_position*d)，不是每层历史 KV。
# vLLM 映射：rotary embedding；核对 pairing 布局、theta、position 与 scaling。
''', [(M, 'precompute_freqs_cis'), (M, 'apply_rotary_pos_emb')], '''
from _shared import *
cfg = config()
q = torch.randn(1, 4, 4, 8)
k = torch.randn(1, 4, 2, 8)
qr, kr = apply_rotary_pos_emb(q, k, *positions(cfg, 0, 4))
torch.testing.assert_close(q.norm(dim=-1), qr.norm(dim=-1))

# 保持内容向量相同，只平移两个绝对位置，验证相对位移。
a, b = torch.randn(1,1,1,8), torch.randn(1,1,1,8)
def dot_at(m, n):
    qa, _ = apply_rotary_pos_emb(a, a, *positions(cfg,m,1))
    _, kb = apply_rotary_pos_emb(b, b, *positions(cfg,n,1))
    return (qa*kb).sum()
torch.testing.assert_close(dot_at(2,5), dot_at(9,12))
print('norm and relative-position identities defined for verification')
''', [('为什么通常不旋转 V？', 'RoPE 通过 Q/K 内积调制位置关系；本实现把 V 保留为聚合内容。'), ('前后半配对能直接改成交错配对吗？', '要同步处理权重/布局；不能只改 rotate_half 就期待同一预训练模型等价。'), ('cache offset 错了有什么影响？', '新 Q/K 与历史 K 的相对角度错误，造成注意力与无缓存结果不一致。')])

add('06_KVCache', '''
# 输入：本轮 IDs:[B,T] 与每层历史 K/V:[B,S_old,Hkv,d]。
# 输出：logits:[B,T,V] 与追加 KV:[B,S_old+T,Hkv,d]。
# prefill 通常处理整段 prompt；普通 decode 每轮只输入新 token。
# cache 复用历史投影与层状态，但每个新 Q 仍读取历史 KV，单步不是 O(1)。
# 当前用 torch.cat 追加，反复分配/复制，不是生产级分页 KV。
# 默认 L=8,Hkv=4,d=96，B=1,S=4096,FP16，持久 cache=48 MiB。
# 替代：预分配连续 buffer、paged cache；分页主要解决分配/寻址/复用问题。
# vLLM 映射：KV blocks、block table、cache manager；不等同于 FlashAttention。
''', [(M, 'Attention.forward'), (M, 'MiniMindModel.forward'), (M, 'MiniMindForCausalLM.generate')], '''
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
''', [('为什么只缓存 K/V 而不累计历史 Q？', '新输出只需要新 Q 查询历史 K/V，历史 Q 对未来查询没有直接用途。'), ('什么时候比较 cache 等价？', 'eval、dropout 关闭、相同位置与mask、相同权重，比较 logits 并使用合理浮点容差。'), ('权重更新后旧 cache 可以复用吗？', '通常不能，旧 KV 来源于旧参数；跨请求复用还必须核对相同前缀和模型状态。')])

add('07_SwiGLU', '''
# 输入/输出:[B,T,D]；中间 gate/up:[B,T,I]。
# y=Wd(SiLU(Wg x) * Wu x)，SiLU(z)=z*sigmoid(z)。
# 默认 I=ceil(D*pi/64)*64，D=768 时 I=2432。
# 三个无 bias 投影参数=3DI；逐 token 变换，不混合序列 T 轴。
# 替代：GELU/ReLU 两层 FFN；比较时应控制参数量而非只控制 I。
# 性能：大矩阵乘、门控中间激活 O(BTI)；融合 gate/up/activation 可能降低访存。
# vLLM 映射：门控激活与并行线性层；不负责持久 KV cache。
''', [(M, 'FeedForward')], '''
from _shared import *
cfg = config()
ffn = FeedForward(cfg).eval()
x = torch.randn(2,3,32)
y = ffn(x)
manual = ffn.down_proj(F.silu(ffn.gate_proj(x))*ffn.up_proj(x))
torch.testing.assert_close(y, manual)
changed=x.clone(); changed[:,2]+=100
torch.testing.assert_close(ffn(changed)[:,:2], y[:,:2])
assert sum(p.numel() for p in ffn.parameters()) == 3*32*64
print(y.shape)
''', [('FFN 会混合 token 吗？', 'Dense FFN 独立处理各 token，跨 token 交互由 Attention 提供。'), ('为什么有三份矩阵？', '两份升维分别生成门与内容，逐元素相乘，再由第三份降维。'), ('扩大 I 的代价？', '参数、矩阵乘工作量和中间激活都随 I 增长。')])

add('08_Block', '''
# 输入隐藏态:[B,T,D]，位置与可选cache；输出同形隐藏态、新cache。
# h=x+Attention(Norm(x))；y=h+MLP(Norm(h))。
# 两个 norm 均位于分支之前，是 Pre-Norm；不能根据 post_attention_layernorm 名字判断架构。
# 残差提供直接信息/梯度通路，归一化控制分支尺度，但均不保证训练必然稳定。
# 替代：Post-Norm、不同 residual scaling；涉及训练和权重行为变化。
# vLLM 映射：residual add+RMSNorm 融合，注意原地修改和残差返回语义。
''', [(M, 'MiniMindBlock')], '''
from _shared import *
cfg = config()
block = MiniMindBlock(0,cfg).eval()
x = torch.randn(1,3,32)
pos = positions(cfg,0,3)
with torch.no_grad():
    result,_ = block(x.clone(),pos)
    attn,_ = block.self_attn(block.input_layernorm(x),pos)
    h = x+attn
    manual = h+block.mlp(block.post_attention_layernorm(h))
torch.testing.assert_close(result,manual)
print(result.shape)
''', [('Pre-Norm 的判据？', '看 norm 在每个子层分支之前，而不是看变量名。'), ('输出尺寸为什么不变？', 'Attention 和 MLP 均投影回 D，才能进行逐元素残差相加。'), ('融合时应核对什么？', '输出数值、residual 保存/更新位置、别名和原地操作、dtype。')])

add('09_Forward', '''
# IDs:[B,T] -> embedding:[B,T,D] -> L Blocks -> final norm -> logits:[B,T,V]。
# 返回 MoeCausalLMOutputWithPast，含 logits/loss/aux_loss/past_key_values/hidden_states。
# logits_to_keep=0：slice(-0,None)=slice(0,None)，保留全部位置。
# logits_to_keep=1 只减少 LM head 位置数，不跳过前面的模型层计算。
# 计算全序列 labels loss 时不要随意裁剪 logits 造成错位或空张量。
# 替代：分离 prefill/decode 执行接口；vLLM 在模型执行器侧组织输入与输出位置。
# 性能：全量 logits O(BTV)，可能明显大于 O(BTD) 隐藏态；cache 另计。
''', [(M, 'MiniMindModel'), (M, 'MiniMindForCausalLM.forward')], '''
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
''', [('hidden_states 与 logits 区别？', '前者最后一维 D 表示隐藏特征；后者最后一维 V 表示词表候选分数。'), ('为什么 keep=0 保留全部？', 'Python 中 -0=0，切片从位置 0 到末尾。'), ('仅最后 logits 是否等于仅算最后 token？', '不是；输入中全部 token 仍经过模型层，仅输出投影裁剪。')])

add('10_Loss', '''
# 输入 logits:[B,T,V]，labels:[B,T]；输出有效 next-token 位置的平均交叉熵标量。
# L=-mean(log softmax(logits[b,t])[labels[b,t+1]])，忽略 label=-100。
# 当前 Dataset 不提前shift，forward 内只shift一次；首token无前驱，末logit无本序列目标。
# label mask 不等于 attention mask；忽略 user loss 仍可读取 user 上下文。
# 全部目标为 -100 时 mean loss 可能 NaN，调用侧需检查有效监督数量。
# 替代：label smoothing、token/sequence 不同归约；会改变优化目标。
# vLLM 常规生成不计算训练 CE；训练需要保留反传图和可能很大的 logits。
''', [(M, 'MiniMindForCausalLM.forward')], '''
from _shared import *
model=tiny_model().eval()
ids=torch.tensor([[1,7,9,2]])
labels=ids.clone(); labels[:,0]=-100
out=model(ids,labels=labels)
logp=out.logits[:,:-1].log_softmax(-1)
targets=labels[:,1:]
valid=targets.ne(-100)
safe=targets.masked_fill(~valid,0)  # gather 不能直接索引 -100。
nll=-logp.gather(-1,safe.unsqueeze(-1)).squeeze(-1)
manual=(nll*valid).sum()/valid.sum()
torch.testing.assert_close(out.loss,manual)
print(out.loss)
# input [A,B,C] 对应 logits[A]->B、logits[B]->C；不要在 Dataset 再 shift。
''', [('next-token shift 在哪里？', 'MiniMindForCausalLM.forward 内：logits去掉末位置，labels去掉首位置。'), ('-100 会禁止 Attention 读取吗？', '不会，它只是 CE 的 ignore_index。'), ('PPL 如何得到？', '相同评测分词与自然对数 NLL 定义下 exp(平均NLL)，不能跨不同tokenizer直接公平比较。')])

add('11_Pretrain', '''
# 数据 text -> 截断/BOS/EOS/pad -> IDs/labels:[B,T] -> CE+aux -> backward -> AdamW。
# 本地 PretrainDataset 对单条样本补齐，不是跨样本 packing。
# lr=lr0*(0.1+0.45*(1+cos(pi*step/total)))；具体见 get_lr。
# 梯度累积：loss/accumulation_steps；当前脚本尾窗不足时仍用固定除数，要核对权重。
# 混合精度：先unscale再clip；完整恢复需要模型、优化器、scaler与进度/RNG等状态。
# 参数半精度导出文件不等于完整恢复checkpoint；trainer_utils 有不同保存用途。
# 替代：packing、动态padding、token数加权累积、DDP；按任务与数据实现选择。
# 内存：参数+梯度+Adam状态+反传激活，不能拿推理参数显存估算训练峰值。
# vLLM 接收训练导出权重做推理，不代替本训练循环。
''', [(D, 'PretrainDataset'), ('trainer/train_pretrain.py', None), ('trainer/trainer_utils.py', 'get_lr'), ('trainer/trainer_utils.py', 'lm_checkpoint')], '''
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
''', [('预训练标签从哪里来？', '直接来自原始token序列，由forward进行next-token对齐；不是人工类别标签。'), ('累积总等于大batch吗？', '还受有效token计数、归约、随机性、尾窗等影响，不能无条件等价。'), ('恢复只加载模型够吗？', '推理通常够；精确续训还需优化器/scaler/进度/RNG及数据顺序等状态。')])

add('12_SFT', '''
# conversations -> chat template -> IDs:[B,T]，labels:[B,T]，非回答目标设 -100。
# 与 Pretrain 复用同一 CE；差异在样本、模板、监督位置、起始权重和超参。
# 当前 SFTDataset 通过 assistant 起止 token 片段扫描生成 labels，细节见真实代码。
# assistant 结束标记可受监督；不能笼统地把所有特殊token删掉。
# user label=-100 不切断上下文通路；回答损失仍能回传到前面输入的表示。
# 替代：全序列监督、多轮回答监督、LoRA；目标和可训练参数集合会改变。
# 全参 SFT 不减少部署模型参数或 KV cache；与 vLLM 对接仍需要正确 chat template。
''', [(D, 'pre_processing_chat'), (D, 'post_processing_chat'), (D, 'SFTDataset'), ('trainer/train_full_sft.py', None)], '''
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
''', [('SFT换了模型结构吗？', '本仓库全参SFT复用相同模型结构与CE，主要改变数据和有效监督位置。'), ('为何保留user输入但mask其labels？', 'user是生成回答的条件，但该设置不要求模型直接学习预测user文本。'), ('截断导致没有assistant目标怎么办？', '检测shift后的有效label数，丢弃/重新构造样本，避免无监督mean loss。')])

add('13_Serving', '''
# 输入 messages/生成参数 -> tokenizer -> IDs -> generate -> token流/HTTP响应。
# 此课是模型到服务边界；完整 HTTP 源码附在下方，仅阅读，不启动服务。
# TTFT=请求至首token耗时；TPOT=输出token间耗时，常需单独计入排队与prefill。
# 当前单请求generate不是continuous batching调度器，HTTP封装不自动提供高并发。
# 替代：vLLM等执行引擎；先核对导出architecture、tokenizer、dtype和具体版本支持。
# 当前convert_model.py提供Qwen3/Qwen3-MoE映射；不要把原生自定义类名直接当作兼容证明。
# 并发显存包含权重、每请求KV、临时激活；取消和结束时须回收请求资源。
# 示例只展示本地greedy生成；代码不发网络请求、不启动服务。
''', [(M, 'MiniMindForCausalLM.generate'), ('scripts/serve_openai_api.py', None), ('scripts/convert_model.py', None)], '''
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
''', [('API服务等于高吞吐引擎吗？', '不等于，还需要调度、批处理、cache管理、取消与资源回收。'), ('首token慢一定是decode慢吗？', '不一定，TTFT包含排队、预处理与prefill，须分阶段观察。'), ('如何迁移vLLM？', '固定版本并核对导出架构/权重/模板，验证logits或生成一致性后再测吞吐与显存。')])

add('14_DPO', '''
# 输入：同prompt的chosen/rejected响应、policy/reference逐token logp与mask。
# 输出：标量loss；reference冻结，policy可训练，无在线环境采样或Critic必需项。
# L=-log sigmoid(beta*((logπ_ch-logπ_rej)-(logref_ch-logref_rej)))。
# 仓库先将响应token logp求和，不是除以长度；batch前半chosen、后半rejected。
# DPODataset已做x/y shift，不能再套LM内置labels shift导致错位。
# -100不能直接用于gather；这里labels用有效token ID，mask单独控制监督位置。
# 替代：其他偏好损失；不同长度归约改变目标，不能悄悄替换。
# 显存：policy训练状态+冻结reference前向；DPO不是PPO式在线RL。
# vLLM可用于生成偏好候选，但不是DPO loss的必需计算环节。
''', [(D, 'DPODataset'), ('trainer/train_dpo.py', None)], '''
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
''', [('reference为什么冻结？', '提供固定偏好相对基线，避免基线跟随policy一起漂移。'), ('为什么chosen/rejected顺序重要？', 'loss按前后半batch对应，顺序或配对错误会优化错误偏好。'), ('DPO需要奖励模型和Critic吗？', '该直接偏好目标不要求显式奖励模型或Critic；需成对偏好与reference基线。')])

add('15_PPO', '''
# actor输入序列->每token logπ；Critic->每状态value；外部reward常在回答末尾赋值。
# ratio=exp(logπ_new-logπ_old)；最大化min(ratio*A,clip(ratio)*A)。
# δ_t=r_t+γV_(t+1)-V_t；GAE_t=δ_t+γλ*GAE_(t+1)，终止处bootstrap为0。
# current仓库另含参考KL惩罚、clipped value loss、adv归一化、DDP同步早停。
# old是采样策略快照；ref是固定基线；不能混为一个模型概念。
# 下方示例是目标函数探针；完整EOS/mask/minibatch实现以真实训练文件为准。
# 替代：GRPO不训练Critic，以同prompt样本组构造相对优势。
# 显存：actor/critic/reference/reward与rollout；采样时no_grad不代表更新时无需反传。
# Infra：采样与更新权重同步，记录实际采样分布，避免old logp口径不一致。
''', [('trainer/train_ppo.py', None)], '''
from _shared import *
rewards=torch.tensor([[0.,0.,1.]])
values=torch.tensor([[0.1,0.2,0.3]])
gamma,lam=0.99,0.95
adv=torch.zeros_like(values); carry=torch.zeros(1)
for t in reversed(range(3)):
    nxt=values[:,t+1] if t<2 else torch.zeros(1)
    delta=rewards[:,t]+gamma*nxt-values[:,t]
    carry=delta+gamma*lam*carry
    adv[:,t]=carry
returns=adv+values
old=torch.tensor([[-1.,-1.,-1.]])
new=torch.tensor([[-0.8,-1.1,-0.9]],requires_grad=True)
ratio=(new-old).exp()
policy_loss=-torch.minimum(ratio*adv,ratio.clamp(0.8,1.2)*adv).mean()
vnew=values.clone().requires_grad_()
vclip=vnew.clamp(values-0.2,values+0.2)
value_loss=0.5*torch.maximum((vnew-returns).square(),(vclip-returns).square()).mean()
(policy_loss+0.5*value_loss).backward()
print(adv,policy_loss.detach(),value_loss.detach())
''', [('old和ref有什么区别？', 'old对应采样行为策略，ref提供固定KL基线；两者用途不同。'), ('clip保证参数完全不变吗？', '不保证；它限制某些方向的代理目标收益，不是参数空间硬约束。'), ('为何需要Critic？', '估计状态价值，构造优势和降低策略梯度方差，同时需额外训练与显存。')])

add('16_GRPO_CISPO', '''
# rewards:[B*G] -> grouped:[B,G] -> A=(reward-组mean)/(组std+1e-4)。
# 当前std为unbiased=False；同组奖励相等时优势为0，不提供组内区分信号。
# logp/old/ref/mask:[B*G,R]；序列级优势广播到回答token。
# GRPO使用PPO式clipped surrogate；KL探针 exp(ref-new)-(ref-new)-1。
# CISPO当前源码：ratio仅上限裁剪后detach，再乘A*new_logp；不是相同clip公式。
# 当前脚本默认loss_type=cispo；每条响应先按有效token平均，再按样本平均。
# 无Critic不等于无reference/reward；G条响应增加rollout和KV成本。
# vLLM迁移：批量多样本rollout、采样策略口径、policy version同步。
''', [('trainer/train_grpo.py', None)], '''
from _shared import *
rewards=torch.tensor([[1.,2.,3.],[4.,4.,4.]])
adv=((rewards-rewards.mean(1,keepdim=True))/(rewards.std(1,keepdim=True,unbiased=False)+1e-4)).flatten()
old=torch.full((6,3),-1.)
ref=torch.full((6,3),-1.1)
mask=torch.tensor([[1,1,0]]*6,dtype=torch.float32)
for mode in ['grpo','cispo']:
    new=torch.full((6,3),-0.9,requires_grad=True)
    ratio=(new-old).exp(); diff=ref-new
    kl=diff.exp()-diff-1
    if mode=='cispo':
        objective=ratio.clamp(max=1.2).detach()*adv[:,None]*new
    else:
        objective=torch.minimum(ratio*adv[:,None],ratio.clamp(.8,1.2)*adv[:,None])
    loss=(((-objective+0.04*kl)*mask).sum(1)/mask.sum(1).clamp(min=1)).mean()
    loss.backward()
    print(mode,loss.detach(),new.grad)
''', [('全组奖励相同会怎样？', '组内优势为零，策略优势项无区分信号；参考KL仍可能贡献梯度。'), ('为什么不能跨不同prompt任意标准化？', '组相对基线要求同一条件下比较；跨prompt难度和奖励尺度可能不同。'), ('CISPO的detach放哪里？', '当前实现对裁剪后的importance ratio detach，再乘可导new logp；不能照搬PPO clip。')])

add('17_RolloutEngine', '''
# 输入prompt_ids/attention_mask:[B,P]、G与max_new_tokens。
# 输出output_ids:[B*G,P+R]、completion_ids/logps/mask:[B*G,R]、prompt_lens:[B*G]。
# 策略更新ratio使用每个回答token的old logp，不能用生成文本长度替代token数量。
# compute_per_token_logps保留n_keep+1个logits，再去末位置，对齐最后n_keep个token。
# 当前实际后端是torch/sglang，没有已经实现的vLLM后端；下方完整工厂代码为准。
# Torch后端本地重算logp；SGLang后端返回logp，须核对温度/截断分布与训练口径。
# old logp、EOS/pad mask与policy版本都必须对应真实采样轨迹。
# 替代/迁移：新增vLLM适配器实现相同结果合同，不宣称仅换URL就兼容。
# 性能：生成常是RL主要成本；异步采样提高吞吐但引入策略滞后和版本管理。
''', [('trainer/rollout_engine.py', None)], '''
from _shared import *
model=tiny_model().eval()
ids=torch.tensor([[1,7,9,2,4]])
n_keep=2
with torch.no_grad():
    logits=model(ids,logits_to_keep=n_keep+1).logits[:,:-1]
    logps=logits.log_softmax(-1).gather(-1,ids[:,-n_keep:,None]).squeeze(-1)
assert logps.shape==(1,2)
print('last response token logps:',logps)
# 如需真正采样，参照真实RolloutEngine；本探针没有调用远端或模拟返回伪造old logp。
''', [('为何保留R+1个logits？', '预测首个回答token需要其前一个位置的logits，去掉末位置后恰好R个。'), ('采样后还能重算old logp吗？', '应使用真实行为策略版本与一致分布；若参数已更新或温度处理不同则不等价。'), ('当前有vLLM实现吗？', '此版本工厂支持torch和sglang；vLLM是后续适配方向。')])

add('18_AgentRL', '''
# messages/tools -> 多轮生成 -> 工具调用/观察 -> 完整轨迹与response_mask。
# assistant生成token是策略动作；工具返回内容是环境观察，通常不纳入策略梯度mask。
# 但工具观察保留在上下文，影响后续assistant动作的logp。
# old logp和mask必须与拼接后的同一token位置对齐，不能按字符串字符数索引。
# 当前仓库提供mock工具和reward；不代表执行了真实天气/时间/汇率查询。
# 算法复用GRPO/CISPO；不能把多轮轨迹简单当成独立单轮响应而丢掉条件。
# 替代：单轮RL、真实环境交互；限制工具回合数/终止条件，评估奖励作弊。
# Infra：多轮上下文增长使prefill/KV成本上升；轨迹与policy版本需要关联。
''', [('trainer/train_agent.py', None), (D, 'AgentRLDataset')], '''
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
''', [('工具输出为何mask为0？', '它是环境观察，不是策略采样的动作；不能把其token概率当作actor动作奖励。'), ('mask为0是否删除工具上下文？', '不是，观察仍是后续生成条件，只是不直接计入策略目标。'), ('轨迹最易错哪里？', '字符串与token错位、工具段错误监督、跨轮old logp错配、EOS和终止规则不一致。')])

add('19_MoE', '''
# x:[B,T,D] -> flatten:[N,D] -> router:[N,E] -> topk:[N,K] -> expert加权散射 -> [B,T,D]。
# y_n=sum_k p_(n,k)*Expert_(idx(n,k))(x_n)，当前没有shared expert。
# top1+norm_topk_prob时权重前向为1，使用straight-through项给router传梯度。
# aux_loss根据专家负载与平均路由概率构造，防止长期集中少数专家；不保证完美均衡。
# active参数少不代表全部权重无需驻留；专家不活跃时仍有总参数存储。
# 替代：Dense、不同top-k/shared experts；多卡expert parallel需考虑all-to-all。
# vLLM映射：fused MoE/expert dispatch；本地Python专家for循环不是高效部署内核。
''', [(M, 'MOEFeedForward')], '''
from _shared import *
cfg=config(use_moe=True,num_experts=4,num_experts_per_tok=1)
moe=MOEFeedForward(cfg).train()
x=torch.randn(2,3,32)
y=moe(x)
loss=y.square().mean()+moe.aux_loss
loss.backward()
print(y.shape,moe.aux_loss.detach())
print('router grad norm:',moe.gate.weight.grad.norm())
''', [('激活参数少是否等于显存同样减少？', '不是，通常仍需存储全部专家权重，另有路由/通信开销。'), ('top1权重归一化为1后router怎么学？', '当前用straight-through表达式与辅助loss传递梯度，直接常量1会丢失该路径。'), ('MoE推理瓶颈？', '专家负载不均、dispatch/gather、通信、碎片化小矩阵乘，不能只看FLOPs。')])

add('20_LoRA', '''
# 线性层 W:[out,in]，A:[r,in]，B:[out,r]，y=xW^T+xA^TB^T。
# 当前实现没有显式alpha/r缩放；增量参数r*(in+out)，通常冻结基础参数再训练adapter。
# A随机、B零初始化使初始增量为0，但首步A梯度可能为0，B可先更新。
# apply_lora仅选择in_features==out_features的Linear；不能宣称自动覆盖全部Q/K/V。
# 当前GQA里K/V投影常不是方阵；实际注入层名必须检查。
# merge为W+BA；卸载adapter路径后才能当作普通线性推理，避免重复加两遍。
# vLLM映射：LoRA加载/多adapter管理；训练显存降低不等于KV cache减少。
''', [('model/model_lora.py', None), ('trainer/train_lora.py', None)], '''
from _shared import *
from model.model_lora import LoRA
base=torch.nn.Linear(8,8,bias=False)
adapter=LoRA(8,8,rank=2)
base.requires_grad_(False)
x=torch.randn(3,8)
torch.testing.assert_close(base(x)+adapter(x),base(x))
optimizer=torch.optim.AdamW(adapter.parameters(),lr=1e-3)
loss=(base(x)+adapter(x)).square().mean()
loss.backward(); optimizer.step()
merged=base.weight.detach()+adapter.B.weight.detach()@adapter.A.weight.detach()
torch.testing.assert_close(base(x)+adapter(x),F.linear(x,merged),atol=1e-5,rtol=1e-4)
print('adapter parameters:',sum(p.numel() for p in adapter.parameters()))
''', [('为何不能A/B都零初始化？', '双零下两者乘积路径的首步梯度都为零，无法从该分支开始学习。'), ('当前LoRA覆盖K/V吗？', '注入筛选方阵，默认GQA K/V非方阵，需逐层核对，不能泛称全部覆盖。'), ('合并后需要额外矩阵乘吗？', '正确折叠并移除adapter路径后可使用普通Linear；只修改权重但保留外挂会重复叠加。')])

add('21_Distillation', '''
# teacher/student logits:[N,V]，有效监督位置筛选后KL(teacher||student)。
# L_KD=T²*KL(softmax(z_teacher/T)||softmax(z_student/T))。
# F.kl_div输入是student log-prob与teacher prob；顺序弄反会改变KL方向。
# 当前总loss=alpha*CE+(1-alpha)*KD，alpha含义不能与其他实现反过来套。
# teacher eval+no_grad；student训练；teacher/student tokenizer ID语义必须对齐。
# 简单裁剪teacher词表到student大小不自动保证ID语义相同，必须验证映射。
# 替代：离线生成数据做SFT、不同层特征蒸馏；成本和目标不同。
# vLLM可服务teacher生成离线数据，但本地logit蒸馏需要teacher分布而非仅文本。
''', [('trainer/train_distillation.py', None)], '''
from _shared import *
temperature=2.0
student=torch.randn(4,8,requires_grad=True)
teacher=torch.randn(4,8)
mask=torch.tensor([True,True,False,True])
student_logp=F.log_softmax(student[mask]/temperature,-1)
teacher_p=F.softmax(teacher[mask]/temperature,-1).detach()
kd=F.kl_div(student_logp,teacher_p,reduction='batchmean')*temperature**2
ce=F.cross_entropy(student[mask],torch.tensor([1,2,3]))
loss=.5*ce+.5*kd
loss.backward()
print('CE/KD:',ce.detach(),kd.detach())
''', [('为什么乘T²？', '补偿温度改变softmax/log-softmax梯度尺度；实际作用还受分布和目标权重影响。'), ('只有teacher文本能算同样的KD吗？', '不能，logit KD需要分布信息；文本可用于另一种离线监督蒸馏。'), ('词表长度相同就对齐吗？', '不是，必须核对同一ID对应相同token与分词规则。')])

add('22_Tokenizer', '''
# 输入文本 -> ByteLevel预切分+BPE -> token IDs；decode把ID映射回文本。
# BPE训练迭代合并高频相邻符号；实际byte映射、特殊token和模板以仓库文件为准。
# 当前上游不建议随意重训tokenizer；先读已有tokenizer，不覆盖model/tokenizer文件。
# 词表变化会改变embedding/LM head的行语义，即使张量形状一致也不代表兼容。
# 更大词表增加V*D参数与输出打分成本；更短序列可能降低Attention/KV成本，存在取舍。
# 替代：其他子词或字符方案；跨tokenizer比较PPL不直接公平。
# vLLM映射：请求文本预处理与chat template，模型与服务必须使用匹配词表。
''', [('trainer/train_tokenizer.py', None)], '''
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
''', [('一个token等于一个字吗？', '不一定，取决于byte/subword切分与语料。'), ('为什么不能随意换tokenizer？', 'ID到词表行的语义改变，embedding/输出权重和模板的匹配被破坏。'), ('增大词表总能提速吗？', '序列可能缩短，但参数与LM head成本增加，要综合权衡并实测。')])


def source_text(path, symbol):
    source = (ROOT / path).read_text(encoding='utf-8')
    if symbol is None:
        return source.rstrip(), 1
    tree = ast.parse(source)
    node = tree
    for part in symbol.split('.'):
        node = next(n for n in node.body if isinstance(n,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==part)
    start = min([node.lineno]+[d.lineno for d in getattr(node,'decorator_list',[])])
    return '\n'.join(source.splitlines()[start-1:node.end_lineno]), start

index = ['# 剩余课次：代码版总入口', '', '```python',
         '# 本批次只编写文件；未运行任何模型、实验、训练、服务或示例。',
         '# 01/02保留此前的讲义；03～22采用代码与注释，含真实源码、独立教学示例、3题与答案。',
         '# 模型源码基线：233d00fa068ec2b8c0d4f07b703934d6391a5a73。',
         '# 原文件代码块是阅读摘录；需要原文件的imports/全局变量，不是独立脚本。',
         '# code_examples中的脚本才是独立入口；依赖与实际运行结果尚未验证。',
         '# 含backward/optimizer/生成调用的代码仅写入文件，未执行。',
         '# 独立示例基于合成输入，不能替代完整训练/评测。',
         '# vLLM均为概念映射，本版本rollout实际支持torch/sglang。',
         '# 学习顺序按编号，不要求一次读完全部文件。', '```', '']
for name, notes, sources, example, qa in LESSONS:
    script = '# 本次仅编写，未运行。\n' + textwrap.dedent(example).strip() + '\n'
    (CODE / (name+'.py')).write_text(script,encoding='utf-8')
    sections = [f'# {name}', '', '```python', '# 本次仅编写，未运行；以下公式/形状/性能说明为教学注释。',
                textwrap.dedent(notes).strip(), '```', '']
    for path, symbol in sources:
        source,line = source_text(path,symbol)
        sections += ['```python', f'# 真实源码：{path}:{line}'+(f'，{symbol}' if symbol else '，完整文件'),
                     '# 以下保持当前源码原样；与上方注释及下方教学例区分。', source, '```', '']
    sections += ['```powershell', '# 仅记录将来手动执行命令，本次没有执行。',
                 f'.\\.venv\\Scripts\\python.exe learning\\code_examples\\{name}.py', '```', '',
                 '```python', '# 独立教学示例；辅助模块见code_examples/_shared.py。', script.rstrip(), '```', '',
                 '```python']
    for i,(q,a) in enumerate(qa,1): sections.append(f'# Q{i}: {q}')
    sections += ['```', '', '<details>', '<summary>参考答案：先独立回答再展开</summary>', '', '```python']
    for i,(q,a) in enumerate(qa,1): sections.append(f'# A{i}: {a}')
    sections += ['```','','</details>', '']
    (LEARN / (name+'.md')).write_text('\n'.join(sections),encoding='utf-8')
    index.append(f'- [{name}]({name}.md) · [示例](code_examples/{name}.py)')
index += ['', '[辅助代码](code_examples/_shared.py)', '',
          '```python', '# 依赖：01/02的CPU环境覆盖多数数学示例。',
          '# 阅读完整训练源码还涉及datasets、requests、swanlab等requirements.txt依赖。',
          '# 本次不安装依赖、不下载模型/数据、不启动任何服务。', '```','']
(LEARN/'CODE_INDEX.md').write_text('\n'.join(index),encoding='utf-8')
print(f'Wrote {len(LESSONS)} Markdown lessons and {len(LESSONS)} example scripts; executed none.')
