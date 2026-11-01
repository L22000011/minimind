# 剩余课次：代码版总入口

```python
# 本批次只编写文件；未运行任何模型、实验、训练、服务或示例。
# 01/02保留此前的讲义；03～22采用代码与注释，含真实源码、独立教学示例、3题与答案。
# 模型源码基线：233d00fa068ec2b8c0d4f07b703934d6391a5a73。
# 原文件代码块是阅读摘录；需要原文件的imports/全局变量，不是独立脚本。
# code_examples中的脚本才是独立入口；依赖与实际运行结果尚未验证。
# 含backward/optimizer/生成调用的代码仅写入文件，未执行。
# 独立示例基于合成输入，不能替代完整训练/评测。
# vLLM均为概念映射，本版本rollout实际支持torch/sglang。
# 学习顺序按编号，不要求一次读完全部文件。
```

- [03_CausalAttention](03_CausalAttention.md) · [示例](code_examples/03_CausalAttention.py)
- [04_GQA](04_GQA.md) · [示例](code_examples/04_GQA.py)
- [05_RoPE](05_RoPE.md) · [示例](code_examples/05_RoPE.py)
- [06_KVCache](06_KVCache.md) · [示例](code_examples/06_KVCache.py)
- [07_SwiGLU](07_SwiGLU.md) · [示例](code_examples/07_SwiGLU.py)
- [08_Block](08_Block.md) · [示例](code_examples/08_Block.py)
- [09_Forward](09_Forward.md) · [示例](code_examples/09_Forward.py)
- [10_Loss](10_Loss.md) · [示例](code_examples/10_Loss.py)
- [11_Pretrain](11_Pretrain.md) · [示例](code_examples/11_Pretrain.py)
- [12_SFT](12_SFT.md) · [示例](code_examples/12_SFT.py)
- [13_Serving](13_Serving.md) · [示例](code_examples/13_Serving.py)
- [14_DPO](14_DPO.md) · [示例](code_examples/14_DPO.py)
- [15_PPO](15_PPO.md) · [示例](code_examples/15_PPO.py)
- [16_GRPO_CISPO](16_GRPO_CISPO.md) · [示例](code_examples/16_GRPO_CISPO.py)
- [17_RolloutEngine](17_RolloutEngine.md) · [示例](code_examples/17_RolloutEngine.py)
- [18_AgentRL](18_AgentRL.md) · [示例](code_examples/18_AgentRL.py)
- [19_MoE](19_MoE.md) · [示例](code_examples/19_MoE.py)
- [20_LoRA](20_LoRA.md) · [示例](code_examples/20_LoRA.py)
- [21_Distillation](21_Distillation.md) · [示例](code_examples/21_Distillation.py)
- [22_Tokenizer](22_Tokenizer.md) · [示例](code_examples/22_Tokenizer.py)

[辅助代码](code_examples/_shared.py)

```python
# 依赖：01/02的CPU环境覆盖多数数学示例。
# 阅读完整训练源码还涉及datasets、requests、swanlab等requirements.txt依赖。
# 本次不安装依赖、不下载模型/数据、不启动任何服务。
```
