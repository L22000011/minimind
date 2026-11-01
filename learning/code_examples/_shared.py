# 仅供你之后手动执行示例；本次没有运行。
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
