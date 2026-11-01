"""Token embedding: real MiniMind modules, deterministic toy weights, CPU only."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch.nn import functional as F
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM


def main():
    torch.manual_seed(42)
    config = MiniMindConfig(
        vocab_size=6, hidden_size=4, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1,
        intermediate_size=8, max_position_embeddings=16,
        dropout=0.0, tie_word_embeddings=True,
    )
    model = MiniMindForCausalLM(config).eval()
    embedding = model.model.embed_tokens
    # Each token gets a visibly different row; these are teaching weights.
    table = torch.arange(24, dtype=torch.float32).reshape(6, 4) / 10
    with torch.no_grad():
        embedding.weight.copy_(table)

    ids = torch.tensor([[1, 3, 1], [0, 2, 5]], dtype=torch.long)
    vectors = embedding(ids)  # [B,T] -> [B,T,D], no attention involved.
    indexed = embedding.weight[ids]
    one_hot = F.one_hot(ids, num_classes=6).to(table.dtype)
    torch.testing.assert_close(vectors, indexed)
    torch.testing.assert_close(vectors, one_hot @ embedding.weight)
    torch.testing.assert_close(vectors[0, 0], vectors[0, 2])
    assert vectors.shape == (2, 3, 4)
    assert embedding.padding_idx is None  # Current repo does not freeze a pad row.
    print('IDs shape:', tuple(ids.shape))
    print('embedding shape:', tuple(vectors.shape))
    print('token 1 vector:', vectors[0, 0].detach().tolist())
    print('lookup == indexing == one_hot @ weight: PASS')

    # Embedding-only path: repeated IDs accumulate row gradients.
    model.zero_grad(set_to_none=True)
    embedding(torch.tensor([[1, 3, 1]])).sum().backward()
    grad = embedding.weight.grad
    torch.testing.assert_close(grad[1], torch.full((4,), 2.0))
    torch.testing.assert_close(grad[3], torch.ones(4))
    torch.testing.assert_close(grad[[0, 2, 4, 5]], torch.zeros(4, 4))
    print('embedding-only gradient rows:', grad.abs().sum(-1).nonzero().flatten().tolist())

    # Same Parameter object: input table [V,D] is also output classifier weight.
    assert embedding.weight is model.lm_head.weight
    hidden = torch.ones(1, 1, 4)  # A probe hidden state; not a complete LM forward.
    logits = model.lm_head(hidden)
    torch.testing.assert_close(logits, hidden @ embedding.weight.T)
    print('shared Parameter:', embedding.weight is model.lm_head.weight)
    print('LM head probe shape:', tuple(logits.shape))

    # Output softmax considers every class, so shared rows can receive gradients
    # even when their token IDs were never present at the embedding input.
    model.zero_grad(set_to_none=True)
    F.cross_entropy(model.lm_head(hidden).reshape(1, 6), torch.tensor([0])).backward()
    active_rows = embedding.weight.grad.abs().sum(-1).nonzero().flatten().tolist()
    assert active_rows == list(range(6))
    print('LM-head CE gradient rows:', active_rows)
    print('ALL CHECKS PASSED')


if __name__ == '__main__':
    main()
