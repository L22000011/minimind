"""Run from any directory with the learning environment; imports real repo code."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from model.model_minimind import RMSNorm


def main():
    torch.manual_seed(42)
    eps = 1e-6
    norm = RMSNorm(2, eps=eps)
    sample = torch.tensor([[[3.0, 4.0]]])
    y = norm(sample)
    expected = sample / torch.sqrt(sample.square().mean(-1, keepdim=True) + eps)
    torch.testing.assert_close(y, expected)
    print('manual output:', y.detach().flatten().tolist())
    print('output mean (not zero):', y.mean().item())
    assert abs(y.mean().item()) > 0.5
    torch.testing.assert_close(norm(sample * 10), y, atol=1e-6, rtol=1e-6)

    # Non-uniform gamma checks actual per-feature affine broadcasting.
    norm4 = RMSNorm(4, eps=eps)
    with torch.no_grad():
        norm4.weight.copy_(torch.tensor([1.0, 2.0, 0.5, -1.0]))
    x = torch.randn(2, 3, 4, requires_grad=True)
    out = norm4(x)
    reference = x / torch.sqrt(x.square().mean(-1, keepdim=True) + eps)
    reference = reference * norm4.weight
    torch.testing.assert_close(out, reference)
    assert out.shape == x.shape
    altered = x.detach().clone()
    altered[:, 1:, :] *= 100
    torch.testing.assert_close(norm4(altered)[:, 0], out.detach()[:, 0])
    print('3D shape:', tuple(out.shape))

    q = torch.randn(2, 3, 2, 4)
    q_out = norm4(q)
    q_ref = q / torch.sqrt(q.square().mean(-1, keepdim=True) + eps) * norm4.weight
    torch.testing.assert_close(q_out, q_ref)
    assert norm4.weight.numel() == 4
    print('4D shape:', tuple(q_out.shape))

    half_x = torch.tensor([[[300.0, 400.0]]], dtype=torch.float16)
    raw_has_inf = torch.isinf(half_x.square()).any().item()
    safe_y = norm(half_x)
    assert raw_has_inf
    assert safe_y.dtype == half_x.dtype and torch.isfinite(safe_y).all()
    print('raw fp16 square has inf:', raw_has_inf)
    print('real RMSNorm fp16 output:', safe_y.detach().flatten().tolist())
    zero_y = norm(torch.zeros(1, 1, 2))
    assert torch.isfinite(zero_y).all() and torch.count_nonzero(zero_y) == 0

    loss = out.square().sum()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert norm4.weight.grad is not None and torch.isfinite(norm4.weight.grad).all()
    assert x.grad.abs().sum() > 0 and norm4.weight.grad.abs().sum() > 0
    print('input and gamma gradients: finite and nonzero')
    print('ALL CHECKS PASSED')


if __name__ == '__main__':
    main()
