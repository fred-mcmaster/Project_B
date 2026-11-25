"""Student-facing behavioral checks; no reference implementation imported."""
import torch
from model import CompressionModel
from codec import compress, decompress


def check_soft_quantize(fn):
    x = torch.tensor([[0.25, 0.0], [1.5, 0.0]], dtype=torch.float64, requires_grad=True)
    codebook = torch.tensor([[0., 0.], [2., 0.], [0., 3.]], dtype=torch.float64, requires_grad=True)
    q, p = fn(x, codebook, 0.7)
    assert q.shape == x.shape and p.shape == (2, 3)
    assert torch.isfinite(q).all() and torch.isfinite(p).all() and (p >= 0).all()
    torch.testing.assert_close(p.sum(-1), torch.ones(2, dtype=p.dtype))
    # Relative probability fixes the distance scale/sign/temperature convention.
    expected_ratio = torch.tensor([3./0.7, -2./0.7], dtype=p.dtype)
    torch.testing.assert_close(p[:, 0].log() - p[:, 1].log(), expected_ratio)
    torch.testing.assert_close(q, p @ codebook)
    cold, _ = fn(x, codebook, 0.001)
    torch.testing.assert_close(cold, codebook[torch.tensor([0, 1])])
    _, hot = fn(x, codebook, 1e7)
    torch.testing.assert_close(hot, torch.full_like(hot, 1/3), atol=1e-6, rtol=1e-6)
    assert torch.autograd.gradcheck(lambda a, b: fn(a, b, 0.7)[0], (x, codebook))
    q.square().sum().backward()
    for t in (x, codebook):
        assert t.grad is not None and torch.isfinite(t.grad).all() and t.grad.abs().sum() > 0
    # An implementation that detaches everything must fail before training.
    print('Soft quantization: shape, values, temperature, and gradients passed.')


def check_pipeline(fn):
    torch.manual_seed(7)
    model = CompressionModel(fn, width=8, latent_dim=4, levels=2, codebook_size=7)
    x = torch.rand(2, 3, 19, 23)
    result = model(x)
    assert result['reconstruction'].shape == x.shape
    loss = (result['reconstruction']-x).square().mean() + .01*result['estimated_bpp']
    loss.backward()
    for name, parameters in [('encoder', model.encoder.parameters()),
                             ('decoder', model.decoder.parameters()),
                             ('codebook', [model.codebook]), ('prior', [model.prior_logits])]:
        grads = [p.grad for p in parameters]
        assert all(g is not None and torch.isfinite(g).all() for g in grads), name
        assert sum(g.abs().sum().item() for g in grads) > 0, name
    model.eval()
    with torch.no_grad():
        expected = model(x[:1])['reconstruction']
        stream = compress(model, x[:1])
        actual = decompress(model, stream)
    torch.testing.assert_close(actual, expected)
    assert actual.min() >= 0 and actual.max() <= 1
    print('End-to-end gradients and odd-size bitstream round trip passed.')
