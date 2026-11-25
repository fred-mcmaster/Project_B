"""Compact, single-scale RDVQ-inspired teaching model; no RDVQ dependency."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1),
                                 nn.SiLU(), nn.Conv2d(channels, channels, 3, padding=1))

    def forward(self, x):
        return x + self.net(x)


class Encoder(nn.Module):
    def __init__(self, width=24, latent_dim=8, levels=3, res_blocks=1):
        super().__init__()
        layers, cin = [], 3
        for i in range(levels):
            cout = width * min(2 ** i, 2)
            layers += [nn.Conv2d(cin, cout, 4, stride=2, padding=1), nn.SiLU()]
            layers += [ResidualBlock(cout) for _ in range(res_blocks)]
            cin = cout
        layers += [nn.Conv2d(cin, latent_dim, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, width=24, latent_dim=8, levels=3, res_blocks=1):
        super().__init__()
        channels = [width * min(2 ** i, 2) for i in range(levels)]
        layers = [nn.Conv2d(latent_dim, channels[-1], 1)]
        cin = channels[-1]
        for cout in reversed(channels):
            layers += [ResidualBlock(cin) for _ in range(res_blocks)]
            layers += [nn.Upsample(scale_factor=2, mode='nearest'),
                       nn.Conv2d(cin, cout, 3, padding=1), nn.SiLU()]
            cin = cout
        layers += [nn.Conv2d(cin, 3, 3, padding=1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)


class CompressionModel(nn.Module):
    """Inject the student's differentiable soft_quantize(flat, codebook, tau).

    Input/output RGB in [0,1]. Training uses soft vectors (or optional hard
    forward/soft backward). Evaluation always uses nearest codebook vectors.
    """
    def __init__(self, soft_quantize, width=24, latent_dim=8, levels=3,
                 codebook_size=64, straight_through=False, res_blocks=1):
        super().__init__()
        if min(width, latent_dim, levels) < 1 or levels > 4 or codebook_size < 2:
            raise ValueError('Require positive width/dim, levels in 1..4, K >= 2')
        if not isinstance(res_blocks, int) or res_blocks < 1:
            raise ValueError('res_blocks must be a positive integer')
        self.config = dict(width=width, latent_dim=latent_dim, levels=levels,
                           codebook_size=codebook_size, straight_through=straight_through)
        # Preserve old checkpoint configs and bitstream fingerprints at depth 1.
        if res_blocks != 1:
            self.config['res_blocks'] = res_blocks
        self.soft_quantize = soft_quantize
        self.factor = 2 ** levels
        self.encoder = Encoder(width, latent_dim, levels, res_blocks)
        self.decoder = Decoder(width, latent_dim, levels, res_blocks)
        self.codebook = nn.Parameter(torch.randn(codebook_size, latent_dim) * 0.1)
        self.prior_logits = nn.Parameter(torch.zeros(codebook_size))

    def encode(self, x):
        h, w = x.shape[-2:]
        padded = F.pad(x, (0, (-w) % self.factor, 0, (-h) % self.factor), mode='replicate')
        return self.encoder(padded)

    def hard_indices(self, flat):
        # Hard nearest-neighbor path is provided; the soft path is the exercise.
        return (flat.square().sum(-1, keepdim=True) + self.codebook.square().sum(-1)
                - 2 * flat @ self.codebook.T).argmin(-1)

    def decode_indices(self, indices, height, width):
        z = F.embedding(indices, self.codebook).permute(0, 3, 1, 2).contiguous()
        return self.decoder(z)[..., :height, :width]

    def forward(self, x, temperature=0.5):
        b, _, h, w = x.shape
        z = self.encode(x)
        _, d, zh, zw = z.shape
        flat = z.permute(0, 2, 3, 1).reshape(-1, d)
        indices = self.hard_indices(flat)
        costs = -F.log_softmax(self.prior_logits, dim=0) / math.log(2)
        if self.training:
            soft, probabilities = self.soft_quantize(flat, self.codebook, temperature)
            quantized = soft
            if self.config['straight_through']:
                hard = F.embedding(indices, self.codebook)
                quantized = soft + (hard - soft).detach()
            bits = (probabilities * costs).sum()
        else:
            quantized = F.embedding(indices, self.codebook)
            bits = costs[indices].sum()
        zq = quantized.reshape(b, zh, zw, d).permute(0, 3, 1, 2).contiguous()
        return dict(reconstruction=self.decoder(zq)[..., :h, :w],
                    estimated_bpp=bits / (b * h * w),
                    indices=indices.reshape(b, zh, zw))
