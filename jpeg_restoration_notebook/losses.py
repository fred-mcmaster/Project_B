"""Differentiable block-DCT and spatial/frequency reconstruction loss."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def dct_matrix(n=8, device=None, dtype=torch.float32):
    C = torch.zeros(n, n, device=device, dtype=dtype)
    for k in range(n):
        alpha = math.sqrt(1 / n) if k == 0 else math.sqrt(2 / n)
        for i in range(n):
            C[k, i] = alpha * math.cos(math.pi * (2 * i + 1) * k / (2 * n))
    return C


def block_dct(x, block=8):
    """Non-overlapping block DCT. x: [B,C,H,W]."""
    B, C, H, W = x.shape
    pad_h = (block - H % block) % block
    pad_w = (block - W % block) % block
    if pad_h or pad_w:
        # Reflection requires each pad to be smaller than the source dimension.
        mode = 'reflect' if pad_h < H and pad_w < W else 'replicate'
        x = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)

    H2, W2 = x.shape[-2:]
    patches = x.unfold(2, block, block).unfold(3, block, block)
    # [B,C,Nh,Nw,block,block]
    Cmat = dct_matrix(block, x.device, x.dtype)
    tmp = torch.matmul(Cmat, patches)
    coeff = torch.matmul(tmp, Cmat.t())
    return coeff


class SpatialFrequencyLoss(nn.Module):
    """
    L = L1_spatial + lambda_freq * L1_DCT
    """
    def __init__(self, lambda_freq=0.0, block=8):
        super().__init__()
        self.lambda_freq = lambda_freq
        self.block = block

    def forward(self, pred, target):
        spatial = F.l1_loss(pred, target)
        if self.lambda_freq <= 0:
            return spatial, {'spatial': spatial.detach(), 'freq': torch.tensor(0., device=pred.device)}

        # Keep transform arithmetic in float32 when the CNN uses CUDA AMP.
        with torch.autocast(device_type=pred.device.type, enabled=False):
            pred_dct = block_dct(pred.float(), self.block)
            target_dct = block_dct(target.float(), self.block)
            freq = F.l1_loss(pred_dct, target_dct)
        total = spatial + self.lambda_freq * freq
        return total
    # , {'spatial': spatial.detach(), 'freq': freq.detach()}
