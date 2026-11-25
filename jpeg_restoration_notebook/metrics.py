"""RGB PSNR and SSIM for normalized images."""
import torch
from torchmetrics.functional.image import structural_similarity_index_measure

def psnr(pred, target, data_range=1.0):
    mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3))
    return 10.0 * torch.log10((data_range ** 2) / torch.clamp(mse, min=1e-10))


def batch_ssim(pred, target):
    return structural_similarity_index_measure(pred.float(), target.float(), data_range=1.0)
