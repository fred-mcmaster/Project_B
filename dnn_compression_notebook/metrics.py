"""RGB SSIM, using the same TorchMetrics protocol as JPEG restoration."""
import numpy as np
import torch
from torchmetrics.functional.image import structural_similarity_index_measure


def ssim_per_image(pred, target):
    """Return [B] SSIM values on [0,1] RGB, Gaussian window 11, sigma 1.5."""
    return structural_similarity_index_measure(pred.float(), target.float(),
        data_range=1.0, gaussian_kernel=True, sigma=1.5, kernel_size=11,
        reduction='none').reshape(-1)


def array_ssim(original, reconstruction):
    a=torch.from_numpy(np.array(original, dtype=np.float32, copy=True)).permute(2,0,1)[None]
    b=torch.from_numpy(np.array(reconstruction, dtype=np.float32, copy=True)).permute(2,0,1)[None]
    return float(ssim_per_image(b,a)[0])
