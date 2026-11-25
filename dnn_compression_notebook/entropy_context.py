"""Fit shared causal index frequencies for lossless latent arithmetic coding."""
from pathlib import Path
import torch
from engine import load_checkpoint


@torch.no_grad()
def train_tables(model, loader, device):
    """Count training-set VQ index pairs for left and upper causal contexts."""
    model.eval()
    k = model.config['codebook_size']
    left_counts = torch.ones(k+1, k, dtype=torch.int64)
    up_counts = torch.ones(k+1, k, dtype=torch.int64)
    for x in loader:
        ids = model(x.to(device))['indices'].cpu().long()
        left = torch.full_like(ids, k)
        up = torch.full_like(ids, k)
        left[:, :, 1:] = ids[:, :, :-1]
        up[:, 1:, :] = ids[:, :-1, :]
        left_counts += torch.bincount((left*k + ids).reshape(-1),
                                     minlength=(k+1)*k).reshape(k+1, k)
        up_counts += torch.bincount((up*k + ids).reshape(-1),
                                   minlength=(k+1)*k).reshape(k+1, k)
    return left_counts.to(torch.int32), up_counts.to(torch.int32)


def fit_context_checkpoint(source, loader, soft_quantize, device, output=None):
    """Save a checkpoint with context tables fitted only from ``loader`` images."""
    source = Path(source)
    output = Path(output) if output is not None else source.with_name(source.stem + '_arithmetic.pt')
    if output.exists():
        raise FileExistsError(output)
    model, state = load_checkpoint(source, soft_quantize, device)
    if any(key.startswith('coder_') for key in state['state_dict']):
        raise ValueError('Source checkpoint already contains entropy context tables')
    left, up = train_tables(model, loader, device)
    state['state_dict']['coder_left'] = left
    state['state_dict']['coder_up'] = up
    state['coding_context'] = dict(training_images=len(loader.dataset),
        method='causal left/up codebook-index frequencies with add-one smoothing')
    torch.save(state, output)
    return output
