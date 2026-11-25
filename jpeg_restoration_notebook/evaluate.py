import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import JPEGRestorationDataset
from model import ResidualRestorationCNN
from metrics import psnr, batch_ssim
import pandas as pd
from tqdm.auto import tqdm


@torch.no_grad()
def evaluate_quality(model, data_dir, quality, patch_size, batch_size, device):
    ds = JPEGRestorationDataset(
        data_dir, patch_size=patch_size, qualities=[quality], train=False, return_bpp=True
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    rows = {'jpeg_psnr': [], 'model_psnr': [],
            'jpeg_ssim': [], 'model_ssim': [], 'jpeg_bpp': []}

    weights = []
    model.eval()
    for degraded, clean, _, bpp in loader:
        degraded, clean = degraded.to(device), clean.to(device)
        restored = model(degraded)
        weights.append(clean.shape[0])
        rows['jpeg_bpp'].append(bpp.mean().item())

        rows['jpeg_psnr'].append(psnr(degraded, clean).mean().item())
        rows['model_psnr'].append(psnr(restored, clean).mean().item())
        rows['jpeg_ssim'].append(batch_ssim(degraded, clean).item())
        rows['model_ssim'].append(batch_ssim(restored, clean).item())

    return {k: float(np.average(v, weights=weights)) for k, v in rows.items()}


def load_checkpoint(path, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    cfg = checkpoint["args"]
    model = ResidualRestorationCNN(cfg["channels"], cfg["depth"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, cfg



@torch.no_grad()
def evaluate_all(models, cfg, test_qualities, device=None, train_qualities=None, *, metric_fns=None):
    # Notebook callers can supply their shared metric functions.
    psnr_fn, ssim_fn = (psnr, batch_ssim) if metric_fns is None else metric_fns
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actual_train_qualities = set(cfg["qualities"] if train_qualities is None else train_qualities)
    for model in models.values():
        model.eval()
    rows = []
    for quality in tqdm(test_qualities, desc="Test quality factors"):
        ds = JPEGRestorationDataset(cfg["test_dir"], cfg["eval_patch_size"],
                                    [quality], train=False, return_bpp=True)
        loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False,
                            num_workers=cfg["num_workers"])
        totals = {name: {"PSNR": 0.0, "SSIM": 0.0}
                  for name in ["JPEG", *models]}
        count, total_bpp = 0, 0.0
        for degraded, clean, _, bpp in loader:
            degraded, clean = degraded.to(device), clean.to(device)
            predictions = {"JPEG": degraded}
            predictions.update({name: model(degraded) for name, model in models.items()})
            n = clean.shape[0]
            for name, pred in predictions.items():
                totals[name]["PSNR"] += psnr_fn(pred, clean).sum().item()
                totals[name]["SSIM"] += ssim_fn(pred, clean).item() * n
            count += n
            total_bpp += bpp.sum().item()
        if not count:
            raise ValueError("Test loader is empty.")
        for name, values in totals.items():
            rows.append({"Q": quality, "Seen in training": quality in actual_train_qualities,
                         "Method": name, "Images": count, "BPP": total_bpp / count,
                         **{key: value / count for key, value in values.items()}})
    return pd.DataFrame(rows)



def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data_dir', required=True)
    p.add_argument('--qualities', type=int, nargs='+', default=[10, 20, 30, 40, 50])
    p.add_argument('--patch_size', type=int, default=256)
    p.add_argument('--batch_size', type=int, default=8)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    train_args = ckpt['args']
    model = ResidualRestorationCNN(train_args['channels'], train_args['depth']).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    print('Q | JPEG BPP | JPEG PSNR | Model PSNR | JPEG SSIM | Model SSIM')
    print('-' * 90)
    for q in args.qualities:
        r = evaluate_quality(model, args.data_dir, q, args.patch_size, args.batch_size, device)
        print(
            f'{q:2d} | {r["jpeg_bpp"]:8.4f} | {r["jpeg_psnr"]:9.3f} | '
            f'{r["model_psnr"]:10.3f} | {r["jpeg_ssim"]:.4f} | '
            f'{r["model_ssim"]:.4f}'
        )


if __name__ == '__main__':
    main()
