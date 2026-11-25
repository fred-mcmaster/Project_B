"""Notebook-friendly training functions and the original command-line interface."""
import argparse
import copy
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from dataset import JPEGRestorationDataset
from model import ResidualRestorationCNN, count_parameters
from losses import SpatialFrequencyLoss
from metrics import psnr, batch_ssim


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True







@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    totals = {"input_psnr": 0.0, "restored_psnr": 0.0,
              "input_ssim": 0.0, "restored_ssim": 0.0}
    count = 0
    for degraded, clean, _ in loader:
        degraded, clean = degraded.to(device), clean.to(device)
        restored = model(degraded)
        n = clean.shape[0]
        totals["input_psnr"] += psnr(degraded, clean).sum().item()
        totals["restored_psnr"] += psnr(restored, clean).sum().item()
        totals["input_ssim"] += batch_ssim(degraded, clean).item() * n
        totals["restored_ssim"] += batch_ssim(restored, clean).item() * n
        count += n
    if not count:
        raise ValueError("Validation loader is empty.")
    return {key: value / count for key, value in totals.items()}



def make_loaders(cfg, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set = JPEGRestorationDataset(cfg["train_dir"], cfg["patch_size"],
                                       cfg["qualities"], train=True)
    val_set = JPEGRestorationDataset(cfg["val_dir"], cfg["patch_size"],
                                     cfg["qualities"], train=False)
    generator = torch.Generator().manual_seed(cfg["seed"])
    common = dict(batch_size=cfg["batch_size"], num_workers=cfg["num_workers"],
                  pin_memory=(device.type == "cuda"))
    return (DataLoader(train_set, shuffle=True, generator=generator, **common),
            DataLoader(val_set, shuffle=False, **common))



def train_experiment(cfg, out_dir, lambda_freq=0.0, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = copy.deepcopy(cfg)
    cfg["lambda_freq"] = float(lambda_freq)
    cfg["out_dir"] = str(out_dir)
    if cfg["epochs"] < 1:
        raise ValueError("epochs must be at least 1")
    seed_everything(cfg["seed"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    train_loader, val_loader = make_loaders(cfg, device)
    model = ResidualRestorationCNN(cfg["channels"], cfg["depth"]).to(device)
    criterion = SpatialFrequencyLoss(lambda_freq)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    print(f"{out_dir.name}: {count_parameters(model):,} parameters; lambda={lambda_freq}")
    best_psnr, history = -float("inf"), []
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        totals = {"train_loss": 0.0, "spatial_loss": 0.0, "freq_loss": 0.0}
        count = 0
        current_lr = optimizer.param_groups[0]["lr"]
        bar = tqdm(train_loader, desc=f"{out_dir.name}: {epoch}/{cfg['epochs']}")
        for degraded, clean, _ in bar:
            loss, parts = train_one_batch(model, degraded, clean, criterion,
                                           optimizer, scaler, device)
            n = clean.shape[0]
            totals["train_loss"] += loss * n
            totals["spatial_loss"] += parts["spatial"].item() * n
            totals["freq_loss"] += parts["freq"].item() * n
            count += n
            bar.set_postfix(loss=f"{loss:.4f}")
        scheduler.step()
        metrics = validate(model, val_loader, device)
        metrics.update({key: value / count for key, value in totals.items()})
        metrics.update(epoch=epoch, lr=current_lr)
        history.append(metrics)
        checkpoint = {"model": model.state_dict(), "args": cfg, "epoch": epoch,
                      "validation": metrics}
        torch.save(checkpoint, out_dir / "last.pt")
        if metrics["restored_psnr"] > best_psnr:
            best_psnr = metrics["restored_psnr"]
            torch.save(checkpoint, out_dir / "best.pt")
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))
        print(f"Epoch {epoch}: loss={metrics['train_loss']:.4f}; "
              f"val PSNR {metrics['input_psnr']:.2f} → {metrics['restored_psnr']:.2f} dB; "
              f"SSIM={metrics['restored_ssim']:.4f}")
    return pd.DataFrame(history)




def main():
    parser = argparse.ArgumentParser(description="Train a JPEG restoration CNN")
    parser.add_argument("--train_dir", required=True)
    parser.add_argument("--val_dir", required=True)
    parser.add_argument("--out_dir", default="runs/baseline")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--qualities", type=int, nargs="+", default=[10, 30, 50])
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lambda_freq", type=float, default=0.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = vars(parser.parse_args())
    train_experiment(args, args["out_dir"], args["lambda_freq"])


if __name__ == "__main__":
    main()
