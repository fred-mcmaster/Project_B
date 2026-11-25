import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision.transforms import functional as TF

from dataset import jpeg_compress_pil, JPEGRestorationDataset
from metrics import psnr, batch_ssim
from model import ResidualRestorationCNN


@torch.no_grad()
def show_comparison(models, cfg, index, quality, results_dir, mode="full", device=None, *, metric_fns=None):
    # Notebook callers can supply their shared metric functions.
    psnr_fn, ssim_fn = (psnr, batch_ssim) if metric_fns is None else metric_fns
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for model in models.values():
        model.eval()
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    ds = JPEGRestorationDataset(cfg["test_dir"], cfg["eval_patch_size"], [quality], train=False,
                                return_bpp=True)
    degraded, clean, _, bpp = ds[index]
    input_batch = degraded.unsqueeze(0).to(device)
    images = {"Clean target": clean, "JPEG": degraded}
    images.update({name: model(input_batch)[0].cpu() for name, model in models.items()})
    size = clean.shape[-1]
    zoom_size = max(16, size // 4)
    start = (size - zoom_size) // 2
    fig, axes = plt.subplots(2, len(images), figsize=(4 * len(images), 7))
    for col, (name, tensor) in enumerate(images.items()):
        array = tensor.permute(1, 2, 0).clamp(0, 1).numpy()
        axes[0, col].imshow(array)
        title = name
        if name != "Clean target":
            value = psnr_fn(tensor.unsqueeze(0), clean.unsqueeze(0)).item()
            score = ssim_fn(tensor.unsqueeze(0), clean.unsqueeze(0)).item()
            title += f"\nPSNR {value:.2f} dB | SSIM {score:.4f}"
        axes[0, col].set_title(title)
        axes[1, col].imshow(array[start:start + zoom_size, start:start + zoom_size],
                            interpolation="nearest")
        axes[1, col].set_title("Center detail")
        for row in range(2):
            axes[row, col].axis("off")
    fig.suptitle(f"{ds.paths[index].name} — JPEG Q={quality}, {bpp:.4f} bpp (same input stream for every method)" +
                 (" — synthetic demo" if mode == "demo" else ""))
    fig.tight_layout()
    path = results_dir / f"comparison_{index:03d}_q{quality}.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print("Saved:", path)

    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--image', required=True)
    p.add_argument('--quality', type=int, default=10)
    p.add_argument('--out', type=str, default='comparison.png')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    cfg = ckpt['args']
    model = ResidualRestorationCNN(cfg['channels'], cfg['depth']).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    clean_pil = Image.open(args.image).convert('RGB')
    degraded_pil, num_bytes = jpeg_compress_pil(clean_pil, args.quality, return_size=True)
    bpp = 8 * num_bytes / (clean_pil.width * clean_pil.height)
    degraded = TF.to_tensor(degraded_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        restored = model(degraded).squeeze(0).cpu()

    clean = TF.to_tensor(clean_pil)
    degraded_cpu = degraded.squeeze(0).cpu()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, img, title in zip(
        axes,
        [clean, degraded_cpu, restored],
        ['Clean target', f'JPEG Q={args.quality}, {bpp:.4f} bpp', f'Restored (same {bpp:.4f} bpp input)']
    ):
        ax.imshow(img.permute(1, 2, 0).clamp(0, 1))
        if title != 'Clean target':
            score_psnr = psnr(img.unsqueeze(0), clean.unsqueeze(0)).item()
            score_ssim = batch_ssim(img.unsqueeze(0), clean.unsqueeze(0)).item()
            title += f'\nPSNR {score_psnr:.2f} dB | SSIM {score_ssim:.4f}'
        ax.set_title(title)
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(args.out, dpi=180, bbox_inches='tight')
    print('Saved:', args.out)


if __name__ == '__main__':
    main()
