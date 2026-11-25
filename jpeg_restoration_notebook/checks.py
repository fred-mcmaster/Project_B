"""Run lightweight mathematical, dataset, and optimization checks."""
import math
import io
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from dataset import JPEGRestorationDataset, jpeg_compress_pil
from model import ResidualRestorationCNN
from metrics import psnr, batch_ssim
from train import seed_everything


def run_correctness_checks(seed=42, device=None, *, metric_fns=None, train_step=None, criterion_fn=None):
    # Notebook callers can supply their shared metric functions.
    psnr_fn, ssim_fn = (psnr, batch_ssim) if metric_fns is None else metric_fns
    if train_step is None:
        from train import train_one_batch
        train_step = train_one_batch
    criterion_fn = F.l1_loss if criterion_fn is None else criterion_fn
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(7)
    rgb = Image.fromarray(np.random.default_rng(7).integers(
        0, 256, size=(13, 9, 3), dtype=np.uint8))
    jpeg = jpeg_compress_pil(rgb, 10)
    assert jpeg.mode == "RGB" and jpeg.size == rgb.size
    measured, byte_count = jpeg_compress_pil(rgb, 10, return_size=True)
    reference = io.BytesIO()
    rgb.save(reference, format='JPEG', quality=10, subsampling=0, optimize=False)
    assert byte_count == len(reference.getvalue())
    np.testing.assert_array_equal(np.asarray(jpeg), np.asarray(measured))
    with tempfile.TemporaryDirectory() as folder:
        rgb.save(Path(folder) / "small.png")
        ds = JPEGRestorationDataset(folder, 32, [10, 30, 50], train=False)
        y, x, q = ds[0]
        y2, x2, q2 = ds[0]
        assert y.shape == x.shape == (3, 32, 32)
        assert torch.equal(y, y2) and torch.equal(x, x2) and q == q2 == 10
        assert 0 <= y.min() <= y.max() <= 1 and 0 <= x.min() <= x.max() <= 1
        ds_train = JPEGRestorationDataset(folder, 32, [10, 30, 50], train=True)
        assert ds_train._choose_train_quality() in {10, 30, 50}
        assert ds_train[0][0].shape == (3, 32, 32)
        measured_ds = JPEGRestorationDataset(folder, 32, [10], train=False, return_bpp=True)
        measured_y, measured_x, measured_q, bpp = measured_ds[0]
        torch.testing.assert_close(measured_y, y)
        crop = Image.fromarray(np.rint(measured_x.permute(1, 2, 0).numpy()*255).astype(np.uint8))
        reference = io.BytesIO()
        crop.save(reference, format='JPEG', quality=measured_q, subsampling=0, optimize=False)
        assert bpp == 8*len(reference.getvalue())/(32*32)

    zeros = torch.zeros(2, 3, 32, 32)
    torch.testing.assert_close(psnr_fn(zeros, torch.full_like(zeros, 0.1)), torch.full((2,), 20.0))
    torch.testing.assert_close(psnr_fn(zeros, zeros), torch.full((2,), 100.0))
    torch.testing.assert_close(ssim_fn(zeros, zeros), torch.tensor(1.0))
    pred, target = torch.rand_like(zeros).requires_grad_(), torch.rand_like(zeros)
    loss0 = criterion_fn(pred, target)
    torch.testing.assert_close(loss0, F.l1_loss(pred, target))
    loss0.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()

    model = ResidualRestorationCNN(8, 4).to(device)
    input_tensor = torch.rand(2, 3, 32, 32, device=device, requires_grad=True)
    output = model(input_tensor)
    assert output.shape == input_tensor.shape and 0 <= output.min() <= output.max() <= 1
    output.mean().backward()
    assert input_tensor.grad is not None
    assert all(p.grad is not None for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    before = [p.detach().clone() for p in model.parameters()]
    total = train_step(model, torch.rand_like(zeros), torch.rand_like(zeros),
                               criterion_fn, optimizer, scaler, device)
    assert math.isfinite(total)
    assert any(not torch.equal(old, new) for old, new in zip(before, model.parameters()))
    seed_everything(seed)
    print("All correctness checks passed.")



if __name__ == "__main__":
    torch.set_num_threads(min(4, torch.get_num_threads()))
    run_correctness_checks()
