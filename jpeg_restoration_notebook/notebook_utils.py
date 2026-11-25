"""Optional dataset preparation and offline demonstration helpers."""
import hashlib
import random
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from dataset import IMG_EXTS


def image_paths(root):
    return sorted(p for p in Path(root).rglob("*")
                  if p.is_file() and p.suffix.lower() in IMG_EXTS)



def prepare_image_splits(source_dir, destination, seed=42):
    """Copy an unsplit source collection into 80%/10%/10% splits; never overwrite."""
    source = Path(source_dir).resolve()
    destination = Path(destination).resolve()
    if source == destination or source in destination.parents:
        raise ValueError("Choose a destination outside the source directory.")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Destination is not empty: {destination}")
    paths = image_paths(source)
    if len(paths) < 10:
        raise ValueError("Use at least 10 images, or provide your own splits.")
    random.Random(seed).shuffle(paths)
    n_val = max(1, len(paths) // 10)
    n_test = max(1, len(paths) // 10)
    n_train = len(paths) - n_val - n_test
    groups = {"train": paths[:n_train],
              "val": paths[n_train:n_train + n_val],
              "test": paths[n_train + n_val:]}
    for split, files in groups.items():
        folder = destination / split
        folder.mkdir(parents=True, exist_ok=True)
        for i, path in enumerate(files):
            shutil.copy2(path, folder / f"{i:06d}_{path.name}")



def make_demo_images(root, counts=None, seed=123):
    """Generate independent synthetic RGB images for an offline pipeline check."""
    counts = counts or {"train": 12, "val": 4, "test": 5}
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:96, 0:96].astype(np.float32) / 95.0
    for split, count in counts.items():
        folder = Path(root) / split
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            frequency = rng.uniform(2.0, 12.0)
            phase = rng.uniform(0, 2 * np.pi)
            wave = 0.5 + 0.5 * np.sin(2 * np.pi * frequency * xx + phase)
            cx, cy = rng.uniform(0.2, 0.8, size=2)
            disk = ((xx - cx)**2 + (yy - cy)**2 < rng.uniform(0.02, 0.10)).astype(float)
            image = np.stack([0.6 * xx + 0.4 * wave,
                              0.65 * yy + 0.35 * disk,
                              0.5 * (1 - xx) + 0.5 * wave * disk], axis=-1)
            image += rng.normal(0, 0.025, image.shape)
            Image.fromarray((255 * np.clip(image, 0, 1)).astype(np.uint8)).save(
                folder / f"demo_{i:03d}.png")



def check_split_overlap(root):
    seen = {}
    summary = []
    for split in ("train", "val", "test"):
        paths = image_paths(Path(root) / split)
        if not paths:
            raise FileNotFoundError(f"No images in {Path(root) / split}")
        for path in paths:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in seen and seen[digest][0] != split:
                raise ValueError(f"Duplicate across splits: {seen[digest][1]} and {path}")
            seen[digest] = (split, str(path))
        summary.append({"split": split, "images": len(paths)})
    return pd.DataFrame(summary)
