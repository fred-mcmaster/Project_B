"""Online JPEG degradation and clean/degraded image pairs."""
import io
import random
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def jpeg_compress_pil(img: Image.Image, quality: int, return_size=False):
    """RGB JPEG round trip; optionally return the exact encoded file byte count.

    Includes JPEG headers/metadata. Keeps the existing 4:4:4, optimize=False
    protocol. Measure this stream, not a re-encoding of the decoded pixels.
    """
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=int(quality), subsampling=0, optimize=False)
    num_bytes = buf.tell()
    buf.seek(0)
    with Image.open(buf) as decoded:
        image = decoded.convert('RGB')
    return (image, num_bytes) if return_size else image


class JPEGRestorationDataset(Dataset):
    """
    Returns:
        degraded: Tensor [3,H,W] in [0,1]
        clean:    Tensor [3,H,W] in [0,1]
        quality:  int
    """
    def __init__(
        self,
        root,
        patch_size=128,
        qualities=(10, 20, 30, 40),
        train=True,
        return_bpp=False,
    ):
        self.root = Path(root)
        self.paths = sorted(p for p in self.root.rglob('*')
                            if p.is_file() and p.suffix.lower() in IMG_EXTS)
        if not self.paths:
            raise RuntimeError(f'No images found in {self.root}')
        if int(patch_size) != patch_size or patch_size <= 0:
            raise ValueError('patch_size must be a positive integer')
        self.patch_size = int(patch_size)
        self.qualities = tuple(int(q) for q in qualities)
        if not self.qualities or any(q < 1 or q > 100 for q in self.qualities):
            raise ValueError('qualities must be nonempty and between 1 and 100')
        self.train = train
        self.return_bpp = return_bpp

    def __len__(self):
        return len(self.paths)

    def _crop(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        ps = self.patch_size
        if min(w, h) < ps:
            scale = ps / min(w, h)
            img = img.resize((round(w * scale), round(h * scale)), Image.BICUBIC)
            w, h = img.size

        if self.train:
            left = random.randint(0, w - ps)
            top = random.randint(0, h - ps)
        else:
            left = max(0, (w - ps) // 2)
            top = max(0, (h - ps) // 2)
        return img.crop((left, top, left + ps, top + ps))

    def _choose_train_quality(self):
        return random.choice(self.qualities)

    def __getitem__(self, idx):
        with Image.open(self.paths[idx]) as image:
            clean = image.convert('RGB')
        clean = self._crop(clean)

        if self.train:
            if random.random() < 0.5:
                clean = TF.hflip(clean)
            if random.random() < 0.5:
                clean = TF.vflip(clean)
            quality = self._choose_train_quality()
        else:
            quality = self.qualities[idx % len(self.qualities)]

        degraded, num_bytes = jpeg_compress_pil(clean, quality, return_size=True)
        bpp = 8 * num_bytes / (clean.width * clean.height)

        clean = TF.to_tensor(clean)
        degraded = TF.to_tensor(degraded)
        if self.return_bpp:
            return degraded, clean, quality, bpp
        return degraded, clean, quality
