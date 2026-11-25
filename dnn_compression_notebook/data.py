"""Clean RGB images, prepared dataset splits, and saved manifests."""
import json
import random
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def source_group(path):
    # For this dataset: BC38_R02.png_crop_370.png -> BC38_R02.png.
    return Path(path).name.split('_crop_')[0]


def split_images(root, manifest, seed=1512):
    root, manifest = Path(root).resolve(), Path(manifest)
    if not root.is_dir():
        raise FileNotFoundError(f'Image directory not found: {root}')
    split_dirs = {s: (root / s).is_dir() for s in ('train', 'val', 'test')}
    prepared = all(split_dirs.values())
    if any(split_dirs.values()) and not prepared:
        raise ValueError('Prepared dataset requires train, val, and test folders')
    if manifest.exists():
        record = json.loads(manifest.read_text())
        if record['seed'] != seed or record['root'] != str(root):
            raise ValueError('Existing manifest uses another root/seed. Choose a new manifest path.')
        splits = record['splits']
    elif prepared:
        splits = {s: [str(p.relative_to(root)) for p in sorted((root / s).rglob('*'))
                      if p.is_file() and p.suffix.lower() in EXTENSIONS]
                  for s in ('train', 'val', 'test')}
        record = dict(root=str(root), seed=seed, split_rule='predefined_directories', splits=splits)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(record, indent=2))
    else:
        paths = sorted(p for p in root.rglob('*') if p.suffix.lower() in EXTENSIONS and p.is_file())
        groups = sorted({source_group(p) for p in paths})
        if len(groups) < 3:
            raise ValueError('At least three source-image groups are required.')
        random.Random(seed).shuffle(groups)
        nv = max(1, round(0.15 * len(groups)))
        nt = max(1, round(0.15 * len(groups)))
        group_sets = dict(val=set(groups[:nv]), test=set(groups[nv:nv+nt]),
                          train=set(groups[nv+nt:]))
        splits = {s: [str(p.relative_to(root)) for p in paths if source_group(p) in gs]
                  for s, gs in group_sets.items()}
        record = dict(root=str(root), seed=seed, group_rule='filename before _crop_', splits=splits)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(record, indent=2))
    seen = set()
    for s in ('train', 'val', 'test'):
        if not splits[s]:
            raise ValueError(f'Empty {s} split')
        if prepared:
            if any(Path(p).parts[0] != s for p in splits[s]):
                raise ValueError('Manifest does not match the prepared split folders')
        else:
            groups = {source_group(p) for p in splits[s]}
            if seen & groups:
                raise ValueError('Source-image leakage across splits')
            seen |= groups
        if not all((root / p).is_file() for p in splits[s]):
            raise FileNotFoundError(f'Missing images in {s} manifest')
    return {s: [root / p for p in paths] for s, paths in splits.items()}


class ImageDataset(Dataset):
    def __init__(self, paths, crop_size=64, train=False):
        self.paths, self.crop_size, self.train = list(paths), int(crop_size), train
        if self.crop_size < 1 or not self.paths:
            raise ValueError('Require positive crop_size and nonempty paths')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with Image.open(self.paths[index]) as image:
            image = image.convert('RGB')
        size = self.crop_size
        w, h = image.size
        if min(w, h) < size:
            scale = size / min(w, h)
            image = image.resize((round(w * scale), round(h * scale)), Image.Resampling.BICUBIC)
        w, h = image.size
        left = random.randint(0, w-size) if self.train else (w-size)//2
        top = random.randint(0, h-size) if self.train else (h-size)//2
        image = image.crop((left, top, left+size, top+size))
        if self.train and random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return torch.from_numpy(np.array(image, dtype=np.float32) / 255).permute(2, 0, 1)


def make_loaders(splits, crop_size=64, batch_size=8, seed=1512, limits=None):
    loaders = {}
    for offset, s in enumerate(('train', 'val', 'test')):
        paths = list(splits[s])
        # Sample across source groups instead of taking a sorted prefix.
        random.Random(seed + offset).shuffle(paths)
        if limits and limits.get(s):
            paths = paths[:limits[s]]
        loaders[s] = DataLoader(ImageDataset(paths, crop_size, train=s == 'train'),
                                batch_size=batch_size, shuffle=s == 'train', num_workers=0,
                                generator=torch.Generator().manual_seed(seed + offset))
    return loaders
