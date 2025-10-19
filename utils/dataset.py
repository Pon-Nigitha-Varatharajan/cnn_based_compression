import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path


class ImagePatchDataset(Dataset):
    """Extract patches from images in a directory (paper-style: 40x40, stride 20)
    """
    def __init__(self, root_dir, patch_size=40, stride=20, transforms=True):
        self.patches = []
        self.patch_size = patch_size
        self.stride = stride
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        files = [p for p in Path(root_dir).glob('*') if p.suffix.lower() in exts]
        if not files:
            raise ValueError(f"No image files found in {root_dir}")
        for f in sorted(files):
            img = cv2.imread(str(f))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype('float32') / 255.0
            h, w = img.shape[:2]
            for i in range(0, max(1, h - patch_size), stride):
                for j in range(0, max(1, w - patch_size), stride):
                    ph = img[i:i+patch_size, j:j+patch_size]
                    if ph.shape[0] != patch_size or ph.shape[1] != patch_size:
                        continue
                    self.patches.append(ph.transpose(2,0,1))

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        import torch
        return torch.tensor(self.patches[idx], dtype=torch.float32)