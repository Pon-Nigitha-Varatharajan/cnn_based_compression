import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

class ImagePatchDataset(Dataset):
    def __init__(self, data_dir, patch_size=40, stride=20):
        self.patch_size = patch_size
        self.stride = stride
        self.patches = []
        
        # Load images
        image_paths = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
            image_paths.extend(Path(data_dir).rglob(ext))
        
        # Extract patches from each image
        for img_path in image_paths:
            img = cv2.imread(str(img_path))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                patches = self.extract_patches(img)
                self.patches.extend(patches)
        
        print(f"Loaded {len(self.patches)} patches from {len(image_paths)} images")
    
    def extract_patches(self, img):
        patches = []
        h, w = img.shape[:2]
        
        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):
                patch = img[y:y+self.patch_size, x:x+self.patch_size]
                patch = patch.astype(np.float32) / 255.0
                patches.append(patch)
        
        return patches
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        patch = self.patches[idx]
        # Convert HWC to CHW and to tensor
        patch_tensor = torch.from_numpy(patch).permute(2, 0, 1)
        return patch_tensor