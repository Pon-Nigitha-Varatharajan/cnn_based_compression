import os
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2

class ImagePatchDataset(Dataset):
    def __init__(self, data_dir, patch_size=40, stride=20, augment=True):
        self.patch_size = patch_size
        self.stride = stride
        self.augment = augment
        
        # Load images
        self.image_paths = []
        for fname in os.listdir(data_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                self.image_paths.append(os.path.join(data_dir, fname))
        
        if not self.image_paths:
            raise ValueError(f"No images found in {data_dir}")
        
        # Extract patches
        self.patches = []
        for img_path in self.image_paths:
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            
            patches = self.extract_patches(img)
            self.patches.extend(patches)
            
            if self.augment:
                # Data augmentation: flips
                augmented = self.augment_image(img)
                for aug_img in augmented:
                    aug_patches = self.extract_patches(aug_img)
                    self.patches.extend(aug_patches)
        
        print(f"Loaded {len(self.patches)} patches from {len(self.image_paths)} images")
    
    def extract_patches(self, img):
        patches = []
        h, w = img.shape[:2]
        
        # Ensure we have at least one patch
        if h < self.patch_size or w < self.patch_size:
            # Resize image to be at least patch_size
            scale = max(self.patch_size / h, self.patch_size / w)
            new_h, new_w = int(h * scale), int(w * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            h, w = img.shape[:2]
        
        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):
                patch = img[y:y+self.patch_size, x:x+self.patch_size]
                patches.append(patch)
        
        return patches
    
    def augment_image(self, img):
        augmented = []
        
        # Horizontal flip
        augmented.append(cv2.flip(img, 1))
        # Vertical flip
        augmented.append(cv2.flip(img, 0))
        
        return augmented
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        patch = self.patches[idx]
        # Convert to tensor [C, H, W]
        patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).float()
        return patch_tensor