import torch
import torch.nn.functional as F
import numpy as np
import cv2
from math import log10

def jpeg_codec(img_np, quality=10):
    """Apply JPEG compression and decompression"""
    if img_np.dtype != np.uint8:
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    
    # Handle grayscale images
    if len(img_np.shape) == 2:
        img_np = np.expand_dims(img_np, axis=2)
    
    # Convert RGB to BGR for OpenCV
    if img_np.shape[2] == 3:
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = img_np
        
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', img_bgr, encode_param)
    
    if not result:
        return img_np
        
    decimg = cv2.imdecode(encimg, cv2.IMREAD_COLOR)
    if decimg is None:
        return img_np
        
    if decimg.shape[2] == 3:
        decimg_rgb = cv2.cvtColor(decimg, cv2.COLOR_BGR2RGB)
    else:
        decimg_rgb = decimg
        
    return decimg_rgb

def psnr_torch(img1, img2):
    """Calculate PSNR between two torch tensors"""
    # Ensure both tensors are on same device and have same shape
    img2 = img2.to(img1.device)
    
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    return 20 * log10(1.0) - 10 * torch.log10(mse)

def ssim_numpy(img1, img2):
    """Calculate SSIM between two numpy arrays"""
    # Ensure both arrays have the same shape
    min_dim = min(img1.shape[0], img2.shape[0]), min(img1.shape[1], img2.shape[1])
    img1 = img1[:min_dim[0], :min_dim[1]]
    img2 = img2[:min_dim[0], :min_dim[1]]
    
    C1 = (0.01 * 1.0) ** 2  # Using 1.0 as data range since images are [0,1]
    C2 = (0.03 * 1.0) ** 2
    
    mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return np.mean(ssim_map)

def bicubic_upsample(img, scale_factor=2):
    """Bicubic upsampling for numpy array"""
    h, w = img.shape[:2]
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)