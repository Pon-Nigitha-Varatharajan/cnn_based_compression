# utils/metrics.py
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def calculate_psnr(img1, img2):
    """
    img1, img2: numpy arrays with values in range [0,1] or [0,255]
    """
    return peak_signal_noise_ratio(img1, img2, data_range=img2.max() - img2.min())

def calculate_ssim(img1, img2):
    """
    img1, img2: numpy arrays with shape HxWxC and values in range [0,1] or [0,255]
    """
    if img1.ndim == 3 and img1.shape[2] == 3:
        return structural_similarity(img1, img2, multichannel=True, data_range=img2.max() - img2.min())
    else:
        return structural_similarity(img1, img2, data_range=img2.max() - img2.min())