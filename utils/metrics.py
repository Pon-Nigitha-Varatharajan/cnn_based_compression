import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def calculate_psnr(img1, img2):
    """
    img1, img2: numpy arrays with values in range [0,1] or [0,255]
    """
    return peak_signal_noise_ratio(img1, img2, data_range=1.0 if img1.max() <= 1.0 else 255.0)

def calculate_ssim(img1, img2):
    """
    img1, img2: numpy arrays with shape HxWxC and values in range [0,1] or [0,255]
    """
    data_range = 1.0 if img1.max() <= 1.0 else 255.0
    if img1.ndim == 3 and img1.shape[2] == 3:
        return structural_similarity(img1, img2, multichannel=True, data_range=data_range)
    else:
        return structural_similarity(img1, img2, data_range=data_range)