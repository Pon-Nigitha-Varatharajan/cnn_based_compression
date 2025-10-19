import io
import cv2
import numpy as np
import torch
from PIL import Image


def jpeg_codec(img, quality=10):
    # Convert NumPy array to tensor if needed
    if isinstance(img, np.ndarray):
        img_tensor = torch.from_numpy(img).float()
        if img_tensor.ndim == 3:  # [H, W, C] -> [1, C, H, W]
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
    else:
        img_tensor = img

    # Now you can safely check dim
    if img_tensor.dim() == 4:
        # convert tensor back to numpy for OpenCV JPEG compression
        img_np = (img_tensor.squeeze(0).permute(1,2,0).numpy()).astype(np.uint8)
    else:
        img_np = img_tensor.numpy().astype(np.uint8)

    # Encode as JPEG in memory
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', img_np, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    decimg = cv2.cvtColor(decimg, cv2.COLOR_BGR2RGB)
    decimg = decimg.astype(np.float32) / 255.0
    return decimg

def bicubic_upsample(img_tensor, scale=2):
    return torch.nn.functional.interpolate(img_tensor, scale_factor=scale, mode='bicubic', align_corners=False)


def psnr_torch(x, y):
    mse = torch.mean((x - y) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()


def ssim_numpy(a, b):
    # small wrapper to use skimage or simple implementation
    from skimage.metrics import structural_similarity
    a = np.clip(a, 0, 1)
    b = np.clip(b, 0, 1)
    # compute per-channel and average
    vals = []
    for ch in range(a.shape[2]):
        vals.append(structural_similarity(a[:,:,ch], b[:,:,ch], data_range=1.0))
    return float(np.mean(vals))