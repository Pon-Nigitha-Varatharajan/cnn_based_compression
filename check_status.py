import torch
import cv2
import numpy as np
from models.comcnn import ComCNN
from models.reccnn import RecCNN
from utils.codec import jpeg_codec, psnr_torch, ssim_numpy
import torch.nn.functional as F

def test_enhanced():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Testing enhanced models on {device}")
    
    # Load enhanced models
    com_cnn = ComCNN().to(device)
    rec_cnn = RecCNN().to(device)
    
    try:
        com_cnn.load_state_dict(torch.load('checkpoints/comcnn.pth', map_location=device))
        rec_cnn.load_state_dict(torch.load('checkpoints/reccnn.pth', map_location=device))
        print("✓ Loaded enhanced models")
    except:
        com_cnn.load_state_dict(torch.load('checkpoints/comcnn.pth', map_location=device))
        rec_cnn.load_state_dict(torch.load('checkpoints/reccnn.pth', map_location=device))
        print("✓ Loaded standard models")
    
    com_cnn.eval()
    rec_cnn.eval()
    
    # Test on a sample image
    test_image_path = 'samples/baby.png'  # Changed to samples folder
    try:
        img = cv2.imread(test_image_path)
        if img is None:
            print(f"❌ Could not load image: {test_image_path}")
            return
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return
    
    # Process through pipeline
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        compressed = com_cnn(img_tensor)
        # Upscale compressed output to match original size
        compressed_up = F.interpolate(compressed, size=img.shape[:2], mode='bilinear', align_corners=False)
    
    compressed_np = compressed_up.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    # JPEG compression
    compressed_uint8 = (np.clip(compressed_np, 0, 1) * 255).astype(np.uint8)
    compressed_jpeg = jpeg_codec(compressed_uint8, quality=10)
    
    # Reconstruction
    rec_input = compressed_jpeg.astype(np.float32) / 255.0
    rec_tensor = torch.from_numpy(rec_input).permute(2, 0, 1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        rec_out = rec_cnn(rec_tensor)
    
    rec_np = rec_out.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    # Ensure both images have same dimensions for metrics
    min_h = min(rec_np.shape[0], img.shape[0])
    min_w = min(rec_np.shape[1], img.shape[1])
    
    rec_np_crop = rec_np[:min_h, :min_w]
    img_crop = img[:min_h, :min_w]
    
    # Metrics
    psnr_val = psnr_torch(
        torch.from_numpy(rec_np_crop).permute(2, 0, 1).unsqueeze(0).to(device),
        torch.from_numpy(img_crop).permute(2, 0, 1).unsqueeze(0).to(device)
    )
    ssim_val = ssim_numpy(rec_np_crop, img_crop)
    
    print(f"\n=== RESULTS ===")
    print(f"Original size: {img.shape[1]}x{img.shape[0]}")
    print(f"Compressed size: {compressed.shape[3]}x{compressed.shape[2]} (before upscale)")
    print(f"Reconstructed size: {rec_np.shape[1]}x{rec_np.shape[0]}")
    print(f"PSNR: {psnr_val:.2f} dB")
    print(f"SSIM: {ssim_val:.4f}")
    print(f"Quality: {'GOOD' if psnr_val > 28 else 'POOR' if psnr_val < 25 else 'AVERAGE'}")

if __name__ == "__main__":
    test_enhanced()