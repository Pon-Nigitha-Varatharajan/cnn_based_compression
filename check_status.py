import torch
import cv2
import numpy as np
from models.comcnn import ComCNN
from models.reccnn import RecCNN
from utils.codec import jpeg_codec, psnr_torch, ssim_numpy
import torch.nn.functional as F
import os
from pathlib import Path

def test_on_image(image_path, com_cnn, rec_cnn, device, quality=50):
    """Test pipeline on a single image"""
    print(f"\n{'='*60}")
    print(f"Testing: {image_path}")
    print(f"{'='*60}")
    
    # Load image
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"❌ Could not load image: {image_path}")
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return None
    
    h_orig, w_orig = img.shape[:2]
    
    # Ensure divisible by 2
    if h_orig % 2 != 0 or w_orig % 2 != 0:
        h_new = h_orig - (h_orig % 2)
        w_new = w_orig - (w_orig % 2)
        img = img[:h_new, :w_new, :]
        print(f"⚠️  Cropped to make divisible by 2: {w_orig}x{h_orig} → {w_new}x{h_new}")
    
    # Process through pipeline
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        compressed = com_cnn(img_tensor)
        # Upscale compressed output to match original size
        compressed_up = F.interpolate(compressed, size=img.shape[:2], mode='bilinear', align_corners=False)
    
    compressed_np = compressed_up.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    # JPEG compression
    compressed_uint8 = (np.clip(compressed_np, 0, 1) * 255).astype(np.uint8)
    compressed_jpeg = jpeg_codec(compressed_uint8, quality=quality)
    
    # Save compressed image for visual inspection
    output_dir = Path('test_results')
    output_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(output_dir / f"{Path(image_path).stem}_compressed_q{quality}.jpg"), 
                cv2.cvtColor(compressed_jpeg, cv2.COLOR_RGB2BGR))
    
    # Reconstruction
    rec_input = compressed_jpeg.astype(np.float32) / 255.0
    rec_tensor = torch.from_numpy(rec_input).permute(2, 0, 1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        rec_out = rec_cnn(rec_tensor)
    
    rec_np = rec_out.squeeze(0).permute(1, 2, 0).cpu().numpy()
    rec_np = np.clip(rec_np, 0, 1)
    
    # Save reconstructed image
    rec_uint8 = (rec_np * 255).astype(np.uint8)
    cv2.imwrite(str(output_dir / f"{Path(image_path).stem}_reconstructed_q{quality}.jpg"), 
                cv2.cvtColor(rec_uint8, cv2.COLOR_RGB2BGR))
    
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
    
    # Calculate file sizes
    original_size = os.path.getsize(image_path) / 1024  # KB
    compressed_path = output_dir / f"{Path(image_path).stem}_compressed_q{quality}.jpg"
    compressed_size = os.path.getsize(compressed_path) / 1024  # KB
    compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
    
    # Print results
    print(f"\n📊 RESULTS:")
    print(f"  Original Size       : {img.shape[1]}x{img.shape[0]} px")
    print(f"  Compressed Size     : {compressed.shape[3]}x{compressed.shape[2]} px (before upscale)")
    print(f"  Reconstructed Size  : {rec_np.shape[1]}x{rec_np.shape[0]} px")
    print(f"\n📈 QUALITY METRICS:")
    print(f"  PSNR               : {psnr_val:.2f} dB")
    print(f"  SSIM               : {ssim_val:.4f}")
    print(f"  Quality Rating     : {'GOOD ✅' if psnr_val > 28 else 'AVERAGE ⚠️' if psnr_val > 25 else 'POOR ❌'}")
    print(f"\n💾 FILE SIZE:")
    print(f"  Original File      : {original_size:.2f} KB")
    print(f"  Compressed File    : {compressed_size:.2f} KB")
    print(f"  Compression Ratio  : {compression_ratio:.2f}:1")
    print(f"  Space Saved        : {((compression_ratio-1)/compression_ratio)*100:.1f}%")
    print(f"\n📁 Saved to:")
    print(f"  Compressed: {compressed_path}")
    print(f"  Reconstructed: {output_dir / f'{Path(image_path).stem}_reconstructed_q{quality}.jpg'}")
    
    return {
        'psnr': psnr_val,
        'ssim': ssim_val,
        'compression_ratio': compression_ratio,
        'image_name': Path(image_path).name
    }

def test_all_samples():
    """Test on all available sample images"""
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                         "cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    
    # Load models
    com_cnn = ComCNN().to(device)
    rec_cnn = RecCNN().to(device)
    
    # Try to load best available model
    model_loaded = False
    for model_path in ['checkpoints/comcnn_best.pth', 
                      'checkpoints/comcnn_enhanced.pth', 
                      'checkpoints/comcnn.pth']:
        if os.path.exists(model_path):
            try:
                com_cnn.load_state_dict(torch.load(model_path, map_location=device))
                rec_cnn.load_state_dict(torch.load(model_path.replace('comcnn', 'reccnn'), map_location=device))
                print(f"✅ Loaded models from: {model_path}")
                model_loaded = True
                break
            except:
                continue
    
    if not model_loaded:
        print("❌ Could not load any models!")
        return
    
    com_cnn.eval()
    rec_cnn.eval()
    
    # Find all test images
    test_dirs = ['samples', 'data/Set5', '.']
    test_images = []
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
                test_images.extend(Path(test_dir).glob(ext))
    
    if not test_images:
        print("❌ No test images found! Please add images to 'samples/' folder")
        return
    
    # Remove duplicates
    test_images = list(set(test_images))
    print(f"\n🔍 Found {len(test_images)} test images")
    
    # Test with different quality settings
    qualities = [50, 70, 90]
    
    results = []
    for quality in qualities:
        print(f"\n\n{'#'*70}")
        print(f"{'#'*70}")
        print(f"  TESTING WITH JPEG QUALITY = {quality}")
        print(f"{'#'*70}")
        print(f"{'#'*70}")
        
        for img_path in test_images[:5]:  # Test first 5 images
            result = test_on_image(img_path, com_cnn, rec_cnn, device, quality)
            if result:
                result['quality'] = quality
                results.append(result)
    
    # Summary
    print(f"\n\n{'='*70}")
    print(f"{'='*70}")
    print(f"  SUMMARY REPORT")
    print(f"{'='*70}")
    print(f"{'='*70}\n")
    
    for quality in qualities:
        quality_results = [r for r in results if r['quality'] == quality]
        if quality_results:
            avg_psnr = sum(r['psnr'] for r in quality_results) / len(quality_results)
            avg_ssim = sum(r['ssim'] for r in quality_results) / len(quality_results)
            avg_ratio = sum(r['compression_ratio'] for r in quality_results) / len(quality_results)
            
            print(f"JPEG Quality {quality}:")
            print(f"  Average PSNR           : {avg_psnr:.2f} dB")
            print(f"  Average SSIM           : {avg_ssim:.4f}")
            print(f"  Average Compression    : {avg_ratio:.2f}:1")
            print(f"  Images Tested          : {len(quality_results)}")
            print()
    
    print("\n✅ All test results saved to 'test_results/' folder")
    print("   Compare compressed vs reconstructed images visually!")

if __name__ == "__main__":
    test_all_samples()