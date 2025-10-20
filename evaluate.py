import argparse
import os
import torch
import numpy as np
import cv2
from models.comcnn import ComCNN
from models.reccnn import RecCNN
from utils.codec import jpeg_codec, psnr_torch, ssim_numpy

def load_img(path):
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not load image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype('float32') / 255.0
    return img

def save_img(img, path):
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)

def evaluate_image(img_path, com_cnn, rec_cnn, device, quality=10):
    # Load image
    img = load_img(img_path)
    h_orig, w_orig = img.shape[:2]

    # Convert to tensor and send to device
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

    # Compress
    with torch.no_grad():
        compressed = com_cnn(img_tensor)
        compressed_np = compressed.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    # JPEG codec
    compressed_uint8 = (np.clip(compressed_np, 0, 1) * 255).astype(np.uint8)
    compressed_jpeg = jpeg_codec(compressed_uint8, quality)

    # Upsample to original size if necessary
    if compressed_jpeg.shape[:2] != (h_orig, w_orig):
        compressed_up = cv2.resize(compressed_jpeg, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
    else:
        compressed_up = compressed_jpeg

    # Convert to tensor and run RecCNN
    rec_input = compressed_up.astype(np.float32) / 255.0
    rec_tensor = torch.from_numpy(rec_input).permute(2, 0, 1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        rec_out = rec_cnn(rec_tensor)
    
    rec_np = rec_out.squeeze(0).permute(1, 2, 0).cpu().numpy()
    rec_np = np.clip(rec_np, 0, 1)

    # Compute metrics
    psnr_val = psnr_torch(
        torch.from_numpy(rec_np).permute(2, 0, 1).unsqueeze(0).to(device),
        torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
    )
    ssim_val = ssim_numpy(rec_np, img)

    return rec_np, psnr_val, ssim_val

def main():
    parser = argparse.ArgumentParser(description='Evaluate CNN Compression Models')
    parser.add_argument('--data_dir', type=str, default='./data/Set5', help='Test images directory')
    parser.add_argument('--ckpt_com', type=str, default='./checkpoints/comcnn.pth', help='ComCNN checkpoint')
    parser.add_argument('--ckpt_rec', type=str, default='./checkpoints/reccnn.pth', help='RecCNN checkpoint')
    parser.add_argument('--out', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--quality', type=int, default=10, help='JPEG quality factor')
    args = parser.parse_args()

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.out, exist_ok=True)

    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' does not exist.")
        return

    # Load models
    com_cnn = ComCNN().to(device)
    rec_cnn = RecCNN().to(device)
    
    try:
        com_cnn.load_state_dict(torch.load(args.ckpt_com, map_location=device))
        rec_cnn.load_state_dict(torch.load(args.ckpt_rec, map_location=device))
        print("✓ Models loaded successfully")
    except FileNotFoundError as e:
        print(f"Error: Model checkpoint not found: {e}")
        return
    
    com_cnn.eval()
    rec_cnn.eval()

    # Find images
    img_files = [f for f in os.listdir(args.data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not img_files:
        print(f"No images found in {args.data_dir}")
        return

    print(f"Found {len(img_files)} images to process:")
    for img_file in img_files:
        print(f"  - {img_file}")

    # Process images
    total_psnr = 0
    total_ssim = 0
    processed_count = 0
    
    for img_file in img_files:
        img_path = os.path.join(args.data_dir, img_file)
        try:
            rec_img, psnr_val, ssim_val = evaluate_image(img_path, com_cnn, rec_cnn, device, args.quality)
            out_path = os.path.join(args.out, img_file)
            save_img(rec_img, out_path)
            print(f"{img_file}: PSNR={psnr_val:.2f} dB, SSIM={ssim_val:.4f}")
            
            total_psnr += psnr_val
            total_ssim += ssim_val
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {img_file}: {e}")

    # Print summary
    if processed_count > 0:
        avg_psnr = total_psnr / processed_count
        avg_ssim = total_ssim / processed_count
        print(f"\n=== SUMMARY (Quality={args.quality}) ===")
        print(f"Average PSNR: {avg_psnr:.2f} dB")
        print(f"Average SSIM: {avg_ssim:.4f}")
        print(f"Processed {processed_count} images")
        print(f"Results saved to {args.out}")

if __name__ == "__main__":
    main()