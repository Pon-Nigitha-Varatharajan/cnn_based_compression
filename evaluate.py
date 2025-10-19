import argparse
import os
import torch
import numpy as np
import cv2
from models.comcnn import ComCNN
from models.reccnn import RecCNN
from utils.codec import jpeg_codec, bicubic_upsample, psnr_torch, ssim_numpy

def load_img(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype('float32') / 255.0
    return img

def save_img(img, path):
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)


def evaluate_image(img_path, com_cnn, rec_cnn, device, quality=10):
    # Load image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # Convert to tensor and send to device
    img_tensor = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(device)

    # Compress
    with torch.no_grad():
        compressed = com_cnn(img_tensor)
        compressed_np = compressed.squeeze(0).permute(1,2,0).cpu().numpy()
    
    # JPEG codec
    compressed_jpeg = jpeg_codec((compressed_np * 255).astype(np.uint8), quality)

    # Upsample to original size if necessary
    if compressed_jpeg.shape != img.shape:
        compressed_up = cv2.resize(compressed_jpeg, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    else:
        compressed_up = compressed_jpeg

    # Convert to tensor and run RecCNN
    rec_tensor = torch.from_numpy(compressed_up.astype(np.float32)).permute(2,0,1).unsqueeze(0).to(device)
    with torch.no_grad():
        rec_out = rec_cnn(rec_tensor)
    rec_np = rec_out.squeeze(0).permute(1,2,0).cpu().numpy()

    # Ensure final size matches original
    if rec_np.shape != img.shape:
        rec_np = cv2.resize(rec_np, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

    # Compute metrics
    psnr_val = psnr_torch(torch.from_numpy(rec_np), torch.from_numpy(img))
    ssim_val = ssim_numpy(rec_np, img)

    return rec_np, psnr_val, ssim_val

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Folder of images to evaluate')
    parser.add_argument('--ckpt_com', type=str, default='./checkpoints/com_cnn.pth')
    parser.add_argument('--ckpt_rec', type=str, default='./checkpoints/rec_cnn.pth')
    parser.add_argument('--out', type=str, default='./outputs', help='Output folder')
    parser.add_argument('--quality', type=int, default=10)
    args = parser.parse_args()

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    os.makedirs(args.out, exist_ok=True)

    # Load models
    com_cnn = ComCNN().to(device)
    rec_cnn = RecCNN().to(device)
    com_cnn.load_state_dict(torch.load(args.ckpt_com, map_location=device))
    rec_cnn.load_state_dict(torch.load(args.ckpt_rec, map_location=device))
    com_cnn.eval()
    rec_cnn.eval()

    # Loop over all images in the folder
    img_files = [f for f in os.listdir(args.data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not img_files:
        print(f"No images found in {args.data_dir}")
        return

    for img_file in img_files:
        img_path = os.path.join(args.data_dir, img_file)
        rec_img, psnr_val, ssim_val = evaluate_image(img_path, com_cnn, rec_cnn, device, args.quality)
        out_path = os.path.join(args.out, img_file)
        save_img(rec_img, out_path)
        print(f"{img_file}: PSNR={psnr_val:.2f}, SSIM={ssim_val:.4f} -> saved to {out_path}")

if __name__ == "__main__":
    main()