import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import io
import time
import os

# Set page config MUST be at the very top
st.set_page_config(
    page_title="CNN Image Compression", 
    layout="wide", 
    page_icon="🖼️"
)

# Import models
try:
    from models.comcnn import ComCNN
    from models.reccnn import RecCNN
    from utils.codec import jpeg_codec, psnr_torch, ssim_numpy
except ImportError as e:
    st.error(f"Import error: {e}. Please check your file structure.")
    st.stop()

# Device configuration
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_models():
    """Load the best available models"""
    try:
        com_cnn = ComCNN().to(device)
        rec_cnn = RecCNN().to(device)
        
        # Try to load models
        model_attempts = [
            ('checkpoints/comcnn.pth', 'checkpoints/reccnn.pth'),
            ('checkpoints/comcnn_enhanced.pth', 'checkpoints/reccnn_enhanced.pth'),
        ]
        
        for com_path, rec_path in model_attempts:
            try:
                if os.path.exists(com_path) and os.path.exists(rec_path):
                    com_cnn.load_state_dict(torch.load(com_path, map_location=device))
                    rec_cnn.load_state_dict(torch.load(rec_path, map_location=device))
                    st.success(f"✓ Loaded models from {com_path}")
                    com_cnn.eval()
                    rec_cnn.eval()
                    return com_cnn, rec_cnn
            except Exception as e:
                st.warning(f"Could not load {com_path}: {e}")
                continue
        
        st.error("❌ Could not load any models. Please train models first.")
        return None, None
        
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

def calculate_compression_ratio(original_img, compressed_img, quality=50):
    """Calculate accurate compression ratio"""
    # Original size
    original_buffer = io.BytesIO()
    original_img.save(original_buffer, format="PNG")
    original_size = len(original_buffer.getvalue())
    
    # Compressed size (using same quality as processing)
    compressed_buffer = io.BytesIO()
    compressed_img.save(compressed_buffer, format="JPEG", quality=quality)
    compressed_size = len(compressed_buffer.getvalue())
    
    if compressed_size > 0:
        return original_size / compressed_size
    return 1.0

def process_image(img, com_cnn, rec_cnn, quality=50):
    """Process image through compression pipeline"""
    try:
        # Convert PIL to numpy
        img_np = np.array(img, dtype=np.float32) / 255.0
        h_orig, w_orig = img_np.shape[:2]
        
        # Store original for metrics
        original_np = img_np.copy()

        # Ensure divisible by 2 for ComCNN (due to stride=2)
        if h_orig % 2 != 0 or w_orig % 2 != 0:
            h_new = h_orig - (h_orig % 2)
            w_new = w_orig - (w_orig % 2)
            img_np = img_np[:h_new, :w_new, :]
            h_orig, w_orig = h_new, w_new

        # Compression stage
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)
        
        with torch.no_grad():
            compressed_tensor = com_cnn(img_tensor)  # This downscales by 2
        
        # Scale compressed output back to original size for JPEG
        compressed_up = torch.nn.functional.interpolate(
            compressed_tensor, 
            size=(h_orig, w_orig), 
            mode='bilinear', 
            align_corners=False
        )
        
        compressed_np = compressed_up.squeeze(0).permute(1, 2, 0).cpu().numpy()
        compressed_np = np.clip(compressed_np, 0, 1)
        
        # JPEG compression with specified quality
        compressed_uint8 = (compressed_np * 255).astype(np.uint8)
        compressed_jpeg = jpeg_codec(compressed_uint8, quality)
        
        compressed_img = Image.fromarray(compressed_jpeg)
        
        # Reconstruction stage
        rec_input_np = compressed_jpeg.astype(np.float32) / 255.0
        rec_tensor = torch.from_numpy(rec_input_np).permute(2, 0, 1).unsqueeze(0).to(device)
        
        with torch.no_grad():
            rec_out_tensor = rec_cnn(rec_tensor)
        
        rec_np = rec_out_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        rec_np = np.clip(rec_np, 0, 1)
        rec_uint8 = (rec_np * 255).astype(np.uint8)
        rec_img = Image.fromarray(rec_uint8)
        
        # Calculate metrics - ensure same dimensions
        min_h = min(rec_np.shape[0], img_np.shape[0])
        min_w = min(rec_np.shape[1], img_np.shape[1])
        
        rec_np_crop = rec_np[:min_h, :min_w]
        img_np_crop = img_np[:min_h, :min_w]
        
        psnr_val = psnr_torch(
            torch.from_numpy(rec_np_crop).permute(2, 0, 1).unsqueeze(0).to(device),
            torch.from_numpy(img_np_crop).permute(2, 0, 1).unsqueeze(0).to(device)
        )
        ssim_val = ssim_numpy(rec_np_crop, img_np_crop)
        
        # Calculate compression ratio
        compression_ratio = calculate_compression_ratio(img, compressed_img, quality)
        
        return compressed_img, rec_img, psnr_val, ssim_val, compression_ratio
        
    except Exception as e:
        raise Exception(f"Image processing error: {str(e)}")

# Streamlit UI
st.title("🖼️ CNN Image Compression Demo")
st.markdown("Compress images using ComCNN + JPEG and reconstruct with RecCNN")

# Sidebar
with st.sidebar:
    st.header("Settings")
    
    # Quality slider
    quality = st.slider("JPEG Quality", min_value=10, max_value=100, value=50, 
                       help="Lower quality = higher compression")
    
    st.header("Model Info")
    st.info(f"Device: {device}")
    st.write("**ComCNN**: 3-layer compression network (downscales by 2)")
    st.write("**RecCNN**: 20-layer reconstruction network")
    
    st.header("Pipeline")
    st.write("1. **ComCNN**: Learned compression + downscaling")
    st.write("2. **JPEG**: Traditional compression")
    st.write("3. **RecCNN**: Learned reconstruction")

# Main content
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Load image
    img = Image.open(uploaded_file).convert("RGB")
    
    # Load models
    with st.spinner("Loading models..."):
        com_cnn, rec_cnn = load_models()
    
    if com_cnn is None or rec_cnn is None:
        st.stop()
    
    # Process image
    with st.spinner("Processing image..."):
        start_time = time.time()
        try:
            compressed_img, rec_img, psnr_val, ssim_val, compression_ratio = process_image(
                img, com_cnn, rec_cnn, quality
            )
            processing_time = time.time() - start_time
        except Exception as e:
            st.error(f"Processing failed: {e}")
            st.stop()
    
    # Display results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Original Image")
        st.image(img, use_container_width=True)
        st.caption(f"Size: {img.size[0]}x{img.size[1]}")
    
    with col2:
        st.subheader("Compressed")
        st.image(compressed_img, use_container_width=True)
        st.caption(f"After ComCNN + JPEG (Quality={quality})")
    
    with col3:
        st.subheader("Reconstructed")
        st.image(rec_img, use_container_width=True)
        st.caption("After RecCNN")
    
    # Metrics
    st.subheader("Quality Metrics")
    col_psnr, col_ssim, col_ratio, col_time = st.columns(4)
    
    with col_psnr:
        st.metric("PSNR", f"{psnr_val:.2f} dB", 
                 help="Peak Signal-to-Noise Ratio (higher is better)")
        st.write("**Quality:** " + ("GOOD" if psnr_val > 28 else "AVERAGE" if psnr_val > 25 else "POOR"))
    
    with col_ssim:
        st.metric("SSIM", f"{ssim_val:.4f}",
                 help="Structural Similarity Index (higher is better)")
    
    with col_ratio:
        st.metric("Compression Ratio", f"{compression_ratio:.2f}:1",
                 help="Original size : Compressed size")
    
    with col_time:
        st.metric("Processing Time", f"{processing_time:.2f}s")
    
    # Download buttons
    st.subheader("Download Results")
    col_dl1, col_dl2 = st.columns(2)
    
    with col_dl1:
        buf_rec = io.BytesIO()
        rec_img.save(buf_rec, format="PNG")
        st.download_button(
            label="📥 Download Reconstructed Image",
            data=buf_rec.getvalue(),
            file_name=f"reconstructed_{uploaded_file.name}",
            mime="image/png"
        )
    
    with col_dl2:
        buf_comp = io.BytesIO()
        compressed_img.save(buf_comp, format="JPEG", quality=quality)
        st.download_button(
            label="📥 Download Compressed Image",
            data=buf_comp.getvalue(),
            file_name=f"compressed_{uploaded_file.name}",
            mime="image/jpeg"
        )

else:
    st.info("👆 Upload an image to start compression")
    st.write("**Expected Results:**")
    st.write("- **Compressed image**: Should show visible but minor quality loss")
    st.write("- **Reconstructed image**: Should be similar to original with PSNR > 25 dB")
    st.write("- **Compression ratio**: Should be reasonable (2:1 to 20:1 depending on quality)")

# Footer
st.markdown("---")
st.caption("CNN-Based Image Compression Framework | Based on the paper 'An End-to-End Compression Framework Based on Convolutional Neural Networks'")