import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import io
import time

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
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        com_cnn = ComCNN().to(device)
        rec_cnn = RecCNN().to(device)
        
        # Try enhanced models first, then regular
        model_attempts = [
            ('checkpoints/comcnn_enhanced.pth', 'checkpoints/reccnn_enhanced.pth', 'enhanced'),
            ('checkpoints/comcnn.pth', 'checkpoints/reccnn.pth', 'standard'),
        ]
        
        for com_path, rec_path, model_type in model_attempts:
            try:
                com_cnn.load_state_dict(torch.load(com_path, map_location=device))
                rec_cnn.load_state_dict(torch.load(rec_path, map_location=device))
                st.success(f"✓ Loaded {model_type} models")
                com_cnn.eval()
                rec_cnn.eval()
                return com_cnn, rec_cnn
            except:
                continue
        
        st.error("❌ Could not load any models")
        return None, None
        
    except Exception as e:
        st.error(f"Error: {e}")
        return None, None

def process_image(img, com_cnn, rec_cnn):
    """Process image through compression pipeline with fixed quality=100"""
    try:
        # Convert PIL to numpy
        img_np = np.array(img, dtype=np.float32) / 255.0
        h_orig, w_orig = img_np.shape[:2]
        
        # Ensure divisible by 2 for ComCNN
        if h_orig % 2 != 0 or w_orig % 2 != 0:
            h_new = h_orig - (h_orig % 2)
            w_new = w_orig - (w_orig % 2)
            img_np = img_np[:h_new, :w_new, :]

        # Compression stage
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)
        
        with torch.no_grad():
            compressed_tensor = com_cnn(img_tensor)
        
        compressed_np = compressed_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        compressed_np = np.clip(compressed_np, 0, 1)
        
        # JPEG compression with FIXED quality=100
        compressed_uint8 = (compressed_np * 255).astype(np.uint8)
        compressed_jpeg = jpeg_codec(compressed_uint8, 100)  # Fixed at maximum quality
        
        # Upsample if needed
        if compressed_jpeg.shape[:2] != (h_orig, w_orig):
            compressed_up = cv2.resize(compressed_jpeg, (w_orig, h_orig), interpolation=cv2.INTER_CUBIC)
        else:
            compressed_up = compressed_jpeg.copy()
        
        compressed_img = Image.fromarray(compressed_up)
        
        # Reconstruction stage
        rec_input_np = compressed_up.astype(np.float32) / 255.0
        rec_tensor = torch.from_numpy(rec_input_np).permute(2, 0, 1).unsqueeze(0).to(device)
        
        with torch.no_grad():
            rec_out_tensor = rec_cnn(rec_tensor)
        
        rec_np = rec_out_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        rec_np = np.clip(rec_np, 0, 1)
        rec_uint8 = (rec_np * 255).astype(np.uint8)
        rec_img = Image.fromarray(rec_uint8)
        
        # Calculate metrics
        psnr_val = psnr_torch(
            torch.from_numpy(rec_np).permute(2, 0, 1).unsqueeze(0).to(device),
            torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)
        )
        ssim_val = ssim_numpy(rec_np, img_np)
        
        return compressed_img, rec_img, psnr_val, ssim_val
        
    except Exception as e:
        raise Exception(f"Image processing error: {str(e)}")

# Streamlit UI
st.set_page_config(
    page_title="CNN Image Compression", 
    layout="wide", 
    page_icon="🖼️"
)

st.title("🖼️ CNN Image Compression Demo")
st.markdown("Compress images using ComCNN + JPEG (Quality=100) and reconstruct with RecCNN")

# Sidebar
with st.sidebar:
    st.header("Settings")
    
    # Fixed quality - no slider
    quality = 50  # Fixed at maximum quality
    st.info(f"**JPEG Quality: {quality}** (Fixed)")
    st.write("**Mode**: Maximum Quality Testing")
    
    st.header("Model Info")
    st.info(f"Device: {device}")
    st.write("**ComCNN**: 3-layer compression network")
    st.write("**RecCNN**: 20-layer reconstruction network")

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
        compressed_img, rec_img, psnr_val, ssim_val = process_image(img, com_cnn, rec_cnn)  # Removed quality parameter
        processing_time = time.time() - start_time
    
    # Display results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Original Image")
        st.image(img, use_container_width=True)
        st.caption(f"Size: {img.size[0]}x{img.size[1]}")
    
    with col2:
        st.subheader("Compressed")
        st.image(compressed_img, use_container_width=True)
        st.caption("After ComCNN + JPEG (Quality=100)")
    
    with col3:
        st.subheader("Reconstructed")
        st.image(rec_img, use_container_width=True)
        st.caption("After RecCNN")
    
    # Metrics
    st.subheader("Quality Metrics")
    col_psnr, col_ssim, col_time = st.columns(3)
    
    with col_psnr:
        st.metric("PSNR", f"{psnr_val:.2f} dB", 
                 help="Peak Signal-to-Noise Ratio (higher is better)")
    
    with col_ssim:
        st.metric("SSIM", f"{ssim_val:.4f}",
                 help="Structural Similarity Index (higher is better)")
    
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
        compressed_img.save(buf_comp, format="JPEG", quality=95)
        st.download_button(
            label="📥 Download Compressed Image",
            data=buf_comp.getvalue(),
            file_name=f"compressed_{uploaded_file.name}",
            mime="image/jpeg"
        )

else:
    st.info("👆 Upload an image to start compression")
    st.write("**How it works:**")
    st.write("1. **ComCNN** compresses the image (learned compression)")
    st.write("2. **JPEG** compresses with maximum quality (100)")
    st.write("3. **RecCNN** reconstructs the high-quality image")
    st.write("4. **Metrics** evaluate the reconstruction quality")
    st.warning("**Note**: JPEG Quality is fixed at 100 for maximum quality testing")

# Footer
st.markdown("---")
st.caption("CNN-Based Image Compression Framework | Based on the paper 'An End-to-End Compression Framework Based on Convolutional Neural Networks'")