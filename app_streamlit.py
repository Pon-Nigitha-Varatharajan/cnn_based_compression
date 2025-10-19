import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from models.comcnn import ComCNN
from models.reccnn import RecCNN
from utils.codec import jpeg_codec, psnr_torch, ssim_numpy

# --- DEVICE ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# --- LOAD MODELS ---
@st.cache_resource
def load_models(ckpt_com, ckpt_rec):
    com_cnn = ComCNN().to(device)
    rec_cnn = RecCNN().to(device)
    com_cnn.load_state_dict(torch.load(ckpt_com, map_location=device))
    rec_cnn.load_state_dict(torch.load(ckpt_rec, map_location=device))
    com_cnn.eval()
    rec_cnn.eval()
    return com_cnn, rec_cnn

# --- IMAGE PROCESSING ---
def process_image(img, com_cnn, rec_cnn, quality):
    img_np = np.array(img).astype(np.float32) / 255.0

    img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).to(device)

    # --- Compression ---
    with torch.no_grad():
        compressed_tensor = com_cnn(img_tensor)

    compressed_np = compressed_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    compressed_np = np.clip(compressed_np, 0, 1)

    compressed_jpeg = jpeg_codec((compressed_np * 255).astype(np.uint8), quality)

    # Upsample
    if compressed_jpeg.shape[:2] != img_np.shape[:2]:
        compressed_up = cv2.resize(
            compressed_jpeg, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_CUBIC
        )
    else:
        compressed_up = compressed_jpeg.copy()

    if compressed_up.ndim == 2:
        compressed_up = cv2.cvtColor(compressed_up, cv2.COLOR_GRAY2RGB)
    elif compressed_up.shape[2] != 3:
        compressed_up = np.repeat(compressed_up, 3, axis=2)
    compressed_up = np.clip(compressed_up, 0, 255).astype(np.uint8)
    compressed_img = Image.fromarray(compressed_up)

    # --- Reconstruction ---
    rec_tensor = torch.from_numpy(compressed_up.astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0).to(device)
    with torch.no_grad():
        rec_out = rec_cnn(rec_tensor)

    rec_np = rec_out.squeeze(0).permute(1,2,0).cpu().numpy()
    rec_np = np.clip(rec_np, 0, 1)

    if rec_np.shape[:2] != img_np.shape[:2]:
        rec_np = cv2.resize(rec_np, (img_np.shape[1], img_np.shape[0]), interpolation=cv2.INTER_CUBIC)

    rec_np = np.clip(rec_np*255, 0, 255).astype(np.uint8)
    if rec_np.ndim == 2:
        rec_np = cv2.cvtColor(rec_np, cv2.COLOR_GRAY2RGB)
    elif rec_np.shape[2] != 3:
        rec_np = np.repeat(rec_np, 3, axis=2)
    rec_img = Image.fromarray(rec_np)

    # Metrics
    psnr_val = psnr_torch(torch.from_numpy(rec_np.astype(np.float32)/255.0),
                          torch.from_numpy(img_np.astype(np.float32)))
    ssim_val = ssim_numpy(rec_np, (img_np*255).astype(np.uint8))

    return compressed_img, rec_img, psnr_val, ssim_val

# --- STREAMLIT UI ---
st.set_page_config(page_title="CNN Image Compression", layout="wide", page_icon="🖼️")

st.markdown(
    """
    <style>
    body {background-color:#0E1117; color:white;}
    .stButton>button {background-color:#1E1E1E; color:white;}
    .stSlider>div>div>input {color:white;}
    .stFileUploader>div>div>label {color:white;}
    .stImage>img {border-radius:15px; box-shadow: 5px 5px 15px #000000;}
    .stMarkdown {color:white;}
    </style>
    """, unsafe_allow_html=True
)

st.title("🖼️ CNN Image Compression Demo")

# --- Upload image ---
uploaded_file = st.file_uploader("Upload an image", type=["png","jpg","jpeg"])
quality = st.slider("JPEG Quality", 1, 100, 10)

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")

    # Load models
    com_cnn, rec_cnn = load_models("checkpoints/comcnn.pth", "checkpoints/reccnn.pth")

    # Process
    compressed_img, rec_img, psnr_val, ssim_val = process_image(img, com_cnn, rec_cnn, quality)

    # Display in cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Original Image**")
        st.image(img, use_container_width=True)
    with col2:
        st.markdown("**Compressed Image**")
        st.image(compressed_img, use_container_width=True)
    with col3:
        st.markdown("**Reconstructed Image**")
        st.image(rec_img, use_container_width=True)

    # Metrics
    st.markdown(f"**PSNR:** {psnr_val:.2f} dB  |  **SSIM:** {ssim_val:.4f}")

    # Save output
    if st.button("Save Reconstructed Image"):
        rec_img.save(f"outputs/reconstructed_{uploaded_file.name}")
        st.success("Reconstructed image saved to outputs/")