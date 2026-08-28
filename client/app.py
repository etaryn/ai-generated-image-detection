import streamlit as st
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
from io import BytesIO
import sys
import os

# Set page config
st.set_page_config(page_title="AIGC Detection Bench", layout="wide")
st.title("🛡️ Robust AIGC Deepfake Detector")

# --- TRANSFORMATION FUNCTIONS ---

def apply_transformations(img, blur, jpeg_q, crop_percent, brightness, contrast, color, noise_level):
    """Applies user-selected transformations to a PIL Image."""
    transformed = img.copy().convert("RGB")
    
    # 1. Crop
    if crop_percent > 0:
        w, h = transformed.size
        crop_px = int((crop_percent / 100.0) * min(w, h) / 2)
        transformed = transformed.crop((crop_px, crop_px, w - crop_px, h - crop_px))
    
    # 2. Blur
    if blur > 0:
        transformed = transformed.filter(ImageFilter.GaussianBlur(radius=blur))
        
    # 3. Color Jitter (Brightness, Contrast, Saturation)
    if brightness != 1.0:
        transformed = ImageEnhance.Brightness(transformed).enhance(brightness)
    if contrast != 1.0:
        transformed = ImageEnhance.Contrast(transformed).enhance(contrast)
    if color != 1.0:
        transformed = ImageEnhance.Color(transformed).enhance(color)
        
    # 4. Gaussian Noise
    if noise_level > 0:
        arr = np.array(transformed, dtype=np.float32)
        noise = np.random.normal(0, noise_level, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        transformed = Image.fromarray(arr)
        
    # 5. JPEG Compression
    if jpeg_q < 100:
        buf = BytesIO()
        transformed.save(buf, format="JPEG", quality=jpeg_q)
        buf.seek(0)
        transformed = Image.open(buf).convert("RGB")
        
    return transformed


def run_inference(image):
    """
    Attempts to import and run your backend inference from infer.py.
    Falls back to a mock score if infer.py or PyTorch model is not ready.
    """
    try:
        # Add parent folder to path to find infer.py
        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        if parent_dir not in sys.path:
            sys.path.append(parent_dir)
            
        import infer
        if hasattr(infer, "predict_image"):
            return infer.predict_image(image)
    except Exception:
        pass
    
    # Mock fallback for UI development
    # Generates a pseudo-deterministic score based on image pixels for realistic real-time feedback
    arr = np.array(image, dtype=np.float32)
    mean_val = float(np.mean(arr))
    score = ((mean_val * 7) % 100) / 100.0
    return score


# --- SIDEBAR CONTROLS ---

st.sidebar.header("Real-World Transformations")

# 1. JPEG & Blur
jpeg_q = st.sidebar.slider("JPEG Compression Quality", 10, 100, 100)
blur_sigma = st.sidebar.slider("Gaussian Blur (σ)", 0.0, 5.0, 0.0, step=0.1)

# 2. Crop
crop_pct = st.sidebar.slider("Crop Border (%)", 0, 30, 0)

# 3. Color Jitter
st.sidebar.subheader("Color Jitter")
bright = st.sidebar.slider("Brightness", 0.5, 1.5, 1.0, step=0.05)
contrast = st.sidebar.slider("Contrast", 0.5, 1.5, 1.0, step=0.05)
color_sat = st.sidebar.slider("Saturation", 0.0, 2.0, 1.0, step=0.1)

# 4. Noise
st.sidebar.subheader("Additive Noise")
noise_val = st.sidebar.slider("Noise Intensity", 0, 50, 0)


# --- MAIN INTERFACE ---

uploaded = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded:
    raw_img = Image.open(uploaded)
    
    # Apply transformations live
    transformed_img = apply_transformations(
        raw_img, blur_sigma, jpeg_q, crop_pct, bright, contrast, color_sat, noise_val
    )
    
    # Run real-time inference on transformed image
    score = run_inference(transformed_img)
    
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.image(transformed_img, caption="Transformed Input Image", width="stretch")
        
    with col2:
        st.subheader("Model Verdict")
        
        # Real-time score display
        if score > 0.5:
            st.error(f"⚠️ **AI-Generated Image**\n\nConfidence: **{score * 100:.1f}%**")
        else:
            st.success(f"✅ **Authentic Camera Image**\n\nConfidence: **{(1 - score) * 100:.1f}%**")
            
        # Real-time progress meter
        st.write("AI Probability Score:")
        st.progress(float(score))