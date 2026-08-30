import streamlit as st
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
from io import BytesIO
import importlib.util
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


# The two detectors are separate projects under server/, not modules next to this
# file. Both expose the same `load_model()` / `predict_image(pil) -> float` API,
# so everything below is written against that contract rather than either model.
def _model_dir(name):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "server", name))


MODELS = {
    "model_01 — CNN + Transformer, trained end to end": {
        "key": "model_01",
        "dir": _model_dir("model_01"),
        "checkpoint": "server/model_01/checkpoints/best.pt",
        "env_var": "AIGC_CHECKPOINT",
        "requirements": "server/requirements.txt",
    },
    "model_02 — frozen DINOv2 + CLIP + FFT → small classifier": {
        "key": "model_02",
        "dir": _model_dir("model_02"),
        "checkpoint": "server/model_02/checkpoints/best.pt",
        "env_var": "AIGC_MODEL02_CHECKPOINT",
        "requirements": "server/model_02/requirements.txt",
    },
    "model_03 — region-aware map → specialist routing → fusion": {
        "key": "model_03",
        "dir": _model_dir("model_03"),
        # No weights of its own: it scores patches with an existing detector,
        # by default a public one pulled from the Hugging Face Hub and cached
        # there. So there is no checkpoint path to point at -- which model it
        # uses is a config/env choice.
        "checkpoint": "no local checkpoint; downloads Organika/sdxl-detector on first use",
        "env_var": "AIGC_MODEL03_BACKEND",
        "requirements": "server/model_03/requirements.txt",
    },
}


@st.cache_resource(show_spinner="Loading detector...")
def _load_backend(model_key, model_dir):
    """Import one model's inference backend, or raise with a usable message.

    Cached per model, so the checkpoint (and for model_02 the frozen DINOv2/CLIP
    backbones) is loaded once per session rather than per upload, and switching
    models back and forth does not reload either one.
    """
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)

    # Both projects ship a top-level `infer.py`. A plain `import infer` would bind
    # whichever model was selected first and then hand that same module back for
    # the other one -- scoring model_02 uploads with model_01's weights, silently.
    # Loading each from its own file path under its own module name keeps them
    # separate in sys.modules.
    mod_name = f"_infer_{model_key}"
    module = sys.modules.get(mod_name)
    if module is None:
        spec = importlib.util.spec_from_file_location(
            mod_name, os.path.join(model_dir, "infer.py")
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            # Don't leave a half-initialised module behind for the next attempt.
            sys.modules.pop(mod_name, None)
            raise

    module.load_model()  # fail here, at startup, rather than on the first upload
    return module


def run_inference(image, model):
    """Score `image` with the selected model.

    Returns `(score, report)`, where `score` is P(AI-generated) in [0, 1] and
    `report` is a region-aware analysis when the backend produces one (model_03)
    or None otherwise. A backend exposing `analyze_image` is called through it
    rather than through `predict_image`, so the pipeline runs once and the UI
    gets both the number and the map; calling both entry points would re-run the
    whole analysis for the same upload.

    There is deliberately no mock fallback. An earlier version scored images with
    `((mean_pixel * 7) % 100) / 100` whenever the import failed, and swallowed the
    reason -- which rendered as a confident "AI-Generated, 87.3%" verdict that had
    never touched the model. A backend that is not wired up must look broken, not
    look like a working demo.
    """
    backend = _load_backend(model["key"], model["dir"])
    analyze = getattr(backend, "analyze_image", None)
    if analyze is not None:
        report = analyze(image)
        return float(report.score), report
    return float(backend.predict_image(image)), None


# --- SIDEBAR CONTROLS ---

st.sidebar.header("Detector")
model_label = st.sidebar.radio(
    "Model", list(MODELS.keys()), index=0,
    help="Both are trained on the same data and report the same score: "
         "P(AI-generated). Switching re-scores the current upload.",
)
model = MODELS[model_label]

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
    try:
        score, report = run_inference(transformed_img, model)
    except Exception as exc:
        st.error(
            f"**{model['key']} is not available, so no score can be shown.**\n\n"
            f"`{type(exc).__name__}: {exc}`\n\n"
            f"Expected the model at `{model['dir']}` and a checkpoint at "
            f"`{model['checkpoint']}` (or set `${model['env_var']}`). "
            f"Install its deps with `pip install -r {model['requirements']}`."
        )
        st.stop()
    
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        st.image(transformed_img, caption="Transformed Input Image", width="stretch")
        
    with col2:
        st.subheader("Model Verdict")
        st.caption(f"Scored by **{model['key']}**")
        
        # Real-time score display
        if score > 0.5:
            st.error(f"⚠️ **AI-Generated Image**\n\nConfidence: **{score * 100:.1f}%**")
        else:
            st.success(f"✅ **Authentic Camera Image**\n\nConfidence: **{(1 - score) * 100:.1f}%**")
            
        # Real-time progress meter
        st.write("AI Probability Score:")
        st.progress(float(score))

        # Both models saturate hard -- a rounded "100.0%" above can be anything from
        # 0.9995 to exactly 1.0, which hides the very movement this bench exists to
        # show. The unrounded score makes the difference visible.
        st.caption(f"raw P(AI-generated) = {score:.6g}")

    # A region-aware backend (model_03) knows more than one number: where the
    # evidence is, what kind it is, and how much of the frame it could not call.
    # Showing only the score would throw away the part that distinguishes it.
    if report is not None:
        st.divider()
        st.subheader("Where the evidence is")
        st.write(f"**Verdict:** `{report.verdict.verdict}` · confidence {report.verdict.confidence:.2f}")
        st.write(report.verdict.explanation)

        backend = _load_backend(model["key"], model["dir"])
        map_col, region_col = st.columns(2)
        with map_col:
            st.image(
                backend.render_overlay(report),
                caption="AI-likelihood map (grey = uncertain, warm = likely AI)",
                width="stretch",
            )
        with region_col:
            st.image(
                backend.render_regions(report),
                caption="Suspicious regions, with the specialist each was routed to",
                width="stretch",
            )

        summary = report.amap.summary()
        cols = st.columns(3)
        cols[0].metric("Likely AI", f"{summary['frac_likely_ai'] * 100:.0f}%")
        cols[1].metric("Uncertain", f"{summary['frac_uncertain'] * 100:.0f}%")
        cols[2].metric("Likely not AI", f"{summary['frac_likely_non_ai'] * 100:.0f}%")

        for finding in report.verdict.findings:
            with st.expander(
                f"Region #{finding.region.region_id} — {finding.label} "
                f"({finding.probability:.2f}, via {finding.route.primary})"
            ):
                st.caption(f"Routed here because {finding.route.reason}.")
                for line in finding.evidence:
                    st.write(f"- {line}")

        for note in report.notes:
            st.info(note)
