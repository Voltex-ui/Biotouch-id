import os
import json
import joblib
import numpy as np
import cv2
import streamlit as st
import pandas as pd
from PIL import Image, UnidentifiedImageError
from skimage.feature import hog, local_binary_pattern

MODEL_DIR = "saved_models"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
METRICS_JSON_PATH = os.path.join(MODEL_DIR, "metrics.json")
BEST_MODEL_INFO_PATH = os.path.join(MODEL_DIR, "best_model_info.json")
GRAPH_PATH = os.path.join(MODEL_DIR, "graph.png")

LBP_RADIUS = 1
LBP_POINTS = 8 * LBP_RADIUS
LBP_BINS = LBP_POINTS + 2

st.set_page_config(
    page_title="Fingerprint Blood Group Detection",
    page_icon="🧬",
    layout="wide"
)

st.markdown("""
    <style>
    .main-title {
        font-size: 34px;
        font-weight: 700;
        color: #1f4e79;
        margin-bottom: 5px;
    }
    .sub-title {
        font-size: 17px;
        color: #555;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #f7f9fc;
        padding: 18px;
        border-radius: 14px;
        border: 1px solid #e6ecf2;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        text-align: center;
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #0b6e4f;
    }
    .metric-label {
        font-size: 14px;
        color: #555;
        margin-top: 4px;
    }
    .section-title {
        font-size: 24px;
        font-weight: 600;
        margin-top: 10px;
        margin-bottom: 10px;
        color: #1f2937;
    }
    .prediction-box {
        background-color: #eef7ff;
        border: 1px solid #cfe8ff;
        border-radius: 14px;
        padding: 18px;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    .prediction-label {
        font-size: 16px;
        color: #444;
    }
    .prediction-value {
        font-size: 30px;
        font-weight: 700;
        color: #0a58ca;
    }
    </style>
""", unsafe_allow_html=True)

# ================= LOAD =================
@st.cache_resource
def load_all():
    required_files = [
        BEST_MODEL_PATH,
        LABEL_ENCODER_PATH,
        METRICS_JSON_PATH,
        BEST_MODEL_INFO_PATH,
        GRAPH_PATH
    ]

    missing = [path for path in required_files if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError("Saved model files not found. Run model.py first.")

    model = joblib.load(BEST_MODEL_PATH)
    le = joblib.load(LABEL_ENCODER_PATH)

    with open(METRICS_JSON_PATH, "r") as f:
        metrics = json.load(f)

    with open(BEST_MODEL_INFO_PATH, "r") as f:
        info = json.load(f)

    return model, le, metrics, info

# ================= PREPROCESS =================
def preprocess_fingerprint(img_pil, size):
    img = np.array(img_pil)

    if len(img.shape) == 2:
        gray = img
    else:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    gray = cv2.resize(gray, (size, size))

    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        gray = gray.astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)

    return gray

# ================= GABOR FEATURES =================
def gabor_features(gray):
    feats = []
    gray_f = gray.astype(np.float32) / 255.0

    for theta in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
        kernel = cv2.getGaborKernel(
            (9, 9), 4.0, theta, 8.0, 0.5, 0, ktype=cv2.CV_32F
        )
        filtered = cv2.filter2D(gray_f, cv2.CV_32F, kernel)
        feats.append(filtered.mean())
        feats.append(filtered.std())

    return np.array(feats, dtype=np.float32)

# ================= FEATURE EXTRACTION =================
def extract_features(img_pil, size):
    gray = preprocess_fingerprint(img_pil, size)

    hog_features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=False,
        feature_vector=True
    ).astype(np.float32)

    lbp = local_binary_pattern(
        gray.astype(np.uint8),
        P=LBP_POINTS,
        R=LBP_RADIUS,
        method="uniform"
    )

    lbp_hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, LBP_BINS + 1),
        range=(0, LBP_BINS)
    )
    lbp_hist = lbp_hist.astype(np.float32)
    lbp_hist /= (lbp_hist.sum() + 1e-7)

    edges = cv2.Canny(gray.astype(np.uint8), 50, 150)
    edge_density = np.array([edges.mean() / 255.0], dtype=np.float32)

    th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    threshold_ratio = np.array([th.mean() / 255.0], dtype=np.float32)

    gabor = gabor_features(gray)

    features = np.hstack([
        hog_features,
        lbp_hist,
        edge_density,
        threshold_ratio,
        gabor
    ]).astype(np.float32)

    return features.reshape(1, -1)

# ================= PREDICT =================
def predict(img_pil, model, le, size):
    x = extract_features(img_pil, size)
    pred = model.predict(x)[0]
    label = le.inverse_transform([pred])[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(x)[0]
            confidence = float(np.max(probs) * 100)
        except Exception:
            confidence = None

    return label, confidence

# ================= APP =================
st.markdown('<div class="main-title">🧬 Fingerprint Blood Group Detection System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">AI-powered blood group prediction using fingerprint image features and machine learning models.</div>',
    unsafe_allow_html=True
)

try:
    model, le, metrics, info = load_all()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# Top summary cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{info['best_model']}</div>
            <div class="metric-label">Best Model</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{info['accuracy']:.2f}%</div>
            <div class="metric-label">Accuracy</div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{info['precision']:.2f}%</div>
            <div class="metric-label">Precision</div>
        </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{info['f1_score']:.2f}%</div>
            <div class="metric-label">F1-Score</div>
        </div>
    """, unsafe_allow_html=True)

st.write("")

left_col, right_col = st.columns([1, 1])

uploaded_image = None

with left_col:
    st.markdown('<div class="section-title">Upload Fingerprint</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Choose a fingerprint image",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"]
    )

    if uploaded is not None:
        try:
            uploaded_image = Image.open(uploaded).convert("RGB")
            st.image(uploaded_image, caption="Uploaded Fingerprint Image", use_container_width=True)
        except UnidentifiedImageError:
            st.error("Unsupported or corrupted image file.")
            st.stop()

with right_col:
    st.markdown('<div class="section-title">Prediction Result</div>', unsafe_allow_html=True)
    st.write(f"**Feature Type:** {info['feature_type']}")
    st.write(f"**Recall:** {info['recall']:.2f}%")
    st.write(f"**Image Size Used:** {info['img_size']} x {info['img_size']}")

    if uploaded_image is not None:
        try:
            result, conf = predict(uploaded_image, model, le, info["img_size"])

            st.markdown(f"""
                <div class="prediction-box">
                    <div class="prediction-label">Predicted Blood Group</div>
                    <div class="prediction-value">{result}</div>
                </div>
            """, unsafe_allow_html=True)

            if conf is not None:
                st.success(f"Confidence: {conf:.2f}%")
            else:
                st.info("Confidence not available for this model.")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
    else:
        st.info("Upload an image to see the prediction.")

st.write("")
st.markdown('<div class="section-title">📊 Model Comparison Table</div>', unsafe_allow_html=True)

rows = []
for model_name, m in metrics.items():
    rows.append({
        "Model": model_name,
        "Accuracy (%)": m["Accuracy"],
        "Precision (%)": m["Precision"],
        "Recall (%)": m["Recall"],
        "F1-Score (%)": m["F1-Score"]
    })

df = pd.DataFrame(rows)
df = df.sort_values(by="Accuracy (%)", ascending=False).reset_index(drop=True)

st.dataframe(df, use_container_width=True, hide_index=True)

st.write("")
st.markdown('<div class="section-title">📈 Accuracy Bar Graph</div>', unsafe_allow_html=True)

if os.path.exists(GRAPH_PATH):
    st.image(GRAPH_PATH, caption="Simulation Results: Model Accuracy Comparison", use_container_width=True)
else:
    st.warning("Graph file not found.")

st.write("")
with st.expander("View technical details"):
    st.write(f"**Selection Rule:** {info.get('selection_rule', 'N/A')}")
    if "training_times_seconds" in info:
        st.write("**Training Times (seconds):**")
        st.json(info["training_times_seconds"])