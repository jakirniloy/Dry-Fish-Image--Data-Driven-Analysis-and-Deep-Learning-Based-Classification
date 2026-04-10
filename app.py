import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
from PIL import Image
import pandas as pd

# -------------------------------
# Streamlit config
# -------------------------------
st.set_page_config(
    page_title="Dry Fish Classifier",
    layout="wide",
    page_icon="🐟",
)

# -------------------------------
# Custom CSS — Dynamic Colorful Light Design
# -------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 40%, #f0fdf4 100%);
    min-height: 100vh;
}

/* ── Hero Title Banner ── */
.hero-banner {
    background: linear-gradient(135deg, #0ea5e9, #6366f1, #ec4899);
    border-radius: 20px;
    padding: 32px 40px;
    margin-bottom: 28px;
    text-align: center;
    box-shadow: 0 8px 32px rgba(99,102,241,0.25);
    animation: gradientShift 6s ease infinite;
    background-size: 200% 200%;
}

@keyframes gradientShift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.hero-banner h1 {
    color: white !important;
    font-size: 2.6rem;
    font-weight: 700;
    margin: 0;
    text-shadow: 0 2px 10px rgba(0,0,0,0.2);
    letter-spacing: 1px;
}

.hero-banner p {
    color: rgba(255,255,255,0.88);
    font-size: 1rem;
    margin: 8px 0 0;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e3a5f 60%, #0e7490 100%) !important;
    border-right: none;
}

section[data-testid="stSidebar"] * {
    color: #e0f2fe !important;
}

section[data-testid="stSidebar"] .stFileUploader label,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #7dd3fc !important;
    font-weight: 600;
}

/* Sidebar buttons */
section[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #0ea5e9, #6366f1);
    color: white !important;
    border: none;
    border-radius: 12px;
    padding: 12px 0;
    font-size: 1rem;
    font-weight: 600;
    margin-top: 8px;
    cursor: pointer;
    transition: transform 0.2s, box-shadow 0.2s;
    box-shadow: 0 4px 14px rgba(14,165,233,0.4);
}

section[data-testid="stSidebar"] .stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(99,102,241,0.5);
}

/* ── Result Card ── */
.result-card {
    background: white;
    border-radius: 20px;
    padding: 28px 32px;
    box-shadow: 0 4px 24px rgba(14,165,233,0.12);
    border-left: 6px solid #6366f1;
    margin-bottom: 24px;
    transition: transform 0.2s;
}

.result-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 32px rgba(99,102,241,0.18);
}

/* ── Metric Badges ── */
.badge-row {
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    margin: 16px 0;
}

.badge {
    border-radius: 40px;
    padding: 10px 22px;
    font-weight: 600;
    font-size: 1rem;
    color: white;
    display: inline-flex;
    align-items: center;
    gap: 8px;
    box-shadow: 0 3px 12px rgba(0,0,0,0.12);
}

.badge-blue  { background: linear-gradient(135deg, #0ea5e9, #6366f1); }
.badge-green { background: linear-gradient(135deg, #10b981, #059669); }
.badge-pink  { background: linear-gradient(135deg, #ec4899, #8b5cf6); }

/* ── Section Headers ── */
.section-title {
    font-size: 1.15rem;
    font-weight: 700;
    color: #0f172a;
    border-bottom: 3px solid #0ea5e9;
    padding-bottom: 6px;
    margin: 20px 0 14px;
    display: inline-block;
}

/* ── Warning Box ── */
.warn-box {
    background: linear-gradient(135deg, #fef9c3, #fde68a);
    border-left: 5px solid #f59e0b;
    border-radius: 14px;
    padding: 16px 20px;
    color: #78350f;
    font-weight: 600;
    margin-top: 12px;
}

/* ── Image containers ── */
.img-card {
    background: white;
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 4px 18px rgba(14,165,233,0.12);
    text-align: center;
}

.img-card h4 {
    color: #6366f1;
    font-weight: 700;
    margin-bottom: 10px;
}

/* ── Description box ── */
.desc-box {
    background: linear-gradient(135deg, #f0f9ff, #e0f2fe);
    border-radius: 14px;
    padding: 16px 20px;
    color: #0c4a6e;
    font-size: 0.95rem;
    border: 1px solid #bae6fd;
    margin-top: 8px;
}

/* ── Footer ── */
.footer {
    margin-top: 48px;
    padding: 22px;
    text-align: center;
    background: linear-gradient(135deg, #0f172a, #1e3a5f);
    border-radius: 16px;
    color: #94a3b8;
    font-size: 0.9rem;
}

.footer a {
    color: #38bdf8;
    text-decoration: none;
    font-weight: 600;
    transition: color 0.2s;
}

.footer a:hover {
    color: #818cf8;
    text-decoration: underline;
}

/* ── Divider ── */
.custom-divider {
    height: 3px;
    background: linear-gradient(90deg, #0ea5e9, #6366f1, #ec4899);
    border-radius: 10px;
    margin: 24px 0;
}

/* streamlit image caption style */
.stImage > div > img {
    border-radius: 14px;
}
</style>
""", unsafe_allow_html=True)

# ── Hero Banner ──
st.markdown("""
<div class="hero-banner">
    <h1>🐟 Dry Fish Classifier</h1>
    <p>AI-powered identification with Grad-CAM visual explanation · 12 Bangladeshi dry fish species</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_trained_model():
    return load_model("hybrid_cnn_attention_dryfish.h5")

model = load_trained_model()
model.trainable = False

# -------------------------------
# Class labels and descriptions
# -------------------------------
class_names = [
    'Bashpata', 'Chanda', 'Chapila', 'Chewa', 'Churi', 'Loitta',
    'Shukna Feuwa', 'Shundori', 'chingri', 'kachki', 'narkeli', 'puti chepa'
]

class_descriptions = {
    'Bashpata': "Flat-bodied dried fish with elongated shape, typically sun-dried.",
    'Chanda': "Small, oval-shaped fish known for its translucent body texture.",
    'Chapila': "Medium-sized dried fish with silver scales, common in river catch.",
    'Chewa': "Slender-bodied fish, popular in coastal markets.",
    'Churi': "Long, ribbon-like fish with sharp head features.",
    'Loitta': "Cylindrical-bodied fish with distinctive dorsal fin pattern.",
    'Shukna Feuwa': "Curved-bodied dried fish, often heavily salted during processing.",
    'Shundori': "Small, slender fish with delicate body structure.",
    'chingri': "Dried shrimp, small, used widely in curries and condiments.",
    'kachki': "Small fish species with narrow bodies, usually sun-dried.",
    'narkeli': "Medium-sized dried fish with firm body texture.",
    'puti chepa': "Flat-bodied dried fish with noticeable fin edges."
}

# Class color accents (cycling)
class_colors = {
    'Bashpata':'#0ea5e9','Chanda':'#6366f1','Chapila':'#10b981',
    'Chewa':'#f59e0b','Churi':'#ec4899','Loitta':'#8b5cf6',
    'Shukna Feuwa':'#14b8a6','Shundori':'#f43f5e','chingri':'#f97316',
    'kachki':'#06b6d4','narkeli':'#84cc16','puti chepa':'#a855f7'
}

# -------------------------------
# Preprocess
# -------------------------------
def preprocess(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

# -------------------------------
# Grad-CAM
# -------------------------------
def manual_gradcam(img_array, input_img):
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break
    if last_conv_layer is None:
        return np.array(input_img)

    conv_model = Model(inputs=model.inputs, outputs=last_conv_layer.output)
    conv_output = conv_model.predict(img_array)[0]

    heatmap = np.mean(conv_output, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-10

    heatmap = cv2.resize(heatmap, (input_img.width, input_img.height))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(np.array(input_img), 0.6, heatmap, 0.4, 0)
    return overlay

# -------------------------------
# Sidebar
# -------------------------------
st.sidebar.markdown("## 🐟 Control Panel")
st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("📂 Upload a fish image", type=["jpg", "jpeg", "png"])
st.sidebar.markdown("---")
predict_btn = st.sidebar.button("🔍 Predict Class")
gradcam_btn = st.sidebar.button("🌡️ Show Grad-CAM")

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="font-size:0.82rem; color:#7dd3fc; line-height:1.6;">
    <b>Supported Species:</b><br>
    Bashpata · Chanda · Chapila · Chewa · Churi · Loitta ·
    Shukna Feuwa · Shundori · Chingri · Kachki · Narkeli · Puti Chepa
</div>
""", unsafe_allow_html=True)

# -------------------------------
# Main Content
# -------------------------------
if not uploaded_file:
    # Welcome state
    st.markdown("""
    <div class="result-card" style="text-align:center; border-left-color:#10b981;">
        <h3 style="color:#0f172a;">👋 Welcome to the Dry Fish Classifier!</h3>
        <p style="color:#475569; font-size:1rem;">
            Upload a dry fish image from the sidebar, then click <b>Predict Class</b> to identify the species,
            or <b>Show Grad-CAM</b> to visualize which areas the AI is focusing on.
        </p>
        <div class="badge-row" style="justify-content:center; margin-top:20px;">
            <span class="badge badge-blue">🧠 CNN + Attention Model</span>
            <span class="badge badge-green">🐠 12 Fish Species</span>
            <span class="badge badge-pink">🌡️ Grad-CAM XAI</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    input_img = Image.open(uploaded_file).convert("RGB")
    img_array = preprocess(input_img)

    # ── Prediction ──
    if predict_btn:
        with st.spinner("🔍 Analyzing image..."):
            preds = model.predict(img_array)
            class_index = int(np.argmax(preds))
            predicted_class = class_names[class_index]
            confidence = preds[0][class_index]
            accent = class_colors.get(predicted_class, '#6366f1')

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

        if confidence < 0.5:
            col_img, col_warn = st.columns([1, 2])
            with col_img:
                st.markdown('<div class="img-card"><h4>📸 Uploaded Image</h4>', unsafe_allow_html=True)
                st.image(input_img.resize((260, 260)), use_container_width=False)
                st.markdown('</div>', unsafe_allow_html=True)
            with col_warn:
                st.markdown("""
                <div class="warn-box">
                    ⚠️ <b>Unrecognized Image</b><br><br>
                    The uploaded image does not appear to be a recognized dry fish species.
                    Please upload a clear, well-lit image of a dry fish.
                </div>
                """, unsafe_allow_html=True)
        else:
            col_img, col_result = st.columns([1, 2])

            with col_img:
                st.markdown('<div class="img-card"><h4>📸 Uploaded Image</h4>', unsafe_allow_html=True)
                st.image(input_img.resize((280, 280)), use_container_width=False)
                st.markdown('</div>', unsafe_allow_html=True)

            with col_result:
                st.markdown(f"""
                <div class="result-card" style="border-left-color:{accent};">
                    <div class="badge-row">
                        <span class="badge badge-blue">🎯 {predicted_class}</span>
                        <span class="badge badge-green">✅ {confidence*100:.2f}% Confidence</span>
                    </div>
                    <p class="section-title">📖 Description</p>
                    <div class="desc-box">{class_descriptions[predicted_class]}</div>
                </div>
                """, unsafe_allow_html=True)

            # Probabilities Chart
            st.markdown('<p class="section-title">📊 Prediction Probabilities</p>', unsafe_allow_html=True)
            prob_dict = {class_names[i]: float(preds[0][i]) * 100 for i in range(len(class_names))}
            prob_df = pd.DataFrame(list(prob_dict.items()), columns=['Class', 'Probability (%)'])
            prob_df = prob_df.sort_values('Probability (%)', ascending=False)
            st.bar_chart(prob_df.set_index('Class')['Probability (%)'], height=320)

    # ── Grad-CAM ──
    if gradcam_btn:
        with st.spinner("🌡️ Generating Grad-CAM visualization..."):
            gradcam_overlay = manual_gradcam(img_array, input_img)

        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        st.markdown('<p class="section-title">🌡️ Grad-CAM Visual Explanation</p>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="img-card"><h4>📷 Original Image</h4>', unsafe_allow_html=True)
            st.image(input_img, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="img-card"><h4>🔥 Grad-CAM Heatmap</h4>', unsafe_allow_html=True)
            st.image(gradcam_overlay, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="desc-box" style="margin-top:12px;">
            🔴 <b>Red/Warm areas</b> — regions the AI focused on most for the prediction. &nbsp;
            🔵 <b>Blue/Cool areas</b> — low attention regions.
        </div>
        """, unsafe_allow_html=True)

# ── Footer ──
st.markdown("""
<div class="custom-divider"></div>
<div class="footer">
    🐟 Dry Fish Classifier · Powered by CNN + Attention &amp; Grad-CAM XAI<br><br>
    Created by <b style="color:#f0f9ff;">Jakir Hossain</b> &nbsp;|&nbsp;
    <a href="https://jakirniloy.github.io" target="_blank">🌐 jakirniloy.github.io</a>
</div>
""", unsafe_allow_html=True)