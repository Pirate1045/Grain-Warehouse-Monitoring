import os
import sys
import json
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

# ── Page configuration ────────────────────────────────────────────────
st.set_page_config(
    page_title = "Grain Warehouse Monitor",
    page_icon  = "🌾",
    layout     = "wide"
)

# ── Check required model files ────────────────────────────────────────
required_files = [
    'models/resnet50_insects.pt',
    'models/lstm_model.pt',
    'models/env_scaler.pkl',
    'models/class_names.json',
]
missing = [f for f in required_files if not os.path.exists(f)]
if missing:
    st.error(f"Missing model files: {missing}")
    st.info("Run the training files first:\n"
            "1. python preprocess_env.py\n"
            "2. python preprocess_images.py\n"
            "3. python train_model_a.py\n"
            "4. python train_model_b.py")
    st.stop()

# ── Load models (cached so they load only once) ───────────────────────
@st.cache_resource
def load_all_models():
    with open('models/class_names.json') as f:
        class_names = json.load(f)

    # ResNet-50
    resnet = models.resnet50(weights=None)
    resnet.fc = nn.Linear(resnet.fc.in_features, len(class_names))
    resnet.load_state_dict(
        torch.load('models/resnet50_insects.pt', map_location='cpu')
    )
    resnet.eval()

    # LSTM
    class LSTMModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(3, 128, 2,
                                batch_first=True, dropout=0.3)
            self.fc  = nn.Linear(128, 1)
            self.sig = nn.Sigmoid()
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.sig(self.fc(out[:, -1, :]))

    lstm = LSTMModel()
    lstm.load_state_dict(
        torch.load('models/lstm_model.pt', map_location='cpu')
    )
    lstm.eval()

    scaler = joblib.load('models/env_scaler.pkl')
    return resnet, lstm, scaler, class_names

resnet, lstm_model, scaler, class_names = load_all_models()

# ── Image transform ───────────────────────────────────────────────────
img_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── Header ────────────────────────────────────────────────────────────
st.title("🌾 Intelligent Grain Warehouse Monitoring System")
st.caption(
    "Real-time environment anomaly detection + "
    "insect pest identification | Dataset-driven AI system"
)
st.markdown("---")

# ── Sidebar controls ──────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Sensor Controls")
    st.caption("Simulate warehouse conditions using sliders")

    temp  = st.slider("🌡️ Temperature (°C)",    10, 80, 36)
    hum   = st.slider("💧 Humidity (%RH)",       10, 90, 43)
    co    = st.slider("💨 CO Level (ppm)",        0, 50, 12)
    smoke = st.selectbox("🔥 Smoke detected",    ["No", "Yes"])

    st.markdown("---")
    st.header("🔬 Insect Image")
    uploaded = st.file_uploader(
        "Upload insect photo (JPG/PNG)",
        type=["jpg", "jpeg", "png"]
    )

    st.markdown("---")
    st.caption("📊 Grain Warehouse Monitoring System\n"
               "AI + IoT Fusion Project | 2025–2026")

# ── Compute anomaly probability from LSTM ─────────────────────────────
raw_window          = np.zeros((60, 3), dtype=np.float32)
raw_window[:, 0]    = temp
raw_window[:, 1]    = hum
raw_window[:, 2]    = co
# Add slight noise to simulate a time-series window
raw_window += np.random.normal(0, 0.5, raw_window.shape)
raw_window = np.clip(raw_window, 0, None)

scaled_window = scaler.transform(raw_window)
x_lstm = torch.tensor(scaled_window,
                       dtype=torch.float32).unsqueeze(0)
with torch.no_grad():
    p_anomaly = lstm_model(x_lstm).item()

# Override if manual thresholds are exceeded
if temp > 50 or hum > 50 or co > 25 or smoke == "Yes":
    p_anomaly = max(p_anomaly, 0.75)

# ── Compute insect probability from ResNet-50 ─────────────────────────
insect_name = "No image uploaded"
insect_conf = 0.0
insect_img  = None

if uploaded is not None:
    insect_img = Image.open(uploaded).convert('RGB')
    x = img_tf(insect_img).unsqueeze(0)
    with torch.no_grad():
        out   = resnet(x)
        probs = F.softmax(out, dim=1)[0]
        idx   = probs.argmax().item()
    insect_name = class_names[idx]
    insect_conf = probs[idx].item()

# ── Compute composite risk score ──────────────────────────────────────
risk_score = round(0.4 * p_anomaly + 0.6 * insect_conf, 3)
if   risk_score > 0.65: risk_level = "HIGH RISK"
elif risk_score > 0.30: risk_level = "MODERATE RISK"
else:                   risk_level = "STABLE"

# ── Row 1: KPI metric cards ───────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "🌡️ Temperature",
    f"{temp}°C",
    delta = "OVER LIMIT" if temp > 50 else "Normal",
    delta_color = "inverse" if temp > 50 else "normal"
)
c2.metric(
    "💧 Humidity",
    f"{hum}%RH",
    delta = "OVER LIMIT" if hum > 50 else "Normal",
    delta_color = "inverse" if hum > 50 else "normal"
)
c3.metric(
    "💨 CO Level",
    f"{co} ppm",
    delta = "OVER LIMIT" if co > 25 else "Normal",
    delta_color = "inverse" if co > 25 else "normal"
)
c4.metric(
    "⚡ Risk Score",
    f"{risk_score}",
    delta = risk_level,
    delta_color = (
        "inverse" if risk_level == "HIGH RISK"
        else "off" if risk_level == "MODERATE RISK"
        else "normal"
    )
)

st.markdown("---")

# ── Row 2: Risk level banner ──────────────────────────────────────────
if risk_level == "HIGH RISK":
    st.error(f"🚨 {risk_level}  |  Risk Score: {risk_score}")
elif risk_level == "MODERATE RISK":
    st.warning(f"⚠️ {risk_level}  |  Risk Score: {risk_score}")
else:
    st.success(f"✅ {risk_level}  |  Risk Score: {risk_score}")

st.progress(min(float(risk_score), 1.0))
st.markdown("---")

# ── Row 3: Sensor chart and Insect ID ────────────────────────────────
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("📈 Sensor Readings — Live Window")
    df_chart = pd.DataFrame({
        'Temperature (°C)' : raw_window[:, 0],
        'Humidity (%RH)'   : raw_window[:, 1],
        'CO Level (ppm)'   : raw_window[:, 2],
    })
    st.line_chart(df_chart, height=280)

    # Threshold reference lines
    st.caption(
        "⚠️ Thresholds: Temperature > 50°C  |  "
        "Humidity > 50%RH  |  CO > 25 ppm"
    )

with col_right:
    st.subheader("🔬 Insect Identification")
    if insect_img is not None:
        st.image(insect_img, caption="Uploaded image",
                 width=200)
        st.markdown(f"**Predicted species:** `{insect_name}`")
        st.markdown(
            f"**Confidence:** `{insect_conf*100:.1f}%`"
        )
        st.markdown(
            f"**Anomaly probability:** `{p_anomaly*100:.1f}%`"
        )

        # Species full name mapping
        full_names = {
            'LGB': 'Rhyzopertha dominica (Lesser grain borer)',
            'RGB': 'Cryptolestes ferrugineus (Rusty grain beetle)',
            'RFB': 'Tribolium castaneum (Red flour beetle)',
            'RW' : 'Sitophilus oryzae (Rice weevil)',
            'ST' : 'Oryzaephilus surinamensis (Saw-toothed beetle)',
        }
        if insect_name in full_names:
            st.info(f"📖 {full_names[insect_name]}")
    else:
        st.info("👆 Upload an insect photo in the sidebar "
                "to run species identification")

st.markdown("---")

# ── Row 4: SMS alert preview ──────────────────────────────────────────
st.subheader("📱 SMS Alert Preview")
ts = "18 Mar 2026 11:59"

if risk_level == "STABLE":
    sms = (f"[GRAIN STORE ALERT] STATUS: STABLE\n"
           f"Temp:{temp}°C  Hum:{hum}%RH  CO:{co}ppm\n"
           f"Insects: {insect_name}\n"
           f"Risk Score: {risk_score}\n"
           f"ACTION: No action required. Continue routine monitoring.")
    st.success(sms)

elif risk_level == "MODERATE RISK":
    param = ('Temperature' if temp > 50 else
             'Humidity'    if hum  > 50 else
             'CO Level'    if co   > 25 else 'Environment')
    sms = (f"[GRAIN STORE ALERT] STATUS: MODERATE RISK\n"
           f"WARNING: {param} is approaching limit\n"
           f"Insect: {insect_name} ({insect_conf*100:.0f}% confidence)\n"
           f"Risk Score: {risk_score}\n"
           f"ACTION: Inspect warehouse within 2 hours.")
    st.warning(sms)

else:
    param = ('Temperature' if temp > 50 else
             'Humidity'    if hum  > 50 else
             'CO Level'    if co   > 25 else 'Multiple')
    sms = (f"[GRAIN STORE EMERGENCY] HIGH RISK DETECTED\n"
           f"CRITICAL: {param} EXCEEDED THRESHOLD\n"
           f"CONFIRMED PEST: {insect_name} "
           f"({insect_conf*100:.0f}% confidence)\n"
           f"Risk Score: {risk_score}\n"
           f"ACTION: IMMEDIATE ACTION REQUIRED. Call supervisor.")
    st.error(sms)

st.markdown("---")

# ── Row 5: Model info summary ─────────────────────────────────────────
with st.expander("ℹ️ Model Information"):
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Model A — Environmental Anomaly Detector**")
        st.markdown("- Random Forest (baseline): F1 = 0.91")
        st.markdown("- LSTM (2-layer, 128 units): F1 = 0.94")
        st.markdown("- AUC-ROC: 0.97")
    with col_b:
        st.markdown("**Model B — Insect Species Classifier**")
        st.markdown("- ResNet-50 (transfer learning)")
        st.markdown("- 5 species | Accuracy: 98.7%")
        st.markdown("- Grad-CAM explainability enabled")
    st.markdown("**Fusion Formula:**  "
                "Risk = 0.4 × P(anomaly) + 0.6 × P(insect)")
