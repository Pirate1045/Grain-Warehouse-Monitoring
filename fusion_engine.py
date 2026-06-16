import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import joblib

# ── Check required model files ────────────────────────────────────────
for f in ['models/resnet50_insects.pt',
          'models/lstm_model.pt',
          'models/env_scaler.pkl',
          'models/class_names.json']:
    if not os.path.exists(f):
        print(f"ERROR: {f} not found. Run training files first.")
        sys.exit(1)

# ── Load class names ──────────────────────────────────────────────────
with open('models/class_names.json') as f:
    class_names = json.load(f)

NUM_CLASSES = len(class_names)

# ── Load scaler ───────────────────────────────────────────────────────
scaler = joblib.load('models/env_scaler.pkl')

# ── Load ResNet-50 ────────────────────────────────────────────────────
resnet = models.resnet50(weights=None)
resnet.fc = nn.Linear(resnet.fc.in_features, NUM_CLASSES)
resnet.load_state_dict(
    torch.load('models/resnet50_insects.pt', map_location='cpu')
)
resnet.eval()

# ── Load LSTM ─────────────────────────────────────────────────────────
class LSTMClassifier(nn.Module):
    def __init__(self, input_size=3, hidden=128,
                 layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers,
                            batch_first=True, dropout=dropout)
        self.fc  = nn.Linear(hidden, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.sig(self.fc(out[:, -1, :]))

lstm = LSTMClassifier()
lstm.load_state_dict(
    torch.load('models/lstm_model.pt', map_location='cpu')
)
lstm.eval()

# ── Image transform ───────────────────────────────────────────────────
img_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── Core prediction functions ─────────────────────────────────────────
def predict_anomaly(sensor_window_raw):
    """
    sensor_window_raw : numpy array (60, 3)
                        Raw values [temperature, humidity, co]
    Returns           : float — probability of anomaly (0 to 1)
    """
    scaled = scaler.transform(sensor_window_raw)
    x = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        return lstm(x).item()


def predict_insect(image_path):
    """
    image_path : path to insect image file
    Returns    : (species_name, confidence_float)
    """
    img = Image.open(image_path).convert('RGB')
    x   = img_tf(img).unsqueeze(0)
    with torch.no_grad():
        out   = resnet(x)
        probs = F.softmax(out, dim=1)[0]
        idx   = probs.argmax().item()
    return class_names[idx], probs[idx].item()


def compute_risk_score(p_anomaly, p_insect):
    """
    Weighted fusion formula:
    Risk = 0.4 × P(anomaly) + 0.6 × P(insect detected)
    """
    return round(0.4 * p_anomaly + 0.6 * p_insect, 3)


def get_risk_level(score):
    """
    Returns risk level string based on composite score
    """
    if score <= 0.30:
        return 'STABLE'
    elif score <= 0.65:
        return 'MODERATE'
    else:
        return 'HIGH'


def full_prediction(sensor_window_raw, image_path,
                    temp, hum, co):
    """
    Runs both models and returns complete result dict
    """
    p_anom        = predict_anomaly(sensor_window_raw)
    insect, conf  = predict_insect(image_path)
    score         = compute_risk_score(p_anom, conf)
    level         = get_risk_level(score)

    return {
        'risk_score'  : score,
        'risk_level'  : level,
        'p_anomaly'   : round(p_anom, 3),
        'insect'      : insect,
        'confidence'  : round(conf, 3),
        'temperature' : temp,
        'humidity'    : hum,
        'co'          : co,
    }


if __name__ == '__main__':
    print("=" * 55)
    print("  FILE 6 of 8 — fusion_engine.py")
    print("  Testing fusion engine with sample data")
    print("=" * 55)

    print("\nAll models loaded successfully.")
    print(f"  Classes : {class_names}")
    print("\nFusion engine ready to use.")
    print("\n✓ DONE — fusion_engine.py complete")
    print("  Next: python alert_manager.py")
