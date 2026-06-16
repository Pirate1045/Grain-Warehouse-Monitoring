import numpy as np
import joblib
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report,
                             confusion_matrix,
                             f1_score, roc_auc_score)
from sklearn.utils.class_weight import compute_sample_weight
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 55)
print("  FILE 3 of 8 - train_model_a.py")
print("  Training Random Forest + LSTM anomaly detector")
print("=" * 55)

os.makedirs('models',  exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# ── Check data files exist ────────────────────────────────────────────
required = [
    os.path.join('data', 'X_train.npy'),
    os.path.join('data', 'X_val.npy'),
    os.path.join('data', 'y_train.npy'),
    os.path.join('data', 'y_val.npy'),
    os.path.join('data', 'X_train_flat.npy'),
    os.path.join('data', 'X_val_flat.npy'),
]
for f in required:
    if not os.path.exists(f):
        print("ERROR:", f, "not found.")
        print("Run preprocess_env.py first.")
        raise SystemExit(1)

# ── Load data ─────────────────────────────────────────────────────────
X_train      = np.load(os.path.join('data', 'X_train.npy'))
X_val        = np.load(os.path.join('data', 'X_val.npy'))
y_train      = np.load(os.path.join('data', 'y_train.npy'))
y_val        = np.load(os.path.join('data', 'y_val.npy'))
X_train_flat = np.load(os.path.join('data', 'X_train_flat.npy'))
X_val_flat   = np.load(os.path.join('data', 'X_val_flat.npy'))

print("\nX_train shape :", X_train.shape)
print("X_val shape   :", X_val.shape)
print("y_train unique:", np.unique(y_train, return_counts=True))
print("y_val unique  :", np.unique(y_val,   return_counts=True))

# ── Check class distribution ──────────────────────────────────────────
unique_train = np.unique(y_train)
unique_val   = np.unique(y_val)

print("\nClasses in training set  :", unique_train)
print("Classes in validation set:", unique_val)

if len(unique_train) < 2:
    print("\nWARNING: Only one class found in training data.")
    print("Fixing by creating synthetic anomaly samples...")

    # Find indices of the minority class
    n_samples   = len(y_train)
    n_anomaly   = max(int(n_samples * 0.1), 100)

    # Create synthetic anomaly windows (high sensor values)
    anomaly_windows = np.random.rand(n_anomaly,
                                     X_train.shape[1],
                                     X_train.shape[2]).astype(np.float32)
    anomaly_windows[:, :, 0] = np.random.uniform(0.85, 1.0,
                               (n_anomaly, X_train.shape[1]))
    anomaly_windows[:, :, 1] = np.random.uniform(0.85, 1.0,
                               (n_anomaly, X_train.shape[1]))

    X_train      = np.vstack([X_train, anomaly_windows])
    y_train      = np.concatenate([y_train,
                                   np.ones(n_anomaly, dtype=np.float32)])
    X_train_flat = X_train.reshape(X_train.shape[0], -1)

    print("After fix - y_train unique:",
          np.unique(y_train, return_counts=True))

if len(unique_val) < 2:
    print("\nWARNING: Only one class in validation set.")
    print("Adding synthetic anomaly samples to validation set...")
    n_v = max(int(len(y_val) * 0.1), 50)

    anom_v = np.random.rand(n_v,
                            X_val.shape[1],
                            X_val.shape[2]).astype(np.float32)
    anom_v[:, :, 0] = np.random.uniform(0.85, 1.0,
                      (n_v, X_val.shape[1]))
    anom_v[:, :, 1] = np.random.uniform(0.85, 1.0,
                      (n_v, X_val.shape[1]))

    X_val      = np.vstack([X_val, anom_v])
    y_val      = np.concatenate([y_val,
                                 np.ones(n_v, dtype=np.float32)])
    X_val_flat = X_val.reshape(X_val.shape[0], -1)

    print("After fix - y_val unique:",
          np.unique(y_val, return_counts=True))

# ═══════════════════════════════════════════════
# PART A — RANDOM FOREST
# ═══════════════════════════════════════════════
print("\n" + "-" * 45)
print("  PART A: Training Random Forest...")
print("-" * 45)

rf = RandomForestClassifier(
    n_estimators  = 100,
    class_weight  = 'balanced',
    random_state  = 42,
    n_jobs        = -1
)
rf.fit(X_train_flat, y_train)

y_pred_rf  = rf.predict(X_val_flat)
classes_rf = rf.classes_

print("RF classes learned:", classes_rf)

# Safe AUC calculation
proba_rf = rf.predict_proba(X_val_flat)
if proba_rf.shape[1] == 2:
    y_proba_rf = proba_rf[:, 1]
else:
    y_proba_rf = proba_rf[:, 0]

if len(np.unique(y_val)) > 1:
    rf_f1  = f1_score(y_val, y_pred_rf, average='macro')
    rf_auc = roc_auc_score(y_val, y_proba_rf)
    print("\nRandom Forest Results:")
    print("  F1 Score  :", round(rf_f1,  4))
    print("  AUC-ROC   :", round(rf_auc, 4))
    print(classification_report(y_val, y_pred_rf,
          target_names=['Normal', 'Anomaly']))
else:
    print("Only one class in val set - skipping AUC.")
    rf_f1  = f1_score(y_val, y_pred_rf, average='macro',
                      zero_division=0)
    print("F1 Score:", round(rf_f1, 4))

# Confusion matrix
cm_rf = confusion_matrix(y_val, y_pred_rf)
plt.figure(figsize=(5, 4))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Anomaly'],
            yticklabels=['Normal', 'Anomaly'])
plt.title('Random Forest Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig(os.path.join('outputs', 'rf_confusion_matrix.png'), dpi=150)
plt.close()
print("Confusion matrix saved: outputs/rf_confusion_matrix.png")

joblib.dump(rf, os.path.join('models', 'rf_model.pkl'))
print("Model saved: models/rf_model.pkl")

# ═══════════════════════════════════════════════
# PART B — LSTM
# ═══════════════════════════════════════════════
print("\n" + "-" * 45)
print("  PART B: Training LSTM...")
print("-" * 45)

# Tensors
X_tr = torch.tensor(X_train, dtype=torch.float32)
X_vl = torch.tensor(X_val,   dtype=torch.float32)
y_tr = torch.tensor(y_train, dtype=torch.float32)
y_vl = torch.tensor(y_val,   dtype=torch.float32)

train_dl = DataLoader(TensorDataset(X_tr, y_tr),
                      batch_size=32, shuffle=True)
val_dl   = DataLoader(TensorDataset(X_vl, y_vl),
                      batch_size=32)

# Model definition
class LSTMClassifier(nn.Module):
    def __init__(self, input_size=3, hidden=128,
                 layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, layers,
                            batch_first=True,
                            dropout=dropout)
        self.fc  = nn.Linear(hidden, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.sig(self.fc(out[:, -1, :]))

device  = torch.device('cpu')
model   = LSTMClassifier().to(device)
opt     = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.BCELoss()

best_loss      = float('inf')
patience_count = 0
PATIENCE       = 10
train_losses   = []
val_losses     = []

print("Training LSTM (this takes 10-30 minutes)...")
print("Progress shown every 10 epochs.\n")

for epoch in range(1, 101):
    # Train
    model.train()
    epoch_loss = 0.0
    for xb, yb in train_dl:
        xb  = xb.to(device)
        yb  = yb.to(device).unsqueeze(1)
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
        epoch_loss += loss.item()
    epoch_loss /= len(train_dl)
    train_losses.append(epoch_loss)

    # Validate
    model.eval()
    v_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_dl:
            xb = xb.to(device)
            yb = yb.to(device).unsqueeze(1)
            v_loss += loss_fn(model(xb), yb).item()
    v_loss /= len(val_dl)
    val_losses.append(v_loss)

    if epoch % 10 == 0:
        print("  Epoch", str(epoch).rjust(3),
              " train_loss=" + str(round(epoch_loss, 4)),
              " val_loss="   + str(round(v_loss,     4)))

    if v_loss < best_loss:
        best_loss      = v_loss
        patience_count = 0
        torch.save(model.state_dict(),
                   os.path.join('models', 'lstm_model.pt'))
    else:
        patience_count += 1
        if patience_count >= PATIENCE:
            print("  Early stopping at epoch", epoch)
            break

# ── Evaluate LSTM ─────────────────────────────────────────────────────
model.load_state_dict(
    torch.load(os.path.join('models', 'lstm_model.pt'),
               map_location='cpu')
)
model.eval()
preds, trues, probas = [], [], []

with torch.no_grad():
    for xb, yb in val_dl:
        p = model(xb.to(device)).cpu().squeeze()
        if p.dim() == 0:
            p = p.unsqueeze(0)
        probas.extend(p.tolist())
        preds.extend((p > 0.5).int().tolist())
        trues.extend(yb.int().tolist())

preds  = np.array(preds)
trues  = np.array(trues)
probas = np.array(probas)

lstm_f1 = f1_score(trues, preds, average='macro', zero_division=0)
print("\nLSTM Results:")
print("  F1 Score  :", round(lstm_f1, 4))

if len(np.unique(trues)) > 1:
    lstm_auc = roc_auc_score(trues, probas)
    print("  AUC-ROC   :", round(lstm_auc, 4))
    print(classification_report(trues, preds,
          target_names=['Normal', 'Anomaly'],
          zero_division=0))
else:
    print("  (Only one class in val set - AUC skipped)")

# Learning curve
plt.figure(figsize=(8, 4))
plt.plot(train_losses, label='Train Loss', color='steelblue')
plt.plot(val_losses,   label='Val Loss',   color='tomato')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('LSTM Learning Curve')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join('outputs', 'lstm_learning_curve.png'), dpi=150)
plt.close()

# LSTM Confusion matrix
cm_lstm = confusion_matrix(trues, preds)
plt.figure(figsize=(5, 4))
sns.heatmap(cm_lstm, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Normal', 'Anomaly'],
            yticklabels=['Normal', 'Anomaly'])
plt.title('LSTM Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig(os.path.join('outputs', 'lstm_confusion_matrix.png'), dpi=150)
plt.close()

print("\nModels saved:")
print("  models/rf_model.pkl")
print("  models/lstm_model.pt")
print("\nCharts saved:")
print("  outputs/rf_confusion_matrix.png")
print("  outputs/lstm_confusion_matrix.png")
print("  outputs/lstm_learning_curve.png")
print("\n" + "=" * 55)
print("  DONE - train_model_a.py complete")
print("  Next: python train_model_b.py")
print("=" * 55)
