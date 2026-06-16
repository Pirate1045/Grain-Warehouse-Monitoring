import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
import sys
import shutil

print("=" * 55)
print("  FILE 1 of 8 - preprocess_env.py")
print("  Cleaning sensor CSV and creating training data")
print("=" * 55)

os.makedirs('models',  exist_ok=True)
os.makedirs('outputs', exist_ok=True)
os.makedirs(os.path.join('data', 'env_csv'), exist_ok=True)

def find_csv():
    base = os.path.dirname(os.path.abspath(__file__))
    search_paths = [
        os.path.join(base, 'data', 'env_csv', 'smoke_detection_iot.csv'),
        os.path.join(base, 'smoke_detection_iot.csv'),
        os.path.join(os.path.expanduser('~'), 'Downloads', 'smoke_detection_iot.csv'),
        os.path.join(os.path.expanduser('~'), 'Desktop',   'smoke_detection_iot.csv'),
        os.path.join(os.path.expanduser('~'), 'Downloads', 'archive', 'smoke_detection_iot.csv'),
    ]
    print("\nSearching for CSV file...")
    for p in search_paths:
        if os.path.isfile(p):
            print("  Found:", p)
            return p
    base2 = os.path.dirname(os.path.abspath(__file__))
    for root, dirs, files in os.walk(base2):
        for f in files:
            if f.endswith('.csv'):
                full = os.path.join(root, f)
                print("  Found CSV:", full)
                return full
    return None

csv_path = find_csv()

if csv_path is None:
    print("\nERROR: CSV file not found.")
    print("Download from:")
    print("  https://www.kaggle.com/datasets/deepcontractor/smoke-detection-dataset")
    dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'env_csv')
    print("Copy smoke_detection_iot.csv into: " + dest)
    sys.exit(1)

print("\nReading:", csv_path)
try:
    df = pd.read_csv(csv_path)
except PermissionError:
    print("Permission denied - copying to temp location...")
    temp = os.path.join(os.path.expanduser('~'), 'Desktop', 'sensor_temp.csv')
    shutil.copy2(csv_path, temp)
    df = pd.read_csv(temp)
    print("Read from:", temp)

print("Loaded. Shape:", df.shape)
print("Columns:", list(df.columns))

def find_col(df, keywords):
    for c in df.columns:
        cl = c.lower().replace(' ', '')
        for k in keywords:
            if k in cl:
                return c
    return None

temp_col = find_col(df, ['temp'])
hum_col  = find_col(df, ['humid'])
co_col   = find_col(df, ['co[', 'co2', 'carbon'])
fire_col = find_col(df, ['fire', 'alarm', 'smoke'])

if co_col is None:
    for c in df.columns:
        if c.strip().lower() in ['co', 'co2', 'co_ppm']:
            co_col = c
            break

print("\nColumns detected:")
print("  Temperature:", temp_col)
print("  Humidity   :", hum_col)
print("  CO         :", co_col)
print("  Fire/Alarm :", fire_col)

if not temp_col or not hum_col or not co_col or not fire_col:
    print("\nCould not detect all columns. Available columns:")
    for i, c in enumerate(df.columns):
        print("  " + str(i) + ": " + c)
    if not temp_col:
        idx = int(input("Enter NUMBER for temperature column: "))
        temp_col = df.columns[idx]
    if not hum_col:
        idx = int(input("Enter NUMBER for humidity column: "))
        hum_col = df.columns[idx]
    if not co_col:
        idx = int(input("Enter NUMBER for CO column: "))
        co_col = df.columns[idx]
    if not fire_col:
        idx = int(input("Enter NUMBER for fire/alarm column: "))
        fire_col = df.columns[idx]

df = df[[temp_col, hum_col, co_col, fire_col]].copy()
df.columns = ['temperature', 'humidity', 'co', 'anomaly']

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

df['anomaly'] = (
    (df['temperature'] > 50) |
    (df['humidity']    > 50) |
    (df['co']          > 25) |
    (df['anomaly']     == 1)
).astype(int)

print("\nTotal rows  :", len(df))
print("Anomaly rows:", df['anomaly'].sum())
print("Normal rows :", (df['anomaly'] == 0).sum())

features = ['temperature', 'humidity', 'co']
scaler   = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])
joblib.dump(scaler, os.path.join('models', 'env_scaler.pkl'))
print("Scaler saved: models/env_scaler.pkl")

WINDOW = 60
X, y   = [], []
for i in range(len(df) - WINDOW):
    X.append(df[features].iloc[i : i + WINDOW].values)
    y.append(df['anomaly'].iloc[i + WINDOW])

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

print("Window shape:", X.shape)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

np.save(os.path.join('data', 'X_train.npy'),      X_train)
np.save(os.path.join('data', 'X_val.npy'),        X_val)
np.save(os.path.join('data', 'y_train.npy'),      y_train)
np.save(os.path.join('data', 'y_val.npy'),        y_val)
np.save(os.path.join('data', 'X_train_flat.npy'), X_train.reshape(X_train.shape[0], -1))
np.save(os.path.join('data', 'X_val_flat.npy'),   X_val.reshape(X_val.shape[0], -1))

print("\n" + "=" * 55)
print("  PREPROCESSING COMPLETE")
print("=" * 55)
print("X_train shape:", X_train.shape)
print("X_val shape  :", X_val.shape)
print("\nNext step: python preprocess_images.py")
