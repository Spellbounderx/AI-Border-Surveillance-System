"""
predictor.py
------------
Trains a logistic regression model on logs.csv to predict HIGH risk time windows.
Addresses Problem Statement Objective #4:
"Analyze historical incident data to predict high-risk zones"
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

LOG_FILE = "logs.csv"
MODEL_FILE = "risk_predictor.pkl"

# ─────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────
def build_features(df):
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Time features
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['is_night'] = ((df['hour'] >= 18) | (df['hour'] <= 6)).astype(int)
    df['is_peak_risk'] = ((df['hour'] >= 22) | (df['hour'] <= 4)).astype(int)

    # Object encoding
    obj_risk = {
        'brassknuckles': 5,
        'switchblades': 5,
        'fire': 4,
        'smoke': 4,
        'Intruder': 2,
        'Hand': 1,
        'Unknown': 0
    }
    df['object_risk_score'] = df['object'].map(obj_risk).fillna(1)

    # Sensor features
    df['sensor_score'] = (
        df['motion'] * 3 +
        (df['infrared'] > 0.7).astype(int) * 2 +
        (df['temperature'] > 35).astype(int)
    )

    # Anomaly as int
    df['anomaly_int'] = df['anomaly'].astype(int)

    # Target: 1 if HIGH risk, 0 otherwise
    df['target'] = (df['risk'] == 'HIGH').astype(int)

    feature_cols = [
        'hour', 'minute', 'is_night', 'is_peak_risk',
        'object_risk_score', 'confidence',
        'motion', 'temperature', 'infrared',
        'sensor_score', 'anomaly_int'
    ]

    return df[feature_cols], df['target']


# ─────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────
def train_model():
    if not os.path.exists(LOG_FILE):
        print(f"[ERROR] {LOG_FILE} not found. Run surveillance first to generate data.")
        return None

    df = pd.read_csv(LOG_FILE)
    if len(df) < 20:
        print(f"[WARN] Only {len(df)} log entries. Need more data for reliable predictions.")
        print("[INFO] Using synthetic augmentation to supplement...")
        df = augment_data(df)

    X, y = build_features(df)

    print(f"[INFO] Training on {len(df)} samples | HIGH risk events: {y.sum()} ({y.mean()*100:.1f}%)")

    if y.nunique() < 2:
        print("[WARN] Only one class in data. Run surveillance longer to capture HIGH risk events.")
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Random Forest — better than logistic regression for this type of data
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=42,
        class_weight='balanced'  # handles imbalanced HIGH vs LOW
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("\n[RESULTS] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['LOW/MEDIUM', 'HIGH']))

    # Save model
    joblib.dump(model, MODEL_FILE)
    print(f"[SAVED] Model saved to {MODEL_FILE}")

    # Feature importance chart
    plot_feature_importance(model, X.columns.tolist())

    # High risk hour heatmap
    plot_risk_heatmap(df)

    # Confusion matrix
    plot_confusion_matrix(y_test, y_pred)

    return model


# ─────────────────────────────────────────
# PREDICT (called from yolov8trained.py)
# ─────────────────────────────────────────
def predict_risk(hour, minute, object_name, confidence, sensor_data, anomaly):
    """
    Returns predicted risk level using trained ML model.
    Falls back to rule-based if model not available.
    """
    if not os.path.exists(MODEL_FILE):
        return None  # fallback to rule-based

    try:
        model = joblib.load(MODEL_FILE)

        obj_risk = {
            'brassknuckles': 5, 'switchblades': 5,
            'fire': 4, 'smoke': 4,
            'Intruder': 2, 'Hand': 1, 'Unknown': 0
        }

        sensor_score = (
            sensor_data['motion'] * 3 +
            (sensor_data['infrared'] > 0.7) * 2 +
            (sensor_data['temperature'] > 35)
        )

        features = np.array([[
            hour, minute,
            int(hour >= 18 or hour <= 6),
            int(hour >= 22 or hour <= 4),
            obj_risk.get(object_name, 1),
            confidence,
            sensor_data['motion'],
            sensor_data['temperature'],
            sensor_data['infrared'],
            sensor_score,
            int(anomaly)
        ]])

        prob = model.predict_proba(features)[0][1]  # probability of HIGH

        if prob >= 0.7:
            return "HIGH"
        elif prob >= 0.35:
            return "MEDIUM"
        else:
            return "LOW"

    except Exception as e:
        print(f"[WARN] Predictor failed: {e}")
        return None


# ─────────────────────────────────────────
# SYNTHETIC DATA AUGMENTATION
# helps when log data is limited
# ─────────────────────────────────────────
def augment_data(df):
    import random
    synthetic_rows = []
    objects = ['Intruder', 'Unknown', 'fire', 'smoke', 'brassknuckles']

    for _ in range(200):
        hour = random.randint(0, 23)
        is_night = hour >= 18 or hour <= 6
        obj = random.choice(objects)
        motion = random.randint(0, 1)
        ir = round(random.uniform(0.1, 1.0), 2)
        temp = random.randint(20, 45)
        anomaly = motion == 1 and ir > 0.7
        conf = round(random.uniform(0.5, 0.99), 2)

        obj_risk = {'brassknuckles': 5, 'switchblades': 5,
                    'fire': 4, 'smoke': 4, 'Intruder': 2,
                    'Hand': 1, 'Unknown': 0}
        risk_score = obj_risk.get(obj, 1) + (4 if anomaly else 0) + (3 if is_night else 0)
        risk = "HIGH" if risk_score >= 8 else ("MEDIUM" if risk_score >= 4 else "LOW")

        from datetime import datetime, timedelta
        ts = datetime.now() - timedelta(hours=random.randint(0, 48))

        synthetic_rows.append({
            'timestamp': ts.strftime("%Y-%m-%d %H:%M:%S"),
            'object': obj, 'confidence': conf, 'risk': risk,
            'motion': motion, 'temperature': temp,
            'infrared': ir, 'anomaly': anomaly
        })

    synthetic_df = pd.DataFrame(synthetic_rows)
    combined = pd.concat([df, synthetic_df], ignore_index=True)
    print(f"[INFO] Augmented dataset: {len(df)} real + {len(synthetic_df)} synthetic = {len(combined)} total")
    return combined


# ─────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────
def plot_feature_importance(model, feature_names):
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(importance)), importance[indices], color='steelblue')
    plt.xticks(range(len(importance)),
               [feature_names[i] for i in indices],
               rotation=30, ha='right')
    plt.title("Feature Importance — Risk Predictor")
    plt.ylabel("Importance Score")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    print("[SAVED] feature_importance.png")


def plot_risk_heatmap(df):
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day'] = df['timestamp'].dt.day_name()

    day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    pivot = df[df['risk'] == 'HIGH'].groupby(['day', 'hour']).size().unstack(fill_value=0)
    pivot = pivot.reindex([d for d in day_order if d in pivot.index])

    if pivot.empty:
        print("[INFO] Not enough HIGH risk events for heatmap yet.")
        return

    plt.figure(figsize=(14, 5))
    sns.heatmap(pivot, cmap='YlOrRd', linewidths=0.5, annot=True, fmt='d')
    plt.title("HIGH Risk Event Heatmap — Day vs Hour")
    plt.xlabel("Hour of Day")
    plt.ylabel("Day of Week")
    plt.tight_layout()
    plt.savefig("risk_heatmap.png")
    print("[SAVED] risk_heatmap.png")


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['LOW/MED', 'HIGH'],
                yticklabels=['LOW/MED', 'HIGH'])
    plt.title("Confusion Matrix — Risk Predictor")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    print("[SAVED] confusion_matrix.png")


# ─────────────────────────────────────────
# RUN STANDALONE
# ─────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  BORDER SURVEILLANCE — RISK PREDICTOR TRAINING")
    print("=" * 50)
    model = train_model()
    if model:
        print("\n[DONE] Predictor ready. It will now be used automatically in yolov8trained.py")
    else:
        print("\n[WARN] Training failed. Check logs above.")