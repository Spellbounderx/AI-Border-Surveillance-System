import pandas as pd
import matplotlib.pyplot as plt
import os

LOG_FILE = "logs.csv"

# ─────────────────────────────────────────
# LOAD & VALIDATE
# ─────────────────────────────────────────
if not os.path.exists(LOG_FILE):
    print(f"[ERROR] {LOG_FILE} not found. Run the surveillance system first.")
    exit()

df = pd.read_csv(LOG_FILE)

if df.empty:
    print("[ERROR] logs.csv is empty. No data to analyze.")
    exit()

df['timestamp'] = pd.to_datetime(df['timestamp'])
df['hour'] = df['timestamp'].dt.hour
df['minute'] = df['timestamp'].dt.strftime('%H:%M')

print(f"[INFO] Loaded {len(df)} log entries.")
print(df.head())

# ─────────────────────────────────────────
# 1. Risk Level Distribution
# ─────────────────────────────────────────
risk_counts = df['risk'].value_counts()
colors = {'HIGH': 'red', 'MEDIUM': 'orange', 'LOW': 'green'}

plt.figure(figsize=(7, 4))
risk_counts.plot(kind='bar', color=[colors.get(r, 'gray') for r in risk_counts.index])
plt.title("Risk Level Distribution")
plt.xlabel("Risk Level")
plt.ylabel("Number of Events")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("risk_distribution.png")
print("[SAVED] risk_distribution.png")

# ─────────────────────────────────────────
# 2. Top Detected Objects
# ─────────────────────────────────────────
object_counts = df['object'].value_counts().head(8)

plt.figure(figsize=(8, 4))
object_counts.plot(kind='bar', color='steelblue')
plt.title("Top Detected Objects")
plt.xlabel("Object")
plt.ylabel("Frequency")
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig("object_frequency.png")
print("[SAVED] object_frequency.png")

# ─────────────────────────────────────────
# 3. Anomaly vs Normal Events
# ─────────────────────────────────────────
anomaly_counts = df['anomaly'].value_counts()

plt.figure(figsize=(5, 4))
anomaly_counts.plot(kind='bar', color=['tomato', 'mediumseagreen'])
plt.title("Anomaly vs Normal Events")
plt.xlabel("Anomaly Detected")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("anomaly_analysis.png")
print("[SAVED] anomaly_analysis.png")

# ─────────────────────────────────────────
# 4. Intruder Detection Trend Over Time
# ─────────────────────────────────────────
intruders = df[df['object'] == "Intruder"]

if not intruders.empty:
    trend = intruders['minute'].value_counts().sort_index()
    plt.figure(figsize=(10, 4))
    trend.plot(kind='line', marker='o', color='crimson')
    plt.title("Intruder Detection Trend Over Time")
    plt.xlabel("Time (HH:MM)")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("intruder_trend.png")
    print("[SAVED] intruder_trend.png")
else:
    print("[INFO] No intruder detections found — skipping intruder trend chart.")

# ─────────────────────────────────────────
# 5. HIGH Risk Events by Hour (NEW)
# ─────────────────────────────────────────
high_risk = df[df['risk'] == 'HIGH']

if not high_risk.empty:
    hourly = high_risk['hour'].value_counts().sort_index()
    plt.figure(figsize=(9, 4))
    hourly.plot(kind='bar', color='darkred')
    plt.title("HIGH Risk Events by Hour of Day")
    plt.xlabel("Hour (24h)")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig("high_risk_by_hour.png")
    print("[SAVED] high_risk_by_hour.png")
else:
    print("[INFO] No HIGH risk events found — skipping hourly risk chart.")

# ─────────────────────────────────────────
# 6. Sensor Correlation Heatmap (NEW)
# ─────────────────────────────────────────
try:
    import seaborn as sns
    sensor_cols = ['motion', 'temperature', 'infrared']
    corr = df[sensor_cols].corr()

    plt.figure(figsize=(5, 4))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Sensor Data Correlation")
    plt.tight_layout()
    plt.savefig("sensor_correlation.png")
    print("[SAVED] sensor_correlation.png")
except ImportError:
    print("[INFO] seaborn not installed — skipping correlation heatmap. Run: pip install seaborn")

# ─────────────────────────────────────────
# 7. Confidence Distribution (NEW)
# ─────────────────────────────────────────
plt.figure(figsize=(7, 4))
df['confidence'].plot(kind='hist', bins=20, color='slateblue', edgecolor='black')
plt.title("Detection Confidence Distribution")
plt.xlabel("Confidence Score")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("confidence_distribution.png")
print("[SAVED] confidence_distribution.png")

print("\n[DONE] All graphs generated successfully.")