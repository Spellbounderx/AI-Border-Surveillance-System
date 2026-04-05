import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("logs.csv")

# Convert timestamp
df['timestamp'] = pd.to_datetime(df['timestamp'])

# -----------------------------
# 1. Risk Distribution
# -----------------------------
risk_counts = df['risk'].value_counts()

plt.figure()
risk_counts.plot(kind='bar')
plt.title("Risk Level Distribution")
plt.xlabel("Risk Level")
plt.ylabel("Number of Events")
plt.tight_layout()
plt.savefig("risk_distribution.png")

# -----------------------------
# 2. Object Frequency
# -----------------------------
object_counts = df['object'].value_counts().head(5)

plt.figure()
object_counts.plot(kind='bar')
plt.title("Top Detected Objects")
plt.xlabel("Object")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("object_frequency.png")

# -----------------------------
# 3. Anomaly Analysis
# -----------------------------
anomaly_counts = df['anomaly'].value_counts()

plt.figure()
anomaly_counts.plot(kind='bar')
plt.title("Anomaly vs Normal Events")
plt.xlabel("Anomaly")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("anomaly_analysis.png")

# -----------------------------
# 4. Intruder Trend Over Time (BONUS)
# -----------------------------
intruders = df[df['object'] == "Intruder"]

if not intruders.empty:
    intruders['minute'] = intruders['timestamp'].dt.strftime('%H:%M')
    trend = intruders['minute'].value_counts().sort_index()

    plt.figure()
    trend.plot(kind='line')
    plt.title("Intruder Detection Trend Over Time")
    plt.xlabel("Time")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("intruder_trend.png")

print("All graphs generated successfully.")