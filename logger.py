import csv
import os
from datetime import datetime

LOG_FILE = "logs.csv"

def initialize_logger():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                "timestamp",
                "object",
                "confidence",
                "risk",
                "motion",
                "temperature",
                "infrared",
                "anomaly"
            ])
        print(f"[INFO] Logger initialized: {LOG_FILE}")

def log_event(object_name, confidence, risk, sensor_data, anomaly):
    try:
        with open(LOG_FILE, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                object_name,
                round(float(confidence), 2),
                risk,
                sensor_data["motion"],
                sensor_data["temperature"],
                round(sensor_data["infrared"], 2),
                anomaly
            ])
        # Print is now OUTSIDE the writerow list (this was the original bug)
        print(f"[LOG] {object_name} | conf={round(float(confidence),2)} | risk={risk} | anomaly={anomaly}")
    except Exception as e:
        print(f"[ERROR] Failed to log event: {e}")