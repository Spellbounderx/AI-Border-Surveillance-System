import csv
import os
from datetime import datetime

LOG_FILE = "logs.csv"

def initialize_logger():
    # Create file with headers if it doesn't exist
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

def log_event(object_name, confidence, risk, sensor_data, anomaly):
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
            anomaly,
            print("LOGGED:", object_name, float(confidence), risk, sensor_data, anomaly)
        ])