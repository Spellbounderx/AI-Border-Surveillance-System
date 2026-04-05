import cv2
from ultralytics import YOLO
import time
import random
import os
import torch
from alert_system import send_alert_email
from logger import initialize_logger, log_event

initialize_logger()

# ─────────────────────────────────────────
# MODEL LOADER — handles broken .pt zips
# ─────────────────────────────────────────
def load_model():
    candidates = [
        r"C:\Users\vaish\OneDrive\Desktop\we-r-vision\best_model2_fixed.pt",
        r"C:\Users\vaish\OneDrive\Desktop\we-r-vision\best_model1_fixed.pt",
        r"C:\Users\vaish\OneDrive\Desktop\we-r-vision\yolov8n.pt",
    ]

    pkl_sources = [
        (r"C:\Users\vaish\OneDrive\Desktop\we-r-vision\best_model2\data.pkl",
         r"C:\Users\vaish\OneDrive\Desktop\we-r-vision\best_model2_fixed.pt"),
        (r"C:\Users\vaish\OneDrive\Desktop\we-r-vision\best_model1.pt\data.pkl",
         r"C:\Users\vaish\OneDrive\Desktop\we-r-vision\best_model1_fixed.pt"),
    ]

    # Try to repair pkl-based models first
    for pkl_path, save_path in pkl_sources:
        if os.path.exists(pkl_path) and not os.path.exists(save_path):
            try:
                print(f"[INFO] Attempting to repair model from {pkl_path}...")
                ckpt = torch.load(pkl_path, map_location="cpu")
                torch.save(ckpt, save_path)
                print(f"[INFO] Saved repaired model to {save_path}")
            except Exception as e:
                print(f"[WARN] Could not repair {pkl_path}: {e}")

    # Load first working model
    for path in candidates:
        if os.path.exists(path):
            try:
                model = YOLO(path)
                print(f"[INFO] Loaded model: {path}")
                return model
            except Exception as e:
                print(f"[WARN] Failed to load {path}: {e}")

    raise RuntimeError("[ERROR] No valid model found. Add yolov8n.pt or a fixed .pt file.")


# ─────────────────────────────────────────
# SENSOR SIMULATION
# ─────────────────────────────────────────
def get_sensor_data():
    return {
        "motion": random.randint(0, 1),
        "temperature": random.randint(20, 45),
        "infrared": round(random.uniform(0.1, 1.0), 2)
    }

def detect_anomaly(sensor):
    return sensor["motion"] == 1 and sensor["infrared"] > 0.7


# ─────────────────────────────────────────
# RISK SCORING
# ─────────────────────────────────────────
def calculate_risk(class_name, anomaly, conf):
    risk = 0

    if class_name in ['brassknuckles', 'switchblades']:
        risk += 5
    elif class_name in ['fire', 'smoke']:
        risk += 4
    elif class_name == 'Intruder':
        risk += 2
    elif class_name == 'Unknown':
        risk += 1

    if anomaly:
        risk += 4

    current_hour = time.localtime().tm_hour
    if current_hour >= 18 or current_hour <= 6:
        risk += 3

    # High confidence detections are more risky
    if conf > 0.85:
        risk += 1

    if risk >= 8:
        return "HIGH"
    elif risk >= 4:
        return "MEDIUM"
    else:
        return "LOW"


# ─────────────────────────────────────────
# FRAME CONSISTENCY — reduce false positives
# ─────────────────────────────────────────
detection_history = {}  # class_name -> consecutive frame count
CONSISTENCY_THRESHOLD = 3  # must appear in 3 consecutive frames to trigger alert

def is_consistent_detection(class_name):
    detection_history[class_name] = detection_history.get(class_name, 0) + 1
    return detection_history[class_name] >= CONSISTENCY_THRESHOLD

def reset_stale_detections(active_classes):
    for cls in list(detection_history.keys()):
        if cls not in active_classes:
            detection_history[cls] = 0


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
model = load_model()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Could not open video source.")
    exit()

suspicious_classes = ['brassknuckles', 'Hand', 'switchblades', 'fire', 'smoke']
last_alert_time = 0
cooldown = 30  # seconds between email alerts

sensor_data = {"motion": 0, "temperature": 0, "infrared": 0}

print("[INFO] Surveillance started. Press ESC to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to read frame.")
        break

    results = model(frame, verbose=False)
    active_classes_this_frame = set()

    for result in results:
        boxes = result.boxes.xyxy
        confs = result.boxes.conf
        class_ids = result.boxes.cls
        names = result.names

        sensor_data = get_sensor_data()
        anomaly = detect_anomaly(sensor_data)

        for i in range(len(boxes)):
            x1, y1, x2, y2 = map(int, boxes[i])
            conf = float(confs[i])
            class_name = names[int(class_ids[i])]

            # Rename person → Intruder
            if class_name == "person":
                class_name = "Intruder"

            # Low confidence → Unknown
            if conf < 0.5:
                class_name = "Unknown"

            active_classes_this_frame.add(class_name)
            risk_level = calculate_risk(class_name, anomaly, conf)
            log_event(class_name, conf, risk_level, sensor_data, anomaly)

            # Color by risk
            color = (0, 255, 0)  # GREEN = LOW
            if risk_level == "HIGH":
                color = (0, 0, 255)    # RED
            elif risk_level == "MEDIUM":
                color = (0, 165, 255)  # ORANGE

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame,
                        f"{class_name} {conf:.2f} | {risk_level}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Alert only if consistent detection + high risk
            current_time = time.time()
            if (class_name in suspicious_classes
                    and risk_level == "HIGH"
                    and is_consistent_detection(class_name)
                    and (current_time - last_alert_time > cooldown)):
                send_alert_email(class_name, risk_level, sensor_data)
                last_alert_time = current_time
                print(f"[ALERT] Email sent for {class_name} | Risk: {risk_level}")

    reset_stale_detections(active_classes_this_frame)

    # Sensor overlay
    cv2.putText(frame,
                f"Motion:{sensor_data['motion']}  Temp:{sensor_data['temperature']}C  IR:{sensor_data['infrared']}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("Border Surveillance Feed", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Surveillance ended.")