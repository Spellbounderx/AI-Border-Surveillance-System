import cv2
from ultralytics import YOLO
import time
from alert_system import send_alert_email  #  Import your external email function
import random
from logger import initialize_logger, log_event

initialize_logger()
def get_sensor_data():
    return {
        "motion": random.randint(0, 1),
        "temperature": random.randint(20, 45),
        "infrared": round(random.uniform(0.1, 1.0), 2)
    }

def detect_anomaly(sensor):
    if sensor["motion"] == 1 and sensor["infrared"] > 0.7:
        return True
    return False

def calculate_risk(class_name, anomaly):
    risk = 0

    if class_name in ['brassknuckles', 'switchblades']:
        risk += 5
    elif class_name in ['fire', 'smoke']:
        risk += 4
    elif class_name == 'Intruder':
        risk += 2

    if anomaly:
        risk += 4

    current_hour = time.localtime().tm_hour
    if current_hour >= 18 or current_hour <= 6:
        risk += 3

    if risk >= 8:
        return "HIGH"
    elif risk >= 4:
        return "MEDIUM"
    else:
        return "LOW"

# Load your trained model
model = YOLO("yolov8n.pt")
# Use webcam or stream link
cap = cv2.VideoCapture(0)  # or use DroidCam URL 

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

# Suspicious classes
suspicious_classes = ['brassknuckles','Hand', 'switchblades','fire', 'smoke']
last_alert_time = 0
cooldown = 30  # in seconds

sensor_data = {"motion": 0, "temperature": 0, "infrared": 0}

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame")
        break

    results = model(frame)

    for result in results:
        boxes = result.boxes.xyxy
        confs = result.boxes.conf
        class_ids = result.boxes.cls
        names = result.names

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            conf = confs[i]
            class_name = names[int(class_ids[i])]
            # Rename logic for better interpretation
            if class_name == "person":
                class_name = "Intruder"

            # Optional: mark low confidence as unknown
            if float(conf) < 0.5:
                class_name = "Unknown"
            sensor_data = get_sensor_data()
            anomaly = detect_anomaly(sensor_data)
            risk_level = calculate_risk(class_name, anomaly)
            log_event(class_name, float(conf), risk_level, sensor_data, anomaly)

            # Draw bounding box and label
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} {float(conf):.2f} | Risk: {risk_level}",
            (int(x1), int(y1)-10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # Send email if suspicious
            current_time = time.time()
            if class_name in suspicious_classes and risk_level == "HIGH" and (current_time - last_alert_time > cooldown):
                send_alert_email(class_name, risk_level, sensor_data)
                last_alert_time = current_time

    cv2.putText(frame,
            f"Motion:{sensor_data['motion']} Temp:{sensor_data['temperature']} IR:{sensor_data['infrared']}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.imshow("Surveillance Feed", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()