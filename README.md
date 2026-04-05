# AI-Powered Border Surveillance & Suspicious Activity Detection System

## Overview
This project presents an intelligent surveillance system designed for border security and high-risk monitoring environments. It integrates real-time video analysis, simulated sensor data, anomaly detection, and risk-based alerting to identify and respond to suspicious activities.

The system is built as a working prototype to demonstrate how AI can enhance surveillance by prioritizing high-risk events instead of generating excessive alerts.

---

## Key Features

- Real-time object detection using YOLOv8
- Intruder detection (person → labeled as Intruder)
- Simulated sensor data (motion, temperature, infrared)
- Anomaly detection based on sensor fusion
- Risk scoring system (LOW / MEDIUM / HIGH)
- Smart alert system with cooldown mechanism
- Email notifications for high-risk events
- CSV-based logging of all detections
- Visual analytics and graph generation

---

## System Architecture

Camera Feed  
→ YOLO Detection  
→ Sensor Simulation  
→ Anomaly Detection  
→ Risk Scoring  
→ Alert Trigger  
→ Logging  
→ Analytics & Visualization  

---

## Technologies Used

- Python
- OpenCV
- Ultralytics YOLOv8
- Pandas
- Matplotlib
- SMTP (Email Alerts)
- Dotenv (Environment Variables)

---

## Project Structure
- **yolov8trained.py** → Main detection + risk logic  
- **alert_system.py** → Email alert handling  
- **logger.py** → Event logging (CSV)  
- **predictor.py** → Graph creation (image)  
- **analytics.py** → Data visualization 

---

## How It Works

1. Captures live video using webcam
2. Detects objects using YOLOv8
3. Labels "person" as "Intruder"
4. Simulates sensor data in real-time
5. Detects anomalies using predefined rules
6. Calculates risk score based on:
   - Object type
   - Sensor anomaly
   - Time of day
7. Triggers alerts only for HIGH-risk events
8. Logs all events for further analysis
9. Generates graphs for insights

---

## Risk Scoring Logic

- Intruder detected → Medium risk
- Suspicious objects → High risk
- Sensor anomaly → Increases risk
- Night time → Additional risk boost

Final Output:
- LOW
- MEDIUM
- HIGH

---

## Setup Instructions

### 1. Clone Repository
git clone https://github.com/yourusername/AI-Border-Surveillance-System.git

cd AI-Border-Surveillance-System
### 2. Install Dependencies

pip install -r requirements.txt


### 3. Configure Environment Variables

Create a `.env` file:


OWNER_EMAIL=your_email@gmail.com

APP_PASSWORD=your_app_password
TO_EMAIL=receiver_email@gmail.com

LIVE_FEED_LINK=http://your_live_feed
---

## Running the System

### Start Surveillance

python yolov8trained.py


### Generate Analytics

python analytics.py


---

## Sample Outputs

- Risk distribution graph
- Object detection frequency
- Anomaly analysis
- Intruder trend over time

---

## Future Enhancements

- Custom-trained weapon detection model
- Real IoT sensor integration
- Live dashboard (Flask/Streamlit)
- Heatmap visualization for high-risk zones
- Database integration (MySQL)

---

## Limitations

- Uses pre-trained YOLO model (not weapon-specific)
- Sensor data is simulated
- Prototype-level deployment

---

## Conclusion

This system demonstrates how AI can transform traditional surveillance into an intelligent monitoring solution by combining video analytics, sensor fusion, and risk-based alert prioritization.

---

## Author

Tarj Vaishnav  
Final Year – Computer Science & Design Engineering
