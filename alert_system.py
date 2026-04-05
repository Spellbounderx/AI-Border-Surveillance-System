import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

SENDER_EMAIL = os.getenv("SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("SENDER_PASSWORD")
RECEIVER_EMAIL = os.getenv("RECEIVER_EMAIL")

def send_alert_email(class_name, risk_level, sensor_data):
    if not all([SENDER_EMAIL, SENDER_PASSWORD, RECEIVER_EMAIL]):
        print("[WARN] Email credentials missing in .env — alert not sent.")
        return

    try:
        subject = f"[BORDER ALERT] {risk_level} Risk — {class_name} Detected"

        body = f"""
        ⚠️ BORDER SURVEILLANCE ALERT ⚠️
        ─────────────────────────────────
        Time       : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        Object     : {class_name}
        Risk Level : {risk_level}

        Sensor Readings:
          Motion     : {sensor_data['motion']}
          Temperature: {sensor_data['temperature']} °C
          Infrared   : {sensor_data['infrared']}
        ─────────────────────────────────
        Immediate review recommended.
        """

        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECEIVER_EMAIL
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())

        print(f"[ALERT] Email sent: {class_name} | {risk_level}")

    except smtplib.SMTPAuthenticationError:
        print("[ERROR] Email authentication failed. Check SENDER_EMAIL and SENDER_PASSWORD in .env")
    except smtplib.SMTPException as e:
        print(f"[ERROR] SMTP error: {e}")
    except Exception as e:
        print(f"[ERROR] Failed to send alert email: {e}")