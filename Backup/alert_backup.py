from email.message import EmailMessage
import smtplib
import os
from dotenv import load_dotenv

load_dotenv()  # Load from .env

OWNER_EMAIL = os.getenv("OWNER_EMAIL")
APP_PASSWORD = os.getenv("APP_PASSWORD")
TO_EMAIL = os.getenv("TO_EMAIL")
LIVE_FEED_LINK = os.getenv("LIVE_FEED_LINK")


def send_alert_email(detected_object):
    msg = EmailMessage()
    msg['Subject'] = 'Surveillance Alert'
    msg['From'] = OWNER_EMAIL
    msg['To'] = TO_EMAIL
    msg.set_content(f"""
    Hello,

    A suspicious object ({detected_object}) was detected on your camera system.

    View the live feed here: {LIVE_FEED_LINK}

    Stay safe,
    Your Smart Surveillance System
    """)

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(OWNER_EMAIL, APP_PASSWORD)
            smtp.send_message(msg)
            print("✅ Email alert sent.")
    except Exception as e:
        print("❌ Email failed:", e)
