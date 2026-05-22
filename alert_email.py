import os
import smtplib
from datetime import datetime
from email.message import EmailMessage
from dotenv import load_dotenv

load_dotenv()


def send_criminal_alert_email(
    person_id,          # kept for DB logging only — NOT shown in email
    person_name,
    confidence,
    station_name,
    platform_number,
    face_image_path,    # always None now — face image not sent
    frame_image_path    # annotated CCTV frame with box + details banner
):
    sender   = os.getenv("MAIL_SENDER")
    password = os.getenv("MAIL_PASSWORD")
    receiver = os.getenv("MAIL_RECEIVER")

    if not sender or not password or not receiver:
        print("Email settings missing in .env")
        return False

    detect_time = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")

    msg = EmailMessage()
    msg["Subject"] = f"URGENT: Criminal Detected — {person_name} @ {station_name}"
    msg["From"]    = sender
    msg["To"]      = receiver

    # ── Email body: person_id / name / station / platform / time
    #    NO confidence score shown ───────────────────────────────────
    msg.set_content(f"""\
URGENT SECURITY ALERT
{'=' * 52}

A known criminal has been detected by the AI Surveillance System.

  Criminal ID     : {person_id}
  Name            : {person_name}
  Station         : {station_name}
  Platform        : {platform_number}
  Detection Time  : {detect_time}
  Status          : WANTED / CRIMINAL

The annotated CCTV frame showing the detection is attached.
Immediate action is required.

{'=' * 52}
This is an automated alert from the AI Surveillance System.
""")

    # ── Attach only the annotated CCTV frame (no face-only image) ────
    if frame_image_path and os.path.exists(frame_image_path):
        with open(frame_image_path, "rb") as f:
            msg.add_attachment(
                f.read(),
                maintype="image",
                subtype="jpeg",
                filename=os.path.basename(frame_image_path)
            )

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(sender, password)
            smtp.send_message(msg)
        print(f"Alert email sent — {person_name}")
        return True

    except Exception as e:
        print("Email sending failed:", e)
        return False
