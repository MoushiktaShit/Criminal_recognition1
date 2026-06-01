# AI Criminal Recognition System

A real-time CCTV surveillance system that detects known criminals using face recognition, triggers automated alerts, and logs evidence to a database.

---

## Overview

This system processes live or recorded video feeds and identifies wanted individuals by matching detected faces against a database of known criminal embeddings. When a match is confirmed, it sends an email alert with an annotated CCTV frame, saves evidence locally, and logs the event to a PostgreSQL database.

---

## Features

- **Real-time face detection** using InsightFace (`buffalo_l` model)
- **Person tracking** using YOLOv8 + ByteTrack
- **Criminal matching** via cosine similarity on face embeddings
- **Two-tier confidence system** — instant lock on high confidence, voting mechanism for lower confidence
- **Group detection** — captures a group evidence frame when all known criminals appear simultaneously
- **Automated email alerts** with annotated CCTV frames attached
- **PostgreSQL logging** of every confirmed detection
- **On-screen suspicious log panel** showing face thumbnails, name, station, platform, and detection time
- **Evidence saved locally** as annotated JPEG frames with details banner

---

## Project Structure

```
Criminal_Recognition/
│
├── main.py                  # Main detection + tracking loop
├── alert_email.py           # Email alert module
├── .env                     # Environment configuration (see setup)
│
├── recognition/
│   └── matcher.py           # load_criminal_embeddings(), match_criminal()
│
├── criminal_evidence/
│   ├── faces/               # (reserved — face crops not saved by default)
│   ├── frames/              # Annotated CCTV frames per detection
│   └── group/               # Group detection frames
│
├── yolov8n.pt               # YOLOv8 nano model weights
├── bytetrack.yaml           # ByteTrack tracker config
└── .insightface/            # InsightFace model cache
```

---

## Requirements

### Python Packages

```
opencv-python
numpy
ultralytics
insightface
psycopg2-binary
python-dotenv
```

Install all dependencies:

```bash
pip install opencv-python numpy ultralytics insightface psycopg2-binary python-dotenv
```

> InsightFace may require additional system libraries. On Windows, install Visual C++ Build Tools. On Linux, install `libgl1` and `libglib2.0`.

### Models

- **YOLOv8n** — downloaded automatically by Ultralytics on first run (`yolov8n.pt`)
- **InsightFace buffalo_l** — downloaded automatically to the path set in `INSIGHTFACE_ROOT`

---

## Configuration

Copy `_env` to `.env` in the project root and fill in your values:

```env
# PostgreSQL database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=criminal_db
DB_USER=postgres
DB_PASS=your_db_password

# Email alert settings (Gmail)
MAIL_SENDER=your_sender@gmail.com
MAIL_PASSWORD=your_gmail_app_password
MAIL_RECEIVER=receiver@gmail.com

# Deployment info (shown in alerts and on-screen panel)
STATION_NAME=Sealdah Metro Station
PLATFORM_NUMBER=12
```

> **Gmail note:** Use an [App Password](https://support.google.com/accounts/answer/185833), not your regular Gmail password. 2-Step Verification must be enabled on the sender account.

---

## Database Setup

Create the PostgreSQL database and table before running:

```sql
CREATE DATABASE criminal_db;

\c criminal_db

CREATE TABLE criminal_alerts (
    id              SERIAL PRIMARY KEY,
    person_id       VARCHAR(50),
    person_name     VARCHAR(100),
    confidence      FLOAT,
    station_name    VARCHAR(100),
    platform_number VARCHAR(20),
    face_image      TEXT,
    frame_image     TEXT,
    email_sent      BOOLEAN,
    detected_at     TIMESTAMP DEFAULT NOW()
);
```

---

## Criminal Embeddings

Place criminal face embeddings in the format expected by `recognition/matcher.py`. The `load_criminal_embeddings()` function should return a list/dict of entries containing:

- `person_id` — unique identifier
- `person_name` — display name
- `embedding` — 512-dimensional numpy array (InsightFace `buffalo_l` output)

Embeddings are typically pre-generated from reference face images and stored as `.npy` files or in the database.

---

## Running the System

1. Set the video source and output paths in `main.py`:

```python
VIDEO_PATH  = r"path/to/your/video.mp4"   # or 0 for webcam
OUTPUT_PATH = r"path/to/output.mp4"
INSIGHTFACE_ROOT = r"path/to/.insightface"
```

2. Run:

```bash
python main.py
```

3. Press `Q` to stop the live window.

---

## Detection Logic

| Stage | Threshold | Behaviour |
|---|---|---|
| Face ignored | below `LOW_CONF_THRESHOLD` (0.52) | No action |
| Vote accumulation | 0.52 – 0.78 | Face votes added to a sliding window; alert fires after `CONFIRM_FRAMES_NEEDED` (3) votes within `CONFIRM_WINDOW` (12) frames |
| Instant lock | above `HIGH_CONF_THRESHOLD` (0.78) | Alert fires immediately |

Once a criminal is confirmed, they are added to `alerted_persons` — subsequent detections of the same individual do not re-trigger alerts.

---

## Alert Output

On confirmed detection, the system:

1. **Saves** an annotated JPEG to `criminal_evidence/frames/` — full CCTV frame with red bounding box, name label, and a details banner showing name, station, platform, detection time, confidence, and status.
2. **Adds** a card to the on-screen suspicious log panel (left side of the display window).
3. **Sends** an email to `MAIL_RECEIVER` with the annotated frame attached.
4. **Logs** the alert to the `criminal_alerts` PostgreSQL table.

### Group Alert

When all known criminals (defined in `KNOWN_CRIMINALS`) appear in the same frame simultaneously, a group evidence image is saved to `criminal_evidence/group/` with bounding boxes for each individual and a group alert banner.

---

## Key Parameters

All tunable parameters are at the top of `main.py`:

| Parameter | Default | Description |
|---|---|---|
| `FRAME_SKIP` | 2 | Run face detection every N frames |
| `HIGH_CONF_THRESHOLD` | 0.78 | Instant alert threshold |
| `LOW_CONF_THRESHOLD` | 0.52 | Minimum score to enter vote queue |
| `MIN_SCORE_GAP` | 0.07 | Minimum gap between top-2 match scores |
| `MIN_FACE_AREA` | 800 px² | Minimum face area to process |
| `CONFIRM_FRAMES_NEEDED` | 3 | Votes required to confirm via voting |
| `CONFIRM_WINDOW` | 12 | Sliding window size for vote accumulation |
| `TRACK_LOST_BUFFER` | 45 frames | Frames before a lost track is removed |

---

## Privacy & Security Notes

- **Person IDs are not displayed** in the live video feed or email body — shown only in the on-screen log panel and database.
- **Raw face-crop images are not sent** via email; only the full annotated CCTV frame is attached.
- **Confidence scores are not shown** on the suspicious log panel cards.
- Store `.env` securely and never commit it to version control.
- Restrict database access to trusted hosts only.
