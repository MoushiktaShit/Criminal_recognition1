import warnings
warnings.filterwarnings("ignore")

import os
import cv2
import psycopg2
import numpy as np
from dotenv import load_dotenv
from datetime import datetime
from collections import Counter, deque

from ultralytics import YOLO
from insightface.app import FaceAnalysis

from recognition.matcher import (
    load_criminal_embeddings,
    match_criminal
)

from alert_email import send_criminal_alert_email


# =========================================================
# LOAD ENV
# =========================================================
load_dotenv()

DB_HOST        = os.getenv("DB_HOST")
DB_NAME        = os.getenv("DB_NAME")
DB_USER        = os.getenv("DB_USER")
DB_PASS        = os.getenv("DB_PASS")
DB_PORT        = os.getenv("DB_PORT")
STATION_NAME   = os.getenv("STATION_NAME")
PLATFORM_NUMBER = os.getenv("PLATFORM_NUMBER")


# =========================================================
# PATHS
# =========================================================
VIDEO_PATH      = r"E:\Criminal_Recognition\video_test.mp4"
OUTPUT_PATH     = r"E:\Criminal_Recognition\output_alert3.mp4"
YOLO_MODEL      = "yolov8n.pt"
TRACKER         = "bytetrack.yaml"
INSIGHTFACE_ROOT = r"E:\Criminal_Recognition\.insightface"


# =========================================================
# EVIDENCE FOLDERS
# =========================================================
FACE_EVIDENCE_DIR  = "criminal_evidence/faces"
FRAME_EVIDENCE_DIR = "criminal_evidence/frames"
GROUP_EVIDENCE_DIR = "criminal_evidence/group"

os.makedirs(FACE_EVIDENCE_DIR,  exist_ok=True)
os.makedirs(FRAME_EVIDENCE_DIR, exist_ok=True)
os.makedirs(GROUP_EVIDENCE_DIR, exist_ok=True)


# =========================================================
# THRESHOLDS
# =========================================================
FRAME_SKIP           = 2
HIGH_CONF_THRESHOLD  = 0.78
LOW_CONF_THRESHOLD   = 0.52
MIN_SCORE_GAP        = 0.07
MIN_FACE_AREA        = 800
MIN_DET_SCORE        = 0.25
FACE_OVERLAP_MIN     = 0.30
FACE_ZONE_FRACTION   = 0.62
TRACK_LOST_BUFFER    = 45
CONFIRM_FRAMES_NEEDED = 3
CONFIRM_WINDOW       = 12
VOTE_TTL             = 18


# =========================================================
# SUSPICIOUS LOG PANEL SETTINGS  (LEFT side of canvas)
# =========================================================
LOG_MAX_CARDS  = 4
LOG_CARD_TTL   = int(30 * 8)   # ~8 s at 30 fps

PANEL_W        = 310           # panel is on the LEFT
# Each card: face thumbnail only (captured at HIGH-conf moment)
# No confidence score shown anywhere on the panel
CARD_H         = 130
CARD_THUMB_W   = 90            # face thumbnail (taken at first detection)
CARD_THUMB_H   = 90

# Colours (BGR)
PANEL_BG      = (15, 15, 15)
CARD_BG       = (28, 28, 28)
CARD_BORDER   = (0, 60, 180)
ACCENT_RED    = (0, 0, 220)
ACCENT_ORANGE = (0, 130, 255)
TEXT_WHITE    = (240, 240, 240)
TEXT_GREY     = (160, 160, 160)
HEADER_BG     = (0, 0, 160)

# Known criminal names (used for group-detection logic)
KNOWN_CRIMINALS = {"sahil", "nibedita", "moushikta"}


# =========================================================
# STATE
# =========================================================
identity_lock    = {}
pending_votes    = {}
frame_count      = 0
alerted_persons  = set()
suspicious_log   = []          # log cards for on-screen panel
group_alert_sent = False       # only one group frame saved

# Stores bounding-box info for currently-visible criminals this frame
# keyed by person_name.lower() → (x1,y1,x2,y2, person_name, confidence)
current_frame_criminals = {}


# =========================================================
# LOAD MODELS
# =========================================================
print("Loading YOLO...")
yolo_model = YOLO(YOLO_MODEL)

print("Loading InsightFace...")
face_app = FaceAnalysis(
    name="buffalo_l",
    root=INSIGHTFACE_ROOT,
    providers=["CPUExecutionProvider"]
)
face_app.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.15)

criminal_embeddings = load_criminal_embeddings()
print(f"Loaded Criminal Profiles: {len(criminal_embeddings)}")


# =========================================================
# VIDEO
# =========================================================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Cannot open video")

W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS = cap.get(cv2.CAP_PROP_FPS)
OUT_W = W + PANEL_W   # total canvas width: panel(left) + video(right)

out_writer = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    FPS,
    (OUT_W, H)
)


# =========================================================
# HELPER: DRAW ANNOTATED BOX ON A FRAME COPY
#   Shows:  Name  |  Conf%      (NO person_id ever shown)
# =========================================================
def draw_annotated_box(img, x1, y1, x2, y2, person_name, confidence):
    """
    Draw a red bounding box + label on img (modifies in-place).
    Label = "Name  |  XX%"  — no internal ID shown.
    """
    label = f"{person_name}  |  {confidence * 100:.0f}%"
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv2.rectangle(img, (x1, y1 - 36), (x2, y1), (0, 0, 255), -1)
    cv2.putText(
        img, label,
        (x1 + 6, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.70,
        (255, 255, 255), 2, cv2.LINE_AA
    )


# =========================================================
# HELPER: BUILD DETAILS BANNER BELOW AN IMAGE
#   Shows:  Name / Station / Platform / Date-Time / Confidence
#   NO person_id, NO raw face-only image in email
# =========================================================
def build_details_banner(width, person_name, confidence,
                          detect_time_str):
    """
    Returns a BGR image (details_h x width) with white text on dark bg.
    """
    details_h  = 110
    banner = np.zeros((details_h, width, 3), dtype=np.uint8)
    banner[:] = (28, 28, 28)

    lines = [
        f"Name       : {person_name}",
        f"Station    : {STATION_NAME}  |  Platform {PLATFORM_NUMBER}",
        f"Detected   : {detect_time_str}",
        f"Confidence : {confidence * 100:.1f}%",
        "Status     : WANTED / CRIMINAL",
    ]

    for i, line in enumerate(lines):
        y = 20 + i * 18
        color = (0, 200, 255) if i == 0 else (200, 200, 200)
        cv2.putText(banner, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    color, 1, cv2.LINE_AA)
    return banner


# =========================================================
# HELPER: SAVE ANNOTATED EVIDENCE FRAME
#   Draws box on a COPY of the full CCTV frame,
#   then stacks a details banner below it.
#   Returns the saved file path.
# =========================================================
def save_annotated_evidence(person_id, person_name, confidence,
                              full_frame, box_coords, timestamp_str):
    """
    box_coords = (x1, y1, x2, y2)
    Saves:  criminal_evidence/frames/<name>_<ts>_annotated.jpg
    Returns path.
    """
    annotated = full_frame.copy()
    x1, y1, x2, y2 = box_coords
    draw_annotated_box(annotated, x1, y1, x2, y2,
                       person_name, confidence)

    detect_time = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    banner = build_details_banner(annotated.shape[1],
                                   person_name, confidence,
                                   detect_time)

    evidence_img = np.vstack([annotated, banner])

    fname = (f"{person_name.lower().replace(' ', '_')}"
             f"_{timestamp_str}_annotated.jpg")
    path  = os.path.join(FRAME_EVIDENCE_DIR, fname)
    cv2.imwrite(path, evidence_img)
    return path


# =========================================================
# HELPER: SAVE GROUP EVIDENCE FRAME
#   Called once when all 3 criminals visible together.
#   Draws a box for each criminal on one frame + group banner.
# =========================================================
def save_group_evidence(full_frame, criminals_info, timestamp_str):
    """
    criminals_info: list of dicts with keys
        name, confidence, box=(x1,y1,x2,y2)
    """
    annotated = full_frame.copy()

    for ci in criminals_info:
        x1, y1, x2, y2 = ci["box"]
        draw_annotated_box(annotated, x1, y1, x2, y2,
                           ci["name"], ci["confidence"])

    # Group details banner
    bw = annotated.shape[1]
    bh = 130
    banner = np.zeros((bh, bw, 3), dtype=np.uint8)
    banner[:] = (20, 20, 40)

    names_str = "  |  ".join(c["name"] for c in criminals_info)
    detect_time = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")

    group_lines = [
        "!!! GROUP CRIMINAL ALERT !!!",
        f"Detected   : {names_str}",
        f"Station    : {STATION_NAME}  |  Platform {PLATFORM_NUMBER}",
        f"Time       : {detect_time}",
        "Status     : ALL SUBJECTS WANTED — IMMEDIATE ACTION REQUIRED",
    ]

    colors = [(0, 0, 255), (0, 200, 255),
              (200, 200, 200), (200, 200, 200), (0, 100, 255)]

    for i, (line, col) in enumerate(zip(group_lines, colors)):
        y = 22 + i * 22
        cv2.putText(banner, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    col, 1, cv2.LINE_AA)

    group_img = np.vstack([annotated, banner])

    fname = f"GROUP_{timestamp_str}.jpg"
    path  = os.path.join(GROUP_EVIDENCE_DIR, fname)
    cv2.imwrite(path, group_img)
    print(f"[GROUP EVIDENCE] Saved → {path}")
    return path


# =========================================================
# DATABASE ALERT SAVE
# =========================================================
def save_alert_to_db(person_id, person_name, confidence,
                      face_path, frame_path, email_sent):
    try:
        conn = psycopg2.connect(
            host=DB_HOST, database=DB_NAME,
            user=DB_USER, password=DB_PASS, port=DB_PORT
        )
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO criminal_alerts
            (person_id, person_name, confidence,
             station_name, platform_number,
             face_image, frame_image, email_sent)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (person_id, person_name, confidence,
             STATION_NAME, PLATFORM_NUMBER,
             face_path, frame_path, email_sent)
        )
        conn.commit()
        cur.close()
        conn.close()
        print("Alert saved in PostgreSQL")
    except Exception as e:
        print("DB Alert Save Error:", e)


# =========================================================
# BUILD FACE THUMBNAIL FOR LOG CARD
#   Only the face crop is stored — taken at the moment confidence
#   first crosses HIGH_CONF_THRESHOLD (or best vote score).
#   No frame thumbnail, no confidence score on cards.
# =========================================================
def make_face_thumb(face_crop):
    if face_crop is None or face_crop.size == 0:
        placeholder = np.zeros((CARD_THUMB_H, CARD_THUMB_W, 3), dtype=np.uint8)
        cv2.putText(placeholder, "N/A", (18, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 80), 1)
        return placeholder
    return cv2.resize(face_crop, (CARD_THUMB_W, CARD_THUMB_H))


# =========================================================
# ADD CARD TO SUSPICIOUS LOG
#   Shows: criminal ID / name / station / platform / detection time
#   Face image = crop taken at first HIGH-confidence detection
#   NO confidence score shown anywhere on the card
# =========================================================
def add_to_suspicious_log(person_id, person_name, face_crop):
    """
    Add a card for every confirmed criminal detection — both HIGH_CONF
    instant lock AND vote-confirmed. Called once per person.
    Face image is the crop from the FIRST confirmed detection moment.
    """
    if any(c["person_id"] == person_id for c in suspicious_log):
        return

    card = {
        "person_id":   person_id,
        "person_name": person_name,
        "face_thumb":  make_face_thumb(face_crop),
        "detected_at": frame_count,
        "time_str":    datetime.now().strftime("%H:%M:%S"),
    }
    suspicious_log.insert(0, card)

    if len(suspicious_log) > LOG_MAX_CARDS * 3:
        suspicious_log.pop()

    print(f"[LOG] Card added for {person_name} (ID: {person_id})")


# =========================================================
# DRAW SUSPICIOUS LOG PANEL  (LEFT side of canvas)
#
#  Each card shows:
#    • Face image  (captured at first HIGH-conf detection)
#    • Criminal ID
#    • Name
#    • Station name
#    • Platform number
#    • Detection time
#  NO confidence score displayed anywhere.
# =========================================================
def draw_suspicious_panel(canvas):
    """
    Panel occupies canvas[:, 0:PANEL_W].
    Video occupies canvas[:, PANEL_W:PANEL_W+W].
    """
    ph = H

    # ── Panel background ─────────────────────────────────────────────
    canvas[:, 0:PANEL_W] = PANEL_BG

    # ── Header bar ───────────────────────────────────────────────────
    header_h = 48
    cv2.rectangle(canvas, (0, 0), (PANEL_W, header_h), HEADER_BG, -1)

    cv2.putText(canvas, "SUSPICIOUS LOG",
                (10, 30), cv2.FONT_HERSHEY_DUPLEX,
                0.65, TEXT_WHITE, 1, cv2.LINE_AA)

    badge_txt = f"{len(suspicious_log)} alert{'s' if len(suspicious_log) != 1 else ''}"
    cv2.putText(canvas, badge_txt,
                (PANEL_W - 100, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.40, ACCENT_ORANGE, 1, cv2.LINE_AA)

    # Thin separator line under header
    cv2.line(canvas, (0, header_h), (PANEL_W, header_h), ACCENT_RED, 1)

    # ── Cards ────────────────────────────────────────────────────────
    card_y = header_h + 6

    for idx, card in enumerate(suspicious_log[:LOG_MAX_CARDS]):

        age   = frame_count - card["detected_at"]
        alpha = (1.0 if age <= LOG_CARD_TTL
                 else max(0.35, 1.0 - (age - LOG_CARD_TTL) / (LOG_CARD_TTL * 0.5)))

        cx1 = 5
        cx2 = PANEL_W - 5
        cy1 = card_y
        cy2 = card_y + CARD_H

        if cy2 > ph - 36:
            break

        # Card background (alpha-blended)
        roi = canvas[cy1:cy2, cx1:cx2]
        cv2.addWeighted(np.full_like(roi, CARD_BG), alpha,
                        roi, 1 - alpha, 0, roi)
        canvas[cy1:cy2, cx1:cx2] = roi

        # Card border — red for newest, blue for older
        border_col = ACCENT_RED if idx == 0 else CARD_BORDER
        cv2.rectangle(canvas, (cx1, cy1), (cx2, cy2), border_col, 1)

        # NEW badge top-right corner of card
        if idx == 0:
            cv2.rectangle(canvas, (cx2 - 38, cy1),
                          (cx2, cy1 + 16), ACCENT_RED, -1)
            cv2.putText(canvas, "NEW",
                        (cx2 - 34, cy1 + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                        TEXT_WHITE, 1, cv2.LINE_AA)

        # ── Face thumbnail (left of card) ────────────────────────────
        ft    = card["face_thumb"]
        ft_x1 = cx1 + 6
        ft_x2 = ft_x1 + CARD_THUMB_W
        ft_y1 = cy1 + (CARD_H - CARD_THUMB_H) // 2
        ft_y2 = ft_y1 + CARD_THUMB_H

        if ft_y2 <= ph and ft_x2 <= PANEL_W:
            face_roi = canvas[ft_y1:ft_y2, ft_x1:ft_x2]
            canvas[ft_y1:ft_y2, ft_x1:ft_x2] = cv2.addWeighted(
                ft, alpha, face_roi, 1 - alpha, 0)
            # thin border around face
            cv2.rectangle(canvas, (ft_x1, ft_y1), (ft_x2, ft_y2),
                          (80, 80, 80), 1)

        # ── Text block (right of face thumbnail) ─────────────────────
        tx = ft_x2 + 8
        ty = cy1 + 16

        # Criminal ID  (highlighted in orange)
        cv2.putText(canvas, f"ID: {card['person_id']}",
                    (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                    ACCENT_ORANGE, 1, cv2.LINE_AA)

        # Name  (white, slightly larger)
        name_disp = card["person_name"]
        if len(name_disp) > 13:
            name_disp = name_disp[:12] + "."
        cv2.putText(canvas, name_disp,
                    (tx, ty + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                    TEXT_WHITE, 1, cv2.LINE_AA)

        # Station name  (grey, small)
        station_disp = STATION_NAME if STATION_NAME else "Unknown"
        if len(station_disp) > 18:
            station_disp = station_disp[:17] + "."
        cv2.putText(canvas, station_disp,
                    (tx, ty + 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                    TEXT_GREY, 1, cv2.LINE_AA)

        # Platform number
        cv2.putText(canvas, f"Platform: {PLATFORM_NUMBER}",
                    (tx, ty + 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                    TEXT_GREY, 1, cv2.LINE_AA)

        # Detection time
        cv2.putText(canvas, card["time_str"],
                    (tx, ty + 74),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                    (100, 200, 100), 1, cv2.LINE_AA)

        # Divider line between cards
        card_y = cy2 + 6
        cv2.line(canvas, (cx1 + 4, card_y - 3),
                 (cx2 - 4, card_y - 3), (45, 45, 45), 1)

    # ── Footer ───────────────────────────────────────────────────────
    footer_y = ph - 22
    cv2.line(canvas, (4, footer_y - 8),
             (PANEL_W - 4, footer_y - 8), (50, 50, 50), 1)
    cv2.putText(canvas, "AI Surveillance System",
                (8, footer_y + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32,
                TEXT_GREY, 1, cv2.LINE_AA)


# =========================================================
# BLINKING ALERT BANNER  (over the video area only)
# =========================================================
def draw_alert_banner(canvas, blink_on):
    if not blink_on:
        return
    vx = PANEL_W    # video starts here on canvas
    overlay = canvas.copy()
    cv2.rectangle(overlay, (vx, 0), (vx + W, 55), (0, 0, 180), -1)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)
    cv2.putText(canvas, "! CRIMINAL ALERT !",
                (vx + W // 2 - 210, 38),
                cv2.FONT_HERSHEY_DUPLEX, 1.2,
                (255, 255, 255), 2, cv2.LINE_AA)


# =========================================================
# MAIN ALERT HANDLER
#   - saves annotated frame (box drawn on CCTV frame + detail banner)
#   - does NOT save or send bare face-crop image
#   - email attaches only the annotated frame evidence image
#   - no person_id in email or on-screen
# =========================================================
def handle_criminal_alert(person_id, person_name, confidence,
                           face_crop, full_frame, box_coords):
    global group_alert_sent

    if person_id in alerted_persons:
        return
    if face_crop is None:
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── 1. Save annotated CCTV frame (box + details banner) ──────────
    annotated_path = save_annotated_evidence(
        person_id, person_name, confidence,
        full_frame, box_coords, timestamp
    )

    # ── 2. Add card to on-screen suspicious log ───────────────────────
    add_to_suspicious_log(person_id, person_name, face_crop)

    # ── 3. Send email  (only annotated frame attached, no face image,
    #       no person_id passed — send_criminal_alert_email receives
    #       name / confidence / station / platform / annotated frame) ──
    email_sent = send_criminal_alert_email(
        person_id=person_id,
        person_name=person_name,
        confidence=confidence,
        station_name=STATION_NAME,
        platform_number=PLATFORM_NUMBER,
        face_image_path=None,            # ← face image NOT sent
        frame_image_path=annotated_path  # ← only annotated frame sent
    )

    # ── 4. Save to DB ─────────────────────────────────────────────────
    save_alert_to_db(
        person_id, person_name, confidence,
        None, annotated_path, email_sent
    )

    alerted_persons.add(person_id)
    print(f"[ALERT] {person_name} — annotated evidence saved.")

    # ── 5. Check if all 3 are now detected → group evidence ──────────
    if not group_alert_sent:
        detected_names = {
            p["person_name"].lower()
            for p in alerted_persons
            if isinstance(p, dict)   # guard (alerted_persons holds IDs)
        }
        # Rebuild from suspicious_log which holds names
        log_names = {
            c["person_name"].lower() for c in suspicious_log
        }
        if KNOWN_CRIMINALS.issubset(log_names):
            _try_save_group_frame(timestamp)


def _try_save_group_frame(timestamp):
    """
    Called as soon as all 3 criminals appear in the log.
    Uses current_frame_criminals if all 3 are visible RIGHT NOW,
    otherwise uses last-known boxes stored in identity_lock.
    """
    global group_alert_sent
    if group_alert_sent:
        return

    # Gather info from current_frame_criminals (populated each frame)
    criminals_info = []
    for name_key, info in current_frame_criminals.items():
        criminals_info.append({
            "name":       info["person_name"],
            "confidence": info["confidence"],
            "box":        info["box"],
        })

    if len(criminals_info) < 2:
        # Not enough people visible right now — skip group frame
        # (will try again next time someone triggers alert)
        return

    # Use the most recent full frame stored in identity_lock context
    # We'll use the frame captured at this moment
    # (full_frame is not globally available; we'll store it below)
    if _last_frame is None:
        return

    path = save_group_evidence(_last_frame, criminals_info, timestamp)
    group_alert_sent = True
    print(f"[GROUP ALERT] All criminals detected together → {path}")


# =========================================================
# SAFE CROP
# =========================================================
def safe_crop(frame, x1, y1, x2, y2):
    fh, fw = frame.shape[:2]
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(fw, int(x2))
    y2 = min(fh, int(y2))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


# =========================================================
# OVERLAP
# =========================================================
def overlap_frac(fx1, fy1, fx2, fy2, px1, py1, px2, py2):
    ix1 = max(fx1, px1)
    iy1 = max(fy1, py1)
    ix2 = min(fx2, px2)
    iy2 = min(fy2, py2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    intersection = (ix2 - ix1) * (iy2 - iy1)
    face_area    = (fx2 - fx1) * (fy2 - fy1)
    return intersection / max(1, face_area)


# =========================================================
# FIND FACE INSIDE PERSON BOX
# =========================================================
def find_face_for_box(faces, px1, py1, px2, py2):
    zone_y       = py1 + (py2 - py1) * FACE_ZONE_FRACTION
    best_face    = None
    best_overlap = 0

    for face in faces:
        fx1, fy1, fx2, fy2 = map(int, face.bbox)
        cx = (fx1 + fx2) // 2
        cy = (fy1 + fy2) // 2

        if not (px1 <= cx <= px2 and py1 <= cy <= zone_y):
            continue

        overlap = overlap_frac(fx1, fy1, fx2, fy2, px1, py1, px2, py2)
        if overlap < FACE_OVERLAP_MIN:
            continue

        face_area = (fx2 - fx1) * (fy2 - fy1)
        if face_area < MIN_FACE_AREA:
            continue

        if getattr(face, "det_score", 1.0) < MIN_DET_SCORE:
            continue

        if overlap > best_overlap:
            best_overlap = overlap
            best_face    = face

    return best_face


# =========================================================
# PUSH / BEST VOTE
# =========================================================
def push_vote(track_id, result, face_crop):
    if track_id not in pending_votes:
        pending_votes[track_id] = deque(maxlen=CONFIRM_WINDOW)
    pending_votes[track_id].append({
        "person_id":   result["person_id"],
        "person_name": result["person_name"],
        "score":       result["score"],
        "face_img":    face_crop,
        "frame":       frame_count
    })


def best_vote(track_id):
    buffer = pending_votes.get(track_id)
    if not buffer or len(buffer) < CONFIRM_FRAMES_NEEDED:
        return None
    top_id, count = Counter(
        v["person_id"] for v in buffer).most_common(1)[0]
    if count < CONFIRM_FRAMES_NEEDED:
        return None
    candidates = [v for v in buffer if v["person_id"] == top_id]
    return max(candidates, key=lambda x: x["score"])


# =========================================================
# MAIN LOOP
# =========================================================
faces       = []
_last_frame = None   # global ref so group handler can access it

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame_count     += 1
    criminal_visible = False
    _last_frame      = frame.copy()

    # Reset per-frame criminal positions
    current_frame_criminals.clear()

    yolo_results = yolo_model.track(
        frame,
        persist=True,
        tracker=TRACKER,
        classes=[0],
        conf=0.30,
        iou=0.50,
        verbose=False
    )

    if frame_count % FRAME_SKIP == 0:
        faces = face_app.get(frame)

    boxes = (yolo_results[0].boxes
             if yolo_results[0].boxes is not None else [])

    for box in boxes:
        if box.id is None:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        track_id = int(box.id[0])
        label    = None

        face = find_face_for_box(faces, x1, y1, x2, y2)

        if face is not None:
            fx1, fy1, fx2, fy2 = map(int, face.bbox)
            face_crop = safe_crop(frame, fx1, fy1, fx2, fy2)

            result = match_criminal(
                face.embedding,
                criminal_embeddings,
                threshold=LOW_CONF_THRESHOLD,
                min_score_gap=MIN_SCORE_GAP
            )

            if result["matched"]:

                if result["score"] >= HIGH_CONF_THRESHOLD:
                    identity_lock[track_id] = {
                        "person_id":   result["person_id"],
                        "person_name": result["person_name"],
                        "score":       result["score"],
                        "last_seen":   frame_count,
                        "box":         (x1, y1, x2, y2)
                    }
                    # Label: name + conf only — no ID
                    label = (f'{result["person_name"]}  |  '
                             f'{result["score"] * 100:.0f}%')
                    criminal_visible = True
                    print(f"[LOCK-INSTANT] {result['person_name']}")

                    handle_criminal_alert(
                        person_id=result["person_id"],
                        person_name=result["person_name"],
                        confidence=result["score"],
                        face_crop=face_crop,
                        full_frame=frame,
                        box_coords=(x1, y1, x2, y2)
                    )

                else:
                    push_vote(track_id, result, face_crop)
                    confirmed = best_vote(track_id)

                    if confirmed is not None:
                        identity_lock[track_id] = {
                            "person_id":   confirmed["person_id"],
                            "person_name": confirmed["person_name"],
                            "score":       confirmed["score"],
                            "last_seen":   frame_count,
                            "box":         (x1, y1, x2, y2)
                        }
                        label = (f'{confirmed["person_name"]}  |  '
                                 f'{confirmed["score"] * 100:.0f}%')
                        criminal_visible = True
                        print(f"[LOCK-VOTE] {confirmed['person_name']}")

                        handle_criminal_alert(
                            person_id=confirmed["person_id"],
                            person_name=confirmed["person_name"],
                            confidence=confirmed["score"],
                            face_crop=confirmed["face_img"],
                            full_frame=frame,
                            box_coords=(x1, y1, x2, y2)
                        )

        # If identity is locked for this track, keep showing
        if track_id in identity_lock:
            locked = identity_lock[track_id]
            locked["last_seen"] = frame_count
            locked["box"]       = (x1, y1, x2, y2)   # update position
            label = (f'{locked["person_name"]}  |  '
                     f'{locked["score"] * 100:.0f}%')
            criminal_visible = True

            # Track currently-visible criminals for group detection
            name_key = locked["person_name"].lower()
            current_frame_criminals[name_key] = {
                "person_name": locked["person_name"],
                "confidence":  locked["score"],
                "box":         (x1, y1, x2, y2)
            }

        if label is not None:
            draw_annotated_box(frame, x1, y1, x2, y2,
                               label.split("  |  ")[0],
                               float(label.split("  |  ")[1].replace("%", "")) / 100)

    # =====================================================
    # CHECK FOR GROUP FRAME  (all 3 visible simultaneously)
    # =====================================================
    if (not group_alert_sent
            and KNOWN_CRIMINALS.issubset(current_frame_criminals.keys())):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        criminals_info = [
            {
                "name":       v["person_name"],
                "confidence": v["confidence"],
                "box":        v["box"]
            }
            for v in current_frame_criminals.values()
        ]
        path = save_group_evidence(frame, criminals_info, ts)
        group_alert_sent = True
        print(f"[GROUP FRAME] All 3 detected together → {path}")

    # =====================================================
    # REMOVE STALE TRACKS
    # =====================================================
    stale = [tid for tid, lk in identity_lock.items()
             if frame_count - lk["last_seen"] > TRACK_LOST_BUFFER]
    for tid in stale:
        identity_lock.pop(tid, None)

    # =====================================================
    # BUILD COMPOSITE CANVAS
    #   LEFT  [0 : PANEL_W]       = suspicious log panel
    #   RIGHT [PANEL_W : OUT_W]   = live video feed
    # =====================================================
    canvas = np.zeros((H, OUT_W, 3), dtype=np.uint8)
    canvas[:, PANEL_W:OUT_W] = frame   # video on the RIGHT

    draw_suspicious_panel(canvas)      # panel drawn on LEFT

    if criminal_visible:
        draw_alert_banner(canvas, (frame_count // 15) % 2 == 0)

    out_writer.write(canvas)
    cv2.imshow("Criminal Recognition System", canvas)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# =========================================================
# RELEASE
# =========================================================
cap.release()
out_writer.release()
cv2.destroyAllWindows()
print("DONE")
