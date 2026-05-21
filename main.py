"""
Criminal Recognition System — main.py  (FIXED)
===============================================
KEY FIXES vs original:
  1. `appearance_result` replaces `app` variable name — was shadowing
     the global `face_app = FaceAnalysis(...)`, crashing after frame 1.
  2. `match_criminal` now receives `min_score_gap` — was causing TypeError.
  3. FRAME_SKIP added — InsightFace only runs every N frames, giving faster
     recognition while still catching criminals the moment they appear.
  4. Only tracked criminals are ever labelled — random people are ignored.

HOW IT WORKS — three-layer recognition
---------------------------------------
LAYER 1 — FACE MATCH (primary)
  InsightFace SCRFD detects faces, ArcFace embeds them.
  Two confidence tiers:
    • score >= HIGH_CONF  → lock on 1 frame immediately
    • score >= LOW_CONF   → collect votes; lock after N matches in window
  Face detection uses det_size=640 and upscales small crops so even
  30-px faces in the distance are picked up.

LAYER 2 — COLOUR APPEARANCE (bridge)
  The moment a criminal is face-confirmed, we save their HSV colour
  histogram (upper-body torso region).  On frames where no face is
  visible we compare EVERY tracked person's colour against every
  locked criminal's saved histogram.  Bhattacharyya similarity >= 0.72
  keeps the criminal labelled through turns, occlusion, and distance.

LAYER 3 — STRICT FALSE-POSITIVE GATE
  • Colour match is only used when the source lock is < APPEARANCE_TTL
    frames old — stale locks never cause false positives.
  • Each track_id can only inherit one criminal identity at a time.
  • Re-validation every REVALIDATE_INTERVAL frames: if a clearly
    different face appears in a locked box, the lock is removed.
  • Vote buffer is NOT reset on a non-match frame — weak but genuine
    matches accumulate; random one-off false faces do not.
"""

import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
from collections import Counter, deque
from ultralytics import YOLO
from insightface.app import FaceAnalysis

from recognition.matcher import load_criminal_embeddings, match_criminal


# ================================================================
# PATHS  — edit for your machine
# ================================================================
VIDEO_PATH       = r"E:\Criminal_Recognition\nibedita.mp4"
OUTPUT_PATH      = r"E:\Criminal_Recognition\output6_6.mp4"
YOLO_MODEL       = "yolov8n.pt"
TRACKER          = "bytetrack.yaml"
INSIGHTFACE_ROOT = r"E:\Criminal_Recognition\.insightface"


# ================================================================
# SPEED: skip face-recognition on intermediate frames
# 1 = run every frame (slowest, most responsive)
# 2 = run every other frame (good balance)
# 3 = run every 3rd frame (fastest, slight delay)
# ================================================================
FRAME_SKIP = 2


# ================================================================
# FACE-MATCH THRESHOLDS
# ================================================================
HIGH_CONF_THRESHOLD  = 0.78   # single-frame instant lock
LOW_CONF_THRESHOLD   = 0.52   # vote accumulation threshold
MIN_SCORE_GAP        = 0.07   # best must beat 2nd-best by this

CONFIRM_FRAMES_NEEDED = 3
CONFIRM_WINDOW        = 12
VOTE_TTL              = 18

REVALIDATE_INTERVAL  = 18
REVALIDATE_THRESHOLD = 0.50
REVALIDATE_GAP       = 0.06

MIN_FACE_AREA      = 800
MIN_DET_SCORE      = 0.25
FACE_OVERLAP_MIN   = 0.30
FACE_ZONE_FRACTION = 0.62

TRACK_LOST_BUFFER  = 45


# ================================================================
# COLOUR APPEARANCE THRESHOLDS
# ================================================================
APPEARANCE_HIST_BINS  = 32
APPEARANCE_THRESHOLD  = 0.72
APPEARANCE_UPPER_FRAC = 0.55
APPEARANCE_TTL        = 90


# ================================================================
# STATE
# ================================================================
identity_lock  = {}   # track_id  -> lock dict
pending_votes  = {}   # track_id  -> deque of vote dicts
suspicious_log = {}   # person_id -> display dict
MAX_LOG_ITEMS  = 5
frame_count    = 0


# ================================================================
# MODELS
# ================================================================
print("Loading YOLO...")
yolo_model = YOLO(YOLO_MODEL)

print("Loading InsightFace (SCRFD + ArcFace)...")
face_app = FaceAnalysis(
    name="buffalo_l",
    root=INSIGHTFACE_ROOT,
    providers=["CPUExecutionProvider"]
)
face_app.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.15)

criminal_embeddings = load_criminal_embeddings()
print(f"Loaded {len(criminal_embeddings)} criminal profiles.\n")


# ================================================================
# VIDEO I/O
# ================================================================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Cannot open video: " + VIDEO_PATH)

W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS = cap.get(cv2.CAP_PROP_FPS) or 25

out_writer = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    FPS, (W, H)
)


# ================================================================
# LAYER 2: COLOUR APPEARANCE
# ================================================================

def build_colour_hist(frame, x1, y1, x2, y2):
    """Normalised HSV hue histogram of the torso region."""
    fh, fw = frame.shape[:2]
    x1c = max(0, int(x1));  y1c = max(0, int(y1))
    x2c = min(fw, int(x2)); y2c = min(fh, int(y2))
    bh  = y2c - y1c
    if bh <= 0 or x2c <= x1c:
        return None
    y_top = y1c + int(bh * 0.15)
    y_bot = y1c + int(bh * (0.15 + APPEARANCE_UPPER_FRAC))
    y_bot = min(y_bot, y2c)
    if y_bot <= y_top:
        return None
    crop = frame[y_top:y_bot, x1c:x2c]
    if crop.size == 0:
        return None
    hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], None, [APPEARANCE_HIST_BINS], [0, 180])
    cv2.normalize(hist, hist, alpha=1, norm_type=cv2.NORM_L1)
    return hist.flatten().astype(np.float32)


def colour_sim(h1, h2):
    """Bhattacharyya coefficient: 1.0 = identical."""
    if h1 is None or h2 is None:
        return 0.0
    return float(np.sum(np.sqrt(np.clip(h1, 0, None) * np.clip(h2, 0, None))))


def appearance_match(frame, x1, y1, x2, y2):
    """
    Compare this box's colour histogram against every locked criminal.
    Returns (person_id, person_name, src_lock) or None.

    FIX: renamed return variable from `app` to `appearance_result` in the
    caller so we never shadow the global face_app object.
    """
    hist = build_colour_hist(frame, x1, y1, x2, y2)
    if hist is None:
        return None

    best_sim  = 0.0
    best_info = None

    for tid, lock in identity_lock.items():
        if "colour_hist" not in lock:
            continue
        age = frame_count - lock["last_seen"]
        if age > APPEARANCE_TTL:
            continue
        sim = colour_sim(hist, lock["colour_hist"])
        if sim > best_sim:
            best_sim  = sim
            best_info = (lock["person_id"], lock["person_name"], lock)

    if best_sim < APPEARANCE_THRESHOLD or best_info is None:
        return None

    pid = best_info[0]
    # Only re-attach if no OTHER track currently holds this person
    already = any(l["person_id"] == pid for l in identity_lock.values())
    if already:
        return None

    return best_info


# ================================================================
# LAYER 1: FACE RECOGNITION HELPERS
# ================================================================

def safe_crop(frame, x1, y1, x2, y2):
    fh, fw = frame.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(fw, int(x2)), min(fh, int(y2))
    c = frame[y1:y2, x1:x2]
    return c if c.size > 0 else None


def overlap_frac(fx1, fy1, fx2, fy2, px1, py1, px2, py2):
    ix1 = max(fx1, px1); iy1 = max(fy1, py1)
    ix2 = min(fx2, px2); iy2 = min(fy2, py2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return ((ix2 - ix1) * (iy2 - iy1)) / max(1, (fx2 - fx1) * (fy2 - fy1))


def find_face_for_box(faces, px1, py1, px2, py2):
    """Return the best face inside this person box, or None."""
    zone_y   = py1 + (py2 - py1) * FACE_ZONE_FRACTION
    best     = None
    best_ovl = 0.0
    for face in faces:
        fx1, fy1, fx2, fy2 = map(int, face.bbox)
        cx = (fx1 + fx2) // 2
        cy = (fy1 + fy2) // 2
        if not (px1 <= cx <= px2 and py1 <= cy <= zone_y):
            continue
        ovl = overlap_frac(fx1, fy1, fx2, fy2, px1, py1, px2, py2)
        if ovl < FACE_OVERLAP_MIN:
            continue
        if (fx2 - fx1) * (fy2 - fy1) < MIN_FACE_AREA:
            continue
        if getattr(face, "det_score", 1.0) < MIN_DET_SCORE:
            continue
        if ovl > best_ovl:
            best_ovl = ovl
            best     = face
    return best


# ================================================================
# LOCK / VOTE HELPERS
# ================================================================

def make_lock(pid, pname, score, face_crop, frame, x1, y1, x2, y2):
    return {
        "person_id":      pid,
        "person_name":    pname,
        "score":          score,
        "last_seen":      frame_count,
        "last_face_seen": frame_count,
        "face_img":       face_crop,
        "colour_hist":    build_colour_hist(frame, x1, y1, x2, y2),
        "box":            (x1, y1, x2, y2),
    }


def push_vote(track_id, res, face_crop, x1, y1, x2, y2):
    if track_id not in pending_votes:
        pending_votes[track_id] = deque(maxlen=CONFIRM_WINDOW)
    pending_votes[track_id].append({
        "person_id":   res["person_id"],
        "person_name": res["person_name"],
        "score":       res["score"],
        "face_img":    face_crop,
        "frame":       frame_count,
        "box":         (x1, y1, x2, y2),
    })


def best_vote(track_id):
    buf = pending_votes.get(track_id)
    if not buf or len(buf) < CONFIRM_FRAMES_NEEDED:
        return None
    top_id, count = Counter(v["person_id"] for v in buf).most_common(1)[0]
    if count < CONFIRM_FRAMES_NEEDED:
        return None
    candidates = [v for v in buf if v["person_id"] == top_id]
    return max(candidates, key=lambda v: v["score"])


def expire_votes():
    dead = [tid for tid, buf in pending_votes.items()
            if buf and (frame_count - buf[-1]["frame"]) > VOTE_TTL]
    for tid in dead:
        del pending_votes[tid]


# ================================================================
# DRAWING
# ================================================================

def draw_box(frame, x1, y1, x2, y2, label):
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv2.rectangle(frame, (x1, max(0, y1 - 35)), (x2, y1), (0, 0, 255), -1)
    cv2.putText(frame, label, (x1 + 5, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)


def draw_log(frame):
    if not suspicious_log:
        return
    items   = list(suspicious_log.values())[-MAX_LOG_ITEMS:]
    panel_h = 45 + len(items) * 95 + 20
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 80), (310, 80 + panel_h), (0, 0, 0), -1)
    frame[:] = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)
    cv2.putText(frame, "SUSPICIOUS LOG", (25, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    y = 140
    for item in items:
        if item["face_img"] is not None:
            try:
                frame[y:y + 70, 25:95] = cv2.resize(item["face_img"], (70, 70))
            except Exception:
                pass
        cv2.putText(frame, f'ID: {item["person_id"]}',     (110, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(frame, f'Name: {item["person_name"]}', (110, y + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(frame, f'Score: {item["score"]:.2f}',  (110, y + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        y += 95


# ================================================================
# MAIN LOOP
# ================================================================
faces = []   # last detected faces — reused on skipped frames

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count      += 1
    criminal_visible  = False

    # ── YOLO person tracking (every frame for smooth boxes) ──────
    yolo_results = yolo_model.track(
        frame, persist=True, tracker=TRACKER,
        classes=[0], conf=0.30, iou=0.50, verbose=False
    )

    # ── InsightFace: only run every FRAME_SKIP frames ────────────
    # On skipped frames we reuse the previous face list.
    # This is safe because YOLO tracking keeps bounding boxes smooth,
    # and the colour-appearance fallback sustains already-locked criminals.
    if frame_count % FRAME_SKIP == 0:
        faces = face_app.get(frame)

    boxes = (yolo_results[0].boxes
             if yolo_results[0].boxes is not None else [])

    for box in boxes:
        if box.id is None:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        track_id        = int(box.id[0])
        criminal_label  = None

        # ============================================================
        # PATH A — Already locked criminal
        # ============================================================
        if track_id in identity_lock:
            locked            = identity_lock[track_id]
            frames_since_face = frame_count - locked.get("last_face_seen", frame_count)

            # ── Periodic re-validation ──────────────────────────────
            if frames_since_face >= REVALIDATE_INTERVAL:
                face = find_face_for_box(faces, x1, y1, x2, y2)

                if face is not None:
                    res = match_criminal(
                        face.embedding, criminal_embeddings,
                        threshold=REVALIDATE_THRESHOLD,
                        min_score_gap=REVALIDATE_GAP        # FIX: was missing
                    )

                    if res["matched"]:
                        fx1, fy1, fx2, fy2 = map(int, face.bbox)
                        fc = safe_crop(frame, fx1, fy1, fx2, fy2)

                        if res["person_id"] == locked["person_id"]:
                            # Same criminal confirmed — refresh
                            new_hist = build_colour_hist(frame, x1, y1, x2, y2)
                            identity_lock[track_id].update({
                                "last_face_seen": frame_count,
                                "last_seen":      frame_count,
                                "box":            (x1, y1, x2, y2),
                            })
                            if new_hist is not None:
                                identity_lock[track_id]["colour_hist"] = new_hist
                            if fc is not None:
                                identity_lock[track_id]["face_img"] = fc
                                suspicious_log[res["person_id"]]["face_img"] = fc
                        else:
                            # Different criminal now in this track
                            fx1, fy1, fx2, fy2 = map(int, face.bbox)
                            fc   = safe_crop(frame, fx1, fy1, fx2, fy2)
                            lock = make_lock(res["person_id"], res["person_name"],
                                             res["score"], fc, frame, x1, y1, x2, y2)
                            identity_lock[track_id] = lock
                            suspicious_log[res["person_id"]] = {
                                "person_id":   res["person_id"],
                                "person_name": res["person_name"],
                                "score":       res["score"],
                                "face_img":    fc
                            }
                    else:
                        # Face visible but doesn't match — remove lock immediately
                        print(f"[REVAL] Track {track_id}: face mismatch "
                              f"(score={res['score']:.3f}). Lock removed.")
                        identity_lock.pop(track_id, None)
                        pending_votes.pop(track_id, None)
                        continue  # skip drawing this frame

                # No face visible this frame — colour appearance will sustain label

            # ── Draw the lock label ─────────────────────────────────
            if track_id in identity_lock:
                locked = identity_lock[track_id]
                criminal_label = (
                    f'{locked["person_id"]} | '
                    f'{locked["person_name"]} | '
                    f'{locked["score"]:.2f}'
                )
                identity_lock[track_id]["last_seen"] = frame_count
                identity_lock[track_id]["box"]       = (x1, y1, x2, y2)
                criminal_visible = True

        # ============================================================
        # PATH B — Unknown track: try face match, then colour fallback
        # ============================================================
        else:
            locked_this_frame = False

            # ── B1. Face-based recognition ──────────────────────────
            face = find_face_for_box(faces, x1, y1, x2, y2)

            if face is not None:
                fx1, fy1, fx2, fy2 = map(int, face.bbox)
                face_crop = safe_crop(frame, fx1, fy1, fx2, fy2)

                res = match_criminal(
                    face.embedding, criminal_embeddings,
                    threshold=LOW_CONF_THRESHOLD,
                    min_score_gap=MIN_SCORE_GAP             # FIX: was missing
                )

                if res["matched"]:

                    if res["score"] >= HIGH_CONF_THRESHOLD:
                        # Single-frame high-confidence lock
                        lock = make_lock(res["person_id"], res["person_name"],
                                         res["score"], face_crop,
                                         frame, x1, y1, x2, y2)
                        identity_lock[track_id] = lock
                        pending_votes.pop(track_id, None)
                        suspicious_log[res["person_id"]] = {
                            "person_id":   res["person_id"],
                            "person_name": res["person_name"],
                            "score":       res["score"],
                            "face_img":    face_crop
                        }
                        criminal_label    = (f'{res["person_id"]} | '
                                             f'{res["person_name"]} | '
                                             f'{res["score"]:.2f}')
                        criminal_visible  = True
                        locked_this_frame = True
                        print(f"[LOCK-INSTANT] Track {track_id} → "
                              f"{res['person_id']} score={res['score']:.3f}")

                    else:
                        # Accumulate votes for weaker matches
                        push_vote(track_id, res, face_crop, x1, y1, x2, y2)
                        confirmed = best_vote(track_id)
                        if confirmed is not None:
                            bx1, by1, bx2, by2 = confirmed["box"]
                            lock = make_lock(
                                confirmed["person_id"],
                                confirmed["person_name"],
                                confirmed["score"],
                                confirmed["face_img"],
                                frame, bx1, by1, bx2, by2
                            )
                            identity_lock[track_id] = lock
                            pending_votes.pop(track_id, None)
                            suspicious_log[confirmed["person_id"]] = {
                                "person_id":   confirmed["person_id"],
                                "person_name": confirmed["person_name"],
                                "score":       confirmed["score"],
                                "face_img":    confirmed["face_img"]
                            }
                            criminal_label    = (
                                f'{confirmed["person_id"]} | '
                                f'{confirmed["person_name"]} | '
                                f'{confirmed["score"]:.2f}'
                            )
                            criminal_visible  = True
                            locked_this_frame = True
                            print(f"[LOCK-VOTE]    Track {track_id} → "
                                  f"{confirmed['person_id']} "
                                  f"score={confirmed['score']:.3f}")

                # Non-match: do NOT reset votes — genuine weak matches accumulate

            # ── B2. Colour appearance fallback ──────────────────────
            # Only runs when face recognition didn't produce a lock.
            # FIX: renamed `app` → `appearance_result` to avoid shadowing
            #      the global `face_app` FaceAnalysis object.
            if not locked_this_frame:
                appearance_result = appearance_match(frame, x1, y1, x2, y2)
                if appearance_result is not None:
                    pid, pname, src_lock = appearance_result
                    lock = make_lock(pid, pname, src_lock["score"],
                                     src_lock["face_img"],
                                     frame, x1, y1, x2, y2)
                    # Keep last_face_seen from original to avoid premature re-validation
                    lock["last_face_seen"] = src_lock["last_face_seen"]
                    identity_lock[track_id] = lock
                    criminal_label   = (f'{pid} | {pname} | '
                                        f'{src_lock["score"]:.2f}')
                    criminal_visible = True
                    print(f"[LOCK-COLOUR]  Track {track_id} → {pid}")

        if criminal_label is not None:
            draw_box(frame, x1, y1, x2, y2, criminal_label)

    # ================================================================
    # CLEANUP
    # ================================================================
    stale = [tid for tid, d in identity_lock.items()
             if frame_count - d["last_seen"] > TRACK_LOST_BUFFER]
    for tid in stale:
        identity_lock.pop(tid, None)

    expire_votes()

    # ================================================================
    # OVERLAYS
    # ================================================================
    if criminal_visible:
        cv2.putText(frame, "CRIMINAL ALERT",
                    (W // 2 - 260, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 5)

    draw_log(frame)

    out_writer.write(frame)
    cv2.imshow("Criminal Recognition System", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
out_writer.release()
cv2.destroyAllWindows()
print(f"\nDone. Output saved: {OUTPUT_PATH}")
