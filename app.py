import warnings
warnings.filterwarnings("ignore")

import cv2
from ultralytics import YOLO
from insightface.app import FaceAnalysis

from recognition.matcher import load_criminal_embeddings, match_criminal


# ===============================
# PATH SETTINGS
# ===============================
VIDEO_PATH = r"E:\Criminal_Recognition\WhatsApp Video 2026-05-14 at 09.32.53.mp4"
OUTPUT_PATH = r"E:\Criminal_Recognition\output_result3.mp4"

YOLO_MODEL = "yolov8n.pt"

FACE_MATCH_THRESHOLD = 0.50

# keep locked identity even if confidence goes down
identity_lock = {}

# keep identity after track disappears shortly
TRACK_LOST_BUFFER = 80


# ===============================
# LOAD YOLO
# ===============================
print("Loading YOLO...")
yolo_model = YOLO(YOLO_MODEL)


# ===============================
# LOAD INSIGHTFACE
# ===============================
print("Loading SCRFD + ArcFace...")
face_app = FaceAnalysis(
    name="buffalo_l",
    root=r"E:\Criminal_Recognition\.insightface",
    providers=["CPUExecutionProvider"]
)

face_app.prepare(
    ctx_id=-1,
    det_size=(1280, 1280),
    det_thresh=0.20
)


# ===============================
# LOAD CRIMINAL EMBEDDINGS
# ===============================
criminal_embeddings = load_criminal_embeddings()


# ===============================
# VIDEO SETUP
# ===============================
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Could not open video:", VIDEO_PATH)
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if fps == 0:
    fps = 25

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    fps,
    (width, height)
)

frame_count = 0


# ===============================
# DRAW CRIMINAL BOX ONLY
# ===============================
def draw_criminal_box(frame, x1, y1, x2, y2, label):
    red = (0, 0, 255)
    white = (255, 255, 255)

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        red,
        3
    )

    cv2.rectangle(
        frame,
        (x1, max(0, y1 - 35)),
        (x2, y1),
        red,
        -1
    )

    cv2.putText(
        frame,
        label,
        (x1 + 5, max(25, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        white,
        2
    )


# ===============================
# MAIN LOOP
# ===============================
while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    results = yolo_model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],
        conf=0.30,
        iou=0.5,
        verbose=False
    )

    faces = face_app.get(frame)

    criminal_visible = False

    if results[0].boxes is not None:

        for box in results[0].boxes:

            if box.id is None:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            track_id = int(box.id[0])

            criminal_label = None

            # ===============================
            # CASE 1: IDENTITY ALREADY LOCKED
            # ===============================
            if track_id in identity_lock:

                locked = identity_lock[track_id]

                criminal_label = (
                    f'{locked["person_id"]} | '
                    f'{locked["person_name"]} | '
                    f'{locked["score"]:.2f}'
                )

                identity_lock[track_id]["last_seen"] = frame_count
                criminal_visible = True

            else:
                # ===============================
                # CASE 2: FIRST TIME FACE MATCHING
                # Unknown persons are checked internally
                # but not drawn on screen
                # ===============================
                for face in faces:

                    fx1, fy1, fx2, fy2 = map(int, face.bbox)

                    face_cx = (fx1 + fx2) // 2
                    face_cy = (fy1 + fy2) // 2

                    # face must be inside person bbox
                    if x1 <= face_cx <= x2 and y1 <= face_cy <= y2:

                        result = match_criminal(
                            face.embedding,
                            criminal_embeddings,
                            threshold=FACE_MATCH_THRESHOLD
                        )

                        if result["matched"]:

                            identity_lock[track_id] = {
                                "person_id": result["person_id"],
                                "person_name": result["person_name"],
                                "score": result["score"],
                                "last_seen": frame_count
                            }

                            criminal_label = (
                                f'{result["person_id"]} | '
                                f'{result["person_name"]} | '
                                f'{result["score"]:.2f}'
                            )

                            criminal_visible = True

                        break

            # ===============================
            # DRAW ONLY CRIMINAL
            # ===============================
            if criminal_label is not None:
                draw_criminal_box(
                    frame,
                    x1,
                    y1,
                    x2,
                    y2,
                    criminal_label
                )

            # IMPORTANT:
            # No else block here.
            # Unknown person is processed internally
            # but no yellow bbox is drawn.

    # ===============================
    # REMOVE OLD LOST LOCKED TRACKS
    # ===============================
    remove_ids = []

    for tid, data in identity_lock.items():
        if frame_count - data["last_seen"] > TRACK_LOST_BUFFER:
            remove_ids.append(tid)

    for tid in remove_ids:
        identity_lock.pop(tid, None)

    # ===============================
    # CRIMINAL ALERT TEXT
    # ===============================
    if criminal_visible:
        cv2.putText(
            frame,
            "CRIMINAL ALERT",
            (width // 2 - 260, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.8,
            (0, 0, 255),
            5
        )

    out.write(frame)

    cv2.imshow("Criminal Recognition System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
out.release()
cv2.destroyAllWindows()

print("Video processing completed")
print("Output saved:", OUTPUT_PATH)