import warnings
warnings.filterwarnings("ignore")

import cv2
from ultralytics import YOLO
from insightface.app import FaceAnalysis

from recognition.matcher import load_criminal_embeddings, match_criminal


VIDEO_PATH = r"E:\Criminal_Recognition\sahil&shew1 (online-video-cutter.com).mp4"
OUTPUT_PATH = r"E:\Criminal_Recognition\output_result3.mp4"

YOLO_MODEL = "yolov8n.pt"

FACE_MATCH_THRESHOLD = 0.50

# identity lock memory
identity_lock = {}

# keep locked identity even if track disappears shortly
TRACK_LOST_BUFFER = 80


print("Loading YOLO...")
yolo_model = YOLO(YOLO_MODEL)

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

criminal_embeddings = load_criminal_embeddings()


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


while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1
    visible_track_ids = set()

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

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if box.id is None:
                continue

            track_id = int(box.id[0])
            visible_track_ids.add(track_id)

            label = "Unknown"
            color = (0, 255, 255)

            # ===============================
            # If identity already locked
            # ===============================
            if track_id in identity_lock:
                locked = identity_lock[track_id]

                label = (
                    f'{locked["person_id"]} | '
                    f'{locked["person_name"]} | '
                    f'{locked["score"]:.2f}'
                )

                color = (0, 0, 255)
                criminal_visible = True

                identity_lock[track_id]["last_seen"] = frame_count

            else:
                # ===============================
                # First-time face matching
                # ===============================
                for face in faces:

                    fx1, fy1, fx2, fy2 = map(int, face.bbox)

                    face_cx = (fx1 + fx2) // 2
                    face_cy = (fy1 + fy2) // 2

                    # face must belong to this person box
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

                            label = (
                                f'{result["person_id"]} | '
                                f'{result["person_name"]} | '
                                f'{result["score"]:.2f}'
                            )

                            color = (0, 0, 255)
                            criminal_visible = True

                        break

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            cv2.putText(
                frame,
                label,
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                color,
                2
            )

    # ===============================
    # Remove old lost tracks
    # ===============================
    remove_ids = []

    for tid, data in identity_lock.items():
        if frame_count - data["last_seen"] > TRACK_LOST_BUFFER:
            remove_ids.append(tid)

    for tid in remove_ids:
        identity_lock.pop(tid, None)

    # ===============================
    # Criminal alert
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