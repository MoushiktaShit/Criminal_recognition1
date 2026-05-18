import os
import cv2
import psycopg2
import numpy as np
from insightface.app import FaceAnalysis


# ===============================
# PostgreSQL Connection
# ===============================
conn = psycopg2.connect(
    host="localhost",
    database="criminal_db",
    user="postgres",
    password="moushikta@1234",
    port="5432"
)

cursor = conn.cursor()
print("PostgreSQL Connected Successfully")


# ===============================
# InsightFace Setup
# ===============================
app = FaceAnalysis(
    name="buffalo_l",
    root=r"E:\Criminal_Recognition\.insightface",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=-1,
    det_size=(320, 320),
    det_thresh=0.20
)


# ===============================
# Criminal Images Folder
# ===============================
IMAGE_DIR = r"E:\Criminal_Recognition\data\criminals"


if not os.path.exists(IMAGE_DIR):
    print("Folder not found:", IMAGE_DIR)
    exit()


stored_count = 0
skipped_count = 0


# ===============================
# Read criminal folders
# Example:
# data/criminals/CR001_Krishna/front.png
# ===============================
for person_folder in os.listdir(IMAGE_DIR):

    person_path = os.path.join(IMAGE_DIR, person_folder)

    if not os.path.isdir(person_path):
        continue

    parts = person_folder.split("_", 1)

    if len(parts) != 2:
        print("Invalid folder name:", person_folder)
        skipped_count += 1
        continue

    person_id = parts[0]
    person_name = parts[1]

    print("\nProcessing criminal:", person_id, person_name)

    for img_name in os.listdir(person_path):

        if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        img_path = os.path.join(person_path, img_name)

        print("Reading:", img_path)

        image = cv2.imread(img_path)

        if image is None:
            print("Could not read image:", img_name)
            skipped_count += 1
            continue

        # ===============================
        # Skip extremely blurry images
        # ===============================
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        blur_score = cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()

        if blur_score < 20:
            print("Too blurry, skipped:", img_name)
            skipped_count += 1
            continue

        # ===============================
        # Face Detection
        # ===============================
        faces = app.get(image)

        if len(faces) == 0:
            print("No face detected:", img_name)
            skipped_count += 1
            continue

        # take largest face
        face = max(
            faces,
            key=lambda f: (
                (f.bbox[2] - f.bbox[0]) *
                (f.bbox[3] - f.bbox[1])
            )
        )

        embedding = face.embedding.astype(np.float32)
        embedding_str = ",".join(map(str, embedding))

        # ===============================
        # Store in PostgreSQL
        # ===============================
        cursor.execute(
            """
            INSERT INTO criminals
            (person_id, person_name, image_path, embedding)
            VALUES (%s, %s, %s, %s)
            """,
            (
                person_id,
                person_name,
                img_path,
                embedding_str
            )
        )

        conn.commit()

        stored_count += 1
        print("Stored:", person_id, person_name, img_name)


cursor.close()
conn.close()

print("\nRegistration Finished")
print("Total Stored :", stored_count)
print("Total Skipped:", skipped_count)