import psycopg2
import numpy as np
from scipy.spatial.distance import cosine


# ==========================================
# DATABASE CONFIG
# ==========================================
DB_HOST = "localhost"
DB_NAME = "criminal_db"
DB_USER = "postgres"
DB_PASSWORD = "moushikta@1234"
DB_PORT = "5432"


# ==========================================
# DATABASE CONNECTION
# ==========================================
def get_connection():
    return psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        port=DB_PORT
    )


# ==========================================
# LOAD CRIMINAL EMBEDDINGS FROM POSTGRESQL
# ==========================================
def load_criminal_embeddings():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT person_id, person_name, embedding
        FROM criminals
        WHERE embedding IS NOT NULL
        """
    )

    rows = cursor.fetchall()

    criminal_embeddings = []

    for person_id, person_name, embedding_text in rows:

        try:
            embedding = np.array(
                list(map(float, embedding_text.split(","))),
                dtype=np.float32
            )

            criminal_embeddings.append({
                "person_id": person_id,
                "person_name": person_name,
                "embedding": embedding
            })

        except Exception as e:
            print("Invalid embedding skipped:", person_id, person_name)
            print(e)

    cursor.close()
    conn.close()

    print("Criminal embeddings loaded:", len(criminal_embeddings))

    return criminal_embeddings


# ==========================================
# MATCH ONE FACE EMBEDDING
# ==========================================
def match_criminal(face_embedding, criminal_embeddings, threshold=0.65):
    """
    Returns:
        matched True/False
        person_id
        person_name
        score
    """

    if face_embedding is None:
        return {
            "matched": False,
            "person_id": None,
            "person_name": "Unknown",
            "score": 0.0
        }

    if len(criminal_embeddings) == 0:
        return {
            "matched": False,
            "person_id": None,
            "person_name": "Unknown",
            "score": 0.0
        }

    face_embedding = np.array(face_embedding, dtype=np.float32)

    best_score = -1.0
    best_person = None

    for criminal in criminal_embeddings:

        db_embedding = criminal["embedding"]

        try:
            similarity = 1 - cosine(face_embedding, db_embedding)
        except Exception:
            continue

        if similarity > best_score:
            best_score = similarity
            best_person = criminal

    if best_person is not None and best_score >= threshold:
        return {
            "matched": True,
            "person_id": best_person["person_id"],
            "person_name": best_person["person_name"],
            "score": float(best_score)
        }

    return {
        "matched": False,
        "person_id": None,
        "person_name": "Unknown",
        "score": float(best_score)
    }


# ==========================================
# TEST MATCHER FILE
# ==========================================
if __name__ == "__main__":
    criminals = load_criminal_embeddings()

    print("\nLoaded Criminal Data:")
    for c in criminals:
        print(c["person_id"], c["person_name"], c["embedding"].shape)