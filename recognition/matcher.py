import psycopg2
import numpy as np
from scipy.spatial.distance import cosine


# ==========================================
# DATABASE CONFIG  — edit for your machine
# ==========================================
DB_HOST     = "localhost"
DB_NAME     = "criminal_db"
DB_USER     = "postgres"
DB_PASSWORD = "moushikta@1234"
DB_PORT     = "5432"


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
    conn   = get_connection()
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
                "person_id":   person_id,
                "person_name": person_name,
                "embedding":   embedding
            })
        except Exception as e:
            print("Invalid embedding skipped:", person_id, person_name, e)

    cursor.close()
    conn.close()

    print(f"Criminal embeddings loaded: {len(criminal_embeddings)}")
    return criminal_embeddings


# ==========================================
# MATCH ONE FACE EMBEDDING
# ==========================================
def match_criminal(face_embedding, criminal_embeddings,
                   threshold=0.65, min_score_gap=0.07):
    """
    Match a face embedding against the criminal database.

    Parameters
    ----------
    face_embedding      : array-like  — ArcFace embedding from InsightFace
    criminal_embeddings : list of dict loaded by load_criminal_embeddings()
    threshold           : float  — minimum cosine similarity to accept a match
    min_score_gap       : float  — best score must exceed 2nd-best by this margin
                          (prevents ambiguous matches between similar-looking people)

    Returns
    -------
    dict with keys: matched, person_id, person_name, score
    """
    _no_match = {
        "matched":     False,
        "person_id":   None,
        "person_name": "Unknown",
        "score":       0.0
    }

    if face_embedding is None or len(criminal_embeddings) == 0:
        return _no_match

    face_embedding = np.array(face_embedding, dtype=np.float32)

    # Collect all scores so we can check the gap between 1st and 2nd
    scores = []
    for criminal in criminal_embeddings:
        try:
            sim = 1.0 - cosine(face_embedding, criminal["embedding"])
        except Exception:
            sim = 0.0
        scores.append((sim, criminal))

    # Sort descending by score
    scores.sort(key=lambda x: x[0], reverse=True)

    best_score,  best_criminal  = scores[0]
    second_score                = scores[1][0] if len(scores) > 1 else 0.0

    # Must beat the threshold AND the 2nd-best by min_score_gap
    if best_score >= threshold and (best_score - second_score) >= min_score_gap:
        return {
            "matched":     True,
            "person_id":   best_criminal["person_id"],
            "person_name": best_criminal["person_name"],
            "score":       float(best_score)
        }

    return {**_no_match, "score": float(best_score)}


# ==========================================
# TEST
# ==========================================
if __name__ == "__main__":
    criminals = load_criminal_embeddings()
    print("\nLoaded Criminal Data:")
    for c in criminals:
        print(c["person_id"], c["person_name"], c["embedding"].shape)
