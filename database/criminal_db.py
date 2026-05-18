import numpy as np

from database.db import cursor

class CriminalDatabase:

    def fetch_all(self):

        query = """
        SELECT name, embedding
        FROM criminals
        """

        cursor.execute(query)

        rows = cursor.fetchall()

        embeddings = []
        names = []

        for row in rows:

            name = row[0]

            embedding = np.array(
                row[1],
                dtype=np.float32
            )

            names.append(name)

            embeddings.append(embedding)

        return embeddings, names