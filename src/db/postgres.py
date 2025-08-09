
import psycopg2

class PostgresDB:
    def __init__(self, config: dict):
        self.conn_params = {
            "dbname": config.get("DB_NAME"),
            "user": config.get("DB_USER"),
            "password": config.get("DB_PASSWORD"),
            "host": config.get("DB_HOST", "localhost"),
            "port": int(config.get("DB_PORT", 5432)),
        }

    def _connect(self):
        """Create a new database connection."""
        return psycopg2.connect(**self.conn_params)
    
    def get_similarity(self, face_embedding, limit):
        conn = self._connect()
        cursor = conn.cursor()

        try:
            query = """
                SELECT person_id, person_name,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM person
                ORDER BY similarity DESC
                LIMIT %s
            """
            cursor.execute(query, (face_embedding.tolist(), limit))
            return cursor.fetchall()
        
        except Exception as e:
            print("Similarity search failed:", e)
            raise
        finally:
            cursor.close()
            conn.close()
