# database/db.py
import os
import psycopg2
from dotenv import load_dotenv

# Load environment variables from the .env file at the root
load_dotenv()

try:
    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME", "criminal_db"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASS", "moushikta@1234"),
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432")
    )
    cursor = conn.cursor()
    print("PostgreSQL Connected Successfully")
except psycopg2.OperationalError as e:
    print(f"Database connection failed: {e}")
    exit(1)