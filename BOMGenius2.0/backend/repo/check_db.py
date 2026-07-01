import os
import sqlite3

# Always point to backend/bomgenius.db regardless of where you run from
HERE = os.path.dirname(os.path.abspath(__file__))          # ...\backend\repo
BACKEND_DIR = os.path.dirname(HERE)                        # ...\backend
DB_PATH = os.path.join(BACKEND_DIR, "bomgenius.db")

print("DB PATH USED:", DB_PATH)

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
tables = [r[0] for r in cur.fetchall()]
print("TABLES:", tables)

cur.execute("SELECT MAX(timestamp) FROM mbom")
latest = cur.fetchone()[0]
print("LATEST TIMESTAMP:", latest)

cur.execute("""
SELECT Confidence_Score, typeof(Confidence_Score)
FROM mbom
WHERE timestamp = (SELECT MAX(timestamp) FROM mbom)
LIMIT 20
""")
rows = cur.fetchall()
print("SAMPLE ROWS (confidence, type):")
for r in rows:
    print(r)

conn.close()