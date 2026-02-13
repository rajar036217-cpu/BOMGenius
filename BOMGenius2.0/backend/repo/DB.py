import sqlite3

DB_NAME = "bomgenius.db"

import sqlite3

with sqlite3.connect("bomgenius.db") as conn:
    print(conn.execute("PRAGMA table_info(mbom);").fetchall())


def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Create table if it doesn't exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS mbom (
            Parent_Part_No   TEXT,
            Child_Part_No    TEXT,
            Description      TEXT,
            Qty_Per          REAL,
            UOM              TEXT,
            BOM_Level        INTEGER,
            Op_Sequence      INTEGER,
            Work_Center      TEXT,
            Make_Buy         TEXT,
            Backflush_Ind    TEXT,
            Scrap_Pct        REAL,
            Plant            TEXT,
            Bin_Location     TEXT,
            timestamp        DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS companies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            created_at TEXT,
            last_login TEXT,
            is_active INTEGER DEFAULT 1
        )
        """)
    
    # Check if timestamp column exists
    cursor.execute("PRAGMA table_info(mbom)")
    columns = [col[1] for col in cursor.fetchall()]

    if "timestamp" not in columns:
        cursor.execute("""
            ALTER TABLE mbom
            ADD COLUMN timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        """)

    conn.commit()
    conn.close()


from datetime import datetime

def save_mbom(df):
    df = df.copy()
    df["timestamp"] = datetime.now().isoformat()

    with sqlite3.connect(DB_NAME) as conn:
        df.to_sql("mbom", conn, if_exists="append", index=False)




def fetch_mbom():
    with sqlite3.connect(DB_NAME) as conn:

        return conn.execute("SELECT * FROM mbom").fetchall()

from pydantic import BaseModel
import sqlite3, datetime

class CompanyCreate(BaseModel):
    name: str

@app.post("/company")
def create_company(data: CompanyCreate):

    with sqlite3.connect("bomgenius.db") as conn:
        conn.execute(
            "INSERT INTO companies (name, created_at) VALUES (?, ?)",
            (data.name, datetime.datetime.now().isoformat())
        )

    return {"status": "company created"}
