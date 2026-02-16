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
            Item_Type        TEXT,
            Confidence_Score REAL,
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

    if "Item_Type" not in columns:
        cursor.execute("ALTER TABLE mbom ADD COLUMN Item_Type TEXT")

    if "Confidence_Score" not in columns:
        cursor.execute("ALTER TABLE mbom ADD COLUMN Confidence_Score REAL")

    if "timestamp" not in columns:
        cursor.execute("ALTER TABLE mbom ADD COLUMN timestamp DATETIME DEFAULT CURRENT_TIMESTAMP")
    
    conn.execute("ALTER TABLE mbom ADD COLUMN Item_Type TEXT;")
    conn.execute("ALTER TABLE mbom ADD COLUMN Confidence_Score REAL;")

    conn.commit()
    conn.close()


from datetime import datetime

def save_mbom(df):
    import sqlite3

    conn = sqlite3.connect(DB_NAME)

    # 🔹 Get DB table columns
    cursor = conn.execute("PRAGMA table_info(mbom);")
    db_columns = [row[1] for row in cursor.fetchall()]

    # 🔹 Keep only columns that exist in DB
    df = df.loc[:, df.columns.intersection(db_columns)]

    # 🔹 Add any missing DB columns as None
    for col in db_columns:
        if col not in df.columns:
            df[col] = None

    # 🔹 Reorder to match DB exactly
    df = df[db_columns]

    # 🔹 Insert safely
    df.to_sql("mbom", conn, if_exists="append", index=False)

    conn.close()