import sqlite3

DB_NAME = "bomgenius.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    conn.execute("""
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
    conn.close()

def save_mbom(df):
    with sqlite3.connect(DB_NAME) as conn:
        df.to_sql("mbom", conn, if_exists="replace", index=False)

def fetch_mbom():
    with sqlite3.connect(DB_NAME) as conn:
        return conn.execute("SELECT * FROM mbom").fetchall()