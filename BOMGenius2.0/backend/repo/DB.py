import sqlite3
from datetime import datetime

DATABASE = "bomgenius.db"


# -------------------------
# Initialize DB
# -------------------------
def init_db():

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

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
            timestamp        TEXT
        )
    """)

    conn.commit()
    conn.close()


# -------------------------
# Add missing columns dynamically
# -------------------------
def ensure_columns(conn, df):

    cursor = conn.cursor()

    cursor.execute("PRAGMA table_info(mbom)")
    existing_columns = [col[1] for col in cursor.fetchall()]

    for column in df.columns:

        if column not in existing_columns:

            print(f"Adding missing column: {column}")

            cursor.execute(
                f'ALTER TABLE mbom ADD COLUMN "{column}" TEXT'
            )

    conn.commit()


# -------------------------
# Save MBOM safely
# -------------------------
def save_mbom(df):

    conn = sqlite3.connect(DATABASE)

    # Add timestamp column
    timestamp = datetime.now().isoformat()
    df["timestamp"] = timestamp

    # Ensure all columns exist
    ensure_columns(conn, df)

    # Save to DB
    df.to_sql(
        "mbom",
        conn,
        if_exists="append",
        index=False
    )

    conn.commit()
    conn.close()

    print("MBOM saved successfully with timestamp:", timestamp)
