import sqlite3
import pandas as pd
from datetime import datetime

DATABASE = "bomgenius.db"


# -------------------------
# Initialize DB
# -------------------------
def init_db():

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()

    # ---- Companies Table (LOGIN TABLE) ----
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS companies (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_name TEXT,
            email TEXT UNIQUE,
            password TEXT,
            created_at TEXT
        )
    """)

    # ---- MBOM Table ----
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS mbom (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company_id INTEGER,
            Parent_Part_No TEXT,
            Child_Part_No TEXT,
            Description TEXT,
            Qty_Per REAL,
            UOM TEXT,
            BOM_Level INTEGER,
            Op_Sequence INTEGER,
            Work_Center TEXT,
            Make_Buy TEXT,
            Backflush_Ind TEXT,
            Scrap_Pct REAL,
            Plant TEXT,
            Bin_Location TEXT,
            Item_Type TEXT,
            Confidence_Score REAL,
            timestamp TEXT,
            FOREIGN KEY (company_id) REFERENCES companies(id)
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
            col_type = "REAL" if pd.api.types.is_numeric_dtype(df[column]) else "TEXT"
            cursor.execute(f'ALTER TABLE mbom ADD COLUMN "{column}" {col_type}')

    conn.commit()

# -------------------------
# Save MBOM safely
# -------------------------
def save_mbom(df, company_id):

    conn = sqlite3.connect(DATABASE)


    # Add timestamp
    timestamp = datetime.now().isoformat()
    df["timestamp"] = timestamp

    # Add company id
    df["company_id"] = company_id


    # Ensure columns exist
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