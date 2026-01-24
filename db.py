import sqlite3

# Create / Connect Database
conn = sqlite3.connect("optibom.db")
cursor = conn.cursor()

# Federated Learning Table
cursor.execute("""
CREATE TABLE IF NOT EXISTS federated_learning_kb (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    wrong_part TEXT UNIQUE,
    correct_part TEXT,
    votes INTEGER DEFAULT 1,
    status TEXT,
    last_updated TEXT,
    sources TEXT
)
""")

# BOM Match Result Table
cursor.execute("""
CREATE TABLE IF NOT EXISTS bom_matches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    design_part TEXT,
    matched_part TEXT,
    confidence REAL,
    unit_cost REAL,
    total_cost REAL,
    stock_status TEXT,
    match_type TEXT,
    created_at TEXT
)
""")

conn.commit()
conn.close()

print("✅ SQLite Database & Tables Created Successfully")

import sqlite3

# -------------------------------
# DB Connection
# -------------------------------
def get_db_connection():
    conn = sqlite3.connect("optibom.db")
    conn.row_factory = sqlite3.Row
    return conn


# -------------------------------
# Save Federated Learning Data
# -------------------------------
def save_federated_learning(wrong_part, correct_part, factory):
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
    INSERT INTO federated_learning_kb 
    (wrong_part, correct_part, votes, status, last_updated, sources)
    VALUES (?, ?, 1, 'Pending', datetime('now'), ?)
    ON CONFLICT(wrong_part)
    DO UPDATE SET
        votes = votes + 1,
        last_updated = datetime('now')
    """, (wrong_part, correct_part, factory))

    conn.commit()
    conn.close()


# -------------------------------
# Load Federated Learning Data
# -------------------------------
def load_federated_learning():
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("SELECT * FROM federated_learning_kb")
    rows = cur.fetchall()

    conn.close()
    return rows

def save_bom_match(design_part, matched_part, confidence,
                   unit_cost, total_cost, stock_status, match_type):

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
    INSERT INTO bom_matches
    (design_part, matched_part, confidence,
     unit_cost, total_cost, stock_status,
     match_type, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
    """, (
        design_part,
        matched_part,
        confidence,
        unit_cost,
        total_cost,
        stock_status,
        match_type
    ))

    conn.commit()
    conn.close()
