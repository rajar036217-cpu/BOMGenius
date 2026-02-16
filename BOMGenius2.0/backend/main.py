import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
import sqlite3
from datetime import datetime

import pandas as pd
from core.ebom_loader import load_ebom
from core.engine import generate_mbom
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from federated.local_trainer import export_local_updates, log_human_feedback
from ocr.ebom_from_image import ebom_from_image
from pydantic import BaseModel
from repo.DB import init_db, save_mbom

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
frontend_path = os.path.join(BASE_DIR, "frontend")


# C:\Users\Raja\OneDrive\Desktop\BOM_Project\BOMGenius2.0\federated\local_trainer.py

app = FastAPI(title="BOMGenius API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/frontend", StaticFiles(directory=frontend_path), name="frontend")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_db()
    yield
    # Shutdown (optional cleanup)
    print("API shutting down...")

#basemodels for request bodies
class LoginRequest(BaseModel):
    company_name: str
    company_id: str
    password: str

class RegisterRequest(BaseModel):
    full_name:str
    email: str
    company_address: str
    password: str
    confirm_password: str
    
class FeedbackRequest(BaseModel):
    message: str

class SettingsRequest(BaseModel):
    company: str
    email: str

@app.get("/")
def home():
    return FileResponse(os.path.join(frontend_path, "home.html"))

@app.post("/login")
def login(data: LoginRequest):
    return {"message": "Login successful"}

@app.post("/register")
def register(data: RegisterRequest):
    return {"message": "User registered successfully"}

@app.post("/forgot-password")
def forgot_password(email: str):
    return {"message": "Reset link sent"}

@app.post("/feedback")
def feedback(data: FeedbackRequest):
    return {"message": "Feedback submitted"}

@app.post("/settings")
def save_settings(data: SettingsRequest):
    return {"message": "Settings saved"}

@app.post("/fullbomconverter")
async def generate_mbom_api(
    ebom: UploadFile = File(...), inventory: Optional[UploadFile] = File(None)
):
    ebom_bytes = await ebom.read()
    ext = Path(ebom.filename).suffix.lower() if ebom.filename else ""

    if ext in [".png", ".jpg", ".jpeg", ".pdf"]:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(ebom_bytes)
            tmp_path = tmp.name

        ebom_df = ebom_from_image(tmp_path)
    else:
        ebom_df = load_ebom(ebom_bytes, ebom.filename)

    if inventory:
        inv_bytes = await inventory.read()
        inv_df = load_ebom(inv_bytes, inventory.filename)
    else:
        inv_df = pd.DataFrame()

    if inventory:
        df_final = generate_mbom(ebom_df, inv_df)
    else:
        df_final = generate_mbom(ebom_df, pd.DataFrame())

    save_mbom(df_final)

    return {
        "columns": list(df_final.columns),
        "rows": df_final.to_dict(orient="records"),
    }

@app.get("/dashboard/analytics")
def dashboard_analytics():
    conn = sqlite3.connect("bomgenius.db")
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("SELECT MAX(timestamp) AS ts FROM mbom")
    ts = cur.fetchone()["ts"]

    if not ts:
        return {"composition": [], "confidence": {"high": 0, "medium": 0, "low": 0}, "timestamp": None}

    # Pie: Item type composition
    cur.execute("""
        SELECT COALESCE(Item_Type, 'Standard Parts') AS item_type, COUNT(*) AS cnt
        FROM mbom
        WHERE timestamp = ?
        GROUP BY COALESCE(Item_Type, 'Standard Parts')
        ORDER BY cnt DESC
    """, (ts,))
    composition = [dict(r) for r in cur.fetchall()]

    # Bar: confidence buckets
    cur.execute("""
        SELECT
          SUM(CASE WHEN Confidence_Score >= 0.90 THEN 1 ELSE 0 END) AS high,
          SUM(CASE WHEN Confidence_Score >= 0.70 AND Confidence_Score < 0.90 THEN 1 ELSE 0 END) AS medium,
          SUM(CASE WHEN Confidence_Score < 0.70 THEN 1 ELSE 0 END) AS low
        FROM mbom
        WHERE timestamp = ?
    """, (ts,))
    conf = dict(cur.fetchone())

    conn.close()
    return {"composition": composition, "confidence": conf, "timestamp": ts}


@app.post("/ebom/ocr")
async def ebom_from_image_api(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower() if file.filename else ""

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    df_ebom = ebom_from_image(tmp_path)

    return {"columns": list(df_ebom.columns), "rows": df_ebom.to_dict(orient="records")}


@app.get("/mbom/history")
def get_mbom_history():

    with sqlite3.connect("bomgenius.db") as conn:
        rows = conn.execute(
            """
            SELECT timestamp, COUNT(*) as total_rows
            FROM mbom
            GROUP BY timestamp
            ORDER BY timestamp DESC
        """
        ).fetchall()

    history = []

    for ts, count in rows:
        dt = datetime.fromisoformat(ts)
        history.append(
            {
                "timestamp": ts,
                "date": dt.strftime("%d-%m-%Y"),
                "time": dt.strftime("%I:%M %p"),
                "rows": count,
            }
        )

    return history


@app.get("/mbom/by-timestamp/{ts}")
def get_mbom_by_timestamp(ts: str):
    import sqlite3

    import pandas as pd

    with sqlite3.connect("bomgenius.db") as conn:
        data = conn.execute("SELECT * FROM mbom WHERE timestamp = ?", (ts,)).fetchall()

    if not data:
        return {"columns": [], "rows": []}

    columns = [
        "parent_part",
        "child_part",
        "description",
        "qty",
        "uom",
        "level",
        "lead_time",
        "work_center",
        "make_buy",
        "phantom",
        "scrap",
        "plant",
        "storage_location",
        "timestamp",
    ]

    df = pd.DataFrame(data, columns=columns)

    return {"columns": list(df.columns), "rows": df.to_dict(orient="records")}

@app.get("/dashboard")
def get_dashboard():

    conn = sqlite3.connect("bomgenius.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Total BOMs
    cursor.execute("""
        SELECT COUNT(DISTINCT Parent_Part_No) AS total_boms
        FROM mbom
    """)
    total_boms = cursor.fetchone()["total_boms"]

    # Total Components
    cursor.execute("""
        SELECT COUNT(*) AS total_components
        FROM mbom
    """)
    total_components = cursor.fetchone()["total_components"]

    # Latest Upload
    cursor.execute("""
        SELECT MAX(timestamp) AS last_uploaded
        FROM mbom
    """)
    last_uploaded = cursor.fetchone()["last_uploaded"]

    # Avg Components per BOM
    cursor.execute("""
        SELECT ROUND(AVG(component_count), 2) AS avg_components
        FROM (
            SELECT COUNT(*) AS component_count
            FROM mbom
            GROUP BY Parent_Part_No
        )
    """)
    avg_components = cursor.fetchone()["avg_components"]

    conn.close()

    return {
        "total_boms": total_boms or 0,
        "total_components": total_components or 0,
        "last_uploaded": last_uploaded,
        "avg_components": avg_components or 0
    }

class Feedback(BaseModel):
    part_no: str
    correct_make_buy: str | None = None
    correct_uom: str | None = None
    correct_work_center: str | None = None


@app.post("/feedback")
def submit_feedback(feedback: Feedback):
    log_human_feedback(feedback.dict())
    return {"status": "feedback recorded"}


@app.get("/federated/export")
def federated_export():
    return export_local_updates()


@app.post("/federated/import")
def federated_import(global_rules: dict):
    import json

    os.makedirs("federated", exist_ok=True)
    with open("federated/global_rules.json", "w") as f:
        json.dump(global_rules, f, indent=2)
    return {"status": "global rules updated"}

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

@app.get("/company")
def get_companies():
    import sqlite3

    with sqlite3.connect("bomgenius.db") as conn:
        rows = conn.execute(
            "SELECT id, name, created_at, last_login, is_active FROM companies"
        ).fetchall()

    return [
        {
            "id": r[0],
            "name": r[1],
            "created_at": r[2],
            "last_login": r[3],
            "is_active": r[4]
        }
        for r in rows
    ]

class CompanyUpdate(BaseModel):
    name: str

@app.put("/company/{cid}")
def update_company(cid: int, data: CompanyUpdate):
    import sqlite3

    with sqlite3.connect("bomgenius.db") as conn:
        conn.execute(
            "UPDATE companies SET name=? WHERE id=?",
            (data.name, cid)
        )

    return {"status": "updated"}

@app.delete("/company/{cid}")
def delete_company(cid: int):
    import sqlite3

    with sqlite3.connect("bomgenius.db") as conn:
        conn.execute(
            "DELETE FROM companies WHERE id=?",
            (cid,)
        )

    return {"status": "deleted"}

@app.patch("/company/{cid}/status")
def toggle_company_status(cid: int):
    import sqlite3

    with sqlite3.connect("bomgenius.db") as conn:
        cur = conn.cursor()

        cur.execute("SELECT is_active FROM companies WHERE id=?", (cid,))
        row = cur.fetchone()

        if not row:
            return {"error": "company not found"}

        current = row[0]
        new_status = 0 if current == 1 else 1

        cur.execute(
            "UPDATE companies SET is_active=? WHERE id=?",
            (new_status, cid)
        )

    return {"status": "changed", "is_active": new_status}