from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File
import pandas as pd
import io
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

import sys, os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
frontend_path = os.path.join(BASE_DIR, "frontend")

from federated.local_trainer import log_human_feedback, export_local_updates
from core.engine import generate_mbom
from core.ebom_loader import load_ebom  
from repo.DB import init_db, save_mbom, fetch_mbom
from ocr.ebom_from_image import ebom_from_image
import tempfile


#C:\Users\Raja\OneDrive\Desktop\BOM_Project\BOMGenius2.0\federated\local_trainer.py

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

@app.get("/health")
def health():
    return {"status": "ok", "service": "BOMGenius API"}

@app.get("/")
def home():
    return FileResponse(os.path.join(frontend_path, "home.html"))

from typing import Optional

@app.post("/fullbomconverter")
async def generate_mbom_api(
    ebom: UploadFile = File(...),
    inventory: Optional[UploadFile] = File(None)
):
    ebom_bytes = await ebom.read()
    ext = os.path.splitext(ebom.filename)[1].lower()

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

    df_final = generate_mbom(ebom_df, inv_df)
    save_mbom(df_final)

    return {
        "columns": list(df_final.columns),
        "rows": df_final.to_dict(orient="records")
    }

@app.post("/ebom/ocr")
async def ebom_from_image_api(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    df_ebom = ebom_from_image(tmp_path)

    return {
        "columns": list(df_ebom.columns),
        "rows": df_ebom.to_dict(orient="records")
    }

@app.get("/mbom/history")
def get_mbom_history():
    import sqlite3
    from datetime import datetime

    with sqlite3.connect("bomgenius.db") as conn:
        rows = conn.execute("""
            SELECT timestamp, COUNT(*) as total_rows
            FROM mbom
            GROUP BY timestamp
            ORDER BY timestamp DESC
        """).fetchall()

    history = []

    for ts, count in rows:
        dt = datetime.fromisoformat(ts)
        history.append({
            "timestamp": ts,
            "date": dt.strftime("%d-%m-%Y"),
            "time": dt.strftime("%I:%M %p"),
            "rows": count
        })

    return history


@app.get("/mbom/by-timestamp/{ts}")
def get_mbom_by_timestamp(ts: str):
    import sqlite3
    import pandas as pd

    with sqlite3.connect("bomgenius.db") as conn:
        data = conn.execute(
            "SELECT * FROM mbom WHERE timestamp = ?",
            (ts,)
        ).fetchall()

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
        "timestamp"
    ]

    df = pd.DataFrame(data, columns=columns)

    return {
        "columns": list(df.columns),
        "rows": df.to_dict(orient="records")
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
