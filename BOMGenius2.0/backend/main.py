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
from repo.DB import init_db, save_mbom, fetch_mbom

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
    ebom_df = pd.read_csv(io.BytesIO(await ebom.read()))

    if inventory:
        inv_df = pd.read_csv(io.BytesIO(await inventory.read()))
    else:
        inv_df = pd.DataFrame()  # empty inventory

    df_final = generate_mbom(ebom_df, inv_df)
    print(df_final)

    save_mbom(df_final)

    return {
        "columns": list(df_final.columns),
        "rows": df_final.to_dict(orient="records")
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


@app.get("/mbom/latest")
def get_mbom():
    data = fetch_mbom()  # this is a list of tuples

    if not data:
        return {"columns": [], "rows": []}

    import pandas as pd

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
        "storage_location"
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