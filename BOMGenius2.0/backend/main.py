from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File
import pandas as pd
import io
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from federated.local_trainer import log_human_feedback, export_local_updates
from core.engine import generate_mbom
from repo.DB import init_db, save_mbom, fetch_mbom

app = FastAPI(title="BOMGenius API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

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
    return FileResponse("frontend/home.html")

@app.post("/generate-mbom")
async def generate_mbom_api(ebom: UploadFile = File(...), inventory: UploadFile = File(...)):
    # Read files into dataframes
    ebom_df = pd.read_csv(io.BytesIO(await ebom.read()))
    inv_df = pd.read_csv(io.BytesIO(await inventory.read()))

    # Process data
    df_final = generate_mbom(ebom_df, inv_df)
    
    # Save to Database
    save_mbom(df_final)

    return {
        "columns": list(df_final.columns),
        "rows": df_final.to_dict(orient="records")
    }

@app.get("/mbom/latest")
def get_mbom():
    df = fetch_mbom()
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

