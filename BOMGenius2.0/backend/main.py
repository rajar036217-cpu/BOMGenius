import os
import io
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from core.engine import generate_mbom  # keep this consistent with your folder structure

try:
    from ocr.ebom_from_image import ebom_from_image
except Exception:
    ebom_from_image = None

try:
    from repo.DB import init_db, save_mbom
except Exception:
    init_db = None
    save_mbom = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
frontend_path = os.path.join(BASE_DIR, "frontend")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if init_db:
        init_db()
    yield
    print("API shutting down...")


app = FastAPI(title="BOMGenius API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.isdir(frontend_path):
    app.mount("/frontend", StaticFiles(directory=frontend_path), name="frontend")


@app.get("/")
def home():
    index_path = os.path.join(frontend_path, "home.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"status": "ok"}


def read_table_from_upload(file_bytes: bytes, filename: str) -> pd.DataFrame:
    ext = Path(filename).suffix.lower()

    if ext == ".csv":
        try:
            return pd.read_csv(io.BytesIO(file_bytes), encoding="utf-8")
        except Exception:
            return pd.read_csv(io.BytesIO(file_bytes), encoding="latin1")

    if ext in [".xlsx", ".xls"]:
        return pd.read_excel(io.BytesIO(file_bytes))

    raise ValueError(f"Unsupported file type: {ext}")


@app.post("/fullbomconverter")
async def generate_mbom_api(
    ebom: UploadFile = File(...),
    inventory: Optional[UploadFile] = File(None),
):
    ebom_bytes = await ebom.read()
    ebom_name = ebom.filename or "ebom.csv"
    ext = Path(ebom_name).suffix.lower()

    # 1) Load eBOM
    if ext in [".png", ".jpg", ".jpeg"]:
        if not ebom_from_image:
            raise ValueError("Image OCR not configured. Upload CSV/XLSX instead.")
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(ebom_bytes)
            tmp_path = tmp.name
        ebom_df = ebom_from_image(tmp_path)
    else:
        ebom_df = read_table_from_upload(ebom_bytes, ebom_name)

    # 2) Load inventory (optional)
    if inventory:
        inv_bytes = await inventory.read()
        inv_name = inventory.filename or "inventory.xlsx"
        inv_df = read_table_from_upload(inv_bytes, inv_name)
    else:
        inv_df = pd.DataFrame()

    # 3) Convert
    df_final = generate_mbom(ebom_df, inv_df)

    # ✅ clean NaN display
    df_final = df_final.fillna("")
    for c in ["Parent Part Number", "Parent Description"]:
        if c in df_final.columns:
            df_final[c] = df_final[c].astype(str).replace({"nan": "", "None": ""}).fillna("")

    # 4) Save to DB (does not affect API output)
    if save_mbom:
        try:
            save_mbom(df_final.copy(deep=True))
        except Exception:
            pass

    # ✅ If timestamp sneaks in, remove from API output
    if "timestamp" in df_final.columns:
        df_final = df_final.drop(columns=["timestamp"])

    # Optional: Mode column
    df_final["Mode"] = "WITH_INVENTORY" if (inventory and not inv_df.empty) else "WITHOUT_INVENTORY"

    return {
        "columns": list(df_final.columns),
        "rows": df_final.to_dict(orient="records"),
    }