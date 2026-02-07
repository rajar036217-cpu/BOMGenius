from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File
import pandas as pd
import io

from core.engine import generate_mbom
from repo.DB import init_db, save_mbom

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # later restrict to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_db()
    yield
    # Shutdown (optional cleanup)
    print("API shutting down...")

app = FastAPI(title="BOMGenius API")

@app.get("/health")
def health():
    return {"status": "ok", "service": "BOMGenius API"}

@app.post("/generate-mbom")
async def generate_mbom_api(ebom: UploadFile = File(...), inventory: UploadFile = File(...)):
    ebom_df = pd.read_csv(io.BytesIO(await ebom.read()))
    inv_df = pd.read_csv(io.BytesIO(await inventory.read()))

    df_final = generate_mbom(ebom_df, inv_df)
    save_mbom(df_final)

    return {
        "columns": list(df_final.columns),
        "rows": df_final.to_dict(orient="records")

    }

from db.repository import fetch_latest_mbom

@app.get("/mbom/latest")
def get_latest_mbom():
    df = fetch_latest_mbom()
    return {
        "columns": list(df.columns),
        "rows": df.to_dict(orient="records")
    }
