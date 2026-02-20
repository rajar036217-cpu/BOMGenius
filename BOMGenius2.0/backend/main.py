import io
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from core.engine import generate_mbom

app = FastAPI(title="BOMGenius API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def read_table(file_bytes: bytes, filename: str) -> pd.DataFrame:
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
async def fullbomconverter(
    ebom: UploadFile = File(...),
    inventory: Optional[UploadFile] = File(None),
):
    ebom_bytes = await ebom.read()
    ebom_df = read_table(ebom_bytes, ebom.filename or "ebom.csv")

    if inventory:
        inv_bytes = await inventory.read()
        inv_df = read_table(inv_bytes, inventory.filename or "inventory.csv")
    else:
        inv_df = pd.DataFrame()

    df_final = generate_mbom(ebom_df, inv_df)

    # API response
    return {
        "columns": list(df_final.columns),
        "rows": df_final.to_dict(orient="records"),
        "mode": "WITH_INVENTORY" if (inventory and not inv_df.empty) else "WITHOUT_INVENTORY",
    }