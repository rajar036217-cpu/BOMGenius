# backend.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import pandas as pd
import sqlite3
import os
import io
import ollama

DB_NAME = "bomgenius.db"
MODEL_NAME = "llama3.2:3b"

app = FastAPI(title="BOMGenius API", version="0.1")

def init_db():
    conn = sqlite3.connect(DB_NAME)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS matches(
            Parent_ID TEXT,
            Item_ID TEXT,
            Item_Name TEXT,
            Qty TEXT,
            Op_Sequence TEXT,
            Work_Center TEXT,
            Make_Buy TEXT,
            MBOM_Item_Type TEXT,
            EBOM_Ref_ID TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.close()

init_db()

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}

@app.post("/api/bom/convert")
async def convert_bom(ebom: UploadFile = File(...), inventory: UploadFile = File(...)):
    ebom_df = pd.read_csv(ebom.file)
    inv_df = pd.read_csv(inventory.file)

    prompt = f"""You are a Manufacturing BOM Engineer AI embedded in a PLM-to-MES integration system.
GOAL:
Convert the given Engineering BOM (eBOM) into a Manufacturing BOM (mBOM) suitable for factory execution.

EBOM (CSV):
{ebom_df.to_csv(index=False)}

INVENTORY (CSV):
{inv_df.to_csv(index=False)}

CONVERSION RULES:
1. Preserve all original parts and quantities from the eBOM.
2. Re-group parts into logical manufacturing sub-assemblies based on build sequence and shop-floor practicality.
3. Create manufacturing sub-assemblies even if they do not exist in the eBOM (use synthetic IDs like SUBASM-001, SUBASM-002).
4. Assign each part to exactly one sub-assembly.
5. Introduce a final top-level assembly node that consumes all sub-assemblies.
6. Add manufacturing attributes for each line:
   - Op_Sequence (10, 20, 30, ...)
   - Work_Center (Welding, Assembly, QC, Packaging only)
   - Make_Buy (default from eBOM; if missing, assume MAKE for custom parts, BUY for standard fasteners)
   - MBOM_Item_Type (RAW, SUBASM, FG)
7. Do NOT remove any part present in the eBOM.
8. Do NOT invent quantities. Quantities must match totals from the eBOM.
9. If ambiguous, choose the most reasonable grouping and list assumptions.

OUTPUT FORMAT (STRICT):
Return ONLY CSV-like rows using this exact schema:
Parent_ID | Item_ID | Item_Name | Qty | Op_Sequence | Work_Center | Make_Buy | MBOM_Item_Type | EBOM_Ref_ID

After the table, add:
ASSUMPTIONS:
- <short bullets>

MAPPING_NOTES:
- <short bullets>

No markdown."""

    response = ollama.generate(model=MODEL_NAME, prompt=prompt)
    raw = response["response"]

    lines = [l.strip() for l in raw.split('\n') if '|' in l]
    df_final = pd.read_csv(io.StringIO('\n'.join(lines)), sep="|", skipinitialspace=True)

    with sqlite3.connect(DB_NAME) as conn:
        df_final.to_sql('matches', conn, if_exists='append', index=False)

    output_path = "mbom_latest.csv"
    df_final.to_csv(output_path, index=False)

    return {"status": "ok", "rows": len(df_final), "download": "/api/mbom/latest"}

@app.get("/api/mbom/latest")
def download_latest_mbom():
    return FileResponse("mbom_latest.csv", media_type="text/csv", filename="mbom_latest.csv")