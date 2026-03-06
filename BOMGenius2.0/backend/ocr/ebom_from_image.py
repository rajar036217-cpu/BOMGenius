import pandas as pd
import os
import re
from datetime import datetime
from .vision_ollama import run_vision_inference

VISION_MODEL = "glm-ocr:q8_0"


# ==========================================================
# MAIN ENTRY
# ==========================================================
def ebom_from_image(image_path: str) -> pd.DataFrame:
    print(image_path)

    prompt = """
Extract BOM table from the image.

Return STRICT JSON.
Return ONLY this format:

[
  {
    "ITEM_NO": "string",
    "PART_NUMBER": "string",
    "DESCRIPTION": "string",
    "QTY": "string"
  }
]

Return nothing else.
"""

    rows = run_vision_inference(
        VISION_MODEL,
        image_path,
        prompt
    )

    if not rows:
        raise ValueError("Vision model returned empty or invalid JSON.")

    parsed_rows = []
    print(rows)

    for row in rows:
        parsed_rows.append({
            "ITEM NO.": row.get("ITEM_NO", ""),
            "PART NUMBER": row.get("PART_NUMBER", ""),
            "DESCRIPTION": row.get("DESCRIPTION", ""),
            "QTY.": row.get("QTY", "")
        })

    raw_df = pd.DataFrame(parsed_rows)

    return transform_to_enterprise_schema(raw_df)


# ==========================================================
# TRANSFORMATION LAYER
# ==========================================================
def transform_to_enterprise_schema(raw_df: pd.DataFrame) -> pd.DataFrame:

    # Normalize headers again (safe)
    raw_df.columns = [c.strip().upper() for c in raw_df.columns]
    print(raw_df.columns)

    required_columns = ["PART NUMBER", "DESCRIPTION", "QTY."]

    for col in required_columns:
        if col not in raw_df.columns:
            raise ValueError(
                f"Missing expected column: {col}. "
                f"Found: {raw_df.columns.tolist()}"
            )

    enterprise_rows = []
    timestamp = datetime.utcnow().isoformat()

    for _, row in raw_df.iterrows():

        part_number = str(row.get("PART NUMBER", "")).strip()
        description = str(row.get("DESCRIPTION", "")).strip()
        qty = str(row.get("QTY.", "")).strip()
        print(part_number)

        enterprise_rows.append({
            "Level": "0",
            "Parent Part Number": part_number,
            "Parent Description": part_number,
            "Child Description": description,
            "UOM": "EA",
            "Node Type": "Component",
            "Make/Buy": "Buy",
            "Work Center": "INCOMING_QC",
            "Procurement Steps": "PR -> PO -> GRN -> Incoming QC -> Putaway -> Issue",
            "Operations (Routing Embedded)": "Incoming Inspection",
            "Consumables": "",
            "Qty": qty,
            "Child Part Number": part_number,
            "Hierarchy Path": part_number,
            "Revision": "A",
            "Effective Date": "2026-01-01",
            "Confidence_Score": "0.95",
            "Inventory Status": "Unknown",
            "Stock_Qty": "0",
            "Store_Location": "MAIN_WH",
            "Procurement Action": "Auto",
            "Approved_Supplier": "",
            "Lead_Time_Days": "7",
            "timestamp": timestamp,
            "company_id": "-1"
        })

    return pd.DataFrame(enterprise_rows)
