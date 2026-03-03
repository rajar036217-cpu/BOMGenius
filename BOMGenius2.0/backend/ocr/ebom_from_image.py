import pandas as pd
import json
import re
from .vision_ollama import run_vision_inference

VISION_MODEL = "glm-ocr:q8_0"


def _extract_json(text: str) -> str:
    """Remove markdown/code fences and extract first JSON block."""
    if not text:
        return ""

    text = text.strip()

    # Remove markdown fences
    text = re.sub(r"```json", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```", "", text).strip()

    # Try to extract first JSON array or object
    start_arr, end_arr = text.find("["), text.rfind("]")
    start_obj, end_obj = text.find("{"), text.rfind("}")

    if start_arr != -1 and end_arr != -1 and end_arr > start_arr:
        return text[start_arr : end_arr + 1]
    if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
        return text[start_obj : end_obj + 1]

    return text


def _normalize_for_engine(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engine expects columns like: Part Number, Description, Qty, UOM
    We'll map common OCR headers to those.
    """
    if df.empty:
        return df

    rename_map = {
        "Part No": "Part Number",
        "PartNo": "Part Number",
        "Part Number": "Part Number",
        "Description": "Description",
        "Desc": "Description",
        "Qty": "Qty",
        "Quantity": "Qty",
        "UOM": "UOM",
        "Unit": "UOM",
    }

    df = df.rename(columns={c: rename_map.get(str(c).strip(), str(c).strip()) for c in df.columns})

    # Ensure required cols exist
    for col in ["Part Number", "Description", "Qty"]:
        if col not in df.columns:
            df[col] = ""

    if "UOM" not in df.columns:
        df["UOM"] = "EA"

    # Clean fields
    df["Part Number"] = df["Part Number"].astype(str).str.strip()
    df["Description"] = df["Description"].astype(str).str.strip()

    def _to_int(x):
        try:
            return int(float(str(x).strip()))
        except Exception:
            return 1

    df["Qty"] = df["Qty"].apply(_to_int)
    df["UOM"] = df["UOM"].astype(str).str.strip()

    # Keep rows with at least description (avoid dropping legit rows)
    df = df[df["Description"] != ""].reset_index(drop=True)

    # Auto part number if missing
    for i in range(len(df)):
        if df.at[i, "Part Number"] in ["", "nan", "None", None]:
            df.at[i, "Part Number"] = f"IMG_PART_{i+1:03d}"

    return df


def ebom_from_image(image_path: str) -> pd.DataFrame:
    prompt = """
You are an OCR + PLM structured data extraction agent.

Your task:
Extract the Engineering BOM table EXACTLY as it appears in the image.

Instructions:
1. Detect the table structure.
2. Identify ALL column headers exactly as written in the image.
3. Preserve header spelling, spacing, capitalization, and symbols.
4. Extract every row under those headers.
5. If a cell is empty, return an empty string "".
6. Do NOT invent columns.
7. Do NOT summarize.
8. Do NOT normalize names.
9. Do NOT drop any columns.

Output Requirements:
- Return ONLY a valid JSON array of objects.
- Each object must contain ALL detected column headers as keys.
- Every row must have the same keys.
- Only raw JSON (no markdown / no extra text).
"""

    raw_text = run_vision_inference(VISION_MODEL, image_path, prompt)

    try:
        json_str = _extract_json(raw_text)
        data = json.loads(json_str)

        # If model returned single object, wrap into list
        if isinstance(data, dict):
            data = [data]

        if not isinstance(data, list):
            raise ValueError(f"Expected list/dict, got {type(data)}")

        # Keep only dict rows
        rows = [r for r in data if isinstance(r, dict)]

        df = pd.DataFrame(rows)

        # ✅ Light normalize ONLY for engine compatibility
        df = _normalize_for_engine(df)

        # Debug: confirm how many rows extracted
        print(f"✅ EBOM rows extracted from image: {len(df)}")
        print(f"✅ EBOM columns: {list(df.columns)}")

        return df

    except Exception as e:
        raise ValueError(
            f"OCR model did not return valid JSON array.\nRaw output:\n{raw_text}\n\nComplete error: {e}"
        ) from e