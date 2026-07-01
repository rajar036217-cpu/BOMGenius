import pandas as pd
import json
import io
import os
import pdfplumber
import chardet
import ezdxf


def safe_normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df


def parse_pdf_ebom(file_bytes):

    import pdfplumber
    import pandas as pd
    import io

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:

        all_tables = []

        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                df = pd.DataFrame(table)
                all_tables.append(df)

    if not all_tables:
        return pd.DataFrame()

    df = pd.concat(all_tables, ignore_index=True)

    # -------------------------
    # AUTO HEADER DETECTION
    # -------------------------
    for i in range(min(10, len(df))):
        row = df.iloc[i].astype(str).str.lower().tolist()

        if any("part" in cell for cell in row):
            df.columns = df.iloc[i]
            df = df[i+1:]
            df = df.reset_index(drop=True)
            return df

    # fallback
    df.columns = df.iloc[0]
    df = df[1:]
    return df.reset_index(drop=True)



def extract_from_dxf(dxf_file):
    extracted_items = []
    try:
        dxf_data = dxf_file.getvalue().decode("utf-8", errors="ignore")
        doc = ezdxf.read(io.StringIO(dxf_data))
        msp = doc.modelspace()
        for entity in msp.query("TEXT MTEXT"):
            text_content = entity.dxf.text if entity.dxftype() == "TEXT" else entity.text
            if len(text_content) > 2:
                extracted_items.append(text_content)
        df = pd.DataFrame(extracted_items, columns=["Description"])
        df["Quantity"] = 1
        return df
    except Exception as e:
        print(f"CAD Parsing Error: {e}")
        return pd.DataFrame()


def load_ebom(file_bytes, filename):

    ext = os.path.splitext(filename)[1].lower()

    # ---------------- CSV ----------------
    if ext == ".csv":
        detected = chardet.detect(file_bytes)
        encoding = detected["encoding"] or "utf-8"
        df = pd.read_csv(io.BytesIO(file_bytes), encoding=encoding)
        return safe_normalize_columns(df)

    # ---------------- EXCEL ----------------
    elif ext in [".xls", ".xlsx"]:
        return pd.read_excel(io.BytesIO(file_bytes))

    # ---------------- JSON ----------------
    elif ext == ".json":
        data = json.loads(file_bytes.decode("utf-8"))

        # CASE 1 → Already list of dicts
        if isinstance(data, list):
            if all(isinstance(row, dict) for row in data):
                return pd.DataFrame(data)
            elif all(isinstance(row, list) for row in data):
                return pd.DataFrame(
                    [{str(i): v for i, v in enumerate(r)} for r in data]
                )

        # CASE 2 → dict containing list
        if isinstance(data, dict):
            for key in ["rows", "data", "body", "table"]:
                if key in data and isinstance(data[key], list):
                    return pd.DataFrame(data[key])

        # fallback
        return pd.json_normalize(data)

    # ---------------- PDF ----------------
    elif ext == ".pdf":
        return parse_pdf_ebom(file_bytes)

    else:
        raise ValueError(f"Unsupported file type: {ext}")


def normalize_ebom_columns(df):
    rename_map = {}

    for col in df.columns:
        c = col.lower()
        if "part" in c and ("no" in c or "number" in c or "id" in c):
            rename_map[col] = "Part_No"
        elif "desc" in c or "name" in c:
            rename_map[col] = "Description"
        elif "qty" in c or "quantity" in c:
            rename_map[col] = "Qty"
        elif "uom" in c or "unit" in c:
            rename_map[col] = "UOM"
        elif "parent" in c or "assembly" in c:
            rename_map[col] = "Parent_Part_No"
    return df.rename(columns=rename_map)            