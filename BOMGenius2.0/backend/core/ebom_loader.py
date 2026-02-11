import pandas as pd
import json
import io
import os
import pdfplumber

def load_ebom(file_bytes, filename):
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".csv":
        return pd.read_csv(io.BytesIO(file_bytes))

    elif ext in [".xls", ".xlsx"]:
        return pd.read_excel(io.BytesIO(file_bytes))

    elif ext == ".json":
        data = json.loads(file_bytes.decode("utf-8"))
        return pd.DataFrame(data)

    elif ext == ".pdf":
        return parse_pdf_ebom(file_bytes)

    else:
        raise ValueError(f"Unsupported file type: {ext}")

def parse_pdf_ebom(file_bytes):
    rows = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            table = page.extract_table()
            if table:
                headers = table[0]
                for row in table[1:]:
                    rows.append(dict(zip(headers, row)))

    return pd.DataFrame(rows)

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
