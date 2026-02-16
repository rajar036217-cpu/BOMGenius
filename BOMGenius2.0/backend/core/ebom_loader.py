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

    extracted_data = []

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:

        for page in pdf.pages:
            table = page.extract_table()

            if table and len(table) > 1:

                df_page = pd.DataFrame(table[1:], columns=table[0])

                # Clean headers
                df_page.columns = [
                    str(col).strip() if col else f"Column_{i}"
                    for i, col in enumerate(df_page.columns)
                ]

                # Remove duplicate columns safely
                df_page = df_page.loc[:, ~df_page.columns.duplicated()]

                extracted_data.append(df_page)

    if not extracted_data:
        return pd.DataFrame()

    # Align columns safely before concat
    all_columns = list(set().union(*[df.columns for df in extracted_data]))

    aligned_dfs = [
        df.reindex(columns=all_columns)
        for df in extracted_data
    ]

    return pd.concat(aligned_dfs, ignore_index=True)



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
    if ext == ".csv":
        detected = chardet.detect(file_bytes)
        encoding = detected["encoding"] or "utf-8"
        df = pd.read_csv(io.BytesIO(file_bytes), encoding=encoding)
        df = safe_normalize_columns(df)
        return df

    elif ext in [".xls", ".xlsx"]:
        return pd.read_excel(io.BytesIO(file_bytes))

    elif ext == ".json":
        data = json.loads(file_bytes.decode("utf-8"))
        return pd.DataFrame(data)

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