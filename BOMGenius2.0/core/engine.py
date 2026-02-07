import pandas as pd
import io
import ollama
import torch
from sentence_transformers import SentenceTransformer, util

torch.set_num_threads(1)

MODEL_NAME = "llama3.2:3b"

CANONICAL_FIELDS = {
    "part_name": ["name", "desc", "description", "item", "part name"],
    "part_no": ["part", "number", "pn", "id", "code"],
    "bin_location": ["bin", "location", "loc", "warehouse", "stock", "store", "depot"],
    "work_center": ["wc", "workcenter", "work center", "dept", "shop", "line"],
}

schema_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

CANONICAL_EMBEDDINGS = {
    k: schema_model.encode(v) for k, v in CANONICAL_FIELDS.items()
}

def learn_inventory_schema(df):
    col_map = {}
    for col in df.columns:
        col_emb = schema_model.encode(col.lower())
        best_match, best_score = None, 0
        for canon, emb_list in CANONICAL_EMBEDDINGS.items():
            score = util.cos_sim(col_emb, emb_list).max().item()
            if score > best_score:
                best_score, best_match = score, canon
        if best_score > 0.55:
            col_map[col] = best_match
    return col_map

def normalize_inventory(df, col_map):
    norm = pd.DataFrame()
    for raw_col, canon in col_map.items():
        norm[canon] = df[raw_col]
    return norm

def generate_mbom(ebom_df, inv_df):
    learned_schema = learn_inventory_schema(inv_df)
    normalized_inv = normalize_inventory(inv_df, learned_schema)

    prompt = f"""
    EBOM:
    {ebom_df.to_csv(index=False)}

    INVENTORY:
    {normalized_inv.to_csv(index=False)}

    OUTPUT STRICT CSV with | delimiter.
    """

    response = ollama.generate(
        model=MODEL_NAME,
        prompt=prompt,
        options={"temperature": 0.0, "num_ctx": 4096, "seed": 42}
    )

    raw = response["response"]
    lines = [l.strip() for l in raw.split("\n") if "|" in l]

    df_final = pd.read_csv(io.StringIO("\n".join(lines)), sep="|", engine="python")
    df_final.columns = [
        "Parent_Part_No","Child_Part_No","Description","Qty_Per","UOM",
        "BOM_Level","Op_Sequence","Work_Center","Make_Buy",
        "Backflush_Ind","Scrap_Pct","Plant","Bin_Location"
    ]

    return df_final