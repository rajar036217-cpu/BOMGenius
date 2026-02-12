import pandas as pd
import io
import ollama
import torch
from sentence_transformers import SentenceTransformer, util
import json
import os

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

GLOBAL_RULES_FILE = "federated/global_rules.json"

def load_global_rules():
    if os.path.exists(GLOBAL_RULES_FILE):
        with open(GLOBAL_RULES_FILE, "r") as f:
            return json.load(f)
    return {}

def learn_ebom_schema(df):
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
        else:
            col_map[col] = "unknown_" + col
    return col_map

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
    ebom_schema = learn_ebom_schema(ebom_df)
    normalized_ebom = normalize_inventory(ebom_df, ebom_schema)

    learned_schema = learn_inventory_schema(inv_df)
    normalized_inv = normalize_inventory(inv_df, learned_schema)

    global_rules = load_global_rules()
    global_context = json.dumps(global_rules, indent=2)
    
    inv_context = normalized_inv.to_csv(index=False)
    ebom_context = normalized_ebom.to_csv(index=False)

    prompt = f"""
You are a Manufacturing Engineer AI. Your task: Transform eBOM to mBOM.
Strictly follow the logic for Make/Buy, UOM, and Hierarchy.
Output format: PIPE-DELIMITED CSV. No Headers. No Markdown.

### INPUT DATA
EBOM:
{ebom_context}

INVENTORY:
{inv_context}

GLOBAL RULES:
{global_context}

### LOGIC RULES
1. Structure: Level 0 (FG) -> Level 1 (SUBASM) -> Level 2+ (Parts).
2. Synthetic IDs: If no subassembly exists, create SUBASM-001, SUBASM-002.
3. Make/Buy: Fasteners=BUY, Fabricated=MAKE. Use Inventory first.
4. UOM: Fasteners=EA, Sheets=KG, Fluids=L.
5. Item Types: RAW, SUBASM, FG.

### OUTPUT EXAMPLE (Follow this format exactly)
PARENT-001|SUBASM-001|WELDED FRAME|1|EA|1|10|Welding|MAKE|No|2|PLANT-01|A1-10
SUBASM-001|PART-99|STEEL PLATE|2|KG|2|10|Welding|BUY|Yes|0|PLANT-01|B2-05

### TASK
Generate the mBOM for the provided input. 
Column Order: Parent_Part_No|Child_Part_No|Description|Qty_Per|UOM|BOM_Level|Op_Sequence|Work_Center|Make_Buy|Backflush_Ind|Scrap_Pct|Plant|Bin_Location

Output the CSV rows first, then a line with "---", then "ASSUMPTIONS:" and "MAPPING_NOTES:".
Do not use markdown blocks.
"""
    response = ollama.generate(
        model=MODEL_NAME,
        prompt=prompt,
        options={"temperature": 0.0, "num_ctx": 4096, "seed": 42}
    )

    raw = response["response"]
    lines = [l.strip() for l in raw.split("\n") if "|" in l]

    EXPECTED_MBOM_COLS = 13

    parsed = [l.split("|") for l in lines]
    bad_rows = [r for r in parsed if len(r) != EXPECTED_MBOM_COLS]

    if bad_rows:
        raise ValueError(f"Invalid mBOM rows from LLM: {bad_rows[:2]}")

    df_final = pd.DataFrame(parsed, columns=[
    "Parent_Part_No","Child_Part_No","Description","Qty_Per","UOM",
    "BOM_Level","Op_Sequence","Work_Center","Make_Buy",
    "Backflush_Ind","Scrap_Pct","Plant","Bin_Location"
])



    return df_final
