import pandas as pd
import io
import ollama
import torch
from sentence_transformers import SentenceTransformer, util
import json
import os
import re

torch.set_num_threads(1)

MODEL_NAME = "llama3.2:3b"

CANONICAL_FIELDS = {
    "part_name": ["name", "desc", "description", "item", "part name"],
    "part_no": ["part", "number", "pn", "id", "code"],
    "bin_location": ["bin", "location", "loc", "warehouse", "stock", "store", "depot"],
    "work_center": ["wc", "workcenter", "work center", "dept", "shop", "line"],
}

FASTENER_KWS = ["bolt", "nut", "washer", "screw", "rivet", "clip", "pin", "stud"]
CONSUMABLE_KWS = ["grease", "oil", "paint", "sealant", "adhesive", "glue", "coolant", "primer", "flux"]
PHANTOM_KWS = ["kit", "phantom", "set", "bundle"]

def classify_item_type(desc: str) -> str:
    d = (desc or "").lower()
    if any(k in d for k in FASTENER_KWS):
        return "Fasteners"
    if any(k in d for k in CONSUMABLE_KWS):
        return "Consumables"
    if any(k in d for k in PHANTOM_KWS):
        return "Phantom/Kits"
    return "Standard Parts"

def compute_confidence(desc: str, make_buy: str, work_center: str, used_inventory: bool) -> float:
    # Base by type
    t = classify_item_type(desc)
    base = {
        "Fasteners": 0.95,
        "Consumables": 0.85,
        "Phantom/Kits": 0.80,
        "Standard Parts": 0.75
    }[t]

    # If key fields missing → reduce
    if not desc or str(desc).strip() in ["NA", "None", ""]:
        base -= 0.20
    if not make_buy or str(make_buy).strip() in ["NA", "None", ""]:
        base -= 0.10
    if not work_center or str(work_center).strip() in ["NA", "None", ""]:
        base -= 0.10

    # Inventory helps confidence
    if used_inventory:
        base += 0.05

    # clamp 0..0.99
    base = max(0.05, min(0.99, base))
    return round(base, 2)

def attach_type_and_confidence(df, used_inventory: bool):
    df = df.copy()
    df["Item_Type"] = df["Description"].apply(classify_item_type)
    df["Confidence_Score"] = df.apply(
        lambda r: compute_confidence(
            r.get("Description"),
            r.get("Make_Buy"),
            r.get("Work_Center"),
            used_inventory
        ),
        axis=1
    )
    return df
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

def generate_mbom(ebom_df, inv_df=None):

    if inv_df is not None and not inv_df.empty:
        return generate_mbom_with_inventory(ebom_df, inv_df)
    else:
        return generate_mbom_without_inventory(ebom_df)

def generate_mbom_without_inventory(ebom_df):
    ebom_schema = learn_ebom_schema(ebom_df)
    normalized_ebom = normalize_inventory(ebom_df, ebom_schema)

    global_rules = load_global_rules()
    global_context = json.dumps(global_rules, indent=2)

    ebom_context = normalized_ebom.to_csv(index=False)
    prompt = f"""
You are a Senior Manufacturing BOM Engineer AI.

Your task:
Convert the given Engineering BOM (eBOM) into a Manufacturing BOM (mBOM) WITHOUT using factory inventory.

EBOM INPUT:
{ebom_context}

You must output ONLY the following columns in this exact order:
Part_Number | Part_Name | Quantity | UOM | BOM_Level | Assembly_or_Subassembly | Assembly_Sequence | Revision | Processing_Steps

STRICT RULES:
1. Output strictly PIPE (|) delimited rows.
2. No markdown, no explanations, no headings.
3. Each row = one mBOM item.
4. Do NOT output null, None, or empty values. If unknown, infer logically.
5. Preserve hierarchy:
   - Top-level product = BOM_Level 0, Assembly_or_Subassembly = Assembly
   - Subassemblies = BOM_Level 1+, Assembly_or_Subassembly = Subassembly
   - Parts under subassemblies = BOM_Level 2+
6. UOM rules:
   - Discrete parts → EA
   - Fluids, chemicals → L, ML, KG (infer from description)
   - Metals by weight → KG
7. Assembly_Sequence:
   - Assign numeric sequence (10, 20, 30...) within each assembly/subassembly
8. Revision:
   - Default = R1 unless eBOM specifies otherwise
9. Processing_Steps:
   - Infer realistic steps like:
     - welding
     - bolting
     - riveting
     - painting
     - coating
     - machining
     - assembly
     - inspection
   - Subassemblies should include multi-step processes.
10. Logical manufacturing rules:
    - Fasteners → bolting
    - Sheet metal parts → stamping + welding + painting
    - Plastic parts → molding + trimming
    - Bought-out parts → inspection + assembly

Return only pipe-delimited CSV rows.
"""

    response = ollama.generate(
        model=MODEL_NAME,
        prompt=prompt,
        options={"temperature": 0.0,"top_p": 0.9}
    )

    raw = response["response"]

    lines = [l.strip() for l in raw.split("\n") if "|" in l]

    EXPECTED_COLUMNS = [ 
    "Part_Number","Part_Name","Quantity","UOM",
    "BOM_Level","Assembly_or_Subassembly","Assembly_Sequence","Revision",
    "Processing_Steps"]
    cleaned_rows = []
    for row in lines:
        row = list(row.split("|"))

        # If row has fewer columns → pad with None
        if len(row) < len(EXPECTED_COLUMNS):
            row += [None] * (len(EXPECTED_COLUMNS) - len(row))

        # If row has more columns → trim extra
        if len(row) > len(EXPECTED_COLUMNS):
            row = row[:len(EXPECTED_COLUMNS)]

        cleaned_rows.append(row)

    df_final = pd.DataFrame(cleaned_rows, columns=EXPECTED_COLUMNS)

    df_final = attach_type_and_confidence(df_final, used_inventory=False)


    return df_final
    
def generate_mbom_with_inventory(ebom_df, inv_df):
    ebom_schema = learn_ebom_schema(ebom_df)
    normalized_ebom = normalize_inventory(ebom_df, ebom_schema)

    learned_schema = learn_inventory_schema(inv_df)
    normalized_inv = normalize_inventory(inv_df, learned_schema)

    global_rules = load_global_rules()
    global_context = json.dumps(global_rules, indent=2)
    
    inv_context = normalized_inv.to_csv(index=False)
    ebom_context = normalized_ebom.to_csv(index=False)

    prompt = f"""
You are a Manufacturing Engineer AI embedded in a PLM → MES → ERP pipeline.

TASK:
Transform the given Engineering BOM (eBOM) into a Manufacturing BOM (mBOM) using FACTORY INVENTORY first, then manufacturing reasoning.

### INPUT DATA
EBOM (CSV):
{ebom_context}

INVENTORY (CSV):
{inv_context}

GLOBAL RULES:
{global_context}

### STRUCTURE & LOGIC RULES
1. Structure:
   - BOM_Level 0 = FG (final product)
   - BOM_Level 1 = SUBASM (sub-assemblies)
   - BOM_Level 2+ = Parts / Consumables / Packaging
2. Synthetic IDs:
   - If sub-assemblies do not exist in eBOM, create SUBASM-001, SUBASM-002, etc.
3. Inventory precedence:
   - If part exists in INVENTORY, use its Bin_Location, Work_Center, Make_Buy, and availability logic.
4. Make/Buy:
   - Use inventory Make/Buy if present.
   - Else: Fasteners=BUY, Fabricated=MAKE, Consumables=BUY.
5. UOM:
   - Fasteners=EA
   - Fluids/adhesives/chemicals=L or ML
   - Sheets/metals=KG
6. Work_Center:
   - Choose only from: Welding, Assembly, QC, Packaging.
7. Backflush:
   - Yes for fasteners & consumables.
   - No for high-value parts.
8. Scrap_Pct:
   - Default 2 for fabricated parts, 0 for bought-out items.
9. Plant:
   - Default PLANT-01 if not present in inventory.

### ADDITIONAL MANUFACTURING COLUMNS (MUST BE INCLUDED)
- Consumables: 
  - List any glue, adhesive, sealant, coolant, grease required for the operation.
  - If none required, write NA.
- Packing_Details:
  - Examples: box, pallet, shrink-wrap, foam insert.
- Item_Alternatives:
  - If inventory shows substitutes/alternates, list part numbers separated by commas.
  - Else NA.
- Effectivity_Date:
  - Use inventory effectivity date if present.
  - Else default = 2025-01-01.

### OUTPUT FORMAT (STRICT)
Return ONLY rows separated by NEWLINES.
Each row MUST use PIPE | as delimiter.
Do NOT add headers.
Do NOT use markdown.
Do NOT output null or None (use NA).

Column Order (STRICT):
Parent_Part_No |
Child_Part_No |
Description |
Qty_Per |
UOM |
BOM_Level |
Op_Sequence |
Work_Center |
Make_Buy |
Backflush_Ind |
Scrap_Pct |
Plant |
Bin_Location |
Consumables |
Packing_Details |
Item_Alternatives |
Effectivity_Date

### OUTPUT EXAMPLE (FORMAT ONLY – DO NOT COPY CONTENT)
PARENT-001|SUBASM-001|WELDED FRAME|1|EA|1|10|Welding|MAKE|No|2|PLANT-01|A1-10|welding flux|pallet|ALT-001,ALT-002|2025-01-01
SUBASM-001|PART-99|STEEL PLATE|2|KG|2|10|Welding|BUY|Yes|0|PLANT-01|B2-05|NA|box|NA|2025-01-01

### TASK
Generate the mBOM for the provided input.

After the table, add:
---
ASSUMPTIONS:
- short bullets
MAPPING_NOTES:
- short bullets

No markdown. No explanations before the table.
"""
    response = ollama.generate(
        model=MODEL_NAME,
        prompt=prompt,
        options={"temperature": 0.0,"top_p": 0.9}
    )

    raw = response["response"]
    lines = [l.strip() for l in raw.split("\n") if "|" in l]

    EXPECTED_COLUMNS = [
    "Part_Number","Part_Name","Quantity","UOM",
    "BOM_Level","Assembly_or_Subassembly","Assembly_Sequence","Revision",
    "Processing_Steps", "Backflush_Ind","Scrap_Pct","Plant","Bin_Location"
]

    cleaned_rows = []

    for row in lines:
        row = list(row.split("|"))

        # If row has fewer columns → pad with None
        if len(row) < len(EXPECTED_COLUMNS):
            row += [None] * (len(EXPECTED_COLUMNS) - len(row))

        # If row has more columns → trim extra
        if len(row) > len(EXPECTED_COLUMNS):
            row = row[:len(EXPECTED_COLUMNS)]

        cleaned_rows.append(row)

    df_final = pd.DataFrame(cleaned_rows, columns=EXPECTED_COLUMNS)

    df_final = attach_type_and_confidence(df_final, used_inventory=True)


    return df_final
