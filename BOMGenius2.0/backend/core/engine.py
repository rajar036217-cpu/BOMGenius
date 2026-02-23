import os
import re
import json
import pandas as pd
import ollama
from datetime import datetime, time
import time
from io import StringIO
from typing import Optional, Dict, List, Tuple

print("=== HYBRID mBOM ENGINE LOADED (AI + Deterministic Validation) ===")

MODEL_NAME = "llama3.2:3b"
GLOBAL_RULES_PATH = "federated/global_rules.json"

# =========================================================
# Utilities
# =========================================================

def load_global_rules():
    if os.path.exists(GLOBAL_RULES_PATH):
        with open(GLOBAL_RULES_PATH, "r") as f:
            return json.load(f)
    return {}

GLOBAL_RULES = load_global_rules()

def _clean_str(x):
    if x is None or pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in ["nan", "none", "null"]:
        return ""
    return s

def smart_find_column(df, candidates):
    norm = {}
    for c in df.columns:
        key = re.sub(r"[\s_]+", "", str(c).strip().lower())
        norm[key] = c

    for cand in candidates:
        key = re.sub(r"[\s_]+", "", str(cand).strip().lower())
        if key in norm:
            return norm[key]

    for cand in candidates:
        key = re.sub(r"[\s_]+", "", str(cand).strip().lower())
        for k, real in norm.items():
            if key in k:
                return real
    return None

# =========================================================
# Normalization
# =========================================================

def normalize_ebom(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    part_no = smart_find_column(df, ["part number", "part_no", "partno"])
    part_name = smart_find_column(df, ["part name", "description"])
    parent_asm = smart_find_column(df, ["parent assembly", "parent"])
    qty = smart_find_column(df, ["quantity", "qty"])
    rev = smart_find_column(df, ["revision"])
    part_type = smart_find_column(df, ["part type"])
    std_custom = smart_find_column(df, ["standard vs custom"])
    valid_from = smart_find_column(df, ["valid from"])

    out = pd.DataFrame()
    out["Part Number"] = df[part_no].apply(_clean_str)
    out["Part Name"] = df[part_name].apply(_clean_str)
    out["Parent Assembly"] = df[parent_asm].apply(_clean_str) if parent_asm else ""
    out["Quantity"] = pd.to_numeric(df[qty], errors="coerce").fillna(1).astype(int) if qty else 1
    out["Revision"] = df[rev].apply(_clean_str) if rev else "NA"
    out["Part Type"] = df[part_type].apply(_clean_str) if part_type else ""
    out["Standard vs Custom"] = df[std_custom].apply(_clean_str) if std_custom else ""
    out["Valid From"] = df[valid_from].apply(_clean_str) if valid_from else ""

    return out.fillna("")

# =========================================================
# Hierarchy
# =========================================================

def build_parent_child(ebom):
    name_to_pn = dict(zip(ebom["Part Name"], ebom["Part Number"]))

    def resolve_parent(x):
        return name_to_pn.get(_clean_str(x), "")

    ebom["Parent Part Number"] = ebom["Parent Assembly"].apply(resolve_parent)
    return ebom

def compute_levels(ebom):
    children = {}
    for _, r in ebom.iterrows():
        parent = _clean_str(r["Parent Part Number"])
        child = _clean_str(r["Part Number"])
        if parent:
            children.setdefault(parent, []).append(child)

    roots = ebom.loc[ebom["Parent Part Number"] == "", "Part Number"].tolist()

    level = {}
    path = {}

    def dfs(node, lvl, stack):
        level[node] = lvl
        path[node] = " > ".join(stack + [node])
        for ch in children.get(node, []):
            dfs(ch, lvl + 1, stack + [node])

    for r in roots:
        dfs(r, 0, [])

    return level, path

# =========================================================
# Deterministic Rules
# =========================================================

def node_type(part_type):
    pt = part_type.lower()
    if "assembly" in pt:
        return "Assembly"
    return "Component"

def makebuy(part_type, std_custom):
    pt = part_type.lower()
    sc = std_custom.lower()

    if "assembly" in pt:
        return "Make"
    if sc == "custom":
        return "Make"
    return "Buy"

def work_center(make_buy, node_type):
    if make_buy == "Make":
        return "Final Assembly Line"
    return "Incoming Inspection (General)"

def procurement_steps(make_buy):
    if make_buy == "Make":
        return "Issue Components -> Assemble -> In-process QC -> Final Test -> FG Receipt"
    return "Vendor Selection -> PR -> PO -> GRN -> Incoming QC -> Putaway -> Issue to Line"

def routing(make_buy):
    if make_buy == "Make":
        return "10: Kitting | 20: Assembly | 30: Functional Test | 40: Packing"
    return "10: PR/PO | 20: Incoming Inspection | 30: Putaway"

# =========================================================
# AI Draft Generator
# =========================================================

def ai_enrich(ebom_df, inv_df):
    prompt = f"""
SYSTEM:
You are a strict ERP/SAP Manufacturing Engineer assistant.
You ONLY output valid JSON. No markdown. No commentary.

TASK:
We already ran a deterministic RULE ENGINE to create base rows (ground truth).
You must enrich each row with ONLY the required AI fields.

HARD RULES:
1) Output MUST be a JSON array of length = {len(ebom_df)}.
2) Preserve row order exactly. 1st output object enriches 1st input row, etc.
3) Do NOT change or add base fields. Only output the enrichment fields listed below.
4) If unknown, use "NA" (string), 0 for numbers, [] for arrays.
5) Never include extra keys. Never include explanations.

ENRICHMENT FIELDS (output keys exactly as below):
- "Node Type" : one of ["Assembly","Sub-Assembly","Component","Material","Process","Packaging"]
- "Make/Buy" : one of ["Make","Buy"]
- "Inventory Status" : one of ["In Stock","Low Stock","Out of Stock","Unknown"]
- "Stock_Qty" : number
- "Store_Location" : string
- "Procurement Action" : one of ["Issue from Stock","Purchase","Manufacture","Expedite","NA"]
- "Approved_Supplier" : string
- "Lead_Time_Days" : number
- "Work Center" : one of ["SMT_LINE","REFLOW_OVEN","AOI_STATION","PCB_TEST","MECH_LINE","QA_STATION","BURN_IN_RACK","PACK_LINE","NA"]
- "Procurement Steps" : array of short strings (max 6 items)
- "Operations (Routing Embedded)" : array of objects with keys:
    - "Op_Seq" (number)
    - "Operation" (string)
    - "Work_Center" (same enum list as above)

CLASSIFICATION HEURISTICS:
- If Child Description contains words like ["process","test","inspection","burn-in","hipot","reflow","smt","aoi"] -> Node Type = "Process"
- If contains ["label","box","packing","carton","manual","sticker"] -> Node Type = "Packaging"
- If contains ["pcb","board","ic","resistor","capacitor","diode","mosfet","transformer","fuse"] -> likely "Component"
- If contains ["assembly","sub assembly","adapter","housing","case"] -> "Assembly" or "Sub-Assembly"
- Cables and housings are usually "Buy" unless explicitly stated otherwise in base row.

INVENTORY MATCHING:
Use inventory_json (if provided) to find matching item by:
1) exact match on Child Part Number if present in inventory
2) else fuzzy match using Child Description keywords
If no match -> Inventory Status = "Unknown", Stock_Qty = 0, Store_Location="NA".

INPUT:
BOM_ROWS (ground truth):
{json.dumps(ebom_df, ensure_ascii=False)}

INVENTORY_JSON:
{inv_df if inv_df is not None else "[]"}

OUTPUT:
Return ONLY the JSON array.
"""

    response = ollama.generate(
        model=MODEL_NAME,
        prompt=prompt,
        options={
            "temperature": 0.0,
            "num_predict": 1500
        }
    )

    text = response["response"]

    try:
        start = text.index("[")
        end = text.rindex("]") + 1
        return json.loads(text[start:end])
    except:
        raise ValueError("AI did not return valid JSON.")

def validate_mbom_output(base_rows, ai_rows, required_columns):

    if not isinstance(ai_rows, list):
        raise ValueError("AI output is not a list.")

    if len(base_rows) != len(ai_rows):
        raise ValueError("AI row count mismatch.")

    for idx, row in enumerate(ai_rows):

        if not isinstance(row, dict):
            raise ValueError(f"Row {idx} is not a dictionary.")

        for col in required_columns:
            if col not in row:
                raise ValueError(f"Missing column '{col}' in row {idx}.")

    return True

# ---------------------------------------------------
# MAIN ENGINE
# ---------------------------------------------------

def generate_mbom_with_inventory(ebom_df, inv_df):

    start_total = time.perf_counter()

    start_pre = time.perf_counter()
    clean_df = normalize_ebom(ebom_df)
    #clean_df = build_hierarchy(clean_df)
    end_pre = time.perf_counter()

    inventory_json = inv_df.to_json(orient="records")
    base_rows = clean_df.to_dict(orient="records")

    start_ai = time.perf_counter()
    ai_data = ai_enrich(base_rows, inventory_json)
    end_ai = time.perf_counter()

    if len(ai_data) != len(base_rows):
        raise ValueError("AI row count mismatch.")

    start_post = time.perf_counter()
    final_rows = []
    for base, ai in zip(base_rows, ai_data):
        merged = {**base, **ai}
        final_rows.append(merged)

    final_df = pd.DataFrame(final_rows)
    end_post = time.perf_counter()

    end_total = time.perf_counter()

    print("\n--- PERFORMANCE METRICS ---")
    print(f"Preprocessing Time   : {end_pre - start_pre:.4f} sec")
    print(f"AI Generation Time   : {end_ai - start_ai:.4f} sec")
    print(f"Post-processing Time : {end_post - start_post:.4f} sec")
    print(f"Total MBOM Time      : {end_total - start_total:.4f} sec\n")

    return final_df


def generate_mbom(ebom_df, inv_df=None):
    if inv_df is None:
        inv_df = pd.DataFrame()
    return generate_mbom_with_inventory(ebom_df, inv_df)
