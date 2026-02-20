import pandas as pd
import ollama
import json
import re
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional, List, Union, Dict, Tuple

import os

GLOBAL_RULES_PATH = "federated/global_rules.json"

def load_global_rules():
    if os.path.exists(GLOBAL_RULES_PATH):
        with open(GLOBAL_RULES_PATH, "r") as f:
            return json.load(f)
    return {}

print("--- Engine.py Loaded: AGGRESSIVE MERGE MODE (Verified) ---")

MODEL_NAME = "llama3.2:3b"

def _col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.strip().lower()
        if key in cols:
            return cols[key]
    return None


def _parse_date(value) -> str:
    if pd.isna(value):
        return ""
    s = str(value).strip()
    for fmt in ("%d.%m.%Y", "%d/%m/%Y", "%Y-%m-%d"):
        try:
            return str(datetime.strptime(s, fmt).date())
        except Exception:
            pass
    return s

def normalize_ebom(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    part_no = _col(df, ["part number", "part_number", "part_no", "partno"])
    part_name = _col(df, ["part name", "part_name", "description", "desc", "item name"])
    parent_asm = _col(df, ["parent assembly", "parent_assembly", "parent", "parent name", "parent_description"])
    qty = _col(df, ["quantity", "qty", "qty_per", "qty per"])
    rev = _col(df, ["revision", "rev"])
    part_type = _col(df, ["part type", "part_type", "type"])
    std_custom = _col(df, ["standard vs custom", "standard_vs_custom", "std_custom"])
    valid_from = _col(df, ["valid from", "valid_from", "effective date", "start date"])

    if not part_no or not part_name:
        raise ValueError("eBOM requires Part Number and Part Name")

    out = pd.DataFrame()
    out["Part Number"] = df[part_no].astype(str).str.strip()
    out["Part Name"] = df[part_name].astype(str).str.strip()
    out["Parent Assembly"] = df[parent_asm].astype(str).str.strip() if parent_asm else ""
    out["Quantity"] = pd.to_numeric(df[qty], errors="coerce").fillna(1).astype(int) if qty else 1
    out["Revision"] = df[rev].astype(str).str.strip() if rev else "NA"
    out["Part Type"] = df[part_type].astype(str).str.strip() if part_type else ""
    out["Standard vs Custom"] = df[std_custom].astype(str).str.strip() if std_custom else ""
    out["Valid From"] = df[valid_from].apply(_parse_date) if valid_from else ""

    return out.replace({"nan": "", "NaN": ""})

def normalize_inventory(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            "Part Number", "Part Name", "In_Inventory", "Stock_Qty",
            "Store_Location", "Approved_Supplier", "Lead_Time_Days"
        ])

    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    pn = _col(df, ["part number", "material", "item code"])
    pname = _col(df, ["part name", "description", "item name"])
    qty = _col(df, ["stock_qty", "quantity", "on hand"])
    loc = _col(df, ["store location", "location", "bin"])
    inv_flag = _col(df, ["in inventory", "available"])
    sup = _col(df, ["approved supplier", "supplier"])
    lead = _col(df, ["lead time", "lead_time_days"])

    out = pd.DataFrame()
    out["Part Number"] = df[pn].astype(str).str.strip() if pn else ""
    out["Part Name"] = df[pname].astype(str).str.strip() if pname else ""
    out["Stock_Qty"] = pd.to_numeric(df[qty], errors="coerce").fillna(0).astype(int) if qty else 0
    out["Store_Location"] = df[loc].astype(str).str.strip() if loc else ""
    out["Approved_Supplier"] = df[sup].astype(str).str.strip() if sup else ""
    out["Lead_Time_Days"] = pd.to_numeric(df[lead], errors="coerce").fillna(0).astype(int) if lead else 0

    if inv_flag:
        out["In_Inventory"] = df[inv_flag].astype(str).str.strip()
    else:
        out["In_Inventory"] = out["Stock_Qty"].apply(lambda x: "Yes" if x > 0 else "No")

    return out.replace({"nan": "", "NaN": ""})

ELECTRONICS_KW = [
    "pcb", "board", "ic", "capacitor", "resistor", "diode",
    "mosfet", "controller", "transformer", "connector", "cable", "wire", "fuse"
]

def node_type_from_part_type(part_type: str) -> str:
    pt = (part_type or "").lower()
    if pt == "assembly":
        return "Assembly"
    if "sub" in pt:
        return "Sub-Assembly"
    return "Component"

def makebuy_rule(part_type: str, std_custom: str, part_name: str) -> str:
    pt = (part_type or "").lower()
    sc = (std_custom or "").lower()
    nm = (part_name or "").lower()

    if "assembly" in pt:
        return "Make"
    if "mechanical" in pt and sc == "custom":
        return "Make"
    if sc == "standard":
        return "Buy"
    if any(k in nm for k in ELECTRONICS_KW):
        return "Buy"
    return "Make" if "mechanical" in pt else "Buy"

def build_parent_child(ebom: pd.DataFrame) -> pd.DataFrame:
    name_to_pn = dict(zip(ebom["Part Name"], ebom["Part Number"]))

    def resolve_parent(parent_name: str):
        if not parent_name:
            return ""
        return name_to_pn.get(parent_name.strip(), "")

    ebom["Parent Part Number"] = ebom["Parent Assembly"].apply(resolve_parent)
    return ebom


def compute_levels(ebom: pd.DataFrame):
    children = {}
    for _, r in ebom.iterrows():
        p = r["Parent Part Number"]
        c = r["Part Number"]
        if p:
            children.setdefault(p, []).append(c)

    roots = ebom.loc[ebom["Parent Part Number"] == "", "Part Number"].tolist()

    level = {}
    path = {}

    def dfs(node, lvl, stack):
        new_stack = stack + [node]
        level[node] = lvl
        path[node] = " > ".join(new_stack)
        for ch in children.get(node, []):
            dfs(ch, lvl + 1, new_stack)

    for r in roots:
        dfs(r, 0, [])

    return level, path, roots

def get_ai_consumable(description, material):
    prompt = f'''
    You're a Manufacturing Engineer.
    Item: {description}
    Material: {material}
    Question: What implies consumable is needed for assembly? (e.g., Glue, Grease, Solder, Cable Tie).
    Answer in 1 word only. If none, say NA.
    '''
    try:
        response = ollama.generate(
            model=MODEL_NAME,
            prompt=prompt,
            options={"temperature": 0.0, "num_predict": 10}
        )
        return response['response'].strip().replace(".", "")
    except:
        return "NA"

def heuristic_fill_consumables(row):
    desc = str(row.get("Description", "")).lower()
    mat = str(row.get("Material", "")).lower()
    
    if "plastic" in mat or "housing" in desc or "shell" in desc: return "Adhesive"
    if "metal" in mat or "screw" in desc or "bolt" in desc: return "Lubricant"
    if "pcb" in desc or "board" in desc: return "Solder"
    if "cable" in desc or "wire" in desc: return "Cable Tie"
    return "NA"

def smart_find_column(df, candidates):
    df_cols_lower = [c.lower().strip() for c in df.columns]
    for candidate in candidates:
        if candidate in df_cols_lower:
            return df.columns[df_cols_lower.index(candidate)]
    return None

def normalize_columns_strictly(df):
    df = df.fillna("NA") 
    new_df = pd.DataFrame()

    print(f"DEBUG: Found CSV Columns: {list(df.columns)}")

    parent_col = smart_find_column(df, ['parent assembly', 'parent','Parent Part Number', 'parent part', 'parent_part_no', 'part_number'])
    if parent_col:
        new_df['Parent_Part_No'] = df[parent_col]
    else:
        print("WARNING: Could not find 'Parent Assembly' column. Defaulting to 'Top Level'.")
        new_df['Parent_Part_No'] = "Top Level"

    desc_col = smart_find_column(df, ['part_name', 'description', 'desc', 'item name','Parent Description'])
    if desc_col:
        new_df['Description'] = df[desc_col]
    else:
        new_df['Description'] = "Unknown Part"

    id_col = smart_find_column(df, ['revision','Revision'])
    if id_col:
        new_df['Revision'] = df[id_col]
    else:
        new_df['Revision'] = "NA"

    qty_col = smart_find_column(df, ['quantity', 'qty', 'qty_per', 'amount','weight','Qty'])
    if qty_col:
        new_df['Qty_Per'] = pd.to_numeric(df[qty_col], errors='coerce').fillna(1)
    else:
        new_df['Qty_Per'] = 1

    mb_col = smart_find_column(df, ['standard_vs_custom', 'make_buy', 'source'])
    if mb_col:
         new_df['Make_Buy'] = df[mb_col].apply(lambda x: "Make" if "Custom" in str(x) else "Buy")
    else:
         new_df['Make_Buy'] = "Buy"

    mat_col = smart_find_column(df, ['material', 'raw material'])
    if mat_col:
        new_df['Material'] = df[mat_col]
    else:
        new_df['Material'] = ""

    return new_df

def generate_mbom_with_inventory(ebom_df, inv_df):
    print(f"Step 1: Input Rows: {len(ebom_df)}")
    
    clean_df = normalize_columns_strictly(ebom_df)
    clean_df = clean_df.fillna("NA")
    
    print("Step 2: Aggregating by NAME (Ignoring Unique IDs)...")
    
    aggregated_df = clean_df.groupby(
        ['Parent_Part_No', 'Description', 'Make_Buy', 'Material'], 
        as_index=False
    ).agg({
        'Qty_Per': 'sum',
        'Revision':'first'
    })
    
    print(f"Step 3: Aggregated Rows: {len(aggregated_df)}")

    if inv_df is not None and not inv_df.empty:
        inv_preview = inv_df.fillna("").to_csv(index=False)
    else:
        inv_preview = "NO INVENTORY DATA PROVIDED"

    ebom_preview = aggregated_df.fillna("").to_csv(index=False)

    global_rules = load_global_rules()

    prompt=f"""
You are a Senior Manufacturing Engineer working in an SAP/ERP environment.

Convert the provided eBOM into a manufacturing-ready mBOM WITH inventory awareness.

You are given:
1. Aggregated eBOM data
2. Inventory Master data

-------------------------------------------------------
eBOM DATA (CSV)
-------------------------------------------------------
{ebom_preview}

-------------------------------------------------------
INVENTORY MASTER (CSV)
-------------------------------------------------------
{inv_preview}

Follow these rules strictly.

STEP 1: BUILD MULTI-LEVEL HIERARCHY
- Use Parent Assembly column to map parent-child relationship.
- If Parent Assembly matches a Part Name, map to its Part Number.
- Level 0 = Final Product (no parent)
- Level 1 = Direct children
- Continue recursively
- Maintain Hierarchy Path for sorting

-------------------------------------------------------
STEP 2: CLASSIFY NODE TYPE
-------------------------------------------------------
- Assembly keywords → Assembly
- Sub keywords → Sub-Assembly
- Else → Component

-------------------------------------------------------
STEP 3: DETERMINE MAKE / BUY
-------------------------------------------------------
Rules:
- Assemblies → Make
- Custom Mechanical parts → Make
- Standard parts → Buy
- Electronics (PCB, capacitor, resistor, IC, connector, cable, fuse, transformer) → Buy

-------------------------------------------------------
STEP 4: INVENTORY VALIDATION
-------------------------------------------------------
If Make item:
   Inventory Status = "N/A (Manufactured Item)"

If Buy item:
   If Stock_Qty > 0:
        Inventory Status = "Available in Stock"
        Procurement Action = "Issue from Stores"
   Else:
        Inventory Status = "Not Available"
        Procurement Action = "Trigger Purchase Requisition"

-------------------------------------------------------
STEP 5: ASSIGN WORK CENTER
-------------------------------------------------------
If Make:
   Assembly → Final Assembly Line
   Mechanical → Injection Molding / Mechanical Assembly
   PCB → SMT Line
   Other → Manufacturing Cell

If Buy:
   Electronics → Incoming Inspection (Electronics)
   Other → Incoming Inspection (General)

-------------------------------------------------------
STEP 6: PROCUREMENT STEPS
-------------------------------------------------------
If Buy & Available:
   Material Issue from Stores -> Line Supply

If Buy & Not Available:
   Vendor Selection -> PR -> PO -> GRN -> Incoming QC -> Putaway -> Issue to Line

If Make:
   Issue Components -> Manufacture/Assemble -> In-process QC -> Final Test -> FG Receipt

-------------------------------------------------------
STEP 7: ROUTING OPERATIONS
-------------------------------------------------------
If Buy:
   10: PR/PO | 20: Incoming Inspection | 30: Putaway/Issue

If Make Assembly:
   10: Kitting/Issue | 20: Assembly | 30: Functional Test | 40: Packing

If Make Component:
   10: Material Issue | 20: Primary Process | 30: Finishing | 40: In-Process Inspection

-------------------------------------------------------
STEP 8: OUTPUT FORMAT (CSV TABLE)
-------------------------------------------------------
Level,Parent Part Number,Parent Description,Child Part Number,Child Description,Qty,UOM,Revision,Node Type,Make/Buy,Inventory Status,Stock_Qty,Store_Location,Procurement Action,Approved_Supplier,Lead_Time_Days,Work Center,Effective Date,Procurement Steps,Operations (Routing Embedded),Hierarchy Path
"""
    try:
        response = ollama.generate(
            model=MODEL_NAME,
            prompt=prompt,
            options={"temperature": 0.2}
        )

        print("AI Manufacturing Reasoning Output:")
        print(response["response"])

    except Exception as e:
        print("AI call failed:", e)

    final_rows = []
    for index, row in aggregated_df.iterrows():
        item = row.to_dict()
        
        cons = heuristic_fill_consumables(item)
        item['Consumables'] = cons
        
        ebom_name = item["Description"]

        if ebom_name in global_rules:
            item["Description"] = global_rules[ebom_name]

        final_rows.append(item)

    final_df = pd.DataFrame(final_rows)
    
    expected_cols = [
        "Parent_Part_No", "Revision", "Description", "Qty_Per", "UOM",
        "Make_Buy", "Work_Center", "Consumables"
    ]
    
    if final_df.empty:
        final_df = pd.DataFrame(columns=expected_cols)
    else:
        final_df['UOM'] = "EA"
        for col in expected_cols:
            if col not in final_df.columns:
                final_df[col] = "NA"
    
    final_df = final_df.fillna("NA")
    
    print(f"Success! Returning {len(final_df)} rows.")
    return final_df[expected_cols]

def generate_mbom(ebom_df, inv_df=None):
    if inv_df is None: inv_df = pd.DataFrame()

    return generate_mbom_with_inventory(ebom_df, inv_df)
