import os
import re
import pandas as pd
import ollama
import json
import re
from pydantic import BaseModel, Field
from typing import Optional, List, Union

print("--- Engine.py Loaded: AGGRESSIVE MERGE MODE (Verified) ---")

MODEL_NAME = "llama3.2:3b"

def get_ai_consumable(description, material):
    prompt = f'''
    Act as a Manufacturing Engineer.
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

def llm_consumables_optional(part_name: str, material: str) -> str:
    """
    Optional LLM (Ollama) call.
    Toggle with env: USE_LLM_CONSUMABLES=1
    Keeps token cost under control; fallback always heuristic.
    """
    if os.getenv("USE_LLM_CONSUMABLES", "0") != "1":
        return heuristic_consumables(part_name, material)

    try:
        import ollama
        model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
        prompt = (
            "You are a manufacturing engineer.\n"
            f"Item: {part_name}\n"
            f"Material: {material}\n"
            "Return ONE consumable word needed for assembly (Glue/Adhesive, Grease, Solder, Cable Tie, Threadlocker).\n"
            "If none, return NA.\n"
            "Answer with ONLY the word."
        )
        resp = ollama.generate(model=model, prompt=prompt, options={"temperature": 0.0, "num_predict": 6})
        ans = str(resp.get("response", "")).strip()
        ans = re.sub(r"[^A-Za-z ]", "", ans).strip()
        if not ans:
            return heuristic_consumables(part_name, material)
        # normalize common outputs
        ans_low = ans.lower()
        if "glue" in ans_low or "adhes" in ans_low:
            return "Adhesive"
        if "solder" in ans_low:
            return "Solder"
        if "tie" in ans_low:
            return "Cable Tie"
        if "grease" in ans_low or "lub" in ans_low:
            return "Lubricant"
        if "thread" in ans_low or "locker" in ans_low:
            return "Thread Locker"
        if ans.upper() == "NA":
            return "NA"
        return ans.title()
    except Exception:
        return heuristic_consumables(part_name, material)

# =============================
# Hierarchy build
# =============================
def build_parent_child(ebom: pd.DataFrame) -> pd.DataFrame:
    ebom = ebom.copy()

    # map name -> part number (first occurrence)
    name_to_pn: Dict[str, str] = {}
    for _, r in ebom.iterrows():
        pn = _clean_str(r.get("Part Number"))
        nm = _clean_str(r.get("Part Name"))
        if pn and nm and nm not in name_to_pn:
            name_to_pn[nm] = pn

    def resolve_parent_pn(parent_asm_name: str) -> str:
        p = _clean_str(parent_asm_name)
        if not p:
            return ""
        return name_to_pn.get(p, "")

    ebom["Parent Part Number"] = ebom["Parent Assembly"].apply(resolve_parent_pn)
    return ebom

def compute_levels(ebom: pd.DataFrame) -> Tuple[Dict[str, int], Dict[str, str], List[str]]:
    children: Dict[str, List[str]] = {}
    for _, r in ebom.iterrows():
        child = _clean_str(r.get("Part Number"))
        parent = _clean_str(r.get("Parent Part Number"))
        if parent:
            children.setdefault(parent, []).append(child)

    roots = (
        ebom.loc[ebom["Parent Part Number"].astype(str).str.strip() == "", "Part Number"]
        .astype(str).str.strip().tolist()
    )

    level: Dict[str, int] = {}
    path: Dict[str, str] = {}

    def dfs(node: str, lvl: int, stack: List[str]):
        if node in stack:
            return
        new_stack = stack + [node]
        if node not in level or lvl < level[node]:
            level[node] = lvl
            path[node] = " > ".join(new_stack)
        for ch in children.get(node, []):
            dfs(ch, lvl + 1, new_stack)

    for rt in roots:
        if rt:
            dfs(rt, 0, [])

    # orphan safety
    for pn in ebom["Part Number"].astype(str).str.strip().tolist():
        if pn and pn not in level:
            dfs(pn, 0, [])

    return level, path, roots

# =============================
# Aggregation (rollup) - SAFE
# =============================
def smart_rollup_aggregation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine rows ONLY within same parent + same manufacturing classification.
    - Qty summed
    - Child Part Number joined (unique)
    - Hierarchy Path keep first
    """
    if df is None or df.empty:
        return df

    # ensure consumables exists for groupby
    if "Consumables" not in df.columns:
        df["Consumables"] = "NA"

    group_cols = [
        "Level",
        "Parent Part Number",
        "Parent Description",
        "Child Description",
        "UOM",
        "Node Type",
        "Make/Buy",
        "Work Center",
        "Procurement Steps",
        "Operations (Routing Embedded)",
        "Consumables",
    ]
    group_cols = [c for c in group_cols if c in df.columns]  # safety

    # build agg dict safely
    agg_dict = {
        "Qty": "sum",
        "Child Part Number": lambda x: ", ".join(pd.Series(x).astype(str).str.strip().replace("nan", "").unique()),
        "Hierarchy Path": "first",
        "Revision": "first",
        "Effective Date": "first",
    }

    # optional inventory cols
    for col in ["Inventory Status", "Stock_Qty", "Store_Location", "Procurement Action", "Approved_Supplier", "Lead_Time_Days"]:
        if col in df.columns:
            agg_dict[col] = "first"

    # Parent info should be stable (already in group_cols mostly)
    out = df.groupby(group_cols, as_index=False).agg(agg_dict)

    # Keep Parent Part Number empty display clean
    if "Parent Part Number" in out.columns:
        out["Parent Part Number"] = out["Parent Part Number"].replace({"nan": ""}).fillna("")

    return out

# =============================
# Generate mBOM
# =============================
def generate_mbom(ebom_df: pd.DataFrame, inv_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    ebom = normalize_ebom(ebom_df)
    ebom = build_parent_child(ebom)
    level_map, path_map, roots = compute_levels(ebom)

    pn_to_name = dict(zip(ebom["Part Number"], ebom["Part Name"]))

    inv_norm = normalize_inventory(inv_df) if (inv_df is not None and not inv_df.empty) else pd.DataFrame()

    inventory_active = (
        inv_norm is not None
        and not inv_norm.empty
        and (("Part Number" in inv_norm.columns and (inv_norm["Part Number"] != "").any())
             or ("Part Name" in inv_norm.columns and (inv_norm["Part Name"] != "").any()))
    )

    def inv_logic(child_pn: str, child_name: str, mb: str):
        if not inventory_active:
            return None

def normalize_columns_strictly(df):
    df = df.fillna("NA") 
    new_df = pd.DataFrame()

    print(f"DEBUG: Found CSV Columns: {list(df.columns)}")

    parent_col = smart_find_column(df, ['parent assembly', 'parent', 'parent part', 'parent_part_no'])
    if parent_col:
        new_df['Parent_Part_No'] = df[parent_col]
    else:
        print("WARNING: Could not find 'Parent Assembly' column. Defaulting to 'Top Level'.")
        new_df['Parent_Part_No'] = "Top Level"

    desc_col = smart_find_column(df, ['part name', 'description', 'desc', 'item name'])
    if desc_col:
        new_df['Description'] = df[desc_col]
    else:
        new_df['Description'] = "Unknown Part"

    id_col = smart_find_column(df, ['part number', 'part_number', 'part no', 'child_part_no', 'child'])
    if id_col:
        new_df['Child_Part_No'] = df[id_col]
    else:
        new_df['Child_Part_No'] = "NA"

    qty_col = smart_find_column(df, ['quantity', 'qty', 'qty_per', 'amount'])
    if qty_col:
        new_df['Qty_Per'] = pd.to_numeric(df[qty_col], errors='coerce').fillna(1)
    else:
        new_df['Qty_Per'] = 1

    mb_col = smart_find_column(df, ['standard vs custom', 'make_buy', 'source'])
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
        'Child_Part_No': 'first'
    })
    
    print(f"Step 3: Aggregated Rows: {len(aggregated_df)}")
    
    final_rows = []
    for index, row in aggregated_df.iterrows():
        item = row.to_dict()
        
        cons = heuristic_fill_consumables(item)
        item['Consumables'] = cons
        
        if item['Make_Buy'] == 'Make':
            item['Work_Center'] = "Assembly Line"
        else:
            item['Work_Center'] = "Store"
            
        final_rows.append(item)

    final_df = pd.DataFrame(final_rows)
    
    expected_cols = [
        "Parent_Part_No", "Child_Part_No", "Description", "Qty_Per", "UOM",
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