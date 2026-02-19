import pandas as pd
import ollama
import json
import re
from pydantic import BaseModel, Field
from typing import Optional, List, Union

import os

GLOBAL_RULES_PATH = "federated/global_rules.json"

def load_global_rules():
    if os.path.exists(GLOBAL_RULES_PATH):
        with open(GLOBAL_RULES_PATH, "r") as f:
            return json.load(f)
    return {}

print("--- Engine.py Loaded: AGGRESSIVE MERGE MODE (Verified) ---")

MODEL_NAME = "llama3.2:3b"

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

    global_rules = load_global_rules()

    prompt=f"""
You are a Senior Manufacturing Engineer working in an SAP/ERP environment.

Convert the provided eBOM into a manufacturing-ready mBOM WITH inventory awareness.

You are given:
1. eBOM data
2. Inventory Master data (Part Number, In_Inventory, Stock_Qty, Store_Location, Approved_Supplier, Lead_Time_Days)

Follow these rules strictly.

-------------------------------------------------------
STEP 1: BUILD MULTI-LEVEL HIERARCHY
-------------------------------------------------------

1. Use Parent Assembly column to map parent-child relationship.
2. If Parent Assembly matches a Part Name, map to its Part Number.
3. Level Rules:
   - Level 0 = Final Product (no parent)
   - Level 1 = Direct children of Level 0
   - Level 2 = Children of Level 1
   - Continue recursively
4. Never skip levels.
5. Maintain Hierarchy Path for sorting.

-------------------------------------------------------
STEP 2: CLASSIFY NODE TYPE
-------------------------------------------------------

- If Part Type = Assembly → Node Type = Assembly
- If Part Type contains "Sub" → Node Type = Sub-Assembly
- Else → Component

-------------------------------------------------------
STEP 3: DETERMINE MAKE / BUY
-------------------------------------------------------

Rules:

- Assemblies → Make
- Custom Mechanical parts → Make
- Standard parts → Buy
- Electronic components (PCB, capacitor, resistor, IC, connector, cable, fuse, transformer, etc.) → Buy

-------------------------------------------------------
STEP 4: INVENTORY VALIDATION LOGIC
-------------------------------------------------------

Use Inventory Master:

If Make item:
   Inventory Status = "N/A (Manufactured Item)"

If Buy item:
   If In_Inventory = Yes AND Stock_Qty > 0:
        Inventory Status = "Available in Stock"
        Procurement Action = "Issue from Stores"
   Else:
        Inventory Status = "Not Available"
        Procurement Action = "Trigger Purchase Requisition"
        Use Approved_Supplier and Lead_Time_Days

-------------------------------------------------------
STEP 5: ASSIGN WORK CENTER
-------------------------------------------------------

If Make:
   Assembly → Final Assembly Line
   Mechanical → Injection Molding / Mechanical Assembly
   PCB (Make) → SMT Line
   Other → Manufacturing Cell

If Buy:
   Electronics → Incoming Inspection (Electronics)
   Other → Incoming Inspection (General)

-------------------------------------------------------
STEP 6: GENERATE PROCUREMENT STEPS
-------------------------------------------------------

If Buy AND Not Available:
   "Vendor Selection -> PR -> PO -> GRN -> Incoming QC -> Putaway -> Issue to Line"

If Buy AND Available:
   "Material Issue from Stores -> Line Supply"

If Make:
   "Issue Components -> Manufacture/Assemble -> In-process QC -> Final Test -> FG Receipt"

-------------------------------------------------------
STEP 7: GENERATE ROUTING OPERATIONS
-------------------------------------------------------

If Buy:
   10: PR/PO (if required)
   20: Incoming Inspection
   30: Putaway or Issue

If Make Assembly:
   10: Kitting/Issue
   20: Assembly
   30: Functional Test
   40: Packing

If Make Component:
   10: Material Issue
   20: Primary Process
   30: Finishing
   40: In-Process Inspection

-------------------------------------------------------
STEP 8: OUTPUT FORMAT
-------------------------------------------------------

Return a SINGLE CONSOLIDATED TABLE with columns:

Level
Parent Part Number
Parent Description
Child Part Number
Child Description
Qty
UOM
Revision
Node Type
Make/Buy
Inventory Status
Stock_Qty
Store_Location
Procurement Action
Approved_Supplier
Lead_Time_Days
Work Center
Effective Date
Procurement Steps
Operations (Routing Embedded)
Hierarchy Path

-------------------------------------------------------
STRICT RULES
-------------------------------------------------------

- Do NOT invent parent-child relationships.
- Use only given data.
- Do NOT hallucinate inventory if not provided.
- Keep structure deterministic.
- Output clean CSV-style table."""

    try:
        response = ollama.generate(
            model=MODEL_NAME,
            prompt=prompt,
            options={"temperature": 0.0}
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
