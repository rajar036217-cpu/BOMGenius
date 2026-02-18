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