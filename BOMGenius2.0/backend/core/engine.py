import pandas as pd
import ollama
from sentence_transformers import SentenceTransformer, util
import json
import os
from pydantic import BaseModel, Field
from typing import Optional, List, Union

# --- CONFIGURATION ---
MODEL_NAME = "llama3.2:3b"
CHUNK_SIZE = 3  # Reduced to 3 for instant CPU processing

# --- PYDANTIC SCHEMA ---
class MBOMItem(BaseModel):
    Parent_Part_No: Optional[str] = Field(default="NA")
    Child_Part_No: Optional[str] = Field(default="NA")
    Description: Optional[str] = Field(default="NA")
    Qty_Per: Union[str, float, int, None] = Field(default=1)
    UOM: Optional[str] = Field(default="EA")      
    BOM_Level: Union[str, int, None] = Field(default=1)
    Op_Sequence: Union[str, int, None] = Field(default=10)
    Work_Center: Optional[str] = Field(default="Assembly")
    Make_Buy: Optional[str] = Field(default="Make")
    Plant: Optional[str] = Field(default="PLANT-01")
    Bin_Location: Optional[str] = Field(default="NA")
    Consumables: Optional[str] = Field(default="NA")

class MBOMItems(BaseModel):
    items: List[MBOMItem] = Field(default_factory=list)

schema = MBOMItems.model_json_schema()

# --- HELPER FUNCTIONS ---
CANONICAL_FIELDS = {
    "part_name": ["name", "desc", "description", "item", "part name"],
    "part_no": ["part", "number", "pn", "id", "code", "part_id"], 
    "bin_location": ["bin", "location", "loc", "warehouse", "stock", "store", "depot"],
    "work_center": ["wc", "workcenter", "work center", "dept", "shop", "line"],
}

schema_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
CANONICAL_EMBEDDINGS = {k: schema_model.encode(v) for k, v in CANONICAL_FIELDS.items()}
GLOBAL_RULES_FILE = "federated/global_rules.json"

def load_global_rules():
    if os.path.exists(GLOBAL_RULES_FILE):
        with open(GLOBAL_RULES_FILE, "r") as f: return json.load(f)
    return {}

def learn_schema(df):
    col_map = {}
    for col in df.columns:
        col_emb = schema_model.encode(col.lower())
        best_match, best_score = None, 0
        for canon, emb_list in CANONICAL_EMBEDDINGS.items():
            score = util.cos_sim(col_emb, emb_list).max().item()
            if score > best_score: best_score, best_match = score, canon
        col_map[col] = best_match if best_score > 0.55 else "unknown_" + col
    return col_map

def normalize_inventory(df, col_map):
    norm = pd.DataFrame()
    for raw_col, canon in col_map.items(): norm[canon] = df[raw_col]
    return norm

def classify_item_type(desc: str) -> str:
    d = (desc or "").lower()
    if any(k in d for k in ["bolt", "nut", "washer", "screw"]): return "Fasteners"
    if any(k in d for k in ["grease", "oil", "paint"]): return "Consumables"
    return "Standard Parts"

def attach_type_and_confidence(df, used_inventory: bool):
    if df.empty: return pd.DataFrame(columns=["Description", "Make_Buy", "Work_Center", "Item_Type", "Confidence_Score"])
    for col in ["Description", "Make_Buy", "Work_Center"]:
        if col not in df.columns: df[col] = "NA"
    df["Confidence_Score"] = 0.85 if used_inventory else 0.75
    df["Item_Type"] = df["Description"].apply(lambda x: classify_item_type(str(x)))
    return df

def normalize_llm_keys(item_dict):
    # Simplified Normalizer
    KEY_MAP = {
        "parent": "Parent_Part_No", "part_no": "Parent_Part_No", "pn": "Parent_Part_No",
        "child": "Child_Part_No", "child_pn": "Child_Part_No", "component": "Child_Part_No",
        "desc": "Description", "name": "Description", "qty": "Qty_Per", "uom": "UOM",
        "level": "BOM_Level", "op": "Op_Sequence", "wc": "Work_Center", "make": "Make_Buy",
        "bin": "Bin_Location", "cons": "Consumables"
    }
    new_item = {}
    for k, v in item_dict.items():
        k_lower = k.lower().replace(" ", "_").strip()
        found = False
        for correct_key in MBOMItem.__annotations__.keys():
            if k_lower == correct_key.lower():
                new_item[correct_key] = v
                found = True
                break
        if not found:
            for keyword, target in KEY_MAP.items():
                if keyword in k_lower:
                    new_item[target] = v
                    found = True
                    break
        if not found: new_item[k] = v
    return new_item

# --- TURBO CHUNKING LOGIC ---

def generate_chunk(ebom_chunk_df, inv_context_str):
    ebom_csv = ebom_chunk_df.to_csv(index=False)
    
    # TURBO PROMPT: Short & Direct
    prompt = f"""
Convert eBOM to mBOM JSON.

INPUT:
{ebom_csv}

INVENTORY:
{inv_context_str}

RULES:
1. Parent_Part_No = 'Parent Assembly' column.
2. Child_Part_No = 'Part_ID' column.
3. Description = 'Part Name' column.
4. Output JSON list.

JSON Example:
[{{ "Parent_Part_No": "A1", "Child_Part_No": "B2", "Description": "Bolt", "Qty_Per": 1, "Make_Buy": "Buy" }}]
"""
    try:
        response = ollama.generate(
            model=MODEL_NAME,
            format=schema, 
            prompt=prompt,
            # TURBO SETTINGS: Stop generation early to prevent hanging
            options={"temperature": 0.0, "num_predict": 256, "top_p": 0.9} 
        )
        response_json = json.loads(response['response'])
        raw_items = response_json.get('items', [])
        return [normalize_llm_keys(item) for item in raw_items]
        
    except Exception as e:
        print(f"Chunk Error: {e}")
        return []

def generate_mbom_with_inventory(ebom_df, inv_df):
    # PRE-PROCESSING: Force 'Part_ID' creation
    ebom_map = learn_schema(ebom_df)
    part_name_col = next((col for col, canon in ebom_map.items() if canon == "part_name"), None)
    
    if part_name_col:
        print(f"Auto-Creating 'Part_ID' from '{part_name_col}'...")
        ebom_df["Part_ID"] = ebom_df[part_name_col]
    
    # Context
    ebom_map = learn_schema(ebom_df)
    inv_map = learn_schema(inv_df)
    ebom_norm = normalize_inventory(ebom_df, ebom_map)
    inv_norm = normalize_inventory(inv_df, inv_map)
    
    inv_context_str = inv_norm.head(20).to_csv(index=False) # Tiny context for speed

    all_items = []
    print(f"Starting Turbo Processing (Chunk size: {CHUNK_SIZE})...")
    
    for i in range(0, len(ebom_norm), CHUNK_SIZE):
        chunk = ebom_norm.iloc[i : i + CHUNK_SIZE]
        print(f"Processing chunk {i}...")
        items = generate_chunk(chunk, inv_context_str)
        print(f"--- Chunk {i} Done ({len(items)} items) ---") # Progress Log
        all_items.extend(items)
        
    df_final = pd.DataFrame(all_items)
    
    expected_cols = [
        "Parent_Part_No", "Child_Part_No", "Description", "Qty_Per", "UOM",
        "BOM_Level", "Op_Sequence", "Work_Center", "Make_Buy", "Plant",
        "Bin_Location", "Consumables"
    ]
    
    for col in expected_cols:
        if col not in df_final.columns: df_final[col] = "NA"

    return attach_type_and_confidence(df_final[expected_cols], used_inventory=True)

def generate_mbom(ebom_df, inv_df=None):
    if inv_df is None: inv_df = pd.DataFrame()
    return generate_mbom_with_inventory(ebom_df, inv_df)