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

def _scalar(x, default="NA"):
    # if x is Series/list/tuple -> take first element
    try:
        if isinstance(x, pd.Series):
            return x.iloc[0] if len(x) else default
        if isinstance(x, (list, tuple)):
            return x[0] if len(x) else default
    except:
        pass
    return default if x is None else x

def attach_type_and_confidence(df, used_inventory: bool):
    df = df.copy()

    # make sure columns exist
    for col in ["Description", "Make_Buy", "Work_Center"]:
        if col not in df.columns:
            df[col] = "NA"

    # item type
    df["Item_Type"] = df["Description"].apply(lambda v: classify_item_type(str(_scalar(v))))

    # confidence (NO apply-return surprises)
    scores = []
    for _, r in df.iterrows():
        desc = str(_scalar(r.get("Description", "NA")))
        mb   = str(_scalar(r.get("Make_Buy", "NA")))
        wc   = str(_scalar(r.get("Work_Center", "NA")))
        scores.append(float(compute_confidence(desc, mb, wc, used_inventory)))

    df["Confidence_Score"] = scores
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
#Part_Number | Part_Name | Quantity | UOM | BOM_Level | Assembly_or_Subassembly | Assembly_Sequence | Revision | Processing_Steps
from pydantic import BaseModel, Field


class MBOMItem(BaseModel):
    Parent_Part_No: str = Field(default=None, description="Part number of the parent assembly. For top-level items, use the final product's part number.")
    Child_Part_No: str = Field(default=None, description="Part number of the child item. For subassemblies, use synthetic IDs like SUBASM-001.")
    Description: str = Field(default=None, description="Name of the part or assembly.")
    Qty_Per: float = Field(default=None, description="Quantity of the child part required per parent assembly.")
    UOM: str = Field(default=None, description="Unit of Measure. Use EA for discrete parts, L/ML/KG for fluids and metals.")      
    BOM_Level: int = Field(default=None, description="BOM level indicating hierarchy. 0 for top-level product, 1 for subassemblies, 2+ for parts under subassemblies.")
    Op_Sequence: int = Field(default=None, description="Numeric sequence for operations within the same assembly level, e.g., 10, 20, 30...")
    Work_Center: str = Field(default=None, description="Assigned work center for the operation, chosen from Welding, Assembly, QC, Packaging.")
    Make_Buy: str = Field(default=None, description="Indicates if the item is Made in-house or Bought out. Use inventory data if available, else infer based on rules.")
    Plant: str = Field(default=None, description="Manufacturing plant where the item is produced or sourced. Default to PLANT-01 if not specified in inventory.")
    Bin_Location: str = Field(default=None, description="Storage location for the item in the factory, derived from inventory if available.")
    Consumables: str = Field(default=None, description="List any consumables like glue, adhesive, sealant, coolant, grease required for the operation. If none required, write NA.")
    Packing_Details: str = Field(default=None, description="Details about packing requirements, e.g., box, pallet, shrink-wrap, foam insert.")
    Item_Alternatives: str = Field(default=None, description="If inventory shows substitutes/alternates, list part numbers separated by commas. Else NA.")
    Effectivity_Date: str = Field(default=None, description="Use inventory effectivity date if present, else default to 2025-01-01.")

class MBOMItems(BaseModel):
    items: list[MBOMItem] = Field(default_factory=list)

schema = MBOMItems.model_json_schema()

def generate_mbom_without_inventory(ebom_df):
    ebom_schema = learn_ebom_schema(ebom_df)
    normalized_ebom = normalize_inventory(ebom_df, ebom_schema)

    global_rules = load_global_rules()
    global_context = json.dumps(global_rules, indent=2)

    ebom_context = normalized_ebom.to_csv(index=False)
    
    # UPDATED PROMPT: Asks for JSON, not Pipes
    prompt = f"""
You are a Senior Manufacturing BOM Engineer AI.

Your task:
Convert the given Engineering BOM (eBOM) into a Manufacturing BOM (mBOM) WITHOUT using factory inventory.

EBOM INPUT:
{ebom_context}

GLOBAL RULES:
{global_context}

### OUTPUT FORMAT (STRICT)
Return strictly valid JSON matching the provided schema.
No markdown.
Do NOT add headers.

### LOGIC RULES:
1. Preserve hierarchy:
   - Top-level product = BOM_Level 0
   - Subassemblies = BOM_Level 1+
2. UOM rules:
   - Discrete parts → EA
   - Fluids, chemicals → L, ML, KG
   - Metals by weight → KG
3. Assembly_Sequence:
   - Assign numeric sequence (10, 20, 30...)
4. Revision:
   - Default = R1 unless eBOM specifies otherwise
5. Processing_Steps (Op_Sequence logic):
   - Fasteners → bolting
   - Sheet metal → stamping + welding + painting
   - Plastic → molding + trimming

Generate the mBOM items now.
"""

    response = ollama.generate(
        model=MODEL_NAME,
        format=schema,
        prompt=prompt,
        options={"temperature": 0.0, "top_p": 0.9}
    )

    # FIXED PARSING LOGIC: Handle JSON response instead of Pipes
    llm_text = response["response"]
    
    structured = None
    try:
        structured = json.loads(llm_text)
    except json.JSONDecodeError:
        # Fallback: try to find JSON if wrapped in other text
        import re
        json_match = re.search(r'\{[\s\S]*\}', llm_text)
        if json_match:
            try:
                structured = json.loads(json_match.group())
            except:
                pass

    if structured is None:
        print(f"LLM Raw Response (Failed Parse): {llm_text}") 
        # Return empty DF to avoid crash, or raise error
        return pd.DataFrame(columns=["Description", "Make_Buy", "Work_Center"])

    items = structured.get("items", [])
    
    # Ensure items is a list
    if not isinstance(items, list):
        items = []

    # Convert to DataFrame
    df_final = pd.DataFrame(items)

    # Attach confidence scores
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
1. Synthetic IDs:
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

### TASK
Generate the mBOM for the provided input.

No markdown. No explanations before the table.
"""
    response = ollama.generate(
        model=MODEL_NAME,
        format=schema,
        prompt=prompt,
        options={"temperature": 0.0,"top_p": 0.9}
    )

    response_dict = json.loads(response)  # Convert string to dictionary
    raw = response_dict["response"]["items"]
    print("LLM Raw Response:", raw)

    lines = [l.strip() for l in raw.split("\n") if "|" in l]

    EXPECTED_COLUMNS = [
    "Parent_Part_No","Child_Part_No","Description","Qty_Per","UOM",
    "BOM_Level","Op_Sequence","Work_Center","Make_Buy",
    "Backflush_Ind","Scrap_Pct","Plant","Bin_Location"
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