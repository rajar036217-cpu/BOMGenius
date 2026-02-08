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

GLOBAL_RULES_FILE = "federated/global_rules.json"

def load_global_rules():
    if os.path.exists(GLOBAL_RULES_FILE):
        with open(GLOBAL_RULES_FILE, "r") as f:
            return json.load(f)
    return {}
    
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

    global_rules = load_global_rules()
    global_context = json.dumps(global_rules, indent=2)
    
    inv_context = normalized_inv.to_csv(index=False)
    ebom_context = ebom_df.to_csv(index=False)

    prompt = f"""
SYSTEM ROLE:
You are a senior Manufacturing BOM Engineer AI operating inside a PLM-to-MES pipeline.

OBJECTIVE:
Transform the given Engineering BOM (eBOM) into a factory-executable Manufacturing BOM (mBOM).

INPUT CONTEXT:
The input may originate from ANY format (CSV, Excel, JSON, PDF, images, CAD screenshots, engineering drawings, OCR outputs, CAD exports).
Assume all such inputs have been pre-parsed and normalized into structured text below.

EBOM (normalized):
{ebom_context}

INVENTORY (normalized):
{inv_context}

GLOBAL LEARNED RULES (from federated learning):
{global_context}

Use these global rules as PRIORITY overrides for Make_Buy, UOM, Work_Center, Backflush_Ind when applicable.

REASONING TASK:
Use manufacturing logic and shop-floor practicality to derive a structured mBOM.
Apply domain knowledge to infer reasonable subassemblies, phantom groupings, and operation sequencing.
If multiple valid structures exist, choose the most production-efficient one and note assumptions.

CONSTRAINTS:
- Preserve all original parts and total quantities from eBOM.
- Re-group parts into logical manufacturing subassemblies.
- Create manufacturing subassemblies even if not present in eBOM (use synthetic IDs: SUBASM-001, SUBASM-002, ...).
- Each part must belong to exactly one subassembly.
- Create exactly one final FG (finished good) node consuming all subassemblies.
- Do NOT remove any eBOM parts.
- Do NOT invent quantities.
- If ambiguous, make reasonable assumptions and report them.

MBOM ATTRIBUTES (infer intelligently if missing):
- Op_Sequence: 10, 20, 30...
- Work_Center: Welding | Assembly | QC | Packaging
- Make_Buy:
  - Use inventory if present.
  - Else infer: standard fasteners = BUY, custom/fabricated parts = MAKE
- MBOM_Item_Type: RAW | SUBASM | FG
- Backflush_Ind: Yes for consumables/fasteners, No otherwise
- Scrap_Pct: numeric default 0–5 based on process
- Plant: PLANT-01 (default)
- Bin_Location: use inventory location else NA

HIERARCHY RULES (MANDATORY):
- FG = BOM_Level 0
- Subassemblies = BOM_Level 1
- Leaf parts = BOM_Level >= 2
- FG must ONLY have subassemblies as children
- Subassemblies must ONLY have parts as children
- Assemblies must explode into at least 2 children
- No leaf node may appear as a parent

UOM INFERENCE:
- Fasteners (bolt, nut, rivet, clip) → EA
- Panels, brackets, frames → EA
- Adhesives, sealants, paint → L or KG
- Welding wire → M or KG
- Sheet metal stock → KG
- Fluids → L
- If description includes kg/litre → use KG or L
- Do NOT default everything to EA

PART NUMBER RULE:
Child_Part_No must NEVER be null.
If missing, generate PN-<UPPERCASE_DESCRIPTION_NO_SPACES>.

OUTPUT FORMAT (STRICT CSV):
- Output ONLY CSV rows (newline-separated)
- NO headers
- NO markdown
- NO commas inside fields
- Pipe character | as delimiter

COLUMN ORDER (MANDATORY):
Parent_Part_No | Child_Part_No | Description | Qty_Per | UOM | BOM_Level | Op_Sequence | Work_Center | Make_Buy | Backflush_Ind | Scrap_Pct | Plant | Bin_Location

After CSV rows, append:

ASSUMPTIONS:
- <short bullets>

MAPPING_NOTES:
- <short bullets>

No markdown.
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

