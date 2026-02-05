import pandas as pd
import streamlit as st
import ollama
import sqlite3
import os
import io
import torch
torch.set_num_threads(1)

# Configuration
DB_NAME = "bomgenius.db"
MODEL_NAME = "llama3.2:3b"

# ============================
# CANONICAL INVENTORY SCHEMA
# ============================

from sentence_transformers import SentenceTransformer, util

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

schema_model = load_embedding_model()


CANONICAL_FIELDS = {
    "part_name": ["name", "desc", "description", "item", "part name"],
    "part_no": ["part", "number", "pn", "id", "code"],
    "bin_location": ["bin", "location", "loc", "warehouse", "stock", "store", "depot"],
    "work_center": ["wc", "workcenter", "work center", "dept", "shop", "line"],
}

def ensure_rules_files():
    if not os.path.exists("main.md"):
        with open("main.md", "w") as f:
            f.write("# MBOM Logic\n1. If Location_ID exists, Code='B', else 'M'.\n2. High-value (CPU, SCREEN) Backflush=False.\n3. Standard Scrap=2%.")
    if not os.path.exists("Process.md"):
        with open("Process.md", "w") as f:
            f.write("# Process\n1. Level 2+ at Op 10.\n2. Level 1 at Op 20.")

def init_db():
    """Initializes the database to match the MBOM structure."""
    conn = sqlite3.connect(DB_NAME)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS mbom (
            Parent_Part_No   TEXT,
            Child_Part_No    TEXT,
            Description      TEXT,
            Qty_Per          REAL,
            UOM              TEXT,
            BOM_Level        INTEGER,
            Op_Sequence      INTEGER,
            Work_Center      TEXT,
            Make_Buy         TEXT,
            Backflush_Ind    TEXT,
            Scrap_Pct        REAL,
            Plant            TEXT,
            Bin_Location     TEXT,
            timestamp        DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.close()

CANONICAL_EMBEDDINGS = {
    k: schema_model.encode(v) for k, v in CANONICAL_FIELDS.items()
}

def learn_inventory_schema(df):
    col_map = {}

    for col in df.columns:
        col_emb = schema_model.encode(col.lower())

        best_match = None
        best_score = 0

        for canon, emb_list in CANONICAL_EMBEDDINGS.items():
            scores = util.cos_sim(col_emb, emb_list)
            max_score = scores.max().item()

            if max_score > best_score:
                best_score = max_score
                best_match = canon

        if best_score > 0.55:
            col_map[col] = best_match

    return col_map


def normalize_inventory(df, col_map):
    norm = pd.DataFrame()
    for raw_col, canon in col_map.items():
        norm[canon] = df[raw_col]
    return norm

def run_llama_matching(ebom_df, inv_df):
    """The core engine: LLaMA performs lookup and matching with flexible column detection."""
    main_rules = open("main.md", "r").read() if os.path.exists("main.md") else ""
    proc_rules = open("Process.md", "r").read() if os.path.exists("Process.md") else ""
    
    learned_schema = learn_inventory_schema(inv_df)
    normalized_inv = normalize_inventory(inv_df, learned_schema)
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
    
    try:
        response = ollama.generate(
            model=MODEL_NAME,
            prompt=prompt,
            options={
                "temperature": 0.0,
                "top_p": 1.0,
                "num_ctx": 4096,
                "seed": 42
            }
        )
        return response["response"]
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    ensure_rules_files()
    init_db()
    st.set_page_config(page_title="BOMGenius AI", layout="wide")
    
    st.title("BOMGenius: Pure LLaMA MBOM Engine")
    st.info("System: SentenceTransformers Disabled. LLaMA is handling all matching.")

    c1, c2 = st.columns(2)
    with c1:
        ebom_file = st.file_uploader("Upload EBOM", type=["csv"])
    with c2:
        inv_file = st.file_uploader("Upload Inventory", type=["csv"])

    if ebom_file and inv_file:
        df_ebom = pd.read_csv(ebom_file)
        df_inv = pd.read_csv(inv_file)

        if st.button("Generate MBOM with LLaMA"):
            with st.spinner("LLaMA is cross-referencing inventory and applying rules..."):
                raw_result = run_llama_matching(df_ebom, df_inv)
                
                raw_lines = [l.strip() for l in raw_result.split('\n') if l.strip()]
                # If model used pipes, use that. Else, fallback to commas.
                if any('|' in l for l in raw_lines):
                    lines = [l for l in raw_lines if '|' in l]
                    sep = '|'
                else:
                    # Model returned comma CSV in one or more lines
                    lines = raw_lines
                    sep = ','
                    
                if len(lines) < 2:
                    st.error("AI did not return a valid table. See raw response below.")
                    st.markdown(raw_result)
                else:
                    try:
                        # 1. Join lines and clean extra pipes
                        table_str = '\n'.join(lines)
                        
                        df_final = pd.read_csv(
                            io.StringIO('\n'.join(lines)),
                            sep=sep,
                            skipinitialspace=True,
                            engine='python'
                            ).dropna(axis=1, how='all')

                        df_final.columns = [
                            "Parent_Part_No",
                            "Child_Part_No",
                            "Description",
                            "Qty_Per",
                            "UOM",
                            "BOM_Level",
                            "Op_Sequence",
                            "Work_Center",
                            "Make_Buy",
                            "Backflush_Ind",
                            "Scrap_Pct",
                            "Plant",
                            "Bin_Location"]


                        # 3. Final Clean: Remove the Markdown separator row (---|---|---)
                        df_final = df_final[~df_final.iloc[:, 0].astype(str).str.contains('---', na=False)]
                        
                        # 4. Clean column names (remove leading/trailing spaces)
                        df_final.columns = [c.strip() for c in df_final.columns]

                        st.subheader("Final Manufacturing BOM (AI Generated)")
                        st.dataframe(df_final, use_container_width=True)
                        
                        # 5. Save to Database
                        with sqlite3.connect(DB_NAME) as conn:
                            df_final.to_sql('mbom', conn, if_exists='replace', index=False)
                        st.success(f"MBOM saved to {DB_NAME}")

                    except Exception as e:
                        st.error(f"Parsing Error: {str(e)}")
                        st.subheader("Raw AI Output")
                        st.markdown(raw_result)
                        
if __name__ == "__main__":
    main()











