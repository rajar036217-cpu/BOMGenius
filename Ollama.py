import pandas as pd
import streamlit as st
import ollama
import sqlite3
import os
import io

# Configuration
DB_NAME = "bomgenius.db"
MODEL_NAME = "llama3.2:3b"

def ensure_rules_files():
    if not os.path.exists("main.md"):
        with open("main.md", "w") as f:
            f.write("# MBOM Logic\n1. If Location_ID exists, Code='B', else 'M'.\n2. High-value (CPU, SCREEN) Backflush=False.\n3. Standard Scrap=2%.")
    if not os.path.exists("Process.md"):
        with open("Process.md", "w") as f:
            f.write("# Process\n1. Level 2+ at Op 10.\n2. Level 1 at Op 20.")

def init_db():
    """Initializes the database to match the 10-column MBOM structure."""
    conn = sqlite3.connect(DB_NAME)
    # This creates a table that matches exactly what LLaMA produces
    conn.execute("""
        CREATE TABLE IF NOT EXISTS matches(
            EBOM_Ref_ID TEXT, 
            Mfg_Part_No TEXT, 
            Make_Buy_Code TEXT, 
            Op_Sequence TEXT, 
            Work_Center TEXT, 
            BOM_Level TEXT, 
            Bin_Location TEXT, 
            Backflush_Ind TEXT, 
            MBOM_Item_Type TEXT, 
            Total_Qty_Req TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.close()

def run_llama_matching(ebom_df, inv_df):
    """The core engine: LLaMA performs lookup and matching with flexible column detection."""
    main_rules = open("main.md", "r").read() if os.path.exists("main.md") else ""
    proc_rules = open("Process.md", "r").read() if os.path.exists("Process.md") else ""
    
    # --- FIX: DYNAMIC COLUMN DETECTION ---
    # This finds the best columns even if they are named slightly differently
    def get_col(df, keywords, default):
        for col in df.columns:
            if any(k.lower() in col.lower() for k in keywords):
                return col
        return default

    inv_name = get_col(inv_df, ["name", "desc"], inv_df.columns[0])
    inv_part = get_col(inv_df, ["part", "number", "id"], inv_df.columns[0])
    inv_loc = get_col(inv_df, ["loc", "bin"], "Location_ID") # default if not found
    inv_wc = get_col(inv_df, ["work", "center", "wc"], "Work_Center") # default if not found

    # Create a simplified inventory view for LLaMA context
    # Use only columns that actually exist to avoid the KeyError
    existing_cols = [c for c in [inv_name, inv_part, inv_loc, inv_wc] if c in inv_df.columns]
    inv_context = inv_df[existing_cols].to_csv(index=False)
    ebom_context = ebom_df.to_csv(index=False)

    prompt = f"""You are a Manufacturing BOM Engineer AI embedded in a PLM-to-MES integration system.

GOAL:
Convert the given Engineering BOM (eBOM) into a Manufacturing BOM (mBOM) suitable for factory execution.

EBOM (CSV):
{ebom_context}

INVENTORY (CSV):
{inv_context}

CONVERSION RULES:
1. Preserve all original parts and quantities from the eBOM.
2. Re-group parts into logical manufacturing sub-assemblies based on build sequence and shop-floor practicality.
3. Create manufacturing sub-assemblies even if they do not exist in the eBOM (use synthetic IDs like SUBASM-001, SUBASM-002).
4. Assign each part to exactly one sub-assembly.
5. Introduce a final top-level assembly node that consumes all sub-assemblies.
6. Add manufacturing attributes for each line:
   - Op_Sequence (10, 20, 30, ...)
   - Work_Center (Welding, Assembly, QC, Packaging only)
   - Make_Buy (default from eBOM; if missing, assume MAKE for custom parts, BUY for standard fasteners)
   - MBOM_Item_Type (RAW, SUBASM, FG)
7. Do NOT remove any part present in the eBOM.
8. Do NOT invent quantities. Quantities must match totals from the eBOM.
9. If ambiguous, choose the most reasonable grouping and list assumptions.

OUTPUT FORMAT (STRICT):
Return ONLY rows separated by NEWLINES.
Each row MUST use the pipe character | as the delimiter.

Do NOT use commas.
Do NOT put everything in one line.
Do NOT add headers.
Do NOT add markdown tables.

Each row MUST follow this exact column order:

Parent_Part_No | Child_Part_No | Description | Qty_Per | UOM | BOM_Level | Op_Sequence | Work_Center | Make_Buy | Backflush_Ind | Scrap_Pct | Plant | Bin_Location

Allowed values:
- UOM: EA
- Work_Center: Welding, Assembly, QC, Packaging
- Make_Buy: MAKE or BUY
- Backflush_Ind: Yes or No
- Scrap_Pct: numeric (e.g., 2)
- BOM_Level: 0 for FG, 1 for sub-assembly, 2+ for parts
- Plant: PLANT-01 (default if not provided by inventory)
- Bin_Location: if inventory location exists use it, else NA

After the table, add:
ASSUMPTIONS:
- <short bullets>

MAPPING_NOTES:
- <short bullets>

No markdown."""

    try:
        response = ollama.generate(
            model=MODEL_NAME,
            prompt=prompt,
            options={
                "temperature": 0.05,
                "top_p": 0.9,
                "num_ctx": 4096
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

                        #EXPECTED_COLS = 9  # Parent_ID | Item_ID | Item_Name | Qty | Op_Sequence | Work_Center | Make_Buy | MBOM_Item_Type | EBOM_Ref_ID

                        #if df_final.shape[1] != EXPECTED_COLS:
                         #   st.error(f"AI returned {df_final.shape[1]} columns, expected {EXPECTED_COLS}. Raw output below.")
                          #  st.markdown(raw_result)
                           # return

                        # 3. Final Clean: Remove the Markdown separator row (---|---|---)
                        df_final = df_final[~df_final.iloc[:, 0].astype(str).str.contains('---', na=False)]
                        
                        # 4. Clean column names (remove leading/trailing spaces)
                        df_final.columns = [c.strip() for c in df_final.columns]

                        st.subheader("Final Manufacturing BOM (AI Generated)")
                        st.dataframe(df_final, use_container_width=True)
                        
                        # 5. Save to Database
                        with sqlite3.connect(DB_NAME) as conn:
                            df_final.to_sql('matches', conn, if_exists='replace', index=False)
                        st.success(f"MBOM saved to {DB_NAME}")

                    except Exception as e:
                        st.error(f"Parsing Error: {str(e)}")
                        st.subheader("Raw AI Output")
                        st.markdown(raw_result)
                        
if __name__ == "__main__":
    main()





