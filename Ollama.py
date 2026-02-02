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
    conn = sqlite3.connect(DB_NAME)
    conn.execute("CREATE TABLE IF NOT EXISTS matches (ebom_item TEXT, mfg_item TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)")
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

    prompt = f"""
    You are a Manufacturing BOM Engineer AI embedded in a PLM-to-MES integration system.

GOAL:
Convert the given Engineering BOM (eBOM) into a Manufacturing BOM (mBOM) suitable for factory execution.

INPUT:
You will receive an eBOM in CSV-like format with one part per line:
Part Name | Part Type | Qty | Parent Assembly | Make/Buy | Material | Weight | Standard/Custom | Fastener

CONVERSION RULES:
1. Preserve all original parts and quantities from the eBOM.
2. Re-group parts into logical manufacturing sub-assemblies based on build sequence and shop-floor practicality.
   - Examples of sub-assemblies: Frame Assembly, Door Shell Assembly, Window Mechanism Assembly, Final Assembly.
3. Create manufacturing sub-assemblies even if they do not exist in the eBOM (use synthetic IDs like SUBASM-001, SUBASM-002).
4. Assign each part to exactly one sub-assembly.
5. Introduce a final top-level assembly node that consumes all sub-assemblies.
6. Add manufacturing attributes for each line:
   - Op_Sequence (10, 20, 30, ...)
   - Work_Center (e.g., Welding, Assembly, QC, Packaging)
   - Make_Buy (default from eBOM; if missing, assume MAKE for custom parts, BUY for standard fasteners)
   - MBOM_Item_Type (RAW, SUBASM, FG)
7. Do NOT remove any part present in the eBOM.
8. Do NOT invent quantities. Quantities must match the total from the eBOM.
9. If parts are ambiguous, choose the most reasonable manufacturing grouping and explain assumptions in a short “Assumptions” section.

OUTPUT FORMAT:
Return the mBOM in CSV-like lines with the following schema:
Parent_ID | Item_ID | Item_Name | Qty | Op_Sequence | Work_Center | Make_Buy | MBOM_Item_Type | EBOM_Ref_ID

Also return a short section:
- Assumptions
- Mapping Notes (how eBOM items were grouped into sub-assemblies)

QUALITY BAR:
The mBOM must be directly usable by MES/ERP for routing and material consumption planning.
No explanations outside the requested output format."""

    try:
        response = ollama.generate(model=MODEL_NAME, prompt=prompt)
        return response["response"]
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    ensure_rules_files()
    init_db()
    st.set_page_config(page_title="BOMGenius AI", layout="wide")
    
    st.title("BOMGenius: Pure LLaMA MBOM Engine")
    st.info("System: SentenceTransformers Disabled. LLaMA is handling all semantic matching and logic.")

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
                
                try:
                    # Extract and parse the table from LLaMA's response
                    lines = [l.strip() for l in raw_result.split('\n') if '|' in l]
                    df_final = pd.read_csv(io.StringIO('\n'.join(lines)), sep="|", skipinitialspace=True).dropna(axis=1, how='all')
                    df_final = df_final[~df_final.iloc[:, 0].str.contains('---', na=False)]
                    df_final.columns = [c.strip() for c in df_final.columns]
                    
                    st.subheader("Final Manufacturing BOM (AI Generated)")
                    st.dataframe(df_final, use_container_width=True)
                    
                    # Log matches to DB
                    with sqlite3.connect(DB_NAME) as conn:
                        df_final[['EBOM_Ref_ID', 'Mfg_Part_No']].to_sql('matches', conn, if_exists='append', index=False)
                        
                except Exception as e:
                    st.error("AI Output Format Error. Raw response shown below.")
                    st.markdown(raw_result)

if __name__ == "__main__":
    main()
