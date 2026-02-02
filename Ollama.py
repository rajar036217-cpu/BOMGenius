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
    [ROLE] You are a Manufacturing Systems Architect.
    [TASK] Match items from the EBOM to the best fit in the Factory Inventory and generate a 10-column MBOM Table.

    [RULES]
    {main_rules}
    {proc_rules}

    [INVENTORY LIST]
    {inv_context}

    [EBOM ITEMS TO MATCH]
    {ebom_context}

    [OUTPUT INSTRUCTIONS]
    1. Match each EBOM item to the most logical part in the Inventory.
    2. Provide ONLY a Markdown Table with exactly these 10 headers:
    EBOM_Ref_ID | Mfg_Part_No | Make_Buy_Code | Op_Sequence | Work_Center | BOM_Level | Bin_Location | Backflush_Ind | MBOM_Item_Type | Total_Qty_Req
    """

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

