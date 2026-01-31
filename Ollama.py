import pandas as pd
import numpy as np
import sqlite3
import streamlit as st
import ollama
import os
from sentence_transformers import SentenceTransformer, CrossEncoder, util

# --------------------------------------------------
# DATABASE SETUP
# --------------------------------------------------
DB_NAME = "bomgenius.db"
MODEL_NAME = "llama3.2:3b"

# Initialize internal rules files if they do not exist
def ensure_rules_files():
    if not os.path.exists("main.md"):
        with open("main.md", "w") as f:
            f.write("# MBOM Logic Rules\n1. Use 10 columns.\n2. Buy code B if Location exists.\n3. Backflush False for electronics.\n4. Standard item type unless consumable.")
    if not os.path.exists("Process.md"):
        with open("Process.md", "w") as f:
            f.write("# Process Rules\n1. Level 2 parts at Op 10.\n2. Level 1 parts at Op 20.\n3. Apply 2% scrap factor.")

def init_db():
    """Initialize SQLite database for match logging."""
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS bom_matches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            design_part TEXT,
            matched_part TEXT,
            confidence REAL,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_bom_match(design_part, matched_part, confidence):
    """Save match results to database."""
    try:
        conn = sqlite3.connect(DB_NAME)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO bom_matches 
            (design_part, matched_part, confidence, created_at)
            VALUES (?, ?, ?, datetime('now'))
        """, (design_part, matched_part, float(confidence.strip('%'))/100))
        conn.commit()
        conn.close()
    except Exception:
        pass

# --------------------------------------------------
# UI CONFIG
# --------------------------------------------------
def configure_ui():
    """Configure Streamlit UI settings."""
    st.set_page_config(page_title="BOMGenius", layout="wide")
    with st.sidebar:
        st.title("System Status")
        st.success("Local AI Mode: Active")
        st.info(f"Model: Ollama ({MODEL_NAME})")

@st.cache_resource
def load_models():
    """Load and cache transformer models."""
    bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return bi_encoder, cross_encoder

# --------------------------------------------------
# KNOWLEDGE BASE LOADER
# --------------------------------------------------
def read_md_file(filename):
    """Read markdown rule files."""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return "Rules configuration not found."

# --------------------------------------------------
# MATCHING ENGINE
# --------------------------------------------------
def find_best_match(query, inventory_records, inventory_names, embeddings, bi, cross, row_data):
    """Semantic matching engine that auto-detects column names."""
    if not isinstance(query, str) or query.strip() == "" or query.lower() == "nan":
        return None

    # 1. Semantic Search
    q_emb = bi.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(q_emb, embeddings, top_k=5)[0]
    candidates = [inventory_names[h["corpus_id"]] for h in hits]
    
    # 2. Reranking
    scores = cross.predict([[query, c] for c in candidates])
    best_idx = int(np.argmax(scores))
    match_name = candidates[best_idx]
    confidence = float(1 / (1 + np.exp(-scores[best_idx])))

    # 3. Robust Record Retrieval (finds row where match_name exists in any column)
    inv_item = next((item for item in inventory_records if any(str(v) == match_name for v in item.values())), {})
    
    # 4. Auto-detect Inventory Columns (Part Number, Location)
    part_no = next((v for k, v in inv_item.items() if "part" in k.lower() or "id" in k.lower()), "UNKNOWN")
    location = next((v for k, v in inv_item.items() if "loc" in k.lower() or "bin" in k.lower()), "")
    work_center = next((v for k, v in inv_item.items() if "center" in k.lower() or "work" in k.lower()), "GENERAL_ASSY")

    # 5. Logic: Make/Buy
    make_buy = "B" if (isinstance(location, str) and len(location) > 1) else "M"

    # 6. Logic: Op_Sequence & Level
    raw_level = str(row_data.get("Level", "2"))
    level = "".join(filter(str.isdigit, raw_level)) or "2"
    op_seq = "10" if int(level) >= 2 else "20"

    # 7. Logic: Quantity (Detects 'Qty' or 'Quantity' or 'Amount')
    try:
        qty_key = next((k for k in row_data.keys() if "qty" in k.lower() or "quant" in k.lower()), None)
        base_qty = float(row_data.get(qty_key, 1)) if qty_key else 1.0
        total_qty = base_qty * 1.02
    except:
        total_qty = 1.02

    # 8. Logic: Backflush
    electronics = ["PCB", "SCREEN", "BATTERY", "MOTOR", "CPU", "RAM", "SSD", "BOARD"]
    is_electronic = any(kw in match_name.upper() for kw in electronics)
    backflush = "False" if is_electronic else "True"

    return {
        "EBOM_Ref_ID": query,
        "Mfg_Part_No": part_no,
        "Make_Buy_Code": make_buy,
        "Op_Sequence": op_seq,
        "Work_Center": work_center,
        "BOM_Level": level,
        "Bin_Location": location if location else "SHORTAGE",
        "Backflush_Ind": backflush,
        "MBOM_Item_Type": "Standard",
        "Total_Qty_Req": round(total_qty, 2),
        "Confidence": f"{confidence:.2%}"
    }

# --------------------------------------------------
# MBOM GENERATION (OLLAMA)
# --------------------------------------------------
def generate_mbom_with_ollama(df_matches):
    """Generate final MBOM table using Ollama LLaMA model."""
    process_rules = read_md_file("Process.md")
    main_flow = read_md_file("main.md")

    # Format input data for the LLM
    context_data = ""
    for _, r in df_matches.iterrows():
        context_data += (
            f"ID: {r['EBOM_Ref_ID']} | Part: {r['Mfg_Part_No']} | Code: {r['Make_Buy_Code']} | "
            f"Seq: {r['Op_Sequence']} | WC: {r['Work_Center']} | Lvl: {r['BOM_Level']} | "
            f"Bin: {r['Bin_Location']} | BF: {r['Backflush_Ind']} | Type: {r['MBOM_Item_Type']} | "
            f"Qty: {r['Total_Qty_Req']}\n"
        )

    prompt = f"""
    You are a Manufacturing Systems Engineer.
    Convert the following matched data into a formal MBOM Markdown Table.

    RULES:
    {process_rules}
    {main_flow}

    FIELD NAMES FOR TABLE:
    1. EBOM_Ref_ID
    2. Mfg_Part_No
    3. Make_Buy_Code
    4. Op_Sequence
    5. Work_Center
    6. BOM_Level
    7. Bin_Location
    8. Backflush_Ind
    9. MBOM_Item_Type
    10. Total_Qty_Req

    INPUT DATA:
    {context_data}

    Output ONLY the Markdown Table. No conversational text.
    """
    
    try:
        response = ollama.generate(model=MODEL_NAME, prompt=prompt)
        return response["response"]
    except Exception as e:
        return f"Error connecting to Ollama: {str(e)}. Please ensure Ollama is running and {MODEL_NAME} is pulled."

# --------------------------------------------------
# STREAMLIT UI
# --------------------------------------------------
def main():
    ensure_rules_files()
    init_db()
    configure_ui()
    bi, cross = load_models()

    st.title("BOMGenius - EBOM to MBOM Converter")

    col1, col2 = st.columns(2)
    with col1:
        ebom_file = st.file_uploader("Upload EBOM (CSV/XLSX)", type=["csv", "xlsx"])
    with col2:
        inv_file = st.file_uploader("Upload Factory Inventory (CSV)", type=["csv"])

    if ebom_file and inv_file:
        # Load EBOM
        if ebom_file.name.endswith(".csv"):
            df_ebom_raw = pd.read_csv(ebom_file)
        else:
            df_ebom_raw = pd.read_excel(ebom_file)

        # Load Inventory
        df_inv_raw = pd.read_csv(inv_file)

        # Basic Column Normalization
        ebom_desc_col = next((c for c in df_ebom_raw.columns if "desc" in c.lower()), df_ebom_raw.columns[0])
        inv_name_col = next((c for c in df_inv_raw.columns if "name" in c.lower()), df_inv_raw.columns[0])

        if st.button("Generate MBOM"):
            inventory_names = df_inv_raw[inv_name_col].astype(str).tolist()
            inventory_data = df_inv_raw.to_dict('records')
            
            with st.spinner("Generating embeddings..."):
                embeddings = bi.encode(inventory_names, convert_to_tensor=True)

            results = []
            with st.status("Matching Parts...", expanded=True) as status:
                for idx, row in df_ebom_raw.iterrows():
                    match_res = find_best_match(
                        str(row[ebom_desc_col]), 
                        inventory_data,
                        inventory_names,
                        embeddings, 
                        bi, 
                        cross,
                        row
                    )
                    if match_res:
                        results.append(match_res)
                        save_bom_match(str(row[ebom_desc_col]), match_res["Mfg_Part_No"], match_res["Confidence"])
                status.update(label="Matching Complete!", state="complete")

            if results:
                df_results = pd.DataFrame(results)
                st.subheader("Semantic Mapping Results")
                st.dataframe(df_results, use_container_width=True)

                st.subheader("Final Manufacturing BOM")
                with st.spinner("AI applying manufacturing logic..."):
                    final_mbom = generate_mbom_with_ollama(df_results)
                    st.markdown(final_mbom)

if __name__ == "__main__":
    main()
