import pandas as pd
import numpy as np
import torch
import json
import io
import pdfplumber
import ezdxf
import sqlite3
import streamlit as st
import google.generativeai as genai
from PIL import Image
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import plotly.express as px
from datetime import datetime

# --------------------------------------------------
# 1. DATABASE LAYER (Integrated from db.py)
# --------------------------------------------------

DB_NAME = "bomgenius.db"

def init_db():
    """Initializes the SQLite database and tables."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    # Federated Learning Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS federated_learning_kb (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        wrong_part TEXT UNIQUE,
        correct_part TEXT,
        votes INTEGER DEFAULT 1,
        status TEXT,
        last_updated TEXT,
        sources TEXT
    )
    """)
    # BOM Match Result Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS bom_matches (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        design_part TEXT,
        matched_part TEXT,
        confidence REAL,
        unit_cost REAL,
        total_cost REAL,
        stock_status TEXT,
        match_type TEXT,
        created_at TEXT
    )
    """)
    conn.commit()
    conn.close()

def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def save_federated_learning(wrong_part, correct_part, factory):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO federated_learning_kb 
    (wrong_part, correct_part, votes, status, last_updated, sources)
    VALUES (?, ?, 1, 'Pending', datetime('now'), ?)
    ON CONFLICT(wrong_part)
    DO UPDATE SET
        votes = votes + 1,
        last_updated = datetime('now'),
        status = CASE WHEN votes >= 2 THEN 'Verified' ELSE 'Pending' END
    """, (wrong_part, correct_part, factory))
    conn.commit()
    conn.close()

def load_verified_kb():
    """Loads verified mappings into a dictionary for fast lookup."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT wrong_part, correct_part FROM federated_learning_kb WHERE status = 'Verified'")
    rows = cur.fetchall()
    conn.close()
    return {row['wrong_part']: row['correct_part'] for row in rows}

def save_bom_match(design_part, matched_part, confidence, unit_cost, total_cost, stock_status, match_type):
    with sqlite3.connect(DB_NAME) as conn:
        cur = conn.cursor()
        cur.execute("""
    INSERT INTO bom_matches
    (design_part, matched_part, confidence, unit_cost, total_cost, stock_status, match_type, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
    """, (design_part, matched_part, confidence, unit_cost, total_cost, stock_status, match_type))
    conn.commit()
    conn.close()

# --------------------------------------------------
# 2. SETUP & CONFIGURATION
# --------------------------------------------------

def configure_application_ui():
    st.set_page_config(page_title="BOMGenius", layout="wide")
    gemini_key = st.secrets.get("GEMINI_API_KEY")
    
    with st.sidebar:
        st.title("System Config")
        if gemini_key:
            genai.configure(api_key=gemini_key)
            st.success("Gemini AI Connected")
        else:
            st.error("Gemini API Key missing")
        st.divider()

@st.cache_resource
def load_models():
    device = "cpu" 
    bi_encoder = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=device)
    return bi_encoder, cross_encoder

# --------------------------------------------------
# 3. EXTRACTION & MATCHING LOGIC
# --------------------------------------------------

def extract_with_gemini(file_bytes, mime_type):
    try:
        # Update model name to 1.5-flash for speed/cost
        model = genai.GenerativeModel('gemini-1.5-flash') 
        
        prompt = "Extract BOM data. Return ONLY a JSON array: [{\"Description\": \"...\", \"Quantity\": 1}]"

        if "image" in mime_type:
            img = Image.open(io.BytesIO(file_bytes))
            response = model.generate_content([prompt, img])
        else:
            response = model.generate_content([
                prompt,
                {"mime_type": "application/pdf", "data": file_bytes}
            ])

        # More robust JSON cleaning
        text = response.text
        if "```json" in text:
            clean_json = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            clean_json = text.split("```")[1].split("```")[0].strip()
        else:
            clean_json = text.strip()

        data = json.loads(clean_json)
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Gemini Error: {e}")
        return None

# --------------------------------------------------
# 4. MULTIMODAL INGESTION ENGINE (Upgraded)
# --------------------------------------------------

def extract_from_pdf_local(pdf_file):
    """
    Robust local PDF table extractor. 
    Cleans newline characters and handles multi-page tables.
    """
    all_tables = []

    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                # Use extract_tables (plural) to find all tables on a page
                tables = page.extract_tables()
                
                for table in tables:
                    if table and len(table) > 1:
                        # Convert to DataFrame
                        df_page = pd.DataFrame(table)
                        
                        # CLEANING STEP 1: Use first row as header
                        df_page.columns = df_page.iloc[0]
                        df_page = df_page[1:].reset_index(drop=True)
                        
                        # CLEANING STEP 2: Remove None values and '\n' characters
                        # This prevents the AI from getting confused by "Bolt\nGrade 8"
                        df_page = df_page.applymap(lambda x: str(x).replace('\n', ' ').strip() if x is not None else "")
                        
                        all_tables.append(df_page)

        if not all_tables:
            return None

        # Combine all tables found in the PDF
        full_df = pd.concat(all_tables, ignore_index=True)
        
        # Remove completely empty rows
        full_df = full_df.replace('', np.nan).dropna(how='all')
        
        return full_df

    except Exception as e:
        st.error(f"Local PDF Parsing Error: {e}")
        return None

def extract_from_dxf(dxf_file):
    extracted_items = []
    try:
        dxf_data = dxf_file.getvalue().decode("utf-8", errors="ignore")
        doc = ezdxf.read(io.StringIO(dxf_data))
        msp = doc.modelspace()
        for entity in msp.query("TEXT MTEXT"):
            text_content = entity.dxf.text if entity.dxftype() == "TEXT" else entity.text
            if len(text_content) > 2:
                extracted_items.append(text_content)
        df = pd.DataFrame(extracted_items, columns=["Description"])
        df["Quantity"] = 1
        return df
    except Exception as e:
        st.error(f"CAD Parsing Error: {e}")
        return pd.DataFrame()

def parse_multimodal_ebom(uploaded_file):
    # We remove st.status from here to prevent UI freezing
    file_ext = uploaded_file.name.split('.')[-1].lower()
    file_bytes = uploaded_file.getvalue()
    
    try:
        df = None
        if file_ext in ['png', 'jpg', 'jpeg']:
            df = extract_with_gemini(file_bytes, f"image/{file_ext}")
        elif file_ext == 'pdf':
            df = extract_with_gemini(file_bytes, "application/pdf")
            if df is None:
                df = extract_from_pdf_local(uploaded_file)
        elif file_ext == 'csv':
            df = pd.read_csv(uploaded_file, encoding_errors='replace')
        elif file_ext in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file)
        elif file_ext == 'json':
            data = json.load(uploaded_file)
            all_items = []
            def walk_json(obj):
                if isinstance(obj, dict):
                    name = obj.get("item_name") or obj.get("name")
                    qty = obj.get("qty") or obj.get("quantity")
                    if name and qty is not None and not any(isinstance(v, list) for v in obj.values()):
                        all_items.append({"Description": str(name), "Quantity": qty})
                    for v in obj.values(): walk_json(v)
                elif isinstance(obj, list):
                    for i in obj: walk_json(i)
            walk_json(data)
            df = pd.DataFrame(all_items)
        elif file_ext == 'dxf':
            df = extract_from_dxf(uploaded_file)

        if df is None or df.empty:
            return None

        # Standardize columns
        final_ebom = pd.DataFrame()
        desc_col = next((c for c in df.columns if any(x in c.lower() for x in ["desc", "item", "name"])), df.columns[0])
        qty_col = next((c for c in df.columns if any(x in c.lower() for x in ["qty", "quant"])), None)
        
        final_ebom["Description"] = df[desc_col].astype(str)
        final_ebom["Quantity"] = pd.to_numeric(df[qty_col], errors='coerce').fillna(1) if qty_col else 1
        st.write("DEBUG extracted df:", df)
        st.write("DEBUG final ebom:", final_ebom)

        return final_ebom
    except Exception as e:
        st.error(f"Ingestion Error: {e}")
        return None

# --------------------------------------------------
# SECTION 5: ADVANCED MATCHING ENGINE
# --------------------------------------------------

def find_best_match(query, inventory_names, inventory_ids, inventory_embeddings, bi_encoder, cross_encoder, kb_map):
    # Check DB Knowledge Base first
    if query in kb_map:
        return {"match_name": kb_map[query], "match_id": "DB-VERIFIED", "score": 1.0, "method": "Verified Override"}

    # AI Matching
    query_embedding = bi_encoder.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, inventory_embeddings, top_k=5)[0]
    candidate_indices = [hit['corpus_id'] for hit in hits]
    candidate_names = [inventory_names[i] for i in candidate_indices]
    
    model_inputs = [[query, candidate] for candidate in candidate_names]
    scores = cross_encoder.predict(model_inputs)
    
    results = sorted([{"index": i, "id": inventory_ids[i], "name": inventory_names[i], "score": s} 
                     for i, s in zip(candidate_indices, scores)], key=lambda x: x['score'], reverse=True)
    
    best = results[0]
    # Sigmoid for display score
    display_score = 1 / (1 + torch.exp(torch.tensor(-best['score']))).item()
    return {"match_name": best['name'], "match_id": best['id'], "score": display_score, "method": "AI Cross-Encoder"}

# --------------------------------------------------
# SECTION 6: UI PAGES
# --------------------------------------------------

def render_dashboard():
    st.title("Executive Dashboard")
    # Metric counts from DB
    conn = get_db_connection()
    total_matches = conn.execute("SELECT COUNT(*) FROM bom_matches").fetchone()[0]
    kb_verified = conn.execute("SELECT COUNT(*) FROM federated_learning_kb WHERE status='Verified'").fetchone()[0]
    conn.close()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Historical Matches", total_matches)
    col2.metric("Verified Knowledge", kb_verified)
    col3.metric("Est. Cost Savings", f"${total_matches * 12:.0f}")
    col4.metric("Active Learning", "Enabled")

    c1, c2 = st.columns(2)
    with c1:
        df_chart = pd.DataFrame({'Category': ['Fasteners', 'Electronics', 'Raw Metal', 'Plastics', 'Consumables'], 'Count': [450, 300, 150, 200, 100]})
        st.plotly_chart(px.pie(df_chart, values='Count', names='Category', title='mBOM Composition'), use_container_width=True)
    with c2:
        df_conf = pd.DataFrame({'Confidence': ['High (>90%)', 'Medium (70-90%)', 'Low (<70%)'], 'Items': [850, 300, 90]})
        st.plotly_chart(px.bar(df_conf, x='Confidence', y='Items', color='Confidence', title='Match Reliability'), use_container_width=True)
# --- Recent Activity Log ---
    st.subheader("Recent System Activity")
    st.dataframe(pd.DataFrame({
        "Timestamp": ["10:05 AM", "10:12 AM", "10:45 AM"],
        "User": ["Factory_Admin_IN", "System_AI", "Factory_Admin_DE"],
        "Action": ["Uploaded eBOM_v2.csv", "Auto-Matched 85 parts", "Flagged 'False Negative'"]
    }), use_container_width=True)

    
def render_bom_converter(bi_encoder, cross_encoder, threshold=0.5):
    st.title("eBOM to mBOM Transformation Engine")
    
    # Initialize session state for results so they persist
    if "final_results" not in st.session_state:
        st.session_state.final_results = None

    col_up1, col_up2 = st.columns(2)
    with col_up1:
        ebom_file = st.file_uploader("Upload eBOM (DXF, PDF, Image, Excel, JSON, CSV)", type=["csv", "xlsx", "json", "png", "jpg", "pdf", "dxf"])
    with col_up2:
        inventory_file = st.file_uploader("Upload Factory Inventory (CSV)", type=["csv"])
    
    if ebom_file and inventory_file:
        # Single Action Button
        if st.button("Run Full AI Transformation"):
            
            kb_map = load_verified_kb()
            df_design = parse_multimodal_ebom(ebom_file)
            
            if df_design is None or df_design.empty:
                st.error("Failed to extract eBOM data. Please check your file format.")
                return
            
            df_inventory = pd.read_csv(inventory_file)
            
            # Inventory Indexing
            inv_desc_col = next((c for c in df_inventory.columns if "name" in c.lower() or "desc" in c.lower()), df_inventory.columns[0])
            inv_id_col = next((c for c in df_inventory.columns if "id" in c.lower()), df_inventory.columns[0])
            inventory_names = df_inventory[inv_desc_col].astype(str).tolist()
            inventory_ids = df_inventory[inv_id_col].astype(str).tolist()
            inventory_embeddings = bi_encoder.encode(inventory_names, convert_to_tensor=True)

            results = []
            with st.status("Matching items..."):
                for _, row in df_design.iterrows():
                    match = find_best_match(row['Description'], inventory_names, inventory_ids, inventory_embeddings, bi_encoder, cross_encoder, kb_map)
                    
                    # Logic to determine if valid
                    is_valid = match['score'] >= threshold
                    res_row = {
                        "Design Part": row['Description'],
                        "Matched Part": match['match_name'] if is_valid else "N/A",
                        "Confidence": f"{match['score']:.2%}",
                        "Status": match['method'] if is_valid else "Low Confidence"
                    }
                    results.append(res_row)
                    
                    # Persistent DB Logging
                    save_bom_match(row['Description'], res_row['Matched Part'], match['score'], 0, 0, "N/A", match['method'])

                    # --- PERSISTENCE: Save to Session State ---
                    st.session_state["persistent_results"] = results
                st.toast("Match Complete!")

            st.session_state.final_results = pd.DataFrame(results)
            st.dataframe(st.session_state.final_results, use_container_width=True)

    # --- THIS PART SHOWS THE DATA EVEN IF YOU SWITCH PAGES ---
    if "persistent_results" in st.session_state and st.session_state["persistent_results"] is not None:
        st.divider()
        st.subheader("Latest Match Results")
        
        # We show the data from the Session Memory
        res_df = pd.DataFrame(st.session_state["persistent_results"])
        st.dataframe(res_df, use_container_width=True)
        
        # The download button will now stay visible
        csv_data = res_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Result CSV",
            data=csv_data,
            file_name="BOMGenius_Export.csv",
            mime="text/csv"
        )

def render_federated_learning():
    st.title("Federated Learning Console")
    c1, c2 = st.columns([1, 2])
    with c1:
        with st.form("learning_form"):
            ebom_input = st.text_input("eBOM Description")
            mbom_input = st.text_input("Correct Inventory Name")
            factory_id = st.selectbox("Voting Entity", ["Factory A", "Factory B", "HQ Engineer"])
            if st.form_submit_button("Submit Vote"):
                save_federated_learning(ebom_input, mbom_input, factory_id)
                st.success("Vote Saved to Database.")
    with c2:
        conn = get_db_connection()
        df_kb = pd.read_sql_query("SELECT wrong_part, correct_part, votes, status FROM federated_learning_kb", conn)
        conn.close()
        st.subheader("Shared Knowledge Base")
        st.dataframe(df_kb, use_container_width=True)

# --------------------------------------------------
# 7. MAIN EXECUTION
# --------------------------------------------------

def main():
    init_db()
    configure_application_ui()
    bi_encoder, cross_encoder = load_models()
    
    page = st.sidebar.radio("Navigate", ["Dashboard", "BOM Converter", "Federated Learning"])
    
    if page == "Dashboard": render_dashboard()
    elif page == "BOM Converter": render_bom_converter(bi_encoder, cross_encoder, 0.65)
    elif page == "Federated Learning": render_federated_learning()

if __name__ == "__main__":
    main()