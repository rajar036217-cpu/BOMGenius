"""
BOMGenius (Final POC Version)
------------------------------------------
Features:
1. AI-Powered Matching (BERT)
2. Federated Learning Simulation (Graph + Feedback)
3. Downloadable Results
4. Professional Dashboard UI
"""

import pandas as pd
import numpy as np
import torch
import json
import io
import pdfplumber
import ezdxf
import streamlit as st
from PIL import Image
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import plotly.express as px
import time

# --------------------------------------------------
# 1. SETUP & CONFIGURATION
# --------------------------------------------------

def configure_application_ui():
    st.set_page_config(page_title="BOMGenius", layout="wide")

# Initialize Session State for Federated Learning (The "Global Knowledge Base")
    if 'knowledge_base' not in st.session_state:
    # Structure: {'eBOM_Name': {'target': 'Inventory_Name', 'votes': set(['FactoryA']), 'status': 'Pending'}}
        st.session_state.knowledge_base = {}

    if 'history_log' not in st.session_state:
        st.session_state.history_log = []

# --------------------------------------------------
# 2. LOAD AI MODEL (Cached)
# --------------------------------------------------

@st.cache_resource
def load_models():
    """
    Loads the Bi-Encoder for fast retrieval and Cross-Encoder for precision ranking.
    """
    with st.spinner("Loading AI Models (Bi-Encoder + Cross-Encoder)..."):
        # Stage 1: Bi-Encoder (Fast, generates embeddings)
        bi_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Stage 2: Cross-Encoder (Slow, accurate, acts as the 'Judge')
        # MS MARCO is trained on Bing search data, good at relevance scoring
        cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
    return bi_encoder, cross_encoder

bi_encoder, cross_encoder = load_models()

# --------------------------------------------------
# 3. MULTIMODAL INGESTION ENGINE
# --------------------------------------------------

def extract_from_pdf(pdf_file):
    extracted_data = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            table = page.extract_table()
            if table:
                df_page = pd.DataFrame(table[1:], columns=table[0])
                extracted_data.append(df_page)
            else:
                text = page.extract_text()
                if text:
                    lines = [line.split() for line in text.split('\n') if len(line.split()) > 1]
                    extracted_data.append(pd.DataFrame(lines))
    
    if extracted_data:
        full_df = pd.concat(extracted_data, ignore_index=True)
        return full_df
    return pd.DataFrame(columns=["Description", "Quantity"])

def extract_from_cad_image(image_file):
    # Mock OCR/Vision logic
    mock_extracted_data = [
        {"Description": "M8x25 Stainless Steel Hexagonal Bolt", "Quantity": 12},
        {"Description": "High-Temp Silicon Gasket 50mm", "Quantity": 4},
        {"Description": "Reinforced Steel Plate 200x200mm", "Quantity": 2},
        {"Description": "Analog Pressure Release Valve 10 Bar", "Quantity": 1}
    ]
    return pd.DataFrame(mock_extracted_data)

def extract_from_dxf(dxf_file):
    extracted_items = []
    try:
        stream = io.BytesIO(dxf_file.read())
        doc = ezdxf.read(stream)
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
    file_ext = uploaded_file.name.split('.')[-1].lower()
    with st.status(f"🔍 Deep-Scanning {file_ext.upper()}...", expanded=True) as status:
        try:
            if file_ext == 'csv':
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
            elif file_ext == 'pdf':
                df = extract_from_pdf(uploaded_file)
            elif file_ext == 'dxf':
                df = extract_from_dxf(uploaded_file)
            elif file_ext in ['png', 'jpg', 'jpeg']:
                st.image(Image.open(uploaded_file), width=300)
                df = extract_from_cad_image(uploaded_file)
            else:
                return None

            if df is None or df.empty:
                return None

            # Standardize columns
            final_ebom = pd.DataFrame()
            desc_col = next((c for c in df.columns if "desc" in c.lower() or "item" in c.lower() or "name" in c.lower()), df.columns[0])
            qty_col = next((c for c in df.columns if "qty" in c.lower() or "quant" in c.lower()), None)
            
            final_ebom["Description"] = df[desc_col].astype(str)
            final_ebom["Quantity"] = pd.to_numeric(df[qty_col], errors='coerce').fillna(1) if qty_col else 1
            
            status.update(label=f"✅ Extracted {len(final_ebom)} components", state="complete")
            return final_ebom
        except Exception as e:
            st.error(f"⚠️ Ingestion Error: {e}")
            return None

# --- SECTION 2: ADVANCED MATCHING ENGINE ---

def find_best_match(query, inventory_names, inventory_ids, inventory_embeddings, bi_encoder, cross_encoder):
    """
    Core AI Logic: Knowledge Base -> Bi-Encoder Retrieval -> Cross-Encoder Re-ranking
    """
    # 1. Knowledge Base Override
    kb = st.session_state.get('knowledge_base', {})
    if query in kb:
        entry = kb[query]
        return {
            "match_name": entry['target'],
            "match_id": "MANUAL-OVERRIDE",
            "score": 1.0,
            "method": f"Knowledge Base ({entry.get('status', 'Verified')})"
        }

    # 2. Retrieval (Bi-Encoder)
    query_embedding = bi_encoder.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, inventory_embeddings, top_k=5)[0]
    
    candidate_indices = [hit['corpus_id'] for hit in hits]
    candidate_names = [inventory_names[i] for i in candidate_indices]
    
    # 3. Re-Ranking (Cross-Encoder)
    model_inputs = [[query, candidate] for candidate in candidate_names]
    scores = cross_encoder.predict(model_inputs)
    
    results = []
    for idx, score in zip(candidate_indices, scores):
        results.append({
            "index": idx,
            "id": inventory_ids[idx],
            "name": inventory_names[idx],
            "score": score
        })
    
    results = sorted(results, key=lambda x: x['score'], reverse=True)
    best = results[0]
    
    # Normalize logit to 0-1
    display_score = 1 / (1 + torch.exp(torch.tensor(-best['score']))).item()

    return {
        "match_name": best['name'],
        "match_id": best['id'],
        "score": display_score,
        "method": "AI Cross-Encoder",
        "inventory_index": best['index']
    }

def compute_semantic_matches(design_bom_df, inventory_df, bi_encoder, cross_encoder, threshold=0.5):
    """
    Orchestrates the matching process for the entire eBOM.
    """
    # Identify Inventory Columns
    inv_desc_col = next((c for c in inventory_df.columns if any(x in str(c).lower() for x in ["name", "desc"]) and "id" not in str(c).lower()), inventory_df.columns[0])
    inv_id_col = next((c for c in inventory_df.columns if "id" in str(c).lower()), inventory_df.columns[0])
    inv_cost_col = next((c for c in inventory_df.columns if any(x in str(c).lower() for x in ["cost", "price", "unit"])), None)
    inv_stock_col = next((c for c in inventory_df.columns if any(x in str(c).lower() for x in ["stock", "qty", "avail", "quantity"])), None)

    # Pre-calculate Inventory Embeddings (Bi-Encoder)
    inventory_names = inventory_df[inv_desc_col].astype(str).tolist()
    inventory_ids = inventory_df[inv_id_col].astype(str).tolist()
    inventory_embeddings = bi_encoder.encode(inventory_names, convert_to_tensor=True)

    results = []

    for _, row in design_bom_df.iterrows():
        query = row["Description"]
        req_qty = row["Quantity"]
        
        # Call the Advanced Matcher
        match_result = find_best_match(
            query, inventory_names, inventory_ids, inventory_embeddings, 
            bi_encoder, cross_encoder
        )
        
        if match_result["score"] >= threshold:
            idx = match_result["inventory_index"]
            matched_name = match_result["match_name"]
            conf_score = f"{round(match_result['score'] * 100, 1)}%"
            status = f"✅ {match_result['method']}"
            
            # Financials & Stock
            unit_cost = float(inventory_df.iloc[idx][inv_cost_col]) if inv_cost_col else 0.0
            total_cost = req_qty * unit_cost
            stock_avail = float(inventory_df.iloc[idx][inv_stock_col]) if inv_stock_col else 0.0
            stock_msg = "📦 In Stock" if stock_avail >= req_qty else f"⚠️ Shortage (-{int(req_qty - stock_avail)})"
        else:
            matched_name, conf_score, unit_cost, total_cost, stock_msg, status = "N/A", f"{round(match_result['score'] * 100, 1)}%", "-", "-", "❓ Unknown", "❌ No Match Found"

        results.append({
            "Design Part (eBOM)": query,
            "Required Qty": req_qty,
            "Matched Inventory (mBOM)": matched_name,
            "Confidence": conf_score,
            "Unit Price ($)": unit_cost,
            "Total Cost ($)": total_cost,
            "Stock Status": stock_msg,
            "Match Status": status
        })

    return pd.DataFrame(results)
# --------------------------------------------------
# 4. MAIN UI PAGES
# --------------------------------------------------

# Sidebar Navigation
st.sidebar.title("BOMGenius")
page = st.sidebar.radio("Navigate", ["Dashboard", "BOM Converter", "Federated Learning"])

def render_dashboard(current_threshold):
    st.title("Executive Dashboard")
    st.markdown("Real-time analytics of the eBOM to mBOM transformation process.")
    
    # Dummy Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Match Accuracy", "94.2%", "+1.2%")
    col2.metric("Auto-Matched Parts", "1,240", "Last 24h")
    col3.metric("Est. Cost Savings", "₹72,450", "Avoided Duplicates")
    col4.metric("Pending KB Votes", len(st.session_state.knowledge_base), "Active Learning")
    
    # Charts
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Inventory Distribution")
        # Dummy Data
        df_chart = pd.DataFrame({
            'Category': ['Fasteners', 'Electronics', 'Raw Metal', 'Plastics', 'Consumables'],
            'Count': [450, 300, 150, 200, 100]
        })
        fig = px.pie(df_chart, values='Count', names='Category', title='mBOM Composition')
        st.plotly_chart(fig, use_container_width=True)
        
    with c2:
        st.subheader("AI Confidence Levels")
        df_conf = pd.DataFrame({
            'Confidence': ['High (>90%)', 'Medium (70-90%)', 'Low (<70%)'],
            'Items': [850, 300, 90]
        })
        fig2 = px.bar(df_conf, x='Confidence', y='Items', color='Confidence', title='Match Reliability')
        st.plotly_chart(fig2, use_container_width=True)

def render_bom_converter(bi_encoder, cross_encoder, threshold=0.5):
    st.title("eBOM to mBOM Transformation Engine")
    st.info("**Multimodal AI Engine**: Upload CAD (DXF), PDFs, Images, JSON, CSV or Excel.")

    col_up1, col_up2 = st.columns(2)
    
    # 1. MULTIMODAL FILE UPLOADERS
    with col_up1:
        ebom_file = st.file_uploader(
            "Upload eBOM (DXF, PDF, Image, Excel, JSON, CSV)", 
            type=["csv", "xlsx", "json", "png", "jpg", "pdf", "dxf"]
        )
    with col_up2:
        inventory_file = st.file_uploader(
            "Upload Factory Inventory (CSV)", 
            type=["csv"]
        )
    
    # 2. DATA LOADING (MOCK OR REAL)
    if ebom_file is None or inventory_file is None:
        st.warning("No files uploaded. Using **Generated Demo Data** for simulation.")
        # Mock eBOM
        df_design = pd.DataFrame({
            'PartID': ['E-101', 'E-102', 'E-103', 'E-104', 'E-105'],
            'Description': [
                'Hex Screw M6x20mm',       # Specific size
                'Steel Plate 10mm',        # Generic material
                'Power Cord US Standard',  # Electrical
                'Safety Glove Large',      # Consumable
                'M8 Bolt'                  # Tricky: M8 vs M6
            ],
            'Quantity': [10, 2, 5, 20, 10]
        })
        # Mock Inventory
        df_inventory = pd.DataFrame({
            'ItemID': ['INV-500', 'INV-501', 'INV-502', 'INV-503', 'INV-504', 'INV-505', 'INV-506'],
            'ItemName': [
                'Screw, Hex Head, M6 x 20, Steel', 
                'Screw, Hex Head, M8 x 20, Steel', 
                'Plate, Sheet Metal, 10mm thick', 
                'Cable, Power, NEMA 5-15P, 3ft', 
                'Gloves, Nitrile, Size L', 
                'Bolt, Carriage, M8', 
                'Bolt, Carriage, M6'
            ],
            'UnitCost': [0.05, 0.08, 15.00, 3.50, 0.20, 0.10, 0.09],
            'StockQty': [5000, 4000, 50, 200, 1000, 3000, 3000]
        })
    else:
        # Real Data Ingestion
        with st.spinner("Ingesting Multimodal Files..."):
            df_design = parse_multimodal_ebom(ebom_file)
            try:
                df_inventory = pd.read_csv(inventory_file, encoding_errors='replace')
            except Exception as e:
                st.error(f"Inventory Load Error: {e}")
                df_inventory = pd.DataFrame()

    # 3. PREVIEW DATA
    if df_design is not None and not df_inventory.empty:
        with st.expander("🔍 View Input Data Previews"):
            c1, c2 = st.columns(2)
            c1.write("**Parsed eBOM**")
            c1.dataframe(df_design.head())
            c2.write("**Inventory mBOM**")
            c2.dataframe(df_inventory.head())

        # 4. ACTION BUTTON
        if st.button("Run AI Matching Process"):
            # Identify Inventory Columns Dynamically
            inv_desc_col = next((c for c in df_inventory.columns if any(x in str(c).lower() for x in ["name", "desc"]) and "id" not in str(c).lower()), df_inventory.columns[0])
            inv_id_col = next((c for c in df_inventory.columns if "id" in str(c).lower()), df_inventory.columns[0])
            inv_cost_col = next((c for c in df_inventory.columns if any(x in str(c).lower() for x in ["cost", "price"])), None)
            inv_stock_col = next((c for c in df_inventory.columns if any(x in str(c).lower() for x in ["stock", "qty"])), None)

            # Pre-compute Inventory Embeddings (Bi-Encoder)
            with st.spinner("Indexing Inventory (Vectorization)..."):
                inventory_names = df_inventory[inv_desc_col].astype(str).tolist()
                inventory_ids = df_inventory[inv_id_col].astype(str).tolist()
                inventory_embeddings = bi_encoder.encode(inventory_names, convert_to_tensor=True)
            
            # Process eBOM rows
            results = []
            progress_bar = st.progress(0)
            
            for i, row in df_design.iterrows():
                query = row['Description']
                req_qty = row.get('Quantity', 1)
                
                # Call the Advanced Hybrid Matcher (find_best_match logic)
                match_data = find_best_match(
                    query, 
                    inventory_names, 
                    inventory_ids, 
                    inventory_embeddings, 
                    bi_encoder, 
                    cross_encoder
                )
                
                # Financials & Stock Logic
                matched_inv_row = df_inventory[df_inventory[inv_id_col] == match_data['match_id']]
                
                if not matched_inv_row.empty and match_data['score'] >= threshold:
                    stock_qty = matched_inv_row[inv_stock_col].values[0] if inv_stock_col else 0
                    unit_cost = matched_inv_row[inv_cost_col].values[0] if inv_cost_col else 0
                    
                    stock_status = "✅ In Stock" if stock_qty >= req_qty else f"⚠️ Shortage ({int(stock_qty - req_qty)})"
                    total_cost = unit_cost * req_qty
                    match_status = f"AI: {match_data['method']}"
                else:
                    stock_status = "❓ Unknown"
                    unit_cost = 0
                    total_cost = 0
                    match_status = "❌ No Match Found"

                results.append({
                    "Design Part (eBOM)": query,
                    "Required Qty": req_qty,
                    "Matched Inventory (mBOM)": match_data['match_name'] if match_data['score'] >= threshold else "N/A",
                    "Confidence": f"{match_data['score']:.2%}",
                    "Status": match_status,
                    "Stock Status": stock_status,
                    "Unit Price": f"${unit_cost:.2f}",
                    "Total Cost": f"${total_cost:.2f}"
                })
                progress_bar.progress((i + 1) / len(df_design))
            
            # 5. DISPLAY RESULTS & EDITING
            st.success("Matching Complete!")
            results_df = pd.DataFrame(results)
            
            st.subheader("Finalized mBOM (Review & Edit)")
            # data_editor allows the user to manually override any AI mistakes
            edited_df = st.data_editor(
                results_df, 
                use_container_width=True,
                column_config={
                    "Confidence": st.column_config.TextColumn("Confidence", help="AI Certainty Score"),
                    "Stock Status": st.column_config.TextColumn("Stock Status")
                }
            )
            
            # 6. DOWNLOAD
            csv = edited_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download mBOM Export (CSV)", 
                data=csv, 
                file_name="mBOM_Transformation.csv", 
                mime="text/csv"
            )
    else:
        st.info("Awaiting file upload or using demo data...")
        
def render_federated_learning():
    st.title("Federated Learning Console")
    st.markdown("""
    **Human-in-the-Loop System:**
    1. **Factory A** proposes a correction. (Status: *Pending*)
    2. **Factory B** (or Manager) confirms the correction. (Status: *Verified*)
    3. The AI **immediately** learns this rule and overrides neural predictions.
    """)
    
    c1, c2 = st.columns([1, 2])
    
    with c1:
        st.subheader("Submit Correction")
        with st.form("learning_form"):
            ebom_input = st.text_input("eBOM Description (Wrong/Unmatched)")
            mbom_input = st.text_input("Correct Inventory Name")
            factory_id = st.selectbox("Voting Entity", ["Factory A", "Factory B", "HQ Engineer"])
            
            submitted = st.form_submit_button("Submit Vote")
            
            if submitted and ebom_input and mbom_input:
                kb = st.session_state.knowledge_base
                
                if ebom_input in kb:
                    # Entry exists, check logic
                    entry = kb[ebom_input]
                    if factory_id not in entry['votes']:
                        entry['votes'].add(factory_id)
                        # If more than 1 entity votes, it becomes verified
                        if len(entry['votes']) >= 2:
                            entry['status'] = "Verified"
                            st.success(f"Vote added! Rule for '{ebom_input}' is now VERIFIED.")
                        else:
                            st.info(f"Vote added. Still Pending verification.")
                    else:
                        st.warning("You have already voted on this rule.")
                else:
                    # New Entry
                    kb[ebom_input] = {
                        'target': mbom_input,
                        'votes': {factory_id},
                        'status': "Pending"
                    }
                    st.success("New rule proposed. Status: Pending.")

    with c2:
        st.subheader("Global Knowledge Ledger")
        
        if not st.session_state.knowledge_base:
            st.info("No manual rules learned yet.")
        else:
            # Convert dict to dataframe for display
            kb_data = []
            for k, v in st.session_state.knowledge_base.items():
                kb_data.append({
                    "eBOM Term": k,
                    "Mapped To": v['target'],
                    "Votes": len(v['votes']),
                    "Voters": ", ".join(list(v['votes'])),
                    "Status": v['status']
                })
            
            kb_df = pd.DataFrame(kb_data)
            
            # Color coding function
            def color_status(val):
                color = '#90EE90' if val == 'Verified' else '#FFFFE0' # LightGreen vs LightYellow
                return f'background-color: {color}; color: black'
            
            st.dataframe(kb_df.style.applymap(color_status, subset=['Status']))

# --------------------------------------------------
# 5. APP EXECUTION
# --------------------------------------------------

def main():
    configure_application_ui()
    
    # --- IMPORTANT FIX: Initialize BOTH Threshold and Brain ---
    if "global_threshold" not in st.session_state:
        st.session_state["global_threshold"] = 0.65  # General AI Strictness

    if "global_knowledge_base" not in st.session_state:
        st.session_state["global_knowledge_base"] = {} # Specific Learned Words

    bi_encoder, cross_encoder = load_models()

    # Page Routing
    if page == "Dashboard":
        render_dashboard(st.session_state["global_threshold"])
    elif page == "BOM Converter":
        render_bom_converter(bi_encoder, cross_encoder, st.session_state["global_threshold"])
    elif page == "Federated Learning":
        render_federated_learning()

if __name__ == "__main__":
    main()