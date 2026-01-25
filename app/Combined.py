import pandas as pd
import numpy as np
import torch
import json
import io
import pdfplumber
import ezdxf
import streamlit as st
import google.generativeai as genai
from PIL import Image
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import plotly.express as px
import time

# --------------------------------------------------
# 1. SETUP & CONFIGURATION
# --------------------------------------------------

def configure_application_ui():
    st.set_page_config(page_title="BOMGenius", layout="wide")
    
    # --- GEMINI API CONFIGURATION (Option 2: Secrets) ---
    gemini_key = st.secrets.get("GEMINI_API_KEY")
    
    with st.sidebar:
        st.title("System Config")
        if gemini_key:
            genai.configure(api_key=gemini_key)
            st.success("Gemini AI Connected (via Secrets)")
        else:
            st.error("Gemini API Key not found in .streamlit/secrets.toml")
            st.info("Please add GEMINI_API_KEY to your secrets file.")
        st.divider()

    # Initialize Session State
    if 'knowledge_base' not in st.session_state:
        st.session_state.knowledge_base = {}
    if 'history_log' not in st.session_state:
        st.session_state.history_log = []

# --------------------------------------------------
# 2. LOAD AI MODEL (Cached)
# --------------------------------------------------

@st.cache_resource
def load_models():
    """
    Loads the Bi-Encoder and Cross-Encoder explicitly on CPU 
    to avoid meta-tensor initialization errors.
    """
    # Force CPU to prevent the 'meta tensor' NotImplementedError
    device = "cpu" 
    
    with st.spinner("Loading AI Models (Bi-Encoder + Cross-Encoder)..."):
        # Stage 1: Bi-Encoder
        bi_encoder = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        
        # Stage 2: Cross-Encoder
        cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device=device)
        
    return bi_encoder, cross_encoder

# --------------------------------------------------
# 3. GEMINI EXTRACTION LOGIC
# --------------------------------------------------

def extract_with_gemini(file_bytes, mime_type):
    """
    Uses Gemini 1.5 Flash to extract BOM tables from Images or PDFs visually.
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = """
        Extract the Bill of Materials (BOM) from this file. 
        Return ONLY a JSON array of objects with keys: "Description" and "Quantity".
        Example: [{"Description": "M8 Bolt", "Quantity": 10}]
        Ensure Quantity is a number. If not specified, default to 1.
        """
        
        if "image" in mime_type:
            img = Image.open(io.BytesIO(file_bytes))
            response = model.generate_content([prompt, img])
        else:
            # PDF Handling
            response = model.generate_content([
                prompt,
                {'mime_type': 'application/pdf', 'data': file_bytes}
            ])
            
        # Clean response and parse JSON
        clean_json = response.text.replace('```json', '').replace('```', '').strip()
        data = json.loads(clean_json)
        return pd.DataFrame(data)
    except Exception as e:
        st.error(f"Gemini Vision Error: {e}")
        return None

# --------------------------------------------------
# 4. MULTIMODAL INGESTION ENGINE (Upgraded)
# --------------------------------------------------

def extract_from_pdf_local(pdf_file):
    """Fallback PDF parser if Gemini is unavailable"""
    extracted_data = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            table = page.extract_table()
            if table:
                df_page = pd.DataFrame(table[1:], columns=table[0])
                extracted_data.append(df_page)
    return pd.concat(extracted_data, ignore_index=True) if extracted_data else None

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
        return final_ebom
    except Exception as e:
        st.error(f"Ingestion Error: {e}")
        return None

# --------------------------------------------------
# SECTION 5: ADVANCED MATCHING ENGINE
# --------------------------------------------------

def find_best_match(query, inventory_names, inventory_ids, inventory_embeddings, bi_encoder, cross_encoder):
    kb = st.session_state.get('knowledge_base', {})
    if query in kb:
        entry = kb[query]
        return {"match_name": entry['target'], "match_id": "MANUAL-OVERRIDE", "score": 1.0, "method": f"Knowledge Base ({entry.get('status', 'Verified')})"}

    query_embedding = bi_encoder.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, inventory_embeddings, top_k=5)[0]
    
    candidate_indices = [hit['corpus_id'] for hit in hits]
    candidate_names = [inventory_names[i] for i in candidate_indices]
    
    model_inputs = [[query, candidate] for candidate in candidate_names]
    scores = cross_encoder.predict(model_inputs)
    
    results = []
    for idx, score in zip(candidate_indices, scores):
        results.append({"index": idx, "id": inventory_ids[idx], "name": inventory_names[idx], "score": score})
    
    results = sorted(results, key=lambda x: x['score'], reverse=True)
    best = results[0]
    display_score = 1 / (1 + torch.exp(torch.tensor(-best['score']))).item()

    return {"match_name": best['name'], "match_id": best['id'], "score": display_score, "method": "AI Cross-Encoder", "inventory_index": best['index']}

# --------------------------------------------------
# SECTION 6: UI PAGES
# --------------------------------------------------

st.sidebar.title("BOMGenius")
page = st.sidebar.radio("Navigate", ["Dashboard", "BOM Converter", "Federated Learning"])

def render_dashboard():
    st.title("Executive Dashboard")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Match Accuracy", "94.2%", "+1.2%")
    col2.metric("Auto-Matched Parts", "1,240", "Last 24h")
    col3.metric("Est. Cost Savings", "72,450", "Avoided Duplicates")
    col4.metric("Pending KB Votes", len(st.session_state.knowledge_base), "Active Learning")
    
    c1, c2 = st.columns(2)
    with c1:
        df_chart = pd.DataFrame({'Category': ['Fasteners', 'Electronics', 'Raw Metal', 'Plastics', 'Consumables'], 'Count': [450, 300, 150, 200, 100]})
        st.plotly_chart(px.pie(df_chart, values='Count', names='Category', title='mBOM Composition'), use_container_width=True)
    with c2:
        df_conf = pd.DataFrame({'Confidence': ['High (>90%)', 'Medium (70-90%)', 'Low (<70%)'], 'Items': [850, 300, 90]})
        st.plotly_chart(px.bar(df_conf, x='Confidence', y='Items', color='Confidence', title='Match Reliability'), use_container_width=True)

def render_bom_converter(bi_encoder, cross_encoder, threshold=0.5):
    st.title("eBOM to mBOM Transformation Engine")
    
    # Initialize Session State
    if "processed_ebom" not in st.session_state:
        st.session_state.processed_ebom = None

    col_up1, col_up2 = st.columns(2)
    with col_up1:
        ebom_file = st.file_uploader("Upload eBOM (DXF, PDF, Image, Excel, JSON, CSV)", type=["csv", "xlsx", "json", "png", "jpg", "pdf", "dxf"])
    with col_up2:
        inventory_file = st.file_uploader("Upload Factory Inventory (CSV)", type=["csv"])
    
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
        if st.button("Run Full AI Transformation", use_container_width=True):
            with st.status("Processing: Ingesting, Analyzing, and Matching...", expanded=True) as status:
                
                # 1. MULTIMODAL INGESTION
                status.write("Extracting data from source file...")
                df_design = parse_multimodal_ebom(ebom_file)
                
                if df_design is None or df_design.empty:
                    status.update(label="Extraction Failed", state="error")
                    st.error("Could not extract data from the eBOM file.")
                    return

                # 2. INVENTORY LOADING
                status.write("Loading factory inventory...")
                try:
                    df_inventory = pd.read_csv(inventory_file, encoding_errors='replace')
                except Exception as e:
                    status.update(label="Inventory Load Failed", state="error")
                    st.error(f"Error loading inventory: {e}")
                    return

                # 3. COLUMN DETECTION & VECTORIZATION
                status.write("Indexing inventory and running AI matching...")
                inv_desc_col = next((c for c in df_inventory.columns if any(x in str(c).lower() for x in ["name", "desc"]) and "id" not in str(c).lower()), df_inventory.columns[0])
                inv_id_col = next((c for c in df_inventory.columns if "id" in str(c).lower()), df_inventory.columns[0])
                inv_cost_col = next((c for c in df_inventory.columns if any(x in str(c).lower() for x in ["cost", "price"])), None)
                inv_stock_col = next((c for c in df_inventory.columns if any(x in str(c).lower() for x in ["stock", "qty"])), None)

                inventory_names = df_inventory[inv_desc_col].astype(str).tolist()
                inventory_ids = df_inventory[inv_id_col].astype(str).tolist()
                inventory_embeddings = bi_encoder.encode(inventory_names, convert_to_tensor=True)
            
                # 4. MATCHING LOOP
                results = []
                for i, row in df_design.iterrows():
                    query = row['Description']
                    req_qty = row.get('Quantity', 1)
                    
                    match_data = find_best_match(query, inventory_names, inventory_ids, inventory_embeddings, bi_encoder, cross_encoder)
                    
                    matched_inv_row = df_inventory[df_inventory[inv_id_col] == match_data['match_id']]
                    
                    if not matched_inv_row.empty and match_data['score'] >= threshold:
                        stock_qty = matched_inv_row[inv_stock_col].values[0] if inv_stock_col else 0
                        unit_cost = matched_inv_row[inv_cost_col].values[0] if inv_cost_col else 0
                        
                        stock_status = "In Stock" if stock_qty >= req_qty else f"Shortage ({int(stock_qty - req_qty)})"
                        total_cost = unit_cost * req_qty
                        match_status = f"AI: {match_data['method']}"
                    else:
                        stock_status, unit_cost, total_cost, match_status = "Unknown", 0, 0, "No Match Found"

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
                
                # Save to session state
                st.session_state.final_results = pd.DataFrame(results)
                status.update(label="Transformation Complete", state="complete")

    # 5. DISPLAY AND DOWNLOAD (Persistent UI)
    if st.session_state.final_results is not None:
        st.divider()
        st.subheader("Finalized mBOM Results")
        
        # Data editor for manual overrides
        edited_df = st.data_editor(st.session_state.final_results, use_container_width=True)
        
        # Download button
        csv = edited_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download mBOM Export (CSV)", 
            data=csv, 
            file_name="mBOM_Transformation.csv", 
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
            if st.form_submit_button("Submit Vote") and ebom_input and mbom_input:
                kb = st.session_state.knowledge_base
                if ebom_input in kb:
                    if factory_id not in kb[ebom_input]['votes']:
                        kb[ebom_input]['votes'].add(factory_id)
                        if len(kb[ebom_input]['votes']) >= 2: kb[ebom_input]['status'] = "Verified"
                else:
                    kb[ebom_input] = {'target': mbom_input, 'votes': {factory_id}, 'status': "Pending"}
                st.success("Vote Registered.")
    with c2:
        if st.session_state.knowledge_base:
            kb_data = [{"eBOM Term": k, "Mapped To": v['target'], "Votes": len(v['votes']), "Status": v['status']} for k, v in st.session_state.knowledge_base.items()]
            st.dataframe(pd.DataFrame(kb_data))

# --------------------------------------------------
# 7. MAIN EXECUTION
# --------------------------------------------------

def main():
    configure_application_ui()
    if "global_threshold" not in st.session_state: st.session_state["global_threshold"] = 0.65
    bi_encoder, cross_encoder = load_models()

    if page == "Dashboard": render_dashboard()
    elif page == "BOM Converter": render_bom_converter(bi_encoder, cross_encoder, st.session_state["global_threshold"])
    elif page == "Federated Learning": render_federated_learning()

if __name__ == "__main__":
    main()