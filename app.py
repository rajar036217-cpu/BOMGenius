import streamlit as st
import pandas as pd
import plotly.express as px
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import time

# -----------------------------------------------------------------------------
# 1. APP CONFIGURATION & STATE MANAGEMENT
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Enterprise BOM Matcher", layout="wide", page_icon="🏭")

# Initialize Session State for Federated Learning (The "Global Knowledge Base")
if 'knowledge_base' not in st.session_state:
    # Structure: {'eBOM_Name': {'target': 'Inventory_Name', 'votes': set(['FactoryA']), 'status': 'Pending'}}
    st.session_state.knowledge_base = {}

if 'history_log' not in st.session_state:
    st.session_state.history_log = []

# -----------------------------------------------------------------------------
# 2. AI MODEL LOADER (Cached for Performance)
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# 3. CORE MATCHING LOGIC
# -----------------------------------------------------------------------------
def find_best_match(query, inventory_names, inventory_ids, inventory_embeddings):
    """
    1. Checks Knowledge Base (Federated Learning).
    2. Uses Bi-Encoder to find Top 5 candidates.
    3. Uses Cross-Encoder to re-rank and pick the winner.
    """
    
    # --- STEP 0: Check Knowledge Base (Federated Learning Override) ---
    kb = st.session_state.knowledge_base
    if query in kb:
        entry = kb[query]
        # Only use if Verified (Green) or if we want to allow Pending (Yellow) matches
        # For Enterprise safety, usually we trust 'Verified', but here we'll take any manual override
        return {
            "match_name": entry['target'],
            "match_id": "MANUAL-OVERRIDE",
            "score": 1.0,
            "method": f"Knowledge Base ({entry['status']})"
        }

    # --- STEP 1: Retrieval (Bi-Encoder) ---
    query_embedding = bi_encoder.encode(query, convert_to_tensor=True)
    
    # Semantic Search: Find top 5 loose matches
    hits = util.semantic_search(query_embedding, inventory_embeddings, top_k=5)
    hits = hits[0] # Get first query results
    
    candidate_indices = [hit['corpus_id'] for hit in hits]
    candidate_names = [inventory_names[i] for i in candidate_indices]
    
    # --- STEP 2: Re-Ranking (Cross-Encoder) ---
    # Prepare pairs: [('M6 Screw', 'Screw M6'), ('M6 Screw', 'Bolt M8')...]
    model_inputs = [[query, candidate] for candidate in candidate_names]
    
    # Predict scores (Logits)
    scores = cross_encoder.predict(model_inputs)
    
    # Combine results
    results = []
    for idx, score in zip(candidate_indices, scores):
        results.append({
            "id": inventory_ids[idx],
            "name": inventory_names[idx],
            "score": score
        })
    
    # Sort by Cross-Encoder score (Descending)
    results = sorted(results, key=lambda x: x['score'], reverse=True)
    best_match = results[0]
    
    # Normalize score roughly to 0-1 for UI display (Sigmoid-ish visualization)
    # Note: Cross-Encoder outputs logits, can be negative.
    display_score = 1 / (1 + torch.exp(torch.tensor(-best_match['score']))).item()

    return {
        "match_name": best_match['name'],
        "match_id": best_match['id'],
        "score": display_score,
        "method": "AI Cross-Encoder"
    }

# -----------------------------------------------------------------------------
# 4. UI LAYOUT & PAGES
# -----------------------------------------------------------------------------

# Sidebar Navigation
st.sidebar.title("🏭 Smart BOM Matcher")
page = st.sidebar.radio("Navigate", ["Dashboard", "BOM Converter", "Federated Learning"])

# --- PAGE 1: DASHBOARD ---
if page == "Dashboard":
    st.title("Executive Dashboard")
    st.markdown("Real-time analytics of the eBOM to mBOM transformation process.")
    
    # Dummy Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Match Accuracy", "94.2%", "+1.2%")
    col2.metric("Auto-Matched Parts", "1,240", "Last 24h")
    col3.metric("Est. Cost Savings", "$12,450", "Avoided Duplicates")
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

# --- PAGE 2: BOM CONVERTER ---
elif page == "BOM Converter":
    st.title("eBOM to mBOM Transformation Engine")
    st.info("The Bi-Encoder retrieves candidates. The Cross-Encoder judges strict specs (e.g., M6 vs M8).")
    
    col_up1, col_up2 = st.columns(2)
    
    # File Uploaders
    ebom_file = col_up1.file_uploader("Upload eBOM (CSV)", type=['csv'])
    inv_file = col_up2.file_uploader("Upload Inventory mBOM (CSV)", type=['csv'])
    
    # GENERATE MOCK DATA IF NO FILES UPLOADED (For Demo Purposes)
    if ebom_file is None or inv_file is None:
        st.warning("⚠️ No files uploaded. Using **Generated Demo Data** for simulation.")
        
        # Mock eBOM (Engineering Descriptions - often messy)
        ebom_df = pd.DataFrame({
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
        
        # Mock Inventory (ERP Data - Clean, strict naming)
        inv_df = pd.DataFrame({
            'ItemID': ['INV-500', 'INV-501', 'INV-502', 'INV-503', 'INV-504', 'INV-505', 'INV-506'],
            'ItemName': [
                'Screw, Hex Head, M6 x 20, Steel',    # Match for E-101
                'Screw, Hex Head, M8 x 20, Steel',    # Trap for E-101
                'Plate, Sheet Metal, 10mm thick',     # Match for E-102
                'Cable, Power, NEMA 5-15P, 3ft',      # Match for E-103
                'Gloves, Nitrile, Size L',            # Match for E-104
                'Bolt, Carriage, M8',                 # Match for E-105
                'Bolt, Carriage, M6'                  # Trap for E-105
            ],
            'UnitCost': [0.05, 0.08, 15.00, 3.50, 0.20, 0.10, 0.09],
            'StockQty': [5000, 4000, 50, 200, 1000, 3000, 3000]
        })
    else:
        ebom_df = pd.read_csv(ebom_file)
        inv_df = pd.read_csv(inv_file)

    # Show Data Previews
    with st.expander("View Input Data"):
        c1, c2 = st.columns(2)
        c1.write("**eBOM Preview**")
        c1.dataframe(ebom_df.head())
        c2.write("**Inventory Preview**")
        c2.dataframe(inv_df.head())

    # ACTION BUTTON
    if st.button("🚀 Run AI Matching Process"):
        
        # 1. Pre-compute Inventory Embeddings (Bi-Encoder)
        # In production, this is done once and stored in a Vector DB (Milvus/Pinecone)
        with st.spinner("Indexing Inventory (Vectorization)..."):
            inventory_names = inv_df['ItemName'].tolist()
            inventory_ids = inv_df['ItemID'].tolist()
            inventory_embeddings = bi_encoder.encode(inventory_names, convert_to_tensor=True)
        
        # 2. Process eBOM rows
        results = []
        progress_bar = st.progress(0)
        
        for i, row in ebom_df.iterrows():
            query = row['Description']
            
            # Run the Hybrid Matcher
            match_data = find_best_match(query, inventory_names, inventory_ids, inventory_embeddings)
            
            # Determine Stock Status
            matched_inv_row = inv_df[inv_df['ItemID'] == match_data['match_id']]
            stock_qty = matched_inv_row['StockQty'].values[0] if not matched_inv_row.empty else 0
            unit_cost = matched_inv_row['UnitCost'].values[0] if not matched_inv_row.empty else 0
            
            status = "✅ In Stock" if stock_qty >= row['Quantity'] else "❌ Shortage"
            
            results.append({
                "Design Part (eBOM)": query,
                "Matched Inventory (mBOM)": match_data['match_name'],
                "Confidence": f"{match_data['score']:.2%}",
                "Method": match_data['method'],
                "Stock Status": status,
                "Price": f"${unit_cost}",
                "Total Cost": f"${unit_cost * row['Quantity']:.2f}"
            })
            progress_bar.progress((i + 1) / len(ebom_df))
            
        # 3. Display Results
        st.success("Matching Complete!")
        result_df = pd.DataFrame(results)
        
        # Styling the dataframe
        def highlight_confidence(val):
            score = float(val.strip('%'))
            color = 'green' if score > 85 else 'orange' if score > 60 else 'red'
            return f'color: {color}'

        st.dataframe(result_df)
        
        # Download
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download mBOM CSV", csv, "mBOM_Export.csv", "text/csv")

# --- PAGE 3: FEDERATED LEARNING ---
elif page == "Federated Learning":
    st.title("🧠 Federated Learning Console")
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