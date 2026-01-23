"""
BOMGenius (Final POC Version)
------------------------------------------
Features:
1. AI-Powered Matching (BERT)
2. Federated Learning Simulation (Graph + Feedback)
3. Downloadable Results
4. Professional Dashboard UI
"""

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import json
import plotly.express as px
from PIL import Image
import io
import pdfplumber
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------
# 1. SETUP & CONFIGURATION
# --------------------------------------------------

def configure_application_ui():
    st.set_page_config(
        page_title="BOMGenius",
        page_icon="🏭",
        layout="wide"
    )
    # Custom CSS for Professional Look
    st.markdown(
        """
        <style>
        .stApp { background-color: #0e1117; color: white; }
        .stButton>button { width: 100%; border-radius: 5px; height: 3em; }
        </style>
        """,
        unsafe_allow_html=True
    )

# --------------------------------------------------
# 2. LOAD AI MODEL (Cached)
# --------------------------------------------------

@st.cache_resource
def load_semantic_engine():
    """Loads the SBERT model once and keeps it in memory."""
    return SentenceTransformer("all-MiniLM-L6-v2")

# --------------------------------------------------
# 3. MULTIMODAL INGESTION ENGINE
# --------------------------------------------------

def extract_from_pdf(pdf_file):
    """
    Extracts table data or text from a PDF file and converts it to a DataFrame.
    """
    extracted_data = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            # Try to extract tables first
            table = page.extract_table()
            if table:
                df_page = pd.DataFrame(table[1:], columns=table[0])
                extracted_data.append(df_page)
            else:
                # Fallback: Extract text if no table is found
                text = page.extract_text()
                if text:
                    # Simple split logic to simulate a table from text
                    lines = [line.split() for line in text.split('\n') if len(line.split()) > 1]
                    extracted_data.append(pd.DataFrame(lines))
    
    if extracted_data:
        full_df = pd.concat(extracted_data, ignore_index=True)
        return full_df
    return pd.DataFrame(columns=["Description", "Quantity"])

def extract_from_cad_image(image_file):
    mock_extracted_data = [
        {"Description": "M8x25 Stainless Steel Hexagonal Bolt", "Quantity": 12},
        {"Description": "High-Temp Silicon Gasket 50mm", "Quantity": 4},
        {"Description": "Reinforced Steel Plate 200x200mm", "Quantity": 2},
        {"Description": "Analog Pressure Release Valve 10 Bar", "Quantity": 1}
    ]
    return pd.DataFrame(mock_extracted_data)

def parse_multimodal_ebom(uploaded_file):
    file_ext = uploaded_file.name.split('.')[-1].lower()
    with st.status(f"🔍 Universal Scanning {file_ext.upper()}...", expanded=True) as status:
        try:
            if file_ext == 'csv':
                df = pd.read_csv(uploaded_file, encoding_errors='replace')
            elif file_ext in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file)
            elif file_ext == 'pdf':
                df = extract_from_pdf(uploaded_file)
            elif file_ext in ['png', 'jpg', 'jpeg']:
                df = extract_from_cad_image(uploaded_file)
            elif file_ext == 'json':
                # --- UNIVERSAL JSON WALKER ---
                data = json.load(uploaded_file)
                all_items = []

                def walk_json(obj):
                    if isinstance(obj, dict):
                        # Look for components specifically
                        if "components" in obj and isinstance(obj["components"], list):
                            for comp in obj["components"]:
                                all_items.append({
                                    "Description": comp.get("item_name") or comp.get("name"),
                                    "Quantity": comp.get("qty") or comp.get("quantity") or 1
                                })
                        # Also look for parts_list (for your other JSON format)
                        elif "parts_list" in obj and isinstance(obj["parts_list"], list):
                            for part in obj["parts_list"]:
                                all_items.append({
                                    "Description": part.get("name") or part.get("item_name"),
                                    "Quantity": part.get("qty") or part.get("quantity") or 1
                                })
                        
                        # Keep walking deeper into the tree
                        for value in obj.values():
                            walk_json(value)
                    elif isinstance(obj, list):
                        for item in obj:
                            walk_json(item)

                walk_json(data)
                df = pd.DataFrame(all_items)
            else:
                return None
            
            # --- NORMALIZATION ---
            if df is None or df.empty:
                return None

            final_ebom = pd.DataFrame()
            # Find the best columns
            d_col = next((c for c in df.columns if any(x in str(c).lower() for x in ["name", "desc", "item"])), df.columns[0])
            q_col = next((c for c in df.columns if any(x in str(c).lower() for x in ["qty", "quantity", "count"])), None)

            final_ebom["Description"] = df[d_col].astype(str)
            if q_col:
                # Force conversion to number
                final_ebom["Quantity"] = pd.to_numeric(df[q_col], errors='coerce').fillna(1)
            else:
                final_ebom["Quantity"] = 1
            
            status.update(label=f"✅ Extracted {len(final_ebom)} items", state="complete")
            return final_ebom

        except Exception as e:
            st.error(f"⚠️ Ingestion Error: {e}")
            return None

def compute_semantic_matches(design_bom_df, inventory_df, semantic_engine, threshold):
    # 1. Detect Inventory Columns (Prioritizing Name over ID)
    inv_desc_col = next((c for c in inventory_df.columns if any(x in str(c).lower() for x in ["name", "desc"]) and "id" not in str(c).lower()), inventory_df.columns[0])
    inv_cost_col = next((c for c in inventory_df.columns if any(x in str(c).lower() for x in ["cost", "price", "unit"])), None)
    inv_stock_col = next((c for c in inventory_df.columns if any(x in str(c).lower() for x in ["stock", "qty", "avail", "quantity"])), None)

    # 2. Prepare Lists for AI
    design_names = design_bom_df["Description"].tolist()
    inventory_names = inventory_df[inv_desc_col].tolist()

    # 3. AI Embeddings
    design_embeddings = semantic_engine.encode(design_names)
    inventory_embeddings = semantic_engine.encode(inventory_names)
    similarity_matrix = cosine_similarity(design_embeddings, inventory_embeddings)

    results = []
    global_mem = st.session_state.get("global_knowledge_base", {})

    # 4. Matching Loop
    for i in range(len(design_names)):
        d_name = design_names[i]
        # Get the EXACT quantity from our normalized eBOM
        req_qty = design_bom_df.iloc[i]["Quantity"]
        
        best_idx = np.argmax(similarity_matrix[i])
        best_score = similarity_matrix[i][best_idx]
        
        # Check Federated Learning Knowledge Base
        is_learned = False
        if d_name in global_mem:
            matched_name = global_mem[d_name]["target"]
            match_rows = inventory_df[inventory_df[inv_desc_col] == matched_name]
            if not match_rows.empty:
                best_idx, best_score, is_learned = match_rows.index[0], 1.0, True

        if best_score >= threshold or is_learned:
            status = "🧠 FL Learned" if is_learned else "✅ AI Auto-Match"
            matched_name = inventory_names[best_idx]
            conf_score = f"{round(float(best_score) * 100, 1)}%"
            
            # Unit Cost calculation
            unit_cost = float(inventory_df.iloc[best_idx][inv_cost_col]) if inv_cost_col else 0.0
            total_cost = req_qty * unit_cost
            
            # Stock check
            stock_avail = float(inventory_df.iloc[best_idx][inv_stock_col]) if inv_stock_col else 0.0
            stock_msg = "📦 In Stock" if stock_avail >= req_qty else f"⚠️ Shortage (-{int(req_qty - stock_avail)})"
        else:
            status, matched_name, conf_score, unit_cost, total_cost, stock_msg = "❌ No Match Found", "N/A", f"{round(float(best_score) * 100, 1)}%", "-", "-", "❓ Unknown"

        results.append({
            "Design Part (eBOM)": d_name,
            "Required Qty": req_qty,
            "Matched Inventory (mBOM)": matched_name,
            "Confidence": conf_score,
            "Unit Price ($)": unit_cost,
            "Total Cost ($)": total_cost,
            "Stock Status": stock_msg,
            "Match Status": status
        })

    return pd.DataFrame(results)

def federated_learning_update(wrong_part, correct_part, factory_name):
    """
    Advanced FL: Implements Voting & Aggregation.
    Only updates if consensus is reached or tracks votes.
    """
    kb = st.session_state["global_knowledge_base"]
    
    # Current Time
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    
    # Check if this mismatch is already tracked
    if wrong_part in kb:
        # Check if the target is the same (Consensus check)
        if kb[wrong_part]["target"] == correct_part:
            # Avoid duplicate votes from same factory
            if factory_name not in kb[wrong_part]["sources"]:
                kb[wrong_part]["votes"] += 1
                kb[wrong_part]["sources"].append(factory_name)
                kb[wrong_part]["last_updated"] = timestamp
                
                # Update Status based on votes
                if kb[wrong_part]["votes"] >= 2:
                    kb[wrong_part]["status"] = "🟢 Verified Global Rule"
        else:
            # Conflict! Overwrite for now (Simple Logic)
            kb[wrong_part] = {
                "target": correct_part,
                "votes": 1,
                "sources": [factory_name],
                "last_updated": timestamp,
                "status": "🟡 Pending Verification"
            }
    else:
        # New Entry
        kb[wrong_part] = {
            "target": correct_part,
            "votes": 1,
            "sources": [factory_name],
            "last_updated": timestamp,
            "status": "🟡 Pending Verification"
        }
    
    return len(kb)

# --------------------------------------------------
# 4. MAIN UI PAGES
# --------------------------------------------------

def render_dashboard(current_threshold):
    st.title("📊 Executive Analytics Dashboard")
    st.markdown("### Real-time Supply Chain Insights")

    # --- Top Metrics Row ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Global Match Accuracy", f"{int(current_threshold*100)}%", "+2.4%")
    col2.metric("Connected Factories", "2 Sites", "India & Germany")
    col3.metric("Pending BOMs", "12 Files", "-3 from yesterday")
    col4.metric("Cost Savings", "$14,200", "Updated 1hr ago")

    st.divider()

    # --- Charts Section (Using Sample Data for Demo) ---
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("📦 Inventory Stock Status")
        # Creating dummy data for visualization
        stock_data = pd.DataFrame({
            "Status": ["In Stock", "Low Stock", "Out of Stock"],
            "Count": [450, 120, 30]
        })
        fig_pie = px.pie(stock_data, values="Count", names="Status", 
                         title="Material Availability Distribution",
                         color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig_pie, width='stretch')

    with c2:
        st.subheader("💰 Cost Analysis by Category")
        cost_data = pd.DataFrame({
            "Category": ["Electronics", "Mechanical", "Fasteners", "Packaging"],
            "Cost ($)": [15000, 8000, 2000, 1500]
        })
        fig_bar = px.bar(cost_data, x="Category", y="Cost ($)", 
                         title="Procurement Cost Breakdown",
                         color="Cost ($)", color_continuous_scale="Viridis")
        st.plotly_chart(fig_bar, width='stretch')

    # --- Recent Activity Log ---
    st.subheader("📝 Recent System Activity")
    st.dataframe(pd.DataFrame({
        "Timestamp": ["10:05 AM", "10:12 AM", "10:45 AM"],
        "User": ["Factory_Admin_IN", "System_AI", "Factory_Admin_DE"],
        "Action": ["Uploaded eBOM_v2.csv", "Auto-Matched 85 parts", "Flagged 'False Negative'"]
    }), width='stretch')

def render_bom_converter(semantic_engine, threshold):
    st.title("🔄 Intelligent eBOM → mBOM Converter")
    
    col1, col2 = st.columns(2)
    with col1:
        # Added PDF to the allowed types
        ebom_file = st.file_uploader("📂 Upload eBOM (PDF, Image, Excel, JSON, CSV)", 
                                     type=["csv", "xlsx", "json", "png", "jpg", "pdf"])
    with col2:
        inventory_file = st.file_uploader("🏭 Upload Factory Inventory (CSV)", type=["csv"])

    if ebom_file and inventory_file:
        st.success("✅ Files Uploaded!")
        
        if st.button("🚀 Run AI Matching Process"):
            with st.spinner("🤖 AI is analyzing semantic meanings..."):
                # Use the Multimodal Parser for the eBOM
                df_design = parse_multimodal_ebom(ebom_file)
                
                # Standard loader for Inventory
                try:
                    df_inventory = pd.read_csv(inventory_file, encoding_errors='replace')
                except:
                    df_inventory = pd.DataFrame()

                if df_design is not None and not df_inventory.empty:
                    results_df = compute_semantic_matches(df_design, df_inventory, semantic_engine, threshold)
                    
                    st.subheader("📋 Match Results")
                    # Use data_editor so user can fix AI mistakes
                    edited_df = st.data_editor(results_df, use_container_width=True)
                    
                    # Download Button
                    csv = edited_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Result CSV",
                        data=csv,
                        file_name="OptiBOM_Results.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("⚠️ Could not process files. Please check formats.")

def render_federated_learning():
    st.title("🌐 Federated Knowledge Network (Enterprise Grade)")

    if "global_knowledge_base" not in st.session_state:
        st.session_state["global_knowledge_base"] = {}

    tab_a, tab_b, tab_hq = st.tabs(["🏭 Factory A (India)", "🏭 Factory B (Germany)", "☁️ Central Server (HQ)"])

    # --- TAB 1: FACTORY A ---
    with tab_a:
        st.subheader("📍 Factory A Console (India)")
        st.info("Status: ✅ Online | Mode: Voting Node")
        
        col1, col2 = st.columns(2)
        with col1:
            wrong_a = st.text_input("Mismatch Name (eBOM):", key="wa", placeholder="e.g., Screw M6")
        with col2:
            correct_a = st.text_input("Correct Map (Inventory):", key="ca", placeholder="e.g., Fastener-Steel-06")

        if st.button("📡 Vote/Push Correction (Factory A)"):
            if wrong_a and correct_a:
                # Passing "Factory A" as the source
                count = federated_learning_update(wrong_a, correct_a, "Factory A")
                st.success(f"✅ Vote Registered for '{wrong_a}'")
            else:
                st.warning("Enter both names.")

    # --- TAB 2: FACTORY B ---
    with tab_b:
        st.subheader("📍 Factory B Console (Germany)")
        st.info("Status: ✅ Online | Mode: Voting Node")
        
        col1, col2 = st.columns(2)
        with col1:
            wrong_b = st.text_input("Mismatch Name (eBOM):", key="wb")
        with col2:
            correct_b = st.text_input("Correct Map (Inventory):", key="cb")

        if st.button("📡 Vote/Push Correction (Factory B)"):
            if wrong_b and correct_b:
                # Passing "Factory B" as the source
                count = federated_learning_update(wrong_b, correct_b, "Factory B")
                st.success(f"✅ Vote Registered for '{wrong_b}'")
            else:
                st.warning("Enter both names.")

    # --- TAB 3: HQ (Analytics Dashboard) ---
    with tab_hq:
        st.subheader("🧠 Global Federated Brain (Analytics)")
        
        kb = st.session_state["global_knowledge_base"]
        
        if kb:
            # 1. Metrics
            total_rules = len(kb)
            verified_rules = sum(1 for v in kb.values() if v["votes"] >= 2)
            
            c1, c2 = st.columns(2)
            c1.metric("Total Learned Patterns", total_rules)
            c2.metric("✅ Verified Rules (Consensus > 1)", verified_rules)
            
            # 2. Detailed Table
            st.write("### 📖 Live Knowledge Ledger")
            
            # Convert Dict to Table for display
            table_data = []
            for wrong, info in kb.items():
                table_data.append({
                    "Mismatch (Input)": wrong,
                    "Correction (Output)": info['target'],
                    "Votes": info['votes'],
                    "Status": info['status'], # 🟢 or 🟡
                    "Contributors": ", ".join(info['sources']),
                    "Last Update": info['last_updated']
                })
            
            st.dataframe(pd.DataFrame(table_data), width='stretch')
        else:
            st.info("System is waiting for updates from Factories...")

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

    semantic_engine = load_semantic_engine()

    # Sidebar Navigation
    with st.sidebar:
        st.title("BOMGenius")
        page = st.radio("Go to:", ["Dashboard", "BOM Converter", "Federated Learning"])
        st.divider()
        st.caption("Powered by L&T TECHgium POC")

    # Page Routing
    if page == "Dashboard":
        render_dashboard(st.session_state["global_threshold"])
    elif page == "BOM Converter":
        render_bom_converter(semantic_engine, st.session_state["global_threshold"])
    elif page == "Federated Learning":
        render_federated_learning()

if __name__ == "__main__":
    main()