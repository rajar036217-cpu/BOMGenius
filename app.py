"""
OptiBOM Enterprise Pro (Final POC Version)
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
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------
# 1. SETUP & CONFIGURATION
# --------------------------------------------------

def configure_application_ui():
    st.set_page_config(
        page_title="OptiBOM Enterprise Pro",
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
# 3. HELPER FUNCTIONS
# --------------------------------------------------

def load_csv_file(uploaded_file, expected_columns):
    """Safe CSV Loader"""
    try:
        df = pd.read_csv(uploaded_file)
        # Check if basic columns exist (Loose check)
        return df
    except Exception as error:
        st.error(f"❌ File loading failed: {error}")
        return None

def compute_semantic_matches(design_bom_df, inventory_df, semantic_engine, threshold):
    """
    Core Engine: Matches Design Parts to Inventory Parts.
    Feature A Upgrade: Calculates Total Cost & Stock Feasibility.
    """
    
    # --- 1. SMART COLUMN DETECTION (Industrial Standard) ---
    # Instead of hardcoding 'column 1', we search for names like 'Description', 'Part Name', etc.
    
    # Find Design Description Column
    design_desc_col = design_bom_df.columns[1] # Default fallback
    for col in design_bom_df.columns:
        if "desc" in col.lower() or "part" in col.lower() or "name" in col.lower():
            design_desc_col = col
            break
            
    # Find Inventory Columns (Description, Cost, Stock)
    inv_desc_col = inventory_df.columns[1] # Default fallback
    inv_cost_col = None
    inv_stock_col = None
    
    for col in inventory_df.columns:
        c_low = col.lower()
        if "desc" in c_low or "item" in c_low or "name" in c_low:
            inv_desc_col = col
        elif "cost" in c_low or "price" in c_low:
            inv_cost_col = col
        elif "stock" in c_low or "qty" in c_low or "available" in c_low:
            inv_stock_col = col

    # --- 2. AI MATCHING PROCESS ---
    design_names = design_bom_df[design_desc_col].astype(str).tolist()
    inventory_names = inventory_df[inv_desc_col].astype(str).tolist()

    design_embeddings = semantic_engine.encode(design_names)
    inventory_embeddings = semantic_engine.encode(inventory_names)

    similarity_matrix = cosine_similarity(design_embeddings, inventory_embeddings)

    results = []

    # --- 3. LOGIC LOOP (MATCHING + BUSINESS MATH) ---
    for i in range(len(design_names)):
        scores = similarity_matrix[i]
        best_idx = np.argmax(scores)
        best_score = scores[best_idx]

        # Logic: Check Threshold
        if best_score >= threshold:
            status = "✅ Auto-Match"
            matched_name = inventory_names[best_idx]
            conf_score = f"{round(best_score * 100, 1)}%"
            
            # --- COST CALCULATION ---
            # Get Required Qty (Default to 1 if missing)
            req_qty = 1
            if "Quantity" in design_bom_df.columns:
                try:
                    req_qty = float(design_bom_df.iloc[i]["Quantity"])
                except:
                    req_qty = 1
            
            # Get Unit Cost
            unit_cost = 0.0
            if inv_cost_col:
                try:
                    unit_cost = float(inventory_df.iloc[best_idx][inv_cost_col])
                except:
                    unit_cost = 0.0
            
            # Get Stock
            stock_avail = 0
            if inv_stock_col:
                try:
                    stock_avail = float(inventory_df.iloc[best_idx][inv_stock_col])
                except:
                    stock_avail = 0

            # The Business Math
            total_cost = req_qty * unit_cost
            
            # Stock Check
            if stock_avail >= req_qty:
                stock_msg = "📦 In Stock"
            else:
                shortage = req_qty - stock_avail
                stock_msg = f"⚠️ Shortage (-{int(shortage)})"

        else:
            # Fallback for No Match
            status = "❌ No Match Found"
            matched_name = "N/A"
            conf_score = f"{round(best_score * 100, 1)}%"
            req_qty = "-"
            unit_cost = "-"
            total_cost = "-"
            stock_msg = "❓ Unknown"

        # Create Output Row
        results.append({
            "Design Part (eBOM)": design_names[i],
            "Required Qty": req_qty,
            "Matched Inventory (mBOM)": matched_name,
            "Confidence": conf_score,
            "Unit Price ($)": unit_cost,
            "Total Cost ($)": total_cost,
            "Stock Status": stock_msg,
            "Match Status": status
        })

    return pd.DataFrame(results)

def federated_learning_update(feedback, current_threshold):
    """Simulates the Federated Learning Update"""
    if feedback == "False Positive (Wrong Match)":
        current_threshold += 0.05 # Make AI stricter
    elif feedback == "False Negative (Missed Match)":
        current_threshold -= 0.05 # Make AI looser
    
    return max(0.4, min(current_threshold, 0.95))

# --------------------------------------------------
# 4. MAIN UI PAGES
# --------------------------------------------------

def render_dashboard(current_threshold):
    st.title("📊 Executive Dashboard")
    col1, col2, col3 = st.columns(3)
    col1.metric("AI Model Status", "Active", "BERT-v2")
    col2.metric("Connected Factories", "2 Sites", "Encrypted")
    col3.metric("Global Match Threshold", f"{int(current_threshold*100)}%")
    
    st.image("https://cdn-icons-png.flaticon.com/512/2620/2620603.png", width=100)
    st.info("System is ready for POC Demonstration.")

def render_bom_converter(semantic_engine, threshold):
    st.title("🔄 Intelligent eBOM → mBOM Converter")
    
    col1, col2 = st.columns(2)
    with col1:
        design_file = st.file_uploader("📂 Upload Engineer eBOM", type=["csv"])
    with col2:
        inventory_file = st.file_uploader("🏭 Upload Factory Inventory", type=["csv"])

    if design_file and inventory_file:
        st.success("✅ Files Ready!")
        
        # THE RUN BUTTON YOU ASKED FOR
        if st.button("🚀 Run AI Matching Process"):
            with st.spinner("🤖 AI is analyzing semantic meanings..."):
                df_design = pd.read_csv(design_file)
                df_inventory = pd.read_csv(inventory_file)
                
                results_df = compute_semantic_matches(df_design, df_inventory, semantic_engine, threshold)
                
                st.subheader("📋 Match Results")
                st.dataframe(results_df, use_container_width=True)
                
                # DOWNLOAD BUTTON
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Result CSV",
                    data=csv,
                    file_name="OptiBOM_Results.csv",
                    mime="text/csv"
                )

def render_federated_learning():
    st.title("🌐 Federated Learning Network (Dual-Client Simulation)")

    # Initialize History
    if "history" not in st.session_state:
        st.session_state["history"] = [st.session_state["global_threshold"]]

    # --- NEW: TAB SYSTEM (The Feature B Upgrade) ---
    tab_a, tab_b, tab_hq = st.tabs(["🏭 Factory A (India)", "🏭 Factory B (Germany)", "☁️ Central Server (HQ)"])

    # --- TAB 1: FACTORY A ---
    with tab_a:
        st.subheader("📍 Factory A Console")
        st.info("Status: ✅ Online | Connection: Secure VPN (India Node)")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            fb_a = st.radio("Technician Feedback (Factory A):", 
                          ["None", "False Positive (Wrong Match)", "False Negative (Missed Match)"], 
                          key="fb_a")
        with col2:
            st.metric("Local Threshold", f"{int(st.session_state['global_threshold']*100)}%")

        if st.button("📡 Push Update from Factory A"):
            if fb_a != "None":
                with st.spinner("Encrypting gradients & syncing with Global Model..."):
                    # Use the helper function to update
                    new_val = federated_learning_update(fb_a, st.session_state["global_threshold"])
                    st.session_state["global_threshold"] = new_val
                    st.session_state["history"].append(new_val)
                    st.success(f"✅ Gradient Update Sent! Global Model Adjusted.")
            else:
                st.warning("⚠️ Select an issue to report first.")

    # --- TAB 2: FACTORY B ---
    with tab_b:
        st.subheader("📍 Factory B Console")
        st.info("Status: ✅ Online | Connection: Secure VPN (Germany Node)")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            fb_b = st.radio("Technician Feedback (Factory B):", 
                          ["None", "False Positive (Wrong Match)", "False Negative (Missed Match)"], 
                          key="fb_b")
        with col2:
            st.metric("Local Threshold", f"{int(st.session_state['global_threshold']*100)}%")

        if st.button("📡 Push Update from Factory B"):
            if fb_b != "None":
                with st.spinner("Encrypting gradients & syncing with Global Model..."):
                    # Use the helper function to update
                    new_val = federated_learning_update(fb_b, st.session_state["global_threshold"])
                    st.session_state["global_threshold"] = new_val
                    st.session_state["history"].append(new_val)
                    st.success(f"✅ Gradient Update Sent! Global Model Adjusted.")
            else:
                st.warning("⚠️ Select an issue to report first.")

    # --- TAB 3: CENTRAL SERVER ---
    with tab_hq:
        st.subheader("🧠 Global Model Aggregation (Headquarters)")
        st.caption("Visualizing real-time weight updates from all connected factories.")
        
        # Professional Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Global Threshold", f"{int(st.session_state['global_threshold']*100)}%", delta="Live")
        m2.metric("Total Updates", len(st.session_state["history"])-1)
        m3.metric("Active Nodes", "2 Sites")

        # The Graph
        st.area_chart(st.session_state["history"], color="#00ff00")
# --------------------------------------------------
# 5. APP EXECUTION
# --------------------------------------------------

def main():
    configure_application_ui()
    
    # Initialize Session State
    if "global_threshold" not in st.session_state:
        st.session_state["global_threshold"] = 0.65 # Default 65% match

    semantic_engine = load_semantic_engine()

    # Sidebar Navigation
    with st.sidebar:
        st.title("🏭 OptiBOM Pro")
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
