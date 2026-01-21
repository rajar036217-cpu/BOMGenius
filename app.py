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
import datetime
import plotly.express as px
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

def compute_semantic_matches(design_bom_df, inventory_df, semantic_engine, threshold):
    # Find Design Description Column
    design_desc_col = design_bom_df.columns[1] 
    for col in design_bom_df.columns:
        if "desc" in col.lower() or "part" in col.lower() or "name" in col.lower():
            design_desc_col = col
            break
            
    # Find Inventory Columns
    inv_desc_col = inventory_df.columns[1]
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

    design_names = design_bom_df[design_desc_col].astype(str).tolist()
    inventory_names = inventory_df[inv_desc_col].astype(str).tolist()

    design_embeddings = semantic_engine.encode(design_names)
    inventory_embeddings = semantic_engine.encode(inventory_names)
    similarity_matrix = cosine_similarity(design_embeddings, inventory_embeddings)

    results = []
    
    # Get Global Brain
    global_mem = st.session_state.get("global_knowledge_base", {})

    for i in range(len(design_names)):
        d_name = design_names[i]
        
        # --- LOGIC CHANGE: Handle Complex Dictionary Structure ---
        is_learned = False
        if d_name in global_mem:
            # NEW: We now access ["target"] because dictionary has votes/status
            matched_name = global_mem[d_name]["target"]
            
            # Verify if this matches verified status (Optional: Require 2 votes)
            # For now, we take it if it exists.
            
            match_rows = inventory_df[inventory_df[inv_desc_col] == matched_name]
            
            if not match_rows.empty:
                best_idx = match_rows.index[0]
                best_score = 1.0
                is_learned = True
            else:
                scores = similarity_matrix[i]
                best_idx = np.argmax(scores)
                best_score = scores[best_idx]
        else:
            scores = similarity_matrix[i]
            best_idx = np.argmax(scores)
            best_score = scores[best_idx]

        if best_score >= threshold or is_learned:
            status = "🧠 FL Learned Match" if is_learned else "✅ AI Auto-Match"
            matched_name = inventory_names[best_idx]
            conf_score = f"{round(best_score * 100, 1)}%"
            
            req_qty = 1
            if "Quantity" in design_bom_df.columns:
                try: req_qty = float(design_bom_df.iloc[i]["Quantity"])
                except: req_qty = 1
            
            unit_cost = 0.0
            if inv_cost_col:
                try: unit_cost = float(inventory_df.iloc[best_idx][inv_cost_col])
                except: unit_cost = 0.0
            
            stock_avail = 0
            if inv_stock_col:
                try: stock_avail = float(inventory_df.iloc[best_idx][inv_stock_col])
                except: stock_avail = 0

            total_cost = req_qty * unit_cost
            
            if stock_avail >= req_qty:
                stock_msg = "📦 In Stock"
            else:
                shortage = req_qty - stock_avail
                stock_msg = f"⚠️ Shortage (-{int(shortage)})"

        else:
            status = "❌ No Match Found"
            matched_name = "N/A"
            conf_score = f"{round(best_score * 100, 1)}%"
            req_qty = "-"
            unit_cost = "-"
            total_cost = "-"
            stock_msg = "❓ Unknown"

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
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        st.subheader("💰 Cost Analysis by Category")
        cost_data = pd.DataFrame({
            "Category": ["Electronics", "Mechanical", "Fasteners", "Packaging"],
            "Cost ($)": [15000, 8000, 2000, 1500]
        })
        fig_bar = px.bar(cost_data, x="Category", y="Cost ($)", 
                         title="Procurement Cost Breakdown",
                         color="Cost ($)", color_continuous_scale="Viridis")
        st.plotly_chart(fig_bar, use_container_width=True)

    # --- Recent Activity Log ---
    st.subheader("📝 Recent System Activity")
    st.dataframe(pd.DataFrame({
        "Timestamp": ["10:05 AM", "10:12 AM", "10:45 AM"],
        "User": ["Factory_Admin_IN", "System_AI", "Factory_Admin_DE"],
        "Action": ["Uploaded eBOM_v2.csv", "Auto-Matched 85 parts", "Flagged 'False Negative'"]
    }), use_container_width=True)

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
                try:
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
                except Exception as e:
                    st.error(f"⚠️ Error during processing: {e}")

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
            
            st.dataframe(pd.DataFrame(table_data), use_container_width=True)
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