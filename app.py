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
    """The AI Logic: Matches eBOM to mBOM"""
    
    # 1. Get Descriptions
    # Assuming the user selects columns, but for now we auto-detect or use 2nd column
    des_col = design_bom_df.columns[1] # Guessing Description is 2nd column
    inv_col = inventory_df.columns[1] 

    # 2. Create Embeddings (Convert words to numbers)
    design_embeddings = semantic_engine.encode(design_bom_df[des_col].astype(str).tolist())
    inventory_embeddings = semantic_engine.encode(inventory_df[inv_col].astype(str).tolist())

    # 3. Calculate Similarity
    similarity_matrix = cosine_similarity(design_embeddings, inventory_embeddings)

    results = []

    for i, scores in enumerate(similarity_matrix):
        best_idx = np.argmax(scores)
        best_score = scores[best_idx]

        status = "✅ Auto-Match" if best_score >= threshold else "⚠️ Manual Review"

        # Create Result Row
        row = {
            "Design Part": design_bom_df.iloc[i][des_col],
            "Matched Inventory Part": inventory_df.iloc[best_idx][inv_col],
            "Confidence Score": f"{round(best_score * 100, 1)}%",
            "Status": status
        }
        
        # Add extra inventory details if available
        if "unit_cost" in inventory_df.columns:
            row["Unit Cost"] = inventory_df.iloc[best_idx]["unit_cost"]
            
        results.append(row)

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
    st.title("🌐 Federated Learning Simulation")
    
    if "history" not in st.session_state:
        st.session_state["history"] = [st.session_state["global_threshold"]]

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Factory Feedback")
        feedback = st.radio("Report Issue:", ["None", "False Positive (Wrong Match)", "False Negative (Missed Match)"])
        
        if st.button("📡 Send Feedback Update"):
            if feedback != "None":
                new_val = federated_learning_update(feedback, st.session_state["global_threshold"])
                st.session_state["global_threshold"] = new_val
                st.session_state["history"].append(new_val)
                st.success(f"Model Updated! New Threshold: {round(new_val, 2)}")
            else:
                st.warning("Select an issue to update.")

    with col2:
        st.subheader("Global Model Evolution")
        st.line_chart(st.session_state["history"])
        st.caption("Graph shows how the AI 'learns' from factory feedback over time.")

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
