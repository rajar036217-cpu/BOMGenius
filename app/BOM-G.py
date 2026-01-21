import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.express as px
import json
import io
from PIL import Image
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------
# 1. SETUP & CONFIGURATION (Failsafe Sidebar)
# --------------------------------------------------

def configure_application_ui():
    # initial_sidebar_state="expanded" forces the menu to stay open
    st.set_page_config(
        page_title="BOMGenius 2.0",
        page_icon="🏭",
        layout="wide",
        initial_sidebar_state="expanded" 
    )
    
    # Clean CSS: Only changes colors, does NOT hide structural elements
    st.markdown(
        """
        <style>
        /* Main Background */
        .stApp {
            background-color: #0e1117;
            color: #FFFFFF;
        }

        /* SIDEBAR VISIBILITY FIX */
        [data-testid="stSidebar"] {
            background-color: #11141a !important;
            border-right: 1px solid #30363d;
            min-width: 250px;
        }

        /* Ensure Sidebar Text is White and Large */
        [data-testid="stSidebar"] .stText, 
        [data-testid="stSidebar"] label, 
        [data-testid="stSidebar"] p {
            color: #FFFFFF !important;
            font-size: 1.1rem !important;
        }

        /* Headers */
        h1, h2, h3 {
            color: #FFBF00 !important;
        }

        /* Metric Cards */
        [data-testid="stMetric"] {
            background-color: #1c1f26;
            border-radius: 10px;
            padding: 15px;
            border-left: 5px solid #FF5F1F;
        }

        /* Buttons */
        .stButton>button {
            width: 100%;
            border-radius: 8px;
            border: 1px solid #FFBF00;
            color: #FFFFFF;
            background-color: transparent;
        }
        .stButton>button:hover {
            background-color: #FFBF00;
            color: #0e1117;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# --------------------------------------------------
# 2. LOAD AI MODEL (Cached)
# --------------------------------------------------

@st.cache_resource
def load_semantic_engine():
    return SentenceTransformer("all-MiniLM-L6-v2")

# --------------------------------------------------
# 3. MULTIMODAL INGESTION ENGINE
# --------------------------------------------------

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
    with st.spinner(f"Processing {file_ext.upper()}..."):
        try:
            if file_ext == 'csv':
                df = pd.read_csv(uploaded_file)
            elif file_ext in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file)
            elif file_ext == 'json':
                df = pd.json_normalize(json.load(uploaded_file))
            elif file_ext in ['png', 'jpg', 'jpeg']:
                st.image(Image.open(uploaded_file), width=300)
                df = extract_from_cad_image(uploaded_file)
            
            # Normalization
            rename_map = {}
            for col in df.columns:
                c_low = col.lower()
                if any(x in c_low for x in ["desc", "name", "item"]): rename_map[col] = "Description"
                elif any(x in c_low for x in ["qty", "quantity"]): rename_map[col] = "Quantity"
            df = df.rename(columns=rename_map)
            return df[["Description", "Quantity"]]
        except Exception as e:
            st.error(f"Error: {e}")
            return None

# --------------------------------------------------
# 4. CORE AI MATCHING ENGINE
# --------------------------------------------------

def compute_semantic_matches(design_bom_df, inventory_df, semantic_engine, threshold):
    inv_desc_col = inventory_df.columns[0]
    inv_cost_col, inv_stock_col = None, None
    for col in inventory_df.columns:
        c_low = col.lower()
        if "desc" in c_low or "item" in c_low: inv_desc_col = col
        elif "cost" in c_low or "price" in c_low: inv_cost_col = col
        elif "stock" in c_low or "qty" in c_low: inv_stock_col = col

    design_names = design_bom_df["Description"].astype(str).tolist()
    inventory_names = inventory_df[inv_desc_col].astype(str).tolist()

    d_emb = semantic_engine.encode(design_names)
    i_emb = semantic_engine.encode(inventory_names)
    sim = cosine_similarity(d_emb, i_emb)

    results = []
    kb = st.session_state.get("global_knowledge_base", {})

    for i in range(len(design_names)):
        d_name = design_names[i]
        if d_name in kb:
            matched_name = kb[d_name]["target"]
            score = 1.0
            status = "🧠 Learned"
        else:
            idx = np.argmax(sim[i])
            score = sim[i][idx]
            matched_name = inventory_names[idx]
            status = "✅ AI Match" if score >= threshold else "❌ No Match"

        results.append({
            "eBOM Item": d_name,
            "Qty": design_bom_df.iloc[i]["Quantity"],
            "mBOM Match": matched_name if status != "❌ No Match" else "N/A",
            "Confidence": f"{int(score*100)}%",
            "Status": status
        })
    return pd.DataFrame(results)

# --------------------------------------------------
# 5. PAGE RENDERING
# --------------------------------------------------

def render_dashboard():
    st.header("📊 Executive Dashboard")
    c1, c2, c3 = st.columns(3)
    c1.metric("System Accuracy", "94%", "+2%")
    c2.metric("Connected Nodes", "2", "Stable")
    c3.metric("Savings", "$14.2k", "Live")
    
    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.pie(names=["In Stock", "Out"], values=[80, 20], title="Inventory Status", color_discrete_sequence=px.colors.sequential.YlOrBr)
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        fig2 = px.bar(x=["Elec", "Mech", "Other"], y=[1500, 800, 400], title="Costs", color_discrete_sequence=["#FFBF00"])
        st.plotly_chart(fig2, use_container_width=True)

def render_converter(engine, threshold):
    st.header("🔄 BOM Converter")
    file_ebom = st.file_uploader("Upload eBOM (Image, Excel, JSON, CSV)", type=["csv", "xlsx", "json", "png", "jpg"])
    file_inv = st.file_uploader("Upload Factory Inventory (CSV)", type=["csv"])

    if file_ebom and file_inv:
        df_ebom = parse_multimodal_ebom(file_ebom)
        df_inv = pd.read_csv(file_inv)
        if df_ebom is not None:
            st.write("### Extracted Data")
            st.dataframe(df_ebom, use_container_width=True)
            if st.button("🚀 Match with Inventory"):
                res = compute_semantic_matches(df_ebom, df_inv, engine, threshold)
                st.write("### Results")
                st.data_editor(res, use_container_width=True)

def render_fl():
    st.header("🌐 Federated Learning")
    st.info("Simulate feedback from global factory nodes.")
    t1, t2 = st.tabs(["Factory Node", "Central HQ"])
    with t1:
        w = st.text_input("Mismatch Found (eBOM Name)")
        c = st.text_input("Correct Inventory Name")
        if st.button("Push Correction"):
            st.session_state["global_knowledge_base"][w] = {"target": c}
            st.success("Rule added to Knowledge Base!")
    with t2:
        st.write(st.session_state["global_knowledge_base"])

# --------------------------------------------------
# 6. MAIN NAVIGATION
# --------------------------------------------------

def main():
    configure_application_ui()
    
    # Initialize State
    if "global_knowledge_base" not in st.session_state:
        st.session_state["global_knowledge_base"] = {}

    engine = load_semantic_engine()

    # --- THE SIDEBAR ---
    with st.sidebar:
        st.title("🏭 BOMGenius")
        st.write("---")
        # This is the navigation menu
        choice = st.radio(
            "Go to Page:",
            ["Dashboard", "BOM Converter", "Federated Learning"],
            index=0
        )
        st.write("---")
        st.caption("POC v2.0 | Multimodal")

    # Routing
    if choice == "Dashboard":
        render_dashboard()
    elif choice == "BOM Converter":
        render_converter(engine, 0.65)
    elif choice == "Federated Learning":
        render_fl()

if __name__ == "__main__":
    main()
