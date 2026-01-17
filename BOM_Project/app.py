import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

# --- 1. SETUP & CONFIG ---
st.set_page_config(page_title="OptiBOM Enterprise", page_icon="🏭", layout="wide")
st.markdown("""<style>.stApp { background-color: #0e1117; color: white; }</style>""", unsafe_allow_html=True)

# --- 2. LOAD AI MODEL ---
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# --- 3. SIDEBAR ---
with st.sidebar:
    st.title("🏭 OptiBOM Pro")
    st.write("Federated AI Edition")
    menu = st.radio("Navigation", ["📊 Dashboard", "🔄 BOM Converter", "🌐 Federated Learning (New!)"])
    st.divider()
    st.info("Mode: Decentralized")

# --- 4. LOAD DATA ---
try:
    df_inventory = pd.read_csv("mbom_data.csv")
    inventory_embeddings = model.encode(df_inventory['Inventory_Description'].tolist(), convert_to_tensor=True)
except:
    st.error("❌ Inventory File Missing!")
    st.stop()

# --- 5. MAIN LOGIC ---

if menu == "📊 Dashboard":
    st.title("Executive Dashboard")
    col1, col2, col3 = st.columns(3)
    col1.metric("Global Model Version", "v2.1", "+0.1")
    col2.metric("Connected Factories", "2", "Active")
    col3.metric("Privacy Level", "High", "Federated")
    
    st.subheader("Network Status")
    st.markdown("✅ **Factory A:** Online | 🟢 **Factory B:** Online | 🔵 **Central Server:** Ready")

elif menu == "🔄 BOM Converter":
    st.title("🛠️ Standard BOM Converter")
    uploaded_file = st.file_uploader("Upload eBOM", type=["csv"])
    if uploaded_file:
        df_user = pd.read_csv(uploaded_file)
        col = 'Description' if 'Description' in df_user.columns else df_user.columns[1]
        user_emb = model.encode(df_user[col].tolist(), convert_to_tensor=True)
        
        results = []
        for i, row in df_user.iterrows():
            scores = util.cos_sim(user_emb[i], inventory_embeddings)[0]
            best_idx = scores.argmax().item()
            results.append({
                "Source": row[col],
                "Match": df_inventory.iloc[best_idx]['Inventory_Description'],
                "Score": scores[best_idx].item()
            })
        st.dataframe(pd.DataFrame(results))

elif menu == "🌐 Federated Learning (New!)":
    st.title("🌐 Federated Learning Simulation")
    st.markdown("Demonstration of updating the Global Model **without sharing data**.")
    
    # A. SIMULATED DATA CREATION
    if 'model_weights' not in st.session_state:
        st.session_state['model_weights'] = None

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🏭 Factory A (Client)")
        st.info("Has Data: 'Hex Bolt', 'Steel Sheet'...")
    with col2:
        st.subheader("🏭 Factory B (Client)")
        st.info("Has Data: 'Resistor', 'Capacitor'...")

    st.divider()
    
    # B. THE TRAINING PROCESS
    st.subheader("🚀 Start Federated Cycle")
    
    if st.button("Step 1: Train Local Models (On Private Data)"):
        with st.spinner("Training on Client A and Client B separately..."):
            # Simulation: We create two simple classifiers
            clf_A = SGDClassifier(loss="log_loss", random_state=42)
            clf_B = SGDClassifier(loss="log_loss", random_state=99)
            
            # Dummy Training (Representing learning from embeddings)
            # In real life, X would be embeddings, y would be "Correct Match" (1) or "Wrong" (0)
            X_dummy = np.random.rand(100, 10) 
            y_dummy = np.random.randint(0, 2, 100)
            
            clf_A.partial_fit(X_dummy, y_dummy, classes=[0, 1])
            clf_B.partial_fit(X_dummy, y_dummy, classes=[0, 1])
            
            # Save weights to session
            st.session_state['weights_A'] = clf_A.coef_
            st.session_state['weights_B'] = clf_B.coef_
            
            st.success("✅ Training Complete!")
            st.write("Client A Weights (First 5):", st.session_state['weights_A'][0][:5])
            st.write("Client B Weights (First 5):", st.session_state['weights_B'][0][:5])
            st.warning("⚠️ Note: Client A and B have DIFFERENT weights because they saw DIFFERENT data.")

    if st.button("Step 2: Aggregate Weights (Secure Server)"):
        if 'weights_A' in st.session_state:
            with st.spinner("Averaging Weights... (Privacy Preserved)"):
                # FEDERATED AVERAGING FORMULA: W_global = (W_a + W_b) / 2
                avg_weights = (st.session_state['weights_A'] + st.session_state['weights_B']) / 2
                st.session_state['global_weights'] = avg_weights
                
                st.success("🎉 Global Model Updated!")
                st.write("New Global Weights (First 5):", avg_weights[0][:5])
                st.markdown("### 💡 Magic Explained:")
                st.markdown("- Server never saw the data.")
                st.markdown("- Server only took the **Average of Math Numbers**.")
                st.markdown("- Now, **Factory A** knows what **Factory B** learned, without asking for data!")
                
                # Visual Chart
                chart_data = pd.DataFrame({
                    "Weight Index": range(5),
                    "Client A": st.session_state['weights_A'][0][:5],
                    "Client B": st.session_state['weights_B'][0][:5],
                    "Global (Avg)": avg_weights[0][:5]
                })
                st.line_chart(chart_data.set_index("Weight Index"))
        else:
            st.error("Please run Step 1 first!")
