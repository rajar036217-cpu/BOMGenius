# 🏭 BOMGenius: AI-Powered Smart BOM Converter

**An Enterprise-grade solution for Automated Engineering BOM (eBOM) to Manufacturing BOM (mBOM) Conversion.**
*Developed for L&T TECHgium POC 2026*

---

## 🚀 Key Features

### 1. AI-Driven Semantic Matching (BERT)
Unlike standard keyword matching, BOMGenius uses **Sentence-BERT (SBERT)** to understand the "meaning" of part descriptions.
* **Result:** Matches parts even if the naming conventions differ (e.g., "Steel Bolt" vs "HEX-STL-BLT").

### 2. Feature A: Advanced Business Intelligence 💰
Integrated a cost and inventory analysis engine into the conversion pipeline.
* **Cost Calculation:** Automatically calculates Total Cost based on required quantity and unit price.
* **Stock Feasibility:** Real-time stock checking with "Shortage" alerts to prevent production delays.

### 3. Feature B: Dual-Client Federated Learning 🌐
Simulates a secure, decentralized AI training network across global manufacturing sites.
* **Decentralized Training:** Factories in India and Germany contribute feedback without sharing their private CSV data.
* **Global Model Aggregation:** A Central Server updates the global matching threshold based on local feedback, improving accuracy for everyone.

---

## 🛠️ Technology Stack
* **Language:** Python
* **Framework:** Streamlit (For Interactive Dashboard)
* **AI Models:** Sentence-Transformers (all-MiniLM-L6-v2)
* **Version Control:** Git & GitHub (Modular Workflow)

---

## 📊 Distributed Architecture
The project simulates a hub-and-spoke model where:
1.  **Nodes:** Regional Factories (India/Germany) provide local feedback.
2.  **Hub:** Central HQ aggregates intelligence to optimize the Global AI Engine.

---

## 🏃 How to Run
1. Clone the repo: `git clone <YOUR_REPO_LINK>`
2. Install dependencies: `pip install streamlit sentence-transformers pandas scikit-learn`
3. Launch App: `streamlit run app.py`