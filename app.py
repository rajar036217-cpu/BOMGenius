# app.py
import streamlit as st
import requests

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="BOMGenius", layout="wide")
st.title("BOMGenius – eBOM → mBOM Converter (API Mode)")

ebom_file = st.file_uploader("Upload EBOM CSV", type=["csv"])
inv_file = st.file_uploader("Upload Inventory CSV", type=["csv"])

if ebom_file and inv_file:
    if st.button("Generate MBOM via API"):
        with st.spinner("Calling BOMGenius API..."):
            files = {
                "ebom": ebom_file,
                "inventory": inv_file
            }
            resp = requests.post(f"{API_BASE}/api/bom/convert", files=files)
            data = resp.json()

            st.success(f"MBOM generated: {data['rows']} rows")
            st.markdown(f"[Download MBOM CSV]({API_BASE}{data['download']})")
