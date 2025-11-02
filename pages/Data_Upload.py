# pages/1_📂_Data_Upload.py
import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from utils.ui_helpers import load_css
load_css()

import streamlit as st
from utils.preprocessing import safe_read_df

st.set_page_config(page_title="Data Upload", layout="wide")
st.title("📂 Step 1 — Upload Dataset")

uploaded = st.file_uploader("Upload CSV or XLSX", type=["csv","xlsx"])
if uploaded:
    try:
        df = safe_read_df(uploaded)
        st.session_state["raw_df"] = df
        st.success(f"Uploaded {uploaded.name} (shape: {df.shape})")
        st.dataframe(df.head(), use_container_width=True)
        c1,c2,c3 = st.columns(3)
        c1.metric("Rows", df.shape[0])
        c2.metric("Cols", df.shape[1])
        c3.metric("Missing %", f"{(df.isna().mean().mean()*100):.2f}%")
    except Exception as e:
        st.error(f"Failed to read file: {e}")
else:
    st.info("Upload a dataset to begin the pipeline.")
