# pages/2_🧼_Data_Cleaning.py
import streamlit as st
from utils.preprocessing import drop_high_missing, simple_impute
import pandas as pd

st.set_page_config(page_title="Data Cleaning", layout="wide")
st.title("🧼 Step 2 — Data Cleaning & Imputation")

if "raw_df" not in st.session_state:
    st.warning("Upload dataset in Data Upload page first.")
    st.stop()

raw = st.session_state["raw_df"].copy()
st.subheader("Preview (Raw)")
st.dataframe(raw.head(), use_container_width=True)

threshold = st.slider("Drop columns with missing ratio >", 0.0, 0.9, 0.4, 0.05)
impute_strategy = st.selectbox("Imputation strategy", ["most_frequent", "median", "mean"])

if st.button("Run cleaning"):
    with st.spinner("Cleaning and imputing — hang tight..."):
        df1, dropped = drop_high_missing(raw, threshold)
        df2 = simple_impute(df1, strategy=impute_strategy)
        st.session_state["clean_df"] = df2
        st.session_state["dropped_cols"] = dropped
        st.success(f"Cleaned dataset saved (shape: {df2.shape}). Dropped: {dropped}")
        st.dataframe(df2.head(), use_container_width=True)

if "clean_df" in st.session_state:
    st.info("Cleaned data available — go to Data Bias page or re-run cleaning with different params.")
