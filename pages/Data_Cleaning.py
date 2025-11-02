# pages/2_🧼_Data_Cleaning.py
import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.ui_helpers import load_css
load_css()

import streamlit as st
from utils.preprocessing import drop_high_missing, simple_impute
import pandas as pd
import numpy as np

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Data Cleaning", layout="wide")
st.title("🧼 Step 2 — Data Cleaning & Imputation")

# -------------------------------
# Session Check
# -------------------------------
if "raw_df" not in st.session_state:
    st.warning("⚠️ Upload dataset in **Data Upload** page first.")
    st.stop()

raw = st.session_state["raw_df"].copy()

# -------------------------------
# Preview
# -------------------------------
st.subheader("📋 Preview (Raw Data)")
st.dataframe(raw.head(), use_container_width=True)

# -------------------------------
# Cleaning Controls
# -------------------------------
threshold = st.slider("📉 Drop columns with missing ratio >", 0.0, 0.9, 0.4, 0.05)
impute_strategy = st.selectbox("🧩 Imputation strategy", ["most_frequent", "median", "mean"])

# -------------------------------
# Cleaning Logic
# -------------------------------
if st.button("🚿 Run Cleaning"):
    with st.spinner("🧠 Cleaning and imputing — please wait..."):
        try:
            # Drop columns with too many NaNs
            df1, dropped = drop_high_missing(raw, threshold)

            # Impute missing values (auto-handles mixed types)
            df2 = simple_impute(df1, strategy=impute_strategy)

            # Save cleaned data in session
            st.session_state["clean_df"] = df2
            st.session_state["dropped_cols"] = dropped

            # -------------------------------
            # Generate Summary
            # -------------------------------
            total_missing_before = raw.isna().sum().sum()
            total_missing_after = df2.isna().sum().sum()
            num_cols = df2.select_dtypes(include=[np.number]).shape[1]
            cat_cols = df2.select_dtypes(exclude=[np.number]).shape[1]

            st.success(f"✅ Cleaned dataset saved successfully!")
            st.info(
                f"""
                **Summary Report:**
                - Original shape: `{raw.shape}`
                - Cleaned shape: `{df2.shape}`
                - Dropped columns: `{dropped if dropped else 'None'}`
                - Missing values before: `{total_missing_before}`
                - Missing values after: `{total_missing_after}`
                - Numeric columns imputed: `{num_cols}`
                - Categorical columns imputed: `{cat_cols}`
                """
            )

            st.dataframe(df2.head(), use_container_width=True)
        except Exception as e:
            st.error(f"❌ Cleaning failed: {e}")

# -------------------------------
# Final Info
# -------------------------------
if "clean_df" in st.session_state:
    st.info("✅ Cleaned data available — proceed to **Data Bias** page or re-run cleaning with new parameters.")
