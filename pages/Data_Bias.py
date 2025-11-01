# pages/3_📊_Data_Bias.py
import streamlit as st
from utils.bias_metrics import calc_spd_binary, calc_di_binary, calc_mutual_info
from utils.visualizations import bias_heatmap
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Data Bias Detection", layout="wide")
st.title("📊 Step 3 — Data-Level Bias Detection")

# -------------------------------
# Load cleaned data
# -------------------------------
if "clean_df" not in st.session_state:
    st.warning("⚠️ Please clean and preprocess data first (Step 2 — Clean Data).")
    st.stop()

df = st.session_state["clean_df"].copy()

# -------------------------------
# Target selection
# -------------------------------
st.markdown("### 🎯 Select Target Column (Optional)")
target_col = st.selectbox("Target Column", ["(none)"] + list(df.columns))

if target_col != "(none)":
    y = df[target_col]
    # Auto-encode target if needed
    if y.dtype == "object" or y.dtype.name == "category":
        le = LabelEncoder()
        df[target_col] = le.fit_transform(y.fillna("Unknown"))
        st.info(f"Target **{target_col}** encoded automatically as numeric labels.")
    st.session_state["target_col"] = target_col
else:
    st.info("No target selected — will compute only Mutual Information.")
    target_col = None

# -------------------------------
# Compute Bias Metrics
# -------------------------------
if st.button("🚀 Compute Bias Metrics"):
    with st.spinner("Analyzing bias across features …"):
        results = []
        for col in df.columns:
            if col == target_col:
                continue
            try:
                spd = calc_spd_binary(df, col, target_col) if target_col else np.nan
                di = calc_di_binary(df, col, target_col) if target_col else np.nan
                mi = calc_mutual_info(df, col, target_col)
            except Exception:
                spd, di, mi = np.nan, np.nan, np.nan
            results.append({"Feature": col, "SPD": spd, "DI": di, "Mutual Info": mi})

        bias_df = pd.DataFrame(results)
        bias_df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Flag high-bias features
        bias_df["Bias Flag"] = np.where(
            (bias_df["SPD"].abs() > 0.2) | (bias_df["DI"] < 0.8),
            "⚠️ High Bias",
            "✅ Fair"
        )

        st.session_state["bias_df"] = bias_df
        st.success("✅ Bias metrics calculated successfully.")
        st.write(f"**Analyzed {len(bias_df)} features**")

        # --- Styled Table ---
        st.dataframe(
            bias_df.style.background_gradient(
                subset=["SPD", "DI", "Mutual Info"],
                cmap="RdYlGn_r"
            ),
            use_container_width=True,
            height=400
        )

        # --- Interactive Bar Plot ---
        melted = bias_df.melt(
            id_vars=["Feature"], 
            value_vars=["SPD", "DI", "Mutual Info"],
            var_name="Metric", value_name="Value"
        ).dropna()
        fig = px.bar(
            melted, x="Feature", y="Value", color="Metric",
            barmode="group", height=500,
            title="📊 Bias Metrics per Feature"
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- Heatmap ---
        try:
            st.markdown("### 🔥 Bias Correlation Heatmap")
            heatmap_fig = bias_heatmap(bias_df)
            st.plotly_chart(heatmap_fig, use_container_width=True)
        except Exception:
            st.warning("⚠️ Heatmap could not be generated for this dataset.")

        # --- Export Results ---
        csv_data = bias_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Bias Report (CSV)",
            data=csv_data,
            file_name="data_bias_report.csv",
            mime="text/csv"
        )

        # Save summary for report
        st.session_state["data_bias_summary"] = {
            "n_features": len(bias_df),
            "high_bias_features": bias_df[bias_df["Bias Flag"] == "⚠️ High Bias"]["Feature"].tolist(),
        }
