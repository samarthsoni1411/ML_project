# pages/5_⚖️_Model_Bias.py
import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from utils.ui_helpers import load_css
load_css()
import streamlit as st
from utils.bias_metrics import demographic_parity, equal_opportunity, predictive_parity
from utils.visualizations import radar_plot
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Model Bias Comparison", layout="wide")
st.title("⚖️ Step 5 — Model-Level Bias Comparison (KNN vs KMeans)")

# -------------------------------
# Validate Session State
# -------------------------------
req = ["models", "X_test", "y_test"]
if not all(k in st.session_state for k in req):
    st.warning("⚠️ Train models first (Step 4 — Train Models).")
    st.stop()

models = st.session_state["models"]
knn = models["knn"]
kmeans = models["kmeans"]
X_test = st.session_state["X_test"]
y_test = st.session_state["y_test"]

# -------------------------------
# Select Sensitive Attribute
# -------------------------------
st.markdown("### 🎯 Choose Sensitive Attribute")

raw = st.session_state.get("raw_df", None)
if raw is not None:
    sensitive_col = st.selectbox("Sensitive Feature", ["(none)"] + list(raw.columns))
else:
    sensitive_col = st.selectbox("Sensitive Feature", ["(none)"])

if sensitive_col != "(none)" and raw is not None:
    sens = raw.loc[X_test.index, sensitive_col].fillna("Unknown")
    sens_enc = LabelEncoder().fit_transform(sens)
else:
    st.info("⚠️ No sensitive column chosen — generating random binary groups for demonstration.")
    sens_enc = np.random.randint(0, 2, size=len(y_test))

# -------------------------------
# Model Predictions
# -------------------------------
try:
    y_knn = knn.predict(X_test)
    y_kmeans = kmeans.predict(X_test)
except Exception as e:
    st.error(f"❌ Prediction failed: {e}")
    st.stop()

# -------------------------------
# Fairness Metric Calculations
# -------------------------------
def safe_metric(func, *args):
    try:
        return float(func(*args))
    except Exception:
        return np.nan

dp_knn = safe_metric(demographic_parity, y_knn, sens_enc)
eo_knn = safe_metric(equal_opportunity, y_test, y_knn, sens_enc)
pp_knn = safe_metric(predictive_parity, y_test, y_knn, sens_enc)

dp_km = safe_metric(demographic_parity, y_kmeans, sens_enc)
eo_km = safe_metric(equal_opportunity, y_test, y_kmeans, sens_enc)
pp_km = safe_metric(predictive_parity, y_test, y_kmeans, sens_enc)

metrics = ["Demographic Parity", "Equal Opportunity", "Predictive Parity"]
knn_vals = [dp_knn, eo_knn, pp_knn]
kmeans_vals = [dp_km, eo_km, pp_km]

# -------------------------------
# Visualization: Radar + Bar
# -------------------------------
st.markdown("### 📊 Fairness Metric Comparison")

fig_radar = radar_plot(metrics, knn_vals, name="KNN")
fig_radar.add_trace(radar_plot(metrics, kmeans_vals, name="KMeans").data[0])
st.plotly_chart(fig_radar, use_container_width=True)

bias_df = pd.DataFrame({
    "Metric": metrics,
    "KNN": knn_vals,
    "KMeans": kmeans_vals
}).set_index("Metric")

# Bias intensity bar plot
bar_fig = px.bar(
    bias_df.reset_index().melt(id_vars="Metric", var_name="Model", value_name="Score"),
    x="Metric", y="Score", color="Model", barmode="group",
    color_discrete_sequence=["#4CAF50", "#FF9800"],
    title="Bias Metric Intensity Comparison"
)
st.plotly_chart(bar_fig, use_container_width=True)

# -------------------------------
# KNN Performance
# -------------------------------
# -------------------------------
# Dynamic Confusion Matrix Display
# -------------------------------
cm = confusion_matrix(y_test, y_knn)
unique_labels = np.unique(y_test)
cm_df = pd.DataFrame(cm,
                     index=[f"True {l}" for l in unique_labels],
                     columns=[f"Pred {l}" for l in unique_labels])

st.subheader("🧾 KNN Confusion Matrix")
st.dataframe(cm_df, use_container_width=True)

st.write("**Precision:**", round(precision_score(y_test, y_knn, zero_division=0), 3))
st.write("**Recall:**", round(recall_score(y_test, y_knn, zero_division=0), 3))

# -------------------------------
# KMeans Cluster Mapping
# -------------------------------
st.subheader("🧩 KMeans Cluster vs True Label Mapping")
contingency = pd.crosstab(y_kmeans, y_test)
st.dataframe(contingency)

# -------------------------------
# Export Safe Report
# -------------------------------
st.markdown("### 📥 Download Bias Comparison Report")

def safe_format(v):
    if isinstance(v, (int, float)):
        return f"{v:.3f}"
    elif pd.isna(v):
        return ""
    return str(v)

export_df = bias_df.copy().reset_index()
for col in export_df.columns:
    export_df[col] = export_df[col].apply(safe_format)

st.download_button(
    label="📄 Download Fairness Comparison CSV",
    data=export_df.to_csv(index=False),
    file_name="faircheck_model_bias_comparison.csv",
    mime="text/csv"
)

# -------------------------------
# Save for Final Report
# -------------------------------
st.session_state["model_bias_comparison"] = {
    "metrics": {"knn": dict(zip(metrics, knn_vals)), "kmeans": dict(zip(metrics, kmeans_vals))},
    "contingency": contingency.to_dict(),
}
st.success("✅ Model bias comparison stored for Final Report generation.")
