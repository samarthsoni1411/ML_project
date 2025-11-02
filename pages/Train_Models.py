# pages/4_🤖_Train_Models.py
import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from utils.ui_helpers import load_css
load_css()
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils.preprocessing import encode_target_if_needed, simple_impute
from utils.model_trainers import train_knn, train_kmeans
import plotly.express as px

st.set_page_config(page_title="Train Models", layout="wide")
st.title("🤖 Step 4 — Train Models (KNN & KMeans)")

# -------------------------------
# Load prerequisites
# -------------------------------
if "clean_df" not in st.session_state or "target_col" not in st.session_state:
    st.warning("⚠️ Please complete Steps 2 & 3 (Data Cleaning → Data Bias) first.")
    st.stop()

df = st.session_state["clean_df"].copy()
target_col = st.session_state["target_col"]
st.markdown(f"**Target Column:** `{target_col}`")

# -------------------------------
# Prepare features and target
# -------------------------------
try:
    y = encode_target_if_needed(df, target_col)
    X = df.drop(columns=[target_col])
    X = pd.get_dummies(X, drop_first=True)
    X = simple_impute(X)

    # Handle potential scaling for distance-based models
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    st.success(f"✅ Data prepared — {X_scaled.shape[0]} rows × {X_scaled.shape[1]} features")
    st.dataframe(X_scaled.head(), use_container_width=True)
except Exception as e:
    st.error(f"❌ Data preparation failed: {e}")
    st.stop()

# -------------------------------
# User controls
# -------------------------------
st.sidebar.header("⚙️ Training Settings")
test_size = st.sidebar.slider("Test size (%)", 10, 50, 30, 5) / 100
n_neighbors = st.sidebar.number_input("KNN neighbors", 1, 20, 5)
st.sidebar.markdown("---")
st.sidebar.info("KNN = Supervised model • KMeans = Unsupervised clusters")

# -------------------------------
# Train
# -------------------------------
if st.button("🚀 Train KNN & KMeans Models"):
    with st.spinner("Training models …"):
        try:
            # Check target validity
            unique_classes = np.unique(y)
            if len(unique_classes) < 2:
                st.error("Target has only one class — cannot train supervised model.")
                st.stop()

            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, random_state=42,
                stratify=y if len(unique_classes) > 1 else None
            )

            # Train both models
            knn = train_knn(X_train, y_train, n_neighbors=int(n_neighbors))
            kmeans = train_kmeans(X_scaled, n_clusters=len(unique_classes))

            # Save to session
            st.session_state["models"] = {"knn": knn, "kmeans": kmeans}
            st.session_state["X_test"] = X_test
            st.session_state["y_test"] = y_test
            st.session_state["scaler"] = scaler

            st.success("✅ Models trained successfully and stored in session.")
            st.info("KNN trained on train split | KMeans trained on entire dataset")

            # -------------------------------
            # Visuals — Target distribution + KMeans clusters
            # -------------------------------
            fig1 = px.histogram(
                x=y, nbins=len(unique_classes),
                title="🎯 Target Class Distribution",
                labels={"x": "Class", "y": "Count"}, color_discrete_sequence=["#2ca02c"]
            )
            st.plotly_chart(fig1, use_container_width=True)

            cluster_labels = kmeans.predict(X_scaled)
            fig2 = px.scatter(
                x=X_scaled.iloc[:, 0], y=X_scaled.iloc[:, 1],
                color=cluster_labels.astype(str),
                title="🔹 KMeans Cluster Preview (First 2 Features)",
                labels={"x": X_scaled.columns[0], "y": X_scaled.columns[1]}
            )
            st.plotly_chart(fig2, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Training failed: {e}")
