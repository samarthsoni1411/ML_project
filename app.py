# app.py
import streamlit as st
from PIL import Image
import os

st.set_page_config(page_title="FairCheck", layout="wide", page_icon="🧪")

# Theme toggle stored in session_state
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = False

def toggle_theme():
    st.session_state["dark_mode"] = not st.session_state["dark_mode"]

# Sidebar - navigation hints
with st.sidebar:
    st.title("FairCheck")
    st.markdown("A bias detection & mitigation pipeline\nUpload → Clean → Data Bias → Train → Model Bias → Report")
    st.button("Toggle Dark/Light", on_click=toggle_theme)
    st.markdown("---")
    st.markdown("**Quick Links**")
    st.markdown("- 1: Data Upload\n- 2: Data Cleaning\n- 3: Data Bias\n- 4: Train Models\n- 5: Model Bias\n- 6: Final Report")
    st.markdown("---")
    st.caption("FairCheck • v2.0")

# Simple CSS for dark mode
if st.session_state["dark_mode"]:
    st.markdown(
        """
        <style>
        .stApp { background-color: #0f1724; color: #e6eef8; }
        .css-1d391kg { color: #e6eef8; }
        </style>
        """, unsafe_allow_html=True
    )

st.title("🧪 FairCheck — Dataset & Model-Level Bias Detection")
st.markdown("**Flow:** Upload → Clean → Data Bias → Train (KNN & KMeans) → Model Bias Comparison → Export")

# show pipeline image if exists
img_path = "assets/faircheck_pipeline.png"
if os.path.exists(img_path):
    st.image(Image.open(img_path), use_column_width=True)
else:
    st.info("Place a pipeline image at `assets/faircheck_pipeline.png` for better visuals.")
