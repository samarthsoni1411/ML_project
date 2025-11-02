# app.py
import streamlit as st
from PIL import Image
from utils.ui_helpers import load_css
import os

st.set_page_config(page_title="FairCheck", layout="wide", page_icon="🧪")

# Load global CSS
load_css()

# ==============================
# Custom CSS (Sidebar + Responsive Workflow Image)
# ==============================
st.markdown("""
<style>
/* --- Sidebar styling --- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #001f3f, #002b5b, #004b8d);
    color: white;
    padding: 1rem;
    border-right: 2px solid rgba(255,255,255,0.1);
}

section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, 
section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] p {
    color: #e3f2fd !important;
}

/* Reorder page buttons */
div[data-testid="stSidebarNav"] ul {
    list-style: none;
    counter-reset: section;
    padding-left: 0;
}

div[data-testid="stSidebarNav"] ul li {
    margin-bottom: 0.5rem;
}

div[data-testid="stSidebarNav"] ul li a {
    color: #dbeafe !important;
    font-weight: 500;
    font-size: 1rem;
    padding: 0.4rem 0.8rem;
    display: block;
    border-radius: 8px;
    transition: all 0.3s ease;
}

div[data-testid="stSidebarNav"] ul li a:hover {
    background-color: rgba(255,255,255,0.1);
    color: white !important;
    transform: translateX(4px);
}

/* Title & version */
.sidebar-title {
    font-size: 1.3rem;
    font-weight: 700;
    text-align: center;
    color: white;
    margin-bottom: 1rem;
}

.sidebar-version {
    text-align: center;
    font-size: 0.8rem;
    color: #a7c7e7;
    margin-top: 1rem;
}

/* --- Workflow Image Styling --- */
.faircheck-pipeline {
    display: flex;
    justify-content: center;
    align-items: center;
    margin: 2rem auto;
    max-width: 900px;
}

.faircheck-pipeline img {
    width: 100%;
    height: auto;
    max-height: 550px;
    object-fit: contain;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    transition: all 0.3s ease-in-out;
}

/* Hover glow effect */
.faircheck-pipeline img:hover {
    transform: scale(1.02);
    box-shadow: 0 6px 20px rgba(0, 153, 255, 0.3);
}

/* --- Responsive scaling --- */
@media (max-width: 1024px) {
    .faircheck-pipeline {
        max-width: 750px;
    }
}

@media (max-width: 768px) {
    .faircheck-pipeline {
        max-width: 90%;
    }
    .faircheck-pipeline img {
        max-height: 450px;
    }
}

@media (max-width: 480px) {
    .faircheck-pipeline img {
        max-height: 350px;
    }
}
</style>
""", unsafe_allow_html=True)

# ==============================
# Sidebar Layout
# ==============================
st.sidebar.markdown("<p class='sidebar-title'>⚙️ FairCheck Navigation</p>", unsafe_allow_html=True)
st.sidebar.info("Navigate using the sidebar menu in proper workflow order ↓")

st.sidebar.markdown("""
1️⃣ **Data Upload**  
2️⃣ **Data Cleaning**  
3️⃣ **Data Bias**  
4️⃣ **Train Models**  
5️⃣ **Model Bias**  
6️⃣ **Final Report**
""")

st.sidebar.markdown("<p class='sidebar-version'>FairCheck v2.0 — Automated Bias Detection</p>", unsafe_allow_html=True)

# ==============================
# MAIN LANDING PAGE
# ==============================
st.title("🧪 FairCheck — AI Bias Detection Framework")

st.markdown("""
Welcome to **FairCheck**, a step-by-step bias detection and analysis framework that helps you:
- Detect **dataset-level** bias (SPD, DI, MI)
- Train & compare **KNN vs KMeans**
- Evaluate **model-level** bias (DP, EO, PP)
- Export a clean **Final Report**
""")

# Display responsive centered image
img_path = "assets/faircheck_pipeline.png"
if os.path.exists(img_path):
    st.markdown("<div class='faircheck-pipeline'>", unsafe_allow_html=True)
    st.image(Image.open(img_path), use_container_width=True, caption="FairCheck Workflow")
    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("Place a pipeline image at `assets/faircheck_pipeline.png` for better visuals.")

st.markdown("""
### 🚀 Steps Overview
1️⃣ Upload your dataset  
2️⃣ Clean and preprocess data  
3️⃣ Detect dataset-level bias  
4️⃣ Train KNN & KMeans models  
5️⃣ Analyze model-level bias  
6️⃣ Generate PDF/CSV report
""")

st.success("✅ Use the left sidebar to navigate step by step in order.")
