# pages/6_Final_Report.py
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import plotly.graph_objects as go

st.set_page_config(page_title="Final Report", layout="wide")
st.title("📄 Step 6 — Final Report Generator")

# -------------------------------
# Load session data
# -------------------------------
bias_df = st.session_state.get("bias_df", None)
model_bias = st.session_state.get("model_bias_comparison", None)
clean_df = st.session_state.get("clean_df", None)
target_col = st.session_state.get("target_col", None)

if (bias_df is None or bias_df.empty) and (model_bias is None or model_bias == {}):
    st.warning("⚠️ No data found in session. Please run Data Bias and Model Bias steps first.")
    st.stop()

# -------------------------------
# Helper: Create styled table
# -------------------------------
def create_table(data, colnames):
    table = Table(data, colWidths=[150]*len(colnames))
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
    ]))
    return table

# -------------------------------
# Helper: Create radar chart image
# -------------------------------
def create_radar_image(metrics_data):
    metrics = list(metrics_data["knn"].keys())
    knn_vals = list(metrics_data["knn"].values())
    km_vals = list(metrics_data["kmeans"].values())

    # Create radar chart
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=knn_vals + [knn_vals[0]],
        theta=metrics + [metrics[0]],
        fill='toself',
        name='KNN',
        line=dict(color='royalblue')
    ))
    fig.add_trace(go.Scatterpolar(
        r=km_vals + [km_vals[0]],
        theta=metrics + [metrics[0]],
        fill='toself',
        name='KMeans',
        line=dict(color='darkorange')
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, max(knn_vals + km_vals) * 1.1])),
        showlegend=True,
        margin=dict(l=20, r=20, t=20, b=20),
        width=500,
        height=400,
        paper_bgcolor='white'
    )

    # Convert to image
    img_bytes = fig.to_image(format="png")
    return BytesIO(img_bytes)

# -------------------------------
# Generate PDF
# -------------------------------
def create_pdf(title, bias_df, model_bias, clean_df, target_col):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    content = []

    # --- Title ---
    content.append(Paragraph(title, styles['Title']))
    content.append(Spacer(1, 12))

    # --- Dataset Overview ---
    if clean_df is not None:
        content.append(Paragraph("📘 Dataset Overview", styles["Heading2"]))
        n_rows, n_cols = clean_df.shape
        missing_pct = clean_df.isna().mean().mean() * 100
        encoded_cols = sum(clean_df.dtypes == "uint8")
        summary_data = [
            ["Total Rows", n_rows],
            ["Total Columns", n_cols],
            ["Average Missing %", f"{missing_pct:.2f}%"],
            ["Encoded Columns", encoded_cols],
            ["Target Column", target_col if target_col else "N/A"],
        ]
        content.append(create_table([["Metric", "Value"]] + summary_data, ["Metric", "Value"]))
        content.append(Spacer(1, 12))

    # --- Data Bias Summary ---
    if bias_df is not None and not bias_df.empty:
        content.append(Paragraph("📊 Data-Level Bias Summary", styles["Heading2"]))
        bias_df = bias_df.copy()
        for c in ["SPD", "DI", "Mutual Info"]:
            bias_df[c] = pd.to_numeric(bias_df[c], errors="coerce")

        mean_spd = bias_df["SPD"].mean(skipna=True)
        mean_di = bias_df["DI"].mean(skipna=True)
        mean_mi = bias_df["Mutual Info"].mean(skipna=True)

        content.append(Paragraph(
            f"Average SPD: {mean_spd:.3f} | Average DI: {mean_di:.3f} | Average MI: {mean_mi:.3f}",
            styles["Normal"]
        ))
        content.append(Spacer(1, 8))

        # Top biased features
        top3 = bias_df.reindex(bias_df["SPD"].abs().sort_values(ascending=False).index).head(3)
        rows = [["Feature", "SPD", "DI", "Mutual Info"]] + [
            [r["Feature"], f"{r['SPD']:.3f}", f"{r['DI']:.3f}", f"{r['Mutual Info']:.3f}"]
            for _, r in top3.iterrows()
        ]
        content.append(create_table(rows, ["Feature", "SPD", "DI", "MI"]))
        content.append(Spacer(1, 15))

    # --- Model Bias Summary ---
    if model_bias:
        content.append(Paragraph("⚖️ Model-Level Bias Comparison", styles["Heading2"]))
        metrics = model_bias.get("metrics", {})
        rows = [["Metric", "KNN", "KMeans"]]
        knn_vals = metrics.get("knn", {})
        km_vals = metrics.get("kmeans", {})

        for metric in knn_vals.keys():
            k1 = knn_vals.get(metric, np.nan)
            k2 = km_vals.get(metric, np.nan)
            rows.append([metric, f"{k1:.3f}", f"{k2:.3f}"])
        content.append(create_table(rows, ["Metric", "KNN", "KMeans"]))
        content.append(Spacer(1, 8))

        # Embed radar chart
        try:
            radar_img = create_radar_image(metrics)
            img = Image(radar_img, width=400, height=300)
            content.append(img)
            content.append(Spacer(1, 12))
        except Exception as e:
            content.append(Paragraph(f"⚠️ Radar chart could not be added: {e}", styles["Normal"]))

        # Fairness summary
        avg_knn = np.nanmean(list(knn_vals.values()))
        avg_km = np.nanmean(list(km_vals.values()))
        better_model = "KNN" if avg_knn < avg_km else "KMeans"
        content.append(Paragraph(
            f"🏆 <b>{better_model}</b> shows lower average bias across metrics, indicating fairer performance.",
            styles["Normal"]
        ))
        content.append(Spacer(1, 15))

    # --- Future Suggestions ---
    content.append(Paragraph("💡 Future Suggestions", styles["Heading2"]))
    suggestions = [
        "✔ Rebalance dataset to reduce bias in dominant groups.",
        "✔ Apply reweighting or resampling for underrepresented classes.",
        "✔ Try mitigation methods like Adversarial Debiasing or FairSampling.",
        "✔ Regularize or drop highly biased features.",
        "✔ Compare fairness metrics post-mitigation to verify improvements."
    ]
    for s in suggestions:
        content.append(Paragraph(s, styles["Normal"]))
    content.append(Spacer(1, 20))

    content.append(Paragraph("<i>Generated with ❤️ by FairCheck</i>", styles["Normal"]))
    doc.build(content)
    buffer.seek(0)
    return buffer

# -------------------------------
# Generate PDF Button
# -------------------------------
if st.button("📄 Generate PDF Report with Chart"):
    try:
        pdf_buf = create_pdf(
            "FairCheck — Automated Bias Detection Report",
            bias_df=bias_df,
            model_bias=model_bias,
            clean_df=clean_df,
            target_col=target_col
        )
        st.download_button(
            label="📥 Download Full PDF Report",
            data=pdf_buf,
            file_name="FairCheck_Report.pdf",
            mime="application/pdf"
        )
        st.success("✅ Report with radar chart generated successfully!")
    except Exception as e:
        st.error(f"PDF generation failed: {e}")
