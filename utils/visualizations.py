# utils/visualizations.py

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# --------------------------------------------------------------------
# 1️⃣ Radar Chart (Fairness Comparison)
# --------------------------------------------------------------------
def radar_plot(metrics, values, name="Model", color=None, bgcolor="#0e1117"):
    """
    Create a smooth and modern radar (spider) chart for fairness metric visualization.

    Parameters:
        metrics (list): Metric labels (e.g., ["DP", "EO", "PP"])
        values (list): Metric values (floats between 0-1)
        name (str): Trace name (e.g., 'KNN', 'KMeans')
        color (str): Optional hex color for line/fill
        bgcolor (str): Background color for chart
    """
    try:
        vals = list(values) + [values[0]]
        labels = list(metrics) + [metrics[0]]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=vals,
            theta=labels,
            fill='toself',
            name=name,
            line_color=color or None,
            opacity=0.7,
            hoverinfo="r+theta+name"
        ))

        fig.update_layout(
            title=dict(text=f"Fairness Radar — {name}", x=0.5, font=dict(size=16, color="#4CAF50")),
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickfont=dict(size=10, color="white"),
                    gridcolor="gray"
                ),
                bgcolor="rgba(0,0,0,0)"
            ),
            paper_bgcolor=bgcolor,
            showlegend=True,
            legend=dict(orientation="h", y=-0.2)
        )
        return fig
    except Exception as e:
        logging.error(f"Radar plot failed: {e}")
        return go.Figure()


# --------------------------------------------------------------------
# 2️⃣ Bias Heatmap (Feature-level Bias)
# --------------------------------------------------------------------
def bias_heatmap(bias_df, title="Feature Bias Heatmap"):
    """
    Plot bias metrics as a heatmap.

    Parameters:
        bias_df (pd.DataFrame): Columns must include Feature, SPD, DI, Mutual Info
    """
    try:
        df = bias_df.copy()
        if "Feature" in df.columns:
            df = df.set_index("Feature")

        cols = [c for c in ["SPD", "DI", "Mutual Info"] if c in df.columns]
        if not cols:
            raise ValueError("Bias DataFrame must contain SPD, DI, or Mutual Info columns")

        df = df[cols].fillna(0)

        fig = px.imshow(
            df.T,
            color_continuous_scale="RdYlGn_r",
            aspect="auto",
            origin="lower",
            labels=dict(x="Feature", y="Metric", color="Bias Intensity"),
        )

        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=16, color="#4CAF50")),
            xaxis=dict(showgrid=False, tickangle=45),
            yaxis=dict(showgrid=False),
            paper_bgcolor="#0e1117",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(t=40, b=40),
            height=400,
        )
        return fig
    except Exception as e:
        logging.error(f"Heatmap generation failed: {e}")
        return go.Figure()


# --------------------------------------------------------------------
# 3️⃣ Metric Bar Chart (Optional Summary)
# --------------------------------------------------------------------
def metric_bar_chart(bias_df, metrics=("SPD", "DI", "Mutual Info")):
    """
    Optional bar chart for bias summary across all features.
    """
    try:
        melted = bias_df.melt(
            id_vars="Feature",
            value_vars=[m for m in metrics if m in bias_df.columns],
            var_name="Metric",
            value_name="Value"
        ).dropna()

        fig = px.bar(
            melted,
            x="Feature",
            y="Value",
            color="Metric",
            barmode="group",
            title="Bias Metrics by Feature",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_layout(
            paper_bgcolor="#0e1117",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(tickangle=45),
            height=450,
        )
        return fig
    except Exception as e:
        logging.error(f"Bar chart failed: {e}")
        return go.Figure()
