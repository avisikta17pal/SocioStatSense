import math
from datetime import date, timedelta
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ------------------------------------------------------------
# Page config
# ------------------------------------------------------------
st.set_page_config(
    page_title="SocioStatSense | Adaptive Socio-Economic Modeling",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------------------------------------
# Custom CSS for a modern, centered, minimalist design
# ------------------------------------------------------------
CUSTOM_CSS = """
<style>
  /* Import modern, readable font */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  :root {
    --bg: #0f1216;               /* dark slate background */
    --panel: #141a21;            /* panel background */
    --panel-2: #10161d;          /* slightly darker */
    --text: #e6ebf1;             /* primary text */
    --muted: #95a1b2;            /* muted text */
    --accent: #5aa9e6;           /* primary accent */
    --accent-2: #7fc8a9;         /* secondary accent */
    --warn: #ffcc66;             /* warning */
    --error: #ff7a7a;            /* error */
    --success: #69db7c;          /* success */
    --card-radius: 14px;
    --soft-shadow: 0 10px 25px rgba(0, 0, 0, 0.25);
    --soft-shadow-2: 0 8px 16px rgba(0, 0, 0, 0.2);
    --border: 1px solid rgba(255,255,255,0.06);
    --maxw: 1200px;
  }

  html, body, [class^="css"]  {
    font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, 'Helvetica Neue', Arial, 'Noto Sans', 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol';
  }

  /* Center the main block and control its width */
  .block-container {
    max-width: var(--maxw);
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    margin: 0 auto;
  }

  /* Dark modern theme overrides */
  body {
    color: var(--text);
    background: radial-gradient(1200px 800px at 20% -10%, rgba(90,169,230,0.08), transparent),
                radial-gradient(1200px 800px at 120% 10%, rgba(127,200,169,0.06), transparent),
                var(--bg);
  }

  /* Headings */
  h1, h2, h3, h4 {
    letter-spacing: 0.2px;
  }

  /* Panel containers */
  .panel {
    background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
    border: var(--border);
    border-radius: var(--card-radius);
    box-shadow: var(--soft-shadow);
    padding: 1.2rem 1.2rem;
  }

  .panel-tight { padding: 0.9rem 1.0rem; }
  .panel-spacious { padding: 1.6rem 1.6rem; }

  /* Metric cards */
  .metric-card {
    background: var(--panel);
    border: var(--border);
    border-radius: var(--card-radius);
    box-shadow: var(--soft-shadow-2);
    padding: 1rem 1.2rem;
    text-align: center;
  }
  .metric-label { color: var(--muted); font-size: 0.9rem; }
  .metric-value { font-size: 1.6rem; font-weight: 700; color: var(--text); }
  .metric-delta { font-size: 0.95rem; }

  /* Section titles */
  .section-title {
    display: flex; align-items: center; gap: 0.6rem;
    font-weight: 600; font-size: 1.1rem; color: var(--text);
  }
  .section-subtitle { color: var(--muted); font-size: 0.95rem; margin-top: 0.15rem; }

  /* Alert banners */
  .banner { border-radius: 12px; padding: 0.9rem 1.1rem; border: var(--border); }
  .banner-info { background: rgba(90,169,230,0.12); color: #d9ecff; }
  .banner-warn { background: rgba(255, 204, 102, 0.18); color: #fff4d6; }
  .banner-error { background: rgba(255, 122, 122, 0.18); color: #ffe3e3; }
  .banner-success { background: rgba(105, 219, 124, 0.18); color: #e6ffed; }

  /* Tooltip helper */
  .help { color: var(--muted); cursor: help; border-bottom: 1px dotted var(--muted); }

  /* Footer */
  .footer { color: var(--muted); text-align: center; font-size: 0.9rem; margin-top: 2.2rem; }

  /* Centering charts */
  .center { display: flex; justify-content: center; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ------------------------------------------------------------
# Sidebar Controls
# ------------------------------------------------------------
with st.sidebar:
    st.markdown("### Controls")
    st.caption("Adjust parameters for scenario exploration. These settings only affect the visualized placeholders.")

    today = date.today()
    default_range: Tuple[date, date] = (today - timedelta(days=60), today)
    drange = st.date_input(
        "Date range",
        value=default_range,
        help="Select the time window to visualize model outputs.",
    )

    sources = st.multiselect(
        "Data sources",
        options=["Government Indicators", "Twitter Sentiment", "Google Trends", "Weather", "Market Prices"],
        default=["Government Indicators", "Market Prices"],
        help="Select which data streams to include in simulated views.",
    )

    scenario = st.selectbox(
        "Scenario",
        options=["Baseline", "Mild Shock", "Severe Shock", "Policy Intervention"],
        help="Choose a scenario to stress-test the simulated outputs.",
    )

    shock_magnitude = st.slider(
        "Shock magnitude",
        min_value=0.0, max_value=3.0, value=1.0, step=0.1,
        help="Controls the amplitude of uncertainty bands in the prediction chart (visual only).",
    )

    show_anomalies = st.toggle("Show anomaly markers", value=True, help="If enabled, displays simulated anomalies as alerts.")

    st.markdown("---")
    st.caption("Tip: Hover elements with the dotted underline for explanations.")


# ------------------------------------------------------------
# Helper functions to generate placeholder visuals
# ------------------------------------------------------------
def _simulate_series(n: int = 90, seed: int = 42, noise_scale: float = 0.8) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = np.arange(n)
    base = 0.05 * x + 3 * np.sin(x / 7) + 0.8 * np.cos(x / 3.5)
    noise = rng.normal(0, noise_scale, size=n)
    mean = base + noise
    ci = np.clip(0.6 + 0.2 * np.sin(x / 4), 0.4, 1.2)
    lower = mean - ci
    upper = mean + ci
    return pd.DataFrame({"t": x, "mean": mean, "lower": lower, "upper": upper})


def build_prediction_figure(df: pd.DataFrame, accent_color: str = "#5aa9e6") -> go.Figure:
    fig = go.Figure()
    # Uncertainty band (lower -> upper)
    fig.add_trace(
        go.Scatter(
            x=df["t"], y=df["upper"],
            line=dict(color="rgba(90,169,230,0)"),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["t"], y=df["lower"],
            fill="tonexty",
            fillcolor="rgba(90,169,230,0.18)",
            line=dict(color="rgba(90,169,230,0)"),
            name="Uncertainty",
            hoverinfo="skip",
        )
    )
    # Mean line
    fig.add_trace(
        go.Scatter(
            x=df["t"], y=df["mean"],
            mode="lines",
            line=dict(color=accent_color, width=3),
            name="Prediction",
        )
    )

    fig.update_layout(
        template="plotly_white",
        height=420,
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(title=None, gridcolor="rgba(255,255,255,0.08)", zeroline=False),
        yaxis=dict(title=None, gridcolor="rgba(255,255,255,0.08)", zeroline=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e6ebf1"),
    )
    return fig


def build_importance_figure(features: List[str], seed: int = 7) -> px.bar:
    rng = np.random.default_rng(seed)
    importance = rng.random(len(features))
    importance = importance / importance.sum()
    df_imp = pd.DataFrame({"feature": features, "importance": importance})
    df_imp = df_imp.sort_values("importance", ascending=True)
    fig = px.bar(
        df_imp,
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale=["#7FC8A9", "#5AA9E6"],
        height=420,
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=10, r=10, t=10, b=10),
        coloraxis_showscale=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e6ebf1"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
    fig.update_yaxes(showgrid=False)
    return fig


def build_causal_network_figure(nodes: List[str]) -> go.Figure:
    # Create a simple directed acyclic structure for placeholder
    num = len(nodes)
    angles = np.linspace(0, 2 * np.pi, num, endpoint=False)
    radius = 1.0
    xs = np.cos(angles) * radius
    ys = np.sin(angles) * radius

    node_positions = {n: (xs[i], ys[i]) for i, n in enumerate(nodes)}

    # Edges: connect i -> i+1 and i -> i+2 as sample causal links
    edges: List[Tuple[str, str]] = []
    for i in range(num):
        edges.append((nodes[i], nodes[(i + 1) % num]))
        if i % 2 == 0:
            edges.append((nodes[i], nodes[(i + 2) % num]))

    edge_x, edge_y = [], []
    for src, dst in edges:
        x0, y0 = node_positions[src]
        x1, y1 = node_positions[dst]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.5, color="rgba(230,235,241,0.35)"),
        hoverinfo="none",
        mode="lines",
    )

    node_trace = go.Scatter(
        x=xs, y=ys,
        mode="markers+text",
        text=nodes,
        textposition="top center",
        marker=dict(
            size=24,
            color="#5aa9e6",
            line=dict(width=2, color="rgba(255,255,255,0.4)"),
            opacity=0.95,
        ),
        hovertemplate="<b>%{text}</b><extra></extra>",
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        template="plotly_white",
        height=460,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    return fig


# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
header = st.container()
with header:
    c1, c2, c3 = st.columns([1, 6, 1])
    with c2:
        st.markdown(
            """
            <div class="panel panel-spacious" style="text-align:center;">
              <h1 style="margin: 0 0 6px 0;">SocioStatSense</h1>
              <div class="section-subtitle">Adaptive socio-economic modeling with real-time insights and uncertainty</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ------------------------------------------------------------
# KPI Metrics
# ------------------------------------------------------------
metrics = st.container()
with metrics:
    m1, m2, m3, m4 = st.columns(4)

    def _metric_card(label: str, value: str, delta: str, delta_color: str) -> None:
        st.markdown(
            f"""
            <div class="metric-card">
              <div class="metric-label">{label}</div>
              <div class="metric-value">{value}</div>
              <div class="metric-delta" style="color:{delta_color}">{delta}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with m1:
        _metric_card("Nowcast GDP Growth", "2.1%", "+0.2 pp vs last week", "#7fc8a9")
    with m2:
        _metric_card("Unemployment Risk", "5.4%", "-0.1 pp", "#7fc8a9")
    with m3:
        _metric_card("Inflation Trend", "3.0%", "+0.3 pp", "#ffcc66")
    with m4:
        _metric_card("Alert Level", "Moderate", "2 anomalies", "#ffcc66")


# ------------------------------------------------------------
# Real-time Predictions with Uncertainty
# ------------------------------------------------------------
predictions = st.container()
with predictions:
    st.markdown(
        """
        <div class="section-title">📈 Real-time Predictions <span class="help" title="Mean predictions with shaded 80% uncertainty bands.">?</span></div>
        <div class="section-subtitle">Simulated placeholder — connect to live model outputs in production.</div>
        """,
        unsafe_allow_html=True,
    )

    # Placeholder data simulation influenced by scenario controls
    base_df = _simulate_series(n=120, seed=13)
    scale = 1.0 + (0.25 * (1 if "Severe" in scenario else 0.12 if "Mild" in scenario else 0.0))
    shock_scale = scale * (1 + (shock_magnitude / 5))
    df_pred = base_df.copy()
    df_pred["lower"] = df_pred["mean"] - (df_pred["upper"] - df_pred["mean"]) * shock_scale
    fig_pred = build_prediction_figure(df_pred)

    c = st.container()
    with c:
        st.plotly_chart(fig_pred, use_container_width=True, theme="streamlit")


# ------------------------------------------------------------
# Variable Importance and Causal Network
# ------------------------------------------------------------
vis_section = st.container()
with vis_section:
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown(
            """
            <div class="section-title">🧠 Variable Importance <span class="help" title="Relative contribution of features to predictions.">?</span></div>
            <div class="section-subtitle">Interactive bars with normalized importances.</div>
            """,
            unsafe_allow_html=True,
        )
        features = ["Unemployment", "CPI", "Interest Rates", "Sentiment", "Retail Sales", "PMI", "Oil Price"]
        fig_imp = build_importance_figure(features)
        st.plotly_chart(fig_imp, use_container_width=True, theme="streamlit")

    with right:
        st.markdown(
            """
            <div class="section-title">🔗 Causal Network <span class="help" title="Illustrative causal relationships among key indicators.">?</span></div>
            <div class="section-subtitle">Nodes and edges are placeholders for demo purposes.</div>
            """,
            unsafe_allow_html=True,
        )
        nodes = ["GDP", "CPI", "Unemployment", "Rates", "Sentiment", "Commodities"]
        fig_net = build_causal_network_figure(nodes)
        st.plotly_chart(fig_net, use_container_width=True, theme="streamlit")


# ------------------------------------------------------------
# Alerts & Anomalies
# ------------------------------------------------------------
alerts = st.container()
with alerts:
    st.markdown(
        """
        <div class="section-title">🚨 Alerts & Anomalies <span class="help" title="Detected deviations from expected behavior.">?</span></div>
        <div class="section-subtitle">Simulated notifications to showcase the UX pattern.</div>
        """,
        unsafe_allow_html=True,
    )

    a1, a2 = st.columns(2)
    with a1:
        st.markdown(
            """
            <div class="banner banner-warn">
              Potential inflation spike detected in the last 48h (CPI surprise). Review monetary policy drivers.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with a2:
        st.markdown(
            """
            <div class="banner banner-info">
              Sentiment dip observed in social signals. Impact on retail sales likely minimal.
            </div>
            """,
            unsafe_allow_html=True,
        )

    if show_anomalies:
        st.markdown(
            """
            <div class="banner banner-error" style="margin-top: 0.8rem;">
              Structural break candidate found (Change-point score above threshold). Validate with domain experts.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div class="footer">
      SocioStatSense · Adaptive Socio-Economic Modeling · v0.1.0<br/>
      Built with Streamlit · Designed for clarity, focus, and decision support
    </div>
    """,
    unsafe_allow_html=True,
)