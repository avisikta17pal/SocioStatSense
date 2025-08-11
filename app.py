from __future__ import annotations

import io
from datetime import date, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import grangercausalitytests


# ------------------------------------------------------------
# Page config
# ------------------------------------------------------------
st.set_page_config(
    page_title="SocioStatSense | Upload & Analyze",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------
# Custom CSS for a modern, centered, minimalist design
# ------------------------------------------------------------
CUSTOM_CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  :root {
    --bg: #0f1216;
    --panel: #141a21;
    --panel-2: #10161d;
    --text: #e6ebf1;
    --muted: #95a1b2;
    --accent: #5aa9e6;
    --accent-2: #7fc8a9;
    --warn: #ffcc66;
    --error: #ff7a7a;
    --success: #69db7c;
    --card-radius: 14px;
    --border: 1px solid rgba(255,255,255,0.06);
    --maxw: 1250px;
  }
  html, body, [class^="css"]  { font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, 'Helvetica Neue', Arial; }
  .block-container { max-width: var(--maxw); padding-top: 1.2rem; padding-bottom: 2rem; margin: 0 auto; }
  body { color: var(--text); background: radial-gradient(1200px 800px at 20% -10%, rgba(90,169,230,0.08), transparent), radial-gradient(1200px 800px at 120% 10%, rgba(127,200,169,0.06), transparent), var(--bg); }
  .panel { background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); border: var(--border); border-radius: var(--card-radius); box-shadow: 0 10px 25px rgba(0, 0, 0, 0.25); padding: 1.1rem 1.2rem; }
  .section-title { display:flex; gap:0.6rem; align-items:center; font-weight:600; font-size:1.1rem; }
  .section-subtitle { color: var(--muted); font-size: 0.95rem; margin-top: 0.2rem; }
  .help { color: var(--muted); cursor: help; border-bottom: 1px dotted var(--muted); }
  .banner { border-radius: 12px; padding: 0.8rem 1.0rem; border: var(--border); margin-bottom: 0.5rem; }
  .banner-info { background: rgba(90,169,230,0.12); color: #d9ecff; }
  .banner-warn { background: rgba(255, 204, 102, 0.18); color: #fff4d6; }
  .banner-error { background: rgba(255, 122, 122, 0.18); color: #ffe3e3; }
  .banner-success { background: rgba(105, 219, 124, 0.18); color: #e6ffed; }
  .footer { color: var(--muted); text-align:center; font-size: 0.9rem; margin-top: 2rem; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ------------------------------------------------------------
# Constants and validation helpers
# ------------------------------------------------------------
REQUIRED_COLUMNS: List[str] = [
    "date",
    "unemployment_rate",
    "inflation_rate",
    "gdp_growth",
    "twitter_sentiment",
    "google_trends_index",
    "avg_temperature",
    "stock_index_close",
    "mobility_change",
    "covid_cases",
]


def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    return (len(missing) == 0, missing)


# ------------------------------------------------------------
# Caching helpers
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_csv(file: io.BytesIO | io.StringIO) -> pd.DataFrame:
    df = pd.read_csv(file)
    return df


# ------------------------------------------------------------
# Sidebar (optional)
# ------------------------------------------------------------
with st.sidebar:
    st.caption("Use the uploader on the main page.")

# ------------------------------------------------------------
# Header + Centered Uploader
# ------------------------------------------------------------
st.markdown(
    """
    <div class="panel" style="text-align:center;">
      <h2 style="margin:0">SocioStatSense — Upload & Analyze</h2>
      <div class="section-subtitle">Upload your dataset, then run modeling, causal analysis, and what‑if scenarios.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns([1, 6, 1])
with c2:
    st.markdown("#### Upload Dataset")
    uploaded = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        help=",".join(REQUIRED_COLUMNS),
    )

# ------------------------------------------------------------
# Load Dataset
# ------------------------------------------------------------
active_df: pd.DataFrame | None = None
errors: List[str] = []

if uploaded is not None:
    try:
        df_up = load_csv(uploaded)
        # Normalize column names (trim spaces)
        df_up.columns = [str(c).strip() for c in df_up.columns]
        # Allow case-insensitive 'date' column and rename to exact 'date'
        if "date" not in df_up.columns:
            for c in df_up.columns:
                if str(c).strip().lower() == "date":
                    df_up = df_up.rename(columns={c: "date"})
                    break
        ok, missing = validate_dataframe(df_up)
        if not ok:
            errors.append(
                f"CSV is missing required columns: {missing}. Expected columns: {REQUIRED_COLUMNS}"
            )
        else:
            active_df = df_up.copy()
            st.markdown(
                "<div class='banner banner-success'>Uploaded dataset loaded successfully.</div>",
                unsafe_allow_html=True,
            )
    except Exception as e:
        errors.append(f"Failed to read CSV: {e}")

if errors:
    for msg in errors:
        st.markdown(f"<div class='banner banner-error'>{msg}</div>", unsafe_allow_html=True)
    st.stop()

if active_df is None:
    st.markdown(
        """
        <div class="banner banner-info">
          Upload a CSV using the sidebar. Required columns:
          <code>date, unemployment_rate, inflation_rate, gdp_growth, twitter_sentiment, google_trends_index, avg_temperature, stock_index_close, mobility_change, covid_cases</code>.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ------------------------------------------------------------
# Data preparation: handle missing values
# ------------------------------------------------------------
df = active_df.copy()
# Ensure date is datetime
# Robust date parsing
try:
    parsed = pd.to_datetime(
        df["date"],
        errors="coerce",
        infer_datetime_format=True,
        utc=False,
    )
    # Fallback: try day-first if many NaT
    if parsed.isna().mean() > 0.2:
        parsed = pd.to_datetime(df["date"], errors="coerce", dayfirst=True, infer_datetime_format=True)
    # Fallback 2: try numeric epoch
    if parsed.isna().mean() > 0.2:
        try:
            as_float = pd.to_numeric(df["date"], errors="coerce")
            # Heuristic: seconds vs ms
            parsed_epoch = pd.to_datetime(
                np.where(as_float > 1e12, as_float, as_float * 1000),
                errors="coerce",
                unit="ms",
                utc=False,
            )
            parsed = parsed.fillna(parsed_epoch)
        except Exception:
            pass
    # Drop rows that are still invalid
    bad = parsed.isna().sum()
    if bad > 0:
        st.markdown(
            f"<div class='banner banner-warn'>Dropped {int(bad)} rows with unparseable dates.</div>",
            unsafe_allow_html=True,
        )
    df = df.assign(date=parsed.dropna().dt.date).dropna(subset=["date"]).reset_index(drop=True)
except Exception:
    st.markdown(
        "<div class='banner banner-error'>Invalid date format in 'date' column (unable to parse).</div>",
        unsafe_allow_html=True,
    )
    st.stop()

# Missing data summary
na_counts = df.drop(columns=["date"]).isna().sum()
if na_counts.sum() > 0:
    st.markdown(
        f"<div class='banner banner-warn'>Missing values detected: {na_counts.to_dict()}. Applying forward/backward fill, then mean imputation.</div>",
        unsafe_allow_html=True,
    )
    df = df.sort_values("date")
    df = df.set_index(pd.to_datetime(df["date"]))
    df = df.drop(columns=["date"])  # temporarily drop
    df = df.ffill().bfill().apply(lambda s: s.fillna(s.mean()))
    df.insert(0, "date", df.index.date)

# ------------------------------------------------------------
# Visualizations: time series of each column
# ------------------------------------------------------------
st.markdown(
    """
<div class="section-title">📈 Time Series Overview <span class="help" title="Each feature visualized across the chosen date range.">?</span></div>
<div class="section-subtitle">Clean, interactive line charts for quick inspection.</div>
""",
    unsafe_allow_html=True,
)

num_cols = [c for c in df.columns if c != "date"]
rows = [num_cols[i:i + 3] for i in range(0, len(num_cols), 3)]
for row in rows:
    cols = st.columns(len(row))
    for col, cname in zip(cols, row):
        with col:
            fig = px.line(df, x="date", y=cname, template="plotly_white")
            fig.update_layout(
                height=280,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e6ebf1"),
            )
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")

# ------------------------------------------------------------
# Modeling: multivariate regression for gdp_growth
# ------------------------------------------------------------
st.markdown(
    """
<div class="section-title">🧠 Modeling <span class="help" title="Multivariate regression predicts gdp_growth from other variables with uncertainty bands from residuals.">?</span></div>
<div class="section-subtitle">Simple, fast prediction demo. Replace with your production models as needed.</div>
""",
    unsafe_allow_html=True,
)

# Fit simple regression model
numeric_cols = [c for c in df.columns if c != "date"]
target = "gdp_growth"
features = [c for c in numeric_cols if c != target]

X_all = df[features].astype(float).values
y = df[target].astype(float).values
model = LinearRegression().fit(X_all, y)

# Predictions and uncertainty via residual std
resid_std = float(np.std(y - model.predict(X_all)))
y_hat = model.predict(X_all)
y_low = y_hat - 1.96 * resid_std
y_up = y_hat + 1.96 * resid_std

pred_df = pd.DataFrame(
    {
        "date": df["date"],
        "gdp_growth": df["gdp_growth"].astype(float).values,
        "y_hat": y_hat,
        "y_low": y_low,
        "y_up": y_up,
    }
)

fig_pred = go.Figure()
fig_pred.add_trace(
    go.Scatter(
        x=pred_df["date"],
        y=pred_df["y_up"],
        line=dict(color="rgba(90,169,230,0)"),
        showlegend=False,
        hoverinfo="skip",
    )
)
fig_pred.add_trace(
    go.Scatter(
        x=pred_df["date"],
        y=pred_df["y_low"],
        fill="tonexty",
        fillcolor="rgba(90,169,230,0.18)",
        line=dict(color="rgba(90,169,230,0)"),
        name="Uncertainty",
        hoverinfo="skip",
    )
)
fig_pred.add_trace(
    go.Scatter(
        x=pred_df["date"],
        y=pred_df["y_hat"],
        mode="lines",
        line=dict(color="#5aa9e6", width=3),
        name="Prediction",
    )
)
fig_pred.add_trace(
    go.Scatter(
        x=pred_df["date"],
        y=pred_df["gdp_growth"],
        mode="lines",
        line=dict(color="#7fc8a9", width=2),
        name="Actual",
    )
)
fig_pred.update_layout(
    template="plotly_white",
    height=420,
    margin=dict(l=20, r=20, t=20, b=20),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e6ebf1"),
)
fig_pred.update_xaxes(showgrid=False)
fig_pred.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
st.plotly_chart(fig_pred, use_container_width=True, theme="streamlit")

# ------------------------------------------------------------
# Causal analysis (simple Granger summary to gdp_growth)
# ------------------------------------------------------------
st.markdown(
    """
<div class="section-title">🔗 Causal Analysis <span class="help" title="Granger tests check if a variable improves forecasting of gdp_growth.">?</span></div>
<div class="section-subtitle">Edges shown where p-value < 0.05.</div>
""",
    unsafe_allow_html=True,
)

try:
    df_for_gc = df.dropna()
    df_for_gc = df_for_gc[["gdp_growth"] + [c for c in num_cols if c != "gdp_growth"]]
    # Build Granger summary
    results = []
    for col in [c for c in df_for_gc.columns if c != "gdp_growth"]:
        try:
            series = df_for_gc[["gdp_growth", col]]
            if len(series) < 30:
                continue
            res = grangercausalitytests(series, maxlag=2, verbose=False)
            min_p = min(res[lag][0]["ssr_ftest"][1] for lag in res.keys())
            results.append({"cause": col, "effect": "gdp_growth", "p_value": float(min_p)})
        except Exception:
            continue
    gc = pd.DataFrame(results).sort_values("p_value")
except Exception:
    gc = pd.DataFrame(columns=["cause", "effect", "p_value"])  # empty if failure

if gc.empty:
    st.caption("No significant Granger-causal effects detected or insufficient data length.")
else:
    st.dataframe(gc, use_container_width=True)
    sig = gc[gc["p_value"] < 0.05]
    if not sig.empty:
        nodes = ["gdp_growth"] + sig["cause"].tolist()
        angles = np.linspace(0, 2 * np.pi, len(nodes), endpoint=False)
        xs = np.cos(angles)
        ys = np.sin(angles)
        pos = {n: (xs[i], ys[i]) for i, n in enumerate(nodes)}
        edge_x, edge_y = [], []
        for _, r in sig.iterrows():
            x0, y0 = pos[r["cause"]]
            x1, y1 = pos[r["effect"]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        fig_net = go.Figure()
        fig_net.add_trace(
            go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line=dict(width=1.5, color="rgba(230,235,241,0.35)"),
                hoverinfo="none",
            )
        )
        fig_net.add_trace(
            go.Scatter(
                x=[pos[n][0] for n in nodes],
                y=[pos[n][1] for n in nodes],
                mode="markers+text",
                text=nodes,
                textposition="top center",
                marker=dict(size=22, color="#5aa9e6", line=dict(width=2, color="rgba(255,255,255,0.4)")),
            )
        )
        fig_net.update_layout(
            template="plotly_white",
            height=420,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        fig_net.update_xaxes(visible=False)
        fig_net.update_yaxes(visible=False)
        st.plotly_chart(fig_net, use_container_width=True, theme="streamlit")

# ------------------------------------------------------------
# What-if scenarios
# ------------------------------------------------------------
st.markdown(
    """
<div class="section-title">🧪 What‑If Scenarios <span class="help" title="Adjust key inputs and see predicted gdp_growth update.">?</span></div>
<div class="section-subtitle">Interactive scenario controls update the prediction for the latest date.</div>
""",
    unsafe_allow_html=True,
)

scenario_cols = [
    c
    for c in features
    if c
    in [
        "unemployment_rate",
        "inflation_rate",
        "twitter_sentiment",
        "google_trends_index",
        "avg_temperature",
        "stock_index_close",
        "mobility_change",
        "covid_cases",
    ]
]

defaults = df.iloc[-1][scenario_cols].to_dict()
sc_vals: Dict[str, float] = {}
cols = st.columns(min(4, len(scenario_cols)))
for i, colname in enumerate(scenario_cols):
    c = cols[i % len(cols)]
    with c:
        current = float(defaults[colname])
        rng = (float(df[colname].min()), float(df[colname].max()))
        step = (rng[1] - rng[0]) / 100.0 if rng[1] > rng[0] else 0.1
        sc_vals[colname] = st.slider(colname, min_value=rng[0], max_value=rng[1], value=current, step=step)

last_row = df.iloc[[-1]].copy()
X_scn = last_row[features].astype(float).copy()
for k, v in sc_vals.items():
    if k in X_scn.columns:
        X_scn.loc[:, k] = v

y_hat_scn = model.predict(X_scn.values)
lo_scn = y_hat_scn - 1.96 * resid_std
up_scn = y_hat_scn + 1.96 * resid_std

st.markdown(
    f"""
    <div class="panel" style="text-align:center;">
        <div class="section-title">Scenario Prediction</div>
        <div class="section-subtitle">Predicted gdp_growth (latest date) with 95% interval.</div>
        <h3 style="margin:0.3rem 0 0;">{y_hat_scn[0]:.2f}%</h3>
        <div style="color:var(--muted);">[{lo_scn[0]:.2f}%, {up_scn[0]:.2f}%]</div>
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
      SocioStatSense · Upload & Analyze · v0.2.1<br/>
      Upload data · Analyze · Explore scenarios
    </div>
    """,
    unsafe_allow_html=True,
)