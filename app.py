from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.stattools import grangercausalitytests


# ------------------------------------------------------------
# Page config
# ------------------------------------------------------------
st.set_page_config(
    page_title="SocioStatSense | Upload or Create Dataset · Analyze",
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


@st.cache_data(show_spinner=False)
def generate_synthetic(
    start: date,
    end: date,
    configs: Dict[str, Dict[str, float | str]],
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start=start, end=end, freq="D")
    n = len(idx)

    def make_series(min_v: float, max_v: float, pattern: str, noise: float) -> np.ndarray:
        span = max_v - min_v if max_v > min_v else 1.0
        x = np.linspace(0, 1, n)
        base: np.ndarray
        if pattern == "increasing":
            base = x
        elif pattern == "decreasing":
            base = 1 - x
        elif pattern == "cyclic":
            base = 0.5 + 0.5 * np.sin(2 * np.pi * 2 * x)
        else:
            base = np.ones_like(x) * 0.5
        noise_vec = rng.normal(0, noise * 0.1, size=n)  # noise normalized
        series = min_v + (base + noise_vec) * span
        return np.clip(series, min(min_v, max_v), max(min_v, max_v))

    data: Dict[str, np.ndarray] = {}
    for col in REQUIRED_COLUMNS:
        if col == "date":
            continue
        cfg = configs[col]
        data[col] = make_series(
            float(cfg["min"]), float(cfg["max"]), str(cfg["pattern"]), float(cfg["noise"])
        )

    df = pd.DataFrame(data, index=idx)
    df.insert(0, "date", idx.date)
    return df.reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def fit_regression_model(df: pd.DataFrame) -> Tuple[LinearRegression, float, List[str]]:
    # Predict gdp_growth from other features (simple multivariate regression)
    numeric_cols = [c for c in df.columns if c != "date"]
    target = "gdp_growth"
    features = [c for c in numeric_cols if c != target]

    X = df[features].astype(float).values
    y = df[target].astype(float).values

    model = LinearRegression()
    model.fit(X, y)

    y_hat = model.predict(X)
    resid = y - y_hat
    resid_std = float(np.std(resid))
    return model, resid_std, features


def predict_with_intervals(model: LinearRegression, resid_std: float, X: np.ndarray, z: float = 1.96) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_hat = model.predict(X)
    lower = y_hat - z * resid_std
    upper = y_hat + z * resid_std
    return y_hat, lower, upper


def granger_summary(df: pd.DataFrame, target: str = "gdp_growth", max_lag: int = 2) -> pd.DataFrame:
    # Returns p-values for Granger causality tests: does col -> target?
    results = []
    cols = [c for c in df.columns if c not in ("date", target)]
    for col in cols:
        try:
            # statsmodels expects 2D array [target, cause]
            series = df[[target, col]].dropna()
            if len(series) < 30:
                continue
            res = grangercausalitytests(series, max_lag, verbose=False)
            min_p = min(res[lag][0]["ssr_ftest"][1] for lag in res.keys())
            results.append({"cause": col, "effect": target, "p_value": float(min_p)})
        except Exception:
            continue
    return pd.DataFrame(results).sort_values("p_value")


# ------------------------------------------------------------
# Sidebar: Upload or Create Dataset
# ------------------------------------------------------------
with st.sidebar:
    st.markdown("### Dataset")
    st.caption("Upload a CSV with required columns or create a synthetic dataset.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"], help=",".join(REQUIRED_COLUMNS))

    st.markdown("---")
    st.markdown("#### Dataset Keeper (Synthetic)")

    default_start = date(2015, 1, 1)
    default_end = date(2024, 12, 31)
    d_start, d_end = st.date_input(
        "Date range",
        value=(default_start, default_end),
        help="Choose start and end dates for the synthetic dataset.",
    )

    pattern_options = ["flat", "increasing", "decreasing", "cyclic"]

    def col_cfg(label: str, min_d: float, max_d: float, pattern_d: str, noise_d: float) -> Dict[str, float | str]:
        with st.expander(f"Configure: {label}", expanded=False):
            c1, c2 = st.columns(2)
            with c1:
                vmin = st.number_input(f"{label} min", value=min_d)
                vmax = st.number_input(f"{label} max", value=max_d)
                if vmax < vmin:
                    st.warning("Max < Min adjusted to Min.")
                    vmax = vmin
            with c2:
                pattern = st.selectbox(f"{label} pattern", options=pattern_options, index=pattern_options.index(pattern_d))
                noise = st.slider(f"{label} noise", 0.0, 2.0, noise_d, 0.1)
            return {"min": vmin, "max": vmax, "pattern": pattern, "noise": noise}

    synth_cfgs: Dict[str, Dict[str, float | str]] = {}
    synth_cfgs["unemployment_rate"] = col_cfg("unemployment_rate", 3.0, 10.0, "flat", 0.5)
    synth_cfgs["inflation_rate"] = col_cfg("inflation_rate", 0.0, 8.0, "cyclic", 0.6)
    synth_cfgs["gdp_growth"] = col_cfg("gdp_growth", -5.0, 7.0, "increasing", 0.7)
    synth_cfgs["twitter_sentiment"] = col_cfg("twitter_sentiment", -1.0, 1.0, "cyclic", 0.3)
    synth_cfgs["google_trends_index"] = col_cfg("google_trends_index", 0.0, 100.0, "flat", 0.4)
    synth_cfgs["avg_temperature"] = col_cfg("avg_temperature", -10.0, 35.0, "cyclic", 0.5)
    synth_cfgs["stock_index_close"] = col_cfg("stock_index_close", 1000.0, 5000.0, "increasing", 0.9)
    synth_cfgs["mobility_change"] = col_cfg("mobility_change", -50.0, 50.0, "flat", 0.5)
    synth_cfgs["covid_cases"] = col_cfg("covid_cases", 0.0, 50000.0, "decreasing", 0.8)

    gen_clicked = st.button("Generate synthetic dataset", type="primary", use_container_width=True)

# ------------------------------------------------------------
# Header
# ------------------------------------------------------------
st.markdown(
    """
    <div class="panel" style="text-align:center;">
      <h2 style="margin:0">SocioStatSense Dataset Studio</h2>
      <div class="section-subtitle">Upload or create a dataset, then run modeling, causal analysis, and what‑if scenarios.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------
# Load / Generate Dataset
# ------------------------------------------------------------
active_df: pd.DataFrame | None = None
errors: List[str] = []

if uploaded is not None:
    try:
        df_up = load_csv(uploaded)
        ok, missing = validate_dataframe(df_up)
        if not ok:
            errors.append(
                f"CSV is missing required columns: {missing}. Expected columns: {REQUIRED_COLUMNS}"
            )
        else:
            active_df = df_up.copy()
            st.markdown("<div class='banner banner-success'>Uploaded dataset loaded successfully.</div>", unsafe_allow_html=True)
    except Exception as e:
        errors.append(f"Failed to read CSV: {e}")

if gen_clicked:
    try:
        if isinstance(d_start, tuple) or isinstance(d_end, tuple):
            # If user picks a single date instead of range by mistake, handle gracefully
            start = default_start
            end = default_end
        else:
            start, end = d_start, d_end
        df_syn = generate_synthetic(start, end, synth_cfgs)
        ok, missing = validate_dataframe(df_syn)
        if not ok:
            errors.append(
                f"Generated dataset missing required columns: {missing}. This should not happen; adjust settings and retry."
            )
        else:
            active_df = df_syn.copy()
            st.markdown("<div class='banner banner-success'>Synthetic dataset generated.</div>", unsafe_allow_html=True)
            st.download_button(
                label="Download synthetic CSV",
                data=df_syn.to_csv(index=False).encode("utf-8"),
                file_name="synthetic_socioeconomic.csv",
                mime="text/csv",
            )
    except Exception as e:
        errors.append(f"Failed to generate synthetic dataset: {e}")

if errors:
    for msg in errors:
        st.markdown(f"<div class='banner banner-error'>{msg}</div>", unsafe_allow_html=True)
    st.stop()

if active_df is None:
    st.markdown(
        """
        <div class="banner banner-info">
          Upload a CSV or generate a synthetic dataset using the sidebar. Required columns:
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
try:
    df["date"] = pd.to_datetime(df["date"]).dt.date
except Exception:
    st.markdown("<div class='banner banner-error'>Invalid date format in 'date' column.</div>", unsafe_allow_html=True)
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
st.markdown("""
<div class="section-title">📈 Time Series Overview <span class="help" title="Each feature visualized across the chosen date range.">?</span></div>
<div class="section-subtitle">Clean, interactive line charts for quick inspection.</div>
""", unsafe_allow_html=True)

num_cols = [c for c in df.columns if c != "date"]

# Display as two rows of charts
rows = [num_cols[i:i+3] for i in range(0, len(num_cols), 3)]
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
st.markdown("""
<div class="section-title">🧠 Modeling <span class="help" title="Multivariate regression predicts gdp_growth from other variables with uncertainty bands from residuals.">?</span></div>
<div class="section-subtitle">Simple, fast prediction demo. Replace with your production models as needed.</div>
""", unsafe_allow_html=True)

model, resid_std, features = fit_regression_model(df)
X_all = df[features].astype(float).values
y_hat, y_low, y_up = predict_with_intervals(model, resid_std, X_all)

pred_df = pd.DataFrame({
    "date": df["date"],
    "gdp_growth": df["gdp_growth"].astype(float).values,
    "y_hat": y_hat,
    "y_low": y_low,
    "y_up": y_up,
})

fig_pred = go.Figure()
fig_pred.add_trace(go.Scatter(x=pred_df["date"], y=pred_df["y_up"], line=dict(color="rgba(90,169,230,0)"), showlegend=False, hoverinfo="skip"))
fig_pred.add_trace(go.Scatter(x=pred_df["date"], y=pred_df["y_low"], fill="tonexty", fillcolor="rgba(90,169,230,0.18)", line=dict(color="rgba(90,169,230,0)"), name="Uncertainty", hoverinfo="skip"))
fig_pred.add_trace(go.Scatter(x=pred_df["date"], y=pred_df["y_hat"], mode="lines", line=dict(color="#5aa9e6", width=3), name="Prediction"))
fig_pred.add_trace(go.Scatter(x=pred_df["date"], y=pred_df["gdp_growth"], mode="lines", line=dict(color="#7fc8a9", width=2), name="Actual"))
fig_pred.update_layout(template="plotly_white", height=420, margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#e6ebf1"))
fig_pred.update_xaxes(showgrid=False)
fig_pred.update_yaxes(gridcolor="rgba(255,255,255,0.08)")
st.plotly_chart(fig_pred, use_container_width=True, theme="streamlit")

# ------------------------------------------------------------
# Causal analysis (simple Granger summary to gdp_growth)
# ------------------------------------------------------------
st.markdown("""
<div class="section-title">🔗 Causal Analysis <span class="help" title="Granger tests check if a variable improves forecasting of gdp_growth.">?</span></div>
<div class="section-subtitle">Edges shown where p-value < 0.05.</div>
""", unsafe_allow_html=True)

try:
    df_for_gc = df.copy()
    # Ensure numeric and no missing
    df_for_gc = df_for_gc.dropna()
    df_for_gc = df_for_gc[["gdp_growth"] + [c for c in num_cols if c != "gdp_growth"]]
    gc = granger_summary(df_for_gc, target="gdp_growth", max_lag=2)
except Exception as e:
    gc = pd.DataFrame(columns=["cause", "effect", "p_value"])  # empty if failure

if gc.empty:
    st.caption("No significant Granger-causal effects detected or insufficient data length.")
else:
    st.dataframe(gc, use_container_width=True)

    # Network visualization to target gdp_growth
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
        fig_net.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=1.5, color="rgba(230,235,241,0.35)"), hoverinfo="none"))
        fig_net.add_trace(go.Scatter(x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes], mode="markers+text", text=nodes, textposition="top center", marker=dict(size=22, color="#5aa9e6", line=dict(width=2, color="rgba(255,255,255,0.4)"))))
        fig_net.update_layout(template="plotly_white", height=420, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        fig_net.update_xaxes(visible=False)
        fig_net.update_yaxes(visible=False)
        st.plotly_chart(fig_net, use_container_width=True, theme="streamlit")

# ------------------------------------------------------------
# What-if scenarios: adjust features to see predicted gdp_growth
# ------------------------------------------------------------
st.markdown("""
<div class="section-title">🧪 What‑If Scenarios <span class="help" title="Adjust key inputs and see predicted gdp_growth update.">?</span></div>
<div class="section-subtitle">Interactive scenario controls update the prediction for the latest date.</div>
""", unsafe_allow_html=True)

scenario_cols = [c for c in features if c in [
    "unemployment_rate", "inflation_rate", "twitter_sentiment", "google_trends_index", "avg_temperature", "stock_index_close", "mobility_change", "covid_cases"
]]

defaults = df.iloc[-1][scenario_cols].to_dict()
sc_vals: Dict[str, float] = {}
cols = st.columns(min(4, len(scenario_cols)))
for i, colname in enumerate(scenario_cols):
    c = cols[i % len(cols)]
    with c:
        current = float(defaults[colname])
        rng = (float(df[colname].min()), float(df[colname].max()))
        sc_vals[colname] = st.slider(colname, min_value=rng[0], max_value=rng[1], value=current, step=(rng[1]-rng[0])/100.0 if rng[1]>rng[0] else 0.1)

# Build scenario feature vector based on last row but replacing selected fields
last_row = df.iloc[[-1]].copy()
X_scn = last_row[features].astype(float).copy()
for k, v in sc_vals.items():
    if k in X_scn.columns:
        X_scn.loc[:, k] = v

y_hat_scn, lo_scn, up_scn = predict_with_intervals(model, resid_std, X_scn.values)

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
      SocioStatSense · Dataset Studio · v0.2.0<br/>
      Upload or synthesize data · Analyze · Explore scenarios
    </div>
    """,
    unsafe_allow_html=True,
)