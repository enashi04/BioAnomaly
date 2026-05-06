"""
BioAnomaly — Streamlit Dashboard
Medical-grade anomaly detection for gene expression data (TCGA RNA-seq)
Palette: #028090  #729EA1  #68C3D4  #EEF8FF
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import warnings
from textwrap import dedent
import streamlit.components.v1 as components

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BioAnomaly — Gene Expression Analysis",
    page_icon="icon_bioanomaly.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
#  PALETTE & GLOBAL CSS
# ─────────────────────────────────────────────────────────────
C1 = "#028090"   # deep teal
C2 = "#729EA1"   # muted teal
C3 = "#68C3D4"   # sky blue
C4 = "#EEF8FF"   # near-white ice
ALERT = "#E05263"
WARN = "#F4A261"
OK = "#2EC4B6"
BG = "#F4FBFD"
CARD = "#FFFFFF"
TEXT = "#0B2E35"
SUBTEXT = "#4A7C84"

# RGBA values for Plotly.
# Plotly does not always accept hex colors with alpha such as "#68C3D444".
C1_RGBA_30 = "rgba(2, 128, 144, 0.30)"
C3_RGBA_08 = "rgba(104, 195, 212, 0.08)"
C3_RGBA_12 = "rgba(104, 195, 212, 0.12)"
C3_RGBA_15 = "rgba(104, 195, 212, 0.15)"
C3_RGBA_25 = "rgba(104, 195, 212, 0.25)"
C3_RGBA_30 = "rgba(104, 195, 212, 0.30)"
C3_RGBA_45 = "rgba(104, 195, 212, 0.45)"
ALERT_RGBA_15 = "rgba(224, 82, 99, 0.15)"
OK_RGBA_15 = "rgba(46, 196, 182, 0.15)"

CLASS_COLORS = {
    "BRCA": C1,
    "KIRC": C2,
    "LUAD": C3,
    "PRAD": "#A8DADC",
    "COAD": "#023E4A",
}

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,300&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {{
    font-family: 'DM Sans', sans-serif;
    background-color: {BG};
    color: {TEXT};
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {C1} 0%, #023E4A 100%);
    border-right: none;
}}

[data-testid="stSidebar"] * {{
    color: {C4} !important;
}}

[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stFileUploader label {{
    color: {C3} !important;
    font-weight: 500;
    font-size: 0.82rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}}

[data-testid="stSidebar"] hr {{
    border-color: rgba(255,255,255,0.12) !important;
}}

/* ── File uploader readability ── */
[data-testid="stFileUploader"] section {{
    background-color: #FFFFFF !important;
    border: 1px dashed {C3} !important;
    border-radius: 12px !important;
}}

[data-testid="stFileUploader"] section * {{
    color: {TEXT} !important;
}}

[data-testid="stFileUploader"] button {{
    background-color: {C1} !important;
    color: #FFFFFF !important;
    border-radius: 8px !important;
    border: none !important;
}}

[data-testid="stFileUploader"] small {{
    color: {SUBTEXT} !important;
}}

/* ── Metric cards ── */
[data-testid="metric-container"] {{
    background: {CARD};
    border: 1px solid rgba(104, 195, 212, 0.30);
    border-radius: 14px;
    padding: 18px 22px !important;
    box-shadow: 0 2px 12px rgba(2,128,144,0.07);
}}

[data-testid="metric-container"] label {{
    color: {SUBTEXT} !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.09em;
    text-transform: uppercase;
}}

[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    color: {C1} !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
}}

[data-testid="metric-container"] [data-testid="stMetricDelta"] {{
    font-size: 0.8rem !important;
}}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tab"] {{
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    font-size: 0.88rem;
    color: {SUBTEXT};
    letter-spacing: 0.04em;
    border-bottom: 2px solid transparent;
    padding: 8px 18px;
}}

[data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
    color: {C1} !important;
    border-bottom: 2px solid {C1} !important;
    font-weight: 700;
}}

/* ── Section headers ── */
.section-header {{
    font-family: 'DM Sans', sans-serif;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: {C2};
    margin-bottom: 4px;
}}

.page-title {{
    font-family: 'DM Sans', sans-serif;
    font-size: 2.1rem;
    font-weight: 700;
    color: {C1};
    line-height: 1.1;
    margin-bottom: 0;
}}

.page-sub {{
    font-size: 0.95rem;
    color: {SUBTEXT};
    margin-top: 4px;
    font-weight: 300;
}}

.badge {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.73rem;
    font-weight: 600;
    letter-spacing: 0.06em;
}}

.badge-anomaly {{
    background: rgba(224, 82, 99, 0.15);
    color: {ALERT};
    border: 1px solid rgba(224, 82, 99, 0.30);
}}

.badge-normal {{
    background: rgba(46, 196, 182, 0.15);
    color: {OK};
    border: 1px solid rgba(46, 196, 182, 0.30);
}}

/* ── Info boxes ── */
.info-card {{
    background: linear-gradient(135deg, rgba(2, 128, 144, 0.07) 0%, rgba(104, 195, 212, 0.06) 100%);
    border-left: 3px solid {C3};
    border-radius: 0 10px 10px 0;
    padding: 14px 18px;
    margin: 12px 0;
    font-size: 0.88rem;
    color: {TEXT};
}}

.warn-card {{
    background: rgba(224, 82, 99, 0.08);
    border-left: 3px solid {ALERT};
    border-radius: 0 10px 10px 0;
    padding: 14px 18px;
    margin: 12px 0;
    font-size: 0.88rem;
    color: {TEXT};
}}

/* ── Buttons ── */
.stButton > button {{
    background: {C1};
    color: white !important;
    border: none;
    border-radius: 10px;
    padding: 10px 24px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 600;
    font-size: 0.88rem;
    letter-spacing: 0.04em;
    transition: all 0.2s;
    width: 100%;
    position: relative;
    z-index: 5;
}}

.stButton > button:hover {{
    background: #016B7A;
    box-shadow: 0 4px 16px rgba(2,128,144,0.3);
    transform: translateY(-1px);
    color: white !important;
}}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {{
    border: 1px solid rgba(104, 195, 212, 0.30);
    border-radius: 12px;
    overflow: hidden;
}}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 6px; height: 6px; }}
::-webkit-scrollbar-track {{ background: {C4}; }}
::-webkit-scrollbar-thumb {{ background: {C2}; border-radius: 3px; }}

/* ── Hide streamlit branding ── */
#MainMenu, footer, header {{ visibility: hidden; }}
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
    <div style='padding: 8px 0 20px 0;'>
        <div style='font-size:1.5rem; font-weight:700; letter-spacing:-0.01em;'>🧬 BioAnomaly</div>
        <div style='font-size:0.78rem; opacity:0.65; margin-top:2px; font-weight:300;'>Gene Expression · TCGA RNA-seq</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown(
        '<div class="section-header" style="color:#68C3D4;">Data Sources</div>',
        unsafe_allow_html=True,
    )

    data_file = st.file_uploader("Expression matrix (CSV)", type="csv", key="data")
    labels_file = st.file_uploader("Labels (CSV)", type="csv", key="labels")

    st.markdown("---")
    st.markdown(
        '<div class="section-header" style="color:#68C3D4;">Model Parameters</div>',
        unsafe_allow_html=True,
    )

    pca_var = st.slider(
        "PCA variance retained",
        0.80,
        0.99,
        0.95,
        0.01,
        help="Fraction of variance to keep after PCA dimensionality reduction",
    )
    iso_contam = st.slider(
        "IF contamination rate",
        0.01,
        0.20,
        0.05,
        0.01,
        help="Expected fraction of anomalies (Isolation Forest)",
    )
    ae_epochs = st.slider("Autoencoder epochs", 20, 150, 80, 10)
    ae_threshold = st.slider(
        "AE threshold percentile",
        90,
        99,
        95,
        1,
        help="Reconstruction error above this percentile = anomaly",
    )

    st.markdown("---")
    run_btn = st.button("▶  Run Analysis", use_container_width=True)

    # Do not use position:absolute here.
    # It can overlap the Run button and block clicks.
    st.markdown(
        """
    <div style='margin-top:28px; text-align:center;
                font-size:0.7rem; opacity:0.55; padding:10px 20px;'>
        BioAnomaly v1.0 · Isolation Forest + Autoencoder · Developed by Ihsane ERRAMI
    </div>
    """,
        unsafe_allow_html=True,
    )


#  PAGE HEADER
st.markdown(
    """
<div style='margin-bottom: 28px;'>
    <div class='page-title'>Anomaly Detection Dashboard</div>
    <div class='page-sub'>Biological Gene Expression Analysis · Isolation Forest + Deep Autoencoder · Developed by Ihsane ERRAMI</div>
</div>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────
#  PIPELINE FUNCTIONS
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(data_bytes, labels_bytes):
    import io

    data = pd.read_csv(io.BytesIO(data_bytes), index_col=0)
    labels = pd.read_csv(io.BytesIO(labels_bytes), index_col=0)
    return data, labels


@st.cache_data(show_spinner=False)
def preprocess(_data, pca_var):
    _data = _data.fillna(_data.mean(numeric_only=True))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(_data)
    pca = PCA(n_components=pca_var, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, pca.n_components_


@st.cache_data(show_spinner=False)
def run_iso_forest(X_pca, contamination):
    iso = IsolationForest(
        contamination=contamination,
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )
    preds = iso.fit_predict(X_pca)
    scores = iso.decision_function(X_pca)
    return preds, scores


@st.cache_data(show_spinner=False)
def run_tsne(X_pca):
    n_samples = X_pca.shape[0]
    perplexity = min(30, max(2, (n_samples - 1) // 3))
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    return tsne.fit_transform(X_pca)


def build_autoencoder(input_dim, latent_dim=32):
    class AE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, input_dim),
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))

    return AE()


@st.cache_data(show_spinner=False)
def run_autoencoder(X_pca, epochs, threshold_pct):
    scaler2 = MinMaxScaler()
    X_norm = scaler2.fit_transform(X_pca).astype(np.float32)
    X_t = torch.tensor(X_norm)
    loader = DataLoader(TensorDataset(X_t), batch_size=32, shuffle=True, drop_last=False)

    model = build_autoencoder(X_norm.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    loss_history = []
    model.train()
    for _ in range(epochs):
        epoch_loss = 0
        n_batches = 0

        for (batch,) in loader:
            out = model(batch)
            loss = loss_fn(out, batch)
            opt.zero_grad()
            loss.backward()
            opt.step()

            epoch_loss += loss.item()
            n_batches += 1

        loss_history.append(epoch_loss / max(n_batches, 1))

    model.eval()
    with torch.no_grad():
        X_recon = model(X_t).numpy()

    errors = np.mean((X_norm - X_recon) ** 2, axis=1)
    threshold = np.percentile(errors, threshold_pct)
    ae_preds = (errors > threshold).astype(int)
    return errors, ae_preds, threshold, loss_history


# ─────────────────────────────────────────────────────────────
#  PLOTLY THEME HELPER
# ─────────────────────────────────────────────────────────────
def fig_layout(fig, title="", height=420):
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(family="DM Sans", size=14, color=TEXT),
            x=0,
        ),
        paper_bgcolor="white",
        plot_bgcolor=BG,
        font=dict(family="DM Sans", color=TEXT),
        height=height,
        margin=dict(l=40, r=20, t=48, b=40),
        legend=dict(
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor=C3_RGBA_30,
            borderwidth=1,
            font=dict(size=11),
        ),
        xaxis=dict(
            gridcolor=C3_RGBA_15,
            zerolinecolor=C3_RGBA_30,
            tickfont=dict(size=10),
        ),
        yaxis=dict(
            gridcolor=C3_RGBA_15,
            zerolinecolor=C3_RGBA_30,
            tickfont=dict(size=10),
        ),
    )
    return fig


# ─────────────────────────────────────────────────────────────
#  MAIN CONTENT
# ─────────────────────────────────────────────────────────────
if not run_btn or data_file is None or labels_file is None:
    components.html(
        f"""
        <div style="
            background: white;
            border: 1px solid rgba(104, 195, 212, 0.30);
            border-radius: 18px;
            padding: 32px 36px;
            box-shadow: 0 4px 20px rgba(2,128,144,0.08);
            margin-bottom: 26px;
            font-family: Arial, sans-serif;
        ">
            <div style="
                font-size: 1rem;
                font-weight: 700;
                color: {C1};
                margin-bottom: 14px;
            ">
                BioAnomaly: Biological Anomaly Detection Dashboard
            </div>

            <div style="
                font-size: 1rem;
                line-height: 1.7;
                color: {TEXT};
                max-width: 950px;
            ">
                BioAnomaly is a machine learning project designed to detect unusual biological samples
                from gene expression data. The goal of this dashboard is to help explore high-dimensional
                RNA-seq data, identify atypical samples, and compare the behavior of different anomaly
                detection methods.

                <br><br>

                The project combines two complementary approaches:
                <b>Isolation Forest</b>, which detects samples located in sparse regions of the feature space,
                and a <b>Deep Autoencoder</b>, which identifies samples that are difficult to reconstruct.
                By combining both methods, the dashboard provides a more robust view of potential anomalies.

                <br><br>

                To start the analysis, upload the expression matrix and the label file from the sidebar,
                then click <b>Run Analysis</b>.
            </div>
        </div>
        """,
        height=420,
    )

    c1, c2, c3 = st.columns(3)
    for col, icon, title, desc in [
        (
            c1,
            "🔬",
            "Isolation Forest",
            "Identifies samples that are easiest to isolate — short path lengths in random trees signal anomalies.",
        ),
        (
            c2,
            "🧠",
            "Deep Autoencoder",
            "Learns the latent structure of normal data. High reconstruction error = anomalous sample.",
        ),
        (
            c3,
            "🔗",
            "Combined Score",
            "Union of both detectors with a normalized composite score for robust, low-false-positive detection.",
        ),
    ]:
        col.markdown(
            f"""
        <div style='background:white; border:1px solid rgba(104, 195, 212, 0.30); border-radius:14px;
                    padding:22px; box-shadow:0 2px 12px rgba(2,128,144,0.06); height:160px;'>
            <div style='font-size:1.6rem; margin-bottom:8px;'>{icon}</div>
            <div style='font-weight:700; color:{C1}; margin-bottom:6px;'>{title}</div>
            <div style='font-size:0.82rem; color:{SUBTEXT}; line-height:1.5;'>{desc}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.stop()

# ── Run pipeline ──
with st.spinner("Loading data…"):
    data, labels = load_data(data_file.getvalue(), labels_file.getvalue())

if "Class" not in labels.columns:
    st.error("The labels CSV must contain a column named 'Class'.")
    st.stop()

cancer_types = labels["Class"].values

if len(cancer_types) != len(data):
    st.error(
        "The number of rows in labels.csv must match the number of samples in the expression matrix."
    )
    st.stop()

progress_bar = st.progress(0, text="Preprocessing & PCA…")
X_pca, n_comp = preprocess(data, pca_var)
progress_bar.progress(20, text="Running Isolation Forest…")

iso_preds, iso_scores = run_iso_forest(X_pca, iso_contam)
progress_bar.progress(40, text="Training Autoencoder…")

recon_errors, ae_preds, ae_thresh, loss_hist = run_autoencoder(
    X_pca, ae_epochs, ae_threshold
)
progress_bar.progress(75, text="Computing t-SNE projection…")

X_tsne = run_tsne(X_pca)
progress_bar.progress(95, text="Building dashboard…")

# Combined score
iso_denominator = iso_scores.max() - iso_scores.min()
ae_denominator = recon_errors.max() - recon_errors.min()

iso_norm = (
    (iso_scores - iso_scores.min()) / iso_denominator
    if iso_denominator != 0
    else np.zeros_like(iso_scores)
)
ae_norm = (
    (recon_errors - recon_errors.min()) / ae_denominator
    if ae_denominator != 0
    else np.zeros_like(recon_errors)
)

combined_score = (1 - iso_norm) * 0.5 + ae_norm * 0.5
combined_preds = ((iso_preds == -1) | (ae_preds == 1)).astype(int)

progress_bar.progress(100, text="Done ✓")
progress_bar.empty()

n_total = len(cancer_types)
n_iso = int((iso_preds == -1).sum())
n_ae = int(ae_preds.sum())
n_combined = int(combined_preds.sum())
n_both = int(((iso_preds == -1) & (ae_preds == 1)).sum())


# ─────────────────────────────────────────────────────────────
#  KPI METRICS
# ─────────────────────────────────────────────────────────────
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total Samples", f"{n_total:,}", f"{data.shape[1]:,} genes")
m2.metric("PCA Components", f"{n_comp}", f"{pca_var * 100:.0f}% variance")
m3.metric("IF Anomalies", f"{n_iso}", f"{n_iso / n_total * 100:.1f}% flagged")
m4.metric("AE Anomalies", f"{n_ae}", f"threshold {ae_thresh:.5f}")
m5.metric("Combined Anomalies", f"{n_combined}", f"{n_both} detected by both")

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🗺️  t-SNE Projection",
        "🌲  Isolation Forest",
        "🧠  Autoencoder",
        "🔗  Combined Analysis",
        "📋  Sample Report",
    ]
)

# ── Tab 1: t-SNE ──────────────────────────────────────────────
with tab1:
    col_left, col_right = st.columns([3, 1])

    with col_left:
        fig = go.Figure()

        for cls in sorted(set(cancer_types)):
            color = CLASS_COLORS.get(cls, C2)
            mask = cancer_types == cls
            if mask.sum() == 0:
                continue

            fig.add_trace(
                go.Scatter(
                    x=X_tsne[mask, 0],
                    y=X_tsne[mask, 1],
                    mode="markers",
                    name=cls,
                    marker=dict(
                        color=color,
                        size=7,
                        opacity=0.75,
                        line=dict(width=0.5, color="white"),
                    ),
                    hovertemplate=f"<b>{cls}</b><br>t-SNE 1: %{{x:.2f}}<br>t-SNE 2: %{{y:.2f}}<extra></extra>",
                )
            )

        anom_mask = combined_preds == 1
        fig.add_trace(
            go.Scatter(
                x=X_tsne[anom_mask, 0],
                y=X_tsne[anom_mask, 1],
                mode="markers",
                name="⚠ Anomaly",
                marker=dict(
                    color="rgba(0,0,0,0)",
                    size=14,
                    line=dict(width=1.8, color=ALERT),
                ),
                hovertemplate="<b>ANOMALY</b><br>Score: %{customdata:.3f}<extra></extra>",
                customdata=combined_score[anom_mask],
            )
        )

        fig_layout(fig, "t-SNE Projection — Cancer Type Clusters", 520)
        fig.update_layout(legend=dict(orientation="v"))
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown(
            "<div class='section-header'>Class Distribution</div>",
            unsafe_allow_html=True,
        )
        class_counts = pd.Series(cancer_types).value_counts()

        fig_pie = go.Figure(
            go.Pie(
                labels=class_counts.index,
                values=class_counts.values,
                hole=0.55,
                marker=dict(
                    colors=[CLASS_COLORS.get(c, C2) for c in class_counts.index],
                    line=dict(color="white", width=2),
                ),
                textinfo="label+percent",
                textfont=dict(size=11, family="DM Sans"),
            )
        )
        fig_pie.update_layout(
            paper_bgcolor="white",
            showlegend=False,
            height=260,
            margin=dict(l=10, r=10, t=10, b=10),
            annotations=[
                dict(
                    text=f"<b>{n_total}</b><br>samples",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=13, family="DM Sans", color=C1),
                )
            ],
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown(
            "<div class='section-header' style='margin-top:12px;'>Anomaly Rate / Class</div>",
            unsafe_allow_html=True,
        )

        rows = []
        for cls in sorted(set(cancer_types)):
            mask = cancer_types == cls
            rate = combined_preds[mask].sum() / mask.sum() * 100
            rows.append(
                {
                    "Type": cls,
                    "Total": int(mask.sum()),
                    "Anomalies": int(combined_preds[mask].sum()),
                    "Rate %": f"{rate:.1f}",
                }
            )

        df_rate = pd.DataFrame(rows)
        st.dataframe(df_rate, hide_index=True, use_container_width=True)


# ── Tab 2: Isolation Forest ───────────────────────────────────
with tab2:
    st.markdown(
        """
    <div class='info-card'>
        <b>Isolation Forest</b> works by randomly selecting a feature and a split point.
        Anomalous samples are isolated faster (shorter average path length) because they
        occupy sparse, low-density regions of the feature space.
    </div>
    """,
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns([2, 1])

    with col_a:
        fig = go.Figure()
        nm = iso_preds == 1
        am = iso_preds == -1

        fig.add_trace(
            go.Scatter(
                x=X_tsne[nm, 0],
                y=X_tsne[nm, 1],
                mode="markers",
                name="Normal",
                marker=dict(
                    color=C3,
                    size=6,
                    opacity=0.55,
                    line=dict(width=0, color="white"),
                ),
            )
        )

        fig.add_trace(
            go.Scatter(
                x=X_tsne[am, 0],
                y=X_tsne[am, 1],
                mode="markers",
                name=f"Anomaly ({am.sum()})",
                marker=dict(
                    color=ALERT,
                    size=11,
                    opacity=0.92,
                    symbol="circle",
                    line=dict(width=1.5, color="white"),
                ),
                hovertemplate="<b>Anomaly</b><br>IF score: %{customdata:.4f}<extra></extra>",
                customdata=iso_scores[am],
            )
        )

        fig_layout(fig, "Isolation Forest — Detected Anomalies (t-SNE space)", 460)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown(
            "<div class='section-header'>Decision Score Distribution</div>",
            unsafe_allow_html=True,
        )

        fig_hist = go.Figure()
        fig_hist.add_trace(
            go.Histogram(
                x=iso_scores[nm],
                name="Normal",
                marker_color=C3,
                opacity=0.75,
                nbinsx=35,
                histnorm="probability density",
            )
        )
        fig_hist.add_trace(
            go.Histogram(
                x=iso_scores[am],
                name="Anomaly",
                marker_color=ALERT,
                opacity=0.85,
                nbinsx=20,
                histnorm="probability density",
            )
        )

        fig_layout(fig_hist, "IF Score Distribution", 280)
        fig_hist.update_layout(barmode="overlay", margin=dict(l=30, r=10, t=40, b=30))
        st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown(
            "<div class='section-header' style='margin-top:6px;'>IF Anomalies per Class</div>",
            unsafe_allow_html=True,
        )

        data_if = pd.DataFrame(
            {"Class": cancer_types, "Anomaly": (iso_preds == -1).astype(int)}
        )
        grp = data_if.groupby(["Class", "Anomaly"]).size().unstack(fill_value=0)

        fig_bar = go.Figure()
        if 0 in grp.columns:
            fig_bar.add_trace(
                go.Bar(x=grp.index, y=grp[0], name="Normal", marker_color=C3, opacity=0.8)
            )
        if 1 in grp.columns:
            fig_bar.add_trace(
                go.Bar(
                    x=grp.index,
                    y=grp[1],
                    name="Anomaly",
                    marker_color=ALERT,
                    opacity=0.9,
                )
            )

        fig_layout(fig_bar, "", 210)
        fig_bar.update_layout(
            barmode="stack", margin=dict(l=30, r=10, t=10, b=30), showlegend=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)


# ── Tab 3: Autoencoder ────────────────────────────────────────
with tab3:
    st.markdown(
        """
    <div class='info-card'>
        <b>Autoencoder</b>: a neural network trained to compress then reconstruct its input.
        Normal samples are well-reconstructed (low MSE). Anomalies have patterns the network
        never learned → high reconstruction error → flagged as outliers.
    </div>
    """,
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns([2, 1])

    with col_a:
        fig = go.Figure(
            go.Scatter(
                x=X_tsne[:, 0],
                y=X_tsne[:, 1],
                mode="markers",
                marker=dict(
                    color=recon_errors,
                    colorscale=[[0, C4], [0.4, C3], [0.75, C1], [1, ALERT]],
                    size=7,
                    opacity=0.85,
                    colorbar=dict(
                        title=dict(
                            text="MSE",
                            font=dict(family="DM Sans")
                        ),
                        tickfont=dict(size=10, family="DM Sans"),
                    ),
                    line=dict(width=0),
                ),
                hovertemplate="Reconstruction Error: %{marker.color:.5f}<extra></extra>",
            )
        )

        ae_anom = ae_preds == 1
        fig.add_trace(
            go.Scatter(
                x=X_tsne[ae_anom, 0],
                y=X_tsne[ae_anom, 1],
                mode="markers",
                name=f"AE Anomaly ({ae_anom.sum()})",
                marker=dict(
                    color="rgba(0,0,0,0)",
                    size=14,
                    line=dict(width=2, color=ALERT),
                ),
            )
        )

        fig_layout(fig, "Autoencoder Reconstruction Error Map (t-SNE)", 460)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown(
            "<div class='section-header'>Training Loss Curve</div>",
            unsafe_allow_html=True,
        )

        fig_loss = go.Figure(
            go.Scatter(
                y=loss_hist,
                mode="lines",
                line=dict(color=C1, width=2.5),
                fill="tozeroy",
                fillcolor=C3_RGBA_15,
            )
        )
        fig_layout(fig_loss, "", 230)
        fig_loss.update_layout(
            margin=dict(l=30, r=10, t=10, b=30),
            xaxis_title="Epoch",
            yaxis_title="MSE Loss",
        )
        st.plotly_chart(fig_loss, use_container_width=True)

        st.markdown(
            "<div class='section-header' style='margin-top:6px;'>Reconstruction Error Distribution</div>",
            unsafe_allow_html=True,
        )

        fig_errhist = go.Figure()
        fig_errhist.add_trace(
            go.Histogram(
                x=recon_errors[ae_preds == 0],
                name="Normal",
                marker_color=C3,
                opacity=0.75,
                nbinsx=40,
                histnorm="probability density",
            )
        )
        fig_errhist.add_trace(
            go.Histogram(
                x=recon_errors[ae_preds == 1],
                name="Anomaly",
                marker_color=ALERT,
                opacity=0.85,
                nbinsx=20,
                histnorm="probability density",
            )
        )
        fig_errhist.add_vline(
            x=ae_thresh,
            line_dash="dash",
            line_color=WARN,
            line_width=2,
            annotation_text=f"p{ae_threshold}",
            annotation_font_color=WARN,
        )

        fig_layout(fig_errhist, "", 220)
        fig_errhist.update_layout(
            barmode="overlay", margin=dict(l=30, r=10, t=10, b=30), showlegend=False
        )
        st.plotly_chart(fig_errhist, use_container_width=True)


# ── Tab 4: Combined ───────────────────────────────────────────
with tab4:
    col_a, col_b = st.columns([3, 2])

    with col_a:
        fig = go.Figure(
            go.Scatter(
                x=X_tsne[:, 0],
                y=X_tsne[:, 1],
                mode="markers",
                marker=dict(
                    color=combined_score,
                    colorscale=[
                        [0, C4],
                        [0.35, C3],
                        [0.65, C1],
                        [0.85, WARN],
                        [1, ALERT],
                    ],
                    size=8,
                    opacity=0.85,
                    colorbar=dict(
                        title=dict(
                            text="Anomaly Score",
                            font=dict(family="DM Sans")
                        ),
                        tickfont=dict(size=10, family="DM Sans"),
                    ),
                    line=dict(width=0),
                ),
                hovertemplate="Combined score: %{marker.color:.3f}<extra></extra>",
                customdata=cancer_types,
            )
        )

        combined_anom = combined_preds == 1
        fig.add_trace(
            go.Scatter(
                x=X_tsne[combined_anom, 0],
                y=X_tsne[combined_anom, 1],
                mode="markers",
                name=f"Flagged ({combined_anom.sum()})",
                marker=dict(
                    color="rgba(0,0,0,0)",
                    size=16,
                    line=dict(width=2, color="white"),
                ),
                hovertemplate="<b>ANOMALY FLAGGED</b><extra></extra>",
            )
        )

        fig_layout(fig, "Combined Anomaly Score Map (IF ∪ Autoencoder)", 500)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown(
            "<div class='section-header'>Detector Comparison</div>",
            unsafe_allow_html=True,
        )

        only_if = int(((iso_preds == -1) & (ae_preds == 0)).sum())
        only_ae = int(((iso_preds == 1) & (ae_preds == 1)).sum())
        both_det = int(((iso_preds == -1) & (ae_preds == 1)).sum())

        fig_venn = go.Figure(
            go.Bar(
                x=["IF only", "AE only", "Both detectors"],
                y=[only_if, only_ae, both_det],
                marker_color=[C2, C3, ALERT],
                text=[only_if, only_ae, both_det],
                textposition="outside",
                textfont=dict(family="DM Sans", size=13, color=TEXT),
            )
        )

        fig_layout(fig_venn, "Anomalies by Detector", 260)
        fig_venn.update_layout(margin=dict(l=20, r=10, t=40, b=30), showlegend=False)
        st.plotly_chart(fig_venn, use_container_width=True)

        st.markdown(
            "<div class='section-header' style='margin-top:4px;'>Top 10 Most Anomalous Samples</div>",
            unsafe_allow_html=True,
        )

        top_idx = np.argsort(combined_score)[::-1][:10]
        df_top = pd.DataFrame(
            {
                "Sample": [data.index[i] for i in top_idx],
                "Class": [cancer_types[i] for i in top_idx],
                "Score": [f"{combined_score[i]:.3f}" for i in top_idx],
                "IF": ["⚠" if iso_preds[i] == -1 else "✓" for i in top_idx],
                "AE": ["⚠" if ae_preds[i] == 1 else "✓" for i in top_idx],
            }
        )
        st.dataframe(df_top, hide_index=True, use_container_width=True)


# ── Tab 5: Sample Report ──────────────────────────────────────
with tab5:
    st.markdown(
        "<div class='section-header'>Full Sample Report</div>",
        unsafe_allow_html=True,
    )

    df_report = pd.DataFrame(
        {
            "Sample": data.index,
            "Class": cancer_types,
            "IF Score": np.round(iso_scores, 5),
            "IF Anomaly": ["⚠ Anomaly" if p == -1 else "Normal" for p in iso_preds],
            "Recon Error": np.round(recon_errors, 6),
            "AE Anomaly": ["⚠ Anomaly" if p == 1 else "Normal" for p in ae_preds],
            "Combined Score": np.round(combined_score, 4),
            "Final Flag": [
                "⚠ ANOMALY" if p == 1 else "Normal" for p in combined_preds
            ],
        }
    )

    fc1, fc2, fc3 = st.columns([1, 1, 2])
    with fc1:
        filter_class = st.multiselect(
            "Filter by class",
            options=sorted(set(cancer_types)),
            default=sorted(set(cancer_types)),
        )
    with fc2:
        filter_flag = st.selectbox(
            "Filter by status", ["All", "Anomaly only", "Normal only"]
        )
    with fc3:
        st.markdown("")

    df_show = df_report[df_report["Class"].isin(filter_class)]

    if filter_flag == "Anomaly only":
        df_show = df_show[df_show["Final Flag"] == "⚠ ANOMALY"]

    if filter_flag == "Normal only":
        df_show = df_show[df_show["Final Flag"] == "Normal"]

    st.markdown(
        f"<div style='font-size:0.82rem; color:{SUBTEXT}; margin-bottom:8px;'>"
        f"Showing {len(df_show)} of {n_total} samples</div>",
        unsafe_allow_html=True,
    )
    st.dataframe(df_show, hide_index=True, use_container_width=True, height=420)

    csv_out = df_report.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇  Download Full Report (CSV)",
        data=csv_out,
        file_name="bioanomaly_report.csv",
        mime="text/csv",
        use_container_width=True,
    )
