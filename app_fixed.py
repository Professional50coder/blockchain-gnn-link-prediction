import streamlit as st
import numpy as np
import pandas as pd
import pickle
import re
import logging
from datetime import datetime

import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from sklearn.metrics import roc_curve, auc

# -----------------------------------------------
# Logging setup
# -----------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# -----------------------------------------------
# Page configuration
# -----------------------------------------------
st.set_page_config(
    page_title="Blockchain GNN Analytics",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------
# Custom CSS — Premium Dark Blockchain Theme
# -----------------------------------------------
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Sora:wght@300;400;600;700;800&display=swap');

    /* ── Global ── */
    html, body, [class*="css"] {
        font-family: 'Sora', sans-serif;
    }

    /* Hide default Streamlit branding */
    #MainMenu, footer { visibility: hidden; }

    /* ── Main header ── */
    .main-header {
        font-family: 'Sora', sans-serif;
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 50%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    .sub-header {
        text-align: center;
        color: #94a3b8;
        font-size: 0.92rem;
        font-weight: 300;
        margin-bottom: 1.8rem;
        letter-spacing: 0.3px;
    }

    /* ── Cards ── */
    .metric-card {
        background: linear-gradient(145deg, rgba(0,210,255,0.08), rgba(58,123,213,0.06));
        padding: 1.1rem 1.3rem;
        border-radius: 12px;
        border: 1px solid rgba(0,210,255,0.18);
        box-shadow: 0 4px 20px rgba(0,0,0,0.25), inset 0 1px 0 rgba(255,255,255,0.05);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover { transform: translateY(-2px); box-shadow: 0 8px 30px rgba(0,210,255,0.12); }

    .fraud-alert {
        background: linear-gradient(145deg, rgba(239,68,68,0.1), rgba(185,28,28,0.06));
        padding: 1.1rem 1.3rem;
        border-radius: 12px;
        border: 1px solid rgba(239,68,68,0.25);
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    }
    .success-box {
        background: linear-gradient(145deg, rgba(34,197,94,0.1), rgba(21,128,61,0.06));
        padding: 1.1rem 1.3rem;
        border-radius: 12px;
        border: 1px solid rgba(34,197,94,0.25);
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    }
    .warning-box {
        background: linear-gradient(145deg, rgba(251,146,60,0.1), rgba(194,65,12,0.06));
        padding: 1.1rem 1.3rem;
        border-radius: 12px;
        border: 1px solid rgba(251,146,60,0.25);
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    }
    .info-box {
        background: linear-gradient(145deg, rgba(99,102,241,0.1), rgba(67,56,202,0.06));
        padding: 1.1rem 1.3rem;
        border-radius: 12px;
        border: 1px solid rgba(99,102,241,0.25);
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    }

    /* ── Risk badges ── */
    .risk-critical {
        background: linear-gradient(135deg, #7f1d1d, #b91c1c);
        color: #fecaca; padding: 4px 13px; border-radius: 20px;
        font-size: 0.76rem; font-weight: 700; letter-spacing: 0.5px;
        border: 1px solid rgba(239,68,68,0.4);
    }
    .risk-high {
        background: linear-gradient(135deg, #991b1b, #dc2626);
        color: #fee2e2; padding: 4px 13px; border-radius: 20px;
        font-size: 0.76rem; font-weight: 700; letter-spacing: 0.5px;
        border: 1px solid rgba(239,68,68,0.3);
    }
    .risk-medium {
        background: linear-gradient(135deg, #92400e, #d97706);
        color: #fef3c7; padding: 4px 13px; border-radius: 20px;
        font-size: 0.76rem; font-weight: 700; letter-spacing: 0.5px;
        border: 1px solid rgba(251,191,36,0.3);
    }
    .risk-low {
        background: linear-gradient(135deg, #14532d, #16a34a);
        color: #dcfce7; padding: 4px 13px; border-radius: 20px;
        font-size: 0.76rem; font-weight: 700; letter-spacing: 0.5px;
        border: 1px solid rgba(34,197,94,0.3);
    }

    /* ── Wallet address display ── */
    .wallet-address-full {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        background: rgba(0,210,255,0.07);
        border: 1px solid rgba(0,210,255,0.2);
        border-radius: 8px;
        padding: 0.55rem 0.9rem;
        color: #7dd3fc;
        word-break: break-all;
        letter-spacing: 0.3px;
        display: block;
        margin: 4px 0;
    }

    /* ── Validation feedback ── */
    .valid-address   { color: #4ade80; font-size: 0.82rem; font-weight: 600; }
    .invalid-address { color: #f87171; font-size: 0.82rem; font-weight: 600; }

    /* ── History pills ── */
    .history-pill {
        display: inline-block;
        background: rgba(0,210,255,0.1);
        color: #7dd3fc;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.73rem;
        font-family: 'JetBrains Mono', monospace;
        margin: 2px;
        border: 1px solid rgba(0,210,255,0.2);
        cursor: pointer;
    }

    /* ── Section divider ── */
    .section-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(0,210,255,0.3), transparent);
        margin: 1.5rem 0;
        border: none;
    }

    /* ── Stat badge for sidebar ── */
    .sidebar-stat {
        background: rgba(255,255,255,0.04);
        border-radius: 8px;
        padding: 0.5rem 0.8rem;
        margin: 4px 0;
        font-size: 0.88rem;
        border: 1px solid rgba(255,255,255,0.07);
    }

    /* ── Footer ── */
    .footer-text {
        text-align: center;
        color: #475569;
        font-size: 0.80rem;
        padding: 2rem 0 1rem;
        letter-spacing: 0.3px;
    }
    .footer-text strong { color: #64748b; }

    /* ── Dataframe: make wallet column wrap ── */
    .stDataFrame td { white-space: pre-wrap !important; word-break: break-all !important; }
    .stDataFrame th { font-family: 'Sora', sans-serif !important; font-weight: 600 !important; }

    /* ── Glowing metric numbers ── */
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-family: 'Sora', sans-serif;
        font-weight: 700;
    }

    /* ── Tab styling ── */
    .stTabs [data-baseweb="tab"] {
        font-family: 'Sora', sans-serif;
        font-weight: 600;
        font-size: 0.88rem;
    }

    /* ── Scrollable code block for addresses ── */
    .address-block {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        color: #93c5fd;
        background: rgba(30,41,59,0.8);
        border: 1px solid rgba(99,102,241,0.25);
        border-radius: 8px;
        padding: 0.7rem 1rem;
        word-break: break-all;
        line-height: 1.6;
    }
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------
# Session-state initialisation
# -----------------------------------------------
if "search_history" not in st.session_state:
    st.session_state.search_history = []


# -----------------------------------------------
# Helpers — validation
# -----------------------------------------------
def is_valid_eth_address(addr: str) -> bool:
    """Return True if addr looks like a valid Ethereum address."""
    return bool(re.fullmatch(r"0x[0-9a-fA-F]{40}", addr.strip()))


def normalise_fraud_score(score: float, score_min: float, score_max: float) -> float:
    """Convert Isolation-Forest score (lower = worse) to 0–100 (higher = worse)."""
    if score_max == score_min:
        return 50.0
    normalised = (score_max - score) / (score_max - score_min) * 100
    return float(np.clip(normalised, 0, 100))


def risk_label(risk_score: float) -> str:
    """Return a risk label from a 0–100 risk score."""
    if risk_score >= 80:
        return "Critical"
    elif risk_score >= 60:
        return "High"
    elif risk_score >= 35:
        return "Medium"
    return "Low"


def risk_badge(risk_score: float) -> str:
    label = risk_label(risk_score)
    css = {"Critical": "risk-critical", "High": "risk-high", "Medium": "risk-medium", "Low": "risk-low"}
    return f'<span class="{css[label]}">{label}</span>'


# -----------------------------------------------
# Helpers — MLP-style link probability
# -----------------------------------------------
def compute_link_probability(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    """
    Improved link-probability estimate using three complementary signals:
      1. Dot-product similarity (original)
      2. Cosine similarity
      3. L2 distance (inverted)
    All three are combined and passed through a sigmoid.
    """
    # 1. Dot product (clipped to avoid overflow)
    dot = float(np.dot(emb_a, emb_b))
    dot = np.clip(dot, -500.0, 500.0)

    # 2. Cosine similarity
    norm_a = np.linalg.norm(emb_a) + 1e-9
    norm_b = np.linalg.norm(emb_b) + 1e-9
    cosine = float(np.dot(emb_a, emb_b) / (norm_a * norm_b))

    # 3. L2 distance (inverted, scaled)
    l2 = float(np.linalg.norm(emb_a - emb_b))
    l2_sim = 1.0 / (1.0 + l2)

    # Weighted combination — tuned heuristically
    combined = 0.5 * dot + 0.3 * cosine * abs(dot) + 0.2 * l2_sim * abs(dot)
    combined = np.clip(combined, -500.0, 500.0)
    return float(1.0 / (1.0 + np.exp(-combined)))


# -----------------------------------------------
# Helpers — graph statistics
# -----------------------------------------------
@st.cache_data
def create_graph_statistics(edges_df: pd.DataFrame) -> dict:
    stats = {
        "total_transactions": len(edges_df),
        "unique_senders": edges_df["from_id"].nunique(),
        "unique_receivers": edges_df["to_id"].nunique(),
        "avg_out_degree": edges_df.groupby("from_id").size().mean(),
        "avg_in_degree": edges_df.groupby("to_id").size().mean(),
        "max_out_degree": int(edges_df.groupby("from_id").size().max()),
        "max_in_degree": int(edges_df.groupby("to_id").size().max()),
    }
    return stats


# -----------------------------------------------
# Data loading with graceful fallback
# -----------------------------------------------
@st.cache_data
def load_data():
    """Load all pre-trained model outputs.  Returns None values on missing files."""
    required_files = {
        "node_embeddings.npy": "NumPy array of GNN node embeddings",
        "edges.csv": "Transaction edge list (from_id, to_id)",
        "fraudulent_wallets.csv": "Fraud detection results",
        "label_encoder.pkl": "LabelEncoder mapping addresses ↔ IDs",
    }

    missing = []
    for fname, desc in required_files.items():
        try:
            open(fname)
        except FileNotFoundError:
            missing.append(f"• **{fname}** — {desc}")

    if missing:
        st.error("### ❌ Required data files are missing")
        st.markdown("\n".join(missing))
        st.info(
            "Run `python generate_sample_data.py` to create demo data, "
            "or export real data from your Colab notebook using `save_dashboard_data.py`."
        )
        st.stop()

    try:
        logger.info("Loading node_embeddings.npy …")
        embeddings = np.load("node_embeddings.npy")

        logger.info("Loading edges.csv …")
        edges = pd.read_csv("edges.csv")

        logger.info("Loading fraudulent_wallets.csv …")
        fraud_df = pd.read_csv("fraudulent_wallets.csv")

        logger.info("Loading label_encoder.pkl …")
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with open("label_encoder.pkl", "rb") as f:
                le = pickle.load(f)

        # Optional files
        loss_history = None
        try:
            loss_history = np.load("loss_history.npy")
            logger.info("Loaded loss_history.npy")
        except FileNotFoundError:
            logger.warning("loss_history.npy not found — loss curve will be unavailable.")

        fpr = tpr = roc_auc_val = None
        try:
            roc_data = np.load("roc_data.npz")
            fpr, tpr, roc_auc_val = roc_data["fpr"], roc_data["tpr"], float(roc_data["auc"])
            logger.info(f"Loaded ROC data — AUC = {roc_auc_val:.4f}")
        except FileNotFoundError:
            logger.warning("roc_data.npz not found — ROC curve will be unavailable.")

        # Pre-compute normalised risk scores and add to fraud_df
        s_min = float(fraud_df["fraud_score"].min())
        s_max = float(fraud_df["fraud_score"].max())
        fraud_df["risk_score"] = fraud_df["fraud_score"].apply(
            lambda s: round(normalise_fraud_score(s, s_min, s_max), 1)
        )
        fraud_df["risk_level"] = fraud_df["risk_score"].apply(risk_label)

        return embeddings, edges, fraud_df, le, loss_history, (fpr, tpr, roc_auc_val)

    except Exception as exc:
        logger.exception("Fatal error while loading data")
        st.error(f"Error loading data: {exc}")
        st.info("Check the terminal for a detailed traceback.")
        st.stop()


# -----------------------------------------------
# Visualisation helpers
# -----------------------------------------------
PLOTLY_TEMPLATE = "plotly_dark"
PRIMARY_COLOR  = "#00d2ff"
DANGER_COLOR   = "#ef4444"
SUCCESS_COLOR  = "#22c55e"
ACCENT_COLOR   = "#8b5cf6"


def plot_loss_curve(loss_history: np.ndarray) -> go.Figure:
    fig = go.Figure()
    epochs = list(range(1, len(loss_history) + 1))
    fig.add_trace(
        go.Scatter(
            x=epochs, y=loss_history,
            mode="lines",
            name="Training Loss",
            line=dict(color=PRIMARY_COLOR, width=2.5),
            fill="tozeroy",
            fillcolor="rgba(0,210,255,0.07)",
        )
    )
    fig.add_annotation(
        x=epochs[-1], y=float(loss_history[-1]),
        text=f"Final: {loss_history[-1]:.4f}",
        showarrow=True, arrowhead=2, ax=-60, ay=-30,
        font=dict(color=PRIMARY_COLOR, size=11),
    )
    fig.update_layout(
        title="GNN Training Loss Curve",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        hovermode="x unified",
        template=PLOTLY_TEMPLATE,
        height=400,
    )
    return fig


def plot_roc_curve(fpr, tpr, auc_score: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fpr, y=tpr,
            mode="lines",
            name=f"ROC Curve (AUC = {auc_score:.3f})",
            line=dict(color=PRIMARY_COLOR, width=2.5),
            fill="tozeroy",
            fillcolor="rgba(0,210,255,0.08)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines",
            name="Random Classifier",
            line=dict(color="gray", width=1.5, dash="dash"),
        )
    )
    fig.update_layout(
        title="ROC Curve — Transaction Link Prediction",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        hovermode="closest",
        template=PLOTLY_TEMPLATE,
        height=400,
    )
    return fig


def plot_fraud_distribution(fraud_df: pd.DataFrame) -> go.Figure:
    """Plot risk-score distribution (0–100, higher = more suspicious)."""
    fig = px.histogram(
        fraud_df, x="risk_score", nbins=50,
        title="Risk Score Distribution (0 = safe, 100 = critical)",
        labels={"risk_score": "Risk Score (0–100)", "count": "Number of Wallets"},
        color_discrete_sequence=[DANGER_COLOR],
    )
    # Add vertical lines for thresholds
    for thresh, label, color in [(35, "Medium", "#f59e0b"), (60, "High", "#ef4444"), (80, "Critical", "#dc2626")]:
        fig.add_vline(x=thresh, line_dash="dot", line_color=color,
                      annotation_text=label, annotation_position="top right",
                      annotation_font_color=color)
    fig.update_layout(template=PLOTLY_TEMPLATE, height=380)
    return fig


def create_network_subgraph(
    edges_df: pd.DataFrame,
    wallet_id: int,
    le,
    fraud_df: pd.DataFrame,
    max_connections: int = 25,
) -> go.Figure | None:
    related = edges_df[
        (edges_df["from_id"] == wallet_id) | (edges_df["to_id"] == wallet_id)
    ].head(max_connections)

    if len(related) == 0:
        return None

    fraudulent_ids = set(fraud_df["wallet_id"].tolist()) if "wallet_id" in fraud_df.columns else set()

    G = nx.DiGraph()
    for _, row in related.iterrows():
        G.add_edge(int(row["from_id"]), int(row["to_id"]))

    pos = nx.spring_layout(G, seed=42, k=1.5)

    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_x += [x0, x1, None]; edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color="rgba(148,163,184,0.35)"),
        hoverinfo="none", mode="lines",
    )

    node_x, node_y, node_text, node_colors, node_sizes = [], [], [], [], []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x); node_y.append(y)

        try:
            addr = le.inverse_transform([node])[0]
            label = addr  # Full address shown on hover
        except Exception:
            label = f"Wallet {node}"

        is_fraud = node in fraudulent_ids
        is_center = node == wallet_id

        node_text.append(label + (" 🚨" if is_fraud else ""))
        node_colors.append(
            "#dc2626" if (is_center and is_fraud) else
            DANGER_COLOR if is_fraud else
            "#3b82f6" if is_center else
            PRIMARY_COLOR
        )
        node_sizes.append(30 if is_center else (20 if is_fraud else 15))

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        hoverinfo="text",
        hovertext=node_text,
        text=[t[:12] + "…" if len(t) > 14 else t for t in node_text],  # short label on graph
        textposition="top center",
        textfont=dict(size=9, color="#94a3b8"),
        marker=dict(
            size=node_sizes, color=node_colors,
            line=dict(width=1.5, color="rgba(255,255,255,0.3)"),
            opacity=0.9,
        ),
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=dict(text=f"Transaction Network — Wallet {wallet_id}", font=dict(size=14, color="#94a3b8")),
        showlegend=False,
        hovermode="closest",
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(15,23,42,0.0)",
        plot_bgcolor="rgba(15,23,42,0.0)",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=540,
        margin=dict(t=50, b=20, l=20, r=20),
    )
    return fig


# -----------------------------------------------
# Load data
# -----------------------------------------------
embeddings, edges, fraud_df, le, loss_history, (fpr, tpr, roc_auc_val) = load_data()
stats = create_graph_statistics(edges)
WALLET_SET = set(le.classes_)

# Fraud ID set for fast lookup
FRAUD_IDS: set = set(fraud_df["wallet_id"].tolist()) if "wallet_id" in fraud_df.columns else set()


# -----------------------------------------------
# Sidebar
# -----------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 0.5rem 0 1rem;">
            <div style="font-size:2.8rem;">🔗</div>
            <div style="font-family:'Sora',sans-serif; font-size:1.05rem; font-weight:700;
                        background:linear-gradient(135deg,#00d2ff,#8b5cf6);
                        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                        background-clip:text; margin-top:4px;">
                GNN Analytics
            </div>
            <div style="color:#475569; font-size:0.72rem; margin-top:2px; letter-spacing:1px;">
                BLOCKCHAIN INTELLIGENCE
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<hr style="border-color:rgba(255,255,255,0.07); margin:0.5rem 0 1rem;">', unsafe_allow_html=True)

    section = st.radio(
        "Navigate",
        [
            "🏠 Overview",
            "📊 Graph Analytics",
            "🔮 Link Prediction",
            "🚨 Fraud Detection",
            "📈 Model Performance",
            "🌐 Network Visualization",
        ],
    )

    st.markdown('<hr style="border-color:rgba(255,255,255,0.07); margin:1rem 0;">', unsafe_allow_html=True)
    st.markdown("**📌 Quick Stats**")
    st.metric("Total Wallets", f"{embeddings.shape[0]:,}")
    st.metric("Total Transactions", f"{stats['total_transactions']:,}")
    st.metric("Suspicious Wallets", f"{len(fraud_df):,}")
    if roc_auc_val is not None:
        st.metric("ROC-AUC Score", f"{roc_auc_val:.4f}")

    st.markdown('<hr style="border-color:rgba(255,255,255,0.07); margin:1rem 0;">', unsafe_allow_html=True)
    st.info(
        "**Project:** Transaction Link Prediction in Blockchain using GNN\n\n"
        "**Model:** GraphSAGE\n\n"
        "**Dataset:** Ethereum Mainnet"
    )


# -----------------------------------------------
# ===  SECTION: Overview  =======================
# -----------------------------------------------
if "🏠 Overview" in section:
    st.markdown(
        '<p class="main-header">🔗 Blockchain GNN Transaction Intelligence</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sub-header">GraphSAGE-powered link prediction & anomaly detection on Ethereum Mainnet</p>',
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Wallet Addresses", f"{embeddings.shape[0]:,}")
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Transactions", f"{stats['total_transactions']:,}")
        st.markdown("</div>", unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Avg Out-Degree", f"{stats['avg_out_degree']:.2f}")
        st.markdown("</div>", unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="fraud-alert">', unsafe_allow_html=True)
        st.metric("Suspicious Wallets", f"{len(fraud_df):,}")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### 📖 Project Overview")
        st.markdown(
            """
This dashboard presents a **Graph Neural Network (GNN)** system for analysing Ethereum blockchain
transactions. It combines a **GraphSAGE** model for link prediction with **Isolation Forest**
anomaly detection to identify suspicious wallet behaviour.

- **Transaction Link Prediction** — predict the probability of a future transaction between any two wallet addresses
- **Fraud Detection** — identify anomalous wallets using unsupervised learning on GNN embeddings
- **Network Visualisation** — explore the transaction graph around any wallet
- **Model Performance** — ROC-AUC analysis, training loss, and architecture details
"""
        )
    with col2:
        st.markdown("### 🎯 Key Features")
        st.markdown(
            """
✅ Real Ethereum mainnet data  
✅ GraphSAGE GNN architecture  
✅ Improved multi-signal link predictor  
✅ Risk score (0–100 scale)  
✅ Input validation for wallet addresses  
✅ Session search history  
✅ Fraud co-occurrence in network graphs  
✅ Graceful error handling & logging  
"""
        )

    st.markdown("---")
    with st.expander("📚 Methodology Details"):
        st.markdown(
            """
#### 1. Data Collection
- Source: Google BigQuery Ethereum Public Dataset
- Transactions: sender, receiver, value, gas, timestamp

#### 2. Graph Construction
- **Nodes:** wallet addresses
- **Edges:** directed transaction relationships
- **Features:** in-degree / out-degree statistics

#### 3. Model Architecture
- **GraphSAGE** — 2-layer Graph Convolutional Network
- **Embedding size:** 64 dimensions
- **Improved decoder:** weighted combination of dot-product, cosine similarity, and L2 distance

#### 4. Training
- Train / Val / Test: 70 / 15 / 15
- Loss: Binary Cross-Entropy
- Optimiser: Adam (lr=0.01)
- Negative sampling: dynamic

#### 5. Fraud Detection
- **Algorithm:** Isolation Forest (unsupervised)
- **Input:** 64-dim node embeddings
- **Output:** Risk score **0–100** (higher = more suspicious)
- **Thresholds:** Low < 35 · Medium 35–60 · High 60–80 · Critical ≥ 80
"""
        )


# -----------------------------------------------
# ===  SECTION: Graph Analytics  ================
# -----------------------------------------------
elif "📊 Graph Analytics" in section:
    st.markdown(
        '<p class="main-header">📊 Graph Analytics & Statistics</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sub-header">Degree distribution, transaction explorer, and network topology metrics</p>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### Network Overview")
        st.metric("Total Nodes (Wallets)", f"{embeddings.shape[0]:,}")
        st.metric("Total Edges (Transactions)", f"{stats['total_transactions']:,}")
        density = stats["total_transactions"] / (embeddings.shape[0] ** 2)
        st.metric("Graph Density", f"{density:.2e}")
    with col2:
        st.markdown("### Degree Statistics")
        st.metric("Avg Out-Degree", f"{stats['avg_out_degree']:.2f}")
        st.metric("Avg In-Degree", f"{stats['avg_in_degree']:.2f}")
        st.metric("Max Out-Degree", f"{stats['max_out_degree']:,}")
    with col3:
        st.markdown("### Node Info")
        st.metric("Unique Senders", f"{stats['unique_senders']:,}")
        st.metric("Unique Receivers", f"{stats['unique_receivers']:,}")
        st.metric("Embedding Dimensions", f"{embeddings.shape[1]}")

    st.markdown("---")

    # Sample transactions with search
    st.markdown("### 📋 Transaction Explorer")
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        search_wallet = st.text_input("🔍 Search by Wallet ID (numeric)", "")
    with col2:
        num_rows = st.selectbox("Rows", [10, 25, 50, 100], index=0)
    with col3:
        sort_col = st.selectbox("Sort by", edges.columns.tolist(), index=0)

    display_edges = edges.sort_values(sort_col)

    if search_wallet:
        try:
            wid = int(search_wallet)
            display_edges = display_edges[
                (display_edges["from_id"] == wid) | (display_edges["to_id"] == wid)
            ]
            st.info(f"Found **{len(display_edges):,}** transactions for wallet `{wid}`")
        except ValueError:
            st.error("Please enter a numeric wallet ID.")

    # Enrich display_edges with full wallet addresses if label encoder available
    display_edges_show = display_edges.head(num_rows).copy()
    try:
        display_edges_show["from_address"] = le.inverse_transform(display_edges_show["from_id"].astype(int))
        display_edges_show["to_address"]   = le.inverse_transform(display_edges_show["to_id"].astype(int))
    except Exception:
        pass

    st.dataframe(
        display_edges_show,
        use_container_width=True,
        column_config={
            "from_address": st.column_config.TextColumn("From Address", width="large"),
            "to_address":   st.column_config.TextColumn("To Address",   width="large"),
        },
    )

    st.markdown("---")
    st.markdown("### 📊 Degree Distribution")
    col1, col2 = st.columns(2)
    with col1:
        out_deg = edges.groupby("from_id").size()
        fig = px.histogram(out_deg, nbins=50, title="Out-Degree Distribution",
                           labels={"value": "Out-Degree"}, color_discrete_sequence=[PRIMARY_COLOR])
        fig.update_layout(template=PLOTLY_TEMPLATE, height=340)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        in_deg = edges.groupby("to_id").size()
        fig = px.histogram(in_deg, nbins=50, title="In-Degree Distribution",
                           labels={"value": "In-Degree"}, color_discrete_sequence=["#ff7f0e"])
        fig.update_layout(template=PLOTLY_TEMPLATE, height=340)
        st.plotly_chart(fig, use_container_width=True)


# -----------------------------------------------
# ===  SECTION: Link Prediction  ================
# -----------------------------------------------
elif "🔮 Link Prediction" in section:
    st.markdown(
        '<p class="main-header">🔮 Transaction Link Prediction</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sub-header">Predict future transaction probability · Dot-product + Cosine similarity + L2 distance signals</p>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # Search history quick-select
    if st.session_state.search_history:
        st.markdown("**Recent lookups:**")
        history_html = " ".join(
            f'<span class="history-pill" title="{a}">{a[:10]}…{a[-6:]}</span>'
            for a in st.session_state.search_history[-6:]
        )
        st.markdown(history_html, unsafe_allow_html=True)
        st.markdown("")

    tab1, tab2 = st.tabs(["🔤 Manual Input", "🎲 Random Prediction"])

    # --- Manual Input ---
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            wallet_a = st.text_input(
                "Sender Wallet Address",
                placeholder="0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb",
                help="Enter a valid Ethereum wallet address (0x + 40 hex chars)",
            )
            if wallet_a:
                if not is_valid_eth_address(wallet_a):
                    st.markdown(
                        '<p class="invalid-address">⚠ Invalid format — must be 0x followed by 40 hex characters</p>',
                        unsafe_allow_html=True,
                    )
                elif wallet_a not in WALLET_SET:
                    st.markdown(
                        '<p class="invalid-address">⚠ Address not found in this dataset</p>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<p class="valid-address">✔ Valid address found in dataset</p>',
                        unsafe_allow_html=True,
                    )

        with col2:
            wallet_b = st.text_input(
                "Receiver Wallet Address",
                placeholder="0x3f5CE5FBFe3E9af3971dD833D26bA9b5C936f0bE",
                help="Enter a valid Ethereum wallet address (0x + 40 hex chars)",
            )
            if wallet_b:
                if not is_valid_eth_address(wallet_b):
                    st.markdown(
                        '<p class="invalid-address">⚠ Invalid format — must be 0x followed by 40 hex characters</p>',
                        unsafe_allow_html=True,
                    )
                elif wallet_b not in WALLET_SET:
                    st.markdown(
                        '<p class="invalid-address">⚠ Address not found in this dataset</p>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<p class="valid-address">✔ Valid address found in dataset</p>',
                        unsafe_allow_html=True,
                    )

        predict_btn = st.button(
            "🔮 Predict Transaction Probability", type="primary", use_container_width=True
        )

        if predict_btn:
            errors = []
            if not wallet_a:
                errors.append("Sender address is empty.")
            elif not is_valid_eth_address(wallet_a):
                errors.append("Sender address has invalid format.")
            elif wallet_a not in WALLET_SET:
                errors.append("Sender address not found in dataset.")

            if not wallet_b:
                errors.append("Receiver address is empty.")
            elif not is_valid_eth_address(wallet_b):
                errors.append("Receiver address has invalid format.")
            elif wallet_b not in WALLET_SET:
                errors.append("Receiver address not found in dataset.")

            if wallet_a and wallet_b and wallet_a == wallet_b:
                errors.append("Sender and receiver must be different wallets.")

            if errors:
                for err in errors:
                    st.error(f"❌ {err}")
            else:
                id_a = int(le.transform([wallet_a])[0])
                id_b = int(le.transform([wallet_b])[0])
                probability = compute_link_probability(embeddings[id_a], embeddings[id_b])

                # Save to history
                for addr in [wallet_a, wallet_b]:
                    if addr not in st.session_state.search_history:
                        st.session_state.search_history.append(addr)

                st.markdown("---")
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown('<div class="success-box">', unsafe_allow_html=True)
                    st.markdown("### 📊 Prediction Result")
                    st.metric(
                        "Transaction Probability",
                        f"{probability:.4f}",
                        delta=f"{probability*100:.1f}%",
                    )
                    if probability > 0.7:
                        st.success("🟢 **High likelihood** of future transaction")
                    elif probability > 0.4:
                        st.warning("🟡 **Moderate likelihood** of future transaction")
                    else:
                        st.info("🔵 **Low likelihood** of future transaction")

                    st.markdown("**Sender:**")
                    st.markdown(f'<span class="wallet-address-full">{wallet_a}</span>', unsafe_allow_html=True)
                    st.markdown("**Receiver:**")
                    st.markdown(f'<span class="wallet-address-full">{wallet_b}</span>', unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("---")
                with st.expander("📈 Detailed Wallet Analysis"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Sender Wallet**")
                        sent = edges[edges["from_id"] == id_a]
                        st.write(f"- Transactions sent: **{len(sent)}**")
                        st.write(f"- Unique receivers: **{sent['to_id'].nunique()}**")
                        is_fraud_a = id_a in FRAUD_IDS
                        st.write(f"- Fraud flag: {'🚨 Flagged' if is_fraud_a else '✅ Clean'}")
                    with col2:
                        st.markdown("**Receiver Wallet**")
                        recv = edges[edges["to_id"] == id_b]
                        st.write(f"- Transactions received: **{len(recv)}**")
                        st.write(f"- Unique senders: **{recv['from_id'].nunique()}**")
                        is_fraud_b = id_b in FRAUD_IDS
                        st.write(f"- Fraud flag: {'🚨 Flagged' if is_fraud_b else '✅ Clean'}")

                    existing = edges[(edges["from_id"] == id_a) & (edges["to_id"] == id_b)]
                    if len(existing) > 0:
                        st.warning(
                            f"⚠️ **{len(existing)} historical transaction(s)** already exist between these wallets"
                        )
                    else:
                        st.info("ℹ️ No prior transactions between these wallets")

                # Export result
                result_df = pd.DataFrame(
                    [{"Sender": wallet_a, "Receiver": wallet_b,
                      "Probability": round(probability, 6),
                      "Likelihood": "High" if probability > 0.7 else ("Medium" if probability > 0.4 else "Low"),
                      "Timestamp": datetime.now().isoformat()}]
                )
                st.download_button(
                    "⬇️ Export Result (CSV)",
                    result_df.to_csv(index=False),
                    file_name="link_prediction_result.csv",
                    mime="text/csv",
                )

    # --- Random Predictions ---
    with tab2:
        st.markdown("### 🎲 Random Wallet Pair Prediction")
        num_predictions = st.slider("Number of random predictions", 5, 30, 10)

        if st.button("🎲 Generate Random Predictions", use_container_width=True):
            rows = []
            for _ in range(num_predictions):
                ia, ib = np.random.choice(embeddings.shape[0], 2, replace=False)
                prob = compute_link_probability(embeddings[ia], embeddings[ib])
                addr_a = le.inverse_transform([ia])[0]
                addr_b = le.inverse_transform([ib])[0]
                rows.append({
                    "Sender": addr_a,
                    "Receiver": addr_b,
                    "Probability": round(prob, 4),
                    "Likelihood": "High" if prob > 0.7 else ("Medium" if prob > 0.4 else "Low"),
                    "Sender Fraud": "🚨" if ia in FRAUD_IDS else "✅",
                    "Receiver Fraud": "🚨" if ib in FRAUD_IDS else "✅",
                })

            df_pred = pd.DataFrame(rows).sort_values("Probability", ascending=False)
            st.dataframe(
                df_pred,
                use_container_width=True,
                height=420,
                column_config={
                    "Sender":   st.column_config.TextColumn("Sender Address",   width="large"),
                    "Receiver": st.column_config.TextColumn("Receiver Address", width="large"),
                    "Probability": st.column_config.NumberColumn("Probability", format="%.4f"),
                },
            )

            st.download_button(
                "⬇️ Export All Predictions (CSV)",
                df_pred.to_csv(index=False),
                file_name="random_predictions.csv",
                mime="text/csv",
            )


# -----------------------------------------------
# ===  SECTION: Fraud Detection  ================
# -----------------------------------------------
elif "🚨 Fraud Detection" in section:
    st.markdown(
        '<p class="main-header">🚨 Fraud Detection Dashboard</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sub-header">Wallets scored 0–100 via Isolation Forest on 64-dim GNN embeddings · Higher = more suspicious</p>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    critical_count = len(fraud_df[fraud_df["risk_level"] == "Critical"])
    high_count = len(fraud_df[fraud_df["risk_level"] == "High"])
    with col1:
        st.markdown('<div class="fraud-alert">', unsafe_allow_html=True)
        st.metric("Critical Risk Wallets", f"{critical_count:,}")
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.metric("High Risk Wallets", f"{high_count:,}")
    with col3:
        st.metric("Total Flagged", f"{len(fraud_df):,}")
    with col4:
        pct = len(fraud_df) / embeddings.shape[0] * 100
        st.metric("Detection Rate", f"{pct:.2f}%")

    st.markdown("---")

    # Risk distribution chart
    st.markdown("### 📊 Risk Score Distribution")
    fig = plot_fraud_distribution(fraud_df)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Filter controls
    st.markdown("### 🔍 Suspicious Wallet Explorer")
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        risk_filter = st.select_slider(
            "Minimum Risk Score",
            options=[0, 35, 60, 80],
            value=35,
            format_func=lambda v: {0: "All (0+)", 35: "Medium+ (35+)", 60: "High+ (60+)", 80: "Critical (80+)"}[v],
        )
    with col2:
        top_n = st.selectbox("Show top N", [10, 25, 50, 100], index=1)
    with col3:
        sort_fraud_by = st.selectbox("Sort by", ["risk_score", "fraud_score"], index=0)

    filtered = (
        fraud_df[fraud_df["risk_score"] >= risk_filter]
        .sort_values(sort_fraud_by, ascending=(sort_fraud_by == "fraud_score"))
        .head(top_n)
        .copy()
    )

    if "wallet_id" in filtered.columns:
        # Add transaction counts
        tx_rows = []
        for wid in filtered["wallet_id"]:
            sent = len(edges[edges["from_id"] == wid])
            recv = len(edges[edges["to_id"] == wid])
            tx_rows.append({"Sent Txs": sent, "Recv Txs": recv, "Total Txs": sent + recv})
        tx_df = pd.DataFrame(tx_rows)
        filtered = pd.concat([filtered.reset_index(drop=True), tx_df], axis=1)

    # Build display table with badge column
    display_cols = {}
    if "wallet_address" in filtered.columns:
        display_cols["wallet_address"] = "Wallet Address"
    if "wallet_id" in filtered.columns:
        display_cols["wallet_id"] = "Wallet ID"
    display_cols["risk_score"] = "Risk Score (0–100)"
    display_cols["risk_level"] = "Risk Level"
    display_cols["fraud_score"] = "Raw IF Score"
    for extra in ["Total Txs", "Sent Txs", "Recv Txs"]:
        if extra in filtered.columns:
            display_cols[extra] = extra

    disp = filtered[[c for c in display_cols if c in filtered.columns]].rename(columns=display_cols)

    # Build column_config so wallet address column renders fully
    col_cfg = {}
    if "Wallet Address" in disp.columns:
        col_cfg["Wallet Address"] = st.column_config.TextColumn("Wallet Address", width="large")
    if "Risk Score (0–100)" in disp.columns:
        col_cfg["Risk Score (0–100)"] = st.column_config.NumberColumn("Risk Score (0–100)", format="%.1f")

    st.dataframe(disp, use_container_width=True, height=420, column_config=col_cfg)

    # Export
    st.download_button(
        "⬇️ Export Fraud Report (CSV)",
        disp.to_csv(index=False),
        file_name="fraud_report.csv",
        mime="text/csv",
    )

    st.markdown("---")

    # Wallet investigation
    st.markdown("### 🕵️ Investigate Specific Wallet")
    inv_col1, inv_col2 = st.columns([3, 1])
    with inv_col1:
        wallet_id_input = st.number_input(
            "Enter Wallet ID",
            min_value=0,
            max_value=int(embeddings.shape[0] - 1),
            value=0,
        )
    with inv_col2:
        # Quick-fill with top suspicious wallet
        if st.button("↑ Load Top Suspicious", use_container_width=True):
            top_id = int(fraud_df.sort_values("risk_score", ascending=False).iloc[0]["wallet_id"])
            st.session_state["_inv_wallet"] = top_id

    if "_inv_wallet" in st.session_state:
        wallet_id_input = st.session_state["_inv_wallet"]

    if st.button("🔍 Investigate Wallet", use_container_width=True):
        info = fraud_df[fraud_df["wallet_id"] == wallet_id_input]
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Wallet Details")
            try:
                addr = le.inverse_transform([wallet_id_input])[0]
                st.markdown(
                    f'<span class="wallet-address-full">📍 {addr}</span>',
                    unsafe_allow_html=True,
                )
            except Exception:
                st.write(f"Wallet ID: `{wallet_id_input}`")

            if len(info) > 0:
                rs = float(info.iloc[0]["risk_score"])
                rl = info.iloc[0]["risk_level"]
                st.markdown('<div class="fraud-alert">', unsafe_allow_html=True)
                st.warning(f"⚠️ **FLAGGED — {rl} Risk**")
                col_a, col_b = st.columns(2)
                col_a.metric("Risk Score", f"{rs:.1f} / 100")
                col_b.metric("Raw IF Score", f"{info.iloc[0]['fraud_score']:.4f}")
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                st.success("✅ No fraud flags detected")
                st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("#### Transaction Activity")
            sent_txs = edges[edges["from_id"] == wallet_id_input]
            recv_txs = edges[edges["to_id"] == wallet_id_input]
            st.metric("Transactions Sent", f"{len(sent_txs):,}")
            st.metric("Transactions Received", f"{len(recv_txs):,}")
            counterparties = len(
                set(sent_txs["to_id"].unique()) | set(recv_txs["from_id"].unique())
            )
            st.metric("Unique Counterparties", f"{counterparties:,}")


# -----------------------------------------------
# ===  SECTION: Model Performance  ==============
# -----------------------------------------------
elif "📈 Model Performance" in section:
    st.markdown(
        '<p class="main-header">📈 Model Performance Metrics</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sub-header">Comprehensive GNN evaluation — ROC-AUC, training loss curve & architecture details</p>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📉 Training Loss Curve")
        if loss_history is not None:
            fig = plot_loss_curve(loss_history)
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("Loss Statistics"):
                reduction = (loss_history[0] - loss_history[-1]) / loss_history[0] * 100
                c1, c2, c3 = st.columns(3)
                c1.metric("Initial Loss", f"{loss_history[0]:.4f}")
                c2.metric("Final Loss", f"{loss_history[-1]:.4f}")
                c3.metric("Reduction", f"{reduction:.1f}%")
                st.write(f"Total epochs: **{len(loss_history)}**")
        else:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.info("Loss history not available. Save `loss_history.npy` during training.")
            st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("### 📊 ROC Curve")
        if all(x is not None for x in [fpr, tpr, roc_auc_val]):
            fig = plot_roc_curve(fpr, tpr, roc_auc_val)
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("ROC Statistics"):
                st.metric("AUC Score", f"{roc_auc_val:.4f}")
                if roc_auc_val > 0.9:
                    st.success("🟢 Excellent — AUC > 0.9")
                elif roc_auc_val > 0.7:
                    st.success("🟢 Good — AUC > 0.7")
                elif roc_auc_val > 0.6:
                    st.warning("🟡 Moderate — AUC > 0.6")
                else:
                    st.error("🔴 Poor — AUC ≤ 0.6")
                st.write(
                    f"The model distinguishes linked from non-linked wallet pairs with "
                    f"**{roc_auc_val*100:.1f}%** accuracy."
                )
        else:
            st.markdown('<div class="info-box">', unsafe_allow_html=True)
            st.info("ROC data not available. Save `roc_data.npz` after evaluation.")
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🏗️ Model Architecture")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### GraphSAGE Configuration")
        st.code(
            "Model          : GraphSAGE\n"
            "Layers         : 2\n"
            "Hidden channels: 64\n"
            "Input features : 2  (in/out degree)\n"
            "Output embedding: 64 dimensions\n"
            "Decoder        : Multi-signal (dot + cosine + L2)",
            language="text",
        )
    with col2:
        st.markdown("#### Training Configuration")
        st.code(
            "Optimiser      : Adam\n"
            "Learning rate  : 0.01\n"
            "Loss function  : Binary Cross-Entropy\n"
            "Epochs         : 100\n"
            "Train/Val/Test : 70 / 15 / 15\n"
            "Negative sampling: Dynamic",
            language="text",
        )

    st.markdown("---")
    st.markdown("### 📊 Evaluation Summary")
    metrics_df = pd.DataFrame(
        {
            "Metric": ["ROC-AUC", "Model Type", "Embedding Dim", "Total Nodes", "Total Edges", "Decoder"],
            "Value": [
                f"{roc_auc_val:.4f}" if roc_auc_val is not None else "N/A",
                "GraphSAGE",
                "64",
                f"{embeddings.shape[0]:,}",
                f"{len(edges):,}",
                "Multi-signal (dot + cosine + L2)",
            ],
            "Status": ["✅", "✅", "✅", "✅", "✅", "✅"],
        }
    )
    st.dataframe(metrics_df, width="stretch", hide_index=True)


# -----------------------------------------------
# ===  SECTION: Network Visualization  ==========
# -----------------------------------------------
elif "🌐 Network Visualization" in section:
    st.markdown(
        '<p class="main-header">🌐 Network Visualization</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sub-header">Explore the transaction graph · Red = fraud-flagged · Blue = central wallet</p>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        wallet_id_viz = st.number_input(
            "Wallet ID",
            min_value=0,
            max_value=int(embeddings.shape[0] - 1),
            value=0,
        )
    with col2:
        max_connections = st.slider("Max connections", 10, 60, 25)
    with col3:
        depth = st.selectbox("Hop depth", ["1-hop", "2-hop (slower)"], index=0)

    if st.button("🌐 Generate Network Graph", type="primary", use_container_width=True):
        with st.spinner("Building graph …"):
            if depth == "2-hop (slower)":
                # extend edges one more hop
                hop1 = edges[
                    (edges["from_id"] == wallet_id_viz) | (edges["to_id"] == wallet_id_viz)
                ]
                hop1_nodes = set(hop1["from_id"].tolist() + hop1["to_id"].tolist())
                hop2 = edges[
                    edges["from_id"].isin(hop1_nodes) | edges["to_id"].isin(hop1_nodes)
                ]
                combined = pd.concat([hop1, hop2]).drop_duplicates().head(max_connections)
                fig = create_network_subgraph(combined, wallet_id_viz, le, fraud_df, max_connections)
            else:
                fig = create_network_subgraph(edges, wallet_id_viz, le, fraud_df, max_connections)

            if fig:
                st.plotly_chart(fig, use_container_width=True)

                # Stats
                st.markdown("---")
                st.markdown("### 📊 Network Statistics")
                out_txs = edges[edges["from_id"] == wallet_id_viz]
                in_txs = edges[edges["to_id"] == wallet_id_viz]
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Outgoing Txs", f"{len(out_txs):,}")
                col2.metric("Incoming Txs", f"{len(in_txs):,}")
                col3.metric("Total Connections", f"{len(out_txs) + len(in_txs):,}")
                is_fraud_node = wallet_id_viz in FRAUD_IDS
                col4.metric("Fraud Status", "🚨 Flagged" if is_fraud_node else "✅ Clean")

                with st.expander("🔍 Connected Wallets"):
                    related = edges[
                        (edges["from_id"] == wallet_id_viz) | (edges["to_id"] == wallet_id_viz)
                    ].copy()
                    try:
                        related["from_address"] = le.inverse_transform(related["from_id"].astype(int))
                        related["to_address"]   = le.inverse_transform(related["to_id"].astype(int))
                    except Exception:
                        pass
                    st.dataframe(
                        related.head(50),
                        use_container_width=True,
                        column_config={
                            "from_address": st.column_config.TextColumn("From Address", width="large"),
                            "to_address":   st.column_config.TextColumn("To Address",   width="large"),
                        },
                    )
            else:
                st.warning(f"No transactions found for wallet {wallet_id_viz}")

    st.markdown("---")

    # Top suspicious wallet networks
    st.markdown("### 🚨 Suspicious Wallet Networks")
    top_n_fraud = st.selectbox("Show top N fraud wallets", [3, 5, 10], index=1)
    if st.button("🔍 Analyse Suspicious Wallet Networks", use_container_width=True):
        top_fraud = fraud_df.sort_values("risk_score", ascending=False).head(top_n_fraud)
        for _, row in top_fraud.iterrows():
            wid = int(row["wallet_id"])
            rs = float(row["risk_score"])
            rl = row["risk_level"]
            with st.expander(f"Wallet {wid} — Risk Score {rs:.1f} ({rl})"):
                fig = create_network_subgraph(edges, wid, le, fraud_df, max_connections=15)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No transactions found for this wallet.")


# -----------------------------------------------
# Footer
# -----------------------------------------------
st.markdown("---")
st.markdown(
    f"""
<div class="footer-text">
    <strong>Transaction Link Prediction in Blockchain using Graph Neural Networks</strong><br>
    Powered by GraphSAGE &nbsp;·&nbsp; Ethereum Mainnet &nbsp;·&nbsp; Built with Streamlit<br>
    <span style="color:#334155; font-size:0.75rem;">
        Dashboard v2.1 &nbsp;·&nbsp; {datetime.now().strftime("%Y-%m-%d")} &nbsp;·&nbsp;
        All wallet addresses displayed in full for auditability
    </span>
</div>
""",
    unsafe_allow_html=True,
)
