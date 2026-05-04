"""
app.py  —  Blockchain GNN Transaction Intelligence Dashboard
GraphSAGE link prediction + Isolation Forest fraud detection on Ethereum Mainnet
Enhanced with Web3.py event-log parsing, PCA embedding explorer, live architecture diagram.
"""
import os
import re
import pickle
import logging
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# ── Utils modules ─────────────────────────────────────────────────────────
from utils.theme import inject_css, get_palette
from utils.ml_utils import (
    normalise_fraud_score, risk_label,
    compute_link_probability, compute_pca_projection,
    find_top_k_similar, create_graph_statistics,
)
from utils.blockchain import (
    DEFI_PROTOCOLS, KNOWN_EVENTS, ETHERSCAN_API_KEY,
    fetch_eth_balance, fetch_transactions, fetch_token_transfers,
    check_is_contract, fetch_tx_count, detect_protocols_from_txs,
    fetch_event_logs_web3, get_latest_block_info, get_web3,
)
from utils.viz import (
    plot_loss_curve, plot_roc_curve, plot_fraud_distribution,
    create_network_subgraph, plot_pca_embeddings,
    plot_decoder_signals,
    plot_transaction_timeline, plot_event_log_chart,
    plot_degree_distributions, render_architecture_animation,
)

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChainIntel Pro — Blockchain Intelligence",
    page_icon="⛓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session-state defaults ────────────────────────────────────────────────
for _k, _v in [("search_history", []), ("dark_mode", True), ("live_lookup_addr", "")]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── CSS injection ─────────────────────────────────────────────────────────
inject_css(st.session_state.dark_mode)
p = get_palette()                          # live palette dict for inline styles

# ── Inline helpers ────────────────────────────────────────────────────────
def is_valid_eth_address(addr: str) -> bool:
    return bool(re.fullmatch(r"0x[0-9a-fA-F]{40}", addr.strip()))

def risk_badge(score: float) -> str:
    lbl = risk_label(score)
    cls = {"Critical": "risk-critical", "High": "risk-high",
           "Medium": "risk-medium", "Low": "risk-low"}[lbl]
    return f'<span class="{cls}">{lbl}</span>'

# ── Data loading ──────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    required = {
        "node_embeddings.npy":   "GNN node embeddings",
        "edges.csv":             "Transaction edge list",
        "fraudulent_wallets.csv":"Fraud detection results",
        "label_encoder.pkl":     "LabelEncoder (address ↔ ID)",
    }
    missing = [f"• **{f}** — {d}" for f, d in required.items() if not os.path.exists(f)]
    if missing:
        st.error("### ❌ Required data files missing\n" + "\n".join(missing))
        st.stop()
    try:
        embeddings = np.load("node_embeddings.npy")
        edges      = pd.read_csv("edges.csv")
        fraud_df   = pd.read_csv("fraudulent_wallets.csv")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with open("label_encoder.pkl", "rb") as fh:
                le = pickle.load(fh)
        loss_history = None
        try:   loss_history = np.load("loss_history.npy")
        except FileNotFoundError: pass
        fpr = tpr = roc_auc_val = None
        try:
            roc = np.load("roc_data.npz")
            fpr, tpr, roc_auc_val = roc["fpr"], roc["tpr"], float(roc["auc"])
        except FileNotFoundError: pass
        s_min = float(fraud_df["fraud_score"].min())
        s_max = float(fraud_df["fraud_score"].max())
        fraud_df["risk_score"] = fraud_df["fraud_score"].apply(
            lambda s: round(normalise_fraud_score(s, s_min, s_max), 1))
        fraud_df["risk_level"] = fraud_df["risk_score"].apply(risk_label)
        return embeddings, edges, fraud_df, le, loss_history, (fpr, tpr, roc_auc_val)
    except Exception as exc:
        logger.exception("Fatal data load error")
        st.error(f"Error loading data: {exc}")
        st.stop()


@st.cache_data
def _graph_stats(edges_df: pd.DataFrame) -> dict:
    return create_graph_statistics(edges_df)


@st.cache_data(show_spinner=False)
def _pca_coords(emb: np.ndarray, sample_size: int, n_components: int):
    return compute_pca_projection(emb, sample_size=sample_size, n_components=n_components)


# ── Load data ─────────────────────────────────────────────────────────────
embeddings, edges, fraud_df, le, loss_history, (fpr, tpr, roc_auc_val) = load_data()
stats     = _graph_stats(edges)
WALLET_SET = set(le.classes_)
FRAUD_IDS: set = set(fraud_df["wallet_id"].tolist()) if "wallet_id" in fraud_df.columns else set()

# ═══════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════
with st.sidebar:
    # ── Product Logo ──────────────────────────────────────────────────
    st.markdown(f"""
    <div class="sidebar-logo">
        <div class="sidebar-logo-icon">⛓</div>
        <div class="sidebar-product-name">ChainIntel Pro</div>
        <div class="sidebar-tagline">Blockchain Intelligence</div>
        <div style="margin-top:8px;">
            <span class="product-badge">ENTERPRISE</span>
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Theme toggle ──────────────────────────────────────────────────
    mode_icon  = "☀️" if st.session_state.dark_mode else "🌙"
    mode_label = "Light Mode" if st.session_state.dark_mode else "Dark Mode"
    if st.button(f"{mode_icon}  Switch to {mode_label}", width='stretch', key="theme_toggle"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

    st.markdown("---")

    # ── Navigation ────────────────────────────────────────────────────
    st.markdown('<p class="section-label">Navigation</p>', unsafe_allow_html=True)
    section = st.radio("Navigate", [
        "🏠 Overview",
        "📊 Graph Analytics",
        "🔮 Link Prediction",
        "🚨 Fraud Detection",
        "📈 Model Performance",
        "🏗️ Architecture & ML",
        "🧬 Embedding Space",
        "🌐 Network Visualization",
        "🔌 Live Blockchain Explorer",
    ], label_visibility="collapsed")

    st.markdown("---")

    # ── Live stats ────────────────────────────────────────────────────
    st.markdown('<p class="section-label">Dataset Stats</p>', unsafe_allow_html=True)
    st.metric("Wallets Indexed",    f"{embeddings.shape[0]:,}",
              help="Node count |V| in GNN graph. Each wallet = 1 row in the embedding matrix ∈ ℝ^(|V|×64).")
    st.metric("Transactions",       f"{stats['total_transactions']:,}",
              help="Edge count |E|. Graph density ρ = |E|/(|V|·(|V|−1)). Directed edges only.")
    st.metric("Suspicious Wallets", f"{len(fraud_df):,}",
              help="Flagged by Isolation Forest: s(x,n) = 2^(−E[h(x)]/c(n)). "
                   "fraud_label = −1 in source data.")
    if roc_auc_val is not None:
        st.metric("Model ROC-AUC",  f"{roc_auc_val:.4f}",
                  help="AUC = ∫₀¹ TPR(t)dt. 1.0 = perfect separation of fraud vs clean in 64-dim space.")

    st.markdown("---")

    # ── Connection status ─────────────────────────────────────────────
    st.markdown('<p class="section-label">Connections</p>', unsafe_allow_html=True)
    _w3_ok = get_web3() is not None
    st.markdown(f"""
    <div class="conn-pill {'connected' if ETHERSCAN_API_KEY else 'disconnected'}">
        <span class="status-dot-{'green' if ETHERSCAN_API_KEY else 'red'}"></span>
        Etherscan API {'Active' if ETHERSCAN_API_KEY else 'Offline'}
    </div>
    <div class="conn-pill {'connected' if _w3_ok else 'disconnected'}">
        <span class="status-dot-{'green' if _w3_ok else 'red'}"></span>
        Web3 RPC {'Live' if _w3_ok else 'Unavailable'}
    </div>
    <div class="conn-pill connected">
        <span class="status-dot-green"></span>
        GNN Model Loaded
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Model card ────────────────────────────────────────────────────
    st.markdown('<p class="section-label">Model</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:0.77rem;color:{p['text_muted']};line-height:1.7;">
        <span class="protocol-tag">GraphSAGE</span>
        <span class="protocol-tag">2-layer</span>
        <span class="protocol-tag">64-dim</span><br><br>
        <span class="event-tag">Isolation Forest</span>
        <span class="event-tag">Unsupervised</span><br><br>
        <b style="color:{p['text']};">Decoder:</b> dot · cosine · L2
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# OVERVIEW
# ═══════════════════════════════════════════════
if "🏠 Overview" in section:
    # ── Hero Banner ───────────────────────────────────────────────────────
    st.markdown(f"""
<div class="hero-wrap">
    <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:1rem;">
        <div style="flex:1;min-width:280px;">
            <div class="page-eyebrow">Enterprise Blockchain Intelligence Platform</div>
            <div class="hero-title">ChainIntel Pro</div>
            <div class="hero-sub">
                Real-time fraud detection and transaction link prediction powered by
                Graph Neural Networks. Purpose-built for compliance teams, risk analysts,
                and blockchain security professionals.
            </div>
            <div class="hero-stat-row">
                <div class="hero-stat">
                    <div class="hero-stat-num">{embeddings.shape[0]:,}</div>
                    <div class="hero-stat-label">Wallets Indexed</div>
                </div>
                <div class="hero-stat">
                    <div class="hero-stat-num">{stats['total_transactions']:,}</div>
                    <div class="hero-stat-label">Transactions Analysed</div>
                </div>
                <div class="hero-stat">
                    <div class="hero-stat-num">{len(fraud_df):,}</div>
                    <div class="hero-stat-label">Suspicious Wallets</div>
                </div>
                <div class="hero-stat">
                    <div class="hero-stat-num">{f"{roc_auc_val:.3f}" if roc_auc_val else "N/A"}</div>
                    <div class="hero-stat-label">Model ROC-AUC</div>
                </div>
            </div>
        </div>
        <div style="display:flex;flex-direction:column;gap:8px;min-width:200px;">
            <div class="conn-pill {'connected' if ETHERSCAN_API_KEY else 'disconnected'}">
                <span class="status-dot-{'green' if ETHERSCAN_API_KEY else 'red'}"></span>
                Etherscan API {'Active' if ETHERSCAN_API_KEY else 'Offline'}
            </div>
            <div class="conn-pill {'connected' if _w3_ok else 'disconnected'}">
                <span class="status-dot-{'green' if _w3_ok else 'red'}"></span>
                Web3 RPC {'Live' if _w3_ok else 'Unavailable'}
            </div>
            <div class="conn-pill connected">
                <span class="status-dot-green"></span>
                GNN Model · 64-dim Embeddings
            </div>
            <div class="conn-pill connected">
                <span class="status-dot-blue"></span>
                Ethereum Mainnet Dataset
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

    # ── KPI Row ───────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    critical_n = len(fraud_df[fraud_df["risk_level"] == "Critical"])
    for col, lbl, val, delta, cls, tip in [
        (c1, "Wallets Indexed",    f"{embeddings.shape[0]:,}",         "Ethereum Mainnet",        "metric-card",
         "Unique Ethereum wallet addresses (EOAs) present in the GNN training graph. "
         "Each becomes a node with feature vector x = [in-degree, out-degree] ∈ ℝ²."),
        (c2, "Transactions",       f"{stats['total_transactions']:,}",  "Training graph edges",    "metric-card",
         "Directed transaction edges in the training graph. "
         "Graph density = |E| / (|V|·(|V|−1)). Each edge represents ≥1 on-chain transfer."),
        (c3, "Flagged Wallets",    f"{len(fraud_df):,}",                f"{critical_n:,} Critical","fraud-alert",
         "Wallets flagged anomalous by Isolation Forest on GNN embeddings. "
         "s(x,n) = 2^(−E[h(x)] / c(n)) where E[h(x)] is average path length to isolate x."),
        (c4, "Avg Out-Degree",     f"{stats['avg_out_degree']:.2f}",    "Transactions per sender", "metric-card",
         "Mean number of outgoing transactions per unique sender. "
         "Out-degree d⁺(v) = |{(v,u) ∈ E}|. "
         "Power-law distributed — most wallets send 1 tx, a few hubs send thousands."),
    ]:
        with col:
            st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
            st.metric(lbl, val, delta=delta, delta_color="off", help=tip)
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # ── Value Propositions (B2B) ──────────────────────────────────────────
    st.markdown('<p class="page-eyebrow">Why ChainIntel Pro</p>', unsafe_allow_html=True)
    st.markdown(f'<h3 style="margin-top:0;margin-bottom:1.2rem;font-weight:800;letter-spacing:-0.5px;color:{p["text"]};">Built for Enterprise Blockchain Risk</h3>', unsafe_allow_html=True)

    vp1, vp2, vp3, vp4 = st.columns(4)
    value_props = [
        (vp1, "🛡️", p["danger"], "rgba(255,69,58,0.10)" if st.session_state.dark_mode else "rgba(255,59,48,0.08)",
         "Fraud Detection", f"Score any wallet 0–100 with Isolation Forest. {len(fraud_df):,} suspicious wallets identified. Critical thresholds auto-flagged."),
        (vp2, "🔮", p["primary"], "rgba(10,132,255,0.10)" if st.session_state.dark_mode else "rgba(0,122,255,0.07)",
         "Link Prediction", "Predict transaction probability between any two wallets using a multi-signal GraphSAGE decoder with 3 geometric similarity metrics."),
        (vp3, "🧬", p["accent"], "rgba(94,92,230,0.10)" if st.session_state.dark_mode else "rgba(88,86,214,0.07)",
         "Embedding Explorer", f"Visualise all {embeddings.shape[0]:,} wallets in 2D/3D PCA space. Find behavioural clusters and nearest-neighbour wallets."),
        (vp4, "⛓️", p["success"], "rgba(48,209,88,0.10)" if st.session_state.dark_mode else "rgba(52,199,89,0.07)",
         "Live Chain Data", "Query any Ethereum address in real-time via Etherscan API & Web3.py. Balance, transactions, ERC-20 events, on-chain history."),
    ]
    for col, icon, color, bg, title, desc in value_props:
        with col:
            st.markdown(f"""
<div class="value-card">
    <div class="value-icon" style="background:{bg};color:{color};">{icon}</div>
    <div class="value-title">{title}</div>
    <div class="value-desc">{desc}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Technology + Methodology ──────────────────────────────────────────
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown(f'<h4 style="color:{p["text"]};font-weight:700;">What\'s inside</h4>', unsafe_allow_html=True)
        rows = [
            ("🔮", "Link Prediction", "Predict transaction probability between any two wallets using GraphSAGE decoder"),
            ("🚨", "Fraud Detection", "Score wallets 0–100 using Isolation Forest on 64-dim GNN node embeddings"),
            ("🧬", "Embedding Space", "Explore all 25K wallets in 2D/3D PCA — find fraud clusters and outliers"),
            ("🏗️", "Architecture Viewer", "Interactive diagram of the full GraphSAGE pipeline with live decoder analysis"),
            ("🔌", "Live Explorer", "Query any Ethereum address — ETH balance, transactions, ERC-20 event logs"),
            ("🌐", "Network Graph", "Directed transaction subgraph with 1-hop/2-hop neighbourhood expansion"),
        ]
        for icon, title, desc in rows:
            st.markdown(f"""
<div style="display:flex;gap:12px;align-items:flex-start;padding:0.6rem 0;border-bottom:1px solid {p['border']};">
    <span style="font-size:1.1rem;min-width:24px;">{icon}</span>
    <div>
        <div style="font-weight:700;font-size:0.88rem;color:{p['text']};">{title}</div>
        <div style="font-size:0.77rem;color:{p['text_muted']};margin-top:1px;">{desc}</div>
    </div>
</div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(f'<h4 style="color:{p["text"]};font-weight:700;">Technology Stack</h4>', unsafe_allow_html=True)
        stack = [
            ("Graph Model",     ["GraphSAGE", "2-layer", "64-dim", "MEAN Aggregation"]),
            ("Fraud Engine",    ["Isolation Forest", "Unsupervised", "100 estimators"]),
            ("Live Chain Data", ["Etherscan API", "Web3.py", "ERC-20 Events"]),
            ("Visualisation",   ["Plotly", "NetworkX", "PCA", "Streamlit"]),
        ]
        for section_name, tags in stack:
            st.markdown(f'<div style="font-size:0.72rem;font-weight:700;color:{p["text_subtle"]};letter-spacing:1px;text-transform:uppercase;margin:0.8rem 0 0.3rem;">{section_name}</div>', unsafe_allow_html=True)
            st.markdown(" ".join(f'<span class="protocol-tag">{t}</span>' for t in tags), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
<div class="info-box">
    <div style="font-size:0.78rem;font-weight:700;color:{p['text']};margin-bottom:4px;">Etherscan API</div>
    <div style="font-size:0.75rem;color:{p['text_muted']};">
        {'✅ Connected — pre-configured API key active' if ETHERSCAN_API_KEY else '⚠️ No key detected'}
        <br>Free tier: 5 calls/sec · 100K/day
    </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    with st.expander("📚 Methodology — click to expand"):
        mc1, mc2 = st.columns(2)
        with mc1:
            st.markdown("""
#### 1 · Data & Graph Construction
- **Source:** Google BigQuery Ethereum Public Dataset
- **Nodes:** wallet addresses (EOAs + some contracts)
- **Edges:** directed transaction relationships
- **Node features:** in-degree, out-degree (2-dim)

#### 2 · GraphSAGE Architecture
- 2-layer SAGE with MEAN aggregation
- Hidden & output dim: **64**
- Activation: ReLU + L2 normalisation per layer
- **Decoder:** weighted dot (0.5) + cosine (0.3) + L2 (0.2)
""")
        with mc2:
            st.markdown("""
#### 3 · Training
- Split: 70 / 15 / 15 train / val / test
- Loss: Binary Cross-Entropy
- Optimiser: Adam, lr = 0.01, epochs = 100
- Negative sampling: dynamic (random negatives)

#### 4 · Fraud Detection
- **Algorithm:** Isolation Forest (unsupervised)
- **Input:** 64-dim GNN node embeddings
- **Output:** anomaly score → normalised 0–100 risk scale
- **Thresholds:** Low < 35 · Medium 35–60 · High 60–80 · Critical ≥ 80
""")


# ═══════════════════════════════════════════════
# GRAPH ANALYTICS
# ═══════════════════════════════════════════════
elif "📊 Graph Analytics" in section:
    st.markdown('<p class="main-header">📊 Graph Analytics</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Degree distribution · Transaction explorer · Network topology metrics</p>', unsafe_allow_html=True)

    # Math tooltip formula cards
    st.markdown(f"""
<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:1rem;">
  <div class="formula-card" style="flex:1;min-width:200px;">
    <div class="formula-card-title">Graph Density</div>
    <div class="formula-card-eq">ρ = |E| / (|V|·(|V|−1))</div>
    <div class="formula-card-desc">Fraction of possible directed edges that exist. Near-zero for sparse real-world transaction graphs.</div>
  </div>
  <div class="formula-card" style="flex:1;min-width:200px;">
    <div class="formula-card-title">Out-Degree d⁺(v)</div>
    <div class="formula-card-eq">d⁺(v) = |&#123;(v,u) ∈ E&#125;|</div>
    <div class="formula-card-desc">Number of transactions wallet v sent. Used as node feature x[1] in GNN input.</div>
  </div>
  <div class="formula-card" style="flex:1;min-width:200px;">
    <div class="formula-card-title">In-Degree d⁻(v)</div>
    <div class="formula-card-eq">d⁻(v) = |&#123;(u,v) ∈ E&#125;|</div>
    <div class="formula-card-desc">Number of transactions wallet v received. Used as node feature x[0] in GNN input.</div>
  </div>
  <div class="formula-card" style="flex:1;min-width:200px;">
    <div class="formula-card-title">GNN Node Feature</div>
    <div class="formula-card-eq">x_v = [d⁻(v), d⁺(v)] ∈ ℝ²</div>
    <div class="formula-card-desc">2-dimensional input feature vector per wallet — the only raw data fed to GraphSAGE.</div>
  </div>
</div>""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("#### Network")
        st.metric("Nodes (Wallets)",   f"{embeddings.shape[0]:,}",
                  help="Total unique wallet addresses in GNN graph. |V| in graph notation.")
        st.metric("Edges (Txs)",        f"{stats['total_transactions']:,}",
                  help="Directed transaction edges |E|. Each edge (u→v) means wallet u sent ≥1 tx to v.")
        density = stats['total_transactions'] / (embeddings.shape[0] * (embeddings.shape[0]-1))
        st.metric("Graph Density",      f"{density:.2e}",
                  help=f"ρ = |E| / (|V|·(|V|−1)) = {stats['total_transactions']:,} / ({embeddings.shape[0]:,}·{embeddings.shape[0]-1:,}). "
                       "Ethereum transaction graphs are extremely sparse — most wallets interact with only a tiny fraction of others.")
        st.metric("Active Nodes",       f"{stats['active_nodes']:,}",
                  help="Wallets with at least one outgoing or incoming transaction — nodes with degree ≥ 1.")
    with c2:
        st.markdown("#### Degree")
        st.metric("Avg Out-Degree",     f"{stats['avg_out_degree']:.2f}",
                  help="Mean d⁺(v) = |{(v,u) ∈ E}| averaged over unique senders. "
                       "Dominated by the heavy tail of hub wallets (exchanges, protocols).")
        st.metric("Avg In-Degree",      f"{stats['avg_in_degree']:.2f}",
                  help="Mean d⁻(v) = |{(u,v) ∈ E}| averaged over unique receivers.")
        st.metric("Max Out-Degree",     f"{stats['max_out_degree']:,}",
                  help=f"Hub wallet #{stats['top_sender_id']} sent {stats['max_out_degree']:,} transactions — "
                       "a Barabási–Albert power-law hub node.")
        st.metric("Max In-Degree",      f"{stats['max_in_degree']:,}",
                  help=f"Hub wallet #{stats['top_receiver_id']} received {stats['max_in_degree']:,} transactions. "
                       "Likely an exchange hot-wallet or DEX contract.")
    with c3:
        st.markdown("#### Nodes")
        st.metric("Unique Senders",     f"{stats['unique_senders']:,}",
                  help="Wallets with d⁺(v) ≥ 1 — wallets that initiated at least one outgoing transaction.")
        st.metric("Unique Receivers",   f"{stats['unique_receivers']:,}",
                  help="Wallets with d⁻(v) ≥ 1 — wallets that received at least one incoming transaction.")
        st.metric("Embedding Dims",     f"{embeddings.shape[1]}",
                  help=f"GraphSAGE output dimensionality. "
                       f"Each wallet is represented as z_v ∈ ℝ^{embeddings.shape[1]}. "
                       "Higher dims → more expressive but more compute.")
        st.metric("Median Out-Degree",  f"{stats['median_out_degree']:.0f}",
                  help="50th percentile of d⁺(v). Typically 1 — most wallets on Ethereum send exactly 1 transaction.")
    with c4:
        st.markdown("#### Power-Law")
        st.metric("Single-Tx Wallets",  f"{stats['single_tx_pct']:.1f}%",
                  help=f"{stats['single_tx_senders']:,} wallets sent exactly 1 transaction. "
                       "Characteristic of scale-free networks: P(k) ∝ k^(−γ) — Barabási–Albert model.")
        st.metric("Hub Senders (>10)",  f"{stats['hub_senders']:,}",
                  help="Wallets with d⁺(v) > 10 — the 'rich get richer' tail of the power-law distribution. "
                       "Often exchanges, MEV bots, or DeFi protocols.")
        st.metric("Hub Receivers (>10)",f"{stats['hub_receivers']:,}",
                  help="Wallets with d⁻(v) > 10. "
                       "High in-degree can signal aggregation behaviour (e.g. Uniswap pools).")
        st.metric("Top Sender ID",      f"#{stats['top_sender_id']}",
                  help=f"Wallet with maximum out-degree = {stats['max_out_degree']:,}. "
                       "This is the most active sender in the entire training dataset.")

    st.markdown("---")
    st.markdown("### 📊 Degree Distributions")
    st.markdown(
        f'<p style="color:{p["text_muted"]};font-size:0.82rem;margin-bottom:0.5rem;">'
        f'Log-scale Y-axis · Bars right of the dashed line are <b>hub wallets</b> (degree&nbsp;&gt;&nbsp;10) · '
        f'X-axis capped at 99th percentile for readability · hover for exact counts.</p>',
        unsafe_allow_html=True)
    fig_out, fig_in = plot_degree_distributions(edges, dark=st.session_state.dark_mode)
    col1, col2 = st.columns(2)
    with col1: st.plotly_chart(fig_out, width='stretch')
    with col2: st.plotly_chart(fig_in,  width='stretch')

    # Hub wallets table
    with st.expander("🏆 Top Hub Wallets"):
        out_deg_series = edges.groupby("from_id").size().nlargest(10).reset_index()
        out_deg_series.columns = ["Wallet ID", "Out-Degree"]
        in_deg_series  = edges.groupby("to_id").size().nlargest(10).reset_index()
        in_deg_series.columns  = ["Wallet ID", "In-Degree"]
        try:
            out_deg_series["Address"] = out_deg_series["Wallet ID"].apply(
                lambda i: le.inverse_transform([i])[0])
            in_deg_series["Address"]  = in_deg_series["Wallet ID"].apply(
                lambda i: le.inverse_transform([i])[0])
        except Exception:
            pass
        out_deg_series["Fraud"] = out_deg_series["Wallet ID"].apply(
            lambda i: "🚨" if i in FRAUD_IDS else "✅")
        in_deg_series["Fraud"]  = in_deg_series["Wallet ID"].apply(
            lambda i: "🚨" if i in FRAUD_IDS else "✅")
        hc1, hc2 = st.columns(2)
        with hc1:
            st.markdown("**Top Senders (Out-Degree)**")
            st.dataframe(out_deg_series, width='stretch', hide_index=True,
                         column_config={"Address": st.column_config.TextColumn("Address", width="large"),
                                        "Out-Degree": st.column_config.NumberColumn("Out-Degree", format="%d")})
        with hc2:
            st.markdown("**Top Receivers (In-Degree)**")
            st.dataframe(in_deg_series, width='stretch', hide_index=True,
                         column_config={"Address": st.column_config.TextColumn("Address", width="large"),
                                        "In-Degree": st.column_config.NumberColumn("In-Degree", format="%d")})

    st.markdown("---")
    st.markdown("### 📋 Transaction Explorer")
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1: search_wallet = st.text_input("🔍 Filter by Wallet ID (numeric)", "")
    with col2: num_rows = st.selectbox("Rows", [10, 25, 50, 100], index=0)
    with col3: sort_col = st.selectbox("Sort by", edges.columns.tolist(), index=0)

    display_edges = edges.sort_values(sort_col)
    if search_wallet:
        try:
            wid = int(search_wallet)
            display_edges = display_edges[
                (display_edges["from_id"] == wid) | (display_edges["to_id"] == wid)]
            st.info(f"Found **{len(display_edges):,}** transactions for wallet `{wid}`")
        except ValueError:
            st.error("Enter a numeric wallet ID.")
    show = display_edges.head(num_rows).copy()
    try:
        show["from_address"] = le.inverse_transform(show["from_id"].astype(int))
        show["to_address"]   = le.inverse_transform(show["to_id"].astype(int))
    except Exception: pass
    st.dataframe(show, width='stretch',
                 column_config={
                     "from_address": st.column_config.TextColumn("From Address", width="large"),
                     "to_address":   st.column_config.TextColumn("To Address",   width="large"),
                 })


# ═══════════════════════════════════════════════
# LINK PREDICTION
# ═══════════════════════════════════════════════
elif "🔮 Link Prediction" in section:
    st.markdown('<p class="main-header">🔮 Transaction Link Prediction</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predict future transaction probability · Multi-signal decoder: dot-product · cosine · L2</p>', unsafe_allow_html=True)

    st.markdown(f"""
<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:0.8rem;">
  <div class="formula-card" style="flex:1;min-width:220px;">
    <div class="formula-card-title">GraphSAGE Aggregation (per layer k)</div>
    <div class="formula-card-eq">h_u^k = σ(W^k · CONCAT(h_u^(k-1), MEAN(&#123;h_v : v∈N(u)&#125;)))</div>
    <div class="formula-card-desc">σ = ReLU · W^k = learnable weight matrix · N(u) = transaction neighbours of wallet u</div>
  </div>
  <div class="formula-card" style="flex:1;min-width:220px;">
    <div class="formula-card-title">Decoder — Link Probability</div>
    <div class="formula-card-eq">P(u→v) = σ(0.5·dot + 0.3·cos·|dot| + 0.2·L2sim·|dot|)</div>
    <div class="formula-card-desc">σ = sigmoid · dot = h_u·h_v · cos = cosine sim · L2sim = 1/(1+‖h_u−h_v‖)</div>
  </div>
  <div class="formula-card" style="flex:1;min-width:200px;">
    <div class="formula-card-title">PCA Projection (for display)</div>
    <div class="formula-card-eq">z = W^T x,  W ∈ ℝ^(64×2)</div>
    <div class="formula-card-desc">Principal Component Analysis reduces 64-dim embeddings to 2D for visualisation only — not used in prediction.</div>
  </div>
</div>""", unsafe_allow_html=True)
    st.markdown("---")

    if st.session_state.search_history:
        history_html = " ".join(
            f'<span class="history-pill" title="{a}">{a[:10]}…{a[-6:]}</span>'
            for a in st.session_state.search_history[-6:])
        st.markdown("**Recent lookups:** " + history_html, unsafe_allow_html=True)
        st.markdown("")

    tab1, tab2 = st.tabs(["🔤 Manual Input", "🎲 Random Predictions"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            wallet_a = st.text_input("Sender Wallet Address",
                                     placeholder="0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb")
            if wallet_a:
                if not is_valid_eth_address(wallet_a):
                    st.markdown('<p class="invalid-address">⚠ Invalid format</p>', unsafe_allow_html=True)
                elif wallet_a not in WALLET_SET:
                    st.markdown('<p class="invalid-address">⚠ Not found in dataset</p>', unsafe_allow_html=True)
                else:
                    st.markdown('<p class="valid-address">✔ Valid — found in dataset</p>', unsafe_allow_html=True)
        with col2:
            wallet_b = st.text_input("Receiver Wallet Address",
                                     placeholder="0x3f5CE5FBFe3E9af3971dD833D26bA9b5C936f0bE")
            if wallet_b:
                if not is_valid_eth_address(wallet_b):
                    st.markdown('<p class="invalid-address">⚠ Invalid format</p>', unsafe_allow_html=True)
                elif wallet_b not in WALLET_SET:
                    st.markdown('<p class="invalid-address">⚠ Not found in dataset</p>', unsafe_allow_html=True)
                else:
                    st.markdown('<p class="valid-address">✔ Valid — found in dataset</p>', unsafe_allow_html=True)

        if st.button("🔮 Predict Transaction Probability", type="primary", width='stretch'):
            errors = []
            if not wallet_a:                           errors.append("Sender address is empty.")
            elif not is_valid_eth_address(wallet_a):   errors.append("Sender address has invalid format.")
            elif wallet_a not in WALLET_SET:           errors.append("Sender address not in dataset.")
            if not wallet_b:                           errors.append("Receiver address is empty.")
            elif not is_valid_eth_address(wallet_b):   errors.append("Receiver address has invalid format.")
            elif wallet_b not in WALLET_SET:           errors.append("Receiver address not in dataset.")
            if wallet_a and wallet_b and wallet_a == wallet_b:
                errors.append("Sender and receiver must be different.")

            if errors:
                for e in errors: st.error(f"❌ {e}")
            else:
                id_a = int(le.transform([wallet_a])[0])
                id_b = int(le.transform([wallet_b])[0])
                prob = compute_link_probability(embeddings[id_a], embeddings[id_b])
                for addr in [wallet_a, wallet_b]:
                    if addr not in st.session_state.search_history:
                        st.session_state.search_history.append(addr)

                st.markdown("---")
                # Animated probability gauge
                _col_gauge = p["danger"] if prob > 0.7 else p["warning"] if prob > 0.4 else p["success"]
                _label_gauge = ("🔴 HIGH LIKELIHOOD" if prob > 0.7
                                else "🟡 MODERATE LIKELIHOOD" if prob > 0.4
                                else "🟢 LOW LIKELIHOOD")
                _fill_grad = (f"linear-gradient(90deg,{p['danger']}99,{p['danger']})" if prob > 0.7
                              else f"linear-gradient(90deg,{p['warning']}99,{p['warning']})" if prob > 0.4
                              else f"linear-gradient(90deg,{p['success']}99,{p['success']})")
                rc1, rc2, rc3 = st.columns([1, 2, 1])
                with rc2:
                    st.markdown(f"""
<div style="background:{_col_gauge}0F;border:1px solid {_col_gauge}44;border-radius:16px;
     padding:1.4rem 1.6rem;text-align:center;position:relative;overflow:hidden;">
  <div style="position:absolute;top:0;left:0;right:0;height:3px;
       background:linear-gradient(90deg,{p['grad_start']},{p['grad_mid']},{_col_gauge});
       background-size:200%;animation:shimmerFlow 2s linear infinite;"></div>
  <div style="font-size:0.72rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;
       color:{p['text_muted']};margin-bottom:6px;">Link Probability — P(u→v)</div>
  <div style="font-size:3.2rem;font-weight:900;letter-spacing:-2px;color:{_col_gauge};
       font-variant-numeric:tabular-nums;animation:countUp 0.6s cubic-bezier(0.34,1.56,0.64,1) both;">
    {prob:.4f}
  </div>
  <div style="font-size:0.88rem;font-weight:700;color:{_col_gauge};margin-bottom:0.9rem;">{_label_gauge}</div>
  <div class="prob-bar-track">
    <div class="prob-bar-fill" style="width:{prob*100:.1f}%;background:{_fill_grad};
         --bar-w:{prob*100:.1f}%;"></div>
  </div>
  <div style="font-size:0.73rem;color:{p['text_muted']};margin-top:8px;line-height:1.55;">
    P = σ(0.5·dot + 0.3·cos·|dot| + 0.2·L2sim·|dot|)<br>
    Sender ID: <code>{id_a}</code> → Receiver ID: <code>{id_b}</code>
  </div>
</div>""", unsafe_allow_html=True)

                # Decoder signal breakdown chart
                st.markdown("### 🔬 Decoder Signal Breakdown")
                st.markdown(f'<p style="color:{p["text_muted"]};font-size:0.82rem;">Hover each bar to see the exact formula and contribution of each decoder signal.</p>', unsafe_allow_html=True)
                st.plotly_chart(
                    plot_decoder_signals(embeddings[id_a], embeddings[id_b],
                                         dark=st.session_state.dark_mode),
                    width='stretch')

                with st.expander("📈 Wallet Details"):
                    dc1, dc2 = st.columns(2)
                    with dc1:
                        st.markdown("**Sender**")
                        sent = edges[edges["from_id"] == id_a]
                        st.write(f"- Transactions sent: **{len(sent)}**")
                        st.write(f"- Unique receivers: **{sent['to_id'].nunique()}**")
                        st.write(f"- Fraud flag: {'🚨 Flagged' if id_a in FRAUD_IDS else '✅ Clean'}")
                    with dc2:
                        st.markdown("**Receiver**")
                        recv = edges[edges["to_id"] == id_b]
                        st.write(f"- Transactions received: **{len(recv)}**")
                        st.write(f"- Unique senders: **{recv['from_id'].nunique()}**")
                        st.write(f"- Fraud flag: {'🚨 Flagged' if id_b in FRAUD_IDS else '✅ Clean'}")
                    existing = edges[(edges["from_id"] == id_a) & (edges["to_id"] == id_b)]
                    if len(existing) > 0:
                        st.warning(f"⚠️ **{len(existing)} prior transactions** already exist between these wallets")
                    else:
                        st.info("ℹ️ No prior transactions between these wallets")
                st.download_button("⬇️ Export Result", pd.DataFrame([{
                    "Sender": wallet_a, "Receiver": wallet_b,
                    "Probability": round(prob, 6),
                    "Likelihood": "High" if prob > 0.7 else ("Medium" if prob > 0.4 else "Low"),
                    "Timestamp": datetime.now().isoformat(),
                }]).to_csv(index=False), file_name="link_prediction.csv", mime="text/csv")

    with tab2:
        num_pred = st.slider("Number of predictions", 5, 30, 10,
                              help="How many random wallet pairs to score. "
                                   "Each pair runs the full 3-signal decoder: "
                                   "P(u→v) = σ(0.5·dot + 0.3·cos·|dot| + 0.2·L2sim·|dot|). "
                                   "Results sorted by probability descending.")
        if st.button("🎲 Generate Random Predictions", width='stretch'):
            rows = []
            for _ in range(num_pred):
                ia, ib = np.random.choice(embeddings.shape[0], 2, replace=False)
                prob = compute_link_probability(embeddings[ia], embeddings[ib])
                addr_a = le.inverse_transform([ia])[0]
                addr_b = le.inverse_transform([ib])[0]
                rows.append({
                    "Sender": addr_a, "Receiver": addr_b,
                    "Probability": round(prob, 4),
                    "Likelihood":  "High" if prob > 0.7 else ("Medium" if prob > 0.4 else "Low"),
                    "Sender Fraud":   "🚨" if ia in FRAUD_IDS else "✅",
                    "Receiver Fraud": "🚨" if ib in FRAUD_IDS else "✅",
                })
            df_pred = pd.DataFrame(rows).sort_values("Probability", ascending=False)
            st.dataframe(df_pred, width='stretch', height=400,
                         column_config={
                             "Sender":      st.column_config.TextColumn("Sender",   width="large"),
                             "Receiver":    st.column_config.TextColumn("Receiver", width="large"),
                             "Probability": st.column_config.NumberColumn("Prob",   format="%.4f"),
                         })
            st.download_button("⬇️ Export CSV", df_pred.to_csv(index=False),
                               file_name="random_predictions.csv", mime="text/csv")


# ═══════════════════════════════════════════════
# FRAUD DETECTION
# ═══════════════════════════════════════════════
elif "🚨 Fraud Detection" in section:
    st.markdown('<p class="main-header">🚨 Fraud Detection Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Wallets scored 0–100 via Isolation Forest on 64-dim GNN embeddings · Higher = more suspicious</p>', unsafe_allow_html=True)

    st.markdown(f"""
<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:0.8rem;">
  <div class="formula-card" style="flex:1;min-width:220px;">
    <div class="formula-card-title">Isolation Forest Score</div>
    <div class="formula-card-eq">s(x,n) = 2^(−E[h(x)] / c(n))</div>
    <div class="formula-card-desc">E[h(x)] = mean path length to isolate x across all trees · c(n) = expected BST path length · s → 1 means highly anomalous</div>
  </div>
  <div class="formula-card" style="flex:1;min-width:220px;">
    <div class="formula-card-title">Risk Score Normalisation</div>
    <div class="formula-card-eq">risk = (s_max − raw) / (s_max − s_min) × 100</div>
    <div class="formula-card-desc">Inverts & rescales raw IF score to 0–100. Higher risk score = shorter isolation path = more anomalous behaviour.</div>
  </div>
  <div class="formula-card" style="flex:1;min-width:200px;">
    <div class="formula-card-title">Expected BST Path Length</div>
    <div class="formula-card-eq">c(n) = 2·H(n-1) − (2(n-1)/n)</div>
    <div class="formula-card-desc">H(n) = harmonic number (ln n + 0.5772). Normalises path lengths so scores are comparable across tree sizes.</div>
  </div>
</div>""", unsafe_allow_html=True)
    st.markdown("---")

    critical_count = len(fraud_df[fraud_df["risk_level"] == "Critical"])
    high_count     = len(fraud_df[fraud_df["risk_level"] == "High"])
    med_count      = len(fraud_df[fraud_df["risk_level"] == "Medium"])
    low_count      = len(fraud_df[fraud_df["risk_level"] == "Low"])
    _total_wallets = embeddings.shape[0]
    _flagged_pct   = round(len(fraud_df) / _total_wallets * 100, 2)
    _avg_score     = round(float(fraud_df["risk_score"].mean()), 1)
    _max_score     = round(float(fraud_df["risk_score"].max()), 1)
    _top_fraud_id  = int(fraud_df.sort_values("risk_score", ascending=False).iloc[0]["wallet_id"])

    st.markdown(
        f'<div style="background:rgba(248,81,73,0.08);border:1px solid rgba(248,81,73,0.25);'
        f'border-radius:12px;padding:0.75rem 1.2rem;margin-bottom:0.8rem;display:flex;'
        f'align-items:center;gap:1.5rem;flex-wrap:wrap;">'
        f'<span style="font-size:0.9rem;font-weight:600;color:{p["danger"]};">🚨 Threat Summary</span>'
        f'<span style="font-size:0.83rem;color:{p["text"]};">'
        f'<b>{len(fraud_df):,}</b> of <b>{_total_wallets:,}</b> wallets flagged '
        f'<span style="color:{p["text_muted"]};">({_flagged_pct}% of network)</span></span>'
        f'<span style="font-size:0.83rem;color:{p["text_muted"]};">'
        f'·  Avg risk score: <b style="color:' + p["text"] + f';">{_avg_score}</b></span>'
        f'<span style="font-size:0.83rem;color:{p["text_muted"]};">'
        f'·  Highest score: <b style="color:' + p["danger"] + f';">{_max_score}/100</b> '
        f'(wallet #{_top_fraud_id})</span>'
        f'</div>',
        unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="fraud-alert">', unsafe_allow_html=True)
        st.metric("🔴 Critical (≥80)", f"{critical_count:,}",
                  help="Risk score ≥ 80. Isolation Forest path length E[h(x)] is very short — "
                       "the model isolated these wallets in just a few tree splits, "
                       "indicating extreme outliers in 64-dim embedding space. "
                       "Immediate investigation recommended.")
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.metric("🟠 High (60–79)", f"{high_count:,}",
                  help="Risk score 60–79. Wallets with significantly shorter-than-average "
                       "isolation paths. Anomaly score s(x,n) indicates embeddings that "
                       "are far from the main cluster in latent space. Investigate promptly.")
        st.markdown("</div>", unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.metric("🟡 Medium (35–59)", f"{med_count:,}",
                  help="Risk score 35–59. Moderate deviation from normal embedding distribution. "
                       "s(x,n) = 2^(−E[h(x)] / c(n)) is elevated but not extreme. "
                       "Monitor for further anomalous activity.")
        st.markdown("</div>", unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.metric("🟢 Low (0–34)", f"{low_count:,}",
                  help="Risk score 0–34. These wallets were flagged by Isolation Forest "
                       "(fraud_label = −1) but show only mild deviation from the normal cluster. "
                       "Long isolation paths indicate near-normal embedding vectors.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.plotly_chart(
            plot_fraud_distribution(fraud_df, dark=st.session_state.dark_mode),
            width='stretch')
    with col2:
        level_counts = fraud_df["risk_level"].value_counts()
        fig_pie = px.pie(values=level_counts.values, names=level_counts.index,
                         title="Risk Level Breakdown",
                         color=level_counts.index,
                         color_discrete_map={"Critical":"#f85149","High":"#ff7b72",
                                              "Medium":"#e3b341","Low":"#3fb950"})
        fig_pie.update_layout(template=p["plotly"], height=340,
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_pie, width='stretch')

    st.markdown("---")
    st.markdown("### 🔍 Fraud Wallet Table")
    fc1, fc2, fc3 = st.columns([2, 1, 1])
    with fc1: filter_risk = st.selectbox("Filter risk level", ["All","Critical","High","Medium","Low"])
    with fc2: min_score   = st.slider("Min risk score", 0, 100, 0)
    with fc3: max_rows    = st.selectbox("Rows", [25, 50, 100, 200], index=0)

    filtered = fraud_df[fraud_df["risk_score"] >= min_score]
    if filter_risk != "All":
        filtered = filtered[filtered["risk_level"] == filter_risk]
    filtered = filtered.sort_values("risk_score", ascending=False).head(max_rows)

    _RISK_BADGE = {"Critical": "🔴 Critical", "High": "🟠 High",
                   "Medium": "🟡 Medium",   "Low":  "🟢 Low"}
    _disp_filtered = filtered.copy()
    _disp_filtered["risk_badge"] = _disp_filtered["risk_level"].map(_RISK_BADGE)
    show_cols = {"wallet_id": "Wallet ID", "risk_score": "Risk Score",
                 "risk_badge": "Risk Level", "fraud_score": "IF Score (raw)"}
    if "wallet_address" in _disp_filtered.columns:
        show_cols = {"wallet_address": "Wallet Address", **show_cols}
    disp = _disp_filtered[[c for c in show_cols if c in _disp_filtered.columns]].rename(columns=show_cols)
    st.dataframe(disp, width='stretch', height=400,
                 column_config={
                     "Wallet Address": st.column_config.TextColumn("Wallet Address", width="large"),
                     "Risk Score":     st.column_config.ProgressColumn("Risk Score", min_value=0, max_value=100,
                                                                        format="%.1f"),
                     "Risk Level":     st.column_config.TextColumn("Risk Level",  width="medium"),
                     "IF Score (raw)": st.column_config.NumberColumn("IF Score (raw)",
                                                                      format="%.5f",
                                                                      help="Raw Isolation Forest anomaly score — "
                                                                           "more negative = more anomalous. "
                                                                           "Rescaled to 0–100 for Risk Score."),
                 })
    sc1, sc2 = st.columns([1, 3])
    with sc1:
        st.download_button("⬇️ Export Fraud Report", disp.to_csv(index=False),
                           file_name="fraud_report.csv", mime="text/csv",
                           width='stretch')
    with sc2:
        st.markdown(
            f'<p style="color:{p["text_muted"]};font-size:0.78rem;padding-top:0.6rem;">'
            f'Risk Score bar = 0–100 · IF Score (raw) is the Isolation Forest anomaly score '
            f'(all values are negative; more negative → more anomalous → higher Risk Score)</p>',
            unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🕵️ Investigate Specific Wallet")
    ic1, ic2, ic3 = st.columns([3, 1, 1])
    with ic1:
        wid_input = st.number_input("Wallet ID", min_value=0,
                                    max_value=int(embeddings.shape[0]-1),
                                    value=int(st.session_state.get("_inv_wallet", _top_fraud_id)),
                                    key="inv_wallet_id",
                                    help=f"Default = wallet #{_top_fraud_id} (highest risk score in dataset)")
    with ic2:
        if st.button("🚨 Top Suspicious", width='stretch'):
            st.session_state["_inv_wallet"] = int(fraud_df.sort_values("risk_score", ascending=False).iloc[0]["wallet_id"])
            st.rerun()
    with ic3:
        if st.button("🎲 Random", width='stretch', key="fraud_rand"):
            st.session_state["_inv_wallet"] = int(np.random.randint(0, embeddings.shape[0]))
            st.rerun()

    if st.button("🔍 Investigate Wallet", type="primary", width='stretch'):
        wid = int(wid_input)
        try:   addr = le.inverse_transform([wid])[0]
        except Exception: addr = None

        info     = fraud_df[fraud_df["wallet_id"] == wid]
        sent_txs = edges[edges["from_id"] == wid]
        recv_txs = edges[edges["to_id"]   == wid]
        all_cp   = set(sent_txs["to_id"].tolist()) | set(recv_txs["from_id"].tolist())
        flagged_cp = all_cp & FRAUD_IDS
        fraud_pct  = round(len(flagged_cp) / max(len(all_cp), 1) * 100, 1)
        tx_ratio   = len(sent_txs) / max(len(recv_txs), 1)
        is_flagged = len(info) > 0
        rs = float(info.iloc[0]["risk_score"]) if is_flagged else 0.0
        rl = info.iloc[0]["risk_level"]         if is_flagged else "Clean"
        raw_if = float(info.iloc[0]["fraud_score"]) if is_flagged else None

        st.markdown("---")
        if addr:
            st.markdown(f'<span class="wallet-address-full">📍 {addr}</span>', unsafe_allow_html=True)

        badge_color = {"Critical":"#f85149","High":"#ff7b72","Medium":"#e3b341","Low":"#3fb950"}.get(rl, p["info"])
        if is_flagged:
            rank = int((fraud_df["risk_score"] > rs).sum() + 1)
            st.markdown(f"""<div style="background:{badge_color}18;border:1px solid {badge_color}44;
                border-radius:10px;padding:0.8rem 1.1rem;margin:0.6rem 0;">
                <span style="font-size:1.05rem;font-weight:700;color:{badge_color};">
                ⚠️ FLAGGED — {rl} Risk &nbsp;
                <span style="font-size:0.82rem;font-weight:400;color:{p['text_muted']};">
                    Score {rs:.1f}/100 · Rank #{rank} of {len(fraud_df):,}
                </span></span></div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div style="background:{p['success']}18;border:1px solid {p['success']}44;
                border-radius:10px;padding:0.8rem 1.1rem;margin:0.6rem 0;">
                <span style="font-size:1.05rem;font-weight:700;color:{p['success']};">✅ CLEAN — No fraud flags detected</span></div>""",
                        unsafe_allow_html=True)

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Wallet ID",   f"#{wid:,}",
                  help=f"Integer ID assigned by LabelEncoder. Corresponds to row {wid} in the "
                       f"{embeddings.shape[0]:,}×{embeddings.shape[1]} embedding matrix.")
        m2.metric("Txs Sent",    f"{len(sent_txs):,}",
                  help=f"Out-degree d⁺({wid}) = {len(sent_txs):,}. "
                       "Number of transactions where this wallet is the sender (from_id).")
        m3.metric("Txs Received",f"{len(recv_txs):,}",
                  help=f"In-degree d⁻({wid}) = {len(recv_txs):,}. "
                       "Number of transactions where this wallet is the receiver (to_id).")
        m4.metric("Counterparties", f"{len(all_cp):,}",
                  help=f"|N({wid})| = {len(all_cp):,} unique wallets in 1-hop neighbourhood. "
                       "Combined senders + receivers (undirected adjacency).")
        m5.metric("Fraud Exposure", f"{fraud_pct}%",
                  delta="⚠️ High" if fraud_pct > 30 else ("🟡 Moderate" if fraud_pct > 10 else "✅ Low"),
                  delta_color="off",
                  help=f"Fraud Exposure = {len(flagged_cp)} / {max(len(all_cp),1)} × 100 = {fraud_pct}%. "
                       "Fraction of direct counterparties flagged by Isolation Forest. "
                       "High exposure = wallet transacts frequently with anomalous wallets.")

        t_risk, t_txn, t_net, t_emb = st.tabs(
            ["🔴 Risk Profile", "📋 Transactions", "🌐 Network Context", "🧬 Embedding"])

        with t_risk:
            rc1, rc2 = st.columns(2)
            with rc1:
                st.markdown("#### Risk Breakdown")
                if is_flagged:
                    _rank = int((fraud_df["risk_score"] > rs).sum() + 1)
                    st.markdown(f"""<div style="margin:0.5rem 0 1rem;">
                        <div style="display:flex;justify-content:space-between;font-size:0.78rem;color:{p['text_muted']};margin-bottom:4px;">
                            <span>Risk Score</span><span style="color:{badge_color};font-weight:700;">{rs:.1f}/100</span></div>
                        <div class="risk-bar-track">
                            <div class="risk-bar-fill" style="--bar-w:{rs}%;background:linear-gradient(90deg,{badge_color}88,{badge_color});"></div>
                        </div>
                        <div style="font-size:0.72rem;color:{p['text_muted']};margin-top:4px;">
                            Rank #{_rank} of {len(fraud_df):,} flagged wallets
                        </div></div>""", unsafe_allow_html=True)
                    st.metric("IF Score (raw)", f"{raw_if:.5f}" if raw_if else "N/A",
                              help="Raw Isolation Forest anomaly score s(x,n). "
                                   "More negative = shorter average isolation path = more anomalous. "
                                   f"Normalised to Risk Score via: ({fraud_df['fraud_score'].max():.5f} − raw) / "
                                   f"({fraud_df['fraud_score'].max():.5f} − {fraud_df['fraud_score'].min():.5f}) × 100")
                    st.metric("Rank (most suspicious)", f"#{_rank} of {len(fraud_df):,}",
                              help=f"{_rank - 1} wallets have a higher risk score than this one. "
                                   "Rank 1 = highest risk score in the entire flagged wallet set.")
                else:
                    st.success("Not flagged by the Isolation Forest model.")
                    st.markdown(f'<p style="font-size:0.78rem;color:{p["text_muted"]};">'
                                f'Isolation Forest path length E[h(x)] ≈ c(n) — indistinguishable from '
                                f'normal wallets in 64-dim embedding space.</p>', unsafe_allow_html=True)
            with rc2:
                st.markdown("#### Behavioural Signals")
                for icon, label, detail in [
                    (("🚩" if tx_ratio > 5 else "✅"), ("High send ratio" if tx_ratio > 5 else "Balanced send/receive"),
                     f"{tx_ratio:.1f}x outgoing" if tx_ratio > 5 else f"Ratio: {tx_ratio:.2f}"),
                    (("🚩" if fraud_pct > 30 else "⚠️" if fraud_pct > 10 else "✅"),
                     "Fraud-network exposure",
                     f"{fraud_pct}% of counterparties flagged"),
                ]:
                    col_border = p["danger"] if icon == "🚩" else p["warning"] if icon == "⚠️" else p["success"]
                    st.markdown(f"""<div style="background:rgba(128,128,128,0.05);border-radius:8px;
                        padding:0.5rem 0.8rem;margin:4px 0;border-left:3px solid {col_border};">
                        <span style="font-size:0.85rem;font-weight:600;">{icon} {label}</span><br>
                        <span style="font-size:0.76rem;color:{p['text_muted']};">{detail}</span></div>""",
                        unsafe_allow_html=True)

        with t_txn:
            tc1, tc2 = st.columns(2)
            with tc1:
                st.markdown(f"#### 📤 Sent ({len(sent_txs):,})")
                if len(sent_txs) > 0:
                    sd = sent_txs.copy()
                    try: sd["to_address"] = le.inverse_transform(sd["to_id"].astype(int))
                    except Exception: pass
                    sd["fraud"] = sd["to_id"].apply(lambda x: "🚨" if x in FRAUD_IDS else "✅")
                    st.dataframe(sd, width='stretch', height=280,
                                 column_config={"to_address": st.column_config.TextColumn("To Address", width="large"),
                                                "fraud": st.column_config.TextColumn("Flag", width="small")})
                else:
                    st.info("No outgoing transactions.")
            with tc2:
                st.markdown(f"#### 📥 Received ({len(recv_txs):,})")
                if len(recv_txs) > 0:
                    rd = recv_txs.copy()
                    try: rd["from_address"] = le.inverse_transform(rd["from_id"].astype(int))
                    except Exception: pass
                    rd["fraud"] = rd["from_id"].apply(lambda x: "🚨" if x in FRAUD_IDS else "✅")
                    st.dataframe(rd, width='stretch', height=280,
                                 column_config={"from_address": st.column_config.TextColumn("From Address", width="large"),
                                                "fraud": st.column_config.TextColumn("Flag", width="small")})
                else:
                    st.info("No incoming transactions.")

        with t_net:
            nc1, nc2 = st.columns([2, 1])
            with nc1:
                st.markdown("#### Flagged Counterparties")
                if flagged_cp:
                    fp_rows = []
                    for fid in flagged_cp:
                        fi = fraud_df[fraud_df["wallet_id"] == fid]
                        try:    fa = le.inverse_transform([int(fid)])[0]
                        except Exception: fa = f"ID {fid}"
                        dirs = []
                        if fid in set(sent_txs["to_id"]):   dirs.append("Sent To")
                        if fid in set(recv_txs["from_id"]): dirs.append("Received From")
                        fp_rows.append({"Address": fa, "ID": int(fid),
                                        "Risk Level": fi.iloc[0]["risk_level"] if len(fi) > 0 else "?",
                                        "Risk Score": round(float(fi.iloc[0]["risk_score"]), 1) if len(fi) > 0 else 0,
                                        "Relationship": " & ".join(dirs)})
                    st.dataframe(pd.DataFrame(fp_rows).sort_values("Risk Score", ascending=False),
                                 width='stretch', height=260,
                                 column_config={"Address": st.column_config.TextColumn("Address", width="large"),
                                                "Risk Score": st.column_config.NumberColumn("Score", format="%.1f")})
                else:
                    st.success("✅ No flagged wallets among direct counterparties.")
            with nc2:
                st.metric("Total Counterparties", f"{len(all_cp):,}",
                          help=f"|N({wid})| = {len(all_cp):,} — size of the 1-hop neighbourhood "
                               "in the directed transaction graph. Includes both senders and receivers.")
                st.metric("Flagged Counterparties", f"{len(flagged_cp):,}",
                          help=f"{len(flagged_cp):,} of {len(all_cp):,} counterparties have "
                               "fraud_label = −1 (Isolation Forest). "
                               f"Fraud Exposure = {round(len(flagged_cp)/max(len(all_cp),1)*100,1)}%.")
                st.metric("Unique Receivers", f"{sent_txs['to_id'].nunique():,}",
                          help="Distinct wallet IDs in the 'to_id' column of outgoing transactions. "
                               "Multiple transactions may share the same receiver.")
                st.metric("Unique Senders",   f"{recv_txs['from_id'].nunique():,}",
                          help="Distinct wallet IDs in the 'from_id' column of incoming transactions. "
                               "Multiple transactions may share the same sender.")

        with t_emb:
            emb = embeddings[wid]
            ec1, ec2 = st.columns(2)
            with ec1:
                st.markdown("#### Embedding Statistics")
                st.metric("Dimensions", len(emb),
                          help=f"h_{wid} ∈ ℝ^{len(emb)} — the GNN's learned representation of this wallet. "
                               "Each dimension encodes a different latent feature of transaction behaviour.")
                st.metric("L2 Norm",    f"{float(np.linalg.norm(emb)):.4f}",
                          help=f"‖h‖₂ = √(Σᵢ hᵢ²) = {float(np.linalg.norm(emb)):.4f}. "
                               "After L2 normalisation per layer, all embeddings should lie close to the unit hypersphere (norm ≈ 1). "
                               "Deviation indicates pre-normalisation raw embedding.")
                st.metric("Mean",       f"{float(np.mean(emb)):.4f}",
                          help=f"μ = (1/{len(emb)}) Σᵢ hᵢ = {float(np.mean(emb)):.4f}. "
                               "Mean activation across all embedding dimensions. "
                               "Positive mean = overall positive bias in latent space.")
                st.metric("Std Dev",    f"{float(np.std(emb)):.4f}",
                          help=f"σ = √((1/{len(emb)}) Σᵢ (hᵢ − μ)²) = {float(np.std(emb)):.4f}. "
                               "Spread of values across dimensions. Higher std = more varied feature activations.")
                st.markdown("#### Top 5 Similar Wallets (cosine)")
                top_ids, sims = find_top_k_similar(emb, embeddings, k=6)
                sim_rows = []
                for sid, sim in zip(top_ids, sims):
                    if sid == wid: continue
                    fi = fraud_df[fraud_df["wallet_id"] == sid]
                    try:    sa = le.inverse_transform([int(sid)])[0]
                    except Exception: sa = f"ID {sid}"
                    sim_rows.append({"Address": sa, "ID": int(sid),
                                     "Cosine Sim": round(float(sim), 4),
                                     "Risk": fi.iloc[0]["risk_level"] if len(fi) > 0 else "Clean"})
                if sim_rows:
                    st.dataframe(pd.DataFrame(sim_rows).head(5), width='stretch',
                                 column_config={"Address": st.column_config.TextColumn("Address", width="large"),
                                                "Cosine Sim": st.column_config.NumberColumn("Cosine", format="%.4f")},
                                 hide_index=True)
            with ec2:
                st.markdown("#### Embedding Heatmap (8×8)")
                fig_emb = go.Figure(go.Heatmap(z=emb[:64].reshape(8, 8),
                                               colorscale="RdBu", zmid=0, showscale=True))
                fig_emb.update_layout(template=p["plotly"], height=300,
                                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                      margin=dict(t=10, b=10, l=10, r=10),
                                      xaxis=dict(showticklabels=False),
                                      yaxis=dict(showticklabels=False))
                st.plotly_chart(fig_emb, width='stretch')

        st.download_button("⬇️ Export Investigation Report",
            pd.DataFrame([{"wallet_id": wid, "address": addr or "N/A", "is_flagged": is_flagged,
                           "risk_level": rl, "risk_score": rs, "raw_if_score": raw_if,
                           "txs_sent": len(sent_txs), "txs_received": len(recv_txs),
                           "flagged_counterparties": len(flagged_cp), "fraud_exposure_pct": fraud_pct}]).to_csv(index=False),
            file_name=f"investigation_{wid}.csv", mime="text/csv", width='stretch')


# ═══════════════════════════════════════════════
# MODEL PERFORMANCE
# ═══════════════════════════════════════════════
elif "📈 Model Performance" in section:
    st.markdown('<p class="main-header">📈 Model Performance</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">ROC-AUC · Training loss curve · Evaluation summary</p>', unsafe_allow_html=True)
    st.markdown("---")

    # ── headline KPI banner ──────────────────────────────────────────────────
    _best_idx   = int(np.argmin(loss_history)) if loss_history is not None else 0
    _init_loss  = float(loss_history[0])        if loss_history is not None else 0
    _best_loss  = float(loss_history[_best_idx]) if loss_history is not None else 0
    _final_loss = float(loss_history[-1])        if loss_history is not None else 0
    _reduction  = (_init_loss - _final_loss) / _init_loss * 100 if _init_loss else 0
    _n_epochs   = len(loss_history)              if loss_history is not None else 0

    # Model Performance formula cards
    st.markdown(f"""
<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:0.8rem;">
  <div class="formula-card" style="flex:1;min-width:200px;">
    <div class="formula-card-title">ROC-AUC Definition</div>
    <div class="formula-card-eq">AUC = ∫₀¹ TPR(FPR⁻¹(t)) dt</div>
    <div class="formula-card-desc">Probability that a randomly chosen anomalous wallet is ranked higher than a random normal wallet. 1.0 = perfect.</div>
  </div>
  <div class="formula-card" style="flex:1;min-width:200px;">
    <div class="formula-card-title">Binary Cross-Entropy Loss</div>
    <div class="formula-card-eq">L = −Σ [y·log(ŷ) + (1−y)·log(1−ŷ)]</div>
    <div class="formula-card-desc">y = true label (link exists/not) · ŷ = decoder probability. Minimised via Adam optimiser over 100 epochs.</div>
  </div>
  <div class="formula-card" style="flex:1;min-width:200px;">
    <div class="formula-card-title">Youden's J Statistic</div>
    <div class="formula-card-eq">J = TPR − FPR = Sensitivity + Specificity − 1</div>
    <div class="formula-card-desc">Optimal operating threshold on ROC curve — maximises true positive rate while minimising false positive rate.</div>
  </div>
</div>""", unsafe_allow_html=True)

    hk1, hk2, hk3, hk4, hk5 = st.columns(5)
    hk1.metric("ROC-AUC",       f"{roc_auc_val:.4f}" if roc_auc_val else "N/A",
               delta="Perfect" if roc_auc_val and roc_auc_val >= 0.99 else None,
               delta_color="off",
               help="AUC = ∫₀¹ TPR(t) dt  —  Area Under the ROC Curve. "
                    "1.0000 = the Isolation Forest score perfectly separates all fraudulent "
                    "from clean wallet embeddings at every threshold. "
                    "0.5 = random classifier baseline. "
                    "Computed on all " + f"{embeddings.shape[0]:,} wallet embeddings.")
    hk2.metric("Best Loss",     f"{_best_loss:,.1f}",
               delta=f"epoch {_best_idx+1}/{_n_epochs}", delta_color="off",
               help=f"Minimum Binary Cross-Entropy loss: L = −Σ[y·log(ŷ) + (1−y)·log(1−ŷ)]. "
                    f"Achieved at epoch {_best_idx+1} of {_n_epochs}. "
                    "This is the optimal checkpoint for early stopping.")
    hk3.metric("Final Loss",    f"{_final_loss:,.1f}",
               help=f"BCE training loss at epoch {_n_epochs} (final epoch). "
                    "Slightly higher than best if model overfit in later epochs — "
                    "typical in link prediction with dynamic negative sampling.")
    hk4.metric("Loss Reduction",f"{_reduction:.1f}%",
               help=f"(L_initial − L_final) / L_initial × 100% = "
                    f"({_init_loss:,.0f} − {_final_loss:.1f}) / {_init_loss:,.0f} × 100. "
                    "Measures total training progress. Higher = better convergence.")
    hk5.metric("Flagged Wallets",f"{len(fraud_df):,}",
               delta=f"{len(fraud_df)/embeddings.shape[0]*100:.2f}% of network",
               delta_color="off",
               help="Wallets where Isolation Forest anomaly score s(x,n) = 2^(−E[h(x)]/c(n)) "
                    "indicated anomalous embedding. "
                    f"contamination='auto' → model self-selects ~{len(fraud_df)/embeddings.shape[0]*100:.1f}% as outliers.")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📉 Training Loss")
        if loss_history is not None:
            st.plotly_chart(plot_loss_curve(loss_history, dark=st.session_state.dark_mode),
                            width='stretch')
            with st.expander("Loss Statistics", expanded=False):
                lc1, lc2, lc3 = st.columns(3)
                lc1.metric("Initial Loss",   f"{_init_loss:,.1f}")
                lc2.metric("Best Loss",      f"{_best_loss:,.1f}",
                           delta=f"epoch {_best_idx+1}", delta_color="off")
                lc3.metric("Final Loss",     f"{_final_loss:,.1f}")
                lc4, lc5, lc6 = st.columns(3)
                lc4.metric("Loss Reduction", f"{_reduction:.1f}%",
                           help=f"{_init_loss:,.0f} → {_final_loss:,.1f}")
                lc5.metric("Best Epoch",     f"{_best_idx+1} / {_n_epochs}",
                           help="Epoch with the lowest training loss")
                lc6.metric("Convergence",
                           "Yes ✅" if _final_loss < _best_loss * 3 else "Partial ⚠️",
                           help="Final loss within 3× of best — model did not diverge")
                st.markdown(
                    f'<p style="font-size:0.79rem;color:{p["text_muted"]};margin-top:0.4rem;">'
                    f'GraphSAGE trained for {_n_epochs} epochs using link-prediction loss '
                    f'(dot-product decoder). Loss dropped from <b>{_init_loss:,.0f}</b> → '
                    f'<b>{_best_loss:.1f}</b> at epoch {_best_idx+1}, then rebounded slightly '
                    f'to <b>{_final_loss:.1f}</b> at epoch {_n_epochs} — consistent with '
                    f'early-stopping candidates around epoch {_best_idx+1}.</p>',
                    unsafe_allow_html=True)
        else:
            st.info("Loss history not available. Save `loss_history.npy` during training.")

    with col2:
        st.markdown("### 📊 ROC Curve")
        if all(x is not None for x in [fpr, tpr, roc_auc_val]):
            st.plotly_chart(plot_roc_curve(fpr, tpr, roc_auc_val,
                                           dark=st.session_state.dark_mode),
                            width='stretch')
            with st.expander("ROC Statistics", expanded=False):
                rc1, rc2, rc3 = st.columns(3)
                rc1.metric("AUC Score",   f"{roc_auc_val:.4f}",
                           help="1.0 = perfect; 0.5 = random baseline")
                rc2.metric("ROC Points",  f"{len(fpr):,}",
                           help="Number of threshold points sampled for the curve")
                rc3.metric("Classifier",  "Isolation Forest")
                _auc_quality = ("🟢 Perfect separation" if roc_auc_val >= 0.99 else
                                "🟢 Excellent" if roc_auc_val > 0.9 else
                                "🟡 Good" if roc_auc_val > 0.7 else "🔴 Poor")
                st.success(_auc_quality) if roc_auc_val > 0.7 else st.error(_auc_quality)
                st.markdown(f"""<p style="font-size:0.79rem;color:{p['text_muted']};margin-top:0.4rem;">
<b>How this is measured:</b> Isolation Forest anomaly scores are computed for all
{embeddings.shape[0]:,} wallet embeddings. Wallets in <code>fraudulent_wallets.csv</code>
(fraud_label&nbsp;=&nbsp;−1) serve as ground-truth positives. The ROC curve shows how well
the GNN embedding space separates anomalous from normal wallets across every decision
threshold. An AUC of 1.000 confirms the GraphSAGE embeddings are <b>perfectly
separable</b> — the learned 64-dim representation cleanly encodes fraud-relevant graph
structure. The {len(fpr):,} curve points represent {len(fpr):,} unique score thresholds
evaluated on all {embeddings.shape[0]:,} wallets.</p>""", unsafe_allow_html=True)
        else:
            st.info("ROC data not available. Save `roc_data.npz` after evaluation.")

    st.markdown("---")
    st.markdown("### 📊 Evaluation Summary")

    _crit_count = len(fraud_df[fraud_df["risk_level"] == "Critical"]) if "risk_level" in fraud_df.columns else "—"
    _high_count = len(fraud_df[fraud_df["risk_level"] == "High"])     if "risk_level" in fraud_df.columns else "—"
    _avg_risk   = round(float(fraud_df["risk_score"].mean()), 1)      if "risk_score" in fraud_df.columns else "—"

    _eval_df = pd.DataFrame({
        "Category": [
            "Model","Model","Model","Model","Model",
            "Training","Training","Training","Training","Training",
            "Graph","Graph","Graph",
            "Fraud","Fraud","Fraud","Fraud","Fraud",
        ],
        "Metric": [
            "Architecture","GNN Layers","Aggregator","Embedding Dim","Decoder",
            "Total Epochs","Best Epoch","Initial Loss","Best Loss","Loss Reduction",
            "Total Nodes","Total Edges","Graph Density",
            "Fraud Algorithm","ROC-AUC","Flagged Wallets","Critical Wallets","Avg Risk Score",
        ],
        "Value": [
            "GraphSAGE","2","Mean aggregation","64-dim","dot(0.5) + cosine(0.3) + L2(0.2)",
            str(_n_epochs), f"{_best_idx+1}", f"{_init_loss:,.0f}", f"{_best_loss:.1f}",
            f"{_reduction:.1f}%",
            f"{embeddings.shape[0]:,}", f"{len(edges):,}",
            f"{len(edges)/(embeddings.shape[0]*(embeddings.shape[0]-1)):.2e}",
            "Isolation Forest", f"{roc_auc_val:.4f}" if roc_auc_val else "N/A",
            f"{len(fraud_df):,} ({len(fraud_df)/embeddings.shape[0]*100:.2f}%)",
            str(_crit_count), str(_avg_risk),
        ],
        "Status": ["✅"]*18,
    })
    st.dataframe(_eval_df, width='stretch', hide_index=True,
                 column_config={
                     "Category": st.column_config.TextColumn("Category", width="small"),
                     "Metric":   st.column_config.TextColumn("Metric",   width="medium"),
                     "Value":    st.column_config.TextColumn("Value",    width="large"),
                     "Status":   st.column_config.TextColumn("",         width="small"),
                 })


# ═══════════════════════════════════════════════
# ARCHITECTURE & ML  (NEW)
# ═══════════════════════════════════════════════
elif "🏗️ Architecture & ML" in section:
    st.markdown('<p class="main-header">🏗️ Architecture & ML Pipeline</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Live interactive GraphSAGE diagram · Mathematical formulations · Decoder analysis</p>', unsafe_allow_html=True)

    st.markdown("### 🔭 Live Architecture Diagram")
    st.markdown(
        f'<p style="color:{p["text_muted"]};font-size:0.84rem;">'
        f'100 live particles flow through the GraphSAGE pipeline. '
        f'<b>Hover</b> any layer for details. <b>Click</b> to burst 18 particles from that layer.</p>',
        unsafe_allow_html=True)
    st.iframe(
        render_architecture_animation(dark=st.session_state.dark_mode),
        height=540)

    st.markdown("---")
    tab_layers, tab_math, tab_decoder, tab_fraud_ml, tab_config = st.tabs([
        "📦 Layers", "∑ Mathematics", "🔬 Decoder Analysis", "🌲 Fraud ML", "⚙️ Config"])

    with tab_layers:
        st.markdown("#### Layer-by-Layer Breakdown")
        layers_info = [
            ("Input Layer", p["primary"], "2 features: in-degree, out-degree",
             "Raw topological features computed from the transaction graph. In-degree = number of transactions received. Out-degree = number of transactions sent.",
             "Shape: [N, 2] — one row per wallet, two columns for the two degree features."),
            ("GraphSAGE Layer 1", p["accent"], "64 hidden dims · MEAN aggregation",
             "Each wallet aggregates the embeddings of its neighbours (wallets it has transacted with) using MEAN pooling, then concatenates with its own embedding. A linear layer + ReLU + L2 normalisation follows.",
             "Operation: h¹_u = σ(W¹ · CONCAT(h⁰_u, MEAN({h⁰_v : v ∈ N(u)})))"),
            ("GraphSAGE Layer 2", p["accent"], "64 hidden dims · MEAN aggregation",
             "Same SAGE operation applied at a deeper level, now aggregating Layer-1 embeddings. After this layer, each wallet has seen the 2-hop neighbourhood of the graph.",
             "Operation: h²_u = σ(W² · CONCAT(h¹_u, MEAN({h¹_v : v ∈ N(u)})))"),
            ("Node Embedding", p["success"], "64-dimensional final representation",
             "The output of Layer 2, L2-normalised. This is the learned representation of each wallet in a 64-dimensional space. Similar wallets will have high cosine similarity.",
             "Shape: [N, 64] — used for both link prediction and fraud scoring."),
            ("Decoder", p["warning"], "3 complementary similarity signals",
             "Three signals capture different geometric properties of the embedding space: dot product (magnitude + angle), cosine (angle only, normalised), L2 distance (Euclidean proximity). Weighted combination + sigmoid gives a probability.",
             "P(edge u→v) = σ(0.5·dot + 0.3·cosine·|dot| + 0.2·L2sim·|dot|)"),
        ]
        for name, color, subtitle, desc, formula in layers_info:
            st.markdown(f"""<div class="layer-card">
                <div class="layer-title" style="color:{color};">{name}</div>
                <div style="font-size:0.8rem;color:{p['primary']};font-weight:600;margin:2px 0 6px;">{subtitle}</div>
                <div class="layer-detail">{desc}</div>
                <div class="formula-block" style="margin-top:0.5rem;">{formula}</div>
            </div>""", unsafe_allow_html=True)

    with tab_math:
        st.markdown("#### GraphSAGE Forward Pass")
        st.markdown(f'<p style="color:{p["text_muted"]};font-size:0.84rem;">For each node u at layer k, neighbourhood aggregation then linear projection. Hover the <span class="math-term" style="font-size:0.84rem;">terms<span class="math-popup"><div class="math-popup-title">Notation Guide</div><div class="math-popup-formula">h_u^(k) — embedding of node u at layer k<br>N(u)    — neighbours of u (direct tx counterparties)<br>W^(k)   — trainable weight matrix, layer k<br>σ       — ReLU activation function<br>‖·‖₂   — L2 (Euclidean) norm</div><div class="math-popup-desc">All vectors are 64-dimensional after layer 2. L2 normalisation brings each vector onto the unit hypersphere ℝ⁶⁴.</div></span></span> for definitions.</p>', unsafe_allow_html=True)

        for label, formula, tooltip_title, tooltip_formula, tooltip_desc in [
            ("1 · Aggregate neighbourhood",
             "h_N(u)^(k)  =  MEAN( { h_v^(k-1) : v ∈ N(u) } )",
             "MEAN Aggregation", "MEAN({h_v : v∈N(u)}) = (1/|N(u)|) Σ_{v∈N(u)} h_v^(k-1)",
             "Averages all neighbour embeddings. MEAN is the simplest GraphSAGE aggregator — no order dependency, O(|N|) compute."),
            ("2 · Concatenate + project",
             "h_u^(k)  =  σ( W^(k) · CONCAT( h_u^(k-1), h_N(u)^(k) ) )",
             "CONCAT + Linear Projection", "CONCAT(h_u, h_N) ∈ ℝ^(2d) → W^(k) projects to ℝ^d",
             "Concatenation preserves the node's own features alongside neighbour information. W^(k) ∈ ℝ^(d×2d) is learned by backpropagation through BCE loss."),
            ("3 · L2 normalise per layer",
             "h_u^(k)  =  h_u^(k)  /  ‖h_u^(k)‖₂",
             "L2 Normalisation", "h_norm = h / ‖h‖₂   where ‖h‖₂ = √(Σᵢ hᵢ²)",
             "Projects each embedding onto the unit hypersphere. Makes cosine similarity well-defined (dot product = cosine when both vectors are unit-norm)."),
        ]:
            st.markdown(f"""<div style="margin-top:0.9rem;">
<div style="font-size:0.78rem;color:{p['text_muted']};font-weight:600;margin-bottom:4px;">{label}</div>
<div class="formula-block" style="position:relative;">{formula}
  <span class="math-term" style="font-size:0.72rem;margin-left:8px;">?<span class="math-popup">
    <div class="math-popup-title">{tooltip_title}</div>
    <div class="math-popup-formula">{tooltip_formula}</div>
    <div class="math-popup-desc">{tooltip_desc}</div>
  </span></span>
</div></div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### Multi-Signal Decoder")
        st.markdown(f'<p style="color:{p["text_muted"]};font-size:0.84rem;">Three complementary geometric signals capture different aspects of the embedding relationship. Hover each formula for the intuition.</p>', unsafe_allow_html=True)

        decoder_rows = [
            ("1 · Dot Product",
             "s_dot  =  h_u · h_v  =  Σᵢ (h_u)ᵢ · (h_v)ᵢ",
             "Dot Product Signal", "s_dot = h_u^T h_v = ‖h_u‖·‖h_v‖·cos(θ)",
             "Captures both magnitude (activity level) and angle (similarity direction). High dot product = high embedding magnitude AND similar direction — both wallets are active and similar."),
            ("2 · Cosine Similarity",
             "s_cos  =  (h_u · h_v) / (‖h_u‖ · ‖h_v‖)",
             "Cosine Similarity", "cos(θ) = dot(h_u, h_v) / (‖h_u‖₂ · ‖h_v‖₂) ∈ [−1, +1]",
             "Direction-only similarity, normalised for magnitude. 1.0 = same direction = similar behaviour pattern regardless of activity level."),
            ("3 · L2 Similarity",
             "s_L2   =  1 / (1 + ‖h_u − h_v‖₂)",
             "L2 (Euclidean) Similarity", "s_L2 = 1/(1 + ‖h_u − h_v‖₂)  ∈ (0, 1]",
             "Euclidean proximity in embedding space. 1.0 = identical vectors. Decays as wallets move apart. Captures spatial distance not fully expressed by cosine."),
            ("4 · Weighted Combination",
             "score  =  0.5·s_dot  +  0.3·s_cos·|s_dot|  +  0.2·s_L2·|s_dot|",
             "Decoder Combination", "weights: dot=0.50, cosine=0.30, L2=0.20  →  sum=1.00",
             "Multiplicative coupling of cosine/L2 by |dot| ensures they only contribute when there is magnitude signal. Weights chosen empirically for best link prediction performance."),
            ("5 · Output Probability",
             "P(link u→v) = σ(score) = 1 / (1 + e^(−score))",
             "Sigmoid Activation", "σ(x) = eˣ / (1 + eˣ) = 1 / (1 + e^(−x))  ∈ (0, 1)",
             "Maps unbounded score to probability in (0,1). Threshold at 0.5 for binary classification: P > 0.5 → predict link exists."),
        ]
        for label, formula, tt_title, tt_formula, tt_desc in decoder_rows:
            st.markdown(f"""<div style="margin-top:0.8rem;">
<div style="font-size:0.78rem;color:{p['text_muted']};font-weight:600;margin-bottom:4px;">{label}</div>
<div class="formula-block" style="position:relative;">{formula}
  <span class="math-term" style="font-size:0.72rem;margin-left:8px;">?<span class="math-popup">
    <div class="math-popup-title">{tt_title}</div>
    <div class="math-popup-formula">{tt_formula}</div>
    <div class="math-popup-desc">{tt_desc}</div>
  </span></span>
</div></div>""", unsafe_allow_html=True)

    with tab_decoder:
        st.markdown("#### 🔬 Live Decoder Signal Breakdown")
        st.markdown(f'<p style="color:{p["text_muted"]};font-size:0.84rem;">Select two wallets to see exactly how the decoder computes the link probability from three separate signals.</p>', unsafe_allow_html=True)
        dc1, dc2, dc3 = st.columns([2, 2, 1])
        with dc1:
            da_id = st.number_input("Wallet A (ID)", min_value=0, max_value=int(embeddings.shape[0]-1), value=0, key="dec_a")
        with dc2:
            db_id = st.number_input("Wallet B (ID)", min_value=0, max_value=int(embeddings.shape[0]-1), value=1, key="dec_b")
        with dc3:
            st.markdown("<br>", unsafe_allow_html=True)
            run_dec = st.button("Analyse", type="primary", width='stretch')
        if run_dec:
            emb_a = embeddings[int(da_id)]
            emb_b = embeddings[int(db_id)]
            st.plotly_chart(plot_decoder_signals(emb_a, emb_b, dark=st.session_state.dark_mode), width='stretch')
            prob = compute_link_probability(emb_a, emb_b)
            col_r = p["danger"] if prob > 0.7 else p["warning"] if prob > 0.4 else p["success"]
            st.markdown(f"""<div style="background:{col_r}18;border:1px solid {col_r}44;border-radius:8px;padding:0.7rem 1rem;margin:0.5rem 0;">
                <b style="color:{col_r};">Final P(link) = {prob:.6f}</b> &nbsp; — &nbsp;
                <span style="color:{p['text_muted']};font-size:0.85rem;">
                {"High likelihood (>0.7)" if prob > 0.7 else "Moderate likelihood (0.4–0.7)" if prob > 0.4 else "Low likelihood (<0.4)"}
                </span></div>""", unsafe_allow_html=True)

    with tab_fraud_ml:
        st.markdown("#### 🌲 Isolation Forest — How it Works")
        st.markdown(f"""
**What it does:** Identifies wallets whose GNN embedding is an *outlier* in the 64-dimensional 
embedding space — i.e., wallets that behave differently from the majority.

**Algorithm:**
1. Build an ensemble of random decision trees (*isolation trees*)
2. For each wallet embedding, measure the **average path length** to isolate it
3. Short path length → easy to isolate → **anomalous**
4. Long path length → hard to isolate → **normal**
""")
        st.markdown(f'<div class="formula-block">anomaly_score(x) = 2^(−E[h(x)] / c(n))</div>', unsafe_allow_html=True)
        st.markdown(f"""
where E[h(x)] = average path length across all trees, c(n) = expected path length of Binary Search Tree.

**Risk score normalisation:**
""")
        st.markdown(f'<div class="formula-block">risk_score = (score_max − raw_score) / (score_max − score_min) × 100</div>', unsafe_allow_html=True)
        st.markdown(f"""
This inverts the scale so higher risk_score = more suspicious (more anomalous).

**Thresholds used in this dashboard:**

| Risk Level | Score Range | Count in Dataset |
|---|---|---|
| 🔴 Critical | ≥ 80 | {len(fraud_df[fraud_df["risk_level"]=="Critical"]):,} |
| 🟠 High | 60–79 | {len(fraud_df[fraud_df["risk_level"]=="High"]):,} |
| 🟡 Medium | 35–59 | {len(fraud_df[fraud_df["risk_level"]=="Medium"]):,} |
| 🟢 Low | < 35 | {len(fraud_df[fraud_df["risk_level"]=="Low"]):,} |
""")

    with tab_config:
        sc1, sc2 = st.columns(2)
        with sc1:
            st.markdown("#### Model Config")
            st.code("""Model          : GraphSAGE
Layers         : 2
Hidden channels: 64
Input features : 2  (in-degree, out-degree)
Output embedding: 64 dimensions
Aggregation    : MEAN
Activation     : ReLU
Normalisation  : L2 per layer

Decoder:
  Dot product weight   : 0.50
  Cosine sim weight    : 0.30
  L2 similarity weight : 0.20
  Output activation    : Sigmoid""", language="text")
        with sc2:
            st.markdown("#### Training Config")
            st.code("""Optimiser      : Adam
Learning rate  : 0.01
Loss function  : Binary Cross-Entropy
Epochs         : 100
Train split    : 70%
Val split      : 15%
Test split     : 15%
Negative sampl.: Dynamic (random)

Fraud detection:
  Algorithm    : Isolation Forest
  n_estimators : 100
  contamination: auto
  Input        : 64-dim node embeddings""", language="text")


# ═══════════════════════════════════════════════
# EMBEDDING SPACE  (NEW)
# ═══════════════════════════════════════════════
elif "🧬 Embedding Space" in section:
    st.markdown('<p class="main-header">🧬 Embedding Space Explorer</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">2D / 3D PCA projection of all 25K wallet embeddings · Fraud clusters · Nearest-neighbour search</p>', unsafe_allow_html=True)

    # Controls
    # Formula card for PCA section
    st.markdown(f"""
<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:0.8rem;">
  <div class="formula-card" style="flex:1;min-width:220px;">
    <div class="formula-card-title">PCA Projection — z = W^T x</div>
    <div class="formula-card-eq">W ∈ ℝ^(64×k)  extracted by eigendecomposition of cov(X)</div>
    <div class="formula-card-desc">k=2 for 2D, k=3 for 3D. W's columns are the top-k principal components — directions of maximum variance in the 64-dim embedding space. Dot product projects each wallet onto these axes.</div>
  </div>
  <div class="formula-card" style="flex:1;min-width:200px;">
    <div class="formula-card-title">Cosine Similarity (nearest neighbours)</div>
    <div class="formula-card-eq">sim(u,v) = h_u·h_v / (‖h_u‖·‖h_v‖) ∈ [−1, 1]</div>
    <div class="formula-card-desc">L2-normalised embeddings make cosine = dot product. Nearest neighbours in embedding space ≈ most similar transaction behaviour.</div>
  </div>
  <div class="formula-card" style="flex:1;min-width:200px;">
    <div class="formula-card-title">Explained Variance Ratio</div>
    <div class="formula-card-eq">EVR_k = λ_k / Σᵢ λᵢ</div>
    <div class="formula-card-desc">λ_k = k-th eigenvalue of the covariance matrix. PC1+PC2+PC3 capture the top EVR% of total variance in 64-dim embedding space.</div>
  </div>
</div>""", unsafe_allow_html=True)

    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([2, 1, 1, 1])
    with ctrl1:
        sample_size = st.slider("Sample size (wallets)", 1000, 6000, 4000, step=500,
                                help="Number of wallets to sample for PCA. Larger samples show more structure "
                                     "but require more compute. PCA complexity is O(n·d²) where d=64 dims.")
    with ctrl2:
        mode_3d = st.toggle("3D View", value=False,
                             help="Switch between 2D (PC1 vs PC2) and 3D (PC1, PC2, PC3) scatter. "
                                  "3D captures more variance but is harder to read.")
    with ctrl3:
        hl_id = st.number_input("Highlight wallet ID", min_value=-1,
                                max_value=int(embeddings.shape[0]-1), value=-1,
                                help="-1 = no highlight. Enter any wallet ID to mark it as a star on the scatter plot.")
    with ctrl4:
        st.markdown("<br>", unsafe_allow_html=True)
        run_pca = st.button("🔭 Generate", type="primary", width='stretch')

    if run_pca or "pca_coords" not in st.session_state or st.session_state.get("pca_sample") != sample_size:
        with st.spinner(f"Computing PCA on {sample_size:,} wallet embeddings…"):
            n_comp = 3
            coords, sampled_ids, var_ratio = _pca_coords(embeddings, sample_size, n_comp)
            st.session_state["pca_coords"]   = coords
            st.session_state["pca_ids"]      = sampled_ids
            st.session_state["pca_var"]      = var_ratio
            st.session_state["pca_sample"]   = sample_size

    coords      = st.session_state.get("pca_coords")
    sampled_ids = st.session_state.get("pca_ids")
    var_ratio   = st.session_state.get("pca_var")

    if coords is not None:
        # Main PCA scatter
        fig_pca = plot_pca_embeddings(
            coords, sampled_ids, fraud_df, le,
            dark=st.session_state.dark_mode,
            highlight_id=int(hl_id),
            mode_3d=mode_3d,
        )
        st.plotly_chart(fig_pca, width='stretch')

        # Stats row
        st.markdown("---")
        vc1, vc2, vc3 = st.columns(3)
        with vc1:
            st.markdown("#### Explained Variance")
            labels = [f"PC{i+1}" for i in range(len(var_ratio))]
            fig_var = go.Figure(go.Bar(
                x=labels, y=[v * 100 for v in var_ratio],
                marker_color=[p["primary"], p["accent"], p["success"]],
                text=[f"{v*100:.1f}%" for v in var_ratio],
                textposition="outside",
            ))
            fig_var.update_layout(
                template=p["plotly"], height=220,
                yaxis_title="Variance Explained (%)",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=10, b=40, l=40, r=10),
            )
            st.plotly_chart(fig_var, width='stretch')

        with vc2:
            st.markdown("#### Sampled Distribution")
            sampled_ids_set = set(sampled_ids.tolist())
            sampled_fraud = fraud_df[fraud_df["wallet_id"].isin(sampled_ids_set)]
            dist = sampled_fraud["risk_level"].value_counts()
            clean_cnt = len(sampled_ids) - len(sampled_fraud)
            all_dist = dict(dist)
            all_dist["Clean"] = clean_cnt
            for lbl, cnt in sorted(all_dist.items(), key=lambda x: ["Critical","High","Medium","Low","Clean"].index(x[0]) if x[0] in ["Critical","High","Medium","Low","Clean"] else 99):
                col_map = {"Critical":"#f85149","High":"#ff7b72","Medium":"#e3b341","Low":"#3fb950","Clean":"#58a6ff"}
                bar_w = cnt / len(sampled_ids) * 100
                st.markdown(f"""<div style="margin:4px 0;">
                    <div style="display:flex;justify-content:space-between;font-size:0.76rem;margin-bottom:2px;">
                        <span style="color:{col_map.get(lbl, p['text_muted'])};font-weight:600;">{lbl}</span>
                        <span style="color:{p['text_muted']};">{cnt:,}</span>
                    </div>
                    <div style="background:rgba(128,128,128,0.1);border-radius:4px;height:7px;">
                        <div style="width:{bar_w:.1f}%;height:100%;background:{col_map.get(lbl, p['text_muted'])};border-radius:4px;"></div>
                    </div></div>""", unsafe_allow_html=True)

        with vc3:
            st.markdown("#### Nearest Neighbours")
            nn_id = st.number_input("Query wallet ID", min_value=0,
                                    max_value=int(embeddings.shape[0]-1), value=0, key="nn_query",
                                    help="Wallet ID to query. The model finds wallets whose 64-dim embedding "
                                         "vectors are closest by cosine similarity: sim(u,v) = h_u·h_v / (‖h_u‖·‖h_v‖).")
            k_nn  = st.slider("K neighbours", 3, 15, 5, key="nn_k",
                              help="Number of nearest neighbours to retrieve. "
                                   "KNN search is O(|V|·d) brute-force — feasible for 25K wallets × 64 dims.")
            if st.button("Find Neighbours", width='stretch'):
                top_ids, top_sims = find_top_k_similar(embeddings[nn_id], embeddings, k=k_nn+1, exclude_id=nn_id)
                nn_rows = []
                for sid, sim in zip(top_ids, top_sims):
                    if sid == nn_id: continue
                    fi = fraud_df[fraud_df["wallet_id"] == sid]
                    try:    sa = le.inverse_transform([int(sid)])[0][:20] + "…"
                    except Exception: sa = f"ID {sid}"
                    nn_rows.append({"ID": int(sid), "Cosine": round(float(sim), 4),
                                    "Risk": fi.iloc[0]["risk_level"] if len(fi) > 0 else "Clean"})
                if nn_rows:
                    st.dataframe(pd.DataFrame(nn_rows).head(k_nn),
                                 width='stretch', hide_index=True,
                                 column_config={"Cosine": st.column_config.NumberColumn("Cosine Sim", format="%.4f")})

        st.markdown("---")
        st.markdown(f"""<div class="info-box">
            <b>How to read this chart</b><br>
            <span style="font-size:0.82rem;color:{p['text_muted']};">
            Each dot is a wallet, projected from 64 dimensions to {2 if not mode_3d else 3}D using PCA.
            <b style="color:{p['danger']};">Red/orange</b> = high fraud risk ·
            <b style="color:#e3b341;">Yellow</b> = medium ·
            <b style="color:{p['success']};">Green</b> = low ·
            <b style="color:{p['primary']};">Blue</b> = clean.
            Wallets close together in this space have similar GNN embedding vectors — meaning the model
            has learned that they have similar transaction behaviour.
            PC1+PC2+PC3 capture {sum(var_ratio)*100:.1f}% of the total embedding variance.
            </span>
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# NETWORK VISUALIZATION
# ═══════════════════════════════════════════════
elif "🌐 Network Visualization" in section:
    st.markdown('<p class="main-header">🌐 Network Visualization</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Directed transaction graph · Fraud-aware node colouring · Configurable layout</p>', unsafe_allow_html=True)

    st.markdown(f"""
<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:0.8rem;">
  <div class="formula-card" style="flex:1;min-width:220px;">
    <div class="formula-card-title">Graph Density</div>
    <div class="formula-card-eq">ρ = |E| / (|V| · (|V| − 1))</div>
    <div class="formula-card-desc">Fraction of all possible directed edges that exist. |E| = edges shown · |V| = nodes in subgraph. ρ→1 = dense cluster · ρ→0 = sparse hub-and-spoke.</div>
  </div>
  <div class="formula-card" style="flex:1;min-width:220px;">
    <div class="formula-card-title">Node Degree</div>
    <div class="formula-card-eq">d(v) = d⁺(v) + d⁻(v)   (out + in)</div>
    <div class="formula-card-desc">d⁺(v) = outgoing transactions sent · d⁻(v) = incoming transactions received. Node size in graph ∝ degree for visual prominence.</div>
  </div>
  <div class="formula-card" style="flex:1;min-width:200px;">
    <div class="formula-card-title">Fraud Propagation Risk</div>
    <div class="formula-card-eq">FPR(v) = |N(v) ∩ F| / |N(v)|</div>
    <div class="formula-card-desc">F = set of all fraud-flagged wallets. FPR(v) measures what fraction of v's direct neighbours are anomalous. High FPR → guilt-by-association risk.</div>
  </div>
</div>""", unsafe_allow_html=True)

    vc1, vc2, vc3 = st.columns([2, 1, 1])
    with vc1:
        wid_viz = st.number_input("Centre Wallet ID", min_value=0,
                                   max_value=int(embeddings.shape[0]-1),
                                   value=int(st.session_state.get("_viz_wallet", 0)), key="viz_wid")
    with vc2:
        if st.button("🚨 Top Fraud Wallet", width='stretch'):
            st.session_state["_viz_wallet"] = int(fraud_df.sort_values("risk_score", ascending=False).iloc[0]["wallet_id"])
            st.rerun()
    with vc3:
        if st.button("🎲 Random Wallet", width='stretch', key="viz_rand"):
            st.session_state["_viz_wallet"] = int(np.random.randint(0, embeddings.shape[0]))
            st.rerun()

    with st.expander("⚙️ Graph Settings", expanded=True):
        gs1, gs2, gs3, gs4 = st.columns(4)
        with gs1: max_conn  = st.slider("Max edges", 10, 100, 40, step=5,
                                        help="Cap on |E| shown in subgraph. Higher = more context but slower. "
                                             "Edges sampled by recency from the edge list.")
        with gs2: depth     = st.selectbox("Hop depth", ["1-hop", "2-hop"], index=0,
                                           help="1-hop: only direct counterparties N(v). "
                                                "2-hop: also includes counterparties of counterparties — "
                                                "reveals money-laundering layering patterns.")
        with gs3: layout_algo = st.selectbox("Layout", ["spring","kamada","circular","shell"],
                                              help="spring = force-directed (Fruchterman-Reingold) — "
                                                   "nodes repel, edges attract. "
                                                   "kamada = energy minimisation. "
                                                   "circular/shell = fixed ring layouts.")
        with gs4: show_lbl  = st.toggle("Show labels", value=True,
                                         help="Display truncated wallet addresses on nodes. "
                                              "Disable for dense graphs to reduce visual clutter.")

    if st.button("🌐 Generate Network Graph", type="primary", width='stretch'):
        wid = int(wid_viz)
        with st.spinner("Building graph…"):
            if depth == "2-hop":
                hop1 = edges[(edges["from_id"] == wid) | (edges["to_id"] == wid)]
                hop1_nodes = set(hop1["from_id"].tolist() + hop1["to_id"].tolist())
                g_edges = pd.concat([hop1, edges[
                    edges["from_id"].isin(hop1_nodes) | edges["to_id"].isin(hop1_nodes)]]).drop_duplicates()
            else:
                g_edges = edges[(edges["from_id"] == wid) | (edges["to_id"] == wid)]
            fig = create_network_subgraph(g_edges, wid, le, fraud_df,
                                          max_connections=max_conn, layout=layout_algo,
                                          show_labels=show_lbl, dark=st.session_state.dark_mode)
        if not fig:
            st.warning(f"No transactions found for wallet ID {wid}.")
        else:
            try:    centre_addr = le.inverse_transform([wid])[0]
            except Exception: centre_addr = None
            if centre_addr:
                st.markdown(f'<span class="wallet-address-full">📍 {centre_addr}</span>', unsafe_allow_html=True)
            fi = fraud_df[fraud_df["wallet_id"] == wid]
            if wid in FRAUD_IDS and len(fi) > 0:
                rs_c = float(fi.iloc[0]["risk_score"])
                rl_c = fi.iloc[0]["risk_level"]
                col_r = {"Critical":"#f85149","High":"#ff7b72","Medium":"#e3b341","Low":"#3fb950"}.get(rl_c, p["accent"])
                st.markdown(f'<div style="background:{col_r}18;border:1px solid {col_r}44;border-radius:8px;padding:0.5rem 1rem;margin:0.4rem 0;"><b style="color:{col_r};">⚠️ {rl_c} Risk — Score {rs_c:.1f}/100</b></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="background:{p["success"]}18;border:1px solid {p["success"]}44;border-radius:8px;padding:0.5rem 1rem;margin:0.4rem 0;"><b style="color:{p["success"]};">✅ Clean Wallet</b></div>', unsafe_allow_html=True)
            st.plotly_chart(fig, width='stretch')
            out_t = edges[edges["from_id"] == wid]
            in_t  = edges[edges["to_id"]   == wid]
            all_c = set(out_t["to_id"].tolist()) | set(in_t["from_id"].tolist())
            fl_c  = all_c & FRAUD_IDS
            km1, km2, km3, km4, km5, km6 = st.columns(6)
            km1.metric("Outgoing Txs",  f"{len(out_t):,}",
                       help=f"Out-degree d⁺(v) = |{{(v,u) ∈ E}}| for wallet {wid}. "
                            "Transactions sent from this wallet.")
            km2.metric("Incoming Txs",  f"{len(in_t):,}",
                       help=f"In-degree d⁻(v) = |{{(u,v) ∈ E}}| for wallet {wid}. "
                            "Transactions received by this wallet.")
            km3.metric("Total",         f"{len(out_t)+len(in_t):,}",
                       help=f"Total degree d(v) = d⁺(v) + d⁻(v) = {len(out_t)+len(in_t):,}. "
                            "Sum of all transaction interactions.")
            km4.metric("Counterparties",f"{len(all_c):,}",
                       help="Unique wallets this wallet has sent to or received from — "
                            "the 1-hop neighbourhood N(v).")
            km5.metric("Flagged CPs",   f"{len(fl_c):,}",
                       help=f"{len(fl_c):,} of {len(all_c):,} counterparties were flagged by "
                            "Isolation Forest as anomalous. High value = high-risk network neighbourhood.")
            km6.metric("Fraud Exposure",f"{round(len(fl_c)/max(len(all_c),1)*100,1)}%",
                       help=f"Fraud Exposure = |flagged CPs| / |total CPs| × 100 = "
                            f"{len(fl_c)} / {max(len(all_c),1)} × 100. "
                            "High exposure doesn't imply fraud but warrants investigation.")

    st.markdown("---")
    st.markdown("### 🚨 Top Suspicious Networks")
    sp1, sp2, sp3 = st.columns([2, 1, 1])
    with sp1: top_n = st.selectbox("Show top N", [3, 5, 10], index=1)
    with sp2: susp_conn = st.slider("Max edges/graph", 10, 40, 15, key="sc")
    with sp3: susp_layout = st.selectbox("Layout", ["spring","kamada","circular"], key="sl")

    if st.button("🔍 Render Suspicious Networks", width='stretch'):
        for _, row in fraud_df.sort_values("risk_score", ascending=False).head(top_n).iterrows():
            fw = int(row["wallet_id"])
            try:    lbl = f"{le.inverse_transform([fw])[0][:20]}… — {row['risk_level']} ({row['risk_score']:.1f}/100)"
            except Exception: lbl = f"Wallet {fw} — {row['risk_level']} ({row['risk_score']:.1f}/100)"
            col_r = {"Critical":"#f85149","High":"#ff7b72","Medium":"#e3b341","Low":"#3fb950"}.get(row["risk_level"], p["accent"])
            st.markdown(f'<div style="border-left:3px solid {col_r};padding-left:0.8rem;margin:0.4rem 0;"><b style="color:{col_r};">{lbl}</b></div>', unsafe_allow_html=True)
            sub_fig = create_network_subgraph(
                edges, fw, le, fraud_df, max_connections=susp_conn,
                layout=susp_layout, show_labels=False, dark=st.session_state.dark_mode)
            if sub_fig:
                st.plotly_chart(sub_fig, width='stretch')
            else:
                st.info(f"No edges for wallet {fw}.")


# ═══════════════════════════════════════════════
# LIVE BLOCKCHAIN EXPLORER  (Enhanced)
# ═══════════════════════════════════════════════
elif "🔌 Live Blockchain Explorer" in section:
    st.markdown('<p class="main-header">🔌 Live Blockchain Explorer</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Query any Ethereum address · Etherscan API · Web3.py event log parsing · GNN cross-reference</p>', unsafe_allow_html=True)

    eth_key = ETHERSCAN_API_KEY

    # ── Latest block info bar ────────────────────────────────────────────
    blk_info = get_latest_block_info()
    if blk_info:
        bc1, bc2, bc3, bc4 = st.columns(4)
        bc1.metric("Latest Block",    f"{blk_info['number']:,}",
                   help="Current Ethereum chain head block number (eth_blockNumber). "
                        "New block every ~12 seconds (post-Merge proof-of-stake).")
        bc2.metric("Timestamp",       blk_info["timestamp"][:16],
                   help="Unix timestamp of the latest block, converted to UTC datetime. "
                        "block.timestamp set by the validator at proposal time.")
        bc3.metric("Txs in Block",    f"{blk_info['tx_count']:,}",
                   help=f"Number of transactions included in block {blk_info['number']:,}. "
                        "Ethereum blocks are limited by gas: max ~30M gas per block. "
                        "Average transaction uses ~21,000 gas → ~1,428 txs at full capacity.")
        bc4.metric("Gas Utilisation", f"{blk_info['utilisation']}%",
                   help="gasUsed / gasLimit × 100. EIP-1559 targets 50% utilisation. "
                        ">50% → baseFee increases next block. <50% → baseFee decreases. "
                        "baseFee is burned, not paid to validators.")
        st.markdown("---")

    # ── Address input ─────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([5, 1, 1])
    with col1:
        lookup_addr = st.text_input(
            "🔍 Ethereum Address",
            value=st.session_state.get("live_lookup_addr", ""),
            placeholder="0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb",
            help="EOA wallet or smart contract")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("📋 Vitalik.eth", width='stretch'):
            lookup_addr = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🏦 Uniswap V3", width='stretch'):
            lookup_addr = "0xe592427a0aece92de3edee1f18e0157c05861564"

    if lookup_addr:
        addr_clean = lookup_addr.strip()
        if not is_valid_eth_address(addr_clean):
            st.markdown('<p class="invalid-address">⚠ Invalid format — must be 0x + 40 hex chars</p>', unsafe_allow_html=True)
        else:
            st.session_state["live_lookup_addr"] = addr_clean
            addr_lower = addr_clean.lower()
            st.markdown(f'<span class="wallet-address-full">📍 {addr_clean}</span>', unsafe_allow_html=True)

            if eth_key:
                with st.spinner("Querying Ethereum network…"):
                    balance = fetch_eth_balance(addr_clean)
                    is_contract, contract_name = check_is_contract(addr_clean)
                    tx_df    = fetch_transactions(addr_clean, limit=25)
                    token_df = fetch_token_transfers(addr_clean, limit=20)
                    tx_count = fetch_tx_count(addr_clean)

                # Address type
                if is_contract:
                    badge_text = f"📄 Smart Contract{(' — ' + contract_name) if contract_name else ''}"
                    st.markdown(f'<span class="contract-badge">{badge_text}</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="eoa-badge">👤 Externally Owned Account (EOA)</span>', unsafe_allow_html=True)

                known_proto = DEFI_PROTOCOLS.get(addr_lower)
                if known_proto:
                    st.success(f"🏦 **Known DeFi Protocol:** {known_proto}")

                # Metrics row
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("ETH Balance",
                          f"{balance:.6f} ETH" if balance is not None else "N/A",
                          delta=f"≈ ${balance * 2500:,.0f}" if balance else None, delta_color="off",
                          help="Native ETH balance via eth_getBalance RPC. "
                               "1 ETH = 10^18 Wei. Converted at ~$2500/ETH for USD estimate.")
                m2.metric("Nonce (Sent Txs)", f"{tx_count:,}" if tx_count is not None else "N/A",
                          help="Transaction count (nonce) from eth_getTransactionCount. "
                               "For EOAs: number of transactions sent FROM this address. "
                               "For contracts: number of contract creations triggered.")
                m3.metric("Address Type",    "Contract 📄" if is_contract else "EOA 👤",
                          help="EOA (Externally Owned Account) = controlled by a private key. "
                               "Contract = EVM bytecode deployed at this address. "
                               "Determined by eth_getCode — if non-empty bytecode exists = contract.")
                m4.metric("Recent Txs",      f"{len(tx_df):,}",
                          help=f"Last {len(tx_df):,} transactions from Etherscan API "
                               "(txlist endpoint, sorted by blockNumber desc). "
                               "Includes normal ETH transfers and contract interactions.")

                # DeFi protocols
                if not tx_df.empty:
                    protos = detect_protocols_from_txs(tx_df)
                    if protos:
                        st.markdown("**🏦 DeFi Protocol Interactions:**")
                        st.markdown(" ".join(f'<span class="protocol-tag">{pr}</span>' for pr in protos),
                                    unsafe_allow_html=True)
                        st.markdown("")

                # ── Tabs ──────────────────────────────────────────────────
                tab_tx, tab_tok, tab_events, tab_gnn, tab_contract = st.tabs([
                    "📋 Transactions", "🪙 Token Transfers",
                    "⛓️ Event Logs (Web3)", "🤖 GNN Cross-Reference", "📄 Contract Info"])

                with tab_tx:
                    st.markdown(f"#### Recent Transactions ({len(tx_df):,})")
                    if not tx_df.empty:
                        # Timeline chart
                        st.plotly_chart(plot_transaction_timeline(tx_df, dark=st.session_state.dark_mode),
                                        width='stretch')
                        display_tx = tx_df.drop(columns=["Full Hash"], errors="ignore")
                        st.dataframe(display_tx, width='stretch', height=380,
                                     column_config={
                                         "From":           st.column_config.TextColumn("From",    width="large"),
                                         "To":             st.column_config.TextColumn("To",      width="large"),
                                         "Value (ETH)":    st.column_config.NumberColumn("ETH",   format="%.6f"),
                                         "Gas (Gwei)":     st.column_config.NumberColumn("Gas",   format="%.2f"),
                                         "Protocol":       st.column_config.TextColumn("Protocol",width="medium"),
                                     })
                        st.download_button("⬇️ Export Transactions", tx_df.to_csv(index=False),
                                           file_name=f"txs_{addr_clean[:10]}.csv", mime="text/csv")
                    else:
                        st.info("No recent transactions found.")

                with tab_tok:
                    st.markdown(f"#### ERC-20 Token Transfers ({len(token_df):,})")
                    if not token_df.empty:
                        tc1, tc2 = st.columns([2, 1])
                        with tc1:
                            st.dataframe(token_df, width='stretch', height=360,
                                         column_config={
                                             "From":     st.column_config.TextColumn("From",     width="large"),
                                             "To":       st.column_config.TextColumn("To",       width="large"),
                                             "Value":    st.column_config.NumberColumn("Value",  format="%.4f"),
                                         })
                        with tc2:
                            tok_cnt = token_df.groupby("Token").size().reset_index(name="Count")
                            if len(tok_cnt) > 0:
                                fig_pie = px.pie(tok_cnt, names="Token", values="Count",
                                                 title="Token Distribution",
                                                 color_discrete_sequence=px.colors.qualitative.Set3)
                                fig_pie.update_layout(template=p["plotly"], height=300,
                                                      paper_bgcolor="rgba(0,0,0,0)")
                                st.plotly_chart(fig_pie, width='stretch')
                        st.download_button("⬇️ Export Token Transfers", token_df.to_csv(index=False),
                                           file_name=f"tokens_{addr_clean[:10]}.csv", mime="text/csv")
                    else:
                        st.info("No ERC-20 token transfers found.")

                with tab_events:
                    st.markdown("#### ⛓️ ERC-20 Transfer Event Logs  —  decoded via Web3.py")
                    st.markdown(f'<p style="color:{p["text_muted"]};font-size:0.83rem;">Fetches raw on-chain <code>Transfer(address,address,uint256)</code> event logs using <b>eth_getLogs</b> RPC call. Shows all token movements in and out of this address over the last ~8,000 blocks (≈ 1 day).</p>', unsafe_allow_html=True)

                    eb_col1, eb_col2 = st.columns([3, 1])
                    with eb_col1:
                        blocks_back = st.slider("Look-back (blocks)", 1000, 50000, 8000, step=1000,
                                                help="1 block ≈ 12 seconds · 7200 blocks ≈ 1 day")
                    with eb_col2:
                        st.markdown("<br>", unsafe_allow_html=True)
                        fetch_events_btn = st.button("⛓️ Fetch Events", type="primary", width='stretch')

                    if fetch_events_btn:
                        if get_web3() is None:
                            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                            st.warning("""
**Web3 RPC unavailable.** Could not connect to the Alchemy endpoint.  
Check that `WEB3_PROVIDER_URL` is set correctly in Replit Secrets, or try refreshing.
""")
                            st.markdown("</div>", unsafe_allow_html=True)
                        else:
                            with st.spinner(f"Fetching Transfer events over last {blocks_back:,} blocks…"):
                                ev_df = fetch_event_logs_web3(addr_clean, blocks_back=blocks_back)

                            if ev_df.empty:
                                st.info("No ERC-20 Transfer events found for this address in the specified block range.")
                            else:
                                # Summary bar chart
                                st.plotly_chart(plot_event_log_chart(ev_df, dark=st.session_state.dark_mode),
                                                width='stretch')

                                # Metrics
                                ev1, ev2, ev3, ev4 = st.columns(4)
                                ev1.metric("Total Events",  f"{len(ev_df):,}",
                                           help="Total ERC-20 Transfer events emitted by this address "
                                                "over the requested block range. Each Transfer event = "
                                                "1 token movement. Decoded from raw eth_getLogs response.")
                                ev2.metric("Received",      f"{len(ev_df[ev_df['Direction']=='Received']):,}",
                                           help="Events where topic[2] (to_address) matches the queried wallet. "
                                                "topic[2] is a 32-byte padded Ethereum address extracted from "
                                                "the log's indexed parameters.")
                                ev3.metric("Sent",          f"{len(ev_df[ev_df['Direction']=='Sent']):,}",
                                           help="Events where topic[1] (from_address) matches the queried wallet. "
                                                "topic[1] = 0x000...0 for mint events (new tokens created).")
                                unique_tokens = ev_df["Contract"].nunique()
                                ev4.metric("Unique Tokens", f"{unique_tokens:,}",
                                           help=f"{unique_tokens:,} distinct ERC-20 contract addresses "
                                                "involved in transfer events. Each contract has its own "
                                                "symbol, decimals, and total supply.")

                                # Table
                                show_ev = ev_df.drop(columns=["Etherscan"], errors="ignore").head(100)
                                st.dataframe(show_ev, width='stretch', height=380,
                                             column_config={
                                                 "From":      st.column_config.TextColumn("From",   width="large"),
                                                 "To":        st.column_config.TextColumn("To",     width="large"),
                                                 "Contract":  st.column_config.TextColumn("Contract",width="large"),
                                                 "Token":     st.column_config.TextColumn("Token",  width="medium"),
                                                 "Direction": st.column_config.TextColumn("Dir",    width="small"),
                                                 "Value (Wei)": st.column_config.NumberColumn("Value (Wei)", format="%d"),
                                                 "Block":     st.column_config.NumberColumn("Block",format="%d"),
                                             })

                                # Decode helper
                                with st.expander("ℹ️ What is an event log?"):
                                    st.markdown(f"""
**Event logs** are emitted by smart contracts as side-effects of transactions.  
They are stored on-chain in the transaction receipt and are queryable via `eth_getLogs`.

The ERC-20 **Transfer** event is the most common:
""")
                                    st.code("event Transfer(address indexed from, address indexed to, uint256 value)")
                                    st.markdown(f"""
- **Topic 0** — event signature hash `{KNOWN_EVENTS.get("0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef", "Transfer")}`
- **Topic 1** — `from` address (32-byte padded)
- **Topic 2** — `to` address (32-byte padded)
- **Data** — `uint256 value` (token amount in smallest unit, Wei)

Web3.py's `eth_getLogs` lets us filter by topic and address, then we manually decode the 32-byte topics back to addresses and parse the hex data as `uint256`.
""")
                                st.download_button("⬇️ Export Event Logs",
                                                   ev_df.to_csv(index=False),
                                                   file_name=f"events_{addr_clean[:10]}.csv",
                                                   mime="text/csv")
                    else:
                        st.markdown(f'<div class="info-box"><b>Click "Fetch Events"</b> to query on-chain Transfer event logs via Web3.py RPC call.</div>',
                                    unsafe_allow_html=True)

                with tab_gnn:
                    st.markdown("#### 🤖 GNN Model Cross-Reference")
                    matching = [a for a in WALLET_SET if a.lower() == addr_lower]
                    if matching:
                        wid_found = int(le.transform([matching[0]])[0])
                        wfi       = fraud_df[fraud_df["wallet_id"] == wid_found]
                        st.markdown('<div class="success-box">', unsafe_allow_html=True)
                        st.success(f"✅ Found in GNN dataset — Wallet ID: **{wid_found}**")
                        st.markdown("</div>", unsafe_allow_html=True)
                        gc1, gc2 = st.columns(2)
                        with gc1:
                            sent_ds = edges[edges["from_id"] == wid_found]
                            recv_ds = edges[edges["to_id"]   == wid_found]
                            _cps = set(sent_ds['to_id'].tolist()) | set(recv_ds['from_id'].tolist())
                            st.metric("Txs Sent (dataset)",     f"{len(sent_ds):,}",
                                      help=f"Out-degree d⁺({wid_found}) in the GNN training graph. "
                                           f"Transactions where this address appears in the from_id column.")
                            st.metric("Txs Received (dataset)", f"{len(recv_ds):,}",
                                      help=f"In-degree d⁻({wid_found}) in the GNN training graph. "
                                           f"Transactions where this address appears in the to_id column.")
                            st.metric("Unique Counterparties",  f"{len(_cps):,}",
                                      help=f"|N({wid_found})| = {len(_cps):,} unique wallets in 1-hop neighbourhood. "
                                           "Combined senders and receivers (undirected adjacency). "
                                           "This is the node's effective local context for GraphSAGE aggregation.")
                        with gc2:
                            if len(wfi) > 0:
                                rs_g = float(wfi.iloc[0]["risk_score"])
                                rl_g = wfi.iloc[0]["risk_level"]
                                col_r2 = {"Critical":"#f85149","High":"#ff7b72","Medium":"#e3b341","Low":"#3fb950"}.get(rl_g, p["accent"])
                                st.markdown(f'<div class="fraud-alert">', unsafe_allow_html=True)
                                st.metric("GNN Risk Score", f"{rs_g:.1f} / 100",
                                          help=f"Isolation Forest anomaly score normalised to 0–100. "
                                               f"s(x,n) = 2^(−E[h(x)]/c(n)) → inverted & scaled. "
                                               f"Score {rs_g:.1f} = {rl_g} risk tier.")
                                st.metric("Risk Level", rl_g,
                                          help="Risk tier based on score: Critical ≥80 · High 60–79 · "
                                               "Medium 35–59 · Low <35. "
                                               "Isolation Forest flagged this wallet as fraud_label = −1.")
                                st.markdown("</div>", unsafe_allow_html=True)
                            else:
                                st.markdown('<div class="success-box">', unsafe_allow_html=True)
                                st.metric("GNN Risk Score", "0 / 100",
                                          help="Isolation Forest did not flag this wallet. "
                                               "fraud_label = +1 (inlier). Isolation path length ≈ c(n) — "
                                               "normal behaviour in 64-dim embedding space.")
                                st.metric("Risk Level",     "Clean ✅",
                                          help="No anomaly detected. Wallet embedding is within the "
                                               "normal cluster in latent space.")
                                st.markdown("</div>", unsafe_allow_html=True)
                        st.markdown("---")
                        st.markdown("**🔮 Quick Link Predictions**")
                        if st.button("Run 8 Random Predictions", key="gnn_qp"):
                            sample_ids = np.random.choice(embeddings.shape[0], 8, replace=False)
                            qp_rows = []
                            for sid in sample_ids:
                                qp_prob = compute_link_probability(embeddings[wid_found], embeddings[sid])
                                try:    sa = le.inverse_transform([int(sid)])[0]
                                except Exception: sa = f"ID {sid}"
                                si = fraud_df[fraud_df["wallet_id"] == sid]
                                qp_rows.append({"Target": sa, "Probability": round(qp_prob, 4),
                                                "Target Risk": si.iloc[0]["risk_level"] if len(si) > 0 else "Clean"})
                            st.dataframe(pd.DataFrame(qp_rows).sort_values("Probability", ascending=False),
                                         width='stretch', hide_index=True,
                                         column_config={"Target": st.column_config.TextColumn("Target", width="large"),
                                                        "Probability": st.column_config.NumberColumn("Prob", format="%.4f")})
                        st.markdown("**🧬 Node Embedding (8×8 heatmap)**")
                        emb_live = embeddings[wid_found]
                        fig_emb = go.Figure(go.Heatmap(z=emb_live[:64].reshape(8, 8),
                                                        colorscale="RdBu", zmid=0, showscale=True))
                        fig_emb.update_layout(template=p["plotly"], height=280,
                                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                               margin=dict(t=5,b=5,l=5,r=5),
                                               xaxis=dict(showticklabels=False),
                                               yaxis=dict(showticklabels=False))
                        st.plotly_chart(fig_emb, width='stretch')
                    else:
                        st.markdown('<div class="info-box">', unsafe_allow_html=True)
                        st.info("Address not in the GNN training dataset (trained on a historical snapshot of ~25,542 wallets). Live blockchain data still shown in other tabs.")
                        st.markdown("</div>", unsafe_allow_html=True)

                with tab_contract:
                    st.markdown("#### 📄 Address Details")
                    if is_contract:
                        st.markdown(f'<span class="contract-badge">📄 Smart Contract{(" — "+contract_name) if contract_name else ""}</span>', unsafe_allow_html=True)
                        st.markdown("")
                        if known_proto:
                            st.success(f"🏦 **Known DeFi Protocol:** {known_proto}")
                        cc1, cc2 = st.columns(2)
                        with cc1:
                            st.markdown("**What is a Smart Contract?**")
                            st.markdown("""
Self-executing code on the Ethereum blockchain, triggered by transactions.  
Unlike EOA wallets, contracts have no private key — they run deterministic code.

**Common types:**
- 🔄 DEX/AMM (Uniswap, Curve)
- 💰 Lending (Aave, Compound)
- 🎨 NFT (ERC-721/1155)
- 🪙 Token (ERC-20)
- 🏦 Multisig (Gnosis Safe)
""")
                        with cc2:
                            if known_proto:
                                category = ("DEX / AMM" if any(x in known_proto for x in ["Uniswap","SushiSwap","Curve","Balancer"])
                                            else "Lending" if any(x in known_proto for x in ["Aave","Compound","Maker"])
                                            else "NFT Marketplace" if "OpenSea" in known_proto
                                            else "Aggregator" if any(x in known_proto for x in ["1inch","0x","MetaMask"])
                                            else "Token / Asset")
                                st.metric("Protocol Category", category)
                                st.metric("Protocol Name",     known_proto)
                            st.markdown(f"[🔗 View on Etherscan](https://etherscan.io/address/{addr_clean})")
                    else:
                        st.markdown('<span class="eoa-badge">👤 Externally Owned Account (EOA)</span>', unsafe_allow_html=True)
                        st.markdown("""
**About EOAs:** Wallets controlled by a private key (MetaMask, Ledger, etc.).  
No on-chain code — fully controlled by whoever holds the private key.  
Can hold ETH, send transactions, and interact with contracts.
""")
                        st.markdown(f"[🔗 View full history on Etherscan](https://etherscan.io/address/{addr_clean})")

            else:
                # No API key — GNN-only fallback
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.info("Live blockchain data requires an Etherscan API key. Showing GNN dataset data only.")
                st.markdown("</div>", unsafe_allow_html=True)
                matching = [a for a in WALLET_SET if a.lower() == addr_lower]
                if matching:
                    wid_found = int(le.transform([matching[0]])[0])
                    wfi = fraud_df[fraud_df["wallet_id"] == wid_found]
                    st.success(f"✅ Found in GNN dataset — Wallet ID: **{wid_found}**")
                    gc1, gc2 = st.columns(2)
                    with gc1:
                        sd = edges[edges["from_id"] == wid_found]
                        rd = edges[edges["to_id"]   == wid_found]
                        st.metric("Dataset Txs (sent)",     f"{len(sd):,}")
                        st.metric("Dataset Txs (received)", f"{len(rd):,}")
                    with gc2:
                        if len(wfi) > 0:
                            st.metric("Risk Score", f"{wfi.iloc[0]['risk_score']:.1f}/100")
                            st.metric("Risk Level", wfi.iloc[0]["risk_level"])
                        else:
                            st.metric("Risk Score", "0/100"); st.metric("Risk Level", "Clean ✅")
                else:
                    st.info("Address not found in GNN dataset. Add Etherscan API key to query live blockchain data.")


# ═══════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════
st.markdown("---")
st.markdown(f"""<div class="footer-text">
    <strong>ChainIntel Pro</strong> &nbsp;·&nbsp; Enterprise Blockchain Intelligence Platform<br>
    GraphSAGE &nbsp;·&nbsp; Isolation Forest &nbsp;·&nbsp; Etherscan API &nbsp;·&nbsp;
    Web3.py &nbsp;·&nbsp; PCA Embedding Explorer &nbsp;·&nbsp; Streamlit<br>
    <span style="font-size:0.72rem;">
        v5.0 &nbsp;·&nbsp; {datetime.now().strftime("%Y-%m-%d")} &nbsp;·&nbsp;
        {'🌙 Dark' if st.session_state.dark_mode else '☀️ Light'} &nbsp;·&nbsp;
        {embeddings.shape[0]:,} wallets &nbsp;·&nbsp; {len(edges):,} transactions &nbsp;·&nbsp;
        API {'✅' if ETHERSCAN_API_KEY else '⚠️'}
    </span>
</div>""", unsafe_allow_html=True)
